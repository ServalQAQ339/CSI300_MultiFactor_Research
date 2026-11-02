"""Test next-open execution timing and cost treatment."""

import numpy as np
import pandas as pd

import portfolio.backtest as backtest
from portfolio.backtest import (
    SCORES,
    evaluate_portfolios,
    rebalance_schedule,
    simulate_holdings,
)


def make_opens(dates: pd.DatetimeIndex, stocks: list[str], seed: int = 0) -> pd.DataFrame:
    """Create random positive open prices.

    Description:
        Simulate independent multiplicative price paths.
    Parameters:
        dates (pd.DatetimeIndex): Trading sessions.
        stocks (list[str]): Instrument codes.
        seed (int): Random generator seed.
    Returns:
        pd.DataFrame: Column 'open' indexed by (datetime, instrument).
    Notes:
        Synthetic data only.
    """
    rng = np.random.default_rng(seed)
    paths = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (len(dates), len(stocks))), axis=0))
    frame = pd.DataFrame(paths, index=dates, columns=stocks)
    return frame.stack(future_stack=True).rename("open").rename_axis(
        ["datetime", "instrument"]).to_frame()


def make_scores(dates, stocks: list[str], seed: int = 1) -> pd.Series:
    """Create random signal scores indexed by (datetime, instrument)."""
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    return pd.Series(rng.normal(size=len(index)), index=index)


def test_holding_simulation_matches_legacy_without_missing_prices() -> None:
    """Verify gross and net period returns equal the legacy evaluator.

    Description:
        With every open valid, full rebalancing must reproduce
        evaluate_portfolios on the same fixed schedule.
    Parameters:
        None: Synthetic prices and scores.
    Returns:
        None: Raises if any period return differs.
    Notes:
        Net parity checks the (1 + gross) * (1 - c) ** 2 - 1 cost identity.
    """
    dates = pd.bdate_range("2022-01-03", periods=40)
    stocks = [f"S{i}" for i in range(8)]
    opens = make_opens(dates, stocks)
    predictions = pd.DataFrame({name: make_scores(dates[:30], stocks, seed=i)
                                for i, name in enumerate(SCORES)})
    legacy, _ = evaluate_portfolios(predictions, opens, top_k=3, holding_days=5)
    schedule = rebalance_schedule(dates, dates[0], dates[29], 5)
    assert schedule.equals(pd.DatetimeIndex(legacy.index))
    for name in SCORES:
        gross = simulate_holdings(predictions[name], opens, schedule, 3, 5, 0.0)[0]
        net = simulate_holdings(predictions[name], opens, schedule, 3, 5, 10.0)[0]
        np.testing.assert_allclose(gross["return"], legacy[f"{name}_gross"], atol=1e-12)
        np.testing.assert_allclose(net["return"], legacy[f"{name}_net"], atol=1e-12)
    proxy = simulate_holdings(predictions[SCORES[0]] * 0, opens, schedule, None, 5, 0.0)[0]
    np.testing.assert_allclose(proxy["return"], legacy["proxy_return"], atol=1e-12)


def locked_market() -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Build a market where the top stock A is suspended at its nominal exit.

    Description:
        Signals on d0 and d5 (entries d1 and d6). A is suspended on d6 and d7
        and resumes on d8; prices are otherwise constant at 100, A resumes at 120.
    Parameters:
        None.
    Returns:
        tuple: Opens, scores, schedule and trading dates.
    Notes:
        A ranks first on both signal dates; its future suspension is unknown.
    """
    dates = pd.bdate_range("2022-01-03", periods=14)
    stocks = ["A", "B", "C"]
    wide = pd.DataFrame(100.0, index=dates, columns=stocks)
    wide.loc[dates[6]:dates[7], "A"] = np.nan
    wide.loc[dates[8]:, "A"] = 120.0
    opens = wide.stack(future_stack=True).rename("open").rename_axis(
        ["datetime", "instrument"]).to_frame()
    index = pd.MultiIndex.from_product([[dates[0], dates[5]], stocks],
                                       names=["datetime", "instrument"])
    scores = pd.Series([3.0, 2.0, 1.0] * 2, index=index)
    return opens, scores, rebalance_schedule(dates, dates[0], dates[5], 5), dates


def test_schedule_is_fixed_and_missing_exit_is_locked_then_sold() -> None:
    """Verify fixed dates, locking, delayed exit and no reuse of locked capital.

    Description:
        Hold A and B; A cannot be sold at d6, so only B's proceeds fund the d6
        purchases. A is sold at its first valid open d8 and the cash waits.
    Parameters:
        None: Uses locked_market.
    Returns:
        None: Assertions check periods, trades, holdings and NAV.
    Notes:
        Cost 0 isolates capital accounting from fees.
    """
    opens, scores, schedule, dates = locked_market()
    assert schedule.equals(pd.DatetimeIndex([dates[0], dates[5]]))
    periods, daily, trades, holdings = simulate_holdings(scores, opens, schedule, 2, 5, 0.0)
    assert periods.index.equals(schedule)
    first = periods.loc[dates[0]]
    assert first["n_locked_at_exit"] == 1 and np.isclose(first["return"], 0.0)

    buys = trades.loc[(trades["side"] == "buy") & (trades["date"] == dates[6])]
    assert set(buys["instrument"]) == {"B", "C"}
    assert np.isclose(buys["notional"].sum(), 0.5)
    sale = trades.loc[(trades["instrument"] == "A") & (trades["side"] == "sell")].iloc[0]
    assert sale["date"] == dates[8] and sale["reason"] == "delayed_exit"
    assert sale["nominal_exit_date"] == dates[6] and sale["delay_sessions"] == 2
    row = holdings.loc[(holdings["signal_date"] == dates[5]) & (holdings["instrument"] == "A")]
    assert row["in_target"].item() and not row["filled"].item()

    assert daily.loc[dates[7], "n_stale"] == 1
    assert np.isclose(daily.loc[dates[7], "locked_weight"], 0.5)
    assert np.isclose(daily.loc[dates[8], "cash"], 0.5 * 1.2)
    assert np.isclose(daily.loc[dates[9], "nav"], 0.5 * 1.2 + 0.5)
    assert np.isclose(periods.loc[dates[5], "return"], (0.6 + 0.5) / 1.0 - 1)


def test_daily_nav_conservation_and_cost_on_executed_notional() -> None:
    """Verify cash plus holdings value is conserved net of fees only.

    Description:
        Under constant prices for traded names, NAV falls only by costs, and
        costs equal the rate times executed notional, excluding locked names.
    Parameters:
        None: Uses locked_market with 10 bps costs.
    Returns:
        None: Assertions reconcile NAV, trades and costs.
    Notes:
        A's resumption gain is added back explicitly.
    """
    opens, scores, schedule, _ = locked_market()
    _, daily, trades, _ = simulate_holdings(scores, opens, schedule, 2, 5, 10.0)
    executed = trades.loc[trades["side"].isin(["buy", "sell"])]
    np.testing.assert_allclose(executed["cost"], executed["notional"] * 0.001)
    assert np.isclose(daily["cost"].sum(), executed["cost"].sum())
    a_shares = 0.5 * 0.999 / 100.0
    gain = a_shares * 20.0
    assert np.isclose(daily["nav"].iloc[-1], 1.0 + gain - daily["cost"].sum())
    replay_cash = (executed.loc[executed["side"] == "sell", "notional"].sum()
                   - executed.loc[executed["side"] == "buy", "notional"].sum()
                   + 1.0 - executed["cost"].sum() + executed.loc[executed["side"] == "buy",
                                                                   "cost"].sum())
    assert np.isclose(daily["cash"].iloc[-1], replay_cash)


def test_unresolved_position_is_flagged_without_cash() -> None:
    """Verify a stock that never resumes is reported, not dropped or monetized.

    Description:
        Suspend A from its nominal exit to the end of the price calendar.
    Parameters:
        None: Synthetic constant prices.
    Returns:
        None: Assertions check the unresolved record and stale NAV.
    Notes:
        The stale mark stays in NAV; no sale cash is created.
    """
    dates = pd.bdate_range("2022-01-03", periods=9)
    wide = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    wide.loc[dates[6]:, "A"] = np.nan
    opens = wide.stack(future_stack=True).rename("open").rename_axis(
        ["datetime", "instrument"]).to_frame()
    scores = pd.Series([2.0, 1.0], index=pd.MultiIndex.from_product(
        [[dates[0]], ["A", "B"]], names=["datetime", "instrument"]))
    schedule = rebalance_schedule(dates, dates[0], dates[0], 5)
    periods, daily, trades, _ = simulate_holdings(scores, opens, schedule, 1, 5, 0.0)
    unresolved = trades.loc[trades["side"] == "unresolved"]
    assert unresolved["instrument"].tolist() == ["A"]
    assert daily["phase"].iloc[-1] == "settlement" and daily["n_locked"].iloc[-1] == 1
    assert np.isclose(daily["cash"].iloc[-1], 0.0) and np.isclose(daily["nav"].iloc[-1], 1.0)
    assert periods.loc[dates[0], "locked_weight_at_exit"] == 1.0


def test_run_backtest_scores_full_universe(tmp_path, monkeypatch) -> None:
    """Verify run_backtest loads unfiltered labels and keeps NaN-label stocks.

    Description:
        Mock data loading with a synthetic panel in which the best Validation
        feature value has a missing future label, then run the full pipeline.
    Parameters:
        tmp_path (pathlib.Path): Pytest temporary output directory.
        monkeypatch (pytest.MonkeyPatch): Replaces file and Qlib loaders.
    Returns:
        None: Assertions check loader arguments, scores and selection.
    Notes:
        Uses the real sample preparation, model fitting and simulation.
    """
    rng = np.random.default_rng(5)
    dates = list(pd.bdate_range("2020-11-02", "2020-12-31")) + list(
        pd.bdate_range("2021-01-04", periods=12))
    stocks = [f"S{i:02d}" for i in range(40)]
    index = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    a = rng.normal(size=len(index))
    panel = pd.DataFrame({"a": a, "b": rng.normal(size=len(index)),
                          "label_5d": a + rng.normal(size=len(index))}, index=index)
    signal = pd.Timestamp("2021-01-04")
    panel.loc[(signal, "S00"), ["a", "label_5d"]] = [10.0, np.nan]
    calls = {}

    def fake_load_panel(data_dir, candidates, require_valid_label=True):
        calls["require_valid_label"] = require_valid_label
        return panel, ["a"], ["b"]

    calendar = pd.bdate_range("2021-01-04", periods=25)
    monkeypatch.setattr(backtest, "load_panel", fake_load_panel)
    monkeypatch.setattr(backtest, "selected_features", lambda path, available: ["b"])
    monkeypatch.setattr(backtest, "load_open_prices",
                        lambda provider, names, start, end: make_opens(calendar, names))
    backtest.run_backtest("unused", "unused", "unused", "unused", str(tmp_path), top_k=5)

    assert calls["require_valid_label"] is False
    scores = pd.read_parquet(tmp_path / "validation_scores.parquet")
    assert (signal, "S00") in scores.index
    holdings = pd.read_csv(tmp_path / "validation_holdings.csv", parse_dates=["signal_date"])
    first = holdings.loc[holdings["signal_date"] == signal]
    assert len(pd.read_csv(tmp_path / "validation_period_returns.csv")) == 3
    assert scores.loc[signal, "ridge_original27"].idxmax() == "S00"
    assert first.loc[(first["model"] == "ridge_original27")
                     & (first["instrument"] == "S00"), "filled"].item()


def test_next_open_and_transaction_cost() -> None:
    """Verify next-open entry and exit and the two-sided cost assumption.

    Description:
        Use deterministic synthetic prices and equal model rankings.
    Parameters:
        None: Synthetic prices are defined in the test.
    Returns:
        None: Raises if the timing or net return differs.
    Notes:
        The signal-day open must never be used for entry.
    """
    dates = pd.bdate_range("2022-01-03", periods=9)
    stocks = ["A", "B"]
    index = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    prices = pd.DataFrame({"open": 100.0}, index=index)
    prices.loc[(dates[1], "A"), "open"] = 100.0
    prices.loc[(dates[6], "A"), "open"] = 110.0
    prices.loc[(dates[1], "B"), "open"] = 100.0
    prices.loc[(dates[6], "B"), "open"] = 100.0
    signal_index = pd.MultiIndex.from_product(
        [[dates[0]], stocks], names=["datetime", "instrument"]
    )
    pred = pd.DataFrame({name: [2.0, 1.0] for name in SCORES}, index=signal_index)
    periods, summary = evaluate_portfolios(pred, prices, top_k=1, holding_days=5,
                                            one_way_cost_bps=10)
    expected = 1.1 * (0.999 ** 2) - 1
    assert np.isclose(periods.iloc[0]["ridge_original27_net"], expected)
    assert periods.iloc[0]["entry_date"] == dates[1]
    assert periods.iloc[0]["exit_date"] == dates[6]
    assert len(summary) == 3


def test_reject_duplicate_prices() -> None:
    """Reject ambiguous execution-price keys.

    Description:
        Verify duplicate price records cannot enter a portfolio simulation.
    Parameters:
        None: Uses a synthetic duplicated index.
    Returns:
        None: Raises AssertionError if duplicate detection fails.
    Notes:
        Duplicate keys could silently distort realized returns.
    """
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2022-01-03"), "A")] * 2,
        names=["datetime", "instrument"],
    )
    prices = pd.DataFrame({"open": [100, 101]}, index=idx)
    pred = pd.DataFrame({name: [1.0] for name in SCORES}, index=idx[:1])
    try:
        evaluate_portfolios(pred, prices)
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("Duplicate price keys were accepted")
