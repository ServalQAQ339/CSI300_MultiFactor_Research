"""Evaluate fixed model scores with non-overlapping five-day open-to-open portfolios."""

from pathlib import Path

import numpy as np
import pandas as pd

from modeling.modeling import (
    fit_predict_models,
    prepare_samples,
    selected_features,
)
from factor_research.selection import VALIDATION, load_panel


SCORES = ("ridge_original27", "lightgbm_original27", "lightgbm_extended")


def load_open_prices(provider: str, instruments: list[str], start: str, end: str) -> pd.DataFrame:
    """Load Qlib daily open prices without using future information.

    Description:
        Retrieve daily open observations for the supplied instrument universe.
    Parameters:
        provider (str): Qlib binary provider directory.
        instruments (list[str]): Instruments already present in the research panel.
        start (str): Inclusive calendar start for price retrieval.
        end (str): Inclusive calendar end, including an exit-price buffer.
    Returns:
        pd.DataFrame: MultiIndex (datetime, instrument) with open prices.
    Notes:
        Price adjustment and historical tradability must be checked separately.
    """
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=provider, kernels=1)
    prices = D.features(instruments, ["$open"], start_time=start, end_time=end)
    prices = prices.rename(columns={"$open": "open"})
    prices.index = prices.index.set_names(["instrument", "datetime"])
    prices = prices.reorder_levels(["datetime", "instrument"]).sort_index()
    if prices.index.has_duplicates:
        raise ValueError("Duplicate open-price keys")
    return prices


def evaluate_portfolios(
    predictions: pd.DataFrame,
    opens: pd.DataFrame,
    top_k: int = 30,
    holding_days: int = 5,
    one_way_cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate non-overlapping next-open portfolios and equal-weight universe proxy.

    Description:
        Rank scores at signal close; enter at the next trading-day open and
        exit after holding_days open-to-open intervals, with fixed transaction costs.
    Parameters:
        predictions (pd.DataFrame): Validation score panel indexed by date and instrument.
        opens (pd.DataFrame): Open prices indexed by date and instrument.
        top_k (int): Maximum names held at each rebalance.
        holding_days (int): Open-to-open holding period and rebalance stride.
        one_way_cost_bps (float): Cost charged on each traded notional side.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Period returns and summary statistics.
    Notes:
        The benchmark is an equal-weight eligible-universe proxy, NOT CSI300.
        Missing execution prices are not treated as zero returns; the entire
        period is skipped if selected holdings lack entry or exit observations.
        Cost is charged at entry and exit each period, a conservative assumption
        that does not model actual share-level turnover or fills.
    """
    if top_k < 1 or holding_days < 1 or one_way_cost_bps < 0:
        raise ValueError("Invalid portfolio parameters")
    if predictions.index.has_duplicates or opens.index.has_duplicates:
        raise ValueError("Duplicate stock-date keys")
    if not set(SCORES).issubset(predictions.columns):
        raise ValueError("Missing model score columns")
    dates = opens.index.get_level_values("datetime").unique().sort_values()
    signals = predictions.index.get_level_values("datetime").unique().sort_values()
    positions = {date: i for i, date in enumerate(dates)}
    rows = []
    next_eligible = 0
    for signal in signals:
        pos = positions.get(signal)
        if pos is None or pos < next_eligible or pos + holding_days + 1 >= len(dates):
            continue
        entry, exit_date = dates[pos + 1], dates[pos + 1 + holding_days]
        day = predictions.xs(signal, level="datetime")
        entry_open = opens.xs(entry, level="datetime")["open"]
        exit_open = opens.xs(exit_date, level="datetime")["open"]
        eligible = day.index.intersection(entry_open.dropna().index)
        returns = (exit_open.reindex(eligible) / entry_open.reindex(eligible) - 1).replace(
            [np.inf, -np.inf], np.nan
        )
        if returns.isna().any():
            continue
        if len(eligible) < top_k:
            continue
        proxy = returns.loc[eligible].mean()
        row = {"signal_date": signal, "entry_date": entry, "exit_date": exit_date,
               "n_eligible": len(eligible), "proxy_return": proxy}
        for model in SCORES:
            ranked = day.loc[eligible, model].dropna().nlargest(top_k)
            if len(ranked) < top_k:
                break
            gross = returns.loc[ranked.index].mean()
            net = (1 + gross) * (1 - one_way_cost_bps / 10000) ** 2 - 1
            row[f"{model}_gross"] = gross
            row[f"{model}_net"] = net
            row[f"{model}_excess_proxy"] = net - proxy
        else:
            rows.append(row)
            next_eligible = pos + holding_days
    if not rows:
        raise ValueError("No valid non-overlapping portfolio periods")
    periods = pd.DataFrame(rows).set_index("signal_date")
    summary_rows = []
    for model in SCORES:
        series = periods[f"{model}_net"]
        excess = periods[f"{model}_excess_proxy"]
        wealth = (1 + series).cumprod()
        periods_per_year = 252 / holding_days
        summary_rows.append({
            "model": model,
            "periods": len(series),
            "mean_period_return": series.mean(),
            "annualized_return": wealth.iloc[-1] ** (periods_per_year / len(series)) - 1,
            "annualized_volatility": series.std(ddof=1) * np.sqrt(periods_per_year),
            "max_drawdown": (wealth / wealth.cummax() - 1).min(),
            "mean_excess_proxy": excess.mean(),
            "annualized_excess_proxy_approx": excess.mean() * periods_per_year,
        })
    return periods, pd.DataFrame(summary_rows)


def rebalance_schedule(calendar: pd.DatetimeIndex, first_signal: pd.Timestamp,
                       last_signal: pd.Timestamp, holding_days: int = 5) -> pd.DatetimeIndex:
    """Fix rebalance signal dates from the trading calendar alone.

    Description:
        Take every holding_days-th trading session from the first signal date
        through the last signal date.
    Parameters:
        calendar (pd.DatetimeIndex): Exchange trading sessions.
        first_signal (pd.Timestamp): First permitted signal date (rounded forward
            to a trading session).
        last_signal (pd.Timestamp): Last permitted signal date.
        holding_days (int): Stride between consecutive rebalance signals.
    Returns:
        pd.DatetimeIndex: Rebalance signal dates.
    Notes:
        Prices, suspensions and missing scores never alter the schedule.
    """
    if holding_days < 1:
        raise ValueError("holding_days must be positive")
    sessions = pd.DatetimeIndex(calendar).unique().sort_values()
    start = sessions.searchsorted(pd.Timestamp(first_signal))
    stop = sessions.searchsorted(pd.Timestamp(last_signal), side="right")
    if start >= stop:
        raise ValueError("No trading sessions in the signal range")
    return sessions[start:stop:holding_days]


def simulate_holdings(
    scores: pd.Series,
    opens: pd.DataFrame,
    schedule: pd.DatetimeIndex,
    top_k: int | None = 30,
    holding_days: int = 5,
    cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate an equal-weight long portfolio with per-stock holding state.

    Description:
        Signals at close t are executed at the open of t+1. On each rebalance
        open every sellable holding is sold, then the top_k highest-scored
        names that trade at that open are bought with the available cash only.
        Holdings whose open is invalid stay held (locked) and are sold at the
        first later session with a valid open; that cash waits for the next
        rebalance. The state is updated and NAV is recorded every session.
    Parameters:
        scores (pd.Series): Signal-time scores indexed by (datetime, instrument).
        opens (pd.DataFrame): Column 'open' indexed by (datetime, instrument); its
            dates define the trading calendar used for execution.
        schedule (pd.DatetimeIndex): Fixed signal dates from rebalance_schedule.
        top_k (int | None): Names to hold; None holds every tradable scored name
            (equal-weight universe proxy).
        holding_days (int): Sessions between rebalance opens.
        cost_bps (float): Cost per traded side as bps of executed notional.
    Returns:
        tuple: (periods, daily, trades, holdings).
            periods: per-rebalance wealth after sales and before purchases
                (wealth_start) to the same point at the next rebalance or the
                final nominal exit (wealth_end), with target/fill counts and the
                locked weight at exit.
            daily: post-trade NAV, cash, locked value/weight, stale-price count,
                traded notional and cost per session; phase 'horizon' up to the
                final nominal exit, 'settlement' afterwards.
            trades: executed buys and sells (with nominal exit and delay in
                sessions) plus 'unresolved' rows for positions never sold.
            holdings: per rebalance, target names (rank on all scored names)
                versus names actually bought.
    Notes:
        A valid open is finite and positive; limit-up/limit-down locks, volume
        and lot sizes are not modeled, so fills are a simplifying assumption.
        Selection never uses labels or future prices; target membership uses
        scores only, and fills use the execution-session open only. Locked
        positions are marked at their last valid open (stale) and their value
        is never reused. Final settlement rule (fixed in advance): after the
        last nominal exit the simulation continues through the end of the
        supplied price calendar, selling locked names at their first valid
        open; names never sold are reported as 'unresolved' at their stale
        mark, with no cash created. With no invalid prices, period returns
        equal (1 + gross) * (1 - cost) ** 2 - 1 of evaluate_portfolios.
    """
    if (top_k is not None and top_k < 1) or holding_days < 1 or cost_bps < 0:
        raise ValueError("Invalid portfolio parameters")
    if scores.index.has_duplicates or opens.index.has_duplicates:
        raise ValueError("Duplicate stock-date keys")
    prices = opens["open"].unstack("instrument").sort_index()
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.where(prices > 0)
    calendar = prices.index
    signal_pos = calendar.get_indexer(pd.DatetimeIndex(schedule))
    if len(signal_pos) == 0 or (signal_pos < 0).any():
        raise ValueError("Schedule dates must be trading sessions in the price calendar")
    if (np.diff(signal_pos) != holding_days).any():
        raise ValueError("Schedule must use a fixed holding_days stride")
    horizon_pos = signal_pos[-1] + 1 + holding_days
    if horizon_pos >= len(calendar):
        raise ValueError("Price calendar does not cover the final nominal exit")

    by_date = {date: day.droplevel("datetime").dropna()
               for date, day in scores.groupby(level="datetime")}
    entries = {pos + 1: signal for pos, signal in zip(signal_pos, schedule)}
    rate = cost_bps / 10000
    cash, shares, last_price, pending = 1.0, {}, {}, {}
    daily, trades, holdings, periods = [], [], [], []
    current = None

    for pos in range(signal_pos[0] + 1, len(calendar)):
        date, today = calendar[pos], prices.iloc[pos]
        rebalance, horizon_end = pos in entries, pos == horizon_pos
        phase = "horizon" if pos <= horizon_pos else "settlement"
        if phase == "settlement" and not shares:
            break
        for name in shares:
            if np.isfinite(today.get(name, np.nan)):
                last_price[name] = today[name]

        # Sell everything at a rebalance or the final nominal exit; otherwise
        # only retry locked names. Failed sales become (or stay) locked.
        traded = paid = 0.0
        for name in list(shares) if rebalance or horizon_end else list(pending):
            price = today.get(name, np.nan)
            if not np.isfinite(price):
                pending.setdefault(name, date)
                continue
            notional = shares.pop(name) * price
            fee = notional * rate
            cash += notional - fee
            traded, paid = traded + notional, paid + fee
            nominal = pending.pop(name, date)
            delay = pos - calendar.get_loc(nominal)
            reason = "settlement" if phase == "settlement" else (
                "delayed_exit" if delay else "rebalance")
            trades.append({"date": date, "instrument": name, "side": "sell", "price": price,
                           "notional": notional, "cost": fee, "reason": reason,
                           "nominal_exit_date": nominal, "delay_sessions": delay})

        # Close the running period: cash plus locked holdings at stale marks.
        if rebalance or horizon_end:
            locked = sum(shares[name] * last_price[name] for name in shares)
            wealth = cash + locked
            if current is not None:
                current.update({"wealth_end": wealth,
                                "return": wealth / current["wealth_start"] - 1,
                                "n_locked_at_exit": len(shares),
                                "locked_weight_at_exit": locked / wealth})
                periods.append(current)
                current = None

        if rebalance:
            signal = entries[pos]
            day = by_date.get(signal, pd.Series(dtype=float))
            target = day.index if top_k is None else day.nlargest(top_k).index
            tradable = day.loc[[name for name in day.index if name not in shares
                                and np.isfinite(today.get(name, np.nan))]]
            fill = tradable.index if top_k is None else tradable.nlargest(top_k).index
            if cash <= 0:
                fill = fill[:0]
            for name in target.union(fill):
                holdings.append({"signal_date": signal, "entry_date": date, "instrument": name,
                                 "in_target": name in target, "filled": name in fill})
            current = {"signal_date": signal, "entry_date": date,
                       "nominal_exit_date": calendar[pos + holding_days],
                       "wealth_start": cash + locked, "n_target": len(target),
                       "n_filled": len(fill), "n_unfilled_target": len(target.difference(fill)),
                       "n_locked_at_entry": len(shares)}
            if len(fill):
                notional = cash / len(fill)
                for name in fill:
                    price = today[name]
                    fee = notional * rate
                    shares[name] = (notional - fee) / price
                    last_price[name] = price
                    traded, paid = traded + notional, paid + fee
                    trades.append({"date": date, "instrument": name, "side": "buy",
                                   "price": price, "notional": notional, "cost": fee,
                                   "reason": "rebalance", "signal_date": signal})
                cash = 0.0

        # Post-trade valuation; names without a valid open today use stale marks.
        value = sum(shares[name] * last_price[name] for name in shares)
        locked = sum(shares[name] * last_price[name] for name in pending)
        nav = cash + value
        daily.append({"date": date, "phase": phase, "nav": nav, "cash": cash,
                      "holdings_value": value, "locked_value": locked,
                      "locked_weight": locked / nav if nav > 0 else np.nan,
                      "n_positions": len(shares), "n_locked": len(pending),
                      "n_stale": sum(not np.isfinite(today.get(name, np.nan)) for name in shares),
                      "traded_notional": traded, "cost": paid})

    last = calendar[min(pos, len(calendar) - 1)]
    for name, nominal in pending.items():
        trades.append({"date": last, "instrument": name, "side": "unresolved",
                       "price": last_price[name], "notional": shares[name] * last_price[name],
                       "cost": 0.0, "reason": "unresolved", "nominal_exit_date": nominal,
                       "delay_sessions": calendar.get_loc(last) - calendar.get_loc(nominal)})
    return (pd.DataFrame(periods).set_index("signal_date"), pd.DataFrame(daily).set_index("date"),
            pd.DataFrame(trades), pd.DataFrame(holdings))


def run_backtest(
    data_dir: str,
    candidates: str,
    selection: str,
    provider: str,
    output: str,
    top_k: int = 30,
    cost_bps: float = 10.0,
) -> None:
    """Run frozen Stage 5 models and a holding-state validation portfolio simulation.

    Description:
        Refit fixed models using Train, score every feature-complete Validation
        stock, then simulate fixed-schedule next-open Top-K portfolios with
        simulate_holdings (gross and net) and an equal-weight universe proxy.
    Parameters:
        data_dir (str): Research data directory, i.e. <project root>/data (not the project root).
        candidates (str): Candidate feature parquet.
        selection (str): Frozen Train-only feature selection CSV.
        provider (str): Qlib provider URI.
        output (str): Output directory for Stage 6 reports.
        top_k (int): Number of selected stocks.
        cost_bps (float): One-way assumed execution cost.
    Returns:
        None: Saves scores, period returns, summary, daily NAV, trades and
            target-versus-filled holdings.
    Notes:
        The prediction universe never depends on future label validity. Period
        exits after the Validation end are flagged; they are realized in 2024.
        evaluate_portfolios is kept only to reproduce historical outputs. No
        test period, validation tuning or claim of executable historical
        CSI300 benchmark tracking is included.
    """
    panel, baseline, available = load_panel(data_dir, candidates, require_valid_label=False)
    chosen = selected_features(selection, available)
    train, validation = prepare_samples(panel, baseline, chosen)
    predictions = fit_predict_models(train, validation, baseline, chosen)
    instruments = validation.index.get_level_values("instrument").unique().tolist()
    opens = load_open_prices(provider, instruments, "2020-12-01", "2024-02-01")
    signals = predictions.index.get_level_values("datetime")
    calendar = opens.index.get_level_values("datetime").unique().sort_values()
    schedule = rebalance_schedule(calendar, signals.min(), signals.max(), 5)

    # Equal-weight universe proxy: every tradable scored name, gross of costs.
    universe = pd.Series(0.0, index=predictions.index)
    proxy = simulate_holdings(universe, opens, schedule, None, 5, 0.0)[0]
    periods = proxy[["entry_date", "nominal_exit_date"]].rename(
        columns={"nominal_exit_date": "exit_date"})
    periods["realized_after_validation_end"] = periods["exit_date"] > pd.Timestamp(VALIDATION[1])
    periods["n_eligible"] = proxy["n_filled"]
    periods["proxy_return"] = proxy["return"]

    daily_frames, trade_frames, holding_frames, summary_rows = [], [], [], []
    periods_per_year = 252 / 5
    for model in SCORES:
        gross, gross_daily, _, _ = simulate_holdings(predictions[model], opens, schedule,
                                                     top_k, 5, 0.0)
        net, daily, trades, holdings = simulate_holdings(predictions[model], opens, schedule,
                                                         top_k, 5, cost_bps)
        periods[f"{model}_gross"] = gross["return"]
        periods[f"{model}_net"] = net["return"]
        periods[f"{model}_excess_proxy"] = net["return"] - periods["proxy_return"]
        periods[f"{model}_n_unfilled_target"] = net["n_unfilled_target"]
        periods[f"{model}_locked_weight_at_exit"] = net["locked_weight_at_exit"]
        daily = daily.join(gross_daily[["nav"]].rename(columns={"nav": "gross_nav"}))
        daily_frames.append(daily.rename(columns={"nav": "net_nav"}).assign(model=model))
        trade_frames.append(trades.assign(model=model))
        holding_frames.append(holdings.assign(model=model))

        series, horizon = net["return"], daily.loc[daily["phase"] == "horizon"]
        wealth = (1 + series).cumprod()
        sells = trades.loc[trades["side"] == "sell"]
        delayed = sells.loc[sells["delay_sessions"] > 0, "delay_sessions"]
        bought = trades.loc[trades["side"] == "buy"].groupby("date")["notional"].sum()
        bought = bought.reindex(net["entry_date"]).fillna(0.0).to_numpy()
        summary_rows.append({
            "model": model,
            "periods": len(series),
            "mean_period_return": series.mean(),
            "annualized_return": wealth.iloc[-1] ** (periods_per_year / len(series)) - 1,
            "annualized_volatility": series.std(ddof=1) * np.sqrt(periods_per_year),
            "max_drawdown": (wealth / wealth.cummax() - 1).min(),
            "mean_excess_proxy": periods[f"{model}_excess_proxy"].mean(),
            "annualized_excess_proxy_approx":
                periods[f"{model}_excess_proxy"].mean() * periods_per_year,
            "gross_annualized_return":
                (1 + gross["return"]).prod() ** (periods_per_year / len(series)) - 1,
            "max_drawdown_daily_nav": (horizon["nav"] / horizon["nav"].cummax() - 1).min(),
            "total_cost": horizon["cost"].sum(),
            "mean_buy_turnover_per_rebalance": (bought / net["wealth_start"].to_numpy()).mean(),
            "max_locked_weight": horizon["locked_weight"].max(),
            "delayed_exits": len(delayed),
            "max_delay_sessions": delayed.max() if len(delayed) else 0,
            "unresolved_positions": int((trades["side"] == "unresolved").sum()),
            "horizon_end": horizon.index.max(),
            "final_net_nav_horizon": horizon["nav"].iloc[-1],
            "final_net_nav_settled": daily["nav"].iloc[-1],
        })

    summary = pd.DataFrame(summary_rows)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    predictions[list(SCORES)].to_parquet(destination / "validation_scores.parquet")
    periods.to_csv(destination / "validation_period_returns.csv")
    summary.to_csv(destination / "validation_portfolio_summary.csv", index=False)
    pd.concat(daily_frames).to_csv(destination / "validation_daily_nav.csv")
    pd.concat(trade_frames).to_csv(destination / "validation_trades.csv", index=False)
    pd.concat(holding_frames).to_csv(destination / "validation_holdings.csv", index=False)
    print(f"Simulated {len(periods)} fixed five-day rebalances "
          f"({int(periods['realized_after_validation_end'].sum())} exit after Validation end).")
    print(summary.to_string(index=False))
