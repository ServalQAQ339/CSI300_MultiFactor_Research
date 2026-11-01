"""Evaluate fixed model scores with non-overlapping five-day open-to-open portfolios."""

from pathlib import Path

import numpy as np
import pandas as pd

from modeling.modeling import (
    fit_predict_models,
    prepare_samples,
    selected_features,
)
from factor_selection.selection import load_panel


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


def run_backtest(
    data_root: str,
    candidates: str,
    selection: str,
    provider: str,
    output: str,
    top_k: int = 30,
    cost_bps: float = 10.0,
) -> None:
    """Run frozen Stage 5 models and a preliminary validation portfolio simulation.

    Description:
        Refit fixed models using Train, then evaluate next-open validation portfolios.
    Parameters:
        data_root (str): Research data root.
        candidates (str): Candidate feature parquet.
        selection (str): Frozen Train-only feature selection CSV.
        provider (str): Qlib provider URI.
        output (str): Output directory for Stage 6 reports.
        top_k (int): Number of selected stocks.
        cost_bps (float): One-way assumed execution cost.
    Returns:
        None: Saves prediction, period and summary CSV files.
    Notes:
        No test period, validation tuning or claim of executable historical
        CSI300 benchmark tracking is included.
    """
    panel, baseline, available = load_panel(data_root, candidates)
    chosen = selected_features(selection, available)
    train, validation = prepare_samples(panel, baseline, chosen)
    predictions = fit_predict_models(train, validation, baseline, chosen)
    instruments = validation.index.get_level_values("instrument").unique().tolist()
    opens = load_open_prices(provider, instruments, "2020-12-01", "2024-02-01")
    periods, summary = evaluate_portfolios(predictions, opens, top_k, 5, cost_bps)

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    predictions[list(SCORES)].to_parquet(destination / "validation_scores.parquet")
    periods.to_csv(destination / "validation_period_returns.csv")
    summary.to_csv(destination / "validation_portfolio_summary.csv", index=False)
    print(f"Evaluated {len(periods)} non-overlapping five-day periods.")
    print(summary.to_string(index=False))
