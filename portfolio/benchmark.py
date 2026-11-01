"""Compare existing validation portfolio returns with the Qlib CSI300 index."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("ridge_original27", "lightgbm_original27", "lightgbm_extended")


def load_benchmark(provider: str, start: str, end: str) -> pd.Series:
    """Load CSI300 opening prices from the local Qlib provider.

    Description:
        Retrieve the SH000300 open series for benchmark return alignment.
    Parameters:
        provider (str): Qlib binary data directory.
        start (str): Earliest requested entry date.
        end (str): Latest requested exit date.
    Returns:
        pd.Series: Daily index opening prices indexed by trading date.
    Notes:
        Index price normalization should be checked before interpreting results.
    """
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=provider, kernels=1)
    raw = D.features(["SH000300"], ["$open"], start_time=start, end_time=end)
    if raw.empty:
        raise ValueError("SH000300 open-price series is empty")
    prices = raw["$open"].droplevel("instrument")
    prices.index = pd.to_datetime(prices.index)
    if prices.index.has_duplicates:
        raise ValueError("Duplicate CSI300 trading dates")
    return prices.sort_index()


def compare_periods(periods: pd.DataFrame, benchmark: pd.Series, holding_days: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare portfolio returns with CSI300 over identical execution windows.

    Description:
        Match each portfolio's entry and exit opens to index opens, then
        calculate period-relative and compounded performance measures.
    Parameters:
        periods (pd.DataFrame): Existing validation period return report.
        benchmark (pd.Series): CSI300 index open prices indexed by date.
        holding_days (int): Trading intervals per non-overlapping portfolio period.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Matched period returns and summary.
    Notes:
        Missing benchmark dates cause a hard failure rather than silent dropping.
        Excess wealth is a relative wealth ratio, not a long-short P&L series.
    """
    if holding_days <= 0:
        raise ValueError("holding_days must be positive")
    frame = periods.copy()
    for column in ("entry_date", "exit_date"):
        if column not in frame:
            raise ValueError(f"Missing required column: {column}")
        frame[column] = pd.to_datetime(frame[column])
    if frame[["entry_date", "exit_date"]].isna().any().any():
        raise ValueError("Missing entry or exit dates")
    if frame["entry_date"].duplicated().any() or not frame["entry_date"].is_monotonic_increasing:
        raise ValueError("Periods must have unique, increasing entry dates")
    if (frame["exit_date"] <= frame["entry_date"]).any():
        raise ValueError("Exit date must follow entry date")
    if (frame["entry_date"].iloc[1:].to_numpy() < frame["exit_date"].iloc[:-1].to_numpy()).any():
        raise ValueError("Overlapping holding periods are not supported")
    benchmark = benchmark.copy()
    benchmark.index = pd.to_datetime(benchmark.index)
    if benchmark.index.has_duplicates:
        raise ValueError("Duplicate benchmark dates")
    required = pd.DatetimeIndex(frame["entry_date"].tolist() + frame["exit_date"].tolist())
    if not required.isin(benchmark.index).all():
        missing = required[~required.isin(benchmark.index)].unique()
        raise ValueError(f"Missing benchmark opens on {list(missing[:5])}")
    entry = benchmark.reindex(frame["entry_date"]).to_numpy(dtype=float)
    exit_price = benchmark.reindex(frame["exit_date"]).to_numpy(dtype=float)
    if not np.isfinite(entry).all() or not np.isfinite(exit_price).all() or (entry <= 0).any() or (exit_price <= 0).any():
        raise ValueError("Invalid CSI300 open prices")
    frame["csi300_return"] = exit_price / entry - 1.0
    periods_per_year = 252 / holding_days
    summary = []
    for model in MODELS:
        column = f"{model}_net"
        if column not in frame:
            raise ValueError(f"Missing strategy return: {column}")
        returns = pd.to_numeric(frame[column], errors="coerce")
        if returns.isna().any() or (returns <= -1).any():
            raise ValueError(f"Invalid strategy returns for {model}")
        excess = returns - frame["csi300_return"]
        frame[f"{model}_excess_csi300"] = excess
        strategy_wealth = (1 + returns).cumprod()
        benchmark_wealth = (1 + frame["csi300_return"]).cumprod()
        relative_wealth = strategy_wealth / benchmark_wealth
        tracking_error = excess.std(ddof=1) * np.sqrt(periods_per_year)
        summary.append({
            "model": model,
            "periods": len(frame),
            "strategy_annualized_return": strategy_wealth.iloc[-1] ** (periods_per_year / len(frame)) - 1,
            "csi300_annualized_return": benchmark_wealth.iloc[-1] ** (periods_per_year / len(frame)) - 1,
            "annualized_return_difference": strategy_wealth.iloc[-1] ** (periods_per_year / len(frame)) - benchmark_wealth.iloc[-1] ** (periods_per_year / len(frame)),
            "mean_period_excess": excess.mean(),
            "annualized_arithmetic_excess": excess.mean() * periods_per_year,
            "tracking_error": tracking_error,
            "information_ratio": excess.mean() * periods_per_year / tracking_error if tracking_error > 0 else np.nan,
            "relative_wealth_final": relative_wealth.iloc[-1],
            "relative_max_drawdown": (relative_wealth / relative_wealth.cummax() - 1).min(),
        })
    return frame, pd.DataFrame(summary)


def main() -> None:
    """Run a benchmark-only comparison without retraining any models.

    Description:
        Read Stage 6 period returns, fetch index opens and save comparisons.
    Parameters:
        None: All paths are supplied by command-line arguments.
    Returns:
        None: Writes matched period and summary CSV files.
    Notes:
        Original portfolio outputs are never overwritten.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holding-days", type=int, default=5)
    args = parser.parse_args()
    run_comparison(args.periods, args.provider, args.output, args.holding_days)


def run_comparison(periods_path: str, provider: str, output: str, holding_days: int = 5) -> None:
    """Match Stage 6 period returns to SH000300 and save the comparison.

    Description:
        Shared body of ``main`` and ``run_backtest.py benchmark``.
    Parameters:
        periods_path (str): Stage 6 validation period-return CSV.
        provider (str): Qlib provider directory used to load index opens.
        output (str): Destination folder for the matched-period and summary CSVs.
        holding_days (int): Expected trading days between entry and exit.
    Returns:
        None: Writes two CSV files and prints the summary.
    Notes:
        Original portfolio outputs are never overwritten.
    """
    periods = pd.read_csv(periods_path)
    benchmark = load_benchmark(provider, str(periods["entry_date"].min()), str(periods["exit_date"].max()))
    matched, summary = compare_periods(periods, benchmark, holding_days)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output / "validation_periods_vs_csi300.csv", index=False)
    summary.to_csv(output / "validation_csi300_summary.csv", index=False)
    print(f"Matched {len(matched)} periods to SH000300.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
