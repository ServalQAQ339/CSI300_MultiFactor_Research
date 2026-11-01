"""Generate membership-preserving CSI300 factors using extended price history."""

from pathlib import Path
import numpy as np
import pandas as pd
from data.history_loader import load_extended_history
from factors.factor_engine import build_factor_panel, DEFAULT_WINDOWS


def generate_backfilled_year(target: pd.DataFrame, calendar: pd.DatetimeIndex, windows: tuple[int, ...] = DEFAULT_WINDOWS, fetcher=None) -> tuple[pd.DataFrame, dict]:
    """Calculate factors using prior price history and preserve eligible keys exactly.

    Description:
        Retrieve sufficient stock-specific prehistory and calculate all technical
        factors before restricting output to the historical target membership.

    Parameters:
        target (pd.DataFrame): Original annual dynamic CSI300 panel.
        calendar (pd.DatetimeIndex): Qlib exchange calendar including pre-2015 dates.
        windows (tuple[int, ...]): Rolling feature windows.
        fetcher (callable | None): Optional injectable Qlib-compatible data fetcher.

    Returns:
        tuple[pd.DataFrame, dict]: Factor panel and diagnostic coverage statistics.

    Notes:
        Uses max(windows)+1 prior sessions for rolling daily-return volatility.
        An instrument's pre-membership OHLCV is not an eligible output observation.
    """
    if not windows or min(windows) < 2:
        raise ValueError("windows must contain values >= 2")
    target_keys = target.reorder_levels(["datetime", "instrument"]).sort_index().index
    history, window_calendar = load_extended_history(target, calendar, lookback=max(windows) + 1, fetcher=fetcher)
    first, last = target_keys.get_level_values("datetime").min(), target_keys.get_level_values("datetime").max()
    computed = build_factor_panel(history, window_calendar, str(first.date()), str(last.date()), windows)
    result = computed.reindex(target_keys).replace([np.inf, -np.inf], np.nan)
    if not result.index.equals(target_keys) or result.index.has_duplicates:
        raise AssertionError("Membership preservation failed")
    stats = {"rows": len(result), "instruments": target_keys.get_level_values("instrument").nunique(), "mean_coverage": float(result.notna().mean().mean()), "history_rows": len(history), "history_start": str(window_calendar[0].date())}
    return result, stats


def run_backfilled_factor_pipeline(data_dir: Path, output_dir: Path, calendar: pd.DatetimeIndex, start_year: int = 2015, end_year: int = 2026, overwrite: bool = False) -> pd.DataFrame:
    """Generate annual extended-history factor files without overwriting baseline.

    Parameters:
        data_dir (Path): Original annual dynamic-membership OHLCV Parquet directory.
        output_dir (Path): Separate destination for enhanced factor Parquet files.
        calendar (pd.DatetimeIndex): Full historical exchange calendar.
        start_year (int): First year to generate.
        end_year (int): Last year to generate.
        overwrite (bool): Whether to overwrite enhanced files.

    Returns:
        pd.DataFrame: Annual factor generation statistics and output paths.

    Notes:
        Qlib must be initialized before calling this function.
        Existing baseline factor files are never overwritten.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in range(start_year, end_year + 1):
        destination = output_dir / f"csi300_factors_{year}.parquet"
        if destination.exists() and not overwrite:
            rows.append({"year": year, "status": "cached", "path": str(destination)})
            continue
        target_path = data_dir / f"csi300_daily_{year}.parquet"
        if not target_path.exists():
            raise FileNotFoundError(target_path)
        target = pd.read_parquet(target_path)
        result, stats = generate_backfilled_year(target, calendar)
        result.to_parquet(destination, engine="pyarrow", compression="snappy")
        rows.append({"year": year, "status": "generated", **stats, "path": str(destination)})
        print(f"Enhanced {year}: {len(result):,} rows, coverage={stats['mean_coverage']:.4f}", flush=True)
    return pd.DataFrame(rows)
