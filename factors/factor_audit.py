"""Audit factor coverage, calendar continuity, and source-panel consistency."""

from pathlib import Path
import numpy as np
import pandas as pd


def load_annual_panels(directory: Path, prefix: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Load and concatenate annual research panels.

    Parameters:
        directory (Path): Folder containing annual Parquet files.
        prefix (str): Filename prefix ending before the year.
        start_year (int): First included calendar year.
        end_year (int): Last included calendar year.

    Returns:
        pd.DataFrame: Chronologically sorted panel indexed by datetime and instrument.

    Notes:
        Raises an error for missing files or duplicated keys instead of silently repairing them.
    """
    paths = [Path(directory) / f"{prefix}{year}.parquet" for year in range(start_year, end_year + 1)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing annual files: {missing}")
    frames = [pd.read_parquet(path) for path in paths]
    panel = pd.concat(frames).reorder_levels(["datetime", "instrument"]).sort_index()
    if panel.index.has_duplicates:
        raise ValueError(f"Duplicate keys in {prefix} panels")
    return panel


def audit_factor_panels(prices: pd.DataFrame, factors: pd.DataFrame, calendar: pd.DatetimeIndex, entry_window: int = 60, boundary_window: int = 20) -> dict[str, pd.DataFrame]:
    """Produce independent data-integrity and factor-coverage audit tables.

    Parameters:
        prices (pd.DataFrame): Original dynamic-universe daily OHLCV panel.
        factors (pd.DataFrame): Generated factor panel with matching observation keys.
        calendar (pd.DatetimeIndex): Official market sessions covering the full research period.
        entry_window (int): Number of market sessions after a stock's first observed panel row.
        boundary_window (int): Number of market sessions used to analyze annual boundaries.

    Returns:
        dict[str, pd.DataFrame]: Tables named integrity, factor_year, year_summary,
            year_boundary, first_observed, stock_year, and cross_section.

    Notes:
        First observed row is not necessarily actual index entry or listing date.
        Nonmissing coverage does not establish point-in-time validity or factor predictive power.
        No imputation, winsorization, or future-label construction is performed.
    """
    if entry_window < 1 or boundary_window < 1:
        raise ValueError("Audit windows must be positive")
    if not isinstance(prices.index, pd.MultiIndex) or not isinstance(factors.index, pd.MultiIndex):
        raise ValueError("Both panels must have MultiIndex keys")
    prices = prices.reorder_levels(["datetime", "instrument"]).sort_index()
    factors = factors.reorder_levels(["datetime", "instrument"]).sort_index()
    calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    if calendar.empty or not factors.index.get_level_values("datetime").isin(calendar).all():
        raise ValueError("Trading calendar does not cover factor observations")
    numeric = factors.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("No numeric factor columns found")
    dates = factors.index.get_level_values("datetime")
    instruments = factors.index.get_level_values("instrument")
    years = dates.year
    finite = pd.DataFrame(np.isfinite(numeric.to_numpy(dtype="float64")), index=numeric.index, columns=numeric.columns)
    valid = finite & numeric.notna()
    index_only_factor = factors.index.difference(prices.index)
    index_only_price = prices.index.difference(factors.index)
    integrity = pd.DataFrame([{"price_rows": len(prices), "factor_rows": len(factors), "factor_only_rows": len(index_only_factor), "price_only_rows": len(index_only_price), "duplicate_factor_keys": int(factors.index.duplicated().sum()), "duplicate_price_keys": int(prices.index.duplicated().sum()), "factor_nonfinite_cells": int((~finite & numeric.notna()).to_numpy().sum()), "factor_columns": len(numeric.columns), "unique_stocks": instruments.nunique(), "unique_dates": dates.nunique()}])
    factor_year_rows = []
    stock_year_rows = []
    for year in sorted(years.unique()):
        mask = years == year
        frame = numeric.loc[mask]
        v = valid.loc[mask]
        for col in numeric.columns:
            s = frame[col]
            finite_s = s.where(v[col])
            factor_year_rows.append({"year": int(year), "factor": col, "rows": len(s), "valid_count": int(v[col].sum()), "coverage": float(v[col].mean()), "nan_count": int(s.isna().sum()), "inf_count": int(np.isinf(s.to_numpy(dtype="float64")).sum()), "unique_valid_values": int(finite_s.nunique()), "is_constant": bool(finite_s.nunique() <= 1), "mean": float(finite_s.mean()), "std": float(finite_s.std()), "p01": float(finite_s.quantile(0.01)), "median": float(finite_s.median()), "p99": float(finite_s.quantile(0.99))})
        by_stock = v.groupby(level="instrument").mean()
        for stock, row in by_stock.iterrows():
            stock_year_rows.append({"year": int(year), "instrument": stock, "rows": int((instruments[mask] == stock).sum()), "mean_coverage": float(row.mean()), "min_factor_coverage": float(row.min())})
    factor_year = pd.DataFrame(factor_year_rows)
    stock_year = pd.DataFrame(stock_year_rows)
    year_summary = factor_year.groupby("year", as_index=False).agg(mean_factor_coverage=("coverage", "mean"), min_factor_coverage=("coverage", "min"), constant_factor_count=("is_constant", "sum"), total_inf_cells=("inf_count", "sum"))
    year_summary["rows"] = [int((years == year).sum()) for year in year_summary["year"]]
    boundary_rows = []
    for year in sorted(years.unique()):
        year_dates = calendar[calendar.year == year]
        first_dates = year_dates[:boundary_window]
        mask = years == year
        first_mask = mask & dates.isin(first_dates)
        rest_mask = mask & ~dates.isin(first_dates)
        for col in numeric.columns:
            boundary_rows.append({"year": int(year), "factor": col, "first_sessions_rows": int(first_mask.sum()), "first_sessions_coverage": float(valid.loc[first_mask, col].mean()) if first_mask.any() else np.nan, "remaining_rows": int(rest_mask.sum()), "remaining_coverage": float(valid.loc[rest_mask, col].mean()) if rest_mask.any() else np.nan})
    year_boundary = pd.DataFrame(boundary_rows)
    session_position = pd.Series(np.arange(len(calendar)), index=calendar)
    first_seen = pd.DataFrame({"instrument": instruments, "datetime": dates}).groupby("instrument")["datetime"].min()
    first_seen_pos = dict(zip(first_seen.index, session_position.reindex(first_seen).to_numpy()))
    observed_pos = session_position.reindex(dates).to_numpy()
    initial_pos = np.array([first_seen_pos[stock] for stock in instruments])
    age = observed_pos - initial_pos
    first_rows = []
    for window_label, mask in (("first_observed_0_to_19", (age >= 0) & (age < min(entry_window, 20))), ("first_observed_0_to_59", (age >= 0) & (age < entry_window)), ("later_observations", age >= entry_window)):
        for col in numeric.columns:
            first_rows.append({"cohort": window_label, "factor": col, "rows": int(mask.sum()), "coverage": float(valid.loc[mask, col].mean()) if mask.any() else np.nan})
    first_observed = pd.DataFrame(first_rows)
    cross_section = pd.DataFrame({"datetime": dates, "valid_factor_count": valid.sum(axis=1).to_numpy()}).groupby("datetime", as_index=False).agg(stocks=("valid_factor_count", "size"), mean_valid_factors=("valid_factor_count", "mean"), zero_valid_factor_stocks=("valid_factor_count", lambda s: int((s == 0).sum())))
    return {"integrity": integrity, "factor_year": factor_year, "year_summary": year_summary, "year_boundary": year_boundary, "first_observed": first_observed, "stock_year": stock_year, "cross_section": cross_section}


def run_factor_audit(price_dir: Path, factor_dir: Path, report_dir: Path, calendar: pd.DatetimeIndex, start_year: int, end_year: int) -> dict[str, pd.DataFrame]:
    """Run the full audit and export each report as a CSV file.

    Parameters:
        price_dir (Path): Directory containing annual source price panels.
        factor_dir (Path): Directory containing annual computed factor panels.
        report_dir (Path): Destination for factor audit CSV reports.
        calendar (pd.DatetimeIndex): Official trading calendar.
        start_year (int): First audited year.
        end_year (int): Last audited year.

    Returns:
        dict[str, pd.DataFrame]: Audit tables also persisted as UTF-8 CSV files.

    Notes:
        This audit checks observed output integrity and coverage, not causality,
        actual index membership announcements, or the adjustment-factor vintage.
    """
    prices = load_annual_panels(price_dir, "csi300_daily_", start_year, end_year)
    factors = load_annual_panels(factor_dir, "csi300_factors_", start_year, end_year)
    reports = audit_factor_panels(prices, factors, calendar)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        frame.to_csv(report_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        print(f"Saved {name}: {len(frame):,} rows", flush=True)
    return reports
