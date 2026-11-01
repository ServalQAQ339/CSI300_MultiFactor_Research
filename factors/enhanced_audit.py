"""Audit baseline and extended-history CSI300 factors without altering source files."""

from pathlib import Path
import numpy as np
import pandas as pd
from factors.factor_backfill import generate_backfilled_year
from factors.factor_engine import DEFAULT_WINDOWS


def compare_factor_frames(baseline: pd.DataFrame, enhanced: pd.DataFrame, price: pd.DataFrame, year: int, rtol: float = 1e-8, atol: float = 1e-10) -> tuple[pd.DataFrame, dict]:
    """Compare yearly factor keys, coverage and previously valid values.

    Description:
        Quantify newly recovered factors, lost factors and changed finite values.
        Index equality is verified as a set and order-independent alignment is used.

    Parameters:
        baseline (pd.DataFrame): Original yearly factor panel.
        enhanced (pd.DataFrame): Backfilled yearly factor panel.
        price (pd.DataFrame): Original dynamic-universe price panel.
        year (int): Calendar year being audited.
        rtol (float): Relative numeric comparison tolerance.
        atol (float): Absolute numeric comparison tolerance.

    Returns:
        tuple[pd.DataFrame, dict]: Per-factor metrics and yearly integrity summary.

    Notes:
        Changed previously valid values are diagnostic, not automatically errors.
        A strict improvement expectation is not imposed on recomputed windows.
    """
    for name, frame in (("baseline", baseline), ("enhanced", enhanced), ("price", price)):
        if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {"datetime", "instrument"} or frame.index.has_duplicates:
            raise ValueError(f"Invalid or duplicate {name} index")
    b, e, p = (frame.reorder_levels(["datetime", "instrument"]).sort_index() for frame in (baseline, enhanced, price))
    same_columns = set(b.columns) == set(e.columns)
    if not same_columns:
        raise ValueError(f"Factor column mismatch: baseline_only={set(b.columns)-set(e.columns)}, enhanced_only={set(e.columns)-set(b.columns)}")
    keys_match = b.index.equals(e.index) and b.index.equals(p.index)
    if not keys_match:
        raise ValueError(f"Index mismatch in {year}: baseline={len(b)}, enhanced={len(e)}, price={len(p)}")
    factors = sorted(b.columns)
    metrics = []
    for factor in factors:
        x = pd.to_numeric(b[factor], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(e[factor], errors="coerce").to_numpy(dtype=float)
        valid_x, valid_y = np.isfinite(x), np.isfinite(y)
        both = valid_x & valid_y
        equal = np.isclose(x[both], y[both], rtol=rtol, atol=atol)
        diff = np.abs(x[both] - y[both])
        metrics.append({"year": year, "factor": factor, "rows": len(b), "baseline_valid": int(valid_x.sum()), "enhanced_valid": int(valid_y.sum()), "baseline_coverage": float(valid_x.mean()), "enhanced_coverage": float(valid_y.mean()), "coverage_delta_pp": float(100 * (valid_y.mean() - valid_x.mean())), "recovered": int((~valid_x & valid_y).sum()), "lost": int((valid_x & ~valid_y).sum()), "both_valid": int(both.sum()), "changed_existing": int((~equal).sum()), "max_abs_diff_existing": float(diff.max()) if diff.size else np.nan, "baseline_inf": int(np.isinf(x).sum()), "enhanced_inf": int(np.isinf(y).sum())})
    details = pd.DataFrame(metrics)
    summary = {"year": year, "rows": len(b), "factor_columns": len(factors), "keys_match": keys_match, "baseline_mean_coverage": float(details["baseline_coverage"].mean()), "enhanced_mean_coverage": float(details["enhanced_coverage"].mean()), "recovered_cells": int(details["recovered"].sum()), "lost_cells": int(details["lost"].sum()), "changed_existing_cells": int(details["changed_existing"].sum()), "enhanced_inf_cells": int(details["enhanced_inf"].sum())}
    return details, summary


def audit_prefix_consistency(target: pd.DataFrame, full_calendar: pd.DatetimeIndex, enhanced: pd.DataFrame, year: int, sample_dates: int = 3, sample_instruments: int = 5, seed: int = 42, rtol: float = 1e-8, atol: float = 1e-10) -> pd.DataFrame:
    """Recompute sampled historical signals using Qlib data truncated at each signal date.

    Description:
        Sample dates and constituent instruments, recompute features from historical
        prices ending on each sampled date, and compare against stored enhanced factors.

    Parameters:
        target (pd.DataFrame): Original yearly membership and OHLCV panel.
        full_calendar (pd.DatetimeIndex): Qlib calendar covering the prior 61 sessions.
        enhanced (pd.DataFrame): Stored enhanced factor panel.
        year (int): Audit year.
        sample_dates (int): Number of dates sampled per year.
        sample_instruments (int): Maximum number of eligible stocks per sampled date.
        seed (int): Reproducible random seed.
        rtol (float): Relative comparison tolerance.
        atol (float): Absolute comparison tolerance.

    Returns:
        pd.DataFrame: Per-date and stock mismatch counts and maximum differences.

    Notes:
        This detects dependence on future observations in factor calculations, but
        cannot prove that retrospectively adjusted Qlib prices were point-in-time
        available. Qlib requests are repeated and can be relatively expensive.
    """
    from data.history_loader import load_extended_history
    from factors.factor_engine import build_factor_panel
    rng = np.random.default_rng(seed + year)
    target = target.reorder_levels(["datetime", "instrument"]).sort_index()
    enhanced = enhanced.reorder_levels(["datetime", "instrument"]).sort_index()
    dates = pd.DatetimeIndex(target.index.get_level_values("datetime").unique()).sort_values()
    chosen_dates = sorted(rng.choice(dates.to_numpy(), size=min(sample_dates, len(dates)), replace=False))
    rows = []
    for date_value in chosen_dates:
        date = pd.Timestamp(date_value)
        day = target.xs(date, level="datetime", drop_level=False)
        instruments = day.index.get_level_values("instrument").unique().to_numpy()
        chosen = rng.choice(instruments, size=min(sample_instruments, len(instruments)), replace=False)
        subset = day.loc[day.index.get_level_values("instrument").isin(chosen)]
        history, calendar = load_extended_history(subset, full_calendar, lookback=max(DEFAULT_WINDOWS) + 1)
        recomputed = build_factor_panel(history, calendar, str(date.date()), str(date.date())).reindex(subset.index)
        expected = enhanced.reindex(subset.index)
        for key in subset.index:
            a = recomputed.loc[key].to_numpy(dtype=float)
            b = expected.loc[key].to_numpy(dtype=float)
            same_missing = np.isnan(a) == np.isnan(b)
            both = np.isfinite(a) & np.isfinite(b)
            numeric_equal = np.isclose(a[both], b[both], rtol=rtol, atol=atol)
            differences = np.abs(a[both] - b[both])
            rows.append({"year": year, "datetime": date, "instrument": key[1], "cells": len(a), "missing_mismatch": int((~same_missing).sum()), "value_mismatch": int((~numeric_equal).sum()), "max_abs_diff": float(differences.max()) if differences.size else np.nan})
    return pd.DataFrame(rows)


def run_enhanced_audit(data_dir: Path, baseline_dir: Path, enhanced_dir: Path, report_dir: Path, calendar: pd.DatetimeIndex | None = None, prefix_check: bool = True, sample_dates: int = 3, sample_instruments: int = 5) -> dict[str, pd.DataFrame]:
    """Audit all available enhanced factor years and export CSV diagnostics.

    Parameters:
        data_dir (Path): Original yearly OHLCV files directory.
        baseline_dir (Path): Original yearly factor files directory.
        enhanced_dir (Path): Backfilled yearly factor files directory.
        report_dir (Path): Destination for CSV audit reports.
        calendar (pd.DatetimeIndex | None): Full historical exchange calendar.
        prefix_check (bool): Whether to recompute randomly sampled historical prefixes.
        sample_dates (int): Number of dates per year for the prefix audit.
        sample_instruments (int): Stocks per sampled date for the prefix audit.

    Returns:
        dict[str, pd.DataFrame]: Year summary, factor comparison and prefix checks.

    Notes:
        Files are loaded one year at a time. No source data are modified.
        A mismatch in keys or factor columns stops the audit immediately.
    """
    data_dir, baseline_dir, enhanced_dir, report_dir = map(Path, (data_dir, baseline_dir, enhanced_dir, report_dir))
    if prefix_check and calendar is None:
        raise ValueError("Full historical calendar is required for prefix checks")
    summaries, comparisons, prefixes = [], [], []
    files = sorted(enhanced_dir.glob("csi300_factors_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No enhanced factor files in {enhanced_dir}")
    for file in files:
        year = int(file.stem.split("_")[-1])
        baseline_file = baseline_dir / file.name
        price_file = data_dir / f"csi300_daily_{year}.parquet"
        if not baseline_file.exists() or not price_file.exists():
            raise FileNotFoundError(f"Missing baseline or price file for {year}")
        baseline, enhanced, price = pd.read_parquet(baseline_file), pd.read_parquet(file), pd.read_parquet(price_file)
        comparison, summary = compare_factor_frames(baseline, enhanced, price, year)
        comparisons.append(comparison)
        summaries.append(summary)
        if prefix_check:
            prefixes.append(audit_prefix_consistency(price, calendar, enhanced, year, sample_dates, sample_instruments))
        print(f"Audited {year}: recovered={summary['recovered_cells']:,}, lost={summary['lost_cells']:,}, changed={summary['changed_existing_cells']:,}", flush=True)
    results = {"year_comparison": pd.DataFrame(summaries), "factor_comparison": pd.concat(comparisons, ignore_index=True)}
    if prefix_check:
        results["prefix_consistency"] = pd.concat(prefixes, ignore_index=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        destination = report_dir / f"{name}.csv"
        frame.to_csv(destination, index=False)
        print(f"Saved {destination}", flush=True)
    return results
