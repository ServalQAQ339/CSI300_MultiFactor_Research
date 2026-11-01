"""Align panels and evaluate cross-sectional predictive information."""

import numpy as np
import pandas as pd


CANDIDATES = [f"{family}{window}" for family in ("CORR", "VSTD", "RSV") for window in (5, 10, 20)]


def align_panels(original: pd.DataFrame, expansion: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Align factor and forward-label observations.

    Description:
        Join the frozen baseline, new candidates, and valid five-day labels.
    Parameters:
        original (pd.DataFrame): Baseline indexed by datetime and instrument.
        expansion (pd.DataFrame): New factors with the same index convention.
        labels (pd.DataFrame): Label panel with signal_date and instrument keys.
    Returns:
        pd.DataFrame: Joined observations on the original factor universe.
    Notes:
        Never forward-fill missing factors or labels. Label statuses must be valid.
    """
    for name, frame in (("original", original), ("expansion", expansion)):
        if list(frame.index.names) != ["datetime", "instrument"] or not frame.index.is_unique:
            raise ValueError(f"{name} must have unique (datetime, instrument) index")
    required = {"signal_date", "instrument", "label_5d", "label_status_5d", "split"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Missing label columns: {sorted(required - set(labels.columns))}")
    label_panel = labels.copy()
    label_panel["signal_date"] = pd.to_datetime(label_panel["signal_date"])
    label_panel = label_panel.set_index(["signal_date", "instrument"])
    label_panel.index.names = ["datetime", "instrument"]
    if not label_panel.index.is_unique:
        raise ValueError("Duplicate label keys")

    combined = original.join(expansion[CANDIDATES], how="left", validate="one_to_one")
    combined = combined.join(label_panel[["label_5d", "label_status_5d", "split"]], how="left", validate="one_to_one")
    return combined


def daily_rank_ic(frame: pd.DataFrame, factors: list[str], minimum: int = 30) -> pd.DataFrame:
    """Calculate daily cross-sectional Spearman information coefficients.

    Description:
        Correlate factor ranks with forward-return ranks for each date.
    Parameters:
        frame (pd.DataFrame): Joined panel with datetime index level and label_5d.
        factors (list[str]): Factor names to evaluate.
        minimum (int): Minimum valid stock pairs on each date.
    Returns:
        pd.DataFrame: Date, factor, valid sample count, and Rank IC.
    Notes:
        Returns NaN for constant or insufficient cross-sections.
    """
    records = []
    for date, day in frame.groupby(level="datetime", sort=True):
        for factor in factors:
            pairs = day[[factor, "label_5d"]].replace([np.inf, -np.inf], np.nan).dropna()
            ic = pairs[factor].corr(pairs["label_5d"], method="spearman") if len(pairs) >= minimum else np.nan
            records.append({"datetime": date, "factor": factor, "n": len(pairs), "rank_ic": ic})
    return pd.DataFrame.from_records(records)


def summarize_ic(daily: pd.DataFrame, hac_lags: int = 5) -> pd.DataFrame:
    """Summarize daily Rank IC with autocorrelation-robust inference.

    Description:
        Compute mean IC, ICIR, and Newey-West HAC t-statistics.
    Parameters:
        daily (pd.DataFrame): Output of daily_rank_ic.
        hac_lags (int): Maximum Newey-West autocovariance lag.
    Returns:
        pd.DataFrame: Factor-level summary statistics.
    Notes:
        ICIR is daily mean divided by daily standard deviation; not annualized.
    """
    rows = []
    for factor, group in daily.groupby("factor"):
        values = group["rank_ic"].dropna().to_numpy(dtype=float)
        count = len(values)
        mean = float(values.mean()) if count else np.nan
        std = float(values.std(ddof=1)) if count > 1 else np.nan
        icir = mean / std if std and np.isfinite(std) else np.nan
        centered = values - mean
        variance = np.dot(centered, centered) / count if count else np.nan
        for lag in range(1, min(hac_lags, count - 1) + 1):
            covariance = np.dot(centered[lag:], centered[:-lag]) / count
            variance += 2 * (1 - lag / (hac_lags + 1)) * covariance
        se = np.sqrt(max(variance, 0) / count) if count else np.nan
        t_stat = mean / se if se and np.isfinite(se) else np.nan
        rows.append({"factor": factor, "days": count, "mean_rank_ic": mean, "icir": icir, "hac_t": t_stat})
    return pd.DataFrame(rows)


def factor_correlations(frame: pd.DataFrame, baseline: list[str], candidates: list[str]) -> pd.DataFrame:
    """Measure time-averaged cross-sectional factor redundancy.

    Description:
        Compute per-date Spearman correlations of each candidate with each baseline.
    Parameters:
        frame (pd.DataFrame): Aligned panel.
        baseline (list[str]): Original27 names.
        candidates (list[str]): Expansion factor names.
    Returns:
        pd.DataFrame: Mean correlations and contributing date counts.
    Notes:
        Uses paired observations separately for every factor pair.
    """
    records = []
    for candidate in candidates:
        for old in baseline:
            daily = []
            for _, day in frame.groupby(level="datetime", sort=True):
                pairs = day[[candidate, old]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(pairs) >= 30:
                    daily.append(pairs[candidate].corr(pairs[old], method="spearman"))
            valid = pd.Series(daily, dtype=float).dropna()
            records.append({"candidate": candidate, "baseline": old, "mean_spearman": valid.mean(), "days": len(valid)})
    return pd.DataFrame(records)
