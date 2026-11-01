"""Compute batch Rank IC, HAC inference and factor redundancy summaries."""

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def daily_rank_ic(panel: pd.DataFrame, factors: list[str], min_stocks: int = 30) -> pd.DataFrame:
    """Calculate vectorized daily Spearman information coefficients.

    Description:
        Rank each cross-section and correlate all features with label ranks.
    Parameters:
        panel (pd.DataFrame): Aligned panel with label_5d and factor columns.
        factors (list[str]): Candidate names to evaluate.
        min_stocks (int): Minimum pairwise-valid observations per date.
    Returns:
        pd.DataFrame: Daily IC values with datetime as the index.
    Notes:
        Pairwise deletion is used; constant columns produce missing IC.
    """
    rows = []
    for date, day in panel.groupby(level="datetime", sort=True):
        ranked = day[factors + ["label_5d"]].replace([np.inf, -np.inf], np.nan).rank()
        valid = ranked[factors].notna().mul(ranked["label_5d"].notna(), axis=0).sum()
        ic = ranked[factors].corrwith(ranked["label_5d"]).where(valid >= min_stocks)
        ic.name = date
        rows.append(ic)
    return pd.DataFrame(rows).rename_axis("datetime")


def hac_summary(daily: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Summarize IC means and Newey-West HAC test statistics.

    Description:
        Calculate daily ICIR, HAC t-tests and BH-FDR adjusted p-values.
    Parameters:
        daily (pd.DataFrame): Date-by-factor daily Rank IC matrix.
        lags (int): Newey-West truncation lag.
    Returns:
        pd.DataFrame: Per-factor summary with inference and FDR flags.
    Notes:
        Normal asymptotic two-sided p-values are used; ICIR is not annualized.
    """
    rows = []
    for factor in daily:
        x = daily[factor].dropna().to_numpy(dtype=float)
        n = len(x)
        if n < 3:
            rows.append({"factor": factor, "days": n, "mean_rank_ic": np.nan, "icir": np.nan, "hac_t": np.nan, "p_value": np.nan})
            continue
        mean = x.mean()
        centered = x - mean
        var = centered @ centered / n
        for lag in range(1, min(lags, n - 1) + 1):
            var += 2 * (1 - lag / (lags + 1)) * (centered[lag:] @ centered[:-lag]) / n
        se = np.sqrt(max(var, 0) / n)
        t = mean / se if se > 0 else np.nan
        rows.append({"factor": factor, "days": n, "mean_rank_ic": mean, "icir": mean / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan, "hac_t": t, "p_value": 2 * norm.sf(abs(t)) if np.isfinite(t) else np.nan})
    result = pd.DataFrame(rows)
    result["fdr_q"] = np.nan
    valid = result["p_value"].notna()
    if valid.any():
        result.loc[valid, "fdr_q"] = multipletests(result.loc[valid, "p_value"], method="fdr_bh")[1]
    return result


def redundancy(panel: pd.DataFrame, baseline: list[str], candidates: list[str]) -> pd.DataFrame:
    """Compute candidate-versus-baseline daily cross-sectional correlations.

    Description:
        Average datewise Spearman coefficients for all candidate/base pairs.
    Parameters:
        panel (pd.DataFrame): Joined research panel.
        baseline (list[str]): Original27 names.
        candidates (list[str]): Candidate feature names.
    Returns:
        pd.DataFrame: Mean correlation and mean absolute daily correlation.
    Notes:
        Mean absolute correlation guards against temporal sign cancellation.
    """
    sums = np.zeros((len(candidates), len(baseline)))
    abs_sums = sums.copy()
    counts = np.zeros_like(sums)
    for _, day in panel.groupby(level="datetime", sort=True):
        corr = day[candidates + baseline].rank().corr().loc[candidates, baseline].to_numpy()
        good = np.isfinite(corr)
        sums += np.where(good, corr, 0)
        abs_sums += np.where(good, abs(corr), 0)
        counts += good
    rows = []
    for i, candidate in enumerate(candidates):
        for j, base in enumerate(baseline):
            n = counts[i, j]
            rows.append({"candidate": candidate, "baseline": base, "mean_spearman": sums[i, j] / n if n else np.nan, "mean_abs_spearman": abs_sums[i, j] / n if n else np.nan, "days": int(n)})
    return pd.DataFrame(rows)
