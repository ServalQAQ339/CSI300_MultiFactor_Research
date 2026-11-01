"""Compute batch Rank IC, HAC inference and factor redundancy summaries."""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, norm
from statsmodels.stats.multitest import multipletests


def cross_sectional_ranks(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Rank columns independently within each trading-date cross-section.

    Description:
        Map each date's cross-section to centered percentile ranks in (-0.5, 0.5].
    Parameters:
        frame (pd.DataFrame): Stock-date panel with a datetime index level.
        columns (list[str]): Predictor or label columns to transform.
    Returns:
        pd.DataFrame: Same-index centered percentile ranks (average ties, NaN kept).
    Notes:
        Uses same-day information only. Shared by Ridge/LightGBM fitting in factor
        expansion, factor selection and modeling; targets are ranked only for fitting.
    """
    return frame.groupby(level="datetime")[columns].rank(pct=True) - 0.5


def daily_rank_ic(panel: pd.DataFrame, factors: list[str], min_stocks: int = 30) -> pd.DataFrame:
    """Calculate daily cross-sectional Spearman information coefficients.

    Description:
        For each date and factor, keep finite (factor, label) pairs, re-rank both
        within that paired sample and take the Pearson correlation of the ranks.
    Parameters:
        panel (pd.DataFrame): Aligned panel with label_5d and factor columns.
        factors (list[str]): Candidate names to evaluate.
        min_stocks (int): Minimum finite paired observations per date.
    Returns:
        pd.DataFrame: Daily IC values with datetime as the index.
    Notes:
        Standard Spearman: invalid pairs are removed before ranking, ties receive
        average ranks, and constant paired samples produce missing IC.
    """
    rows = []
    for date, day in panel.groupby(level="datetime", sort=True):
        values = day[factors + ["label_5d"]].replace([np.inf, -np.inf], np.nan)
        valid = values[factors].notna().mul(values["label_5d"].notna(), axis=0).sum()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            ic = values[factors].corrwith(values["label_5d"], method="spearman")
        ic = ic.where(valid >= min_stocks)
        ic.name = date
        rows.append(ic)
    return pd.DataFrame(rows).rename_axis("datetime")


def hac_summary(daily: pd.DataFrame, lags: int = 5, fdr: bool = True) -> pd.DataFrame:
    """Summarize IC means and Newey-West HAC test statistics.

    Description:
        Calculate daily ICIR, HAC t-tests and BH-FDR adjusted p-values.
    Parameters:
        daily (pd.DataFrame): Date-by-factor daily Rank IC matrix.
        lags (int): Newey-West truncation lag.
        fdr (bool): Add BH-FDR q-values across columns; False omits the fdr_q column
            (used for matched model comparisons, where no multiple-testing claim is made).
    Returns:
        pd.DataFrame: Per-factor days, mean IC, ICIR, HAC t, p-value and optional fdr_q.
    Notes:
        Missing days are dropped before inference; fewer than three valid days gives NaN.
        Bartlett weights use the requested lags. Normal asymptotic two-sided p-values are
        used; ICIR is not annualized.
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
    if not fdr:
        return result
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
        Mean absolute correlation guards against temporal sign cancellation. Each
        pair uses standard Spearman on its own finite paired sample (re-ranked).
    """
    sums = np.zeros((len(candidates), len(baseline)))
    abs_sums = sums.copy()
    counts = np.zeros_like(sums)
    for _, day in panel.groupby(level="datetime", sort=True):
        values = day[candidates + baseline].replace([np.inf, -np.inf], np.nan)
        corr = values.corr(method="spearman").loc[candidates, baseline].to_numpy()
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
