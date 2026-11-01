"""Independent same-sample IC and quintile-return consistency audit."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .core import hac_mean
from .config import SEGMENTS


def audit_factor(panel: pd.DataFrame, factor: str, min_n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Description: Independently recompute daily rank IC and quintiles on identical paired observations.

    Parameters:
        panel (pd.DataFrame): Label-left joined panel containing signal_date, instrument, research_label_5d and the factor.
        factor (str): Numeric factor column to audit.
        min_n (int): Minimum paired observations for a valid daily cross-section.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Daily diagnostics and split-level descriptive/HAC summaries.
    Notes:
        Equal-count quintiles use factor/instrument stable sorting. Raw returns remain unchanged; trimmed means are diagnostic only.
        Forward five-day returns are overlapping labels, not daily executable portfolio PnL.
    """
    if min_n < 5: raise ValueError('min_n must be >= 5')
    required = {'signal_date', 'instrument', 'research_label_5d', factor}
    if not required.issubset(panel.columns): raise ValueError(f'Missing columns: {required-set(panel.columns)}')
    rows = []
    for date, day in panel.groupby('signal_date', sort=True):
        z = day[['instrument', factor, 'research_label_5d']].copy()
        for col in (factor, 'research_label_5d'): z[col] = pd.to_numeric(z[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        z = z.dropna(subset=[factor, 'research_label_5d'])
        if len(z) < min_n or z[factor].nunique() < 2 or z['research_label_5d'].nunique() < 2: continue
        rank_ic = z[factor].rank(method='average').corr(z['research_label_5d'].rank(method='average'))
        z = z.sort_values([factor, 'instrument'], kind='mergesort').reset_index(drop=True)
        z['quantile'] = (np.floor(np.arange(len(z)) * 5 / len(z)).astype(int) + 1)
        means = z.groupby('quantile')['research_label_5d'].mean(); medians = z.groupby('quantile')['research_label_5d'].median()
        low = z.loc[z['quantile'].eq(1), 'research_label_5d']; high = z.loc[z['quantile'].eq(5), 'research_label_5d']
        def trim_mean(s: pd.Series) -> float:
            v = np.sort(s.to_numpy(dtype=float)); k = int(np.floor(0.05 * len(v))); return float(v[k:len(v)-k].mean()) if len(v) > 2*k else np.nan
        rows.append({'signal_date': date, 'factor': factor, 'n_paired': len(z), 'rank_ic_independent': rank_ic, 'q1': means.loc[1], 'q2': means.loc[2], 'q3': means.loc[3], 'q4': means.loc[4], 'q5': means.loc[5], 'spread_mean': means.loc[5]-means.loc[1], 'spread_median': medians.loc[5]-medians.loc[1], 'spread_trimmed_5pct': trim_mean(high)-trim_mean(low), 'q1_max_return': low.max(), 'q5_max_return': high.max(), 'q1_min_return': low.min(), 'q5_min_return': high.min()})
    daily = pd.DataFrame(rows)
    summary = []
    for split, (start, end) in SEGMENTS.items():
        d = daily.loc[daily.signal_date.between(start, end)] if not daily.empty else daily
        if d.empty: continue
        s = d.spread_mean; r = d.rank_ic_independent
        summary.append({'factor': factor, 'split': split, 'n_days': len(d), 'rank_ic_mean': r.mean(), 'spread_mean': s.mean(), 'spread_median_over_dates': s.median(), 'spread_positive_ratio': (s > 0).mean(), 'mean_daily_median_spread': d.spread_median.mean(), 'mean_daily_trimmed_spread': d.spread_trimmed_5pct.mean(), 'sign_disagreement': bool(np.sign(r.mean()) != np.sign(s.mean())), **{f'spread_{k}': v for k, v in hac_mean(s, 4).items()}})
    return daily, pd.DataFrame(summary)


def compare_saved(daily: pd.DataFrame, original_ic: pd.DataFrame, original_quantiles: pd.DataFrame, tolerance: float = 1e-10) -> pd.DataFrame:
    """Description: Compare independently recomputed daily metrics against saved v1.2 results.

    Parameters:
        daily (pd.DataFrame): Independent daily audit metrics for one factor.
        original_ic (pd.DataFrame): Saved daily IC output, filtered to one factor.
        original_quantiles (pd.DataFrame): Saved daily quantile output for that factor.
        tolerance (float): Maximum acceptable absolute floating-point difference.
    Returns:
        pd.DataFrame: Per-date differences, mismatches and sample-count comparisons.
    Notes:
        A missing saved date is an audit failure. Comparison is performed on daily outputs, not just aggregated means.
    """
    i = original_ic[['signal_date', 'rank_ic', 'n_paired']].rename(columns={'rank_ic':'saved_rank_ic', 'n_paired':'saved_ic_n'})
    q = original_quantiles[['signal_date', 'long_short_forward_spread', 'n_paired']].rename(columns={'long_short_forward_spread':'saved_spread', 'n_paired':'saved_quantile_n'})
    m = daily.merge(i, on='signal_date', how='left', validate='one_to_one').merge(q, on='signal_date', how='left', validate='one_to_one')
    m['rank_ic_abs_diff'] = (m.rank_ic_independent-m.saved_rank_ic).abs(); m['spread_abs_diff'] = (m.spread_mean-m.saved_spread).abs()
    m['pass'] = m[['saved_rank_ic', 'saved_spread', 'saved_ic_n', 'saved_quantile_n']].notna().all(axis=1) & m.rank_ic_abs_diff.le(tolerance) & m.spread_abs_diff.le(tolerance) & m.n_paired.eq(m.saved_ic_n) & m.n_paired.eq(m.saved_quantile_n)
    return m
