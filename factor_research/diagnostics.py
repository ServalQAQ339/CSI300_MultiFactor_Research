"""Independent diagnostics for cross-sectional factor research."""
import numpy as np
import pandas as pd


def ic_diagnostics(
    ic: pd.DataFrame, segments: dict, horizon: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Description: Evaluate IC autocorrelation and five non-overlapping calendar phases.

    Parameters:
        ic (pd.DataFrame): Daily IC panel with signal_date, factor, ic and rank_ic.
        segments (dict): Named inclusive start/end date pairs.
        horizon (int): Forward-return holding period in trading sessions.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ACF diagnostics and phase-level IC means.
    Notes:
        Phase selection uses the common sorted signal-date calendar, not factor-specific valid dates.
        Disjoint windows within one phase need not be statistically independent.
    """
    required = {'signal_date', 'factor', 'ic', 'rank_ic'}
    if not required.issubset(ic.columns):
        raise ValueError(f'Missing daily IC fields: {required-set(ic.columns)}')
    if horizon < 1: raise ValueError('horizon must be positive')
    d = ic.copy(); d['signal_date'] = pd.to_datetime(d['signal_date']).dt.normalize()
    if d.duplicated(['signal_date', 'factor']).any(): raise ValueError('Duplicate date-factor keys')
    dates = pd.Index(sorted(d.signal_date.unique()))
    date_phase = pd.Series(np.arange(len(dates)) % horizon, index=dates)
    d['phase'] = d.signal_date.map(date_phase)
    acf_rows, phase_rows = [], []

    for segment, (start, end) in segments.items():
        subset = d.loc[d.signal_date.between(start, end)]

        for factor, group in subset.groupby('factor'):
            for metric in ('ic', 'rank_ic'):
                group = group.sort_values('signal_date')
                s = (
                    group.set_index('signal_date')[metric]
                    .astype(float)
                    .replace([np.inf, -np.inf], np.nan)
                )

                acf_rows.append(
                    {
                        'split': segment,
                        'factor': factor,
                        'metric': metric,
                        'n_valid': int(s.notna().sum()),
                        **{
                            f'acf_lag_{lag}': (
                                float(s.corr(s.shift(lag))) if s.notna().sum() > lag + 2 else np.nan
                            )
                            for lag in (1, 2, 3, 4, 5, 9, 19)
                        },
                    }
                )

                for phase in range(horizon):
                    vals = (
                        group.loc[group.phase.eq(phase), metric]
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                    )

                    phase_rows.append(
                        {
                            'split': segment,
                            'factor': factor,
                            'metric': metric,
                            'phase': phase,
                            'n_days': len(vals),
                            'mean': float(vals.mean()) if len(vals) else np.nan,
                            'std': float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                        }
                    )

    return pd.DataFrame(acf_rows), pd.DataFrame(phase_rows)


def bh_fdr(pvalues: pd.Series) -> pd.Series:
    """Description: Adjust finite two-sided p-values with Benjamini-Hochberg FDR.

    Parameters:
        pvalues (pd.Series): Unadjusted p-values for one prespecified test family.
    Returns:
        pd.Series: Adjusted q-values preserving original index and missing values.
    Notes:
        Does not correct for arbitrary dependence among tests; report family definition.
    """
    p = pd.to_numeric(pvalues, errors='coerce')
    good = p.notna() & np.isfinite(p) & p.between(0, 1)
    result = pd.Series(np.nan, index=p.index, dtype=float)
    vals = p.loc[good].to_numpy(); m = len(vals)
    if not m: return result
    order = np.argsort(vals, kind='stable')
    ranked = vals[order]
    adjusted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1].clip(0, 1)
    restored = np.empty(m)
    restored[order] = adjusted
    result.loc[good] = restored

    return result


def daily_factor_correlations(
    factors: pd.DataFrame, factor_cols: list[str], min_n: int = 30
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Description: Average within-day pairwise Spearman correlations without pooling years.

    Parameters:
        factors (pd.DataFrame): Factor panel containing signal_date and numeric columns.
        factor_cols (list[str]): Explicit feature names to evaluate.
        min_n (int): Minimum finite paired stocks for each daily factor pair.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Mean correlation matrix and paired-day count matrix.
    Notes:
        Correlations are computed on pairwise-complete stocks; ties use average ranks.
        Dates contribute equal weight; matrix is descriptive and not guaranteed PSD.
    """
    if min_n < 3: raise ValueError('min_n must be at least 3')
    names = list(factor_cols)
    sums = np.zeros((len(names), len(names)))
    counts = np.zeros_like(sums, dtype=int)

    for _, group in factors.groupby('signal_date', sort=True):
        x = group[names].replace([np.inf, -np.inf], np.nan)
        valid = x.notna().astype(int).T.dot(x.notna().astype(int))
        corr = x.corr(method='spearman', min_periods=min_n)
        a = corr.to_numpy()
        eligible = np.isfinite(a) & (valid.to_numpy() >= min_n)
        sums[eligible] += a[eligible]
        counts[eligible] += 1

    avg = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts>0)

    return pd.DataFrame(avg, index=names, columns=names), pd.DataFrame(
        counts, index=names, columns=names
    )


def quantile_diagnostics(frame: pd.DataFrame, factor: str, segments: dict) -> pd.DataFrame:
    """Description: Summarize five-bin forward-return gradients across signal dates.

    Parameters:
        frame (pd.DataFrame): Existing v1.2 quantile output with q1_return through q5_return.
        factor (str): Factor identifier.
        segments (dict): Named inclusive date boundaries.
    Returns:
        pd.DataFrame: Five quantile means, long-short spread, and monotonicity diagnostics.
    Notes:
        Mean forward returns are not daily portfolio returns or annualized PnL.
    """
    required = {
        'signal_date',
        'q1_return',
        'q2_return',
        'q3_return',
        'q4_return',
        'q5_return',
        'long_short_forward_spread',
    }

    if not required.issubset(frame.columns):
        raise ValueError(f'Unsupported quantile schema: {list(frame.columns)}')
    d = frame.copy(); d['signal_date'] = pd.to_datetime(d['signal_date']); rows = []

    for segment, (start, end) in segments.items():
        p = d.loc[d.signal_date.between(start, end)]
        means = [p[f'q{i}_return'].mean() for i in range(1, 6)]
        rows.append(
            {
                'factor': factor,
                'split': segment,
                'n_days': len(p),
                **{f'q{i}_mean': float(means[i - 1]) for i in range(1, 6)},
                'q5_minus_q1': float(p.long_short_forward_spread.mean()),
                'monotone_increasing': bool(all(means[i] <= means[i + 1] for i in range(4))),
                'monotone_decreasing': bool(all(means[i] >= means[i + 1] for i in range(4))),
            }
        )

    return pd.DataFrame(rows)
