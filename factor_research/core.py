"""Auditable factor-label alignment, IC statistics, and forward-return quantiles."""
from __future__ import annotations
import numpy as np
import pandas as pd


def align(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    date_col: str = 'date',
    asset_col: str = 'instrument',
) -> tuple[pd.DataFrame, dict]:
    """Description: Join factor and label observations without imputing missing data.

    Parameters:
        factors (pd.DataFrame): Factor panel with unique signal-date and instrument keys.
        labels (pd.DataFrame): Forward-return panel with unique keys.
        factor_cols (list[str]): Names of candidate factor columns.
        label_col (str): Forward-return label column name.
        date_col (str): Signal-date key column.
        asset_col (str): Instrument key column.
    Returns:
        tuple[pd.DataFrame, dict]: Left-joined factor panel and alignment audit counts.
    Notes:
        Universe eligibility must be validated upstream against PIT membership; a left join does not certify tradability.
    """
    keys = [date_col, asset_col]

    for name, frame in [('factors', factors), ('labels', labels)]:
        if frame.duplicated(keys).any():
            raise ValueError(f'Duplicate keys in {name}')

    if label_col in factors.columns:
        raise ValueError('Label column already exists in factors')
    joined = factors[keys + factor_cols].merge(
        labels[keys + [label_col]], on=keys, how='left', validate='one_to_one', indicator=True
    )
    audit = {
        'factor_rows': len(factors),
        'label_rows': len(labels),
        'matched_rows': int(joined['_merge'].eq('both').sum()),
        'label_finite_rows': int(
            np.isfinite(pd.to_numeric(joined[label_col], errors='coerce')).sum()
        ),
    }

    return joined.drop(columns='_merge'), audit


def daily_ic(
    panel: pd.DataFrame,
    factor_cols: list[str],
    label_col: str,
    date_col: str = 'date',
    min_n: int = 30,
) -> pd.DataFrame:
    """Description: Compute Pearson and tie-aware Spearman correlations per signal date.

    Parameters:
        panel (pd.DataFrame): Aligned factor and label observations.
        factor_cols (list[str]): Factor columns to evaluate.
        label_col (str): Future-return column.
        date_col (str): Signal-date column.
        min_n (int): Minimum finite paired observations per date.
    Returns:
        pd.DataFrame: Daily IC, Rank IC, paired counts, and status by factor.
    Notes:
        Non-finite values are excluded factor-by-factor; constant vectors produce NaN correlations.
    """
    out = []

    for date, day in panel.groupby(date_col, sort=True):
        y = pd.to_numeric(day[label_col], errors='coerce').replace([np.inf, -np.inf], np.nan)

        for factor in factor_cols:
            x = pd.to_numeric(day[factor], errors='coerce').replace([np.inf, -np.inf], np.nan)
            ok = x.notna() & y.notna()
            n = int(ok.sum())
            status = (
                'ok'
                if n >= min_n and x[ok].nunique() > 1 and y[ok].nunique() > 1
                else ('insufficient' if n < min_n else 'constant')
            )

            out.append(
                {
                    date_col: date,
                    'factor': factor,
                    'n_paired': n,
                    'ic': x[ok].corr(y[ok], method='pearson') if status == 'ok' else np.nan,
                    'rank_ic': x[ok].corr(y[ok], method='spearman') if status == 'ok' else np.nan,
                    'status': status,
                }
            )

    return pd.DataFrame(out)


def hac_mean(values: pd.Series, lags: int = 4) -> dict:
    """Description: Estimate mean IC significance using Bartlett-kernel HAC covariance.

    Parameters:
        values (pd.Series): Chronologically ordered daily IC observations.
        lags (int): Maximum included autocovariance lag.
    Returns:
        dict: Mean, sample standard deviation, ICIR, HAC standard error, t-statistic, and count.
    Notes:
        Uses autocovariances normalized by T, equivalent to Bartlett HAC for an intercept-only regression.
        Missing dates are not assumed independent; inspect temporal gaps before inference.
    """
    x = (
        pd.to_numeric(values, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    n = len(x)
    if lags < 0:
        raise ValueError('lags must be nonnegative')

    if n < 2:
        return {
            'n': n,
            'mean': np.nan,
            'std': np.nan,
            'icir': np.nan,
            'hac_se': np.nan,
            'hac_t': np.nan,
        }

    mean = float(x.mean()); std = float(x.std(ddof=1)); e = x - mean
    lag = min(lags, n - 1)
    long_var = float(np.dot(e, e) / n)
    for k in range(1, lag + 1):
        long_var += 2 * (1 - k / (lag + 1)) * float(np.dot(e[k:], e[:-k]) / n)
    se = float(np.sqrt(max(long_var, 0) / n))

    return {
        'n': n,
        'mean': mean,
        'std': std,
        'icir': mean / std if std > 0 else np.nan,
        'hac_se': se,
        'hac_t': mean / se if se > 0 else np.nan,
    }


def quantile_returns(
    panel: pd.DataFrame,
    factor: str,
    label_col: str,
    date_col: str = 'date',
    asset_col: str = 'instrument',
    groups: int = 5,
    min_n: int = 30,
) -> pd.DataFrame:
    """Description: Compute daily equal-weighted forward returns of deterministic factor quantiles.

    Parameters:
        panel (pd.DataFrame): Factor-label panel.
        factor (str): Factor to sort ascending.
        label_col (str): Future-return label.
        date_col (str): Signal date.
        asset_col (str): Instrument identifier for deterministic tie breaking.
        groups (int): Number of equal-count groups.
        min_n (int): Minimum valid observations.
    Returns:
        pd.DataFrame: Daily group returns, group counts, and high-minus-low forward spread.
    Notes:
        These overlapping forward returns are diagnostics, not executable daily portfolio PnL.
    """
    if groups < 2:
        raise ValueError('groups must be >= 2')
    rows = []

    for date, day in panel.groupby(date_col, sort=True):
        z = day[[asset_col, factor, label_col]].copy()
        z[factor] = pd.to_numeric(z[factor], errors='coerce')
        z[label_col] = pd.to_numeric(z[label_col], errors='coerce')
        z = z.replace([np.inf, -np.inf], np.nan).dropna(subset=[factor, label_col])
        if len(z) < max(min_n, groups) or z[factor].nunique() < 2:
            continue
        z = z.sort_values([factor, asset_col], kind='mergesort').reset_index(drop=True)
        z['group'] = np.floor(np.arange(len(z)) * groups / len(z)).astype(int) + 1
        means = z.groupby('group')[label_col].mean(); counts = z.groupby('group')[label_col].size()
        row = {date_col: date, 'n_paired': len(z), 'n_unique_factor': z[factor].nunique()}
        row.update({f'q{q}_return': float(means[q]) for q in range(1, groups + 1)})
        row.update({f'q{q}_n': int(counts[q]) for q in range(1, groups + 1)})
        row['long_short_forward_spread'] = row[f'q{groups}_return'] - row['q1_return']
        rows.append(row)

    return pd.DataFrame(rows)
