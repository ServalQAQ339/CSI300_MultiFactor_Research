"""Calendar-preserving Newey-West inference for daily information coefficients."""
import numpy as np
import pandas as pd


def hac_summary(daily, metric, lag=4):
    """Description: Estimate a Bartlett HAC t-statistic retaining missing calendar slots.

    Parameters:
        daily (pd.DataFrame): One row per trading signal date with the metric column.
        metric (str): Column holding daily IC or rank IC.
        lag (int): Nonnegative lag in observed signal-date positions.
    Returns:
        dict: Mean, standard deviation, ICIR, HAC standard error, t-statistic, valid days.
    Notes:
        Missing IC dates remain calendar slots; autocovariance uses valid pairs only,
        normalized by the valid sample count. This is a conservative explicit convention,
        not a general missing-data-robust HAC estimator.
    """
    if lag < 0: raise ValueError('lag must be nonnegative')
    x = pd.to_numeric(daily[metric], errors='coerce').to_numpy(dtype=float)
    x[~np.isfinite(x)] = np.nan
    valid = np.isfinite(x); n = int(valid.sum())
    result = {
        'n': n,
        'mean': np.nan,
        'std': np.nan,
        'icir': np.nan,
        'hac_se': np.nan,
        'hac_t': np.nan,
    }
    if n < 2: return result
    mean = float(np.nanmean(x)); std = float(np.nanstd(x, ddof=1)); e = x - mean
    if std <= 1e-14: result.update(mean=mean, std=0.0); return result
    long_var = float(np.nansum(e * e) / n)
    for k in range(1, min(lag, len(x)-1)+1):
        a, b = e[k:], e[:-k]; mask = np.isfinite(a) & np.isfinite(b)
        long_var += 2 * (1-k/(lag+1)) * float(np.sum(a[mask]*b[mask])/n)
    se = float(np.sqrt(max(long_var, 0)/n))
    result.update(
        mean=mean,
        std=std,
        icir=mean / std if std > 0 else np.nan,
        hac_se=se,
        hac_t=mean / se if se > 0 else np.nan,
    )
    return result
