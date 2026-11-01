"""Synthetic tests for independent factor value verification."""
import numpy as np
import pandas as pd
import pytest
from factor_verification.verification import (
    FACTORS,
    compare_values,
    independent_factors,
    normalize_panel,
    recompute_panel,
)


def sample_panel(n=100):
    """Description: Create synthetic, calendar-aligned OHLCV data.

    Parameters:
        n (int): Number of sessions.
    Returns:
        pd.DataFrame: Valid synthetic OHLCV indexed by dates.
    Notes:
        Deterministic values permit exact formula checks.
    """
    dates = pd.bdate_range('2020-01-01', periods=n)
    close = pd.Series(100 + np.arange(n, dtype=float), index=dates)
    return pd.DataFrame(
        {
            'open': close - 1,
            'high': close + 2,
            'low': close - 3,
            'close': close,
            'volume': 1000 + np.arange(n),
        },
        index=dates,
    )


def test_formulas_and_warmup():
    """Description: Verify 27 formulas, 60-session warm-up, and selected known values.

    Parameters:
        None: Uses synthetic panel.
    Returns:
        None: Raises assertion errors on mismatch.
    Notes:
        The 60-day return volatility requires 61 prices.
    """
    panel = sample_panel()
    f = independent_factors(panel)
    assert list(f.columns) == list(FACTORS)
    assert pd.isna(f.iloc[59]['VOLATILITY_60'])
    assert np.isfinite(f.iloc[60]['VOLATILITY_60'])
    assert np.isclose(f.iloc[20]['MOM_20'], 120 / 100 - 1)
    assert np.isclose(f.iloc[20]['OPEN_CLOSE_RATIO'], 119 / 120 - 1)
    assert np.isclose(f.iloc[20]['HIGH_POSITION_20'], 120 / 122 - 1)


def test_comparison_detects_missingness_and_numeric_error():
    """Description: Ensure missingness and numeric differences are independently counted.

    Parameters:
        None: Uses synthetic factors.
    Returns:
        None: Raises assertion errors on mismatch.
    Notes:
        Both-NaN pairs are treated as equal.
    """
    panel = sample_panel()
    f = independent_factors(panel)
    f['instrument'] = 'SH600000'
    f.index.name = 'datetime'
    f = f.reset_index().set_index(['datetime', 'instrument'])
    altered = f.copy()
    altered.iloc[70, altered.columns.get_loc('MOM_5')] += 0.01
    altered.iloc[71, altered.columns.get_loc('MOM_5')] = np.nan
    summary, examples = compare_values(f, altered)
    row = summary.set_index('factor').loc['MOM_5']
    assert row.numeric_fail == 1 and row.missing_mismatch == 1 and row.failed == 2
    assert len(examples) == 2


def test_duplicate_keys_rejected():
    """Description: Reject duplicate (datetime, instrument) keys.

    Parameters:
        None: Uses synthetic duplicate rows.
    Returns:
        None: Raises assertion errors on mismatch.
    Notes:
        Duplicate keys must not be silently deduplicated.
    """
    df = pd.DataFrame({'datetime': ['2020-01-01'] * 2, 'instrument': ['A'] * 2, 'open': [1, 2]})
    with pytest.raises(ValueError, match='Duplicate'):
        normalize_panel(df, ('open',))


def test_calendar_gap_does_not_compress_rolling_window():
    """Description: Ensure a missing exchange session invalidates complete rolling windows.

    Parameters:
        None: Uses a synthetic calendar gap.
    Returns:
        None: Raises assertion errors on mismatch.
    Notes:
        Reindexing must precede factor computation.
    """
    panel = sample_panel(80)
    missing_day = panel.index[65]
    source = panel.drop(missing_day).copy()
    source['instrument'] = 'A'
    source.index.name = 'datetime'
    source = source.reset_index().set_index(['datetime', 'instrument'])
    target = pd.MultiIndex.from_tuples([(panel.index[70], 'A')], names=['datetime', 'instrument'])
    recomputed = recompute_panel(source, panel.index, target)
    assert np.isnan(recomputed.iloc[0]['VOLATILITY_20'])
