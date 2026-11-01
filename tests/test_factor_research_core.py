"""Deterministic unit tests for factor evaluation edge cases."""
import numpy as np
import pandas as pd
import pytest
from factor_research.core import align, daily_ic, hac_mean, quantile_returns


def test_ic_perfect_monotonic():
    """Description: Check exact perfect positive IC and Rank IC.

    Parameters: None.
    Returns: None.
    Notes: Uses a single four-stock cross-section.
    """
    d = pd.DataFrame(
        {
            'date': ['2020-01-01'] * 4,
            'instrument': list('ABCD'),
            'x': [1, 2, 3, 4],
            'y': [2, 4, 6, 8],
        }
    )

    a = daily_ic(d, ['x'], 'y', min_n=3).iloc[0]
    assert np.isclose(a.ic, 1) and np.isclose(a.rank_ic, 1)


def test_ties_and_constant():
    """Description: Verify tied ranks and constant factor handling.

    Parameters: None.
    Returns: None.
    Notes: Spearman rank ties are averaged by pandas.
    """
    d = pd.DataFrame(
        {
            'date': ['2020-01-01'] * 4,
            'instrument': list('ABCD'),
            'x': [1, 1, 2, 3],
            'c': [2] * 4,
            'y': [1, 2, 3, 4],
        }
    )

    a = daily_ic(d, ['x', 'c'], 'y', min_n=3).set_index('factor')
    assert np.isclose(a.loc['x', 'rank_ic'], pd.Series(d.x).corr(pd.Series(d.y), method='spearman'))
    assert a.loc['c', 'status'] == 'constant' and np.isnan(a.loc['c', 'ic'])


def test_duplicate_keys_rejected():
    """Description: Reject nonunique date-instrument keys.

    Parameters: None.
    Returns: None.
    Notes: Duplicate keys must not silently multiply observations.
    """
    f = pd.DataFrame({'date': [1, 1], 'instrument': ['A', 'A'], 'x': [1, 2]})
    y = pd.DataFrame({'date': [1], 'instrument': ['A'], 'y': [.01]})
    with pytest.raises(ValueError):
        align(f, y, ['x'], 'y')


def test_hac_zero_lag_and_quantiles():
    """Description: Check zero-lag HAC and five-group forward-return spread.

    Parameters: None.
    Returns: None.
    Notes: Quantile spread is not a portfolio daily return.
    """
    x = pd.Series([.1, .2, -.1, .3, .2])
    result = hac_mean(x, 0)
    assert np.isclose(result['hac_se'], np.std(x, ddof=0) / np.sqrt(len(x)))
    d = pd.DataFrame(
        {
            'date': ['2020-01-01'] * 10,
            'instrument': list('ABCDEFGHIJ'),
            'x': range(10),
            'y': np.arange(10) / 100,
        }
    )

    q = quantile_returns(d, 'x', 'y', min_n=5).iloc[0]
    assert np.isclose(q.long_short_forward_spread, .08)
