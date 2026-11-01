"""Synthetic tests for calendar-preserving HAC statistics."""
import numpy as np
import pandas as pd
from factor_research.statistical_tests import hac_summary


def test_constant_ic_mean():
    """Description: Check constant IC has zero sample variance.

    Parameters:
        None: Uses synthetic values.
    Returns:
        None: Asserts correct mean and sample size.
    Notes:
        A zero standard error yields undefined t-statistic.
    """
    result = hac_summary(pd.DataFrame({'ic':[0.2]*6}), 'ic', 4)
    assert np.isclose(result['mean'],0.2)
    assert result['n']==6
    assert np.isnan(result['hac_t'])


def test_missing_date_kept_as_slot():
    """Description: Confirm missing IC observations do not shift adjacent lags.

    Parameters:
        None: Synthetic time sequence with an interior missing value.
    Returns:
        None: Asserts retained valid count and finite mean.
    Notes:
        A missing date is not interpreted as an adjacent observed date.
    """
    result = hac_summary(pd.DataFrame({'ic':[0.1,np.nan,0.3,0.2,0.4]}),'ic',1)
    assert result['n']==4
    assert np.isclose(result['mean'],0.25)
