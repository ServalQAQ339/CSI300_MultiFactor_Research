"""Synthetic tests for factor research v1.3 diagnostics."""
import numpy as np
import pandas as pd
import pytest
from factor_research.diagnostics import ic_diagnostics, bh_fdr, daily_factor_correlations, quantile_diagnostics


def test_fdr():
    """Description: Verify BH adjustment against hand-calculated values.

    Parameters:
        None: Uses synthetic p-values.
    Returns:
        None: Raises on numerical mismatch.
    Notes:
        Missing values must remain missing.
    """
    got = bh_fdr(pd.Series([0.01, 0.04, 0.03, np.nan])); np.testing.assert_allclose(got.iloc[:3], [0.03, 0.04, 0.04]); assert np.isnan(got.iloc[3])


def test_calendar_phase_preserved():
    """Description: Ensure missing factor-day IC does not shift non-overlap calendar phases.

    Parameters:
        None: Uses a six-day synthetic calendar.
    Returns:
        None: Checks phase counts.
    Notes:
        Phase assignment is shared across factors.
    """
    dates = pd.bdate_range('2024-01-01', periods=6); rows = [{'signal_date': t, 'factor': f, 'ic': float(i)/10, 'rank_ic': float(i)/10} for i,t in enumerate(dates) for f in ['A','B'] if not (f=='B' and i==1)]
    _, phase = ic_diagnostics(pd.DataFrame(rows), {'test': ('2024-01-01','2024-01-31')}, horizon=5)
    assert phase.query("factor == 'A' and metric == 'ic' and phase == 0").n_days.iloc[0] == 2
    assert phase.query("factor == 'B' and metric == 'ic' and phase == 1").n_days.iloc[0] == 0


def test_factor_corr():
    """Description: Validate positive and negative Spearman factor relationships.

    Parameters:
        None: Synthetic single-day cross section.
    Returns:
        None: Asserts pairwise ranks and counts.
    Notes:
        Uses min_n three.
    """
    frame = pd.DataFrame({'signal_date':['2024-01-02']*4, 'A':[1,2,3,4], 'B':[4,3,2,1], 'C':[1,2,3,4]}); corr, count = daily_factor_correlations(frame, ['A','B','C'], 3)
    assert corr.loc['A','B'] == pytest.approx(-1); assert corr.loc['A','C'] == pytest.approx(1); assert count.loc['A','B'] == 1


def test_quantile_schema():
    """Description: Check descending five-bin returns and spread consistency.

    Parameters:
        None: One synthetic signal date.
    Returns:
        None: Asserts decreasing quantiles.
    Notes:
        The spread is a forward-return difference, not daily PnL.
    """
    d = pd.DataFrame([{'signal_date':'2024-01-02', **{f'q{i}_return': (6-i)/100 for i in range(1,6)}, 'long_short_forward_spread': -0.04}]); out = quantile_diagnostics(d, 'A', {'test':('2024-01-01','2024-12-31')})
    assert bool(out.monotone_decreasing.iloc[0]); assert out.q5_minus_q1.iloc[0] == pytest.approx(-0.04)
