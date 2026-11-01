"""Synthetic regression tests for independent IC/quintile consistency audit."""
import numpy as np
import pandas as pd
from factor_research.consistency_audit import audit_factor, compare_saved


def sample_panel() -> pd.DataFrame:
    """Description: Build a two-date synthetic panel with a rank/mean-spread sign reversal.

    Parameters:
        None: Uses fixed deterministic synthetic data.
    Returns:
        pd.DataFrame: Ten stocks per date with valid factor/label observations.
    Notes:
        Extreme high-quantile return makes average spread positive despite negative rank IC.
    """
    rows = []
    for date in ['2022-01-04', '2022-01-05']:
        for i in range(10): rows.append({'signal_date': pd.Timestamp(date), 'instrument': f'S{i:02}', 'VOLATILITY_20': float(i), 'research_label_5d': float(100 if i == 9 else 10-i)})
    return pd.DataFrame(rows)


def test_sign_disagreement() -> None:
    """Description: Verify a negative rank IC can coexist with a positive top-minus-bottom mean spread.

    Parameters:
        None: Uses synthetic panel.
    Returns:
        None: Raises on unexpected result.
    Notes:
        Tests a statistical possibility, not an observed CSI300 result.
    """
    daily, _ = audit_factor(sample_panel(), 'VOLATILITY_20', min_n=5)
    assert (daily.rank_ic_independent < 0).all() and (daily.spread_mean > 0).all()


def test_daily_comparison() -> None:
    """Description: Verify exact-match comparison and deliberate mismatch detection.

    Parameters:
        None: Uses synthetic daily values.
    Returns:
        None: Raises on mismatch in pass/fail logic.
    Notes:
        Comparison must reject incorrect saved spread values.
    """
    daily, _ = audit_factor(sample_panel(), 'VOLATILITY_20', min_n=5)
    ic = daily[['signal_date', 'rank_ic_independent', 'n_paired']].rename(columns={'rank_ic_independent': 'rank_ic'})
    q = daily[['signal_date', 'spread_mean', 'n_paired']].rename(columns={'spread_mean': 'long_short_forward_spread'})
    assert compare_saved(daily, ic, q)['pass'].all()
    q.loc[q.index[0], 'long_short_forward_spread'] += 0.01
    assert (~compare_saved(daily, ic, q)['pass']).sum() == 1
