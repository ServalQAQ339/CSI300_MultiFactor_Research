"""Synthetic unit tests for v1.5 screening diagnostics."""
import numpy as np
import pandas as pd
from factor_research.selection import conditional_rank_ic, train_redundancy_groups


def test_conditional_rank_ic_perfect_rank() -> None:
    """Description: Verify positive monotone candidate-label association.

    Parameters:
        None: Uses synthetic panel.
    Returns:
        None: Asserts exact correlation behavior.
    Notes:
        No project data is needed.
    """
    d = pd.DataFrame(
        {
            'signal_date': pd.Timestamp('2020-01-02'),
            'x': np.arange(50),
            'research_label_5d': np.arange(50),
        }
    )
    assert abs(conditional_rank_ic(d,'x',[],30).conditional_rank_ic.iloc[0]-1)<1e-12


def test_residualized_duplicate_is_uninformative() -> None:
    """Description: Ensure a factor fully explained by a control has undefined incremental correlation.

    Parameters:
        None: Uses synthetic duplicate factors.
    Returns:
        None: Asserts residual correlation is NaN.
    Notes:
        Duplicate factors cannot add independent rank information.
    """
    d = pd.DataFrame(
        {
            'signal_date': pd.Timestamp('2020-01-02'),
            'x': np.arange(50),
            'z': np.arange(50),
            'research_label_5d': np.arange(50),
        }
    )
    assert np.isnan(conditional_rank_ic(d,'x',['z'],30).conditional_rank_ic.iloc[0])


def test_connected_components() -> None:
    """Description: Verify correlation graph components and isolated factors.

    Parameters:
        None: Uses a synthetic correlation matrix.
    Returns:
        None: Asserts group membership.
    Notes:
        This is not a test of statistical significance.
    """
    m = pd.DataFrame([[1,.9,.1],[.9,1,.2],[.1,.2,1]],index=['a','b','c'],columns=['a','b','c'])
    g = train_redundancy_groups(m,.8).set_index('factor').redundancy_group
    assert g['a']==g['b'] and g['a']!=g['c']
