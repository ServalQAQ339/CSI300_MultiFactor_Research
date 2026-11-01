"""Synthetic regression checks for the lightweight neutralization pipeline."""
import numpy as np
import pandas as pd
from factor_neutralization.fasttrack import neutralize_day, summarize_ic


def test_size_residual_orthogonality():
    """Description: Check that OLS residuals are orthogonal to size on synthetic observations.

    Parameters:
        None: Synthetic inputs are constructed in the test.
    Returns:
        None: Raises an assertion error if neutralization fails.
    Notes:
        Does not certify real historical market-cap data.
    """
    rng=np.random.default_rng(7); size=rng.normal(size=100); d=pd.DataFrame({'datetime':pd.Timestamp('2020-01-02'),'instrument':[f'SH{i:06d}' for i in range(100)],'factor':2*size+rng.normal(size=100),'log_market_cap_candidate':size}); z=neutralize_day(d,'factor'); assert abs(np.corrcoef(z['size_neutral'],size)[0,1])<1e-10


def test_ic_positive():
    """Description: Confirm Spearman IC computation on monotonically aligned returns.

    Parameters:
        None: Synthetic factor and return data are created in the test.
    Returns:
        None: Raises an assertion error for incorrect IC.
    Notes:
        The synthetic example is not a real performance claim.
    """
    d=pd.DataFrame({'datetime':np.repeat(pd.date_range('2020-01-01',periods=8),40),'factor':list(range(40))*8,'label':list(range(40))*8}); assert abs(summarize_ic(d,'factor','label')['mean_rank_ic']-1)<1e-12
