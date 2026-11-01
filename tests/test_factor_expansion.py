"""Synthetic regression checks for unified factor expansion."""

import numpy as np
import pandas as pd

from factor_expansion.extra_factors import compute_extra_factors
from factor_expansion.inspired_factors import compute_batch_factors
from factor_expansion.qlib_factors import compute_factors
from factor_expansion.registry import build_registry
from factor_expansion.statistics import daily_rank_ic, hac_summary


def sample_bars() -> pd.DataFrame:
    """Build deterministic OHLCV observations.

    Description:
        Construct one-stock synthetic daily bars.
    Parameters:
        None.
    Returns:
        pd.DataFrame: Indexed OHLCV observations.
    Notes:
        Used exclusively for unit tests.
    """
    index = pd.MultiIndex.from_product([pd.date_range("2020-01-01", periods=90), ["TEST"]], names=["datetime", "instrument"])
    x = np.arange(90, dtype=float)
    return pd.DataFrame({"open": 100 + x, "close": 101 + x, "high": 103 + x, "low": 99 + x, "volume": 1000 + 2 * x}, index=index)


def test_generation_and_registry() -> None:
    """Check all feature families and known formula exclusions.

    Description:
        Validate expected column counts and redundancy flags.
    Parameters:
        None.
    Returns:
        None: Assertions validate generation.
    Notes:
        No local Qlib provider is required.
    """
    bars = sample_bars()
    features = pd.concat([compute_factors(bars), compute_batch_factors(bars), compute_extra_factors(bars)], axis=1)
    assert features.shape == (90, 99)
    registry = build_registry(features.columns.tolist())
    assert registry.loc[registry.factor.eq("VOLUME_TREND_20"), "excluded"].iloc[0]


def test_no_future_leakage() -> None:
    """Check that changing a future bar leaves historical features unchanged.

    Description:
        Compare feature outputs before a synthetic future perturbation.
    Parameters:
        None.
    Returns:
        None: Assertions enforce trailing-only feature behavior.
    Notes:
        Uses one instrument to isolate temporal calculations.
    """
    bars = sample_bars()
    changed = bars.copy()
    changed.iloc[-1, changed.columns.get_loc("close")] *= 1.5
    before = compute_extra_factors(bars).iloc[:-1]
    after = compute_extra_factors(changed).iloc[:-1]
    pd.testing.assert_frame_equal(before, after)


def test_rank_ic_and_hac() -> None:
    """Verify Rank IC sign and HAC output on deterministic observations.

    Description:
        Check vectorized cross-sectional rank correlation.
    Parameters:
        None.
    Returns:
        None: Assertions check expected statistical behavior.
    Notes:
        Uses more than thirty stocks per date.
    """
    index = pd.MultiIndex.from_product([pd.date_range("2021-01-01", periods=12), [f"S{i}" for i in range(40)]], names=["datetime", "instrument"])
    values = np.tile(np.arange(40), 12)
    panel = pd.DataFrame({"factor": values, "label_5d": -values}, index=index)
    daily = daily_rank_ic(panel, ["factor"])
    assert np.allclose(daily.factor, -1)
    assert hac_summary(daily).iloc[0]["days"] == 12
