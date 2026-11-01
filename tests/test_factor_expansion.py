"""Synthetic regression checks for unified factor expansion."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from factors.expansion.extra_factors import compute_extra_factors
from factors.expansion.inspired_factors import compute_batch_factors
from factors.expansion.qlib_factors import compute_factors
from factors.expansion.registry import build_registry
from factors.expansion.statistics import daily_rank_ic, hac_summary, redundancy


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


def one_day_panel(columns: dict[str, list[float]]) -> pd.DataFrame:
    """Wrap equal-length columns as a single-date stock panel.

    Description:
        Build a (datetime, instrument) index for one synthetic cross-section.
    Parameters:
        columns (dict[str, list[float]]): Column name to per-stock values.
    Returns:
        pd.DataFrame: One-date panel.
    Notes:
        Used exclusively for unit tests.
    """
    n = len(next(iter(columns.values())))
    index = pd.MultiIndex.from_product([[pd.Timestamp("2021-01-04")], [f"S{i}" for i in range(n)]],
                                       names=["datetime", "instrument"])
    return pd.DataFrame(columns, index=index)


def test_rank_ic_reranks_finite_pairs_only() -> None:
    """Verify Rank IC drops invalid pairs before ranking.

    Description:
        Rank-then-delete gives 0.0822 here; standard Spearman on the five finite
        pairs gives exactly zero.
    Parameters:
        None.
    Returns:
        None: Assertions check the standard Spearman definition.
    Notes:
        Covers NaN factor values and the documented min_stocks threshold.
    """
    panel = one_day_panel({"f": [1, 2, np.nan, 4, 5, 6], "label_5d": [6, 1, 2, 3, 4, 5]})
    assert np.isclose(daily_rank_ic(panel, ["f"], min_stocks=5)["f"].iloc[0], 0.0)
    assert np.isnan(daily_rank_ic(panel, ["f"], min_stocks=6)["f"].iloc[0])


def test_rank_ic_matches_scipy_with_nan_inf_and_ties() -> None:
    """Verify Rank IC equals scipy Spearman on finite pairs with ties.

    Description:
        Mix NaN and inf in factors and labels, plus heavily tied factor values.
    Parameters:
        None.
    Returns:
        None: Assertions compare to scipy.stats.spearmanr.
    Notes:
        Constant paired samples must return NaN rather than a number.
    """
    rng = np.random.default_rng(3)
    label = rng.normal(size=80)
    tied = np.round(0.5 * label + rng.normal(size=80))
    tied[[1, 7, 20]] = np.nan
    tied[[3, 40]] = np.inf
    label[[5, 60]] = np.nan
    label[11] = -np.inf
    panel = one_day_panel({"tied": tied, "const": np.ones(80), "label_5d": label})
    daily = daily_rank_ic(panel, ["tied", "const"])
    ok = np.isfinite(tied) & np.isfinite(label)
    assert np.isclose(daily["tied"].iloc[0], spearmanr(tied[ok], label[ok]).statistic)
    assert np.isnan(daily["const"].iloc[0])


def test_redundancy_uses_pairwise_finite_spearman() -> None:
    """Verify candidate-baseline correlations re-rank each pair's finite sample.

    Description:
        Candidate and baseline have different missing and infinite rows.
    Parameters:
        None.
    Returns:
        None: Assertions compare to scipy.stats.spearmanr.
    Notes:
        Single-date panel, so the daily mean equals the daily value.
    """
    rng = np.random.default_rng(4)
    base = np.round(rng.normal(size=50), 1)
    cand = base + rng.normal(size=50)
    base[[2, 9]] = np.nan
    cand[[4, 30]] = np.nan
    cand[17] = np.inf
    result = redundancy(one_day_panel({"C": cand, "B": base}), ["B"], ["C"]).iloc[0]
    ok = np.isfinite(cand) & np.isfinite(base)
    assert np.isclose(result["mean_spearman"], spearmanr(cand[ok], base[ok]).statistic)
    assert result["days"] == 1


def test_hac_summary_fdr_flag_only_drops_fdr_column() -> None:
    """Verify fdr=False returns the same inference without BH-FDR q-values.

    Description:
        Matched model comparisons use fdr=False; values must match the FDR variant.
    Parameters:
        None.
    Returns:
        None: Assertions compare both outputs.
    Notes:
        Includes a series with fewer than three valid days.
    """
    rng = np.random.default_rng(5)
    daily = pd.DataFrame({"a": rng.normal(0.02, 0.1, 60), "b": [0.1, 0.2] + [np.nan] * 58})
    with_fdr = hac_summary(daily, lags=5)
    without = hac_summary(daily, lags=5, fdr=False)
    assert "fdr_q" not in without.columns
    pd.testing.assert_frame_equal(with_fdr.drop(columns="fdr_q"), without)
    assert np.isnan(without.set_index("factor").loc["b", "hac_t"])
