"""Synthetic regression tests for train-only selection and matched Ridge IC."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from factor_research.selection import _pooled_rank_corr, compare_ridge, select_features


def make_panel(seed: int = 10) -> pd.DataFrame:
    """Create deterministic synthetic stock-date factor observations.

    Description:
        Simulate two predictors and a weakly informative return label.
    Parameters:
        seed (int): Random generator seed.
    Returns:
        pd.DataFrame: Unique stock-date panel.
    Notes:
        These data are not empirical market observations.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=25)
    index = pd.MultiIndex.from_product([dates, [f"S{i:03d}" for i in range(60)]],
                                       names=["datetime", "instrument"])
    base = rng.normal(size=len(index))
    extra = rng.normal(size=len(index))
    return pd.DataFrame({"BASE": base, "EXTRA": extra, "DUP": base,
                         "label_5d": base + extra + rng.normal(size=len(index))}, index=index)


def test_train_only_selection_and_redundancy() -> None:
    """Verify redundant and registry-excluded candidates are not selected.

    Description:
        Check selection decisions using synthetic training observations.
    Parameters:
        None: Uses locally generated synthetic data.
    Returns:
        None: Assertions check selection behavior.
    Notes:
        Thresholds are fixed before calling the selection function.
    """
    panel = make_panel()
    registry = pd.DataFrame({"factor": ["EXTRA", "DUP"], "excluded": [False, False]})
    result = select_features(panel, ["BASE"], ["EXTRA", "DUP"], registry,
                             max_features=2, min_days=10)
    assert result.set_index("factor").loc["EXTRA", "selected"]
    assert not result.set_index("factor").loc["DUP", "selected"]


def test_matched_ridge_and_delta_identity() -> None:
    """Verify both models use identical Validation observations.

    Description:
        Check daily IC differences and row counts after missing-value filtering.
    Parameters:
        None: Uses synthetic data with one missing candidate value.
    Returns:
        None: Assertions verify matched-sample model evaluation.
    Notes:
        No model is fitted on Validation labels.
    """
    panel = make_panel()
    dates = panel.index.get_level_values("datetime")
    train = panel.loc[dates <= dates.unique()[14]]
    validation = panel.loc[dates > dates.unique()[14]].copy()
    validation.iloc[0, validation.columns.get_loc("EXTRA")] = np.nan
    daily, summary = compare_ridge(train, validation, ["BASE"], ["EXTRA"])
    assert (daily["n_stocks"] == 59).sum() == 1
    assert np.allclose(daily["delta_ic"], daily["extended"] - daily["baseline"])
    assert set(summary["factor"]) == {"baseline", "extended", "delta_ic"}


def test_selection_ic_is_standard_spearman() -> None:
    """Verify Train IC and IC days follow standard Spearman on finite pairs.

    Description:
        Add NaN, inf and ties to a candidate, plus a date with one finite pair.
    Parameters:
        None: Uses synthetic data.
    Returns:
        None: Assertions compare to per-date scipy Spearman.
    Notes:
        A date with fewer than two finite pairs has no IC and is not counted.
    """
    panel = make_panel()
    rng = np.random.default_rng(1)
    extra = np.round(panel["EXTRA"].to_numpy(), 0)
    extra[rng.random(len(extra)) < 0.2] = np.nan
    extra[rng.random(len(extra)) < 0.03] = np.inf
    panel["EXTRA"] = extra
    first_date = panel.index.get_level_values("datetime")[0]
    on_first = panel.index.get_level_values("datetime") == first_date
    panel.loc[on_first, "EXTRA"] = np.nan
    panel.loc[panel.index[0], "EXTRA"] = 1.0
    registry = pd.DataFrame({"factor": ["EXTRA"], "excluded": [False]})
    row = select_features(panel, ["BASE"], ["EXTRA"], registry, min_days=1,
                          min_coverage=0.0).set_index("factor").loc["EXTRA"]

    expected = []
    for _, day in panel.groupby(level="datetime"):
        x, y = day["EXTRA"].to_numpy(), day["label_5d"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 2:
            expected.append(spearmanr(x[ok], y[ok]).statistic)
    assert row["ic_days"] == len(expected) == 24
    assert np.isclose(row["train_ic"], np.mean(expected))


def test_pooled_rank_corr_reranks_pair_sample() -> None:
    """Verify the redundancy proxy ranks each pair on its own finite rows.

    Description:
        Compare with manual datewise percentile ranks on the paired sample.
    Parameters:
        None: Uses synthetic data.
    Returns:
        None: Assertions check the pooled correlation.
    Notes:
        Infinite values are treated as missing.
    """
    panel = make_panel()
    rng = np.random.default_rng(2)
    panel.loc[rng.random(len(panel)) < 0.2, "EXTRA"] = np.nan
    panel.loc[rng.random(len(panel)) < 0.2, "BASE"] = np.inf
    pair = panel[["BASE", "EXTRA"]].replace(np.inf, np.nan).dropna()
    ranks = pair.groupby(level="datetime").rank(pct=True)
    expected = np.corrcoef(ranks["BASE"], ranks["EXTRA"])[0, 1]
    assert np.isclose(_pooled_rank_corr(panel, "BASE", "EXTRA"), expected)
