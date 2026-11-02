"""Test Stage 5 sample matching, label purging and daily IC comparisons."""

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from modeling.modeling import (
    cross_sectional_ranks,
    evaluate_predictions,
    fit_predict_models,
    prepare_samples,
    selected_features,
)


def make_split_panel(seed: int = 3) -> pd.DataFrame:
    """Create a synthetic panel spanning the Train/Validation boundary.

    Description:
        Simulate two features and a weakly informative label for 40 stocks.
    Parameters:
        seed (int): Random generator seed.
    Returns:
        pd.DataFrame: Stock-date panel with a, b and label_5d columns.
    Notes:
        Enough Train rows exist for the fixed LightGBM min_child_samples.
    """
    rng = np.random.default_rng(seed)
    dates = list(pd.bdate_range("2020-11-02", "2020-12-31")) + list(
        pd.bdate_range("2021-01-04", periods=6)
    )
    index = pd.MultiIndex.from_product([dates, [f"S{i:02d}" for i in range(40)]],
                                       names=["datetime", "instrument"])
    a, b = rng.normal(size=len(index)), rng.normal(size=len(index))
    return pd.DataFrame({"a": a, "b": b, "label_5d": a + rng.normal(size=len(index))},
                        index=index)


def test_common_samples_and_purge():
    """Ensure the Train tail is purged and Validation rows are shared.

    Description:
        Construct a small synthetic panel and apply Stage 5 sample filtering.
    Parameters:
        None: Synthetic inputs are generated in the test.
    Returns:
        None: Assertions verify the resulting sample boundaries.
    Notes:
        Uses 35 stocks per date to meet the minimum cross-section size.
    """
    dates = list(pd.bdate_range("2020-12-14", "2020-12-31")) + list(
        pd.bdate_range("2021-01-04", periods=7)
    )
    index = pd.MultiIndex.from_product([dates, range(35)], names=["datetime", "instrument"])
    frame = pd.DataFrame({"a": np.arange(len(index)) % 31,
                          "b": np.arange(len(index)) % 19,
                          "label_5d": np.arange(len(index)) % 13}, index=index)
    train, validation = prepare_samples(frame, ["a"], ["b"])
    assert train.index.get_level_values("datetime").max() < pd.Timestamp("2020-12-25")
    assert validation.index.get_level_values("datetime").min() == pd.Timestamp("2021-01-04")
    assert len(validation) == 7 * 35


def test_train_drops_invalid_labels_validation_keeps_them():
    """Ensure label validity filters Train fitting rows but not Validation predictions.

    Description:
        Blank some labels in both splits and one Validation feature value.
    Parameters:
        None: Synthetic inputs are generated in the test.
    Returns:
        None: Assertions verify row retention in each split.
    Notes:
        Missing features remain a legitimate signal-time exclusion.
    """
    frame = make_split_panel()
    train_missing = (pd.Timestamp("2020-11-02"), "S00")
    valid_missing = (pd.Timestamp("2021-01-05"), "S01")
    frame.loc[train_missing, "label_5d"] = np.nan
    frame.loc[valid_missing, "label_5d"] = np.nan
    frame.loc[(pd.Timestamp("2021-01-05"), "S02"), "b"] = np.nan
    train, validation = prepare_samples(frame, ["a"], ["b"])
    assert train_missing not in train.index
    assert train["label_5d"].notna().all()
    assert valid_missing in validation.index
    assert np.isnan(validation.loc[valid_missing, "label_5d"])
    assert (pd.Timestamp("2021-01-05"), "S02") not in validation.index
    assert len(validation) == 6 * 40 - 1


def test_validation_scores_independent_of_future_labels():
    """Verify feature ranks and model scores ignore Validation label validity.

    Description:
        Fit models twice, once with several Validation labels blanked.
    Parameters:
        None: Synthetic inputs are generated in the test.
    Returns:
        None: Assertions compare ranks, rows and scores.
    Notes:
        Train data are identical, so fitted models must coincide.
    """
    full = make_split_panel()
    masked = full.copy()
    dates = masked.index.get_level_values("datetime")
    blank = (dates >= pd.Timestamp("2021-01-04")) & masked.index.get_level_values(
        "instrument").isin(["S00", "S05", "S09"])
    masked.loc[blank, "label_5d"] = np.nan

    train_a, valid_a = prepare_samples(full, ["a"], ["b"])
    train_b, valid_b = prepare_samples(masked, ["a"], ["b"])
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_index_equal(valid_a.index, valid_b.index)
    pd.testing.assert_frame_equal(cross_sectional_ranks(valid_a, ["a", "b"]),
                                  cross_sectional_ranks(valid_b, ["a", "b"]))

    scores = ["ridge_original27", "lightgbm_original27", "lightgbm_extended"]
    pred_a = fit_predict_models(train_a, valid_a, ["a"], ["b"])
    pred_b = fit_predict_models(train_b, valid_b, ["a"], ["b"])
    pd.testing.assert_frame_equal(pred_a[scores], pred_b[scores])
    assert pred_b["label_5d"].isna().sum() == blank.sum()


def test_ic_uses_only_valid_labels():
    """Verify daily Rank IC is standard Spearman on the valid-label subsample.

    Description:
        Blank labels on two dates and compare against scipy Spearman.
    Parameters:
        None: Test inputs are generated locally.
    Returns:
        None: Assertions check IC values, counts and skipped dates.
    Notes:
        A date with fewer valid labels than min_stocks is not evaluated.
    """
    index = pd.MultiIndex.from_product(
        [pd.bdate_range("2021-01-04", periods=4), range(40)],
        names=["datetime", "instrument"],
    )
    rng = np.random.default_rng(11)
    names = ["ridge_original27", "lightgbm_original27", "lightgbm_extended"]
    frame = pd.DataFrame(rng.normal(size=(len(index), 4)), index=index,
                         columns=names + ["label_5d"])
    first, last = index.levels[0][0], index.levels[0][-1]
    frame.loc[(first, slice(0, 4)), "label_5d"] = np.nan
    frame.loc[(last, slice(0, 19)), "label_5d"] = np.nan

    daily, _ = evaluate_predictions(frame, min_stocks=30)
    assert last not in daily.index
    assert daily.loc[first, "n_predicted"] == 40
    assert daily.loc[first, "n_stocks"] == 35
    valid = frame.xs(first, level="datetime").dropna()
    for name in names:
        expected = spearmanr(valid[name], valid["label_5d"]).statistic
        assert np.isclose(daily.loc[first, name], expected)
    target = valid["label_5d"].rank(pct=True) - 0.5
    assert np.isclose(daily.loc[first, "mse_ridge_original27"],
                      ((valid["ridge_original27"] - target) ** 2).mean())


def test_daily_delta_identity():
    """Verify that paired daily IC differences match their model columns.

    Description:
        Evaluate deterministic synthetic predictions on multiple dates.
    Parameters:
        None: Test inputs are generated locally.
    Returns:
        None: Assertions validate both paired difference definitions.
    Notes:
        No model fitting or market data is needed.
    """
    index = pd.MultiIndex.from_product(
        [pd.bdate_range("2021-01-04", periods=6), range(40)],
        names=["datetime", "instrument"],
    )
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(len(index), 4)), index=index,
                         columns=["ridge_original27", "lightgbm_original27",
                                  "lightgbm_extended", "label_5d"])
    daily, summary = evaluate_predictions(frame)
    np.testing.assert_allclose(daily["delta_expansion"],
                               daily["lightgbm_extended"] - daily["lightgbm_original27"])
    assert len(summary) == 5


def test_selection_file(tmp_path):
    """Verify that only frozen selected features are loaded.

    Description:
        Read a small Stage 4-compatible selection CSV.
    Parameters:
        tmp_path (pathlib.Path): Pytest temporary directory.
    Returns:
        None: Asserts selected feature names and order.
    Notes:
        The function does not inspect Validation labels.
    """
    path = tmp_path / "selected.csv"
    pd.DataFrame({"factor": ["a", "b", "c"],
                  "selected": [True, False, True]}).to_csv(path, index=False)
    assert selected_features(str(path), ["a", "b", "c"]) == ["a", "c"]
