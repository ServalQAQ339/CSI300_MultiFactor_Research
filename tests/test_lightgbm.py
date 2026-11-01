"""Test Stage 5 sample matching, label purging and daily IC comparisons."""

import numpy as np
import pandas as pd

from modeling.modeling import (
    cross_sectional_ranks,
    evaluate_predictions,
    prepare_samples,
    selected_features,
)


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
