"""Synthetic regression tests for cross-sectional preprocessing and temporal isolation."""

import numpy as np
import pandas as pd
from factors.factor_preprocessor import (
    PreprocessConfig,
    preprocess_factor_panel,
    compare_prefix_invariance,
)


def run_tests() -> None:
    """Description:
        Test tie handling, missingness, robustness, cross-sectional isolation and prefix invariance.

    Parameters:
        None.

    Returns:
        None: Raises AssertionError on failure; prints a success message otherwise.
    """
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    index = pd.MultiIndex.from_product(
        [dates, ["A", "B", "C", "D", "E"]], names=["datetime", "instrument"]
    )
    panel = pd.DataFrame(
        {
            "MOM_5": [1, 2, 2, 4, np.nan, 3, 4, 5, 6, 1000, 10, 11, 12, 13, 14],
            "MOM_20": [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10],
        },
        index=index,
    )
    for method in ("raw", "rank_uniform", "rank_gaussian", "robust"):
        config = PreprocessConfig(method=method, min_valid=3)
        result = preprocess_factor_panel(panel, config)
        assert result.index.equals(panel.index) and list(result.columns) == list(panel.columns)
        assert np.isnan(result.loc[(dates[0], "E"), "MOM_5"])
        assert compare_prefix_invariance(panel, dates[1], config)["value_mismatch"] == 0
        assert compare_prefix_invariance(panel, dates[1], config)["missing_mismatch"] == 0
        changed = panel.copy()
        changed.loc[(dates[2], "A"), "MOM_5"] = 999999
        assert (
            result.loc[(dates[0], "A"), "MOM_5"]
            == preprocess_factor_panel(changed, config).loc[(dates[0], "A"), "MOM_5"]
        )
    ranks = preprocess_factor_panel(panel, PreprocessConfig(method="rank_uniform", min_valid=3))
    assert ranks.loc[(dates[0], "B"), "MOM_5"] == ranks.loc[(dates[0], "C"), "MOM_5"]
    robust = preprocess_factor_panel(panel, PreprocessConfig(method="robust", min_valid=3))
    assert robust.loc[dates[0], "MOM_20"].isna().all()
    print("Synthetic factor preprocessing tests passed.")


if __name__ == "__main__":
    run_tests()
