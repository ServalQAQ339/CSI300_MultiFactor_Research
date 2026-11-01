"""Synthetic regression tests for preprocessing bias diagnostics."""

import numpy as np
import pandas as pd
from factors.factor_preprocessor import PreprocessConfig, preprocess_factor_panel
from preprocessing_validation.preprocessing_validator import validate_one_year


def test_preprocessing_validation() -> None:
    """Description:
        Verify temporal invariance, rank preservation, and missingness diagnostics.

    Parameters:
        None.

    Returns:
        None: Raises AssertionError if a synthetic validation fails.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=8)
    instruments = [f"SH{i:06d}" for i in range(20)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    panel = pd.DataFrame({"MOM_20": rng.normal(size=len(index)), "VOLATILITY_20": rng.normal(size=len(index))}, index=index)
    panel.loc[(dates[0], instruments[0]), "MOM_20"] = np.nan
    panel.loc[(dates[1], instruments[1]), "MOM_20"] = np.inf
    for method in ("raw", "rank_uniform", "rank_gaussian", "robust"):
        records = validate_one_year(panel, method)
        assert all(record["passed"] for record in records), records
        transformed = preprocess_factor_panel(panel, PreprocessConfig(method=method))
        assert np.isnan(transformed.loc[(dates[0], instruments[0]), "MOM_20"])
    print("Synthetic preprocessing validation tests passed.")


if __name__ == "__main__":
    test_preprocessing_validation()
