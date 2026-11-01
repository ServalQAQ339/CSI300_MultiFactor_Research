"""Synthetic regression tests for enhanced factor audit comparisons."""

import numpy as np
import pandas as pd
from audits.enhanced_audit import compare_factor_frames


def test_compare_factor_frames() -> None:
    """Check recovery, loss, changed-value and membership assertions.

    Parameters:
        None.

    Returns:
        None: Raises an assertion on failure.
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    index = pd.MultiIndex.from_product([dates, ["SH600000"]], names=["datetime", "instrument"])
    price = pd.DataFrame({"close": [1, 2, 3, 4]}, index=index)
    baseline = pd.DataFrame({"MOM_5": [np.nan, 1.0, 2.0, 3.0]}, index=index)
    enhanced = pd.DataFrame({"MOM_5": [0.5, 1.0, np.nan, 3.1]}, index=index)
    details, summary = compare_factor_frames(baseline, enhanced, price, 2020)
    assert details.loc[0, "recovered"] == 1
    assert details.loc[0, "lost"] == 1
    assert details.loc[0, "changed_existing"] == 1
    assert summary["rows"] == 4

    try:
        compare_factor_frames(baseline, enhanced.iloc[:-1], price, 2020)
        raise AssertionError("Index mismatch should raise")
    except ValueError:
        pass


if __name__ == "__main__":
    test_compare_factor_frames()
    print("Synthetic enhanced audit tests passed.")
