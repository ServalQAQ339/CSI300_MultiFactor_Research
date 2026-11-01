"""Synthetic tests for CSI300 benchmark alignment and performance calculations."""

import numpy as np
import pandas as pd
import pytest

from compare_csi300_benchmark import MODELS, compare_periods


def make_periods() -> tuple[pd.DataFrame, pd.Series]:
    """Build two deterministic, non-overlapping periods.

    Description:
        Supply aligned synthetic portfolio returns and index opening prices.
    Parameters:
        None: Synthetic values are fixed.
    Returns:
        tuple[pd.DataFrame, pd.Series]: Period report and benchmark series.
    Notes:
        This fixture does not access the Qlib provider.
    """
    dates = pd.to_datetime(["2022-01-03", "2022-01-10", "2022-01-17"])
    frame = pd.DataFrame({"entry_date": dates[:-1], "exit_date": dates[1:]})
    for model in MODELS:
        frame[f"{model}_net"] = [0.02, -0.01]
    benchmark = pd.Series([100.0, 101.0, 103.0], index=dates)
    return frame, benchmark


def test_benchmark_matches_period_dates() -> None:
    """Verify exact open-to-open benchmark matching.

    Description:
        Compare index returns and calculated period excess returns.
    Parameters:
        None: Uses the synthetic fixture.
    Returns:
        None: Assertions validate calculations.
    Notes:
        Benchmark periods are never approximated using calendar offsets.
    """
    periods, benchmark = make_periods()
    matched, summary = compare_periods(periods, benchmark)
    assert np.isclose(matched.iloc[0]["csi300_return"], 0.01)
    assert np.isclose(matched.iloc[1]["csi300_return"], 103 / 101 - 1)
    assert np.isclose(matched.iloc[0]["ridge_original27_excess_csi300"], 0.01)
    assert len(summary) == 3


def test_missing_benchmark_date_fails() -> None:
    """Reject unmatched index execution dates.

    Description:
        Prevent silent date dropping and misleading comparisons.
    Parameters:
        None: Uses the synthetic fixture.
    Returns:
        None: Expects ValueError.
    Notes:
        All strategy periods must have matching benchmark prices.
    """
    periods, benchmark = make_periods()
    with pytest.raises(ValueError, match="Missing benchmark opens"):
        compare_periods(periods, benchmark.iloc[:-1])
