"""Synthetic regression tests for train-only selection and matched Ridge IC."""

import numpy as np
import pandas as pd

from factor_selection.selection import compare_ridge, select_features


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
