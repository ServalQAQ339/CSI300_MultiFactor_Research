"""Test temporal maturity and no-future-exit selection constraints."""

import numpy as np
import pandas as pd

from portfolio.expanding import MODELS, mature_train_mask, simulate


def test_label_maturity() -> None:
    """Verify labels crossing a quarterly boundary are purged.

    Description:
        Compare known and unavailable label endpoints on a trading calendar.
    Parameters:
        None: Synthetic trading calendar.
    Returns:
        None: Assertions validate the mask.
    Notes:
        A six-session endpoint must precede first forecast signal.
    """
    calendar = pd.bdate_range("2023-12-01", periods=30)
    dates = calendar[:20]
    mask = mature_train_mask(dates, calendar, calendar[20])
    assert mask[13]
    assert not mask[14]
    assert not mask[19]


def test_missing_selected_exit_is_not_replaced() -> None:
    """Reject periods with missing exits for stocks selected at signal time.

    Description:
        Ensure a top-ranked stock is not swapped for another after observing
        that its future exit price is unavailable.
    Parameters:
        None: Synthetic prices and model scores.
    Returns:
        None: Raises if a biased period is accepted.
    Notes:
        A missing exit price cannot be used for pre-trade filtering.
    """
    dates = pd.bdate_range("2024-01-02", periods=9)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
    opens = pd.DataFrame({"open": 100.0}, index=idx)
    opens.loc[(dates[6], "A"), "open"] = np.nan
    signal_idx = pd.MultiIndex.from_product([[dates[0]], ["A", "B"]], names=["datetime", "instrument"])
    scores = pd.DataFrame({name: [2.0, 1.0] for name in MODELS}, index=signal_idx)
    benchmark = pd.Series(100.0, index=dates)
    try:
        simulate(scores, opens, benchmark, top_k=1)
    except ValueError as error:
        assert "No complete" in str(error)
    else:
        raise AssertionError("Future missing exit was used to replace a selected stock")
