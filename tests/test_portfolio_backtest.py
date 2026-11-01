"""Test next-open execution timing and cost treatment."""

import numpy as np
import pandas as pd

from portfolio.backtest import evaluate_portfolios, SCORES


def test_next_open_and_transaction_cost() -> None:
    """Verify next-open entry and exit and the two-sided cost assumption.

    Description:
        Use deterministic synthetic prices and equal model rankings.
    Parameters:
        None: Synthetic prices are defined in the test.
    Returns:
        None: Raises if the timing or net return differs.
    Notes:
        The signal-day open must never be used for entry.
    """
    dates = pd.bdate_range("2022-01-03", periods=9)
    stocks = ["A", "B"]
    index = pd.MultiIndex.from_product([dates, stocks], names=["datetime", "instrument"])
    prices = pd.DataFrame({"open": 100.0}, index=index)
    prices.loc[(dates[1], "A"), "open"] = 100.0
    prices.loc[(dates[6], "A"), "open"] = 110.0
    prices.loc[(dates[1], "B"), "open"] = 100.0
    prices.loc[(dates[6], "B"), "open"] = 100.0
    signal_index = pd.MultiIndex.from_product(
        [[dates[0]], stocks], names=["datetime", "instrument"]
    )
    pred = pd.DataFrame({name: [2.0, 1.0] for name in SCORES}, index=signal_index)
    periods, summary = evaluate_portfolios(pred, prices, top_k=1, holding_days=5,
                                            one_way_cost_bps=10)
    expected = 1.1 * (0.999 ** 2) - 1
    assert np.isclose(periods.iloc[0]["ridge_original27_net"], expected)
    assert periods.iloc[0]["entry_date"] == dates[1]
    assert periods.iloc[0]["exit_date"] == dates[6]
    assert len(summary) == 3


def test_reject_duplicate_prices() -> None:
    """Reject ambiguous execution-price keys.

    Description:
        Verify duplicate price records cannot enter a portfolio simulation.
    Parameters:
        None: Uses a synthetic duplicated index.
    Returns:
        None: Raises AssertionError if duplicate detection fails.
    Notes:
        Duplicate keys could silently distort realized returns.
    """
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2022-01-03"), "A")] * 2,
        names=["datetime", "instrument"],
    )
    prices = pd.DataFrame({"open": [100, 101]}, index=idx)
    pred = pd.DataFrame({name: [1.0] for name in SCORES}, index=idx[:1])
    try:
        evaluate_portfolios(pred, prices)
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("Duplicate price keys were accepted")
