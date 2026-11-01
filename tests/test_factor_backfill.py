"""Synthetic tests for membership-preserving pre-entry factor history."""
# %%
import numpy as np
import pandas as pd
from factors.factor_backfill import generate_backfilled_year


def test_pre_entry_prices_and_no_future():
    """Verify pre-entry lookback, exact eligible output and future truncation.

    Parameters:
        None.

    Returns:
        None: Raises AssertionError when an invariant is violated.
    """
    dates = pd.bdate_range("2020-01-01", periods=90)
    entry = dates[65]
    eligible = pd.MultiIndex.from_product(
        [dates[65:75], ["SH600000"]], names=["datetime", "instrument"]
    )
    target = pd.DataFrame(index=eligible, data={"close": 1.0})
    prices = pd.DataFrame(
        {
            "open": 100.0 + np.arange(90),
            "high": 101.0 + np.arange(90),
            "low": 99.0 + np.arange(90),
            "close": 100.0 + np.arange(90),
            "volume": 1000.0,
        },
        index=dates,
    )

    def fetcher(instruments, fields, start_time, end_time, freq):
        frame = prices.loc[start_time:end_time].copy()
        frame.columns = ["$open", "$high", "$low", "$close", "$volume"]
        return (
            frame.assign(instrument=instruments[0])
            .reset_index(names="datetime")
            .set_index(["instrument", "datetime"])
        )

    result, stats = generate_backfilled_year(target, dates, windows=(5, 60), fetcher=fetcher)
    assert result.index.equals(eligible)
    assert np.isclose(
        result.loc[(entry, "SH600000"), "MOM_60"],
        prices.loc[entry, "close"] / prices.loc[dates[5], "close"] - 1,
    )

    assert pd.notna(result.loc[(entry, "SH600000"), "VOLATILITY_60"])
    assert stats["history_rows"] > len(target)
    truncated = target.loc[(slice(None, dates[68]), slice(None)), :]
    earlier, _ = generate_backfilled_year(truncated, dates, windows=(5, 60), fetcher=fetcher)
    pd.testing.assert_frame_equal(earlier, result.loc[earlier.index])


if __name__ == "__main__":
    test_pre_entry_prices_and_no_future()
    print("Synthetic backfill tests passed.")

# %%
