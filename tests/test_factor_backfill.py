"""Synthetic tests for membership-preserving pre-entry factor history."""
# %%
import numpy as np
import pandas as pd
from data.history_loader import fetch_qlib_features
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


def test_fetch_qlib_features_batches_and_skips_empty():
    """Verify batched Qlib extraction, column naming and empty-batch handling.

    Parameters:
        None.

    Returns:
        None: Raises AssertionError when an invariant is violated.

    Notes:
        A fake fetcher replaces D.features; no Qlib data is read.
    """
    calls = []

    def fetcher(instruments, fields, start_time, end_time, freq):
        calls.append(list(instruments))
        known = [code for code in instruments if code != "MISSING"]
        if not known:
            return pd.DataFrame(columns=fields)
        index = pd.MultiIndex.from_product(
            [known, pd.to_datetime(["2020-01-02", "2020-01-03"])],
            names=["instrument", "datetime"],
        )
        return pd.DataFrame({field: 1.0 for field in fields}, index=index)

    result = fetch_qlib_features(
        ["A", "B", "MISSING"], ["$open", "$factor"], "2020-01-01", "2020-01-31",
        batch_size=2, fetcher=fetcher,
    )
    assert calls == [["A", "B"], ["MISSING"]]
    assert list(result.columns) == ["instrument", "datetime", "open", "factor"]
    assert result["instrument"].tolist() == ["A", "A", "B", "B"]

    empty = fetch_qlib_features(["MISSING"], ["$open"], "2020-01-01", "2020-01-31",
                                fetcher=fetcher)
    assert empty.empty and list(empty.columns) == ["instrument", "datetime", "open"]
