"""Calculate additional trailing OHLCV features for batch screening."""

import numpy as np
import pandas as pd


WINDOWS = (5, 10, 20, 30, 60)


def compute_extra_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate thirty trailing OHLCV features.

    Description:
        Compute six distinct feature families over five rolling windows.
    Parameters:
        frame (pd.DataFrame): Unique datetime/instrument-indexed OHLCV panel.
    Returns:
        pd.DataFrame: Thirty trailing candidate columns on the same index.
    Notes:
        Signals assume the daily bar is known at the end of the signal day.
    """
    if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {"datetime", "instrument"}:
        raise ValueError("Expected datetime/instrument MultiIndex")
    if not frame.index.is_unique:
        raise ValueError("Duplicate OHLCV keys")
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing OHLCV columns: {sorted(required - set(frame.columns))}")

    results = []
    for instrument, stock in frame.reorder_levels(["instrument", "datetime"]).sort_index().groupby(level=0):
        stock = stock.droplevel(0)
        close = stock["close"].astype(float).where(lambda x: x > 0)
        open_ = stock["open"].astype(float).where(lambda x: x > 0)
        high = stock["high"].astype(float).where(lambda x: x > 0)
        low = stock["low"].astype(float).where(lambda x: x > 0)
        volume = stock["volume"].astype(float).where(lambda x: x >= 0)
        ret = close.pct_change(fill_method=None)
        log_ret = np.log(close).diff()
        spread = (high - low) / close
        turnover_proxy = close * volume
        out = pd.DataFrame(index=stock.index)

        for window in WINDOWS:
            out[f"LOGRET_SUM_{window}"] = log_ret.rolling(window, min_periods=window).sum()
            out[f"RET_AUTOCORR_{window}"] = ret.rolling(window, min_periods=window).corr(ret.shift(1))
            out[f"RANGE_STD_{window}"] = spread.rolling(window, min_periods=window).std()
            out[f"BODY_MEAN_{window}"] = ((close - open_) / open_).rolling(window, min_periods=window).mean()
            out[f"DOLLAR_VOLUME_CV_{window}"] = (turnover_proxy.rolling(window, min_periods=window).std() / turnover_proxy.rolling(window, min_periods=window).mean().replace(0, np.nan))
            out[f"UPDOWN_VOL_RATIO_{window}"] = (volume.where(ret > 0, 0).rolling(window, min_periods=window).sum() / volume.where(ret < 0, 0).rolling(window, min_periods=window).sum().replace(0, np.nan))

        out["instrument"] = instrument
        results.append(out.reset_index().set_index(["datetime", "instrument"]))

    return pd.concat(results).sort_index().replace([np.inf, -np.inf], np.nan)
