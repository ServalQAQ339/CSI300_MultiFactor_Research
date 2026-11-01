"""Generate additive, trailing OHLCV features inspired by Alpha158 research."""

import numpy as np
import pandas as pd


WINDOWS = (5, 10, 20, 30, 60)
FAMILIES = (
    "RET_STD", "RET_SKEW", "RET_KURT", "RET_POS_RATIO", "RET_NEG_RATIO",
    "RET_VOLUME_CORR", "ABSRET_VOLUME_CORR", "VOLUME_CV", "VOLUME_TREND",
    "RANGE_MEAN", "CLOSE_RANGE_POSITION", "DOWNSIDE_STD",
)


def compute_batch_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute 60 non-forward-looking rolling feature candidates.

    Description:
        Compute twelve OHLCV-derived feature families over five trailing windows.
    Parameters:
        frame (pd.DataFrame): Close, high, low and volume indexed by
            (datetime, instrument), with one observation per key.
    Returns:
        pd.DataFrame: Sixty candidate columns indexed by (datetime, instrument).
    Notes:
        Features use current and earlier observations only. They assume the
        signal is formed after the current daily bar is fully available.
        Outputs are inspired features, not certified Qlib Alpha158 expressions.
    """
    required = {"close", "high", "low", "volume"}
    if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {
        "datetime", "instrument"
    }:
        raise ValueError("Expected MultiIndex levels datetime and instrument")
    if not frame.index.is_unique:
        raise ValueError("Duplicate factor keys")
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")

    data = frame.reorder_levels(["instrument", "datetime"]).sort_index()
    outputs = []

    for instrument, stock in data.groupby(level="instrument", sort=False):
        stock = stock.droplevel("instrument")
        close = stock["close"].astype(float).where(lambda x: x > 0)
        high = stock["high"].astype(float).where(lambda x: x > 0)
        low = stock["low"].astype(float).where(lambda x: x > 0)
        volume = stock["volume"].astype(float).where(lambda x: x >= 0)
        ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        abs_ret = ret.abs()
        downside = ret.clip(upper=0)
        intraday_range = (high - low) / close
        result = pd.DataFrame(index=stock.index)

        for window in WINDOWS:
            roll_ret = ret.rolling(window, min_periods=window)
            roll_vol = volume.rolling(window, min_periods=window)
            vol_mean = roll_vol.mean().replace(0, np.nan)
            high_max = high.rolling(window, min_periods=window).max()
            low_min = low.rolling(window, min_periods=window).min()
            span = (high_max - low_min).replace(0, np.nan)
            positive = ret.gt(0).where(ret.notna()).rolling(window, min_periods=window).mean()
            negative = ret.lt(0).where(ret.notna()).rolling(window, min_periods=window).mean()

            result[f"RET_STD_{window}"] = roll_ret.std(ddof=1)
            result[f"RET_SKEW_{window}"] = roll_ret.skew()
            result[f"RET_KURT_{window}"] = roll_ret.kurt()
            result[f"RET_POS_RATIO_{window}"] = positive
            result[f"RET_NEG_RATIO_{window}"] = negative
            result[f"RET_VOLUME_CORR_{window}"] = roll_ret.corr(np.log1p(volume))
            result[f"ABSRET_VOLUME_CORR_{window}"] = abs_ret.rolling(
                window, min_periods=window
            ).corr(np.log1p(volume))
            result[f"VOLUME_CV_{window}"] = roll_vol.std(ddof=1) / vol_mean
            result[f"VOLUME_TREND_{window}"] = volume / vol_mean - 1
            result[f"RANGE_MEAN_{window}"] = intraday_range.rolling(
                window, min_periods=window
            ).mean()
            result[f"CLOSE_RANGE_POSITION_{window}"] = (close - low_min) / span
            result[f"DOWNSIDE_STD_{window}"] = downside.rolling(
                window, min_periods=window
            ).std(ddof=1)

        result["instrument"] = instrument
        outputs.append(result.reset_index().set_index(["datetime", "instrument"]))

    if not outputs:
        return pd.DataFrame(columns=[f"{name}_{w}" for w in WINDOWS for name in FAMILIES])
    return pd.concat(outputs).sort_index().replace([np.inf, -np.inf], np.nan)
