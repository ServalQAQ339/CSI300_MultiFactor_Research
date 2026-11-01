"""Compute a small, explicitly defined subset of Qlib Alpha158 factors."""

import numpy as np
import pandas as pd


WINDOWS = (5, 10, 20)


def compute_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute selected Qlib-equivalent trailing features.

    Description:
        Calculate rolling correlation, volume dispersion, and price-range location.
    Parameters:
        frame (pd.DataFrame): OHLCV observations indexed by datetime and instrument.
    Returns:
        pd.DataFrame: Selected factors with the original two-level index.
    Notes:
        Input prices must share a consistent adjustment convention. Rows are
        sorted chronologically within each instrument; no forward fill is used.
    """
    required = {"close", "high", "low", "volume"}
    if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {"datetime", "instrument"}:
        raise ValueError("Expected MultiIndex with datetime and instrument levels")
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing OHLCV columns: {sorted(required - set(frame.columns))}")
    if not frame.index.is_unique:
        raise ValueError("Duplicate instrument-date keys")

    data = frame.reorder_levels(["instrument", "datetime"]).sort_index().copy()
    data.index = data.index.set_names(["instrument", "datetime"])
    pieces = []
    for instrument, stock in data.groupby(level="instrument", sort=False):
        stock = stock.droplevel("instrument")
        close = stock["close"].astype(float)
        high = stock["high"].astype(float)
        low = stock["low"].astype(float)
        volume = stock["volume"].astype(float)
        log_volume = np.log(volume.where(volume >= 0) + 1)
        result = pd.DataFrame(index=stock.index)
        for window in WINDOWS:
            result[f"CORR{window}"] = close.rolling(window, min_periods=window).corr(log_volume)
            result[f"VSTD{window}"] = (
                volume.rolling(window, min_periods=window).std(ddof=1) / (volume + 1e-12)
            )
            maximum = high.rolling(window, min_periods=window).max()
            minimum = low.rolling(window, min_periods=window).min()
            result[f"RSV{window}"] = (close - minimum) / (maximum - minimum + 1e-12)
        pieces.append((instrument, result))

    # Instrument names are restored using group keys rather than DataFrame metadata.
    outputs = []
    for instrument, result in pieces:
        result["instrument"] = instrument
        outputs.append(result.reset_index().set_index(["datetime", "instrument"]))
    return pd.concat(outputs).sort_index().replace([np.inf, -np.inf], np.nan)
