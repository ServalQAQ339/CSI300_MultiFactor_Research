"""Canonical names and provisional metadata for the original 27 CSI300 factors."""
from __future__ import annotations

BASE = ['OPEN_CLOSE_RATIO', 'HIGH_LOW_RANGE', 'CLOSE_LOCATION']
FAMILIES = {
    'MOM': 'momentum',
    'MA_RATIO': 'moving_average',
    'VOLATILITY': 'volatility',
    'VOLUME_RATIO': 'volume',
    'HIGH_POSITION': 'price_position',
    'LOW_POSITION': 'price_position',
}
WINDOWS = (5, 10, 20, 60)
FACTOR_NAMES = tuple(BASE + [f'{prefix}_{window}' for window in WINDOWS for prefix in FAMILIES])


def metadata_for(name: str) -> dict:
    """Description: Build provisional metadata without inventing source formulas.

    Parameters:
        name (str): Existing factor column name.
    Returns:
        dict: Factor family, lookback window, and unverified-definition flags.
    Notes:
        Exact expressions must be audited against factors/factor_engine.py.
    """
    if name not in FACTOR_NAMES:
        raise ValueError(f'Unknown original factor: {name}')
    if name in BASE:
        return {'family': 'candlestick', 'window': None}
    prefix, window = name.rsplit('_', 1)
    return {'family': FAMILIES[prefix], 'window': int(window)}
