"""Research configuration for CSI300 raw-factor five-day analysis."""
from pathlib import Path
DATA_ROOT = Path(r'D:\QuantResearch\CSI300_MultiFactor_Research')
SEGMENTS = {
    'train': ('2015-04-01', '2020-12-31'),
    'validation': ('2021-01-01', '2023-12-31'),
    'test': ('2024-01-01', '2026-09-30'),
}
START, END = '2015-04-01', '2026-09-30'
FACTOR_NAMES = ['OPEN_CLOSE_RATIO', 'HIGH_LOW_RANGE', 'CLOSE_LOCATION'] + [
    f'{name}_{window}'
    for window in (5, 10, 20, 60)
    for name in ('MOM', 'MA_RATIO', 'VOLATILITY', 'VOLUME_RATIO', 'HIGH_POSITION', 'LOW_POSITION')
]
