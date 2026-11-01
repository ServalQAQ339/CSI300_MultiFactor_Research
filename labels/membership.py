"""Parse historical CSI300 constituent intervals and expand them on exchange dates."""
from pathlib import Path
import pandas as pd


def load_membership_intervals(path: str | Path) -> pd.DataFrame:
    """Description: Load historical index membership intervals. Parameters: path (str | Path): Qlib csi300.txt location. Returns: pd.DataFrame: instrument, start, end intervals. Notes: Interval endpoints are treated as inclusive; publication-time availability is not certified."""
    m = pd.read_csv(
        path,
        sep=r'\s+',
        header=None,
        names=['instrument', 'start', 'end'],
        dtype={'instrument': str},
    )
    if m.empty or m.isna().any().any():
        raise ValueError('Empty or incomplete constituent intervals')
    m['start'], m['end'] = pd.to_datetime(m['start']), pd.to_datetime(m['end'])
    if (m['start'] > m['end']).any():
        raise ValueError('Constituent interval starts after its end')
    return m.sort_values(['instrument', 'start', 'end']).reset_index(drop=True)


def expand_membership(
    intervals: pd.DataFrame, calendar: pd.DatetimeIndex, start: str, end: str
) -> pd.DataFrame:
    """Description: Expand index intervals into dated signal eligibility. Parameters: intervals (pd.DataFrame): inclusive membership ranges; calendar (pd.DatetimeIndex): exchange trading dates; start, end (str): research signal date bounds. Returns: pd.DataFrame: unique instrument, datetime, eligible rows. Notes: Overlapping source intervals are deduplicated, and PIT availability remains unverified."""
    cal = pd.DatetimeIndex(calendar).normalize().unique().sort_values()
    cal = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    pieces = []
    for row in intervals.itertuples(index=False):
        left, right = cal.searchsorted(row.start, side='left'), cal.searchsorted(
            row.end, side='right'
        )
        if right > left:
            pieces.append(pd.DataFrame({'instrument': row.instrument, 'datetime': cal[left:right]}))
    if not pieces:
        raise ValueError('No index members in the requested trading period')
    result = pd.concat(pieces, ignore_index=True).drop_duplicates(['instrument', 'datetime'])
    result['eligible'] = True
    return result.sort_values(['datetime', 'instrument']).reset_index(drop=True)
