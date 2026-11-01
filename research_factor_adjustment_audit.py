"""Audit Qlib adjustment factors within the historical CSI300 research universe.

Description:
    Read the Qlib market calendar, historical membership intervals and full factor
    observations; compute market-day changes before filtering by membership.
    Write diagnostic reports without changing existing research datasets.
Notes:
    Threshold exceedances are candidates, not verified corporate actions.
    Historical membership effective dates are not proof of publication-time PIT.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

DEFAULT_QLIB = Path(r'D:\QuantResearch\qlib_data\cn_data\qlib_bin')
DEFAULT_DATA = Path(r'D:\QuantResearch\CSI300_MultiFactor_Research')
START = pd.Timestamp('2015-04-01')
END = pd.Timestamp('2026-09-30')
THRESHOLDS = (0.0001, 0.001, 0.01, 0.05)


def load_membership(path, start=START, end=END):
    """Load intersecting CSI300 membership intervals.

    Parameters:
        path (str or pathlib.Path): Qlib csi300.txt path.
        start (pandas.Timestamp): Research start, inclusive.
        end (pandas.Timestamp): Research end, inclusive.
    Returns:
        pandas.DataFrame: instrument, start, end, including repeated intervals.
    Notes:
        Date endpoints are assumed inclusive; this is not a PIT certification.
    """
    frame = pd.read_csv(
        path, sep='\t', header=None, names=['instrument', 'start', 'end'], usecols=[0, 1, 2]
    )
    frame['start'] = pd.to_datetime(frame['start'])
    frame['end'] = pd.to_datetime(frame['end'])
    if (
        frame[['instrument', 'start', 'end']].isna().any().any()
        or (frame['start'] > frame['end']).any()
    ):
        raise ValueError('Invalid membership intervals')
    return frame.loc[(frame['start'] <= end) & (frame['end'] >= start)].copy()


def compute_changes(frame, calendar):
    """Compute changes only across consecutive market-calendar sessions.

    Parameters:
        frame (pandas.DataFrame): instrument, datetime, factor, open, close.
        calendar (sequence): Complete ordered market trading dates.
    Returns:
        pandas.DataFrame: Original observations and previous factor, factor_change,
            consecutive, valid_pair and price_missing columns.
    Notes:
        Never forward-fills factors or skips missing market sessions.
    """
    d = frame.copy()
    d['datetime'] = pd.to_datetime(d['datetime'])
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).unique().sort_values()
    mapping = pd.Series(np.arange(len(cal), dtype=int), index=cal)
    if d.duplicated(['instrument', 'datetime']).any():
        raise ValueError('Duplicate instrument-date keys')
    d['calendar_pos'] = d['datetime'].map(mapping)
    if d['calendar_pos'].isna().any(): raise ValueError('Price dates absent from market calendar')
    d = d.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    group = d.groupby('instrument', sort=False)
    d['previous_factor'] = group['factor'].shift()
    d['previous_calendar_pos'] = group['calendar_pos'].shift()
    d['consecutive'] = (d['calendar_pos'] - d['previous_calendar_pos']).eq(1)
    d['valid_factor'] = np.isfinite(d['factor']) & (d['factor'] > 0)
    prev_valid = np.isfinite(d['previous_factor']) & (d['previous_factor'] > 0)
    d['valid_pair'] = d['consecutive'] & d['valid_factor'] & prev_valid
    d['factor_change'] = np.where(d['valid_pair'], d['factor'] / d['previous_factor'] - 1, np.nan)
    d['price_missing'] = (
        ~np.isfinite(d['open']) | ~np.isfinite(d['close']) | (d['open'] <= 0) | (d['close'] <= 0)
    )
    return d


def apply_membership(frame, intervals):
    """Keep factor observations inside at least one effective membership interval.

    Parameters:
        frame (pandas.DataFrame): Factor change observations with datetime.
        intervals (pandas.DataFrame): instrument, start, end membership intervals.
    Returns:
        pandas.DataFrame: Matching observations without duplication on overlaps.
    Notes:
        Factor changes must be computed before this filter so entry dates retain
        their immediately preceding market-session factor when available.
    """
    parts = []
    for instrument, d in frame.groupby('instrument', sort=False):
        ranges = intervals.loc[intervals['instrument'].eq(instrument), ['start', 'end']]
        if ranges.empty: continue
        eligible = np.zeros(len(d), dtype=bool)
        for a, b in ranges.itertuples(index=False, name=None):
            eligible |= d['datetime'].between(a, b).to_numpy()
        if eligible.any(): parts.append(d.loc[eligible])
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[:0].copy()


def summarize_thresholds(frame):
    """Summarize absolute adjustment-factor changes at specified thresholds.

    Parameters:
        frame (pandas.DataFrame): Research-eligible factor observations.
    Returns:
        pandas.DataFrame: Thresholds, counts, rates and missing-price counts.
    Notes:
        All threshold counts share the same eligible comparison denominator.
    """
    eligible = frame.loc[frame['valid_pair']].copy()
    n = len(eligible)
    values = eligible['factor_change'].abs()
    return pd.DataFrame(
        [
            {
                'threshold': t,
                'threshold_pct': t * 100,
                'eligible_comparisons': n,
                'exceedances': int((values > t).sum()),
                'exceedance_rate': float((values > t).sum() / n) if n else None,
                'exceedances_with_missing_price': int(
                    ((values > t) & eligible['price_missing']).sum()
                ),
            }
            for t in THRESHOLDS
        ]
    )


def label_overlap(events, labels, horizon=5):
    """Count large factor-change events within each valid label holding interval.

    Parameters:
        events (pandas.DataFrame): instrument, datetime, factor_change events.
        labels (pandas.DataFrame): instrument, signal_date, entry_date_{horizon}d,
            exit_date_{horizon}d, label_{horizon}d and label_status_{horizon}d.
        horizon (int): Holding-period label horizon.
    Returns:
        pandas.DataFrame: Valid labels with event counts and extreme-return flags.
    Notes:
        Event overlap is descriptive, not evidence of causal contamination.
        Holding intervals include entry and exit dates for conservative screening.
    """
    suffix = f'{horizon}d'
    entry = f'entry_date_{suffix}'
    exit_col = f'exit_date_{suffix}'
    label = f'label_{suffix}'
    status = f'label_status_{suffix}'
    required = ['instrument', 'signal_date', entry, exit_col, label, status]
    missing = set(required) - set(labels.columns)
    if missing: raise ValueError(f'Missing label columns: {sorted(missing)}')
    d = labels.loc[labels[status].eq('ok'), required].copy()
    d[entry] = pd.to_datetime(d[entry])
    d[exit_col] = pd.to_datetime(d[exit_col])
    d['significant_factor_events'] = 0
    ev = events.loc[
        events['valid_pair'] & events['factor_change'].abs().gt(0.01), ['instrument', 'datetime']
    ].copy()
    ev['datetime'] = pd.to_datetime(ev['datetime'])
    for instrument, group in d.groupby('instrument', sort=False):
        dates = np.sort(
            ev.loc[ev['instrument'].eq(instrument), 'datetime'].to_numpy(dtype='datetime64[ns]')
        )
        if not len(dates): continue
        left = group[entry].to_numpy(dtype='datetime64[ns]')
        right = group[exit_col].to_numpy(dtype='datetime64[ns]')
        d.loc[group.index, 'significant_factor_events'] = np.searchsorted(
            dates, right, side='right'
        ) - np.searchsorted(dates, left, side='left')
    d['absolute_return'] = d[label].abs()
    d['extreme_return_20pct'] = d['absolute_return'].gt(0.2)
    return d


def run(qlib_dir=DEFAULT_QLIB, data_dir=DEFAULT_DATA, batch_size=80):
    """Run a read-only research-universe factor adjustment audit.

    Parameters:
        qlib_dir (str or pathlib.Path): Qlib binary provider directory.
        data_dir (str or pathlib.Path): Existing D-drive research data root.
        batch_size (int): Number of instruments per Qlib features request.
    Returns:
        dict: Counts and report paths written to the research audit directory.
    Notes:
        Never overwrites the existing full-history factor audit or labels.
    """
    import qlib
    from qlib.data import D
    qlib_dir = Path(qlib_dir)
    data_dir = Path(data_dir)
    out = data_dir / 'reports' / 'factor_adjustment_audit' / 'research_universe'
    out.mkdir(parents=True, exist_ok=True)
    membership = load_membership(qlib_dir / 'instruments' / 'csi300.txt')
    stocks = sorted(membership['instrument'].unique())
    qlib.init(provider_uri=str(qlib_dir), region='cn')
    cal = pd.DatetimeIndex(D.calendar(start_time='2015-01-01', end_time='2026-09-30', freq='day'))
    frames = []
    for j in range(0, len(stocks), batch_size):
        batch = stocks[j : j + batch_size]
        x = D.features(
            batch,
            ['$factor', '$open', '$close'],
            start_time='2015-01-01',
            end_time='2026-09-30',
            freq='day',
        ).reset_index()
        x = x.rename(columns={'$factor': 'factor', '$open': 'open', '$close': 'close'})
        frames.append(x)
        print(f'Fetched {min(j + batch_size, len(stocks))}/{len(stocks)} instruments', flush=True)
    panel = pd.concat(frames, ignore_index=True)
    panel['datetime'] = pd.to_datetime(panel['datetime'])
    changes = compute_changes(panel, cal)
    research = apply_membership(changes, membership)
    research = research.loc[research['datetime'].between(START, END)].copy()
    thresholds = summarize_thresholds(research)
    thresholds.to_csv(out / 'research_universe_thresholds.csv', index=False)
    events = research.loc[research['valid_pair'] & research['factor_change'].abs().gt(0.01)].copy()
    events['absolute_change'] = events['factor_change'].abs()
    events.sort_values('absolute_change', ascending=False).head(200).to_csv(
        out / 'research_universe_largest_changes.csv', index=False
    )
    by_year = (
        research.assign(year=research['datetime'].dt.year)
        .groupby('year')
        .agg(
            rows=('instrument', 'size'),
            comparisons=('valid_pair', 'sum'),
            price_missing=('price_missing', 'sum'),
            invalid_factor=('valid_factor', lambda x: int((~x).sum())),
        )
        .reset_index()
    )
    by_year.to_csv(out / 'research_universe_by_year.csv', index=False)
    labels_path = data_dir / 'data' / 'labels' / 'forward_labels.parquet'; overlap_info = {}
    if labels_path.exists():
        cols = [
            'instrument',
            'signal_date',
            'entry_date_5d',
            'exit_date_5d',
            'label_5d',
            'label_status_5d',
        ]
        labels = pd.read_parquet(labels_path, columns=cols)
        overlap = label_overlap(research, labels)
        grouped = (
            overlap.groupby(['extreme_return_20pct', overlap['significant_factor_events'].gt(0)])
            .size()
            .rename('count')
            .reset_index()
        )
        grouped = grouped.rename(
            columns={'significant_factor_events': 'has_significant_factor_event'}
        )
        grouped.to_csv(out / 'research_universe_label_overlap.csv', index=False)
        overlap_info = {
            'valid_5d_labels': len(overlap),
            'labels_with_event': int(overlap['significant_factor_events'].gt(0).sum()),
            'extreme_5d_labels': int(overlap['extreme_return_20pct'].sum()),
        }
    summary = {
        'historical_research_stocks': len(stocks),
        'research_member_rows_with_qlib_keys': len(research),
        'valid_consecutive_comparisons': int(research['valid_pair'].sum()),
        'price_missing_rows': int(research['price_missing'].sum()),
        'invalid_factor_rows': int((~research['valid_factor']).sum()),
        'thresholds': thresholds.to_dict('records'),
        'label_overlap': overlap_info,
        'limitations': [
            'Membership publication-time PIT not verified',
            'Qlib adjustment factor and raw prices not independently verified',
            'Price availability does not prove trade executability',
            'Large factor changes are not verified corporate actions',
        ],
    }
    (out / 'research_universe_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('Reports:', out)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Audit Qlib adjustment factors within the dynamic CSI300 research universe.'
    )
    parser.add_argument('--qlib-dir', type=Path, default=DEFAULT_QLIB)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--batch-size', type=int, default=80)
    args = parser.parse_args()
    run(args.qlib_dir, args.data_dir, args.batch_size)
