"""Read-only validators for historical CSI300 membership, prices, and labels."""
from pathlib import Path
import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10)
SPLITS = {
    'train': ('2015-04-01', '2020-12-31'),
    'valid': ('2021-01-01', '2023-12-31'),
    'validation': ('2021-01-01', '2023-12-31'),
    'test': ('2024-01-01', '2026-09-30'),
}


def read_membership(path: Path) -> pd.DataFrame:
    """Description: Read Qlib historical membership intervals. Parameters: path (Path): csi300.txt path. Returns: pd.DataFrame: instrument, start, end. Notes: Inclusive intervals are assumed; publication-time PIT is not certified."""
    frame = pd.read_csv(path, sep=r'\s+', header=None, names=['instrument', 'start', 'end'])
    frame['start'] = pd.to_datetime(frame['start']); frame['end'] = pd.to_datetime(frame['end'])
    if (
        frame[['instrument', 'start', 'end']].isna().any().any()
        or (frame['start'] > frame['end']).any()
    ):
        raise ValueError('Invalid membership intervals')
    return frame


def membership_diagnostics(
    intervals: pd.DataFrame, processed_dir: Path, calendar: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Description: Reconcile all market-day membership keys with yearly processed keys. Parameters: intervals (pd.DataFrame): membership intervals; processed_dir (Path): annual Parquet directory; calendar (pd.DatetimeIndex): market dates. Returns: tuple[pd.DataFrame,pd.DataFrame]: yearly summary and missing/extra key details. Notes: No assumptions are made about the reason for missing processed rows."""
    calendar = pd.DatetimeIndex(calendar).normalize().unique().sort_values()
    records = []
    for instrument, group in intervals.groupby('instrument', sort=False):
        dates = pd.DatetimeIndex([])
        for row in group.itertuples():
            dates = dates.union(
                calendar[
                    (calendar >= max(row.start, pd.Timestamp('2015-04-01')))
                    & (calendar <= min(row.end, pd.Timestamp('2026-09-30')))
                ]
            )
        records.extend((date, instrument) for date in dates)
    expected = pd.MultiIndex.from_tuples(records, names=['datetime', 'instrument']).unique()
    summaries, details = [], []
    for year in range(2015, 2027):
        target = expected[expected.get_level_values('datetime').year == year]
        file = processed_dir / f'csi300_daily_{year}.parquet'
        if not file.exists():
            summaries.append(
                {
                    'year': year,
                    'file_exists': False,
                    'expected': len(target),
                    'processed': np.nan,
                    'extra': np.nan,
                    'missing': np.nan,
                }
            )
            continue
        actual = pd.read_parquet(file).reset_index()[['datetime', 'instrument']]
        actual['datetime'] = pd.to_datetime(actual['datetime']).dt.normalize()
        actual = actual.loc[actual['datetime'].between('2015-04-01', '2026-09-30')]
        duplicate_count = int(actual.duplicated().sum())
        observed = pd.MultiIndex.from_frame(actual.drop_duplicates())
        missing, extra = target.difference(observed), observed.difference(target)
        summaries.append(
            {
                'year': year,
                'file_exists': True,
                'expected': len(target),
                'processed': len(observed),
                'extra': len(extra),
                'missing': len(missing),
                'duplicate_keys': duplicate_count,
            }
        )
        details.extend(
            {'year': year, 'datetime': date, 'instrument': stock, 'difference': kind}
            for kind, keys in [('missing_processed', missing), ('extra_processed', extra)]
            for date, stock in keys
        )
    return pd.DataFrame(summaries), pd.DataFrame(
        details, columns=['year', 'datetime', 'instrument', 'difference']
    )


def price_diagnostics(
    issues: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Description: Summarize price issues and factor-relative changes by stock. Parameters: issues (pd.DataFrame): instrument/datetime/issue; candidates (pd.DataFrame): instrument/datetime/factor. Returns: tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]: issue attribution, threshold counts, largest changes. Notes: Candidate file may exclude unchanged factor rows; therefore comparisons are between consecutive candidate observations, not necessarily consecutive market days. Threshold counts are diagnostic only."""
    issue = issues.copy(); issue['year'] = pd.to_datetime(issue['datetime']).dt.year
    attribution = (
        issue.groupby(['year', 'issue'], dropna=False).size().rename('count').reset_index()
    )
    c = candidates[['instrument', 'datetime', 'factor']].copy()
    c['datetime'] = pd.to_datetime(c['datetime'])
    c = c.sort_values(['instrument', 'datetime'])
    previous = c.groupby('instrument')['factor'].shift(1)
    c['previous_factor'] = previous
    c['relative_change'] = (c['factor'] / previous - 1).where(
        previous.gt(0) & c['factor'].gt(0) & np.isfinite(previous) & np.isfinite(c['factor'])
    )
    c['gap_days'] = c.groupby('instrument')['datetime'].diff().dt.days
    valid = c['relative_change'].abs().dropna()
    thresholds = pd.DataFrame(
        [
            {
                'threshold': t,
                'count': int(valid.gt(t).sum()),
                'denominator_valid_pairs': len(valid),
                'percentage': float(valid.gt(t).mean() * 100) if len(valid) else np.nan,
            }
            for t in (0.0001, 0.001, 0.01, 0.05)
        ]
    )
    top = (
        c.loc[c['relative_change'].notna()]
        .assign(abs_change=lambda x: x['relative_change'].abs())
        .nlargest(100, 'abs_change')
    )
    return attribution, thresholds, top


def label_diagnostics(
    labels: pd.DataFrame, calendar: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Description: Recompute all label returns and validate offsets, statuses, and split boundaries. Parameters: labels (pd.DataFrame): forward_labels panel; calendar (pd.DatetimeIndex): market trading dates. Returns: tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]: grouped violations, normalized status counts, example violations. Notes: Dates must be market-calendar aligned; unverified tradability and PIT are outside scope."""
    calendar = pd.DatetimeIndex(calendar).normalize().unique().sort_values()
    dates = pd.to_datetime(labels['signal_date']).dt.normalize()
    split = labels['split'].astype(str).str.lower()
    positions = calendar.get_indexer(dates)
    all_counts, all_status, examples = [], [], []
    for h in HORIZONS:
        suffix = f'{h}d'
        status = labels[f'label_status_{suffix}'].astype(str)
        y = pd.to_numeric(labels[f'label_{suffix}'], errors='coerce')
        entry = pd.to_datetime(labels[f'entry_date_{suffix}'], errors='coerce').dt.normalize()
        exit_ = pd.to_datetime(labels[f'exit_date_{suffix}'], errors='coerce').dt.normalize()
        e = pd.to_numeric(labels[f'entry_open_adjusted_{suffix}'], errors='coerce')
        x = pd.to_numeric(labels[f'exit_open_adjusted_{suffix}'], errors='coerce')
        ok = status.eq('ok'); computed = x.div(e).sub(1)
        expected_entry = pd.Series(pd.NaT, index=labels.index, dtype='datetime64[ns]')
        expected_exit = expected_entry.copy()
        valid_entry = (positions >= 0) & (positions + 1 < len(calendar))
        valid_exit = (positions >= 0) & (positions + h + 1 < len(calendar))
        expected_entry.loc[valid_entry] = calendar[positions[valid_entry] + 1].values
        expected_exit.loc[valid_exit] = calendar[positions[valid_exit] + h + 1].values
        split_end = split.map({k: pd.Timestamp(v[1]) for k, v in SPLITS.items()})
        problems = {
            'duplicate_key': labels.duplicated(['instrument', 'signal_date']),
            'unknown_split': ~split.isin(SPLITS),
            'signal_not_on_calendar': positions < 0,
            'entry_date_mismatch': ok & entry.ne(expected_entry),
            'exit_date_mismatch': ok & exit_.ne(expected_exit),
            'invalid_ok_price': ok & (~np.isfinite(e) | ~np.isfinite(x) | e.le(0) | x.le(0)),
            'return_mismatch': ok
            & (~np.isfinite(y) | ~np.isclose(y, computed, rtol=1e-8, atol=1e-10, equal_nan=False)),
            'non_ok_has_label': ~ok & y.notna(),
            'split_boundary_leakage': ok & exit_.gt(split_end),
            'signal_outside_split': dates.lt(
                split.map({k: pd.Timestamp(v[0]) for k, v in SPLITS.items()})
            )
            | dates.gt(split_end),
        }
        for name, mask in problems.items():
            all_counts.append({'horizon': h, 'check': name, 'violations': int(mask.sum())})
            if mask.any():
                sample = labels.loc[mask, ['instrument', 'signal_date', 'split']].head(30).copy()
                sample['horizon'] = h
                sample['check'] = name
                examples.append(sample)
        counts = labels.groupby(['split', f'label_status_{suffix}'], dropna=False).size()
        all_status.extend(
            {'split': a, 'horizon': h, 'status': b, 'count': int(n)} for (a, b), n in counts.items()
        )
    return (
        pd.DataFrame(all_counts),
        pd.DataFrame(all_status),
        (
            pd.concat(examples, ignore_index=True)
            if examples
            else pd.DataFrame(columns=['instrument', 'signal_date', 'split', 'horizon', 'check'])
        ),
    )


def alignment_diagnostics(labels: pd.DataFrame, factors_dir: Path) -> pd.DataFrame:
    """Description: Compare label signal keys against annual enhanced factor keys. Parameters: labels (pd.DataFrame): label panel; factors_dir (Path): enhanced annual Parquet directory. Returns: pd.DataFrame: yearly key and duplicate statistics. Notes: Key alignment does not certify factor availability timing or factor value quality."""
    result = []
    keys = labels[['signal_date', 'instrument']].rename(columns={'signal_date': 'datetime'}).copy()
    keys['datetime'] = pd.to_datetime(keys['datetime']).dt.normalize()
    for year in range(2015, 2027):
        file = factors_dir / f'csi300_factors_{year}.parquet'
        target = keys.loc[keys['datetime'].dt.year.eq(year)]
        if not file.exists():
            result.append({'year': year, 'file_exists': False, 'label_keys': len(target)}); continue
        f = pd.read_parquet(file).reset_index()[['datetime', 'instrument']]
        f['datetime'] = pd.to_datetime(f['datetime']).dt.normalize()
        f = f.loc[f['datetime'].between('2015-04-01', '2026-09-30')]
        label_idx = pd.MultiIndex.from_frame(target.drop_duplicates())
        factor_idx = pd.MultiIndex.from_frame(f.drop_duplicates())
        result.append(
            {
                'year': year,
                'file_exists': True,
                'label_keys': len(label_idx),
                'factor_keys': len(factor_idx),
                'labels_without_factors': len(label_idx.difference(factor_idx)),
                'factors_without_labels': len(factor_idx.difference(label_idx)),
                'label_duplicate_keys': int(target.duplicated().sum()),
                'factor_duplicate_keys': int(f.duplicated().sum()),
            }
        )
    return pd.DataFrame(result)
