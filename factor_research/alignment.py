"""Annual factor ingestion and auditable label-left alignment."""
import numpy as np
import pandas as pd
from .config import FACTOR_NAMES, SEGMENTS, START, END


def load_inputs(root):
    """Description: Read twelve annual factor panels and forward labels.

    Parameters:
        root (Path): D-drive research data root.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Normalized factor and label tables.
    Notes:
        Rejects duplicate keys, unexpected schemas and missing annual files.
    """
    paths = sorted((root / 'data/factors_preprocessed/raw').glob('csi300_factors_*.parquet'))
    expected = [f'csi300_factors_{year}.parquet' for year in range(2015, 2027)]
    if [p.name for p in paths] != expected:
        raise ValueError('Annual factor files do not match 2015-2026 manifest')
    frames = []

    for path in paths:
        frame = pd.read_parquet(path)
        if list(frame.columns) != FACTOR_NAMES or list(frame.index.names) != [
            'datetime',
            'instrument',
        ]:
            raise ValueError(f'Unexpected factor schema: {path}')
        frames.append(frame.reset_index().rename(columns={'datetime': 'signal_date'}))

    factors = pd.concat(frames, ignore_index=True)
    labels = pd.read_parquet(root / 'data/labels/forward_labels.parquet')
    required = {
        'instrument',
        'signal_date',
        'split',
        'label_5d',
        'label_status_5d',
        'entry_date_5d',
        'exit_date_5d',
    }

    if not required.issubset(labels.columns):
        raise ValueError(f'Missing label columns: {required-set(labels.columns)}')

    for name, frame in [('factors', factors), ('labels', labels)]:
        frame['signal_date'] = pd.to_datetime(frame['signal_date'], errors='raise').dt.normalize()
        frame['instrument'] = frame['instrument'].astype(str).str.strip()
        if (
            frame[['signal_date', 'instrument']].isna().any().any()
            or frame.duplicated(['signal_date', 'instrument']).any()
        ):
            raise ValueError(f'Invalid or duplicate keys: {name}')

    return factors, labels


def prepare_panel(factors, labels):
    """Description: Left-join labels to factors and audit matching and validity.

    Parameters:
        factors (pd.DataFrame): Normalized annual factor panel.
        labels (pd.DataFrame): Forward labels and metadata.
    Returns:
        tuple[pd.DataFrame, dict]: Full research panel and audit counts.
    Notes:
        Valid label requires status ok, finite return and chronological entry/exit dates.
        Universe eligibility is inherited from upstream and not independently PIT-certified.
    """
    panel = labels.merge(
        factors[['signal_date', 'instrument'] + FACTOR_NAMES],
        on=['signal_date', 'instrument'],
        how='left',
        validate='one_to_one',
        indicator=True,
    )

    panel = panel.loc[panel.signal_date.between(START, END)].copy()
    panel['entry_date_5d'] = pd.to_datetime(panel['entry_date_5d'], errors='coerce')
    panel['exit_date_5d'] = pd.to_datetime(panel['exit_date_5d'], errors='coerce')
    value = pd.to_numeric(panel['label_5d'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    valid = (
        panel.label_status_5d.eq('ok')
        & value.notna()
        & panel.entry_date_5d.gt(panel.signal_date)
        & panel.exit_date_5d.gt(panel.entry_date_5d)
    )

    panel['research_label_5d'] = value.where(valid)
    panel['factor_matched'] = panel['_merge'].eq('both')

    for split, (start, end) in SEGMENTS.items():
        if not panel.loc[panel.split.eq(split), 'signal_date'].between(start, end).all():
            raise ValueError(f'Invalid split boundary: {split}')

    if not panel.split.isin(SEGMENTS).all(): raise ValueError('Unknown split values')
    audit = {
        'research_rows': len(panel),
        'factor_matched_rows': int(panel.factor_matched.sum()),
        'unmatched_label_rows': int((~panel.factor_matched).sum()),
        'valid_5d_label_rows': int(valid.sum()),
        'valid_5d_matched_rows': int((valid & panel.factor_matched).sum()),
        'factor_count': len(FACTOR_NAMES),
        'independent_membership_pit_verified': False,
        'execution_tradability_verified': False,
    }

    panel['purged_for_split'] = False

    for split, (_, end) in SEGMENTS.items():
        panel.loc[panel.split.eq(split), 'purged_for_split'] = (
            panel.loc[panel.split.eq(split), 'exit_date_5d'].gt(pd.Timestamp(end)).fillna(False)
        )

    audit['valid_labels_crossing_split_end'] = int((valid & panel.purged_for_split).sum())

    return panel.drop(columns='_merge'), audit
