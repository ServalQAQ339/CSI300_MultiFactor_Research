"""Audit complete Qlib adjustment-factor histories without modifying research data."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

DEFAULT_QLIB = Path(r'D:\QuantResearch\qlib_data\cn_data\qlib_bin')
DEFAULT_ROOT = Path(r'D:\QuantResearch\CSI300_MultiFactor_Research')
THRESHOLDS = (0.0001, 0.001, 0.01, 0.05)


def load_membership_stocks(path):
    """Description: Read historical CSI300 constituent identifiers. Parameters: path (Path): Qlib tab-separated interval file. Returns: list[str]: Sorted unique stock identifiers. Notes: Intervals are used only to select stocks, not to assert publication-time availability."""
    table = pd.read_csv(
        path, sep='\t', header=None, names=['instrument', 'start', 'end'], dtype=str
    )
    if table.empty or table['instrument'].isna().any():
        raise ValueError('Membership file is empty or contains missing instruments')
    return sorted(table['instrument'].unique().tolist())


def analyze_factor_panel(panel, calendar):
    """Description: Compute consecutive-market-day adjustment-factor changes and classify price availability. Parameters: panel (DataFrame): Columns instrument, datetime, factor, open, close, volume. calendar (DatetimeIndex or sequence): Complete market trading calendar. Returns: DataFrame: One row per stock observation with prior factor, calendar gap, relative change, and price status. Notes: Missing factors never use implicit forward fill; gaps between valid factor observations are flagged and excluded from strict consecutive-day threshold statistics."""
    required = {'instrument', 'datetime', 'factor', 'open', 'close', 'volume'}
    if not required.issubset(panel.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(panel.columns))}')
    d = panel.copy()
    d['datetime'] = pd.to_datetime(d['datetime']).dt.normalize()
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
    if cal.empty or d.duplicated(['instrument', 'datetime']).any():
        raise ValueError('Empty calendar or duplicate instrument-date keys')
    d = d.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    positions = pd.Series(np.arange(len(cal)), index=cal)
    d['calendar_pos'] = d['datetime'].map(positions)
    if d['calendar_pos'].isna().any():
        raise ValueError('Panel dates not found in market calendar')
    f = pd.to_numeric(d['factor'], errors='coerce')
    d['factor_valid'] = np.isfinite(f) & (f > 0)
    d['valid_factor'] = f.where(d['factor_valid'])
    groups = d.groupby('instrument', sort=False)
    d['previous_factor'] = groups['valid_factor'].shift(1)
    d['previous_calendar_pos'] = groups['calendar_pos'].shift(1)
    d['calendar_gap'] = d['calendar_pos'] - d['previous_calendar_pos']
    d['consecutive'] = d['calendar_gap'].eq(1) & d['previous_factor'].notna() & d['factor_valid']
    d['relative_change'] = (d['valid_factor'] / d['previous_factor'] - 1).where(d['consecutive'])
    d['abs_change'] = d['relative_change'].abs()
    d['price_missing'] = (
        d[['open', 'close', 'volume']]
        .apply(lambda s: ~np.isfinite(pd.to_numeric(s, errors='coerce')))
        .any(axis=1)
    )

    return d


def summarize_changes(audit, thresholds=THRESHOLDS):
    """Description: Summarize strict consecutive-day factor movements by threshold. Parameters: audit (DataFrame): Output of analyze_factor_panel. thresholds (sequence[float]): Absolute relative-change cutoffs. Returns: DataFrame: Threshold, eligible comparisons, and exceedance counts. Notes: Counts are observations, not verified corporate actions."""
    values = audit.loc[audit['consecutive'], 'abs_change'].dropna()

    return pd.DataFrame(
        [
            {
                'threshold': float(t),
                'threshold_pct': 100 * float(t),
                'eligible_comparisons': int(len(values)),
                'exceedances': int(values.gt(t).sum()),
                'exceedance_rate': float(values.gt(t).mean()) if len(values) else np.nan,
            }
            for t in thresholds
        ]
    )


def fetch_qlib(stocks, qlib_dir, start, end, batch_size=80):
    """Description: Retrieve full historical Qlib factor and daily trading fields in bounded stock batches. Parameters: stocks (list[str]): Instruments. qlib_dir (Path): Qlib binary directory. start (str): First date. end (str): Last date. batch_size (int): Stocks per request. Returns: DataFrame: Normalized stock-date price panel. Notes: Qlib may return missing rows and price values; no implicit filling is performed."""
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(qlib_dir), region='cn')
    chunks = []

    for k in range(0, len(stocks), batch_size):
        batch = D.features(
            stocks[k : k + batch_size],
            ['$factor', '$open', '$close', '$volume'],
            start_time=start,
            end_time=end,
        ).reset_index()

        chunks.append(
            batch.rename(
                columns={
                    '$factor': 'factor',
                    '$open': 'open',
                    '$close': 'close',
                    '$volume': 'volume',
                }
            )
        )
        print(f'Fetched {min(k + batch_size, len(stocks))}/{len(stocks)} instruments', flush=True)

    return (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=['instrument', 'datetime', 'factor', 'open', 'close', 'volume'])
    )


def run(qlib_dir, root, start, end):
    """Description: Execute read-only adjustment-factor diagnostics and export audit reports. Parameters: qlib_dir (Path): Qlib binary root. root (Path): Research data root. start (str): Audit start date. end (str): Audit end date. Returns: dict: Summary statistics and output paths. Notes: Does not overwrite labels, factors, or existing audit reports."""
    import qlib
    from qlib.data import D

    stocks = load_membership_stocks(qlib_dir / 'instruments' / 'csi300.txt')
    panel = fetch_qlib(stocks, qlib_dir, start, end)
    calendar = pd.DatetimeIndex(D.calendar(start_time=start, end_time=end, freq='day'))
    audit = analyze_factor_panel(panel, calendar)
    threshold = summarize_changes(audit)
    output = root / 'reports' / 'factor_adjustment_audit'
    output.mkdir(parents=True, exist_ok=True)

    threshold.to_csv(output / 'factor_change_thresholds.csv', index=False)
    audit.nlargest(200, 'abs_change')[
        [
            'instrument',
            'datetime',
            'previous_factor',
            'factor',
            'relative_change',
            'calendar_gap',
            'price_missing',
        ]
    ].to_csv(output / 'largest_factor_changes.csv', index=False)
    audit.assign(year=audit['datetime'].dt.year).groupby('year').agg(
        observations=('instrument', 'size'),
        valid_factor=('factor_valid', 'sum'),
        consecutive_comparisons=('consecutive', 'sum'),
        price_missing=('price_missing', 'sum'),
    ).reset_index().to_csv(output / 'factor_audit_by_year.csv', index=False)
    audit.loc[
        (audit['calendar_gap'].gt(1)) | (~audit['factor_valid']),
        ['instrument', 'datetime', 'factor', 'calendar_gap', 'factor_valid', 'price_missing'],
    ].to_csv(output / 'factor_gap_or_invalid.csv', index=False)

    summary = {
        'stocks': len(stocks),
        'panel_rows': len(audit),
        'consecutive_comparisons': int(audit['consecutive'].sum()),
        'price_missing_rows': int(audit['price_missing'].sum()),
        'invalid_factor_rows': int((~audit['factor_valid']).sum()),
        'gap_rows': int(audit['calendar_gap'].gt(1).sum()),
        'thresholds': threshold.to_dict(orient='records'),
        'limitations': [
            'Threshold exceedances are not verified corporate actions',
            'Factor adjustment convention and raw prices require independent source checks',
            'Membership publication-time PIT remains unverified',
        ],
    }

    (output / 'factor_audit_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Reports: {output}')

    return summary


def main():
    """Description: Parse command-line paths and run the audit. Parameters: None. Returns: None. Notes: Default paths match the existing Windows CSI300 research project."""
    parser = argparse.ArgumentParser(description='Read-only Qlib factor adjustment audit')
    parser.add_argument('--qlib-dir', type=Path, default=DEFAULT_QLIB)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2026-09-30')
    args = parser.parse_args()
    run(args.qlib_dir, args.root, args.start, args.end)


if __name__ == '__main__':
    main()
