"""Audit CSI300 forward-return labels against Qlib endpoint prices and factors."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_ROOT = r'D:\QuantResearch\CSI300_MultiFactor_Research'
DEFAULT_QLIB = r'D:\QuantResearch\qlib_data\cn_data\qlib_bin'
HORIZONS = (1, 5, 10)


def calculate_endpoint_checks(
    labels: pd.DataFrame, prices: pd.DataFrame, horizon: int, tolerance: float = 1e-8
) -> pd.DataFrame:
    """Description: Independently recompute adjusted and factor-decomposed endpoint returns for a single horizon.
    Parameters: labels (pd.DataFrame): label table with instrument, signal date, endpoint dates, adjusted prices and status; prices (pd.DataFrame): Qlib observations with instrument, datetime, open and factor; horizon (int): holding period in market days; tolerance (float): absolute numeric discrepancy tolerance.
    Returns: pd.DataFrame: one record per valid label with endpoint checks, factor ratio, raw-price return, errors and event flags.
    Notes: The decomposition is algebraic and does not independently verify corporate actions. Missing Qlib endpoint keys are explicitly counted as failures. Factor event flag measures the net factor change between entry and exit, not every intervening event.
    """
    h = f'{horizon}d'
    subset = labels.loc[
        labels[f'label_status_{h}'].eq('ok'),
        [
            'instrument',
            'signal_date',
            f'entry_date_{h}',
            f'exit_date_{h}',
            f'label_{h}',
            f'entry_open_adjusted_{h}',
            f'exit_open_adjusted_{h}',
        ],
    ].copy()

    subset = subset.rename(
        columns={
            f'entry_date_{h}': 'entry_date',
            f'exit_date_{h}': 'exit_date',
            f'label_{h}': 'label',
            f'entry_open_adjusted_{h}': 'stored_entry',
            f'exit_open_adjusted_{h}': 'stored_exit',
        }
    )

    subset['signal_date'] = pd.to_datetime(subset['signal_date'])
    for col in ('entry_date', 'exit_date'):
        subset[col] = pd.to_datetime(subset[col])
    p = prices[['instrument', 'datetime', 'open', 'factor']].copy()
    p['datetime'] = pd.to_datetime(p['datetime'])
    if p.duplicated(['instrument', 'datetime']).any():
        raise ValueError('Qlib price keys are not unique')

    for side in ('entry', 'exit'):
        endpoint = p.rename(
            columns={'datetime': f'{side}_date', 'open': f'{side}_open', 'factor': f'{side}_factor'}
        )
        subset = subset.merge(
            endpoint,
            on=['instrument', f'{side}_date'],
            how='left',
            validate='many_to_one',
            indicator=f'{side}_merge',
        )

    a, b = subset['entry_open'].to_numpy(dtype=float), subset['exit_open'].to_numpy(dtype=float)
    fa, fb = subset['entry_factor'].to_numpy(dtype=float), subset['exit_factor'].to_numpy(
        dtype=float
    )
    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & np.isfinite(fa)
        & np.isfinite(fb)
        & (a > 0)
        & (b > 0)
        & (fa > 0)
        & (fb > 0)
    )

    subset['endpoint_valid'] = valid
    subset['recomputed_return'] = np.where(valid, b / a - 1, np.nan)
    subset['factor_ratio'] = np.where(valid, fb / fa, np.nan)
    subset['raw_price_return'] = np.where(valid, (b / fb) / (a / fa) - 1, np.nan)
    subset['decomposed_return'] = np.where(
        valid, (1 + subset['raw_price_return']) * subset['factor_ratio'] - 1, np.nan
    )
    subset['label_error'] = (subset['label'] - subset['recomputed_return']).abs()
    subset['decomposition_error'] = (
        subset['recomputed_return'] - subset['decomposed_return']
    ).abs()
    subset['entry_price_error'] = (subset['stored_entry'] - subset['entry_open']).abs()
    subset['exit_price_error'] = (subset['stored_exit'] - subset['exit_open']).abs()
    subset['net_factor_change_gt_1pct'] = (subset['factor_ratio'] - 1).abs() > 0.01
    subset['net_factor_change_gt_5pct'] = (subset['factor_ratio'] - 1).abs() > 0.05
    subset['audit_pass'] = valid & subset[
        ['label_error', 'decomposition_error', 'entry_price_error', 'exit_price_error']
    ].le(tolerance).all(axis=1)
    subset['horizon'] = horizon

    return subset


def summarize_checks(checks: pd.DataFrame) -> dict:
    """Description: Aggregate audit checks by horizon. Parameters: checks (pd.DataFrame): endpoint audit results. Returns: dict: JSON-serializable counts and maximum errors. Notes: Numeric success is not a claim of economic correctness."""
    result = {}

    for h, part in checks.groupby('horizon'):
        result[str(h)] = {
            'valid_labels': int(len(part)),
            'missing_qlib_endpoint_keys': int(
                ((part['entry_merge'] != 'both') | (part['exit_merge'] != 'both')).sum()
            ),
            'invalid_endpoint_values': int((~part['endpoint_valid']).sum()),
            'audit_failures': int((~part['audit_pass']).sum()),
            'net_factor_change_gt_1pct': int(part['net_factor_change_gt_1pct'].sum()),
            'net_factor_change_gt_5pct': int(part['net_factor_change_gt_5pct'].sum()),
            'max_label_error': (
                float(part['label_error'].max()) if part['label_error'].notna().any() else None
            ),
            'max_decomposition_error': (
                float(part['decomposition_error'].max())
                if part['decomposition_error'].notna().any()
                else None
            ),
        }

    return result


def fetch_prices(provider_uri: str, stocks: list[str], start: str, end: str) -> pd.DataFrame:
    """Description: Read unfiltered Qlib adjusted opening prices and adjustment factors in batches. Parameters: provider_uri (str): Qlib binary directory; stocks (list[str]): label stock codes; start (str): first date; end (str): final date. Returns: pd.DataFrame: instrument-date open/factor panel. Notes: Does not filter to index membership, preserving post-removal exit prices."""
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=provider_uri, region='cn')
    chunks = []

    for offset in range(0, len(stocks), 80):
        batch = stocks[offset:offset + 80]
        part = D.features(batch, ['$open', '$factor'], start_time=start, end_time=end, freq='day')
        if not part.empty:
            chunks.append(part.reset_index().rename(columns={'$open': 'open', '$factor': 'factor'}))
        print(f'Fetched {min(offset + 80, len(stocks))}/{len(stocks)} instruments', flush=True)

    if not chunks:
        raise ValueError('Qlib returned no price data')

    return pd.concat(chunks, ignore_index=True)


def main() -> None:
    """Description: Run read-only return consistency checks and write reports. Parameters: None; configurable CLI paths. Returns: None; writes CSV and JSON files. Notes: Reports do not modify labels or historical data."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=DEFAULT_ROOT)
    parser.add_argument('--qlib-dir', default=DEFAULT_QLIB)
    parser.add_argument('--tolerance', type=float, default=1e-8)
    args = parser.parse_args()
    root = Path(args.data_root)
    labels = pd.read_parquet(root / 'data' / 'labels' / 'forward_labels.parquet')
    stocks = sorted(labels['instrument'].unique().tolist())
    prices = fetch_prices(args.qlib_dir, stocks, '2015-01-01', '2026-09-30')
    reports = root / 'reports' / 'internal_return_audit'
    reports.mkdir(parents=True, exist_ok=True)
    all_checks = []

    for h in HORIZONS:
        checks = calculate_endpoint_checks(labels, prices, h, args.tolerance)
        all_checks.append(checks)
        print(
            f'Checked {h}D: {len(checks)} valid labels, {(~checks["audit_pass"]).sum()} failures',
            flush=True,
        )

    combined = pd.concat(all_checks, ignore_index=True)
    summary = {
        'source_label_rows': int(len(labels)),
        'stocks': int(len(stocks)),
        'tolerance': args.tolerance,
        'by_horizon': summarize_checks(combined),
        'limitations': [
            'Internal algebraic agreement cannot certify corporate-action correctness',
            'Net endpoint factor changes may hide offsetting intra-holding-period changes',
            'Qlib data do not independently establish price PIT, index PIT or executable prices',
        ],
    }

    combined.loc[~combined['audit_pass']].to_csv(
        reports / 'return_consistency_failures.csv', index=False
    )
    combined.loc[
        combined['net_factor_change_gt_5pct'],
        [
            'instrument',
            'signal_date',
            'horizon',
            'entry_date',
            'exit_date',
            'label',
            'raw_price_return',
            'factor_ratio',
            'audit_pass',
        ],
    ].to_csv(reports / 'large_net_factor_change_labels.csv', index=False)
    pd.DataFrame(
        [{'horizon': int(h), **stats} for h, stats in summary['by_horizon'].items()]
    ).to_csv(reports / 'return_consistency_summary.csv', index=False)
    (reports / 'internal_return_audit_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Reports: {reports}')


if __name__ == '__main__':
    main()
