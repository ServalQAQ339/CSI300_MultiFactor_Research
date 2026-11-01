"""Run CSI300 label engineering using Qlib historical constituent intervals and prices."""
import argparse
import json
from pathlib import Path
import pandas as pd
from labels.label_engine import LabelConfig, build_labels
from labels.membership import load_membership_intervals, expand_membership
from labels.price_audit import summarize_price_audit

DEFAULT_QLIB = r'D:\QuantResearch\qlib_data\cn_data\qlib_bin'
DEFAULT_DATA = r'D:\QuantResearch\CSI300_MultiFactor_Research'


def fetch_qlib_prices(
    provider_uri: str, instruments: list[str], start: str, end: str
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Description: Extract all relevant stock prices and the shared market calendar. Parameters: provider_uri (str): Qlib binary directory; instruments (list[str]): historical members across the full period; start, end (str): inclusive extraction bounds. Returns: tuple[pd.DataFrame, pd.DatetimeIndex]: adjusted-price panel and market trading calendar. Notes: Queries stocks directly, including their post-removal trading dates; does not use index-filtered processed prices."""
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=provider_uri, region='cn')
    calendar = pd.DatetimeIndex(D.calendar(start_time=start, end_time=end, freq='day'))
    chunks = []
    for offset in range(0, len(instruments), 100):
        batch = instruments[offset:offset + 100]
        part = D.features(
            batch, ['$open', '$close', '$factor'], start_time=start, end_time=end, freq='day'
        )
        if not part.empty:
            chunks.append(
                part.reset_index().rename(
                    columns={'$open': 'open', '$close': 'close', '$factor': 'factor'}
                )
            )
        print(
            f'Fetched {min(offset + 100, len(instruments))}/{len(instruments)} instruments',
            flush=True,
        )
    if not chunks:
        raise ValueError('Qlib returned no prices for historical members')
    prices = pd.concat(chunks, ignore_index=True)
    if not {'instrument', 'datetime', 'open', 'close', 'factor'}.issubset(prices.columns):
        raise ValueError('Unexpected Qlib feature fields')
    return prices, calendar


def audit_universe(membership: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Description: Compare membership keys to existing yearly processed panel keys. Parameters: membership (pd.DataFrame): expanded signal universe; data_root (Path): existing research data directory. Returns: pd.DataFrame: per-year counts of mismatched instrument-date keys. Notes: Missing processed files are recorded, not treated as successful matches."""
    rows = []
    for year in range(2015, 2027):
        file = data_root / 'data' / 'processed' / f'csi300_daily_{year}.parquet'
        expected = membership.loc[
            membership['datetime'].dt.year.eq(year), ['datetime', 'instrument']
        ]
        if not file.exists():
            rows.append(
                {
                    'year': year,
                    'file_exists': False,
                    'expected': len(expected),
                    'processed': None,
                    'extra': None,
                    'missing': None,
                }
            )
            continue
        actual = pd.read_parquet(file).reset_index()[['datetime', 'instrument']]
        actual['datetime'] = pd.to_datetime(actual['datetime']).dt.normalize()
        actual = actual.loc[
            actual['datetime'].between('2015-04-01', '2026-09-30')
        ].drop_duplicates()
        exp_keys = pd.MultiIndex.from_frame(expected)
        act_keys = pd.MultiIndex.from_frame(actual)
        rows.append(
            {
                'year': year,
                'file_exists': True,
                'expected': len(exp_keys),
                'processed': len(act_keys),
                'extra': int((~act_keys.isin(exp_keys)).sum()),
                'missing': int((~exp_keys.isin(act_keys)).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Description: Build all 1/5/10-day labels and integrity reports with fixed paths by default. Parameters: None (CLI options). Returns: None; writes Parquet, CSV and JSON outputs. Notes: Does not certify index PIT publication timing, tradability, delisting returns, or independent adjustment accuracy."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--qlib-dir', default=DEFAULT_QLIB)
    parser.add_argument('--data-root', default=DEFAULT_DATA)
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2026-09-30')
    args = parser.parse_args()
    root, qlib_dir = Path(args.data_root), Path(args.qlib_dir)
    interval_file = qlib_dir / 'instruments' / 'csi300.txt'
    intervals = load_membership_intervals(interval_file)
    # Fetch calendar independently of the stock panel to preserve suspended trading days.
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=str(qlib_dir), region='cn')
    calendar = pd.DatetimeIndex(D.calendar(start_time=args.start, end_time=args.end, freq='day'))
    membership = expand_membership(intervals, calendar, '2015-04-01', args.end)
    stocks = sorted(membership['instrument'].unique().tolist())
    print(f'Signal rows: {len(membership)}, historical stocks: {len(stocks)}', flush=True)
    prices, calendar = fetch_qlib_prices(str(qlib_dir), stocks, args.start, args.end)
    issues, changes = summarize_price_audit(prices)
    labels = build_labels(prices, calendar, membership, LabelConfig(test_end=args.end))
    universe_audit = audit_universe(membership, root)
    output = root / 'data' / 'labels'
    report = root / 'reports' / 'label_engineering'
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output / 'forward_labels.parquet', index=False)
    issues.to_csv(report / 'price_audit.csv', index=False)
    changes.to_csv(report / 'factor_change_candidates.csv', index=False)
    universe_audit.to_csv(report / 'membership_audit.csv', index=False)
    quality = pd.concat(
        [
            labels.groupby(['split', f'label_status_{h}d'])
            .size()
            .rename('count')
            .reset_index()
            .assign(horizon=h)
            for h in (1, 5, 10)
        ],
        ignore_index=True,
    )
    quality.to_csv(report / 'label_quality.csv', index=False)
    summary = {
        'signal_rows': len(labels),
        'historical_stocks': len(stocks),
        'price_issue_rows': len(issues),
        'factor_change_candidates': len(changes),
        'universe_extra_rows': int(universe_audit['extra'].fillna(0).sum()),
        'universe_missing_rows': int(universe_audit['missing'].fillna(0).sum()),
        'status_counts': {
            str(h): labels[f'label_status_{h}d'].value_counts().to_dict() for h in (1, 5, 10)
        },
        'limitations': [
            'Index membership publication-time PIT not verified',
            'Qlib price adjustments not independently verified',
            'No executable limit-up/limit-down or liquidity validation',
            'Delisting payoff not independently reconstructed',
        ],
    }
    (report / 'label_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
