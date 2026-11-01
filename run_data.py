"""Data-stage entrypoint: yearly OHLCV panels, forward labels and label validation.

Usage:
    python run_data.py panels [--mode {missing,full}]
    python run_data.py labels [--qlib-dir DIR] [--data-root DIR] [--start DATE] [--end DATE]
    python run_data.py check-labels [--data-root DIR] [--qlib-dir DIR]

Every heavy dependency (pandas, Qlib, project data modules) is imported inside the subcommand
handlers, so importing this module never touches data.
"""

import argparse
import json
from pathlib import Path

from config import AUDIT_OUTPUT_DIR, DATA_OUTPUT_DIR, DATA_ROOT, QLIB_DATA_DIR

DEFAULT_QLIB = str(QLIB_DATA_DIR)
DEFAULT_DATA = str(DATA_ROOT)


def run_complete_data_pipeline(overwrite: bool = False) -> None:
    """
    Execute the CSI300 data extraction and auditing workflow.

    Parameters:
        overwrite (bool): Whether to regenerate existing annual Parquet files.

    Returns:
        None: Saves annual datasets and audit reports and prints summaries.

    Notes:
        Qlib should be initialized with kernels=1 to limit resource usage.
        Annual datasets are processed sequentially to reduce peak memory usage.
    """
    from data.data_audit import run_data_audit
    from data.data_loader import build_daily_dataset, initialize_qlib

    initialize_qlib()

    print("\n[1/2] BUILDING DAILY STOCK PANEL", flush=True)

    summary = build_daily_dataset(overwrite=overwrite)

    print(summary.to_string(index=False))

    print("\n[2/2] RUNNING DATA QUALITY AUDIT", flush=True)

    audit = run_data_audit()

    print(audit.to_string(index=False))

    print("\nDATA PIPELINE COMPLETED", flush=True)
    print(f"Processed data: {DATA_OUTPUT_DIR}")
    print(f"Audit reports: {AUDIT_OUTPUT_DIR}")


def run_missing_diagnostics():
    """
    Analyze missing price observations across all cached annual datasets.

    Parameters:
        None.

    Returns:
        pd.DataFrame: Combined stock-level missing data diagnostics.

    Notes:
        Annual panels are loaded sequentially to reduce memory consumption.
        The resulting report is saved to the configured audit directory.
    """
    import pandas as pd
    from data.data_audit import diagnose_missing_patterns

    results = []

    for file in sorted(DATA_OUTPUT_DIR.glob("csi300_daily_*.parquet")):
        year = int(file.stem.split("_")[-1])
        df = pd.read_parquet(file)
        results.append(diagnose_missing_patterns(df, year))
        print(f"Missing diagnostics completed: {year}", flush=True)

    report = pd.concat(results, ignore_index=True)
    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report.to_csv(AUDIT_OUTPUT_DIR / "missing_patterns.csv", index=False)

    print("\nTOP 20 STOCK-YEAR MISSING PATTERNS")
    print(report.head(20).to_string(index=False))

    return report


def cmd_panels(args: argparse.Namespace) -> None:
    """Run the panel step in the selected mode.

    Notes:
        The default mode is "missing", matching the hard-coded RUN_MODE of the former
        run_data_pipeline.py; "full" builds the yearly panels with overwrite=False.
    """
    if args.mode == "full":
        run_complete_data_pipeline(overwrite=False)
    else:
        run_missing_diagnostics()


def fetch_qlib_prices(provider_uri: str, instruments: list[str], start: str, end: str):
    """Description: Extract all relevant stock prices and the shared market calendar. Parameters: provider_uri (str): Qlib binary directory; instruments (list[str]): historical members across the full period; start, end (str): inclusive extraction bounds. Returns: tuple[pd.DataFrame, pd.DatetimeIndex]: adjusted-price panel and market trading calendar. Notes: Queries stocks directly, including their post-removal trading dates; does not use index-filtered processed prices."""
    import pandas as pd
    import qlib
    from qlib.data import D
    from data.history_loader import fetch_qlib_features

    qlib.init(provider_uri=provider_uri, region='cn')
    calendar = pd.DatetimeIndex(D.calendar(start_time=start, end_time=end, freq='day'))
    prices = fetch_qlib_features(
        instruments, ['$open', '$close', '$factor'], start, end, batch_size=100
    )
    if prices.empty:
        raise ValueError('Qlib returned no prices for historical members')
    if not {'instrument', 'datetime', 'open', 'close', 'factor'}.issubset(prices.columns):
        raise ValueError('Unexpected Qlib feature fields')

    return prices, calendar


def audit_universe(membership, data_root: Path):
    """Description: Compare membership keys to existing yearly processed panel keys. Parameters: membership (pd.DataFrame): expanded signal universe; data_root (Path): existing research data directory. Returns: pd.DataFrame: per-year counts of mismatched instrument-date keys. Notes: Missing processed files are recorded, not treated as successful matches."""
    import pandas as pd

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


def cmd_labels(args: argparse.Namespace) -> None:
    """Description: Build all 1/5/10-day labels and integrity reports with fixed paths by default. Parameters: args (argparse.Namespace): qlib_dir, data_root, start, end. Returns: None; writes Parquet, CSV and JSON outputs. Notes: Does not certify index PIT publication timing, tradability, delisting returns, or independent adjustment accuracy."""
    import pandas as pd
    from labels.label_engine import LabelConfig, build_labels
    from labels.membership import expand_membership, load_membership_intervals
    from labels.price_audit import summarize_price_audit

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


def cmd_check_labels(args: argparse.Namespace) -> None:
    """Description: Validate historical membership, label integrity, price issues, and factor alignment. Parameters: args (argparse.Namespace): data_root, qlib_dir. Returns: None; writes CSV and JSON reports. Notes: Never overwrites input data; PIT, tradability and corporate-action accuracy remain unverified."""
    import pandas as pd
    from labels.label_validation import (
        alignment_diagnostics,
        label_diagnostics,
        membership_diagnostics,
        price_diagnostics,
    )
    from labels.membership import load_membership_intervals

    root = Path(args.data_root); qroot = Path(args.qlib_dir)
    source = root / 'reports' / 'label_engineering'
    out = root / 'reports' / 'label_validation'
    out.mkdir(parents=True, exist_ok=True)

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(qroot), region='cn')
    calendar = pd.DatetimeIndex(
        D.calendar(start_time='2015-01-01', end_time='2026-09-30', freq='day')
    )
    labels = pd.read_parquet(root / 'data' / 'labels' / 'forward_labels.parquet')
    intervals = load_membership_intervals(qroot / 'instruments' / 'csi300.txt')
    membership, missing = membership_diagnostics(intervals, root / 'data' / 'processed', calendar)

    membership.to_csv(out / 'membership_reconciliation.csv', index=False)
    missing.to_csv(out / 'membership_difference_details.csv', index=False)

    issues = pd.read_csv(source / 'price_audit.csv')
    candidates = pd.read_csv(source / 'factor_change_candidates.csv')
    price, thresholds, largest = price_diagnostics(issues, candidates)

    price.to_csv(out / 'price_issue_attribution.csv', index=False)
    thresholds.to_csv(out / 'factor_change_thresholds.csv', index=False)
    largest.to_csv(out / 'factor_change_largest_100.csv', index=False)

    integrity, status, examples = label_diagnostics(labels, calendar)

    integrity.to_csv(out / 'label_integrity.csv', index=False)
    status.to_csv(out / 'label_missingness.csv', index=False)
    examples.to_csv(out / 'label_integrity_examples.csv', index=False)

    alignment = alignment_diagnostics(labels, root / 'data' / 'factors_enhanced')
    alignment.to_csv(out / 'factor_label_alignment.csv', index=False)

    processed = sorted((root / 'data' / 'processed').glob('csi300_daily_*.parquet'))
    yearly = []

    for file in processed:
        frame = pd.read_parquet(file)
        idx = (
            frame.index.get_level_values('datetime')
            if 'datetime' in frame.index.names
            else pd.to_datetime(frame['datetime'])
        )

        dates = pd.to_datetime(idx)
        yearly.append(
            {
                'year': int(file.stem[-4:]),
                'all_rows': len(frame),
                'warmup_rows': int((dates < pd.Timestamp('2015-04-01')).sum()),
                'research_rows': int(
                    (
                        (dates >= pd.Timestamp('2015-04-01'))
                        & (dates <= pd.Timestamp('2026-09-30'))
                    ).sum()
                ),
            }
        )

    sample = pd.DataFrame(yearly); sample.to_csv(out / 'sample_reconciliation.csv', index=False)

    summary = {
        'label_rows': len(labels),
        'processed_total_rows': int(sample['all_rows'].sum()),
        'processed_warmup_rows': int(sample['warmup_rows'].sum()),
        'processed_research_rows': int(sample['research_rows'].sum()),
        'processed_minus_label_rows': int(sample['research_rows'].sum() - len(labels)),
        'membership_missing': int(membership['missing'].fillna(0).sum()),
        'membership_extra': int(membership['extra'].fillna(0).sum()),
        'label_integrity_violations': int(integrity['violations'].sum()),
        'factor_alignment_missing': int(
            alignment.get('labels_without_factors', pd.Series(dtype=float)).fillna(0).sum()
        ),
        'limitations': [
            'Factor-change candidate file may omit unchanged observations; gap-aware event reconstruction needs complete factor series',
            'Membership announcement-time PIT not established',
            'No tradability or delisting payoff verification',
            'Price adjustment not independently verified',
        ],
    }

    (out / 'validation_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('Reports:', out)


def build_parser() -> argparse.ArgumentParser:
    """Build the data-stage parser.

    Returns:
        argparse.ArgumentParser: Parser whose subcommands set ``func`` to their handler.
    """
    parser = argparse.ArgumentParser(description="CSI300 data panels and forward labels")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("panels", help="Yearly OHLCV panels or missing-price diagnostics")
    p.add_argument("--mode", choices=["missing", "full"], default="missing")
    p.set_defaults(func=cmd_panels)

    p = sub.add_parser("labels", help="Build 1/5/10-day forward labels")
    p.add_argument('--qlib-dir', default=DEFAULT_QLIB)
    p.add_argument('--data-root', default=DEFAULT_DATA)
    p.add_argument('--start', default='2015-01-01')
    p.add_argument('--end', default='2026-09-30')
    p.set_defaults(func=cmd_labels)

    p = sub.add_parser("check-labels", help="Read-only label quality diagnostics")
    p.add_argument('--data-root', default=str(DATA_ROOT))
    p.add_argument('--qlib-dir', default=str(QLIB_DATA_DIR))
    p.set_defaults(func=cmd_check_labels)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and dispatch to the selected subcommand."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
