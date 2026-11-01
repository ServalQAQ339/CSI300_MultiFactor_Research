"""Run read-only CSI300 label quality diagnostics against local research data."""
import argparse
import json
from pathlib import Path
import pandas as pd
from label_validation.validators import (
    read_membership,
    membership_diagnostics,
    price_diagnostics,
    label_diagnostics,
    alignment_diagnostics,
)


def main() -> None:
    """Description: Validate historical membership, label integrity, price issues, and factor alignment. Parameters: None; CLI accepts --data-root and --qlib-dir. Returns: None; writes CSV and JSON reports. Notes: Never overwrites input data; PIT, tradability and corporate-action accuracy remain unverified."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=r'D:\QuantResearch\CSI300_MultiFactor_Research')
    parser.add_argument('--qlib-dir', default=r'D:\QuantResearch\qlib_data\cn_data\qlib_bin')
    args = parser.parse_args(); root = Path(args.data_root); qroot = Path(args.qlib_dir)
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
    intervals = read_membership(qroot / 'instruments' / 'csi300.txt')
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


if __name__ == '__main__':
    main()
