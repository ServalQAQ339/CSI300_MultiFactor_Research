"""Run read-only independent IC/quintile consistency audit on local CSI300 research data."""
import argparse
import json
from pathlib import Path
import pandas as pd
from factor_research.config import DATA_ROOT
from factor_research.alignment import load_inputs, prepare_panel
from factor_research.consistency_audit import audit_factor, compare_saved


def main() -> None:
    """Description: Recompute daily diagnostics and compare to saved v1.2 reports.

    Parameters:
        None: CLI flags select paths, factors, and tolerance.
    Returns:
        None: Writes independent audit CSV and JSON outputs to a new directory.
    Notes:
        Does not modify any original factor, label, or v1.2 research outputs.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=DATA_ROOT)
    p.add_argument('--v12', type=Path, default=DATA_ROOT/'reports/factor_research/raw_5d_v12')
    p.add_argument(
        '--output', type=Path, default=DATA_ROOT / 'reports/factor_research/raw_5d_v14_consistency'
    )
    p.add_argument(
        '--factors',
        nargs='+',
        default=[
            'VOLATILITY_5',
            'VOLATILITY_10',
            'VOLATILITY_20',
            'VOLATILITY_60',
            'MOM_20',
            'OPEN_CLOSE_RATIO',
        ],
    )

    p.add_argument('--min-n', type=int, default=30)
    p.add_argument('--tolerance', type=float, default=1e-10)
    args = p.parse_args()
    if args.output.resolve() == args.v12.resolve():
        raise ValueError('Output must differ from v1.2 report directory')
    factors, labels = load_inputs(args.data_root); panel, alignment = prepare_panel(factors, labels)
    saved_ic = pd.read_parquet(args.v12/'daily_ic_raw_5d.parquet')
    saved_ic['signal_date'] = pd.to_datetime(saved_ic['signal_date'])
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, comparisons, details = [], [], []

    for factor in args.factors:
        if factor not in factors.columns: raise ValueError(f'Unknown factor: {factor}')
        daily, summary = audit_factor(panel, factor, args.min_n)
        saved_q = pd.read_parquet(args.v12/f'quantiles_{factor}_raw_5d.parquet')
        saved_q['signal_date'] = pd.to_datetime(saved_q['signal_date'])
        comp = compare_saved(
            daily, saved_ic.loc[saved_ic.factor.eq(factor)], saved_q, args.tolerance
        )
        summaries.append(summary)
        comparisons.append(comp.assign(factor=factor))
        details.append(daily)
        print(f'{factor}: audited={len(comp)}, failed={int((~comp["pass"]).sum())}', flush=True)

    summary = pd.concat(summaries, ignore_index=True)
    comparison = pd.concat(comparisons, ignore_index=True)
    detail = pd.concat(details, ignore_index=True)

    summary.to_csv(args.output/'split_consistency_summary.csv', index=False)
    comparison.to_csv(args.output/'daily_numeric_comparison.csv', index=False)
    detail.to_parquet(args.output/'independent_daily_diagnostics.parquet', index=False)

    report = {
        'factors': args.factors,
        'days_compared': len(comparison),
        'failed_daily_comparisons': int((~comparison['pass']).sum()),
        'max_rank_ic_abs_diff': float(comparison.rank_ic_abs_diff.max()),
        'max_spread_abs_diff': float(comparison.spread_abs_diff.max()),
        'alignment': alignment,
        'audit_scope': 'numerical consistency, not PIT or execution certification',
    }

    (args.output / 'consistency_audit.json').write_text(
        json.dumps(report, indent=2, default=str), encoding='utf-8'
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == '__main__': main()
