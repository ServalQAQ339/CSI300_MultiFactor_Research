"""Command-line entry point for independent Raw x 5D factor research."""
import argparse
from pathlib import Path
import pandas as pd
from factor_research.core import align, daily_ic, hac_mean, quantile_returns


def main() -> None:
    """Description: Load validated panel files and write reproducible IC and quantile reports.

    Parameters:
        None: Inputs are provided through command-line flags.
    Returns:
        None: Writes research CSV outputs and audit summaries.
    Notes:
        Input key and label column names must be explicitly checked against upstream schemas.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--factors', required=True, help='Raw factors parquet file')
    p.add_argument('--labels', required=True, help='Forward labels parquet file')
    p.add_argument('--label-col', required=True, help='Verified 5D label column')
    p.add_argument('--date-col', default='date'); p.add_argument('--asset-col', default='instrument')
    p.add_argument('--output', required=True); p.add_argument('--min-n', type=int, default=30)
    args = p.parse_args()
    factors = pd.read_parquet(args.factors); labels = pd.read_parquet(args.labels)
    keys = [args.date_col, args.asset_col]
    if any(k not in factors.columns or k not in labels.columns for k in keys):
        raise ValueError('Required key columns missing: inspect upstream parquet schemas')
    factor_cols = [c for c in factors.columns if c not in keys and pd.api.types.is_numeric_dtype(factors[c])]
    if not factor_cols:
        raise ValueError('No numeric factor columns found')
    merged, audit = align(factors, labels, factor_cols, args.label_col, *keys)
    merged[args.date_col] = pd.to_datetime(merged[args.date_col])
    merged = merged.loc[merged[args.date_col].between('2015-04-01', '2026-09-30')]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ic = daily_ic(merged, factor_cols, args.label_col, args.date_col, args.min_n)
    ic.to_csv(out / 'daily_ic.csv', index=False)
    rows = []
    for factor in factor_cols:
        d = ic.loc[ic.factor == factor]
        for segment, start, end in [('train', '2015-04-01', '2020-12-31'), ('validation', '2021-01-01', '2023-12-31'), ('test', '2024-01-01', '2026-09-30')]:
            part = d.loc[d[args.date_col].between(start, end)]
            for metric in ['ic', 'rank_ic']:
                rows.append({'factor': factor, 'segment': segment, 'metric': metric, **hac_mean(part[metric], 4)})
        quantile_returns(merged, factor, args.label_col, args.date_col, args.asset_col, 5, args.min_n).to_csv(out / f'quantiles_{factor}.csv', index=False)
    pd.DataFrame(rows).to_csv(out / 'ic_summary.csv', index=False)
    pd.Series({**audit, 'factors_detected': len(factor_cols), 'aligned_research_rows': len(merged)}).to_csv(out / 'alignment_audit.csv')
    print(f'Completed: {len(factor_cols)} factors; outputs in {out}')


if __name__ == '__main__':
    main()
