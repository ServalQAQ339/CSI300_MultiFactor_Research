"""Run non-destructive v1.3 robustness and redundancy diagnostics on v1.2 outputs."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from factor_research.config import DATA_ROOT, SEGMENTS, FACTOR_NAMES
from factor_research.diagnostics import (
    ic_diagnostics,
    bh_fdr,
    daily_factor_correlations,
    quantile_diagnostics,
)


def main() -> None:
    """Description: Audit v1.2 daily IC, quantiles, and raw factor redundancy.

    Parameters:
        None: Optional command-line arguments override default D-drive locations.
    Returns:
        None: Writes versioned v1.3 CSV diagnostics without modifying v1.2 files.
    Notes:
        FDR is applied separately to the 27 factor tests per split and metric at HAC lag 4.
        Non-overlap phases are based on the global signal-date calendar.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DATA_ROOT)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--min-n', type=int, default=30)
    args = parser.parse_args()
    source = args.input or args.root / 'reports/factor_research/raw_5d_v12'
    dest = args.output or args.root / 'reports/factor_research/raw_5d_v13_diagnostics'
    dest.mkdir(parents=True, exist_ok=True)
    ic = pd.read_parquet(source / 'daily_ic_raw_5d.parquet')
    acf, phases = ic_diagnostics(ic, SEGMENTS)
    acf.to_csv(dest / 'ic_autocorrelation.csv', index=False)
    phases.to_csv(dest / 'nonoverlap_phase_ic.csv', index=False)
    summary = pd.read_csv(source / 'factor_ic_summary.csv')
    print('Summary columns:', summary.columns.tolist())
    if 'hac_t' not in summary.columns: raise ValueError('Expected hac_t in v1.2 summary')
    from math import erfc, sqrt
    selected = summary.loc[summary.hac_lag.eq(4)].copy()
    selected['p_normal_approx'] = selected.hac_t.map(
        lambda x: erfc(abs(float(x)) / sqrt(2)) if pd.notna(x) else np.nan
    )
    selected['fdr_q'] = selected.groupby(['split', 'metric'])['p_normal_approx'].transform(bh_fdr)
    selected.to_csv(dest / 'hac_lag4_fdr.csv', index=False)
    qrows = []
    for factor in FACTOR_NAMES:
        path = source / f'quantiles_{factor}_raw_5d.parquet'
        if not path.exists(): raise FileNotFoundError(path)
        qrows.append(quantile_diagnostics(pd.read_parquet(path), factor, SEGMENTS))
    pd.concat(qrows, ignore_index=True).to_csv(dest / 'quantile_monotonicity.csv', index=False)
    factor_paths = sorted(
        (args.root / 'data/factors_preprocessed/raw').glob('csi300_factors_*.parquet')
    )
    if len(factor_paths) != 12:
        raise ValueError(f'Expected 12 annual factor files, found {len(factor_paths)}')
    from factor_research.alignment import load_inputs
    factors, _ = load_inputs(args.root)
    factors = factors.loc[pd.to_datetime(factors.signal_date).between('2015-04-01', '2026-09-30')]
    for split, (start, end) in SEGMENTS.items():
        subset = factors.loc[pd.to_datetime(factors.signal_date).between(start, end)]
        matrix, n_days = daily_factor_correlations(subset, FACTOR_NAMES, args.min_n)
        matrix.to_csv(dest / f'factor_spearman_{split}.csv')
        n_days.to_csv(dest / f'factor_pair_days_{split}.csv')
    print(f'Completed v1.3 diagnostics: {dest}')


if __name__ == '__main__':
    main()
