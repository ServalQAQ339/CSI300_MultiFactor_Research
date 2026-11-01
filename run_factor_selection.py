"""Standalone v1.5 train-only candidate screening and validation diagnostics."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from factor_research.alignment import load_inputs, prepare_panel
from factor_research.config import DATA_ROOT, FACTOR_NAMES
from factor_research.core import hac_mean
from factor_research.selection import conditional_rank_ic, train_redundancy_groups


def main() -> None:
    """Description: Screen factors using train data and report conditional Rank IC in train and validation.

    Parameters:
        None: Command-line options control paths and research thresholds.
    Returns:
        None: Writes CSV reports without altering source data.
    Notes:
        Does not access Test samples for selection; screening uses historical descriptive scores, not fitted portfolio returns.
    """
    p = argparse.ArgumentParser(); p.add_argument('--data-root', type=Path, default=DATA_ROOT); p.add_argument('--output', type=Path, default=DATA_ROOT / 'reports/factor_research/raw_5d_v15_selection'); p.add_argument('--corr-threshold', type=float, default=0.8); p.add_argument('--min-n', type=int, default=30); p.add_argument('--max-candidates', type=int, default=8); args = p.parse_args()
    factors, labels = load_inputs(args.data_root); panel, audit = prepare_panel(factors, labels)
    train = panel.loc[panel.split.eq('train') & ~panel.purged_for_split]; validation = panel.loc[panel.split.eq('validation') & ~panel.purged_for_split]
    summary_path = args.data_root / 'reports/factor_research/raw_5d_v12/factor_ic_summary.csv'
    if not summary_path.exists(): raise FileNotFoundError(f'Missing prerequisite report: {summary_path}')
    summary = pd.read_csv(summary_path)
    rank = summary.loc[summary.metric.eq('rank_ic') & summary.split.eq('train') & summary.hac_lag.eq(4)].copy()
    if len(rank) != len(FACTOR_NAMES) or rank.factor.duplicated().any(): raise ValueError('Unexpected train IC summary')
    rank['abs_train_rank_ic'] = rank['mean'].abs(); rank = rank.sort_values(['abs_train_rank_ic','factor'], ascending=[False,True])
    corr_path = args.data_root / 'reports/factor_research/raw_5d_v13_diagnostics/factor_spearman_train.csv'
    if not corr_path.exists(): raise FileNotFoundError(f'Missing train factor correlation matrix: {corr_path}')
    corr = pd.read_csv(corr_path, index_col=0)
    if set(corr.index) != set(FACTOR_NAMES) or set(corr.columns) != set(FACTOR_NAMES): raise ValueError('Unexpected correlation matrix factors')
    groups = train_redundancy_groups(corr.loc[FACTOR_NAMES, FACTOR_NAMES], args.corr_threshold)
    selected = []
    for factor in rank.factor:
        if all(pd.isna(corr.loc[factor, other]) or abs(corr.loc[factor, other]) < args.corr_threshold for other in selected): selected.append(factor)
        if len(selected) >= args.max_candidates: break
    if not selected: raise ValueError('No selected factors')
    args.output.mkdir(parents=True, exist_ok=True)
    groups.to_csv(args.output / 'train_redundancy_groups.csv', index=False); rank.to_csv(args.output / 'train_univariate_screen.csv', index=False)
    pd.DataFrame({'selection_order': range(1,len(selected)+1), 'factor': selected}).to_csv(args.output / 'train_candidate_shortlist.csv', index=False)
    rows = []
    for i, factor in enumerate(selected):
        controls = selected[:i]
        for split, data in [('train',train),('validation',validation)]:
            daily = conditional_rank_ic(data, factor, controls, args.min_n)
            stats = hac_mean(daily.conditional_rank_ic, 4)
            rows.append({'factor':factor,'selection_order':i+1,'split':split,'controls':','.join(controls),'common_sample_method':'complete_case','statistic':'candidate_residual_vs_label_rank','horizon':'5d','hac_lag':4,**stats})
    pd.DataFrame(rows).to_csv(args.output / 'conditional_rank_ic_diagnostics.csv', index=False)
    print(f'Completed v1.5: {len(selected)} train-selected candidates; test not accessed for screening; output={args.output}')
    print(f'Alignment reference: {audit["factor_matched_rows"]} factor-matched rows')


if __name__ == '__main__': main()
