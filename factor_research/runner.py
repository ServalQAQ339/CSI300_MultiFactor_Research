"""End-to-end audited Raw x 5D factor research runner."""
from pathlib import Path
import numpy as np
import pandas as pd
from .alignment import load_inputs, prepare_panel
from .config import FACTOR_NAMES, SEGMENTS
from .ic_analysis import daily_ic
from .quantile_analysis import quantile_returns
from .statistical_tests import hac_summary
from .reporting import write_audit


def run(root: Path, output: Path, min_n: int = 30):
    """Description: Run full cross-sectional factor research on validated input files.

    Parameters:
        root (Path): Root of D-drive CSI300 data.
        output (Path): Destination directory for reports.
        min_n (int): Minimum paired observations per daily cross-section.
    Returns:
        dict: Alignment audit counts and research limitations.
    Notes:
        Reports are descriptive; 5D forward returns are not executable portfolio PnL.
        No input file is changed. Existing output filenames may be replaced on rerun.
    """
    factors, labels = load_inputs(root)
    panel, audit = prepare_panel(factors, labels)
    output.mkdir(parents=True, exist_ok=True)
    ic = daily_ic(panel, FACTOR_NAMES, 'research_label_5d', 'signal_date', min_n)
    ic.to_parquet(output / 'daily_ic_raw_5d.parquet', index=False)
    summary, coverage, yearly = [], [], []
    for factor in FACTOR_NAMES:
        x = pd.to_numeric(panel[factor], errors='coerce').replace([np.inf, -np.inf], np.nan)
        y = panel['research_label_5d']; paired = x.notna() & y.notna()
        for split, (start, end) in SEGMENTS.items():
            mask = panel.split.eq(split); n = int(mask.sum()); valid_y = int((mask & y.notna()).sum())
            coverage.append({'factor': factor, 'split': split, 'universe_rows': n, 'factor_valid': int((mask & x.notna()).sum()), 'label_valid': valid_y, 'paired_valid': int((mask & paired).sum()), 'universe_coverage': float((mask & x.notna()).sum()/n) if n else np.nan, 'paired_coverage': float((mask & paired).sum()/valid_y) if valid_y else np.nan})
            part = ic.loc[(ic.factor == factor) & ic.signal_date.between(start, end)].sort_values('signal_date')
            for metric in ('ic', 'rank_ic'):
                for lag in (0, 4, 9, 19): summary.append({'factor': factor, 'split': split, 'metric': metric, 'hac_lag': lag, 'positive_ratio': float((part[metric].dropna()>0).mean()) if part[metric].notna().any() else np.nan, **hac_summary(part, metric, lag)})
        q = quantile_returns(panel, factor, 'research_label_5d', 'signal_date', 'instrument', 5, min_n)
        q.to_parquet(output / f'quantiles_{factor}_raw_5d.parquet', index=False)
    pd.DataFrame(summary).to_csv(output / 'factor_ic_summary.csv', index=False)
    pd.DataFrame(coverage).to_csv(output / 'factor_coverage.csv', index=False)
    yearly = ic.assign(year=ic.signal_date.dt.year).groupby(['factor', 'year'], as_index=False).agg(ic_mean=('ic','mean'), rank_ic_mean=('rank_ic','mean'), valid_days=('rank_ic','count'))
    yearly.to_csv(output / 'yearly_stability.csv', index=False)
    write_audit(output / 'alignment_audit.json', audit)
    return audit
