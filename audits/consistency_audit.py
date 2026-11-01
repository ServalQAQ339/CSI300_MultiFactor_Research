"""Independent same-sample IC and quintile-return consistency audit."""
from __future__ import annotations
import numpy as np
import pandas as pd
from factor_research.core import hac_mean
from factor_research.config import SEGMENTS


def audit_factor(
    panel: pd.DataFrame, factor: str, min_n: int = 30
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Description: Independently recompute daily rank IC and quintiles on identical paired observations.

    Parameters:
        panel (pd.DataFrame): Label-left joined panel containing signal_date, instrument, research_label_5d and the factor.
        factor (str): Numeric factor column to audit.
        min_n (int): Minimum paired observations for a valid daily cross-section.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Daily diagnostics and split-level descriptive/HAC summaries.
    Notes:
        Equal-count quintiles use factor/instrument stable sorting. Raw returns remain unchanged; trimmed means are diagnostic only.
        Forward five-day returns are overlapping labels, not daily executable portfolio PnL.
    """
    if min_n < 5: raise ValueError('min_n must be >= 5')
    required = {'signal_date', 'instrument', 'research_label_5d', factor}
    if not required.issubset(panel.columns):
        raise ValueError(f'Missing columns: {required-set(panel.columns)}')
    rows = []

    for date, day in panel.groupby('signal_date', sort=True):
        z = day[['instrument', factor, 'research_label_5d']].copy()
        for col in (factor, 'research_label_5d'):
            z[col] = pd.to_numeric(z[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        z = z.dropna(subset=[factor, 'research_label_5d'])
        if len(z) < min_n or z[factor].nunique() < 2 or z['research_label_5d'].nunique() < 2:
            continue
        rank_ic = (
            z[factor].rank(method='average').corr(z['research_label_5d'].rank(method='average'))
        )
        z = z.sort_values([factor, 'instrument'], kind='mergesort').reset_index(drop=True)
        z['quantile'] = (np.floor(np.arange(len(z)) * 5 / len(z)).astype(int) + 1)
        means = z.groupby('quantile')['research_label_5d'].mean()
        medians = z.groupby('quantile')['research_label_5d'].median()
        low = z.loc[z['quantile'].eq(1), 'research_label_5d']
        high = z.loc[z['quantile'].eq(5), 'research_label_5d']

        def trim_mean(s: pd.Series) -> float:
            v = np.sort(s.to_numpy(dtype=float))
            k = int(np.floor(0.05 * len(v)))
            return float(v[k : len(v) - k].mean()) if len(v) > 2 * k else np.nan

        rows.append(
            {
                'signal_date': date,
                'factor': factor,
                'n_paired': len(z),
                'rank_ic_independent': rank_ic,
                'q1': means.loc[1],
                'q2': means.loc[2],
                'q3': means.loc[3],
                'q4': means.loc[4],
                'q5': means.loc[5],
                'spread_mean': means.loc[5] - means.loc[1],
                'spread_median': medians.loc[5] - medians.loc[1],
                'spread_trimmed_5pct': trim_mean(high) - trim_mean(low),
                'q1_max_return': low.max(),
                'q5_max_return': high.max(),
                'q1_min_return': low.min(),
                'q5_min_return': high.min(),
            }
        )

    daily = pd.DataFrame(rows)
    summary = []

    for split, (start, end) in SEGMENTS.items():
        d = daily.loc[daily.signal_date.between(start, end)] if not daily.empty else daily
        if d.empty: continue
        s = d.spread_mean; r = d.rank_ic_independent
        summary.append(
            {
                'factor': factor,
                'split': split,
                'n_days': len(d),
                'rank_ic_mean': r.mean(),
                'spread_mean': s.mean(),
                'spread_median_over_dates': s.median(),
                'spread_positive_ratio': (s > 0).mean(),
                'mean_daily_median_spread': d.spread_median.mean(),
                'mean_daily_trimmed_spread': d.spread_trimmed_5pct.mean(),
                'sign_disagreement': bool(np.sign(r.mean()) != np.sign(s.mean())),
                **{f'spread_{k}': v for k, v in hac_mean(s, 4).items()},
            }
        )

    return daily, pd.DataFrame(summary)


def compare_saved(
    daily: pd.DataFrame,
    original_ic: pd.DataFrame,
    original_quantiles: pd.DataFrame,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Description: Compare independently recomputed daily metrics against saved v1.2 results.

    Parameters:
        daily (pd.DataFrame): Independent daily audit metrics for one factor.
        original_ic (pd.DataFrame): Saved daily IC output, filtered to one factor.
        original_quantiles (pd.DataFrame): Saved daily quantile output for that factor.
        tolerance (float): Maximum acceptable absolute floating-point difference.
    Returns:
        pd.DataFrame: Per-date differences, mismatches and sample-count comparisons.
    Notes:
        A missing saved date is an audit failure. Comparison is performed on daily outputs, not just aggregated means.
    """
    i = original_ic[['signal_date', 'rank_ic', 'n_paired']].rename(
        columns={'rank_ic': 'saved_rank_ic', 'n_paired': 'saved_ic_n'}
    )
    q = original_quantiles[['signal_date', 'long_short_forward_spread', 'n_paired']].rename(
        columns={'long_short_forward_spread': 'saved_spread', 'n_paired': 'saved_quantile_n'}
    )
    m = daily.merge(i, on='signal_date', how='left', validate='one_to_one').merge(
        q, on='signal_date', how='left', validate='one_to_one'
    )
    m['rank_ic_abs_diff'] = (m.rank_ic_independent - m.saved_rank_ic).abs()
    m['spread_abs_diff'] = (m.spread_mean - m.saved_spread).abs()
    m['pass'] = (
        m[['saved_rank_ic', 'saved_spread', 'saved_ic_n', 'saved_quantile_n']].notna().all(axis=1)
        & m.rank_ic_abs_diff.le(tolerance)
        & m.spread_abs_diff.le(tolerance)
        & m.n_paired.eq(m.saved_ic_n)
        & m.n_paired.eq(m.saved_quantile_n)
    )

    return m


def build_parser():
    """Build the CLI parser of ``python -m audits.consistency_audit``.

    Returns:
        argparse.ArgumentParser: Parser with data, v1.2 report and output paths, factors and
        numerical tolerance.
    """
    import argparse
    from pathlib import Path
    from factor_research.config import DATA_ROOT

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
    return p


def main(argv: list[str] | None = None) -> None:
    """Description: Recompute daily diagnostics and compare to saved v1.2 reports.

    Parameters:
        argv (list[str] | None): CLI arguments; None reads ``sys.argv``.
    Returns:
        None: Writes independent audit CSV and JSON outputs to a new directory.
    Notes:
        Former ``run_factor_consistency_audit.py``. Does not modify any original factor,
        label, or v1.2 research outputs.
    """
    import json
    from factor_research.alignment import load_inputs, prepare_panel

    args = build_parser().parse_args(argv)
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


if __name__ == '__main__':
    main()
