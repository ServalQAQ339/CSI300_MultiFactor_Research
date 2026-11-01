"""Research-stage entrypoint: Original27 Rank IC, candidate expansion and train-only selection.

Usage:
    python run_research.py ic [--data-root DIR] [--output DIR] [--min-n N]
    python run_research.py diagnostics [--data-root DIR] [--input DIR] [--output DIR] [--min-n N]
    python run_research.py generate-candidates --output DIR [--provider DIR] [--start D] [--end D]
    python run_research.py analyze-candidates --output DIR [--data-dir DIR] [--candidates FILE]
    python run_research.py select --data-dir DIR --candidates FILE --registry FILE --output DIR

Every heavy dependency is imported inside the subcommand handlers, so importing this module
never touches data.
"""

import argparse
from pathlib import Path

from config import DATA_DIR, QLIB_DATA_DIR
from factor_research.config import DATA_ROOT


def cmd_ic(args: argparse.Namespace) -> None:
    """Description: Execute Raw x 5D factor research with project defaults and optional overrides.

    Parameters:
        args (argparse.Namespace): data_root, output, min_n.
    Returns:
        None: Prints audit and saves reports under the data root.
    Notes:
        Does not modify the existing upstream factor or label data.
    """
    from factor_research.runner import run

    audit = run(args.data_root, args.output, args.min_n)
    print(f'Completed: {audit}; output={args.output}')


def cmd_diagnostics(args: argparse.Namespace) -> None:
    """Description: Audit v1.2 daily IC, quantiles, and raw factor redundancy.

    Parameters:
        args (argparse.Namespace): data_root, input, output, min_n.
    Returns:
        None: Writes versioned v1.3 CSV diagnostics without modifying v1.2 files.
    Notes:
        FDR is applied separately to the 27 factor tests per split and metric at HAC lag 4.
        Non-overlap phases are based on the global signal-date calendar.
    """
    import numpy as np
    import pandas as pd
    from factor_research.config import FACTOR_NAMES, SEGMENTS
    from factor_research.diagnostics import (
        bh_fdr,
        daily_factor_correlations,
        ic_diagnostics,
        quantile_diagnostics,
    )

    source = args.input or args.data_root / 'reports/factor_research/raw_5d_v12'
    dest = args.output or args.data_root / 'reports/factor_research/raw_5d_v13_diagnostics'
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
        (args.data_root / 'data/factors_preprocessed/raw').glob('csi300_factors_*.parquet')
    )
    if len(factor_paths) != 12:
        raise ValueError(f'Expected 12 annual factor files, found {len(factor_paths)}')

    from factor_research.alignment import load_inputs

    factors, _ = load_inputs(args.data_root)
    factors = factors.loc[pd.to_datetime(factors.signal_date).between('2015-04-01', '2026-09-30')]

    for split, (start, end) in SEGMENTS.items():
        subset = factors.loc[pd.to_datetime(factors.signal_date).between(start, end)]
        matrix, n_days = daily_factor_correlations(subset, FACTOR_NAMES, args.min_n)
        matrix.to_csv(dest / f'factor_spearman_{split}.csv')
        n_days.to_csv(dest / f'factor_pair_days_{split}.csv')

    print(f'Completed v1.3 diagnostics: {dest}')


def cmd_generate_candidates(args: argparse.Namespace) -> None:
    """Generate the additive candidate panel, its registry and coverage.

    Notes:
        An existing candidates.parquet is never overwritten. Former
        ``run_factor_expansion.py generate``.
    """
    import pandas as pd
    from factors.expansion.data import load_bars
    from factors.expansion.extra_factors import compute_extra_factors
    from factors.expansion.inspired_factors import compute_batch_factors
    from factors.expansion.qlib_factors import compute_factors
    from factors.expansion.registry import build_registry

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    bars = load_bars(args.provider, args.start, args.end)
    blocks = [compute_factors(bars), compute_batch_factors(bars), compute_extra_factors(bars)]
    candidates = pd.concat(blocks, axis=1)
    if not candidates.columns.is_unique:
        raise ValueError("Duplicate feature names")
    destination = output / "candidates.parquet"
    if destination.exists():
        raise FileExistsError(destination)
    candidates.to_parquet(destination)
    registry = build_registry(candidates.columns.tolist())
    registry.to_csv(output / "candidate_registry.csv", index=False)
    candidates.notna().mean().rename("coverage").to_csv(output / "candidate_coverage.csv")
    print(f"Generated {len(candidates):,} rows, {len(candidates.columns)} candidate columns")


def cmd_analyze_candidates(args: argparse.Namespace) -> None:
    """Run train/validation Rank IC, redundancy, conditional IC and nested Ridge analysis.

    Notes:
        Former ``run_factor_expansion.py analyze``. Expensive conditional analysis is
        restricted to the first nine Qlib candidates.
    """
    import pandas as pd
    from factors.expansion.data import TRAIN, VALIDATION, load_research_panel
    from factors.expansion.incremental import conditional_rank_ic, nested_ridge_ic
    from factors.expansion.registry import build_registry
    from factors.expansion.statistics import daily_rank_ic, hac_summary, redundancy

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    candidate_path = Path(args.candidates) if args.candidates else output / "candidates.parquet"
    candidates = pd.read_parquet(candidate_path)
    panel, baseline = load_research_panel(args.data_dir, candidates)
    registry = build_registry(candidates.columns.tolist())
    selected = registry.loc[~registry["excluded"], "factor"].tolist()
    for split_name, bounds in (("train", TRAIN), ("validation", VALIDATION)):
        subset = panel.loc[(panel.index.get_level_values("datetime") >= bounds[0]) & (panel.index.get_level_values("datetime") <= bounds[1])]
        print(f"{split_name}: {len(subset):,} valid rows")
        daily = daily_rank_ic(subset, baseline + selected)
        daily.to_csv(output / f"{split_name}_daily_rank_ic.csv")
        hac_summary(daily).to_csv(output / f"{split_name}_rank_ic_summary.csv", index=False)
        if split_name == "train":
            redundancy(subset, baseline, selected).to_csv(output / "train_redundancy.csv", index=False)
            train = subset
        else:
            validation = subset

    core = [name for name in selected if name.startswith(("CORR", "VSTD", "RSV"))]
    conditional_rank_ic(train, baseline, core).to_csv(output / "train_conditional_rank_ic.csv")
    conditional_rank_ic(validation, baseline, core).to_csv(output / "validation_conditional_rank_ic.csv")
    daily, summary = nested_ridge_ic(train, validation, baseline, selected)
    daily.to_csv(output / "validation_nested_daily_ic.csv")
    summary.to_csv(output / "validation_nested_summary.csv", index=False)
    registry.to_csv(output / "candidate_registry.csv", index=False)
    print("Completed unified factor analysis")


def cmd_select(args: argparse.Namespace) -> None:
    """Run train-only factor selection and the matched-sample Ridge comparison."""
    from factor_research.selection import run_selection

    run_selection(args.data_dir, args.candidates, args.registry, args.output,
                  args.max_features, args.min_coverage, args.max_corr, args.alpha)


def _add_expansion_args(parser: argparse.ArgumentParser) -> None:
    """Add the option set shared by both former ``run_factor_expansion.py`` modes.

    Notes:
        Both modes accepted every option; options a mode does not use are still accepted and
        ignored, as before.
    """
    parser.add_argument("--provider", default=str(QLIB_DATA_DIR))
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--start", default="2014-09-01")
    parser.add_argument("--end", default="2026-09-30")


def build_parser() -> argparse.ArgumentParser:
    """Build the research-stage parser.

    Returns:
        argparse.ArgumentParser: Parser whose subcommands set ``func`` to their handler.
    """
    parser = argparse.ArgumentParser(description="CSI300 factor research and selection")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ic", help="Original27 Raw x 5D Rank IC research")
    p.add_argument('--data-root', type=Path, default=DATA_ROOT)
    p.add_argument(
        '--output', type=Path, default=DATA_ROOT / 'reports/factor_research/raw_5d_v12'
    )
    p.add_argument('--min-n', type=int, default=30)
    p.set_defaults(func=cmd_ic)

    p = sub.add_parser("diagnostics", help="IC robustness and redundancy diagnostics")
    p.add_argument('--data-root', type=Path, default=DATA_ROOT)
    p.add_argument('--input', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--min-n', type=int, default=30)
    p.set_defaults(func=cmd_diagnostics)

    p = sub.add_parser("generate-candidates", help="Generate the candidate factor panel")
    _add_expansion_args(p)
    p.set_defaults(func=cmd_generate_candidates)

    p = sub.add_parser("analyze-candidates", help="Analyze the candidate factor panel")
    _add_expansion_args(p)
    p.set_defaults(func=cmd_analyze_candidates)

    p = sub.add_parser("select", help="Train-only factor selection")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--registry", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-features", type=int, default=20)
    p.add_argument("--min-coverage", type=float, default=0.80)
    p.add_argument("--max-corr", type=float, default=0.85)
    p.add_argument("--alpha", type=float, default=100.0)
    p.set_defaults(func=cmd_select)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and dispatch to the selected subcommand."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
