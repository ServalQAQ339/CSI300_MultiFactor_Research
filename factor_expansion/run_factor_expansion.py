"""Unified command-line entrypoint for factor generation and research."""

import argparse
from pathlib import Path

import pandas as pd

from ..data import TRAIN, VALIDATION, load_bars, load_research_panel
from .extra_factors import compute_extra_factors
from .incremental import conditional_rank_ic, nested_ridge_ic
from .inspired_factors import compute_batch_factors
from .qlib_factors import compute_factors
from .registry import build_registry
from .statistics import daily_rank_ic, hac_summary, redundancy


def main() -> None:
    """Run the unified additive expansion and statistical research pipeline.

    Description:
        Generate candidates or analyze a previously generated candidate panel.
    Parameters:
        None: Arguments are parsed from the command line.
    Returns:
        None: Writes new Parquet and CSV files under explicit output paths.
    Notes:
        Existing output files are never silently overwritten.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["generate", "analyze"])
    parser.add_argument("--provider", default=r"D:\QuantResearch\qlib_data\cn_data\qlib_bin")
    parser.add_argument("--data-root", default=r"D:\QuantResearch\CSI300_MultiFactor_Research\data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--start", default="2014-09-01")
    parser.add_argument("--end", default="2026-09-30")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.mode == "generate":
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
        return

    candidate_path = Path(args.candidates) if args.candidates else output / "candidates.parquet"
    candidates = pd.read_parquet(candidate_path)
    panel, baseline = load_research_panel(args.data_root, candidates)
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

    # Expensive conditional analysis is restricted to the first nine Qlib candidates.
    core = [name for name in selected if name.startswith(("CORR", "VSTD", "RSV"))]
    conditional_rank_ic(train, baseline, core).to_csv(output / "train_conditional_rank_ic.csv")
    conditional_rank_ic(validation, baseline, core).to_csv(output / "validation_conditional_rank_ic.csv")
    daily, summary = nested_ridge_ic(train, validation, baseline, selected)
    daily.to_csv(output / "validation_nested_daily_ic.csv")
    summary.to_csv(output / "validation_nested_summary.csv", index=False)
    registry.to_csv(output / "candidate_registry.csv", index=False)
    print("Completed unified factor analysis")


if __name__ == "__main__":
    main()
