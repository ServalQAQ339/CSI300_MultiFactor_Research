"""Run a reproducible, baseline-aligned Alpha158 diagnostic report."""

import argparse
from pathlib import Path

import pandas as pd

from incremental_alpha_research.analysis import CANDIDATES, align_panels, daily_rank_ic, factor_correlations, summarize_ic


PERIODS = {"train": ("2015-04-01", "2020-12-31"), "validation": ("2021-01-01", "2023-12-31")}


def main() -> None:
    """Produce non-destructive train and validation reports.

    Description:
        Read baseline annual files, expansion features, and existing labels.
    Parameters:
        None: Command-line arguments define data and report roots.
    Returns:
        None: Writes CSV reports to a new output directory.
    Notes:
        The previously inspected test period is excluded from all reports.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.data_root)
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty report directory: {destination}")
    originals = [pd.read_parquet(root / f"factors_preprocessed/raw/csi300_factors_{year}.parquet") for year in range(2015, 2024)]
    original = pd.concat(originals).sort_index()
    expansion = pd.read_parquet(root / "alpha158_expansion/stage1_candidates.parquet")
    labels = pd.read_parquet(root / "labels/forward_labels.parquet").reset_index()
    if "signal_date" not in labels.columns and "datetime" in labels.columns:
        raise ValueError("Labels must use signal_date; inspect the actual label schema")
    joined = align_panels(original, expansion, labels)
    baseline = list(original.columns)
    destination.mkdir(parents=True, exist_ok=True)

    for split, (start, end) in PERIODS.items():
        dates = joined.index.get_level_values("datetime")
        period = joined.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))].copy()
        period = period.loc[period["split"].astype(str).str.lower().eq(split)]
        # Inspect the real status vocabulary before applying a validity filter.
        statuses = period["label_status_5d"].value_counts(dropna=False).to_dict()
        print(f"{split} label statuses: {statuses}")
        period = period.loc[period["label_5d"].notna()]
        print(f"{split}: {len(period):,} rows, {period.index.get_level_values('datetime').nunique()} dates")
        daily = daily_rank_ic(period, baseline + CANDIDATES)
        daily.to_csv(destination / f"{split}_daily_rank_ic.csv", index=False)
        summarize_ic(daily).to_csv(destination / f"{split}_rank_ic_summary.csv", index=False)
        factor_correlations(period, baseline, CANDIDATES).to_csv(destination / f"{split}_factor_correlations.csv", index=False)
    print(f"Saved Stage 2 diagnostic reports to {destination}")


if __name__ == "__main__":
    main()
