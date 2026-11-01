
"""
Main entry point for the CSI300 data engineering pipeline.

This module extracts annual historical stock panels, saves Parquet
datasets, and performs basic data quality auditing.
"""

import pandas as pd
from data.data_loader import initialize_qlib, build_daily_dataset
from data.data_audit import run_data_audit, diagnose_missing_patterns
from config import DATA_OUTPUT_DIR, AUDIT_OUTPUT_DIR


def run_complete_data_pipeline(overwrite: bool = False) -> None:
    """
    Execute the CSI300 data extraction and auditing workflow.

    Parameters:
        overwrite (bool): Whether to regenerate existing annual Parquet files.

    Returns:
        None: Saves annual datasets and audit reports and prints summaries.

    Notes:
        Qlib should be initialized with kernels=1 to limit resource usage.
        Annual datasets are processed sequentially to reduce peak memory usage.
    """
    initialize_qlib()

    print("\n[1/2] BUILDING DAILY STOCK PANEL", flush=True)
    summary = build_daily_dataset(overwrite=overwrite)
    print(summary.to_string(index=False))

    print("\n[2/2] RUNNING DATA QUALITY AUDIT", flush=True)
    audit = run_data_audit()
    print(audit.to_string(index=False))

    print("\nDATA PIPELINE COMPLETED", flush=True)
    print(f"Processed data: {DATA_OUTPUT_DIR}")
    print(f"Audit reports: {AUDIT_OUTPUT_DIR}")


def run_missing_diagnostics() -> pd.DataFrame:
    """
    Analyze missing price observations across all cached annual datasets.

    Parameters:
        None.

    Returns:
        pd.DataFrame: Combined stock-level missing data diagnostics.

    Notes:
        Annual panels are loaded sequentially to reduce memory consumption.
        The resulting report is saved to the configured audit directory.
    """
    results = []

    for file in sorted(DATA_OUTPUT_DIR.glob("csi300_daily_*.parquet")):
        year = int(file.stem.split("_")[-1])
        df = pd.read_parquet(file)
        results.append(diagnose_missing_patterns(df, year))
        print(f"Missing diagnostics completed: {year}", flush=True)

    report = pd.concat(results, ignore_index=True)
    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(AUDIT_OUTPUT_DIR / "missing_patterns.csv", index=False)

    print("\nTOP 20 STOCK-YEAR MISSING PATTERNS")
    print(report.head(20).to_string(index=False))

    return report

if __name__ == "__main__":
    RUN_MODE = "missing"

    if RUN_MODE == "full":
        run_complete_data_pipeline(overwrite=False)
    elif RUN_MODE == "missing":
        run_missing_diagnostics()
    else:
        raise ValueError(f"Unknown run mode: {RUN_MODE}")