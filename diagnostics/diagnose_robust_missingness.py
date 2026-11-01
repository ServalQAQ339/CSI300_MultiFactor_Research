"""Identify exact causes of valid-to-missing cells in daily robust scaling."""
from pathlib import Path
import numpy as np
import pandas as pd
from config import DATA_ROOT
from factors.factor_preprocessor import PreprocessConfig


def diagnose_robust_missingness(data_root: Path = DATA_ROOT, research_start: str = "2015-04-01", min_valid: int = 10, iqr_epsilon: float = 1e-12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Description:
        Attribute every valid-to-missing robust-scaled cell to insufficient cross-sectional sample size or degenerate IQR.

    Parameters:
        data_root (Path): Root directory containing enhanced and preprocessed factor datasets.
        research_start (str): Earliest included research date in YYYY-MM-DD format.
        min_valid (int): Minimum finite stock observations per date and factor.
        iqr_epsilon (float): IQR threshold below which robust scaling yields missing values.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Per-date-factor diagnostic groups and year-reason summary.

    Notes:
        Uses the same thresholds as PreprocessConfig and compares saved robust output to enhanced raw factors.
        Only valid-to-missing cells are attributed; missing-to-valid cells are checked separately.
    """
    config = PreprocessConfig(method="robust", min_valid=min_valid, iqr_epsilon=iqr_epsilon)
    rows = []
    source_files = sorted((data_root / "data" / "factors_enhanced").glob("csi300_factors_*.parquet"))
    if not source_files:
        raise FileNotFoundError("No enhanced factor files found")
    for source_file in source_files:
        year = int(source_file.stem.split("_")[-1])
        raw = pd.read_parquet(source_file).sort_index()
        raw = raw.loc[raw.index.get_level_values("datetime") >= pd.Timestamp(research_start)]
        saved_path = data_root / "data" / "factors_preprocessed" / "robust" / source_file.name
        if not saved_path.exists():
            raise FileNotFoundError(saved_path)
        scaled = pd.read_parquet(saved_path).sort_index()
        if not raw.index.equals(scaled.index) or not raw.columns.equals(scaled.columns):
            raise ValueError(f"Index or column mismatch for {year}")
        for date, section in raw.groupby(level="datetime", sort=False):
            output = scaled.loc[section.index]
            for factor in raw.columns:
                values = pd.to_numeric(section[factor], errors="coerce").replace([np.inf, -np.inf], np.nan)
                valid = values.dropna()
                lost = int((values.notna() & output[factor].isna()).sum())
                created = int((values.isna() & output[factor].notna()).sum())
                if not lost and not created:
                    continue
                iqr = float(valid.quantile(0.75) - valid.quantile(0.25)) if len(valid) >= config.min_valid else np.nan
                if len(valid) < config.min_valid:
                    reason = "insufficient_valid_stocks"
                elif not np.isfinite(iqr) or iqr <= config.iqr_epsilon:
                    reason = "degenerate_iqr"
                else:
                    reason = "unexpected_missingness"
                rows.append({"year": year, "datetime": date, "factor": factor, "reason": reason, "valid_stocks": len(valid), "iqr": iqr, "lost_cells": lost, "created_cells": created})
        print(f"Diagnosed {year}", flush=True)
    details = pd.DataFrame(rows, columns=["year", "datetime", "factor", "reason", "valid_stocks", "iqr", "lost_cells", "created_cells"])
    summary = details.groupby(["year", "reason"], as_index=False)[["lost_cells", "created_cells"]].sum() if not details.empty else pd.DataFrame(columns=["year", "reason", "lost_cells", "created_cells"])
    output_dir = data_root / "reports" / "preprocessing_validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "robust_missingness_details.csv", index=False)
    summary.to_csv(output_dir / "robust_missingness_summary.csv", index=False)
    total_lost = int(details["lost_cells"].sum())
    total_created = int(details["created_cells"].sum())
    print("\nROBUST MISSINGNESS SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nTOTAL LOST: {total_lost}; TOTAL CREATED: {total_created}")
    print(f"EXPECTED LOST FROM PRIOR AUDIT: 1482; MATCH: {total_lost == 1482}")
    print(f"UNEXPECTED LOST: {int(details.loc[details['reason'] == 'unexpected_missingness', 'lost_cells'].sum())}")
    return details, summary


if __name__ == "__main__":
    diagnose_robust_missingness()
