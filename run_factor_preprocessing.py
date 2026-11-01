"""Run deterministic, daily cross-sectional preprocessing for enhanced CSI300 factors."""

from pathlib import Path
import pandas as pd
from config import DATA_ROOT
from factors.factor_preprocessor import (
    PreprocessConfig,
    preprocess_factor_panel,
    compare_prefix_invariance,
)

RESEARCH_START = "2015-04-01"
INPUT_DIR = DATA_ROOT / "data" / "factors_enhanced"
OUTPUT_ROOT = DATA_ROOT / "data" / "factors_preprocessed"
METHODS = ("rank_gaussian", "robust", "rank_uniform", "raw")
MIN_VALID = 10


def run_preprocessing(methods: tuple[str, ...] = METHODS, overwrite: bool = False) -> pd.DataFrame:
    """Description:
        Read enhanced annual factor files and save independently transformed daily panels.

    Parameters:
        methods (tuple[str, ...]): Preprocessing methods to run.
        overwrite (bool): Whether to regenerate existing transformed annual files.

    Returns:
        pd.DataFrame: Summary with method, year, rows, coverage, and output path.

    Notes:
        Warm-up rows are excluded only from final outputs, after factor calculation.
        Factor columns are read from the enhanced factor files; no labels are processed.
    """
    files = sorted(INPUT_DIR.glob("csi300_factors_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No enhanced factor files found in {INPUT_DIR}")
    summaries = []

    for file in files:
        panel = pd.read_parquet(file)
        panel = panel.loc[panel.index.get_level_values("datetime") >= pd.Timestamp(RESEARCH_START)]

        for method in methods:
            output_dir = OUTPUT_ROOT / method
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / file.name

            if output_path.exists() and not overwrite:
                summaries.append(
                    {
                        "method": method,
                        "year": int(file.stem.split("_")[-1]),
                        "status": "cached",
                        "path": str(output_path),
                    }
                )
                continue

            transformed = preprocess_factor_panel(
                panel, PreprocessConfig(method=method, min_valid=MIN_VALID)
            )
            transformed.to_parquet(output_path, engine="pyarrow", compression="snappy")

            summaries.append(
                {
                    "method": method,
                    "year": int(file.stem.split("_")[-1]),
                    "status": "generated",
                    "rows": len(transformed),
                    "mean_coverage": float(transformed.notna().mean().mean()),
                    "path": str(output_path),
                }
            )
            print(
                f"{method} {file.stem}: {len(transformed):,} rows, coverage={transformed.notna().mean().mean():.4f}",
                flush=True,
            )

    report = pd.DataFrame(summaries)
    report_dir = DATA_ROOT / "reports" / "factor_preprocessing"
    report_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_dir / "preprocessing_summary.csv", index=False)

    return report


if __name__ == "__main__":
    print(run_preprocessing().to_string(index=False))
