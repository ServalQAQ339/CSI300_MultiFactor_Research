"""Validate temporal isolation, rank preservation, missingness, and saved factor transformations."""

from pathlib import Path
import numpy as np
import pandas as pd
from factors.factor_preprocessor import PreprocessConfig, preprocess_factor_panel

METHODS = ("raw", "rank_uniform", "rank_gaussian", "robust")


def _compare(reference: pd.DataFrame, candidate: pd.DataFrame, atol: float = 1e-10) -> dict:
    """Description:
        Compare aligned factor panels including missing-value patterns and finite values.

    Parameters:
        reference (pd.DataFrame): Reference factor panel.
        candidate (pd.DataFrame): Recomputed factor panel.
        atol (float): Maximum allowed absolute numerical difference.

    Returns:
        dict: Counts of index, column, missingness, and numerical discrepancies.
    """
    keys_match = reference.index.equals(candidate.index) and reference.columns.equals(
        candidate.columns
    )

    if not keys_match:
        return {
            "keys_match": False,
            "missing_mismatch": -1,
            "value_mismatch": -1,
            "max_abs_difference": np.nan,
        }

    a, b = reference.to_numpy(dtype=float), candidate.to_numpy(dtype=float)
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    missing_mismatch = int(np.count_nonzero(finite_a != finite_b))
    both = finite_a & finite_b
    differences = np.abs(a[both] - b[both])

    return {
        "keys_match": True,
        "missing_mismatch": missing_mismatch,
        "value_mismatch": int(np.count_nonzero(differences > atol)),
        "max_abs_difference": float(differences.max()) if differences.size else 0.0,
    }


def validate_one_year(
    panel: pd.DataFrame, method: str, saved_path: Path | None = None, min_valid: int = 10
) -> list[dict]:
    """Description:
        Run deterministic preprocessing bias diagnostics for one annual factor panel.

    Parameters:
        panel (pd.DataFrame): Enhanced raw factor panel with (datetime, instrument) index.
        method (str): Cross-sectional preprocessing method.
        saved_path (Path | None): Optional path to the corresponding preprocessed Parquet file.
        min_valid (int): Minimum valid observations for a cross-sectional transformation.

    Returns:
        list[dict]: One diagnostic result per validation test.

    Notes:
        Prefix and perturbation checks test algorithmic causality, not vendor data availability.
        The membership check requires an independently validated point-in-time universe upstream.
    """
    config = PreprocessConfig(method=method, min_valid=min_valid)
    panel = panel.sort_index()
    dates = panel.index.get_level_values("datetime").unique().sort_values()
    if len(dates) < 3:
        raise ValueError("At least three trading dates are required")
    cutoff = dates[len(dates) // 2]
    historical = panel.loc[panel.index.get_level_values("datetime") <= cutoff]
    future_mask = panel.index.get_level_values("datetime") > cutoff
    prefix_reference = preprocess_factor_panel(historical, config)
    full = preprocess_factor_panel(panel, config)
    records = []

    def add(test: str, comparison: dict, detail: str = "") -> None:
        """Description:
            Append a diagnostic record with a deterministic pass/fail decision.

        Parameters:
            test (str): Diagnostic name.
            comparison (dict): Numeric comparison results.
            detail (str): Additional context.

        Returns:
            None: Appends a record to the result list.
        """
        passed = (
            comparison["keys_match"]
            and comparison["missing_mismatch"] == 0
            and comparison["value_mismatch"] == 0
        )

        records.append(
            {
                "method": method,
                "test": test,
                "passed": bool(passed),
                "cutoff": str(cutoff.date()),
                "rows_checked": len(historical),
                **comparison,
                "detail": detail,
            }
        )

    add("prefix_invariance", _compare(full.loc[historical.index], prefix_reference))
    modified = panel.copy()
    if future_mask.any():
        modified.loc[future_mask, :] = 123456.789
    modified_result = preprocess_factor_panel(modified, config)
    add(
        "future_perturbation",
        _compare(full.loc[historical.index], modified_result.loc[historical.index]),
    )

    target_date = dates[len(dates) // 3]
    date_mask = panel.index.get_level_values("datetime") == target_date
    date_reference = full.loc[date_mask]
    isolated = preprocess_factor_panel(panel.loc[date_mask], config)
    add("date_isolation", _compare(date_reference, isolated), str(target_date.date()))

    expected_valid = np.isfinite(panel.to_numpy(dtype=float))
    actual_valid = np.isfinite(full.to_numpy(dtype=float))
    created = int(np.count_nonzero(~expected_valid & actual_valid))
    removed = int(np.count_nonzero(expected_valid & ~actual_valid))
    records.append(
        {
            "method": method,
            "test": "missingness_integrity",
            "passed": created == 0,
            "cutoff": "",
            "rows_checked": len(panel),
            "keys_match": True,
            "missing_mismatch": created,
            "value_mismatch": removed,
            "max_abs_difference": 0.0,
            "detail": f"new_valid_from_missing={created}; valid_to_missing={removed} (allowed for min_valid/zero IQR)",
        }
    )

    rank_violations = 0

    for date in dates:
        source = panel.xs(date, level="datetime")
        output = full.xs(date, level="datetime")

        for col in panel.columns:
            a, b = source[col].to_numpy(dtype=float), output[col].to_numpy(dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 2:
                continue
            order = np.argsort(a[mask], kind="mergesort")
            x, y = a[mask][order], b[mask][order]
            rank_violations += int(np.count_nonzero(np.diff(y)[np.diff(x) > 0] <= -1e-10))
            rank_violations += int(np.count_nonzero(np.abs(np.diff(y)[np.diff(x) == 0]) > 1e-10))

    records.append(
        {
            "method": method,
            "test": "rank_preservation",
            "passed": rank_violations == 0,
            "cutoff": "",
            "rows_checked": len(panel),
            "keys_match": True,
            "missing_mismatch": 0,
            "value_mismatch": rank_violations,
            "max_abs_difference": 0.0,
            "detail": "Checks monotonicity and equality of ties",
        }
    )

    if saved_path is not None:
        if not saved_path.exists():
            records.append(
                {
                    "method": method,
                    "test": "saved_output",
                    "passed": False,
                    "cutoff": "",
                    "rows_checked": 0,
                    "keys_match": False,
                    "missing_mismatch": -1,
                    "value_mismatch": -1,
                    "max_abs_difference": np.nan,
                    "detail": f"Missing file: {saved_path}",
                }
            )
        else:
            saved = pd.read_parquet(saved_path).sort_index()
            add("saved_output", _compare(full, saved), str(saved_path))

    return records


def run_validation(
    data_root: Path, research_start: str = "2015-04-01", years: tuple[int, ...] | None = None
) -> pd.DataFrame:
    """Description:
        Validate all available enhanced factor files and preprocessed outputs.

    Parameters:
        data_root (Path): Root directory containing data and reports folders.
        research_start (str): Earliest date included in final research samples.
        years (tuple[int, ...] | None): Optional years to check; None checks all files.

    Returns:
        pd.DataFrame: Detailed validation results saved as CSV.
    """
    source_dir = data_root / "data" / "factors_enhanced"
    files = sorted(source_dir.glob("csi300_factors_*.parquet"))
    if years is not None:
        files = [file for file in files if int(file.stem.split("_")[-1]) in years]
    if not files:
        raise FileNotFoundError(f"No matching enhanced factor files found: {source_dir}")
    results = []

    for file in files:
        year = int(file.stem.split("_")[-1])
        panel = pd.read_parquet(file)
        panel = panel.loc[panel.index.get_level_values("datetime") >= pd.Timestamp(research_start)]
        for method in METHODS:
            saved_path = data_root / "data" / "factors_preprocessed" / method / file.name
            for record in validate_one_year(panel, method, saved_path=saved_path):
                results.append({"year": year, **record})
            print(f"Validated {year}: {method}", flush=True)

    report = pd.DataFrame(results)
    report_dir = data_root / "reports" / "preprocessing_validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_dir / "preprocessing_validation.csv", index=False)

    summary = report.groupby(["method", "test"], as_index=False).agg(
        checks=("passed", "size"),
        failed=("passed", lambda s: int((~s).sum())),
        total_value_mismatches=("value_mismatch", "sum"),
    )

    summary.to_csv(report_dir / "validation_summary.csv", index=False)
    print("\nVALIDATION SUMMARY\n" + summary.to_string(index=False), flush=True)

    return report
