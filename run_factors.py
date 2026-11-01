"""Factor-stage entrypoint: Original27 factors, warm-up backfill, preprocessing and registry.

Usage:
    python run_factors.py baseline
    python run_factors.py backfill
    python run_factors.py preprocess
    python run_factors.py check-preprocess
    python run_factors.py registry [--output FILE] [--sample-parquet FILE] [--overwrite]

The first four steps have no options; they take every path from ``config.py``. Every heavy
dependency is imported inside the subcommand handlers, so importing this module never touches
data.
"""

import argparse
import json
from pathlib import Path

from config import DATA_DIR, DATA_OUTPUT_DIR, DATA_ROOT, END_DATE, REPORT_OUTPUT_DIR, START_DATE

PREPROCESS_RESEARCH_START = "2015-04-01"
PREPROCESS_INPUT_DIR = DATA_ROOT / "data" / "factors_enhanced"
PREPROCESS_OUTPUT_ROOT = DATA_ROOT / "data" / "factors_preprocessed"
PREPROCESS_METHODS = ("rank_gaussian", "robust", "rank_uniform", "raw")
PREPROCESS_MIN_VALID = 10


def cmd_baseline(args: argparse.Namespace) -> None:
    """Generate baseline Original27 factors without warm-up backfill into data/factors.

    Notes:
        The former run_factor_pipeline.py ran this at import time; here it runs only when
        the subcommand is dispatched. Existing yearly files are kept (overwrite=False).
    """
    from data.data_loader import get_trading_calendar, initialize_qlib
    from factors.factor_engine import run_factor_engine

    initialize_qlib()

    calendar = get_trading_calendar(START_DATE, END_DATE)

    summary = run_factor_engine(
        data_dir=DATA_OUTPUT_DIR,
        output_dir=DATA_ROOT / "data" / "factors",
        calendar=calendar,
        start_year=2015,
        end_year=2026,
        overwrite=False
    )

    print(summary.to_string(index=False))


def run_complete_backfill(overwrite: bool = False):
    """Generate enhanced factors with pre-membership historical OHLCV.

    Parameters:
        overwrite (bool): Whether to regenerate existing enhanced factor files.

    Returns:
        pd.DataFrame: Annual generation summary.

    Notes:
        The calendar begins in 2014 to support 2015 warm-up calculations.
        The original baseline factor files remain untouched.
    """
    from data.data_loader import get_trading_calendar, initialize_qlib
    from factors.factor_backfill import run_backfilled_factor_pipeline

    initialize_qlib()
    calendar = get_trading_calendar("2014-01-01", END_DATE)
    summary = run_backfilled_factor_pipeline(
        DATA_OUTPUT_DIR,
        DATA_ROOT / "data" / "factors_enhanced",
        calendar,
        2015,
        int(END_DATE[:4]),
        overwrite,
    )
    print(summary.to_string(index=False))

    return summary


def cmd_backfill(args: argparse.Namespace) -> None:
    """Run the warm-up backfill with overwrite=False."""
    run_complete_backfill(overwrite=False)


def run_preprocessing(methods: tuple[str, ...] = PREPROCESS_METHODS, overwrite: bool = False):
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
    import pandas as pd
    from factors.factor_preprocessor import PreprocessConfig, preprocess_factor_panel

    files = sorted(PREPROCESS_INPUT_DIR.glob("csi300_factors_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No enhanced factor files found in {PREPROCESS_INPUT_DIR}")
    summaries = []

    for file in files:
        panel = pd.read_parquet(file)
        panel = panel.loc[
            panel.index.get_level_values("datetime") >= pd.Timestamp(PREPROCESS_RESEARCH_START)
        ]

        for method in methods:
            output_dir = PREPROCESS_OUTPUT_ROOT / method
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
                panel, PreprocessConfig(method=method, min_valid=PREPROCESS_MIN_VALID)
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


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Run all four preprocessing methods with overwrite=False and print the summary."""
    print(run_preprocessing().to_string(index=False))


def cmd_check_preprocess(args: argparse.Namespace) -> None:
    """Run point-in-time preprocessing diagnostics on cached factor panels."""
    from factors.preprocessing_validator import run_validation

    run_validation(data_root=DATA_ROOT, research_start="2015-04-01")


def cmd_registry(args: argparse.Namespace) -> None:
    """Description: Create the versioned registry and optionally audit a raw parquet schema.

    Parameters:
        args (argparse.Namespace): output, sample_parquet, overwrite.
    Returns:
        None: Writes JSON registry and prints audit results.
    Notes:
        Existing registry files are never overwritten without --overwrite.
    """
    from factors.registry import (
        build_registry,
        validate_parquet_columns,
        validate_registry,
        write_registry,
    )

    registry = build_registry()
    audit = validate_registry(registry)
    if not audit['passed']:
        raise RuntimeError(audit)
    path = write_registry(args.output, registry, overwrite=args.overwrite)

    if Path(args.sample_parquet).exists():
        audit['parquet_schema'] = validate_parquet_columns(args.sample_parquet, registry)
        if not audit['parquet_schema']['passed']:
            raise RuntimeError(audit)

    print(json.dumps({'registry_path': str(path), **audit}, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build the factor-stage parser.

    Returns:
        argparse.ArgumentParser: Parser whose subcommands set ``func`` to their handler.
    """
    parser = argparse.ArgumentParser(description="CSI300 Original27 factor construction")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("baseline", help="Baseline Original27 factors (no warm-up backfill)")
    p.set_defaults(func=cmd_baseline)
    p = sub.add_parser("backfill", help="Original27 factors with warm-up backfill")
    p.set_defaults(func=cmd_backfill)
    p = sub.add_parser("preprocess", help="Daily cross-sectional preprocessing")
    p.set_defaults(func=cmd_preprocess)
    p = sub.add_parser("check-preprocess", help="Point-in-time preprocessing validation")
    p.set_defaults(func=cmd_check_preprocess)

    p = sub.add_parser("registry", help="Build and validate the Original27 registry")
    p.add_argument(
        '--output',
        default=str(REPORT_OUTPUT_DIR / 'factor_registry' / 'original27_registry_v1.json'),
    )
    p.add_argument(
        '--sample-parquet',
        default=str(DATA_DIR / 'factors_preprocessed' / 'raw' / 'csi300_factors_2015.parquet'),
    )
    p.add_argument('--overwrite', action='store_true')
    p.set_defaults(func=cmd_registry)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and dispatch to the selected subcommand."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
