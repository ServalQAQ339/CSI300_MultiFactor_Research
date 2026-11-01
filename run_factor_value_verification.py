"""Command-line entry point for the independent original-27 factor value audit."""
import argparse
import json
from pathlib import Path
from factor_verification.verification import run_audit


def main() -> None:
    """Description: Run read-only historical factor verification and write reports.

    Parameters:
        None: Arguments are read from the command line.
    Returns:
        None: Prints the final audit summary and writes reports.
    Notes:
        Requires explicit OHLCV source path to prevent accidental use of the wrong vintage.
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        '--ohlcv-dir', type=Path, required=True, help='Folder with csi300_daily_YYYY.parquet files'
    )
    p.add_argument(
        '--raw-dir',
        type=Path,
        default=Path(r'D:\QuantResearch\CSI300_MultiFactor_Research\data\factors_preprocessed\raw'),
    )
    p.add_argument(
        '--output',
        type=Path,
        default=Path(
            r'D:\QuantResearch\CSI300_MultiFactor_Research\reports\factor_verification\original27_value_audit_v1'
        ),
    )
    p.add_argument('--calendar-file', type=Path, default=None)
    p.add_argument('--enhanced-dir', type=Path, default=None)
    p.add_argument('--atol', type=float, default=1e-8)
    p.add_argument('--rtol', type=float, default=1e-7)
    args = p.parse_args()
    print(
        json.dumps(
            run_audit(
                args.ohlcv_dir,
                args.raw_dir,
                args.output,
                args.calendar_file,
                args.enhanced_dir,
                args.atol,
                args.rtol,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
