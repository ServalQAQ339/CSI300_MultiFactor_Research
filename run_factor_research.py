"""Command-line entry point for CSI300 Raw x 5D factor research."""
import argparse
from pathlib import Path
from factor_research.config import DATA_ROOT
from factor_research.runner import run


def main() -> None:
    """Description: Execute factor research with project defaults and optional overrides.

    Parameters:
        None: Uses command-line arguments.
    Returns:
        None: Prints audit and saves reports on D drive.
    Notes:
        Does not modify the existing upstream factor or label data.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output', type=Path, default=DATA_ROOT / 'reports/factor_research/raw_5d_v12')
    parser.add_argument('--min-n', type=int, default=30)
    args = parser.parse_args()
    audit = run(args.data_root, args.output, args.min_n)
    print(f'Completed: {audit}; output={args.output}')


if __name__ == '__main__': main()
