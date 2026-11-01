"""Run fixed LightGBM comparisons from the CSI300 project root."""

import argparse

from modeling.modeling import run_modeling


def main() -> None:
    """Parse paths and execute the fixed three-model experiment.

    Description:
        Launch Stage 5 using the existing factor and selection artifacts.
    Parameters:
        None: Paths are provided as command-line arguments.
    Returns:
        None: Saves reports and prints model comparison statistics.
    Notes:
        Run from the project root; no Validation hyperparameter tuning is used.
    """
    parser = argparse.ArgumentParser(description="CSI300 fixed LightGBM comparison")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_modeling(args.data_root, args.candidates, args.selection, args.output)


if __name__ == "__main__":
    main()
