"""Run the compact Stage 4 train-only selection and Ridge comparison."""

import argparse

from factor_selection.selection import run_selection


def main() -> None:
    """Parse command-line settings and execute Stage 4.

    Description:
        Run factor selection and matched-sample Ridge validation.
    Parameters:
        None: Configuration is supplied by command-line arguments.
    Returns:
        None: Prints the summary and writes CSV reports.
    Notes:
        Run from the CSI300 project root on Windows.
    """
    parser = argparse.ArgumentParser(description="Train-only factor selection")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-features", type=int, default=20)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--max-corr", type=float, default=0.85)
    parser.add_argument("--alpha", type=float, default=100.0)
    args = parser.parse_args()

    run_selection(args.data_root, args.candidates, args.registry, args.output,
                  args.max_features, args.min_coverage, args.max_corr, args.alpha)


if __name__ == "__main__":
    main()
