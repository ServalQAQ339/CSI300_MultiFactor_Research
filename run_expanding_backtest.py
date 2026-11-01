"""Run fixed quarterly expanding-window CSI300 diagnostic backtest."""

import argparse

from portfolio.expanding import run


def main() -> None:
    """Parse command-line paths and run Stage 7.

    Description:
        Execute quarterly expanding model fitting and diagnostic portfolios.
    Parameters:
        None: Reads arguments from the command line.
    Returns:
        None: Writes reports to the requested output directory.
    Notes:
        Model settings and selected features are frozen from prior stages.
    """
    parser = argparse.ArgumentParser()
    for name in ("data-root", "candidates", "selection", "provider", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    run(args.data_root, args.candidates, args.selection, args.provider,
        args.output, args.top_k, args.cost_bps)


if __name__ == "__main__":
    main()
