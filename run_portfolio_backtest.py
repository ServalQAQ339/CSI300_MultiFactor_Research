"""Run the preliminary CSI300 next-open portfolio experiment."""

import argparse

from portfolio.backtest import run_backtest


def main() -> None:
    """Parse inputs and execute the frozen validation backtest.

    Description:
        Launch Stage 6 from the project root.
    Parameters:
        None: Command-line arguments specify inputs and cost assumptions.
    Returns:
        None: Writes validation reports.
    Notes:
        No tuning or Test evaluation is performed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    run_backtest(args.data_root, args.candidates, args.selection, args.provider,
                 args.output, args.top_k, args.cost_bps)


if __name__ == "__main__":
    main()
