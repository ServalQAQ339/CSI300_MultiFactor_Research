"""Model and portfolio entrypoint: model comparison, validation portfolios and walk-forward.

Usage:
    python run_backtest.py models --data-dir DIR --candidates FILE --selection FILE --output DIR
    python run_backtest.py portfolio --data-dir DIR --candidates FILE --selection FILE
                                     --provider DIR --output DIR [--top-k K] [--cost-bps C]
    python run_backtest.py benchmark --periods FILE --provider DIR --output DIR
    python run_backtest.py walk-forward --data-dir DIR --candidates FILE --selection FILE
                                        --provider DIR --output DIR [--top-k K] [--cost-bps C]

Every heavy dependency is imported inside the subcommand handlers, so importing this module
never touches data or fits a model.
"""

import argparse


def cmd_models(args: argparse.Namespace) -> None:
    """Run the fixed Ridge / LightGBM comparison on existing factor and selection artifacts."""
    from modeling.modeling import run_modeling

    run_modeling(args.data_dir, args.candidates, args.selection, args.output)


def cmd_portfolio(args: argparse.Namespace) -> None:
    """Run the frozen next-open Top-K validation backtest; no tuning or test evaluation."""
    from portfolio.backtest import run_backtest

    run_backtest(args.data_dir, args.candidates, args.selection, args.provider,
                 args.output, args.top_k, args.cost_bps)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Compare validation period returns with the SH000300 price index."""
    from portfolio.benchmark import run_comparison

    run_comparison(args.periods, args.provider, args.output, args.holding_days)


def cmd_walk_forward(args: argparse.Namespace) -> None:
    """Run the quarterly expanding-window diagnostic with frozen features and model settings."""
    from portfolio.expanding import run

    run(args.data_dir, args.candidates, args.selection, args.provider,
        args.output, args.top_k, args.cost_bps)


def build_parser() -> argparse.ArgumentParser:
    """Build the model and portfolio parser.

    Returns:
        argparse.ArgumentParser: Parser whose subcommands set ``func`` to their handler.
    """
    parser = argparse.ArgumentParser(description="CSI300 models, portfolios and walk-forward")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="Fixed Ridge / LightGBM comparison")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_models)

    for name, handler, text in (
        ("portfolio", cmd_portfolio, "Validation Top-K portfolio backtest"),
        ("walk-forward", cmd_walk_forward, "Quarterly expanding-window walk-forward"),
    ):
        p = sub.add_parser(name, help=text)
        for option in ("data-dir", "candidates", "selection", "provider", "output"):
            p.add_argument(f"--{option}", required=True)
        p.add_argument("--top-k", type=int, default=30)
        p.add_argument("--cost-bps", type=float, default=10.0)
        p.set_defaults(func=handler)

    p = sub.add_parser("benchmark", help="Validation portfolios versus SH000300")
    p.add_argument("--periods", required=True)
    p.add_argument("--provider", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--holding-days", type=int, default=5)
    p.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and dispatch to the selected subcommand."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
