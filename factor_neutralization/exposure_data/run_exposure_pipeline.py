"""Orchestrate read-only historical universe extraction and cached-share validation."""
import argparse
from pathlib import Path
from .build_universe import main as universe_main
from .validate_shares import main as validate_main


def main() -> None:
    """Description: Print the safe staged pipeline commands without making remote requests.

    Parameters:
        None: Uses root and output command-line arguments.
    Returns:
        None: Prints reproducible commands for manual execution.
    Notes:
        Downloading hundreds of stocks requires deliberate user initiation.
    """
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.root)
    out = Path(a.output)
    print(
        f'python -m factor_neutralization.exposure_data.build_universe --root "{root}" --output "{out}"'
    )
    print(
        f'python -m factor_neutralization.exposure_data.download --symbols-file "{out / "symbols.txt"}" --output "{root / "data" / "external" / "shares"}" --start 20100101 --end 20260930 --delay 2'
    )
    print(
        f'python -m factor_neutralization.exposure_data.validate_shares --symbols-file "{out / "symbols.txt"}" --cache "{root / "data" / "external" / "shares"}" --output "{out}"'
    )


if __name__=='__main__': main()
