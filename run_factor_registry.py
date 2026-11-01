"""Generate and validate the original CSI300 factor registry without modifying factor data."""
from pathlib import Path
import argparse
import json
from factor_registry import (
    build_registry,
    validate_registry,
    validate_parquet_columns,
    write_registry,
)


def main() -> None:
    """Description: Create the versioned registry and optionally audit a raw parquet schema.

    Parameters:
        None: Configuration is supplied through CLI flags.
    Returns:
        None: Writes JSON registry and prints audit results.
    Notes:
        Existing registry files are never overwritten without --overwrite.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        default=r'D:\QuantResearch\CSI300_MultiFactor_Research\reports\factor_registry\original27_registry_v1.json',
    )
    parser.add_argument(
        '--sample-parquet',
        default=r'D:\QuantResearch\CSI300_MultiFactor_Research\data\factors_preprocessed\raw\csi300_factors_2015.parquet',
    )
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
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


if __name__ == '__main__':
    main()
