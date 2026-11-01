"""Independent schema checks for factor registries and parquet factor panels."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .definitions import FACTOR_NAMES


def validate_registry(registry: dict) -> dict:
    """Description: Check uniqueness, count, and metadata completeness.

    Parameters:
        registry (dict): Loaded registry object.
    Returns:
        dict: Validation counts and error messages.
    Notes:
        Formula verification is a separate source-code audit.
    """
    rows = registry.get('factors', [])
    names = [row.get('name') for row in rows]
    ids = [row.get('factor_id') for row in rows]
    errors = []
    if len(names) != len(set(names)) or len(ids) != len(set(ids)):
        errors.append('Duplicate factor name or factor ID')
    if set(names) != set(FACTOR_NAMES) or len(rows) != 27:
        errors.append('Original factor inventory mismatch')
    if any(not row.get('source_file') or not row.get('status') for row in rows):
        errors.append('Missing source or research status')
    if any(row.get('formula_verified') and not row.get('formula') for row in rows):
        errors.append('Verified formula has no expression')
    return {'passed': not errors, 'factor_count': len(rows), 'formula_verified_count': sum(bool(r.get('formula_verified')) for r in rows), 'errors': errors}


def validate_parquet_columns(path: str | Path, registry: dict) -> dict:
    """Description: Compare parquet feature names with the registered inventory.

    Parameters:
        path (str | Path): A yearly raw-factor parquet file.
        registry (dict): Validated registry object.
    Returns:
        dict: Missing and unexpected feature names, index names, and pass status.
    Notes:
        Checks schema only; does not certify factor formulas or data values.
    """
    frame = pd.read_parquet(Path(path))
    expected = {row['name'] for row in registry['factors']}
    actual = set(frame.columns)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    index_names = list(frame.index.names)
    return {'passed': not missing and not extra and index_names == ['datetime', 'instrument'], 'missing': missing, 'unexpected': extra, 'index_names': index_names, 'rows': len(frame)}
