"""Versioned factor metadata registry with explicit research provenance."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from .definitions import FACTOR_NAMES, metadata_for


def build_registry(source_file: str = 'factors/factor_engine.py') -> dict:
    """Description: Build a non-destructive registry of original factor metadata.

    Parameters:
        source_file (str): Relative source-code location requiring formula verification.
    Returns:
        dict: Registry schema, factor definitions, and provenance fields.
    Notes:
        Formula fields are null until independently verified from source code.
    """
    factors = []
    for name in FACTOR_NAMES:
        factors.append(
            {
                'factor_id': f'original27:{name}:v1',
                'name': name,
                'library': 'original27',
                'version': 1,
                **metadata_for(name),
                'formula': None,
                'formula_verified': False,
                'source_file': source_file,
                'status': 'implemented_definition_pending_audit',
                'requires': [],
                'availability_rule': None,
            }
        )
    return {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'factor_count': len(factors),
        'factors': factors,
    }


def write_registry(path: str | Path, registry: dict, overwrite: bool = False) -> Path:
    """Description: Write metadata to JSON without silently overwriting prior versions.

    Parameters:
        path (str | Path): Destination JSON file.
        registry (dict): Registry object to persist.
        overwrite (bool): Whether explicitly to replace an existing file.
    Returns:
        Path: Destination path.
    Notes:
        Use overwrite only after reviewing existing registry changes.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f'Registry already exists: {destination}')
    destination.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding='utf-8')
    return destination


def load_registry(path: str | Path) -> dict:
    """Description: Read a saved registry.

    Parameters:
        path (str | Path): Existing registry JSON path.
    Returns:
        dict: Parsed registry metadata.
    Notes:
        Run validate_registry before relying on its contents.
    """
    return json.loads(Path(path).read_text(encoding='utf-8'))
