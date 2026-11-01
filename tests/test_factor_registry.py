"""Unit tests for registry metadata and non-destructive persistence."""
import pytest
from factor_registry import (
    FACTOR_NAMES,
    build_registry,
    load_registry,
    validate_registry,
    write_registry,
)


def test_inventory():
    """Description: Check exact factor inventory and uniqueness.

    Parameters:
        None: Uses canonical inventory.
    Returns:
        None: Asserts expected names and count.
    Notes:
        Does not check numerical factor expressions.
    """
    assert len(FACTOR_NAMES) == len(set(FACTOR_NAMES)) == 27
    assert validate_registry(build_registry())['passed']


def test_no_unverified_formula_claims():
    """Description: Prevent unverified formulas from being recorded as verified.

    Parameters:
        None: Uses generated metadata.
    Returns:
        None: Asserts all definitions are explicitly pending verification.
    Notes:
        Source expressions must be inspected separately.
    """
    assert all(
        row['formula'] is None and not row['formula_verified']
        for row in build_registry()['factors']
    )


def test_persistence_does_not_overwrite(tmp_path):
    """Description: Verify JSON roundtrip and overwrite protection.

    Parameters:
        tmp_path (Path): Pytest temporary directory fixture.
    Returns:
        None: Asserts persisted registry and overwrite failure.
    Notes:
        Never writes to the user's real project folders.
    """
    path = tmp_path / 'registry.json'
    write_registry(path, build_registry())
    assert len(load_registry(path)['factors']) == 27
    with pytest.raises(FileExistsError):
        write_registry(path, build_registry())


def test_duplicate_rejected():
    """Description: Reject duplicate factor identifiers.

    Parameters:
        None: Uses deliberately corrupted registry metadata.
    Returns:
        None: Asserts validation failure.
    Notes:
        Duplicate IDs are invalid even if names differ.
    """
    registry = build_registry()
    registry['factors'][1]['factor_id'] = registry['factors'][0]['factor_id']
    assert not validate_registry(registry)['passed']
