"""Maintain formula provenance and deterministic duplicate exclusions."""

import pandas as pd


BASE_FAMILIES = ("OPEN_CLOSE_RATIO", "HIGH_LOW_RANGE", "CLOSE_LOCATION")
WINDOWS = (5, 10, 20, 60)


def build_registry(columns: list[str]) -> pd.DataFrame:
    """Classify candidates by provenance and known formula overlap.

    Description:
        Mark known exact/affine duplicates while retaining all raw features.
    Parameters:
        columns (list[str]): Available candidate feature names.
    Returns:
        pd.DataFrame: Feature metadata and exclusion reasons.
    Notes:
        This registry does not claim all mathematical equivalences are found.
    """
    rows = []
    for name in columns:
        if name in BASE_FAMILIES or any(name.startswith(x + "_") for x in ("MOM", "MA_RATIO", "VOLATILITY", "VOLUME_RATIO", "HIGH_POSITION", "LOW_POSITION")):
            origin = "Original27"
        elif name.startswith(("CORR", "VSTD", "RSV")) and name[4 if name.startswith("CORR") or name.startswith("VSTD") else 3:].isdigit():
            origin = "Qlib Alpha158 subset"
        else:
            origin = "OHLCV-inspired"
        reason = ""
        if name.startswith("VOLUME_TREND_"):
            reason = "Affine duplicate of Original27 VOLUME_RATIO for matching windows" if name.rsplit("_", 1)[-1] in {str(w) for w in WINDOWS} else ""
        if name.startswith("CLOSE_RANGE_POSITION_") and name.rsplit("_", 1)[-1] in {"5", "10", "20"}:
            reason = "Formula overlap with RSV; verify zero-denominator convention"
        if name.startswith("RET_NEG_RATIO_"):
            reason = "Complement of RET_POS_RATIO when all returns are nonzero; conditional overlap"
        rows.append({"factor": name, "origin": origin, "excluded": bool(reason), "reason": reason})
    return pd.DataFrame(rows)
