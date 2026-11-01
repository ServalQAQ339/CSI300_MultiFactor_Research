"""Write factor research diagnostics without modifying upstream data."""
import json


def write_audit(path, audit):
    """Description: Write the research audit as human-readable JSON.

    Parameters:
        path (Path): Destination JSON path.
        audit (dict): Audited counts and research limitations.
    Returns:
        None: Writes the JSON file.
    Notes:
        Parent output directory must already exist.
    """
    path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
