"""Description: Audit local historical industry and capitalization source availability without altering data. Parameters: CLI paths to project data and output directory. Returns: JSON inventory and candidate field metadata. Notes: File presence never establishes point-in-time validity."""

import argparse
import json
from pathlib import Path

KEYWORDS = {
    'industry': ('industry', 'sector', 'sw_level', 'citic', 'shenwan', '行业', '申万', '中信'),
    'capitalization': (
        'market_cap',
        'mktcap',
        'total_mv',
        'circ_mv',
        'shares',
        'capitalization',
        '市值',
        '股本',
    ),
    'pit_timing': ('announce', 'publish', 'effective', 'available', 'disclosure', '公告', '生效'),
}
EXTENSIONS = {'.csv', '.parquet', '.feather', '.xlsx', '.json'}


def classify_fields(names):
    """Description: Identify candidate industry, capitalization and timing fields by name. Parameters: names (iterable[str]) field names. Returns: dict[str,list[str]] matching candidate names. Notes: Heuristic identification is not proof of data quality or PIT correctness."""
    fields = [str(x) for x in names]
    return {
        kind: [field for field in fields if any(token in field.lower() for token in tokens)]
        for kind, tokens in KEYWORDS.items()
    }


def inspect_file(path):
    """Description: Inspect supported tabular files without modifying them. Parameters: path (pathlib.Path) file location. Returns: dict with inspection status, columns and candidate matches. Notes: Parquet schema inspection uses pyarrow; large CSV files read header only."""
    result = {'path': str(path), 'status': 'not_inspected', 'columns': [], 'candidates': {}}
    try:
        if path.suffix.lower() == '.parquet':
            import pyarrow.parquet as pq
            columns = pq.ParquetFile(path).schema_arrow.names
        elif path.suffix.lower() == '.csv':
            import pandas as pd
            columns = list(pd.read_csv(path, nrows=0).columns)
        elif path.suffix.lower() == '.feather':
            import pyarrow.feather as feather
            columns = feather.read_table(path, memory_map=True).column_names
        elif path.suffix.lower() == '.xlsx':
            import pandas as pd
            columns = list(pd.read_excel(path, nrows=0).columns)
        elif path.suffix.lower() == '.json':
            result['status'] = 'filename_only'; return result
        else:
            return result
        result.update(status='schema_read', columns=columns, candidates=classify_fields(columns))
    except Exception as exc:
        result.update(status='inspection_error', error=f'{type(exc).__name__}: {exc}')
    return result


def audit(root, output, max_files=10000):
    """Description: Inventory possible historical industry and capitalization files. Parameters: root (Path) local data root; output (Path) JSON destination; max_files (int) scanning limit. Returns: dict containing findings and explicit PIT assessment. Notes: Does not read row-level values or certify historical availability."""
    root, output = Path(root), Path(output)
    if not root.is_dir():
        raise FileNotFoundError(f'Data root does not exist: {root}')
    paths = sorted(
        (p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS), key=str
    )
    selected = [
        p
        for p in paths
        if any(token in p.name.lower() for values in KEYWORDS.values() for token in values)
    ]
    results = [inspect_file(p) for p in selected[:max_files]]
    report = {
        'root': str(root),
        'total_supported_files': len(paths),
        'candidate_filename_files': len(selected),
        'inspected_files': len(results),
        'truncated': len(selected) > max_files,
        'files': results,
        'pit_certified': False,
        'reason': 'Historical dates, disclosure timing, classification version, and market-cap price/share consistency require separate row-level verification.',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Non-destructive local PIT source inventory')
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-files', type=int, default=10000)
    args = parser.parse_args()
    result = audit(Path(args.root), Path(args.output), args.max_files)
    print(
        json.dumps({k: v for k, v in result.items() if k != 'files'}, ensure_ascii=False, indent=2)
    )
