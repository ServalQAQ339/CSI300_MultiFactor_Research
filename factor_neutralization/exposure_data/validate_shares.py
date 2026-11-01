"""Validate downloaded CNINFO share-event caches before constructing exposures."""
import argparse
import json
from pathlib import Path
import pandas as pd


def audit_share_cache(cache: Path, symbols: list[str]) -> tuple[pd.DataFrame,dict]:
    """Description: Inspect share-event records and preserve failures and uncertainty.

    Parameters:
        cache (Path): Directory containing per-symbol CNINFO CSV files.
        symbols (list[str]): Expected six-digit symbols from observed historical universe.
    Returns:
        tuple[pd.DataFrame, dict]: Concatenated valid-schema records and aggregate audit report.
    Notes:
        Valid schema and nonnegative values do not certify share units or historical PIT.
    """
    required = ['证券代码', '公告日期', '变动日期', '变动原因', '总股本', '已流通股份']
    chunks = []
    problems = {}
    missing = []
    counts = {}
    for code in symbols:
        paths=sorted(cache.glob(f'cninfo_shares_{code}_*.csv'))
        if not paths: missing.append(code); continue
        if len(paths) > 1:
            problems[code] = 'Multiple cache snapshots: select an explicit version'
            continue
        try:
            d = pd.read_csv(
                paths[0],
                dtype={'证券代码': 'string', '变动原因编码': 'string'},
                encoding='utf-8-sig',
            )
            if not set(required).issubset(d.columns): raise ValueError('Missing required fields')
            if d.empty: raise ValueError('Empty share-event history')
            d['证券代码']=d['证券代码'].str.zfill(6)
            if '变动原因编码' in d.columns:
                d['变动原因编码'] = d['变动原因编码'].str.zfill(6)
            if not d['证券代码'].eq(code).all(): raise ValueError('Unexpected symbol in cached file')
            for field in ['公告日期','变动日期']: d[field]=pd.to_datetime(d[field],errors='coerce')
            for field in ['总股本','已流通股份']: d[field]=pd.to_numeric(d[field],errors='coerce')
            counts[code] = {
                'rows': len(d),
                'invalid_total_shares': int((d['总股本'].isna() | (d['总股本'] <= 0)).sum()),
                'missing_announcement': int(d['公告日期'].isna().sum()),
                'missing_effective': int(d['变动日期'].isna().sum()),
                'periodic_snapshots': int(d['变动原因'].astype(str).str.contains('定期报告').sum()),
            }
            chunks.append(d)
        except Exception as exc:
            problems[code] = f'{type(exc).__name__}: {exc}'
    report = {
        'requested_symbols': len(symbols),
        'cached_symbols': len(counts),
        'missing_symbols': missing,
        'invalid_symbols': problems,
        'per_symbol': counts,
        'share_unit_certified': False,
        'strict_pit_certified': False,
    }
    return (
        pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=required)
    ), report


def main() -> None:
    """Description: Write an audit report for cached historical share events.

    Parameters:
        None: Command-line options define input and output paths.
    Returns:
        None: Writes report and consolidated event parquet to a separate directory.
    Notes:
        Reports missing symbols without fabricating records.
    """
    p=argparse.ArgumentParser()
    p.add_argument('--symbols-file',required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    symbols = [
        s.strip()
        for s in Path(a.symbols_file).read_text(encoding='utf-8-sig').splitlines()
        if s.strip()
    ]
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)

    d, r = audit_share_cache(Path(a.cache), symbols)
    for col in d.select_dtypes(include=['object']).columns:
        d[col] = d[col].astype('string')
    d.to_parquet(out/'share_events_raw.parquet', index=False)

    (out / 'share_cache_audit.json').write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps({k:v for k,v in r.items() if k!='per_symbol'},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
