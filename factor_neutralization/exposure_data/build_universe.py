"""Build observed CSI300 instrument-date universes from existing processed parquet files."""
import argparse
import json
from pathlib import Path
import pandas as pd


def extract_universe(root: Path, start: str = '2015-04-01', end: str = '2026-09-30') -> pd.DataFrame:
    """Description: Extract unique observed instrument-date pairs without inventing missing membership.

    Parameters:
        root (Path): Research data root containing data/processed/csi300_daily_YYYY.parquet.
        start (str): Inclusive first signal date.
        end (str): Inclusive last signal date.
    Returns:
        pd.DataFrame: Sorted datetime, instrument, and six-digit symbol columns.
    Notes:
        Observed price rows are not a separately PIT-certified membership calendar.
    """
    folder = Path(root) / 'data' / 'processed'; files = sorted(folder.glob('csi300_daily_*.parquet'))
    if not files: raise FileNotFoundError(f'No processed yearly parquet files found under {folder}')
    parts = []
    for path in files:
        d = pd.read_parquet(path, columns=[])
        if not isinstance(d.index, pd.MultiIndex) or list(d.index.names) != ['datetime','instrument']: raise ValueError(f'Unexpected index structure: {path}')
        x = d.index.to_frame(index=False)[['datetime','instrument']]; x['datetime'] = pd.to_datetime(x['datetime']); x = x[x['datetime'].between(pd.Timestamp(start),pd.Timestamp(end))]; parts.append(x)
    out = pd.concat(parts,ignore_index=True).drop_duplicates(['datetime','instrument']).sort_values(['datetime','instrument']).reset_index(drop=True)
    if out.empty: raise ValueError('No observed rows in requested date range')
    if not out['instrument'].str.fullmatch(r'(SH|SZ|BJ)\d{6}').all(): raise ValueError('Unexpected instrument code format')
    out['symbol'] = out['instrument'].str[-6:]
    return out


def main() -> None:
    """Description: Save the observed universe and a source-provenance summary.

    Parameters:
        None: Values supplied via command-line arguments.
    Returns:
        None: Writes parquet, symbol list, and JSON to an independent output directory.
    Notes:
        Does not modify existing factor, label, or processed data files.
    """
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--output',required=True); p.add_argument('--start',default='2015-04-01'); p.add_argument('--end',default='2026-09-30'); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); d=extract_universe(Path(a.root),a.start,a.end); d.to_parquet(out/'observed_universe.parquet',index=False); symbols=sorted(d['symbol'].unique()); (out/'symbols.txt').write_text('\n'.join(symbols)+'\n',encoding='utf-8'); summary={'rows':len(d),'unique_symbols':len(symbols),'unique_instruments':int(d['instrument'].nunique()),'first_date':str(d['datetime'].min().date()),'last_date':str(d['datetime'].max().date()),'source':'data/processed/csi300_daily_YYYY.parquet','membership_pit_certified':False}; (out/'universe_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
