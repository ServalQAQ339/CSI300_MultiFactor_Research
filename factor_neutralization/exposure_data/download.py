"""Download and cache CNINFO share records with bounded request rate and resumability."""
import argparse
import json
import time
from pathlib import Path
import pandas as pd


def download_shares(symbols: list[str], output: Path, start_date: str, end_date: str, delay: float = 2.0) -> dict:
    """Description: Fetch per-stock CNINFO share records, caching successful results.

    Parameters:
        symbols (list[str]): Six-digit stock codes to fetch.
        output (Path): Directory for immutable per-symbol CSV snapshots.
        start_date (str): YYYYMMDD starting date.
        end_date (str): YYYYMMDD ending date.
        delay (float): Minimum seconds between requests.
    Returns:
        dict: Success, skipped and failed counts and error details.
    Notes:
        Never calls Eastmoney Push2. No automatic retries after errors.
        Source snapshots are not PIT-certified merely because they were downloaded.
    """
    import akshare as ak
    output.mkdir(parents=True,exist_ok=True); status = {'success':0,'skipped':0,'failed':{}}
    for symbol in symbols:
        code = str(symbol).zfill(6); path = output / f'cninfo_shares_{code}_{start_date}_{end_date}.csv'
        if path.exists(): status['skipped'] += 1; continue
        try:
            df = ak.stock_share_change_cninfo(symbol=code,start_date=start_date,end_date=end_date)
            tmp = path.with_suffix('.tmp'); df.to_csv(tmp,index=False,encoding='utf-8-sig'); tmp.replace(path); status['success'] += 1
        except Exception as exc: status['failed'][code] = f'{type(exc).__name__}: {exc}'
        time.sleep(max(0.0,delay))
    (output/'download_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    return status


def main() -> None:
    """Description: Run controlled share-history acquisition from a symbol text file.

    Parameters:
        None: Command-line arguments provide paths, dates and delay.
    Returns:
        None: Writes source snapshots and a status JSON.
    Notes:
        This tool never alters existing files.
    """
    p=argparse.ArgumentParser(); p.add_argument('--symbols-file',required=True); p.add_argument('--output',required=True); p.add_argument('--start',default='20100101'); p.add_argument('--end',default='20260930'); p.add_argument('--delay',type=float,default=2.0); a=p.parse_args()
    symbols=[line.strip() for line in Path(a.symbols_file).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    print(json.dumps(download_shares(symbols,Path(a.output),a.start,a.end,a.delay),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
