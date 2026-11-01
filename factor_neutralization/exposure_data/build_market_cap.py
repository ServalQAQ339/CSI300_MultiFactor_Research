"""Build candidate market-cap exposures with explicit non-certified price and unit assumptions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import normalize_shares, build_share_panel


def construct_candidate_exposures(universe: pd.DataFrame, events_raw: pd.DataFrame, root: Path, unit_multiplier: float, allow_unverified_price: bool = False) -> pd.DataFrame:
    """Description: Join as-known share disclosures with Qlib-derived candidate unadjusted prices.

    Parameters:
        universe (pd.DataFrame): Observed datetime, instrument, symbol rows.
        events_raw (pd.DataFrame): Original CNINFO event fields from cached files.
        root (Path): Research data root with annual processed OHLCV parquet files.
        unit_multiplier (float): Explicit shares-per-reported-unit conversion, not inferred by code.
        allow_unverified_price (bool): Required explicit acknowledgement of unverified close/factor reconstruction.
    Returns:
        pd.DataFrame: Candidate total and float market-cap exposures with PIT and source flags.
    Notes:
        This is exploratory, not certified PIT. Share events use a conservative next-calendar-day proxy.
        Never backfills missing shares and never interprets adjusted Qlib close as raw close.
    """
    if not allow_unverified_price: raise ValueError('Candidate Qlib close/factor reconstruction requires --allow-unverified-price')
    if not np.isfinite(unit_multiplier) or unit_multiplier<=0: raise ValueError('Invalid share unit multiplier')
    if universe.duplicated(['datetime','instrument']).any(): raise ValueError('Duplicate universe keys')
    events=normalize_shares(events_raw,unit_multiplier); shares=build_share_panel(events,universe[['datetime','instrument']]); shares['instrument']=shares['instrument'].astype(str).str.zfill(6)
    price_parts=[]
    for year in sorted(universe['datetime'].dt.year.unique()):
        path=Path(root)/'data'/'processed'/f'csi300_daily_{year}.parquet'
        if not path.exists(): raise FileNotFoundError(path)
        d=pd.read_parquet(path,columns=['close','factor']).reset_index(); d['datetime']=pd.to_datetime(d['datetime']); d=d[d['datetime'].dt.year.eq(year)]; d['symbol']=d['instrument'].astype(str).str[-6:]; d['factor']=pd.to_numeric(d['factor'],errors='coerce'); d['close']=pd.to_numeric(d['close'],errors='coerce'); d['raw_close_candidate']=d['close'].div(d['factor'].where(d['factor']>0)); d.loc[~np.isfinite(d['raw_close_candidate']) | (d['raw_close_candidate']<=0),'raw_close_candidate']=np.nan; price_parts.append(d[['datetime','instrument','raw_close_candidate']])
    prices=pd.concat(price_parts,ignore_index=True)
    if prices.duplicated(['datetime','instrument']).any(): raise ValueError('Duplicate price keys')
    u=universe[['datetime','instrument','symbol']].copy(); u['datetime']=pd.to_datetime(u['datetime']); u=u.merge(prices,on=['datetime','instrument'],how='left',validate='one_to_one'); u=u.merge(shares.drop(columns=['datetime']).assign(datetime=shares['datetime']),left_on=['datetime','symbol'],right_on=['datetime','instrument'],how='left',validate='one_to_one',suffixes=('','_shares'))
    u['total_market_cap_candidate']=u['raw_close_candidate']*u['total_shares']; u['float_market_cap_candidate']=u['raw_close_candidate']*u['float_shares']; u['log_market_cap_candidate']=np.log(u['total_market_cap_candidate'].where(u['total_market_cap_candidate']>0)); u['price_status']='QLIB_CLOSE_DIV_FACTOR_UNVERIFIED'; u['share_unit_status']='USER_ASSUMPTION_UNVERIFIED'; u['membership_status']='OBSERVED_PROCESSED_ROWS_NOT_INDEPENDENTLY_PIT_CERTIFIED'; u['strict_pit_certified']=False
    return u.drop(columns=['instrument_shares'],errors='ignore').sort_values(['datetime','instrument']).reset_index(drop=True)


def main() -> None:
    """Description: Save an explicitly exploratory candidate market-cap exposure panel.

    Parameters:
        None: CLI options specify paths, share units, and unverified price acknowledgement.
    Returns:
        None: Writes a parquet file and summary report.
    Notes:
        Do not use the resulting panel as certified production risk exposures.
    """
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--universe',required=True); p.add_argument('--events',required=True); p.add_argument('--output',required=True); p.add_argument('--unit-multiplier',type=float,required=True); p.add_argument('--allow-unverified-price',action='store_true'); a=p.parse_args(); universe=pd.read_parquet(a.universe); events=pd.read_parquet(a.events); d=construct_candidate_exposures(universe,events,Path(a.root),a.unit_multiplier,a.allow_unverified_price); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); d.to_parquet(out/'market_cap_candidate.parquet',index=False); summary={'rows':len(d),'matched_shares':int(d['total_shares'].notna().sum()),'valid_candidate_market_cap':int(d['log_market_cap_candidate'].notna().sum()),'coverage':float(d['log_market_cap_candidate'].notna().mean()),'share_unit_multiplier_assumed':a.unit_multiplier,'strict_pit_certified':False,'price_certified':False}; (out/'market_cap_candidate_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
