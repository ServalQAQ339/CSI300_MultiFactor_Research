"""Build exploratory historical market-cap exposures with disclosed-share provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def normalize_events(raw: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """Description: Normalize disclosed CNINFO shares and conservative availability dates.

    Parameters:
        raw (pd.DataFrame): Original CNINFO event table with Chinese column names.
        multiplier (float): Explicit assumed conversion from reported share units to shares.
    Returns:
        pd.DataFrame: Normalized events with source dates, quantities, and snapshot flags.
    Notes:
        A next-calendar-day rule is a conservative date-only proxy, not PIT certification.
    """
    required=['证券代码','公告日期','变动日期','变动原因','总股本','已流通股份']
    if not set(required).issubset(raw.columns): raise ValueError(f'Missing columns: {set(required)-set(raw.columns)}')
    if not np.isfinite(multiplier) or multiplier<=0: raise ValueError('Positive explicit multiplier required')
    e=raw[required].copy().rename(columns={'证券代码':'symbol','公告日期':'announcement_date','变动日期':'effective_date','变动原因':'reason','总股本':'total_shares','已流通股份':'float_shares'})
    e['symbol']=e['symbol'].astype('string').str.replace(r'^(SH|SZ|BJ)','',regex=True,case=False).str.zfill(6)
    for c in ['announcement_date','effective_date']: e[c]=pd.to_datetime(e[c],errors='coerce').dt.normalize()
    for c in ['total_shares','float_shares']: e[c]=pd.to_numeric(e[c],errors='coerce')*multiplier
    e['snapshot']=e['reason'].astype('string').str.contains('定期报告',na=False)
    e['usable_date']=e[['announcement_date','effective_date']].max(axis=1)+pd.Timedelta(days=1)
    e.loc[e[['announcement_date','effective_date']].isna().any(axis=1),'usable_date']=pd.NaT
    e.loc[e['total_shares']<=0,'total_shares']=np.nan
    e.loc[e['float_shares']<0,'float_shares']=np.nan
    return e


def asof_shares(universe: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Description: Match each observed stock-date with latest known eligible effective share state.

    Parameters:
        universe (pd.DataFrame): Unique datetime and instrument rows.
        events (pd.DataFrame): Normalized events from normalize_events.
    Returns:
        pd.DataFrame: Observations with as-known share counts and source dates.
    Notes:
        Superseded historical snapshots cannot overwrite a later effective-date state.
        Same-effective-date revisions remain exploratory; no original-publication archive exists.
    """
    u=universe[['datetime','instrument']].copy(); u['datetime']=pd.to_datetime(u['datetime']).dt.normalize(); u['symbol']=u['instrument'].astype('string').str[-6:]
    if u.duplicated(['datetime','instrument']).any(): raise ValueError('Duplicate observation keys')
    e=events.dropna(subset=['usable_date','effective_date','total_shares']).copy()
    fields=['total_shares','float_shares','announcement_date','effective_date','usable_date','snapshot','reason']
    result=[]
    for symbol,g in u.groupby('symbol',sort=False):
        h=e.loc[e['symbol'].eq(symbol)].sort_values(['usable_date','effective_date'],kind='stable'); dates=g['datetime'].sort_values().unique(); records=[]; active=None; j=0; rows=h.to_dict('records')
        for dt in dates:
            while j<len(rows) and rows[j]['usable_date']<=dt:
                candidate=rows[j]; j+=1
                if active is None or candidate['effective_date']>active['effective_date'] or (candidate['effective_date']==active['effective_date'] and candidate['usable_date']>=active['usable_date']): active=candidate
            records.append({'datetime':dt,**({k:active.get(k) for k in fields} if active is not None else {k:pd.NaT if k.endswith('date') else np.nan for k in fields})})
        state=pd.DataFrame(records); result.append(g.merge(state,on='datetime',how='left',validate='many_to_one'))
    return pd.concat(result,ignore_index=True) if result else u


def build_candidate(universe: pd.DataFrame, events: pd.DataFrame, root: Path, multiplier: float) -> pd.DataFrame:
    """Description: Create explicitly unverified candidate total and float market capitalization.

    Parameters:
        universe (pd.DataFrame): Observed instrument-date panel.
        events (pd.DataFrame): Raw CNINFO share records.
        root (Path): Research root containing processed annual Qlib prices.
        multiplier (float): Assumed shares-per-reported-unit conversion.
    Returns:
        pd.DataFrame: Candidate size exposures with audit flags and event provenance.
    Notes:
        Qlib close/factor is an unverified price reconstruction, not certified raw price.
    """
    s=asof_shares(universe,normalize_events(events,multiplier)); prices=[]
    for year in sorted(s['datetime'].dt.year.unique()):
        path=root/'data'/'processed'/f'csi300_daily_{year}.parquet'; d=pd.read_parquet(path,columns=['close','factor']).reset_index(); d['datetime']=pd.to_datetime(d['datetime']).dt.normalize(); d=d[d['datetime'].dt.year.eq(year)]; d['close']=pd.to_numeric(d['close'],errors='coerce'); d['factor']=pd.to_numeric(d['factor'],errors='coerce'); d['raw_close_candidate']=d['close']/d['factor'].where(d['factor']>0); prices.append(d[['datetime','instrument','raw_close_candidate']])
    p=pd.concat(prices,ignore_index=True)
    if p.duplicated(['datetime','instrument']).any(): raise ValueError('Duplicate price keys')
    d=s.merge(p,on=['datetime','instrument'],how='left',validate='one_to_one'); d.loc[~np.isfinite(d['raw_close_candidate']) | (d['raw_close_candidate']<=0),'raw_close_candidate']=np.nan
    d['total_market_cap_candidate']=d['raw_close_candidate']*d['total_shares']; d['float_market_cap_candidate']=d['raw_close_candidate']*d['float_shares']; d['log_market_cap_candidate']=np.log(d['total_market_cap_candidate'].where(d['total_market_cap_candidate']>0)); d['strict_pit_certified']=False; d['price_certified']=False; d['share_unit_certified']=False
    return d.sort_values(['datetime','instrument']).reset_index(drop=True)


def main() -> None:
    """Description: Run local exploratory exposure construction and write an audit summary.

    Parameters:
        None: Inputs are specified using command-line arguments.
    Returns:
        None: Saves candidate parquet and JSON report under a new output directory.
    Notes:
        No network calls are made; outputs must not be labeled PIT-certified.
    """
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--universe',required=True); p.add_argument('--events',required=True); p.add_argument('--output',required=True); p.add_argument('--unit-multiplier',type=float,required=True); p.add_argument('--acknowledge-unverified',action='store_true'); a=p.parse_args()
    if not a.acknowledge_unverified: p.error('Exploratory reconstruction requires --acknowledge-unverified')
    d=build_candidate(pd.read_parquet(a.universe),pd.read_parquet(a.events),Path(a.root),a.unit_multiplier); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); d.to_parquet(out/'market_cap_candidate_v04.parquet',index=False)
    summary={'rows':len(d),'matched_shares':int(d['total_shares'].notna().sum()),'matched_candidate_market_cap':int(d['log_market_cap_candidate'].notna().sum()),'coverage':float(d['log_market_cap_candidate'].notna().mean()),'assumed_share_multiplier':a.unit_multiplier,'strict_pit_certified':False,'price_certified':False,'share_unit_certified':False}; (out/'market_cap_candidate_v04_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
