"""Construct conservative as-known share histories and market capitalization panels."""
import numpy as np
import pandas as pd


def normalize_shares(raw: pd.DataFrame, unit_multiplier: float) -> pd.DataFrame:
    """Description: Normalize CNINFO share events without inferring undisclosed units.

    Parameters:
        raw (pd.DataFrame): CNINFO share history containing announcement, effective, reason and share columns.
        unit_multiplier (float): Explicit verified conversion from reported units to shares.
    Returns:
        pd.DataFrame: Cleaned event table with dates, share counts and availability dates.
    Notes:
        Conservative next-calendar-day availability is a proxy, not verified intraday PIT.
        Historical periodic-report snapshots are available only after disclosure and must not be treated as new corporate actions.
    """
    required = ['证券代码', '公告日期', '变动日期', '变动原因', '总股本', '已流通股份']
    if any(c not in raw.columns for c in required):
        raise ValueError(f'Missing required columns: {set(required)-set(raw.columns)}')
    if not np.isfinite(unit_multiplier) or unit_multiplier <= 0:
        raise ValueError('unit_multiplier must be positive and explicitly verified')
    d = (
        raw[required]
        .copy()
        .rename(
            columns={
                '证券代码': 'instrument',
                '公告日期': 'announcement_date',
                '变动日期': 'effective_date',
                '变动原因': 'event_type',
                '总股本': 'total_shares',
                '已流通股份': 'float_shares',
            }
        )
    )
    d['instrument'] = d['instrument'].astype('string').str.zfill(6)
    for c in ['announcement_date','effective_date']: d[c] = pd.to_datetime(d[c], errors='coerce')
    for c in ['total_shares', 'float_shares']:
        d[c] = pd.to_numeric(d[c], errors='coerce') * unit_multiplier
    d['available_date'] = d[['announcement_date', 'effective_date']].max(axis=1) + pd.Timedelta(
        days=1
    )
    d['is_periodic_snapshot'] = d['event_type'].astype('string').str.contains('定期报告',na=False)
    d.loc[d['announcement_date'].isna() | d['effective_date'].isna(), 'available_date'] = pd.NaT
    d.loc[d['total_shares'] <= 0, 'total_shares'] = np.nan
    d.loc[d['float_shares'] < 0, 'float_shares'] = np.nan
    d['pit_status'] = 'CONSERVATIVE_DATE_PROXY_UNVERIFIED'
    return d.sort_values(['instrument','available_date','effective_date']).reset_index(drop=True)


def build_share_panel(events: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """Description: As-of join share disclosures to signal dates, without future record leakage.

    Parameters:
        events (pd.DataFrame): Output of normalize_shares.
        signals (pd.DataFrame): Table with instrument and datetime columns.
    Returns:
        pd.DataFrame: Signal rows with most recently available disclosed share values.
    Notes:
        Coverage prior to the first available event is intentionally missing.
        For equal availability dates, the latest effective date is selected deterministically.
    """
    s = signals.copy()
    s['instrument'] = (
        s['instrument']
        .astype('string')
        .str.replace(r'^(SH|SZ|BJ)', '', regex=True, case=False)
        .str.zfill(6)
    )
    s['datetime'] = pd.to_datetime(s['datetime'])
    e = (
        events.dropna(subset=['available_date'])
        .sort_values(['instrument', 'available_date', 'effective_date'])
        .drop_duplicates(['instrument', 'available_date'], keep='last')
    )
    parts = []
    for code, group in s.groupby('instrument',sort=False):
        g = group.sort_values('datetime')
        h = e[e['instrument'] == code].drop(columns='instrument').sort_values('available_date')
        if h.empty:
            for c in [
                'total_shares',
                'float_shares',
                'available_date',
                'announcement_date',
                'effective_date',
                'pit_status',
            ]:
                g[c] = pd.NaT if c.endswith('date') else np.nan
            parts.append(g)
        else:
            parts.append(
                pd.merge_asof(
                    g, h, left_on='datetime', right_on='available_date', direction='backward'
                )
            )
    return pd.concat(parts,ignore_index=True) if parts else s


def build_market_cap_panel(shares: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Description: Construct market capitalization from externally verified unadjusted prices and share counts.

    Parameters:
        shares (pd.DataFrame): Signal-level share panel with total_shares and float_shares.
        prices (pd.DataFrame): Signal-level prices with instrument, datetime and raw_close.
    Returns:
        pd.DataFrame: Market capitalization and log-size fields.
    Notes:
        The function deliberately does not infer Qlib adjustment formulas or price units.
    """
    keys = ['instrument', 'datetime']
    p = prices[keys + ['raw_close']].copy()
    p['datetime'] = pd.to_datetime(p['datetime'])
    d = shares.merge(p, on=keys, how='left', validate='one_to_one')
    d['raw_close'] = pd.to_numeric(d['raw_close'], errors='coerce')
    d.loc[d['raw_close'] <= 0, 'raw_close'] = np.nan
    d['total_market_cap'] = d['raw_close'] * d['total_shares']
    d['float_market_cap'] = d['raw_close'] * d['float_shares']
    d['log_market_cap'] = np.log(d['total_market_cap'].where(d['total_market_cap']>0))
    return d


def normalize_industry(raw: pd.DataFrame) -> pd.DataFrame:
    """Description: Preserve historical SW classifications as retrospective, non-PIT-certified events.

    Parameters:
        raw (pd.DataFrame): SW history with stock code, inclusion date, industry code and update date.
    Returns:
        pd.DataFrame: Normalized dated industry records with explicit retrospective risk flag.
    Notes:
        Update dates are not assumed to be first-publication dates.
    """
    cols = {'股票代码':'instrument','计入日期':'effective_date','行业代码':'industry_code','更新日期':'update_date'}
    if not set(cols).issubset(raw.columns): raise ValueError('Missing SW industry columns')
    d = raw[list(cols)].rename(columns=cols).copy()
    d['instrument'] = d['instrument'].astype('string').str.zfill(6)
    for c in ['effective_date','update_date']: d[c] = pd.to_datetime(d[c],errors='coerce')
    d['pit_status'] = 'RETROSPECTIVE_INDUSTRY_UNVERIFIED_PIT'
    return d
