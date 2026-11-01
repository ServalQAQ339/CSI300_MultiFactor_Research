"""Build calendar-aligned forward-return labels and audit adjusted Qlib prices."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    """Description: Immutable label configuration. Parameters: horizons (tuple[int]): holding periods; train_end, validation_end, test_end (str): split endpoints. Returns: LabelConfig: configuration. Notes: All offsets refer to a shared exchange calendar."""
    horizons: tuple = (1, 5, 10)
    train_end: str = '2020-12-31'
    validation_end: str = '2023-12-31'
    test_end: str = '2026-09-30'
    research_start: str = '2015-04-01'


def audit_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Description: Audit Qlib adjusted prices and factors without asserting point-in-time validity. Parameters: prices (pd.DataFrame): instrument, datetime, open, close, factor. Returns: tuple[pd.DataFrame, pd.DataFrame]: enriched prices and per-row issue records. Notes: Recovered raw prices are diagnostic only; independent raw-price comparison is required."""
    required = {'instrument', 'datetime', 'open', 'close', 'factor'}
    if not required.issubset(prices.columns):
        raise ValueError(f'Missing price fields: {sorted(required - set(prices.columns))}')
    p = prices.copy()
    p['datetime'] = pd.to_datetime(p['datetime']).dt.normalize()
    if p.duplicated(['instrument', 'datetime']).any():
        raise ValueError('Duplicate instrument-date observations')
    for field in ['open', 'close', 'factor']:
        p[field] = pd.to_numeric(p[field], errors='coerce').replace([np.inf, -np.inf], np.nan)
    issues = []
    for field in ['open', 'close', 'factor']:
        bad = p[field].isna() | (p[field] <= 0)
        if bad.any():
            q = p.loc[bad, ['instrument', 'datetime']].copy()
            q['issue'] = f'invalid_{field}'
            issues.append(q)
    p['raw_open_estimate'] = (p['open'] / p['factor']).where(p['factor'] > 0)
    p['raw_close_estimate'] = (p['close'] / p['factor']).where(p['factor'] > 0)
    p = p.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    p['factor_changed'] = p.groupby('instrument')['factor'].transform(
        lambda x: x.notna() & x.shift().notna() & ~np.isclose(x, x.shift(), rtol=1e-8, atol=1e-12)
    )
    issue_df = (
        pd.concat(issues, ignore_index=True)
        if issues
        else pd.DataFrame(columns=['instrument', 'datetime', 'issue'])
    )
    return p, issue_df


def build_labels(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    membership: pd.DataFrame,
    config: LabelConfig = LabelConfig(),
) -> pd.DataFrame:
    """Description: Construct forward open-to-open returns on a shared market calendar. Parameters: prices (pd.DataFrame): audited Qlib prices; calendar (pd.DatetimeIndex): complete exchange trading days through test end; membership (pd.DataFrame): instrument, datetime, eligible, with membership status known as of signal time; config (LabelConfig): research settings. Returns: pd.DataFrame: per-signal rows with labels, dates, missing reasons and split eligibility. Notes: Membership is signal-date only; missing exit prices are not forward-filled; no tradability claim is made."""
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
    if len(cal) == 0 or cal.has_duplicates:
        raise ValueError('A nonempty unique exchange calendar is required')
    required = {'instrument', 'datetime', 'open', 'factor'}
    if not required.issubset(prices.columns):
        raise ValueError(f'Missing price fields: {sorted(required - set(prices.columns))}')
    if not {'instrument', 'datetime', 'eligible'}.issubset(membership.columns):
        raise ValueError('Membership must contain instrument, datetime, eligible')
    p = prices.copy()
    p['datetime'] = pd.to_datetime(p['datetime']).dt.normalize()
    if p.duplicated(['instrument', 'datetime']).any():
        raise ValueError('Duplicate price keys')
    m = membership.copy()
    m['datetime'] = pd.to_datetime(m['datetime']).dt.normalize()
    if m.duplicated(['instrument', 'datetime']).any():
        raise ValueError('Duplicate membership keys')
    if m['eligible'].isna().any():
        raise ValueError('Unknown membership cannot be treated as eligible')
    if not m['datetime'].isin(cal).all() or not p['datetime'].isin(cal).all():
        raise ValueError('Input dates outside supplied exchange calendar')
    m = m.loc[
        m['eligible'].astype(bool) & m['datetime'].between(config.research_start, config.test_end)
    ].copy()
    m['calendar_pos'] = cal.get_indexer(m['datetime'])
    if (m['calendar_pos'] < 0).any():
        raise ValueError('Unmapped signal dates')
    p['open'] = pd.to_numeric(p['open'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    p['factor'] = pd.to_numeric(p['factor'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    p['valid_open'] = p['open'].gt(0) & p['factor'].gt(0)
    price_map = p.set_index(['instrument', 'datetime'])[['open', 'valid_open']]
    result = (
        m[['instrument', 'datetime', 'calendar_pos']]
        .rename(columns={'datetime': 'signal_date'})
        .copy()
    )
    result['split'] = np.select(
        [
            result['signal_date'].le(config.train_end),
            result['signal_date'].le(config.validation_end),
            result['signal_date'].le(config.test_end),
        ],
        ['train', 'validation', 'test'],
        default='outside',
    )
    end_map = {
        'train': pd.Timestamp(config.train_end),
        'validation': pd.Timestamp(config.validation_end),
        'test': pd.Timestamp(config.test_end),
    }
    for h in config.horizons:
        if not isinstance(h, int) or h < 1:
            raise ValueError('Horizons must be positive integers')
        entry_pos, exit_pos = (
            result['calendar_pos'].to_numpy() + 1,
            result['calendar_pos'].to_numpy() + h + 1,
        )
        entry_dates = pd.Series(pd.NaT, index=result.index, dtype='datetime64[ns]')
        exit_dates = pd.Series(pd.NaT, index=result.index, dtype='datetime64[ns]')
        valid_entry, valid_exit = entry_pos < len(cal), exit_pos < len(cal)
        entry_dates.loc[valid_entry] = cal.take(entry_pos[valid_entry]).to_numpy()
        exit_dates.loc[valid_exit] = cal.take(exit_pos[valid_exit]).to_numpy()
        result[f'entry_date_{h}d'], result[f'exit_date_{h}d'] = entry_dates, exit_dates
        entry_keys = pd.MultiIndex.from_arrays([result['instrument'], entry_dates])
        exit_keys = pd.MultiIndex.from_arrays([result['instrument'], exit_dates])
        entry_open = pd.Series(price_map['open'].reindex(entry_keys).to_numpy(), index=result.index)
        exit_open = pd.Series(price_map['open'].reindex(exit_keys).to_numpy(), index=result.index)
        entry_valid = pd.Series(
            price_map['valid_open'].reindex(entry_keys).eq(True).to_numpy(dtype=bool),
            index=result.index,
        )
        exit_valid = pd.Series(
            price_map['valid_open'].reindex(exit_keys).eq(True).to_numpy(dtype=bool),
            index=result.index,
        )
        boundary = pd.Series([end_map[s] for s in result['split']], index=result.index)
        reason = np.select(
            [~valid_exit, exit_dates.gt(boundary), ~valid_entry, ~entry_valid, ~exit_valid],
            [
                'calendar_end',
                'split_boundary',
                'missing_entry',
                'invalid_entry_price',
                'invalid_exit_price',
            ],
            default='ok',
        )
        label = (exit_open / entry_open - 1).where(reason == 'ok')
        result[f'label_{h}d'] = label
        result[f'label_status_{h}d'] = reason
        result[f'entry_open_adjusted_{h}d'] = entry_open
        result[f'exit_open_adjusted_{h}d'] = exit_open
    return (
        result.drop(columns='calendar_pos')
        .sort_values(['signal_date', 'instrument'])
        .reset_index(drop=True)
    )
