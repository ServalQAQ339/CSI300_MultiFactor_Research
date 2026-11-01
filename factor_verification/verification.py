"""Independently recompute original 27 technical factors and audit stored values."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20, 60)
BASE = ('OPEN_CLOSE_RATIO', 'HIGH_LOW_RANGE', 'CLOSE_LOCATION')
FAMILIES = ('MOM', 'MA_RATIO', 'VOLATILITY', 'VOLUME_RATIO', 'HIGH_POSITION', 'LOW_POSITION')
FACTORS = BASE + tuple(f'{family}_{w}' for w in WINDOWS for family in FAMILIES)
KEYS = ['datetime', 'instrument']


def normalize_panel(frame: pd.DataFrame, required: tuple[str, ...]) -> pd.DataFrame:
    """Description: Normalize a panel to a unique datetime/instrument MultiIndex.

    Parameters:
        frame (pd.DataFrame): DataFrame with key columns or named MultiIndex.
        required (tuple[str, ...]): Required non-key column names.
    Returns:
        pd.DataFrame: Date-normalized, key-sorted panel.
    Notes:
        Duplicate keys and missing required columns raise errors; no rows are silently removed.
    """
    df = frame.reset_index() if isinstance(frame.index, pd.MultiIndex) else frame.copy()
    missing = set(KEYS + list(required)) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    df['datetime'] = pd.to_datetime(df['datetime'], errors='raise').dt.normalize()
    df['instrument'] = df['instrument'].astype(str)
    if df.duplicated(KEYS).any():
        raise ValueError(f'Duplicate panel keys: {int(df.duplicated(KEYS).sum())}')
    return df.set_index(KEYS).sort_index()


def independent_factors(stock: pd.DataFrame) -> pd.DataFrame:
    """Description: Independently compute the 27 source-defined factors from one calendar-aligned OHLCV series.

    Parameters:
        stock (pd.DataFrame): Chronologically indexed OHLCV panel for one instrument.
    Returns:
        pd.DataFrame: 27 numeric factor columns indexed by exchange sessions.
    Notes:
        Replicates the uploaded source's formulas, ddof=1, complete windows, and invalid-price rules without importing factor_engine.
    """
    x = (
        stock[['open', 'high', 'low', 'close', 'volume']]
        .apply(pd.to_numeric, errors='coerce')
        .replace([np.inf, -np.inf], np.nan)
        .copy()
    )

    bad = (
        x[['open', 'high', 'low', 'close']].le(0).any(axis=1)
        | x.high.lt(x.low)
        | x.high.lt(x[['open', 'close']].max(axis=1))
        | x.low.gt(x[['open', 'close']].min(axis=1))
    )

    x.loc[bad, ['open', 'high', 'low', 'close']] = np.nan
    x.loc[x.volume.lt(0), 'volume'] = np.nan
    c, o, h, l, v = x.close, x.open, x.high, x.low, x.volume
    out = pd.DataFrame(index=x.index)
    out['OPEN_CLOSE_RATIO'] = o / c - 1
    out['HIGH_LOW_RANGE'] = (h - l) / c
    out['CLOSE_LOCATION'] = (c - l) / (h - l).replace(0, np.nan)
    r = c / c.shift(1) - 1

    for w in WINDOWS:
        out[f'MOM_{w}'] = c / c.shift(w) - 1
        out[f'MA_RATIO_{w}'] = c / c.rolling(w, min_periods=w).mean() - 1
        out[f'VOLATILITY_{w}'] = r.rolling(w, min_periods=w).std(ddof=1)
        out[f'VOLUME_RATIO_{w}'] = v / v.rolling(w, min_periods=w).mean().replace(0, np.nan)
        out[f'HIGH_POSITION_{w}'] = c / h.rolling(w, min_periods=w).max() - 1
        out[f'LOW_POSITION_{w}'] = c / l.rolling(w, min_periods=w).min() - 1

    return out.replace([np.inf, -np.inf], np.nan)[list(FACTORS)]


def recompute_panel(
    source: pd.DataFrame, calendar: pd.DatetimeIndex, target_keys: pd.MultiIndex
) -> pd.DataFrame:
    """Description: Recompute factors using a common exchange calendar and retain exactly requested keys.

    Parameters:
        source (pd.DataFrame): Normalized OHLCV panel including historical warm-up.
        calendar (pd.DatetimeIndex): Complete exchange sessions for source dates.
        target_keys (pd.MultiIndex): Stored-factor keys to verify.
    Returns:
        pd.DataFrame: Recomputed factor panel on target keys, with missing source rows preserved as NaN.
    Notes:
        Never fills missing OHLCV with future information.
    """
    parts = []
    instruments = target_keys.get_level_values('instrument').unique()

    for asset in instruments:
        if asset not in source.index.get_level_values('instrument'):
            continue
        stock = source.xs(asset, level='instrument').reindex(calendar)
        calc = independent_factors(stock)
        requested = target_keys[
            target_keys.get_level_values('instrument') == asset
        ].get_level_values('datetime')
        calc = calc.reindex(requested)
        calc['instrument'] = asset
        parts.append(calc.reset_index().rename(columns={'index': 'datetime'}).set_index(KEYS))

    return (
        pd.concat(parts).reindex(target_keys)
        if parts
        else pd.DataFrame(index=target_keys, columns=list(FACTORS), dtype=float)
    )


def compare_values(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    atol: float = 1e-8,
    rtol: float = 1e-7,
    max_examples: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Description: Compare values with separate missingness and numeric discrepancy accounting.

    Parameters:
        expected (pd.DataFrame): Independently recomputed factor panel.
        actual (pd.DataFrame): Stored factor panel with same key index.
        atol (float): Absolute numeric tolerance.
        rtol (float): Relative numeric tolerance.
        max_examples (int): Maximum discrepancy examples retained across factors.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Per-factor audit summary and discrepancy examples.
    Notes:
        Missingness mismatches always fail; both-NaN values are considered consistent. No stored data is altered.
    """
    if not expected.index.equals(actual.index):
        raise ValueError('Key mismatch: expected and stored indexes differ')
    rows, examples = [], []

    for factor in FACTORS:
        a = expected[factor].to_numpy(dtype=float)
        b = actual[factor].to_numpy(dtype=float)
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        missing_mismatch = finite_a ^ finite_b
        both = finite_a & finite_b
        diff = np.full(len(a), np.nan)
        diff[both] = np.abs(a[both] - b[both])
        numeric_fail = both & ~np.isclose(a, b, atol=atol, rtol=rtol)
        failed = missing_mismatch | numeric_fail
        rows.append(
            {
                'factor': factor,
                'rows': len(a),
                'both_finite': int(both.sum()),
                'both_missing': int((~finite_a & ~finite_b).sum()),
                'missing_mismatch': int(missing_mismatch.sum()),
                'numeric_fail': int(numeric_fail.sum()),
                'failed': int(failed.sum()),
                'max_abs_diff': float(np.nanmax(diff)) if both.any() else None,
            }
        )

        for j in np.flatnonzero(failed)[:max(0, max_examples - len(examples))]:
            dt, asset = expected.index[j]
            examples.append(
                {
                    'datetime': str(dt.date()),
                    'instrument': asset,
                    'factor': factor,
                    'recomputed': float(a[j]) if finite_a[j] else None,
                    'stored': float(b[j]) if finite_b[j] else None,
                    'abs_diff': float(diff[j]) if both[j] else None,
                    'reason': 'missing_mismatch' if missing_mismatch[j] else 'numeric_mismatch',
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(
        examples,
        columns=['datetime', 'instrument', 'factor', 'recomputed', 'stored', 'abs_diff', 'reason'],
    )


def load_years(
    folder: Path, template: str, years: range, required: tuple[str, ...]
) -> pd.DataFrame:
    """Description: Load and normalize annual Parquet panels, failing on missing files.

    Parameters:
        folder (Path): Directory containing yearly files.
        template (str): Filename pattern containing {year}.
        years (range): Inclusive set of years to load.
        required (tuple[str, ...]): Required data columns.
    Returns:
        pd.DataFrame: Concatenated unique-key panel.
    Notes:
        No automatic fallback to alternative datasets or price adjustment versions.
    """
    frames = []

    for year in years:
        path = folder / template.format(year=year)
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(normalize_panel(pd.read_parquet(path), required))

    result = pd.concat(frames).sort_index()
    if result.index.has_duplicates:
        raise ValueError('Duplicate keys across annual files')

    return result


def run_audit(
    ohlcv_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    calendar_file: Path | None = None,
    enhanced_dir: Path | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-7,
) -> dict:
    """Description: Independently verify stored annual Raw factors against historical OHLCV and optionally compare Enhanced values.

    Parameters:
        ohlcv_dir (Path): Directory with csi300_daily_YYYY.parquet OHLCV files.
        raw_dir (Path): Directory with csi300_factors_YYYY.parquet Raw files.
        output_dir (Path): New audit report directory, which must not already exist.
        calendar_file (Path | None): Optional verified exchange calendar text file with YYYY-MM-DD dates.
        enhanced_dir (Path | None): Optional Enhanced directory for overlap-only comparisons, not assumed identical by default.
        atol (float): Numeric absolute tolerance.
        rtol (float): Numeric relative tolerance.
    Returns:
        dict: Audit status, row counts, failure counts, and limitations.
    Notes:
        Requires OHLCV source matching factor-generation vintage; if no verified calendar is supplied, calendar completeness is unverified.
    """
    if output_dir.exists():
        raise FileExistsError(f'Output exists; choose a new directory: {output_dir}')
    source = load_years(
        ohlcv_dir,
        'csi300_daily_{year}.parquet',
        range(2015, 2027),
        ('open', 'high', 'low', 'close', 'volume'),
    )

    raw = load_years(raw_dir, 'csi300_factors_{year}.parquet', range(2015, 2027), FACTORS)

    if calendar_file:
        calendar = (
            pd.DatetimeIndex(pd.to_datetime(pd.read_csv(calendar_file, header=None).iloc[:, 0]))
            .normalize()
            .sort_values()
            .unique()
        )
        if not source.index.get_level_values('datetime').isin(calendar).all():
            raise ValueError('Verified calendar omits source dates')
        calendar_status = 'provided_calendar_not_independently_certified'
    else:
        calendar = source.index.get_level_values('datetime').unique().sort_values()
        calendar_status = 'inferred_from_source_dates_incomplete_exchange_calendar_possible'

    expected = recompute_panel(source, calendar, raw.index)
    summary, examples = compare_values(expected, raw, atol, rtol)
    missing_source_keys = int((~raw.index.isin(source.index)).sum())
    report = {
        'status': (
            'PASS'
            if int(summary.failed.sum()) == 0 and missing_source_keys == 0
            else 'FAIL_REQUIRES_ATTRIBUTION'
        ),
        'raw_rows': len(raw),
        'source_rows': len(source),
        'factors': len(FACTORS),
        'total_numeric_fail': int(summary.numeric_fail.sum()),
        'total_missing_mismatch': int(summary.missing_mismatch.sum()),
        'stored_keys_without_ohlcv': missing_source_keys,
        'calendar_status': calendar_status,
        'atol': atol,
        'rtol': rtol,
        'limitations': [
            'Source OHLCV version and adjustment lineage not independently authenticated',
            'Historical universe membership PIT not independently certified',
            'Enhanced values may differ by design and are not presumed equal to Raw',
        ],
    }

    output_dir.mkdir(parents=True)

    summary.to_csv(output_dir / 'factor_value_summary.csv', index=False)
    examples.to_csv(output_dir / 'factor_value_discrepancies.csv', index=False)

    if enhanced_dir:
        enhanced = load_years(
            enhanced_dir, 'csi300_factors_{year}.parquet', range(2015, 2027), FACTORS
        )
        common = raw.index.intersection(enhanced.index)
        enhanced_summary, enhanced_examples = compare_values(
            raw.loc[common, list(FACTORS)], enhanced.loc[common, list(FACTORS)], atol, rtol
        )

        enhanced_summary.to_csv(output_dir / 'raw_enhanced_overlap_summary.csv', index=False)
        enhanced_examples.to_csv(output_dir / 'raw_enhanced_overlap_examples.csv', index=False)

        report['raw_enhanced_overlap_rows'] = len(common)
        report['raw_enhanced_difference_count'] = int(enhanced_summary.failed.sum())

    (output_dir / 'factor_value_audit.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    return report
