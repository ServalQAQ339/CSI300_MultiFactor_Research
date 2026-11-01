"""Build point-in-time daily technical factors from historical CSI300 OHLCV panels."""

from pathlib import Path
import numpy as np
import pandas as pd

PRICE_COLUMNS = ("open", "high", "low", "close")
REQUIRED_COLUMNS = (*PRICE_COLUMNS, "volume")
DEFAULT_WINDOWS = (5, 10, 20, 60)


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate and sanitize an OHLCV panel.

    Parameters:
        panel (pd.DataFrame): Panel indexed by (datetime, instrument) with OHLCV columns.

    Returns:
        pd.DataFrame: Sorted panel with invalid or nonfinite market values set to NaN.

    Notes:
        Nonpositive prices, negative volume and inconsistent OHLC are invalidated.
        Zero volume is preserved and not automatically classified as a suspension.
    """
    if not isinstance(panel.index, pd.MultiIndex) or set(panel.index.names) != {"datetime", "instrument"}:
        raise ValueError("Panel index must contain datetime and instrument levels")
    if panel.index.has_duplicates:
        raise ValueError("Duplicate (datetime, instrument) index detected")
    missing = set(REQUIRED_COLUMNS).difference(panel.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    
    df = panel.reorder_levels(["datetime", "instrument"]).sort_index().copy()
    
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    invalid_price = df[list(PRICE_COLUMNS)].le(0).any(axis=1) | df["high"].lt(df["low"]) | df["high"].lt(df[["open", "close"]].max(axis=1)) | df["low"].gt(df[["open", "close"]].min(axis=1))
    df.loc[invalid_price, list(PRICE_COLUMNS)] = np.nan
    df.loc[df["volume"].lt(0), "volume"] = np.nan
    
    return df


def calculate_stock_factors(stock: pd.DataFrame, windows: tuple[int, ...] = DEFAULT_WINDOWS) -> pd.DataFrame:
    """Compute past-only daily factors for one calendar-aligned instrument.

    Parameters:
        stock (pd.DataFrame): One instrument's OHLCV data indexed by market trading dates.
        windows (tuple[int, ...]): Positive lookback lengths measured in trading sessions.

    Returns:
        pd.DataFrame: Technical factor values with NaN for incomplete histories.

    Notes:
        Rolling operations require complete valid windows; missing dates remain NaN.
        All factors are available only after the corresponding trading day's close.
        These factors are an Alpha158-inspired subset, not an exact Alpha158 reproduction.
    """
    close, opening, high, low, volume = (stock[c] for c in ("close", "open", "high", "low", "volume"))
    
    result = pd.DataFrame(index=stock.index)
    result["OPEN_CLOSE_RATIO"] = opening.div(close).sub(1)
    result["HIGH_LOW_RANGE"] = high.sub(low).div(close)
    result["CLOSE_LOCATION"] = close.sub(low).div(high.sub(low).replace(0, np.nan))
    daily_return = close.div(close.shift(1)).sub(1)
    
    for window in windows:
        result[f"MOM_{window}"] = close.div(close.shift(window)).sub(1)
        result[f"MA_RATIO_{window}"] = close.div(close.rolling(window, min_periods=window).mean()).sub(1)
        result[f"VOLATILITY_{window}"] = daily_return.rolling(window, min_periods=window).std(ddof=1)
        result[f"VOLUME_RATIO_{window}"] = volume.div(volume.rolling(window, min_periods=window).mean().replace(0, np.nan))
        result[f"HIGH_POSITION_{window}"] = close.div(high.rolling(window, min_periods=window).max()).sub(1)
        result[f"LOW_POSITION_{window}"] = close.div(low.rolling(window, min_periods=window).min()).sub(1)
    
    return result.replace([np.inf, -np.inf], np.nan)


def build_factor_panel(panel: pd.DataFrame, calendar: pd.DatetimeIndex, output_start: str, output_end: str, windows: tuple[int, ...] = DEFAULT_WINDOWS) -> pd.DataFrame:
    """Compute factors on a complete exchange calendar and retain original universe rows.

    Parameters:
        panel (pd.DataFrame): Historical dynamic-universe OHLCV panel including warm-up history.
        calendar (pd.DatetimeIndex): Full exchange trading calendar covering the input period.
        output_start (str): First output date in YYYY-MM-DD format.
        output_end (str): Last output date in YYYY-MM-DD format.
        windows (tuple[int, ...]): Lookback lengths in exchange trading sessions.

    Returns:
        pd.DataFrame: Factor panel indexed by (datetime, instrument) on original input rows.

    Notes:
        Calendar alignment prevents rolling windows from silently skipping absent dates.
        Missing out-of-universe historical rows are not synthesized as tradable rows.
    """
    if not windows or any(not isinstance(w, int) or w < 2 for w in windows):
        raise ValueError("windows must contain integers >= 2")
    
    df = validate_panel(panel)
    
    calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    if calendar.empty or not df.index.get_level_values("datetime").isin(calendar).all():
        raise ValueError("Calendar must contain all input trading dates")
    
    outputs = []
    for instrument, stock in df.groupby(level="instrument", sort=False):
        stock = stock.droplevel("instrument").reindex(calendar)
        features = calculate_stock_factors(stock, windows)
        features["instrument"] = instrument
        outputs.append(features.loc[output_start:output_end].reset_index().rename(columns={"index": "datetime"}))
    if not outputs:
        return pd.DataFrame()
    
    factors = pd.concat(outputs, ignore_index=True).set_index(["datetime", "instrument"]).sort_index()
    original_rows = df.loc[(slice(output_start, output_end), slice(None)), :].index
    
    return factors.reindex(original_rows).sort_index()


def run_factor_engine(data_dir: Path, output_dir: Path, calendar: pd.DatetimeIndex, start_year: int, end_year: int, windows: tuple[int, ...] = DEFAULT_WINDOWS, overwrite: bool = False) -> pd.DataFrame:
    """Generate annual factor Parquet files with cross-year historical warm-up.

    Parameters:
        data_dir (Path): Directory containing csi300_daily_YYYY.parquet files.
        output_dir (Path): Destination for csi300_factors_YYYY.parquet files.
        calendar (pd.DatetimeIndex): Exchange calendar covering all available source years.
        start_year (int): First target output year.
        end_year (int): Last target output year.
        windows (tuple[int, ...]): Factor lookback windows in market trading sessions.
        overwrite (bool): Whether to regenerate existing factor files.

    Returns:
        pd.DataFrame: Yearly output summary with row counts and factor coverage.

    Notes:
        Historical warm-up uses at least max(windows)+1 prior market sessions.
        Historical data before the first cached year cannot be reconstructed here.
        Earlier membership gaps may cause NaNs even when prices exist elsewhere.
    """
    calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = []
    for year in range(start_year, end_year + 1):
        output_file = output_dir / f"csi300_factors_{year}.parquet"
        
        if output_file.exists() and not overwrite:
            summary.append({"year": year, "status": "cached", "path": str(output_file)})
            continue
        target_file = Path(data_dir) / f"csi300_daily_{year}.parquet"
        
        if not target_file.exists():
            raise FileNotFoundError(target_file)
        
        target = pd.read_parquet(target_file)
        target_dates = target.index.get_level_values("datetime")
        first_date, last_date = target_dates.min(), target_dates.max()
        first_pos = calendar.get_indexer([first_date])[0]
        
        if first_pos < 0:
            raise ValueError(f"Calendar does not include {first_date}")
        warmup_start = calendar[max(0, first_pos - max(windows) - 1)]
        source_files = [Path(data_dir) / f"csi300_daily_{source_year}.parquet" for source_year in range(warmup_start.year, year + 1)]
        missing_files = [str(path) for path in source_files if not path.exists()]
        
        if missing_files:
            raise FileNotFoundError(f"Missing warm-up files: {missing_files}")
        
        source = pd.concat([pd.read_parquet(path) for path in source_files]).sort_index()
        source = source.loc[(slice(warmup_start, last_date), slice(None)), :]
        factors = build_factor_panel(source, calendar[(calendar >= warmup_start) & (calendar <= last_date)], str(first_date.date()), str(last_date.date()), windows)
        factors.to_parquet(output_file, engine="pyarrow", compression="snappy")
        summary.append({"year": year, "status": "generated", "rows": len(factors), "mean_factor_coverage": float(factors.notna().mean().mean()), "path": str(output_file)})
        print(f"Generated factors for {year}: {len(factors):,} rows", flush=True)
    
    return pd.DataFrame(summary)
