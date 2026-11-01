from pathlib import Path
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

from config import QLIB_DATA_DIR, START_DATE, END_DATE, DATA_OUTPUT_DIR


def initialize_qlib() -> None:
    """
    Initialize the Qlib data provider using the configured local dataset.

    Parameters:
        None.

    Returns:
        None: Initializes the global Qlib data environment.
    """
    if not QLIB_DATA_DIR.exists():
        raise FileNotFoundError(f"Qlib data directory not found: {QLIB_DATA_DIR}")

    qlib.init(provider_uri=str(QLIB_DATA_DIR), region=REG_CN, kernels=1)


def get_trading_calendar(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Retrieve the trading calendar within the specified research period.

    Parameters:
        start_date (str): First date of the requested period, formatted as YYYY-MM-DD.
        end_date (str): Last date of the requested period, formatted as YYYY-MM-DD.

    Returns:
        pd.DatetimeIndex: Chronologically sorted trading dates.
    """
    calendar = D.calendar(start_time=start_date, end_time=end_date, freq="day")
    return pd.DatetimeIndex(calendar)


def load_annual_panel(year: int) -> pd.DataFrame:
    """
    Load daily OHLCV data for the historical CSI300 constituent universe.

    Parameters:
        year (int): Calendar year to retrieve.

    Returns:
        pd.DataFrame: Daily stock panel indexed by (datetime, instrument), containing
        open, high, low, close, volume, and factor.

    Notes:
        The historical CSI300 universe is resolved by Qlib for the requested period.
        Prices may be adjusted or normalized and must not be treated as execution prices.
    """
    start_date = max(f"{year}-01-01", START_DATE)
    end_date = min(f"{year}-12-31", END_DATE)

    if start_date > end_date:
        return pd.DataFrame()

    universe = D.instruments(market="csi300")
    fields = ["$open", "$high", "$low", "$close", "$volume", "$factor"]

    df = D.features(
        instruments=universe, fields=fields, start_time=start_date, end_time=end_date, freq="day"
    )

    df = df.rename(columns={field: field.replace("$", "") for field in fields})
    df = df.reorder_levels(["datetime", "instrument"]).sort_index()

    return df


def save_annual_panel(df: pd.DataFrame, year: int) -> Path:
    """
    Save one annual stock panel to a compressed Parquet file.

    Parameters:
        df (pd.DataFrame): Stock panel with datetime and instrument index levels.
        year (int): Calendar year corresponding to the dataset.

    Returns:
        pathlib.Path: Absolute path to the saved Parquet file.
    """
    DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_OUTPUT_DIR / f"csi300_daily_{year}.parquet"
    df.to_parquet(output_path, engine="pyarrow", compression="snappy")
    return output_path


def build_daily_dataset(overwrite: bool = False) -> pd.DataFrame:
    """
    Build annual CSI300 stock panels and save them as Parquet files.

    Parameters:
        overwrite (bool): Whether to overwrite existing annual datasets.
            False skips existing files to reduce repeated extraction.

    Returns:
        pd.DataFrame: Annual extraction summary containing year, row count,
            unique stock count, date range, and output file path.

    Notes:
        Each annual dataset is loaded independently to limit memory usage.
        Existing files are reused only when overwrite is False.
    """
    summary = []

    for year in range(int(START_DATE[:4]), int(END_DATE[:4]) + 1):
        output_path = DATA_OUTPUT_DIR / f"csi300_daily_{year}.parquet"

        if output_path.exists() and not overwrite:
            df = pd.read_parquet(output_path)
            print(f"Using cached data: {year}", flush=True)
        else:
            print(f"Downloading Qlib data: {year}", flush=True)
            df = load_annual_panel(year)

            if df.empty:
                print(f"Warning: No observations found for {year}", flush=True)
                continue

            save_annual_panel(df, year)

        summary.append(
            {
                "year": year,
                "rows": len(df),
                "stocks": df.index.get_level_values("instrument").nunique(),
                "first_date": df.index.get_level_values("datetime").min(),
                "last_date": df.index.get_level_values("datetime").max(),
                "output_path": str(output_path),
            }
        )
        print(f"Completed {year}: {len(df):,} observations", flush=True)
        del df

    return pd.DataFrame(summary)
