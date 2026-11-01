from pathlib import Path
import numpy as np
import pandas as pd

from config import DATA_OUTPUT_DIR, AUDIT_OUTPUT_DIR


def audit_annual_panel(df: pd.DataFrame, year: int) -> dict:
    """
    Evaluate the structural integrity and basic quality of an annual stock panel.

    Parameters:
        df (pd.DataFrame): Daily stock panel indexed by (datetime, instrument).
        year (int): Calendar year associated with the panel.

    Returns:
        dict: Data quality metrics including duplicate records, missing values,
        nonpositive prices, invalid OHLC relationships, and coverage statistics.

    Notes:
        OHLC checks assume prices share a consistent adjustment basis.
        Zero volume is flagged for investigation rather than automatically removed.
    """
    result = {
        "year": year,
        "rows": len(df),
        "unique_stocks": df.index.get_level_values("instrument").nunique(),
        "unique_dates": df.index.get_level_values("datetime").nunique(),
    }
    result["duplicate_index"] = int(df.index.duplicated().sum())

    for col in ["open", "high", "low", "close", "volume", "factor"]:
        result[f"missing_{col}"] = float(df[col].isna().mean())
        result[f"infinite_{col}"] = int(np.isinf(df[col].to_numpy()).sum())

    prices = df[["open", "high", "low", "close"]]
    result["nonpositive_price_rows"] = int((prices <= 0).any(axis=1).sum())
    result["negative_volume_rows"] = int((df["volume"] < 0).sum())
    result["zero_volume_rows"] = int((df["volume"] == 0).sum())
    result["invalid_high_low_rows"] = int((df["high"] < df["low"]).sum())
    result["invalid_ohlc_rows"] = int(
        (
            (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
        ).sum()
    )

    return result


def run_data_audit() -> pd.DataFrame:
    """
    Audit all locally cached annual CSI300 panels and export a summary report.

    Parameters:
        None.

    Returns:
        pd.DataFrame: Annual data quality summary with one row per cached year.
    """
    files = sorted(DATA_OUTPUT_DIR.glob("csi300_daily_*.parquet"))

    if not files:
        raise FileNotFoundError(f"No annual Parquet files found in {DATA_OUTPUT_DIR}")

    results = []

    for file in files:
        year = int(file.stem.split("_")[-1])
        df = pd.read_parquet(file)
        results.append(audit_annual_panel(df, year))
        print(f"Audited {year}", flush=True)
        del df

    report = pd.DataFrame(results).sort_values("year")
    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(AUDIT_OUTPUT_DIR / "annual_data_audit.csv", index=False)

    return report


def diagnose_missing_patterns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Identify stock-level missing price patterns within an annual CSI300 panel.

    Parameters:
        df (pd.DataFrame): Daily stock panel indexed by (datetime, instrument).
        year (int): Calendar year associated with the dataset.

    Returns:
        pd.DataFrame: Stock-level diagnostic report containing observation counts,
            missing price counts, missing ratios, and the longest consecutive
            missing observation sequence.

    Notes:
        Consecutive missing observations are measured within each stock's
        available panel rows. Missing dates absent from the panel are not counted.
        Missing prices are not automatically classified as suspensions.
    """
    results = []

    for instrument, stock_df in df.groupby(level="instrument"):
        stock_df = stock_df.sort_index()
        missing = stock_df["close"].isna()
        groups = missing.ne(missing.shift()).cumsum()
        longest_missing = int(missing.groupby(groups).sum().max()) if len(missing) else 0

        results.append(
            {
                "year": year,
                "instrument": instrument,
                "observations": len(stock_df),
                "missing_count": int(missing.sum()),
                "missing_ratio": float(missing.mean()),
                "longest_missing_sequence": longest_missing,
            }
        )

    return pd.DataFrame(results).sort_values(["missing_ratio", "missing_count"], ascending=False)
