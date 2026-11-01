"""Load Qlib bars and align frozen baseline with forward labels."""

from pathlib import Path

import pandas as pd


TRAIN = ("2015-04-01", "2020-12-31")
VALIDATION = ("2021-01-01", "2023-12-31")


def load_bars(provider: str, start: str, end: str) -> pd.DataFrame:
    """Load OHLCV from the local Qlib provider.

    Description:
        Retrieve historical CSI300 observations with an explicit warmup period.
    Parameters:
        provider (str): Local Qlib binary provider path.
        start (str): Earliest requested trading date.
        end (str): Latest requested trading date.
    Returns:
        pd.DataFrame: OHLCV panel indexed by datetime and instrument.
    Notes:
        Qlib csi300 universe membership requires separate PIT verification.
    """
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=provider, region="cn")
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    bars = D.features(D.instruments("csi300"), fields, start_time=start, end_time=end)
    bars = bars.rename(columns=dict(zip(fields, ["open", "high", "low", "close", "volume"])))
    if set(bars.index.names) != {"datetime", "instrument"} or not bars.index.is_unique:
        raise ValueError("Unexpected or duplicated Qlib keys")
    return bars.reorder_levels(["datetime", "instrument"]).sort_index()


def load_original27(data_dir: str, years: range | None = None) -> pd.DataFrame:
    """Load the frozen Original27 raw factor panel from annual Parquet files.

    Description:
        Concatenate annual csi300_factors_YYYY files and sort by (datetime, instrument).
    Parameters:
        data_dir (str): Research data directory, i.e. <project root>/data.
        years (range | None): Explicit years to read; a missing year raises. None reads
            every csi300_factors_20??.parquet present and raises only if none exist.
    Returns:
        pd.DataFrame: Original27 factors indexed by datetime and instrument.
    Notes:
        Key uniqueness is checked by callers, which differ in their validation rules.
    """
    folder = Path(data_dir) / "factors_preprocessed" / "raw"
    if years is None:
        files = sorted(folder.glob("csi300_factors_20??.parquet"))
        if not files:
            raise FileNotFoundError("No Original27 annual files found")
    else:
        files = [folder / f"csi300_factors_{year}.parquet" for year in years]
    return pd.concat([pd.read_parquet(file) for file in files]).sort_index()


def load_forward_labels(data_dir: str, required: set[str]) -> pd.DataFrame:
    """Load forward labels keyed by signal date and instrument.

    Description:
        Read labels/forward_labels.parquet and index it as (datetime, instrument).
    Parameters:
        data_dir (str): Research data directory, i.e. <project root>/data.
        required (set[str]): Columns that must exist, including signal_date and instrument.
    Returns:
        pd.DataFrame: Labels with a unique (datetime, instrument) index, where datetime
            is the signal date.
    Notes:
        No status filtering is applied here; callers filter label_status_5d explicitly.
    """
    labels = pd.read_parquet(Path(data_dir) / "labels" / "forward_labels.parquet")
    if not required.issubset(labels.columns):
        raise ValueError(f"Missing label fields: {sorted(required - set(labels.columns))}")
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels = labels.set_index(["signal_date", "instrument"])
    labels.index.names = ["datetime", "instrument"]
    if not labels.index.is_unique:
        raise ValueError("Duplicate label keys")
    return labels


def load_research_panel(data_dir: str, candidates: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Align all factor observations to the fixed Original27 baseline.

    Description:
        Join annual baseline factors, candidates and status-filtered labels.
    Parameters:
        data_dir (str): Research data directory, i.e. <project root>/data (not the project root).
        candidates (pd.DataFrame): Candidate features with datetime/instrument index.
    Returns:
        tuple[pd.DataFrame, list[str]]: Aligned valid panel and baseline columns.
    Notes:
        Labels must have explicit signal_date, instrument and status fields.
    """
    original = load_original27(data_dir, range(2015, 2027))
    if not original.index.is_unique or not candidates.index.is_unique:
        raise ValueError("Duplicate factor keys")
    baseline = original.columns.tolist()
    if set(baseline) & set(candidates.columns):
        raise ValueError("Candidate/baseline column collision")
    labels = load_forward_labels(
        data_dir, {"signal_date", "instrument", "label_5d", "label_status_5d", "split"}
    )
    panel = original.join(candidates, how="left", validate="one_to_one")
    panel = panel.join(labels[["label_5d", "label_status_5d", "split"]], how="left", validate="one_to_one")
    panel = panel.loc[panel["label_status_5d"].eq("ok") & panel["label_5d"].notna()].copy()
    return panel, baseline
