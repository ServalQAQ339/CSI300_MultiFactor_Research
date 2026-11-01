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


def load_research_panel(data_root: str, candidates: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Align all factor observations to the fixed Original27 baseline.

    Description:
        Join annual baseline factors, candidates and status-filtered labels.
    Parameters:
        data_root (str): Research data directory on D drive.
        candidates (pd.DataFrame): Candidate features with datetime/instrument index.
    Returns:
        tuple[pd.DataFrame, list[str]]: Aligned valid panel and baseline columns.
    Notes:
        Labels must have explicit signal_date, instrument and status fields.
    """
    root = Path(data_root)
    original = pd.concat([pd.read_parquet(root / f"factors_preprocessed/raw/csi300_factors_{year}.parquet") for year in range(2015, 2027)])
    original = original.sort_index()
    if not original.index.is_unique or not candidates.index.is_unique:
        raise ValueError("Duplicate factor keys")
    baseline = original.columns.tolist()
    if set(baseline) & set(candidates.columns):
        raise ValueError("Candidate/baseline column collision")
    labels = pd.read_parquet(root / "labels/forward_labels.parquet")
    required = {"signal_date", "instrument", "label_5d", "label_status_5d", "split"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Missing label fields: {sorted(required - set(labels.columns))}")
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels = labels.set_index(["signal_date", "instrument"])
    labels.index.names = ["datetime", "instrument"]
    if not labels.index.is_unique:
        raise ValueError("Duplicate label keys")
    panel = original.join(candidates, how="left", validate="one_to_one")
    panel = panel.join(labels[["label_5d", "label_status_5d", "split"]], how="left", validate="one_to_one")
    panel = panel.loc[panel["label_status_5d"].eq("ok") & panel["label_5d"].notna()].copy()
    return panel, baseline
