"""Retrieve historical OHLCV for eligible CSI300 instruments without changing membership."""

import numpy as np
import pandas as pd

FIELDS = ("$open", "$high", "$low", "$close", "$volume")


def load_extended_history(
    target: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    lookback: int = 61,
    batch_size: int = 80,
    fetcher=None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Load historical prices for all instruments eligible during a target year.

    Description:
        Fetch each target instrument's OHLCV from the earliest required session
        through the last target session, irrespective of its prior index membership.

    Parameters:
        target (pd.DataFrame): Historical eligible panel indexed by (datetime, instrument).
        calendar (pd.DatetimeIndex): Exchange sessions including sufficient prehistory.
        lookback (int): Number of prior exchange sessions needed by factor formulas.
        batch_size (int): Number of instruments fetched per Qlib request.
        fetcher (callable | None): Optional test double with D.features signature.

    Returns:
        tuple[pd.DataFrame, pd.DatetimeIndex]: Extended OHLCV panel and the exact
            calendar window used for computation.

    Notes:
        Eligibility is never inferred from the returned price history. Qlib price
        adjustment may be retrospective and needs a separate point-in-time audit.
        Missing or nonpositive market values are left to the factor engine to handle.
    """
    if (
        not isinstance(target.index, pd.MultiIndex)
        or set(target.index.names) != {"datetime", "instrument"}
        or target.empty
    ):
        raise ValueError("Target must have nonempty (datetime, instrument) index")
    if lookback < 1 or batch_size < 1:
        raise ValueError("lookback and batch_size must be positive")
    calendar = pd.DatetimeIndex(calendar).sort_values().unique()
    target_dates = pd.DatetimeIndex(target.index.get_level_values("datetime"))
    first, last = target_dates.min(), target_dates.max()
    first_pos = calendar.get_indexer([first])[0]
    if first_pos < 0 or last not in calendar:
        raise ValueError("Target dates must exist in exchange calendar")
    start = calendar[max(0, first_pos - lookback)]
    window_calendar = calendar[(calendar >= start) & (calendar <= last)]
    instruments = sorted(target.index.get_level_values("instrument").unique().tolist())
    if fetcher is None:
        from qlib.data import D
        fetch = D.features
    else:
        fetch = fetcher
    parts = []
    for offset in range(0, len(instruments), batch_size):
        batch = instruments[offset:offset + batch_size]
        frame = fetch(
            instruments=batch,
            fields=list(FIELDS),
            start_time=str(start.date()),
            end_time=str(last.date()),
            freq="day",
        )
        if frame is None or frame.empty:
            continue
        if not isinstance(frame.index, pd.MultiIndex) or set(frame.index.names) != {
            "datetime",
            "instrument",
        }:
            raise ValueError("Qlib returned an unexpected index")
        frame = (
            frame.rename(columns={name: name[1:] for name in FIELDS})
            .reorder_levels(["datetime", "instrument"])
            .sort_index()
        )
        frame = frame.loc[frame.index.get_level_values("datetime").isin(window_calendar)]
        frame = frame.loc[frame.index.get_level_values("instrument").isin(batch)]
        parts.append(frame)
    if not parts:
        raise ValueError("No extended OHLCV history was returned")
    history = pd.concat(parts).sort_index()
    if history.index.has_duplicates:
        raise ValueError("Duplicate historical OHLCV rows returned")
    return history, window_calendar
