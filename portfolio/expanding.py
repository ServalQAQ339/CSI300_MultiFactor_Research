"""Quarterly expanding-window model fitting and diagnostic next-open backtesting."""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from modeling.modeling import BASE_PARAMS, selected_features


MODELS = ("ridge_original27", "lightgbm_original27", "lightgbm_extended")


def mature_train_mask(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex,
                      first_signal: pd.Timestamp, horizon: int = 6) -> np.ndarray:
    """Identify labels fully realized before the model's first prediction.

    Description:
        Require the open-price label endpoint at signal position plus horizon
        to precede the first forecast signal date, conservatively.
    Parameters:
        dates (pd.DatetimeIndex): Training signal dates.
        calendar (pd.DatetimeIndex): Ordered trading calendar including label exits.
        first_signal (pd.Timestamp): First signal date of the forecast quarter.
        horizon (int): Trading-session distance from signal to label exit.
    Returns:
        np.ndarray: Boolean eligibility mask for mature labels.
    Notes:
        A signal at t uses open(t+1) through open(t+6); endpoint must be
        earlier than first_signal. No extra embargo is applied.
    """
    positions = calendar.get_indexer(dates)
    cutoff = calendar.searchsorted(first_signal)
    return (positions >= 0) & (positions + horizon < cutoff)


def quarterly_scores(panel: pd.DataFrame, baseline: list[str], chosen: list[str],
                     calendar: pd.DatetimeIndex, start: str = "2024-01-01",
                     end: str = "2026-09-30") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit fixed models on expanding mature-label histories each quarter.

    Description:
        Freeze model specifications and features, then refit three models at
        each quarter boundary and score same-day cross-sectional feature ranks.
    Parameters:
        panel (pd.DataFrame): Original27, candidates and valid 5-day labels.
        baseline (list[str]): Original27 feature names.
        chosen (list[str]): Frozen Train-only selected candidate names.
        calendar (pd.DatetimeIndex): Ordered trading sessions.
        start (str): First diagnostic signal date.
        end (str): Last diagnostic signal date.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Model scores and quarterly fit metadata.
    Notes:
        Test labels are not required for prediction. Complete-case filtering
        uses only signal-day features; label availability affects fitting only.
    """
    features = baseline + chosen
    frame = panel[features + ["label_5d"]].copy()
    dates = frame.index.get_level_values("datetime")
    target = frame.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    quarters = target.index.get_level_values("datetime").to_period("Q")
    scores, records = [], []

    for quarter in sorted(quarters.unique()):
        forecast = target.loc[quarters == quarter, features].dropna()
        if forecast.empty:
            continue
        first = forecast.index.get_level_values("datetime").min()
        historical = frame.loc[(frame.index.get_level_values("datetime") >= pd.Timestamp("2015-04-01")) & (frame.index.get_level_values("datetime") < first)].dropna()
        hist_dates = historical.index.get_level_values("datetime")
        historical = historical.loc[mature_train_mask(hist_dates, calendar, first)]
        if historical.empty:
            raise ValueError(f"No mature training labels for {quarter}")

        x_train = historical.groupby(level="datetime")[features].rank(pct=True) - 0.5
        y_train = historical.groupby(level="datetime")["label_5d"].rank(pct=True) - 0.5
        x_forecast = forecast.groupby(level="datetime")[features].rank(pct=True) - 0.5
        ridge = Ridge(alpha=100.0).fit(x_train[baseline], y_train)
        lgb_base = LGBMRegressor(**BASE_PARAMS).fit(x_train[baseline], y_train)
        lgb_extended = LGBMRegressor(**BASE_PARAMS).fit(x_train[features], y_train)

        output = pd.DataFrame(index=forecast.index)
        output[MODELS[0]] = ridge.predict(x_forecast[baseline])
        output[MODELS[1]] = lgb_base.predict(x_forecast[baseline])
        output[MODELS[2]] = lgb_extended.predict(x_forecast[features])
        scores.append(output)
        records.append({"quarter": str(quarter), "first_signal": first,
                        "last_train_signal": historical.index.get_level_values("datetime").max(),
                        "train_rows": len(historical), "forecast_rows": len(forecast)})
        print(f"{quarter}: train={len(historical):,}; forecast={len(forecast):,}", flush=True)

    if not scores:
        raise ValueError("No forecast dates in requested range")
    return pd.concat(scores).sort_index(), pd.DataFrame(records)


def simulate(scores: pd.DataFrame, opens: pd.DataFrame, index_open: pd.Series,
             top_k: int = 30, holding: int = 5, cost_bps: float = 10.0
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate next-open non-overlapping portfolios without exit-date screening.

    Description:
        Select stocks using signal scores only, then check execution prices.
        Missing selected-stock entry or exit prices invalidate the whole period.
    Parameters:
        scores (pd.DataFrame): Model scores indexed by signal date and instrument.
        opens (pd.DataFrame): Stock open prices indexed by date and instrument.
        index_open (pd.Series): SH000300 open prices indexed by date.
        top_k (int): Number of equally weighted names.
        holding (int): Open-to-open holding intervals.
        cost_bps (float): Assumed one-way cost in basis points.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Period returns and annualized summary.
    Notes:
        No future exit-price availability is used in stock selection. Missing
        selected exit prices cause a reported skipped period, not silent removal.
        Tradability, limit-up/down, and adjusted-price execution remain unverified.
    """
    if top_k < 1 or holding < 1 or cost_bps < 0:
        raise ValueError("Invalid portfolio parameters")
    if scores.index.has_duplicates or opens.index.has_duplicates:
        raise ValueError("Duplicate stock-date keys")
    calendar = index_open.index.sort_values().unique()
    positions = {date: i for i, date in enumerate(calendar)}
    rows = []
    next_position = 0
    for date, group in scores.groupby(level="datetime", sort=True):
        pos = positions.get(date)
        if pos is None or pos < next_position or pos + holding + 1 >= len(calendar):
            continue
        entry, exit_date = calendar[pos + 1], calendar[pos + holding + 1]
        entry_px = opens.xs(entry, level="datetime")["open"] if entry in opens.index.get_level_values("datetime") else pd.Series(dtype=float)
        exit_px = opens.xs(exit_date, level="datetime")["open"] if exit_date in opens.index.get_level_values("datetime") else pd.Series(dtype=float)
        ranked = group.droplevel("datetime")
        record = {"signal_date": date, "entry_date": entry, "exit_date": exit_date,
                  "benchmark": index_open.loc[exit_date] / index_open.loc[entry] - 1}
        valid_period = True
        for model in MODELS:
            names = ranked[model].dropna().nlargest(top_k).index
            if len(names) != top_k:
                valid_period = False
                break
            initial = entry_px.reindex(names)
            final = exit_px.reindex(names)
            if initial.isna().any() or final.isna().any() or (initial <= 0).any() or (final <= 0).any():
                valid_period = False
                break
            gross = (final / initial - 1).mean()
            record[model] = (1 + gross) * (1 - cost_bps / 10000) ** 2 - 1
        if valid_period:
            rows.append(record)
        next_position = pos + holding
    if not rows:
        raise ValueError("No complete executable periods; inspect missing prices")
    periods = pd.DataFrame(rows).set_index("signal_date")
    annual_periods = 252 / holding
    summary = []
    benchmark = periods["benchmark"]
    for model in MODELS:
        returns = periods[model]
        relative = (1 + returns).cumprod() / (1 + benchmark).cumprod()
        summary.append({"model": model, "periods": len(periods),
                        "strategy_annualized": (1 + returns).prod() ** (annual_periods / len(returns)) - 1,
                        "benchmark_annualized": (1 + benchmark).prod() ** (annual_periods / len(benchmark)) - 1,
                        "annualized_return_difference": (1 + returns).prod() ** (annual_periods / len(returns)) - (1 + benchmark).prod() ** (annual_periods / len(benchmark)),
                        "relative_wealth_final": relative.iloc[-1]})
    return periods, pd.DataFrame(summary)


def run(data_dir: str, candidates: str, selection: str, provider: str,
        output: str, top_k: int = 30, cost_bps: float = 10.0) -> None:
    """Run the quarterly diagnostic using existing data and frozen models.

    Description:
        Load factor panel, fetch prices and calendar, score quarters, simulate.
    Parameters:
        data_dir (str): Research data directory, i.e. <project root>/data (not the project root).
        candidates (str): Existing candidate parquet path.
        selection (str): Frozen Train-only selection CSV.
        provider (str): Qlib binary data directory.
        output (str): New report directory.
        top_k (int): Number of stocks selected per rebalance.
        cost_bps (float): Assumed one-way transaction cost.
    Returns:
        None: Writes predictions, quarterly metadata and period reports.
    Notes:
        Historical membership and executable pricing are not PIT-certified.
    """
    import qlib
    from qlib.data import D

    root = Path(data_dir)
    files = sorted((root / "factors_preprocessed" / "raw").glob("csi300_factors_20??.parquet"))
    original = pd.concat([pd.read_parquet(path) for path in files]).sort_index()
    expansion = pd.read_parquet(candidates).sort_index()
    baseline = original.columns.tolist()
    available = expansion.columns.tolist()
    chosen = selected_features(selection, available)
    labels = pd.read_parquet(root / "labels" / "forward_labels.parquet")
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels = labels.set_index(["signal_date", "instrument"])
    labels.index.names = ["datetime", "instrument"]
    if any(not item.index.is_unique for item in (original, expansion, labels)):
        raise ValueError("Duplicate stock-date keys")
    panel = original.join(expansion[chosen], how="left")
    panel = panel.join(labels[["label_5d", "label_status_5d"]], how="left")
    panel.loc[~panel["label_status_5d"].eq("ok"), "label_5d"] = np.nan
    panel = panel.replace([np.inf, -np.inf], np.nan)
    qlib.init(provider_uri=provider, kernels=1)
    index_frame = D.features(["SH000300"], ["$open"], start_time="2015-01-01",
                             end_time="2026-11-30")
    index_open = index_frame["$open"].droplevel("instrument").sort_index().dropna()
    if index_open.empty:
        raise ValueError("No SH000300 benchmark data")
    scores, metadata = quarterly_scores(panel, baseline, chosen, index_open.index)
    from portfolio.backtest import load_open_prices

    names = scores.index.get_level_values("instrument").unique().tolist()
    opens = load_open_prices(provider, names, "2023-12-01", "2026-11-30")
    periods, summary = simulate(scores, opens, index_open, top_k, 5, cost_bps)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(destination / "diagnostic_scores.parquet")
    metadata.to_csv(destination / "quarterly_fits.csv", index=False)
    periods.to_csv(destination / "diagnostic_period_returns.csv")
    summary.to_csv(destination / "diagnostic_summary.csv", index=False)
    print(summary.to_string(index=False))
