"""Compare fixed Ridge and LightGBM factor models on matched stock-date samples."""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from factors.expansion.statistics import cross_sectional_ranks, hac_summary
from factor_research.selection import TRAIN, VALIDATION, load_panel


BASE_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 100,
    "colsample_bytree": 0.85,
    "subsample": 1.0,
    "reg_lambda": 10.0,
    "n_jobs": 4,
    "random_state": 42,
    "verbosity": -1,
}


def selected_features(selection_file: str, available: list[str]) -> list[str]:
    """Read the frozen Stage 4 training-selected feature list.

    Description:
        Require selected flags and preserve selection-file order.
    Parameters:
        selection_file (str): Stage 4 train_factor_selection.csv path.
        available (list[str]): Candidate columns in the factor panel.
    Returns:
        list[str]: Validated expansion feature names.
    Notes:
        Do not reselect features using Validation performance.
    """
    frame = pd.read_csv(selection_file)
    if not {"factor", "selected"}.issubset(frame.columns):
        raise ValueError("Selection file must contain factor and selected columns")
    flags = frame["selected"].astype(str).str.lower().isin(["true", "1"])
    names = frame.loc[flags, "factor"].tolist()
    if not names or len(names) != len(set(names)):
        raise ValueError("Selected features must be nonempty and unique")
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"Selected features absent from candidate data: {missing}")
    return names


def prepare_samples(panel: pd.DataFrame, baseline: list[str], selected: list[str],
                    min_stocks: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct identical finite-feature samples for all three models.

    Description:
        Train rows require complete features and a valid label. Validation
        prediction rows require complete features only; label_5d is carried
        along (possibly NaN) for ex-post evaluation.
    Parameters:
        panel (pd.DataFrame): Joined factor and label panel, with label_5d set to
            NaN for invalid statuses (load_panel(..., require_valid_label=False)).
        baseline (list[str]): Original27 feature names.
        selected (list[str]): Frozen additional feature names.
        min_stocks (int): Minimum cross-sectional observations per date.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and Validation panels.
    Notes:
        Training labels must be mature before Validation starts; the 5-day
        horizon is conservatively purged using the last five Train dates.
        Validation membership never depends on future label availability; the
        Validation min_stocks rule counts feature-complete (predictable) stocks.
    """
    features = baseline + selected
    columns = features + ["label_5d"]
    dates = panel.index.get_level_values("datetime")
    train = panel.loc[(dates >= TRAIN[0]) & (dates <= TRAIN[1]), columns].dropna()
    validation = panel.loc[(dates >= VALIDATION[0]) & (dates <= VALIDATION[1]), columns]
    validation = validation.dropna(subset=features)
    if train.empty or validation.empty:
        raise ValueError("No complete-case rows in Train or Validation")

    # Purge the last five observed Train dates to avoid label-horizon overlap.
    train_dates = train.index.get_level_values("datetime").unique().sort_values()
    if len(train_dates) <= 5:
        raise ValueError("Insufficient training dates for five-day purge")
    train = train.loc[train.index.get_level_values("datetime") < train_dates[-5]]
    train = train.groupby(level="datetime").filter(lambda x: len(x) >= min_stocks)
    validation = validation.groupby(level="datetime").filter(lambda x: len(x) >= min_stocks)
    if train.empty or validation.empty:
        raise ValueError("Insufficient stocks after common-sample filtering")
    return train, validation


def fit_predict_models(train: pd.DataFrame, validation: pd.DataFrame,
                       baseline: list[str], selected: list[str]) -> pd.DataFrame:
    """Fit three fixed-specification models and return Validation scores.

    Description:
        Fit Ridge Original27, LightGBM Original27 and LightGBM Extended.
    Parameters:
        train (pd.DataFrame): Purged training complete-case panel.
        validation (pd.DataFrame): Validation feature-complete panel; label_5d may be NaN.
        baseline (list[str]): Original27 feature names.
        selected (list[str]): Train-selected expansion feature names.
    Returns:
        pd.DataFrame: Three scores and the future-return label (NaN if invalid)
            for every predictable Validation observation.
    Notes:
        No early stopping, Validation-based tuning or future-date fitting occurs.
        Validation feature ranks use the full same-day predictable cross-section,
        so scores do not depend on future label validity.
    """
    features = baseline + selected
    x_train = cross_sectional_ranks(train, features)
    x_validation = cross_sectional_ranks(validation, features)
    y_train = cross_sectional_ranks(train, ["label_5d"])["label_5d"]

    ridge = Ridge(alpha=100.0).fit(x_train[baseline], y_train)
    lgb_base = LGBMRegressor(**BASE_PARAMS).fit(x_train[baseline], y_train)
    lgb_extended = LGBMRegressor(**BASE_PARAMS).fit(x_train[features], y_train)

    predictions = pd.DataFrame(index=validation.index)
    predictions["ridge_original27"] = ridge.predict(x_validation[baseline])
    predictions["lightgbm_original27"] = lgb_base.predict(x_validation[baseline])
    predictions["lightgbm_extended"] = lgb_extended.predict(x_validation[features])
    predictions["label_5d"] = validation["label_5d"]
    return predictions


def evaluate_predictions(predictions: pd.DataFrame, min_stocks: int = 30
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate matched daily Rank IC, rank MSE and paired HAC inference.

    Description:
        After scores exist, keep rows with a valid ex-post label, rank scores and
        labels within that subsample by date and compare both model differences.
    Parameters:
        predictions (pd.DataFrame): Same-row model predictions and labels; label_5d
            is NaN where the realized label is invalid.
        min_stocks (int): Minimum valid-label stocks required for a daily correlation.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Daily IC/MSE and HAC summary tables.
            Daily n_predicted counts all scored rows; n_stocks counts the evaluated
            valid-label rows.
    Notes:
        Daily IC differences use identical observations for all models. Rows with
        missing labels are excluded from evaluation only, never from prediction.
        mse_<model> compares scores with centered percentile label ranks, the
        training target scale.
    """
    names = ["ridge_original27", "lightgbm_original27", "lightgbm_extended"]
    rows = []
    for date, day in predictions.groupby(level="datetime", sort=True):
        labeled = day.dropna(subset=["label_5d"])
        if len(labeled) < min_stocks:
            continue
        ranked = labeled[names + ["label_5d"]].rank()
        target = labeled["label_5d"].rank(pct=True) - 0.5
        row = {"datetime": date, "n_predicted": len(day), "n_stocks": len(labeled)}
        for name in names:
            row[name] = ranked[name].corr(ranked["label_5d"])
            row[f"mse_{name}"] = ((labeled[name] - target) ** 2).mean()
        rows.append(row)
    if not rows:
        raise ValueError("No eligible Validation dates")
    daily = pd.DataFrame(rows).set_index("datetime")
    daily["delta_nonlinear"] = daily["lightgbm_original27"] - daily["ridge_original27"]
    daily["delta_expansion"] = daily["lightgbm_extended"] - daily["lightgbm_original27"]
    summary = hac_summary(daily[names + ["delta_nonlinear", "delta_expansion"]], lags=5, fdr=False)
    return daily, summary


def run_modeling(data_dir: str, candidate_file: str, selection_file: str,
                 output_dir: str) -> None:
    """Execute the fixed three-model experiment and save concise outputs.

    Description:
        Reuse Stage 4 data loading and selection, then evaluate matched models.
    Parameters:
        data_dir (str): Research data directory, i.e. <project root>/data (not the project root).
        candidate_file (str): Existing factor expansion Parquet file.
        selection_file (str): Stage 4 Train-selected feature CSV file.
        output_dir (str): New report directory for Stage 5 results.
    Returns:
        None: Writes full Validation predictions, daily IC, summary and metadata.
    Notes:
        Validation has been inspected before; results are not blind-test claims.
        Labels are loaded without row filtering so that the prediction universe
        never depends on future label validity.
    """
    panel, baseline, candidates = load_panel(data_dir, candidate_file, require_valid_label=False)
    chosen = selected_features(selection_file, candidates)
    train, validation = prepare_samples(panel, baseline, chosen)
    predictions = fit_predict_models(train, validation, baseline, chosen)
    daily, summary = evaluate_predictions(predictions)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(destination / "validation_predictions.parquet")
    daily.to_csv(destination / "validation_daily_ic.csv")
    summary.to_csv(destination / "validation_model_summary.csv", index=False)
    pd.DataFrame({"selected_feature": chosen}).to_csv(
        destination / "frozen_selected_features.csv", index=False
    )
    labeled = predictions["label_5d"].notna().sum()
    print(f"Train rows (purged): {len(train):,}; Validation predicted rows: "
          f"{len(predictions):,}; with valid labels: {labeled:,}")
    print(f"Selected expansion features: {len(chosen)}; Validation days: {len(daily)}")
    print(summary.to_string(index=False))
