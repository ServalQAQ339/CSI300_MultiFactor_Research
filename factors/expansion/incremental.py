"""Evaluate conditional ranks and out-of-sample nested ridge-model IC."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .statistics import cross_sectional_ranks, daily_rank_ic, hac_summary


def conditional_rank_ic(panel: pd.DataFrame, baseline: list[str], candidates: list[str], alpha: float = 10.0, min_stocks: int = 50) -> pd.DataFrame:
    """Estimate daily conditional rank correlations.

    Description:
        Residualize candidate and label ranks against baseline ranks daily.
    Parameters:
        panel (pd.DataFrame): Valid aligned panel for a specified split.
        baseline (list[str]): Original27 predictors.
        candidates (list[str]): Candidate predictors.
        alpha (float): Ridge regularization for residualization.
        min_stocks (int): Minimum complete observations per date.
    Returns:
        pd.DataFrame: Daily partial rank correlation matrix.
    Notes:
        Descriptive conditional association, not a tradeable prediction model.
    """
    records = []
    for date, day in panel.groupby(level="datetime"):
        row = {"datetime": date}
        for candidate in candidates:
            subset = day[baseline + [candidate, "label_5d"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(subset) < min_stocks:
                row[candidate] = np.nan
                continue
            ranked = subset.rank(pct=True)
            x = ranked[baseline].to_numpy()
            y = ranked["label_5d"].to_numpy()
            z = ranked[candidate].to_numpy()
            model_y = Ridge(alpha=alpha).fit(x, y)
            model_z = Ridge(alpha=alpha).fit(x, z)
            ry = y - model_y.predict(x)
            rz = z - model_z.predict(x)
            row[candidate] = np.corrcoef(ry, rz)[0, 1] if ry.std() > 0 and rz.std() > 0 else np.nan
        records.append(row)
    return pd.DataFrame(records).set_index("datetime")


def nested_ridge_ic(train: pd.DataFrame, validation: pd.DataFrame, baseline: list[str], candidates: list[str], alpha: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare baseline and expanded ridge predictions on identical samples.

    Description:
        Fit models using Train only and evaluate Validation daily Rank IC.
    Parameters:
        train (pd.DataFrame): Valid Train observations.
        validation (pd.DataFrame): Valid Validation observations.
        baseline (list[str]): Frozen Original27 columns.
        candidates (list[str]): Incremental candidate set.
        alpha (float): Fixed ridge regularization parameter.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Daily IC and HAC delta summary.
    Notes:
        All predictors use within-date percentile ranks; complete-case rows
        are identical across the two models. No validation label is fitted.
    """
    features = baseline + candidates
    training = train[features + ["label_5d"]].replace([np.inf, -np.inf], np.nan).dropna()
    testing = validation[features + ["label_5d"]].replace([np.inf, -np.inf], np.nan).dropna()
    if training.empty or testing.empty:
        raise ValueError("No complete-case observations for nested comparison")
    x_train = cross_sectional_ranks(training, features)
    x_test = cross_sectional_ranks(testing, features)
    y_train = cross_sectional_ranks(training, ["label_5d"])["label_5d"]
    base = Ridge(alpha=alpha).fit(x_train[baseline], y_train)
    extended = Ridge(alpha=alpha).fit(x_train[features], y_train)
    predictions = pd.DataFrame(index=testing.index)
    predictions["baseline"] = base.predict(x_test[baseline])
    predictions["extended"] = extended.predict(x_test[features])
    predictions["label_5d"] = testing["label_5d"]
    daily = daily_rank_ic(predictions, ["baseline", "extended"])
    daily["delta_ic"] = daily["extended"] - daily["baseline"]
    summary = hac_summary(daily[["baseline", "extended", "delta_ic"]])
    return daily, summary
