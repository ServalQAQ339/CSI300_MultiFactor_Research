"""Select incremental features on Train only and evaluate matched Ridge models."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from factors.expansion.data import load_forward_labels, load_original27
from factors.expansion.statistics import cross_sectional_ranks, daily_rank_ic, hac_summary


TRAIN = ("2015-04-01", "2020-12-31")
VALIDATION = ("2021-01-01", "2023-12-31")


def load_panel(data_dir: str, candidate_file: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load frozen baseline, expansion features and valid forward labels.

    Description:
        Join unique stock-date observations and filter invalid label statuses.
    Parameters:
        data_dir (str): Research data directory, i.e. <project root>/data (not the project root).
        candidate_file (str): Path to the existing candidate parquet file.
    Returns:
        tuple: Joined panel, baseline names and candidate names.
    Notes:
        Historical universe PIT status is not independently certified.
    """
    baseline_frame = load_original27(data_dir)
    candidates_frame = pd.read_parquet(candidate_file).sort_index()
    baseline = baseline_frame.columns.tolist()
    candidates = [col for col in candidates_frame if col not in baseline]
    if set(baseline) & set(candidates_frame.columns):
        raise ValueError("Baseline and candidate names overlap")
    if not baseline_frame.index.is_unique or not candidates_frame.index.is_unique:
        raise ValueError("Duplicate factor keys")
    if baseline_frame.index.names != ["datetime", "instrument"]:
        raise ValueError("Unexpected baseline index names")
    if candidates_frame.index.names != ["datetime", "instrument"]:
        raise ValueError("Unexpected candidate index names")

    labels = load_forward_labels(
        data_dir, {"signal_date", "instrument", "label_5d", "label_status_5d"}
    )

    panel = baseline_frame.join(candidates_frame[candidates], how="left", validate="one_to_one")
    panel = panel.join(labels[["label_5d", "label_status_5d"]], how="left", validate="one_to_one")
    panel = panel.loc[panel["label_status_5d"].eq("ok") & panel["label_5d"].notna()]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel, baseline, candidates


def _pooled_rank_corr(train: pd.DataFrame, first: str, second: str) -> float:
    """Pooled Train correlation of datewise percentile ranks for one factor pair.

    Description:
        Keep finite pairs, rank both factors within each date on that paired sample,
        then correlate the pooled ranks across all Train dates.
    Parameters:
        train (pd.DataFrame): Training-only panel with a datetime index level.
        first (str): First factor name.
        second (str): Second factor name.
    Returns:
        float: Pooled rank correlation; NaN if fewer than two paired observations.
    Notes:
        Inexpensive redundancy proxy, not a datewise clustering estimator.
    """
    pair = train[[first, second]].replace([np.inf, -np.inf], np.nan).dropna()
    ranks = pair.groupby(level="datetime").rank(pct=True)
    return ranks[first].corr(ranks[second])


def select_features(train: pd.DataFrame, baseline: list[str], candidates: list[str],
                    registry: pd.DataFrame, max_features: int = 20,
                    min_coverage: float = 0.80, max_corr: float = 0.85,
                    min_days: int = 200) -> pd.DataFrame:
    """Choose expansion factors using training information exclusively.

    Description:
        Filter registry exclusions, coverage and IC; greedily reduce rank redundancy.
    Parameters:
        train (pd.DataFrame): Training-only labeled stock-date panel.
        baseline (list[str]): Frozen Original27 feature names.
        candidates (list[str]): Expansion feature names.
        registry (pd.DataFrame): Candidate registry with excluded flags.
        max_features (int): Maximum selected expansion features.
        min_coverage (float): Minimum finite observation fraction on Train.
        max_corr (float): Maximum absolute pooled rank correlation to selected/base factors.
        min_days (int): Minimum days with calculable cross-sectional IC.
    Returns:
        pd.DataFrame: All candidates with scores, reasons and selection flags.
    Notes:
        Selection never consults Validation labels. Daily IC is standard Spearman on
        finite pairs. Redundancy is an inexpensive pooled Train rank proxy computed on
        each pair's finite sample, not a datewise clustering estimator.
    """
    if not train.index.is_unique or max_features < 1:
        raise ValueError("Invalid training index or max_features")
    flags = registry.set_index("factor")["excluded"].to_dict() if not registry.empty else {}
    # No explicit daily minimum is specified here; 2 pairs is the mathematical minimum.
    ic = daily_rank_ic(train, candidates, min_stocks=2)
    coverage = train[candidates].notna().mean()
    summary = pd.DataFrame({"factor": candidates})
    summary["coverage"] = summary["factor"].map(coverage)
    summary["train_ic"] = summary["factor"].map(ic.mean())
    summary["ic_days"] = summary["factor"].map(ic.count()).fillna(0).astype(int)
    summary["excluded_registry"] = summary["factor"].map(flags).fillna(False).astype(bool)
    summary["score"] = summary["train_ic"].abs()
    summary["selected"] = False
    summary["reason"] = ""

    # Rank features only within Train; rank-based correlation reduces scale effects.
    selected = []
    reference = list(baseline)
    for index in summary.sort_values(["score", "factor"], ascending=[False, True], na_position="last").index:
        item = summary.loc[index]
        name = item["factor"]
        if item["excluded_registry"]:
            reason = "registry_excluded"
        elif item["coverage"] < min_coverage or item["ic_days"] < min_days:
            reason = "insufficient_coverage_or_days"
        elif not np.isfinite(item["score"]):
            reason = "invalid_ic"
        elif len(selected) >= max_features:
            reason = "selection_cap"
        else:
            correlations = pd.Series(
                {ref: _pooled_rank_corr(train, ref, name) for ref in reference}, dtype=float
            ).abs()
            reason = "redundant" if correlations.max(skipna=True) >= max_corr else "selected"
        summary.loc[index, "reason"] = reason
        if reason == "selected":
            selected.append(name)
            reference.append(name)
            summary.loc[index, "selected"] = True
    return summary.sort_values(["selected", "score"], ascending=[False, False]).reset_index(drop=True)


def compare_ridge(train: pd.DataFrame, validation: pd.DataFrame,
                  baseline: list[str], selected: list[str],
                  alpha: float = 100.0, min_stocks: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed-alpha nested Ridge models on identical Validation rows.

    Description:
        Fit both models on Train only, then compare daily Spearman Rank IC.
    Parameters:
        train (pd.DataFrame): Train observations with valid labels.
        validation (pd.DataFrame): Validation observations with valid labels.
        baseline (list[str]): Original27 predictors.
        selected (list[str]): Train-selected incremental predictors.
        alpha (float): Ridge penalty fixed before Validation evaluation.
        min_stocks (int): Minimum valid stocks per evaluation day.
    Returns:
        tuple: Daily matched IC frame and HAC inference summary.
    Notes:
        Both models use the exact same complete-case rows in each split.
        The comparison is conditional on the fixed Ridge specification.
    """
    if not selected:
        raise ValueError("No additional features were selected")
    features = baseline + selected
    fitting = train[features + ["label_5d"]].dropna()
    testing = validation[features + ["label_5d"]].dropna()
    if fitting.empty or testing.empty:
        raise ValueError("No common complete-case rows")
    x_train = cross_sectional_ranks(fitting, features)
    x_test = cross_sectional_ranks(testing, features)
    y_train = cross_sectional_ranks(fitting, ["label_5d"])["label_5d"]
    base_model = Ridge(alpha=alpha).fit(x_train[baseline], y_train)
    ext_model = Ridge(alpha=alpha).fit(x_train[features], y_train)

    scores = pd.DataFrame(index=testing.index)
    scores["baseline"] = base_model.predict(x_test[baseline])
    scores["extended"] = ext_model.predict(x_test[features])
    scores["label"] = testing["label_5d"]
    rows = []
    for date, day in scores.groupby(level="datetime"):
        if len(day) < min_stocks:
            continue
        ranks = day.rank()
        rows.append({"datetime": date, "n_stocks": len(day),
                     "baseline": ranks["baseline"].corr(ranks["label"]),
                     "extended": ranks["extended"].corr(ranks["label"])})
    daily = pd.DataFrame(rows).set_index("datetime")
    daily["delta_ic"] = daily["extended"] - daily["baseline"]
    return daily, hac_summary(daily[["baseline", "extended", "delta_ic"]], fdr=False)


def run_selection(data_dir: str, candidates_file: str, registry_file: str,
                  output_dir: str, max_features: int = 20,
                  min_coverage: float = 0.80, max_corr: float = 0.85,
                  alpha: float = 100.0) -> None:
    """Run selection and comparison, saving concise research outputs.

    Description:
        Load existing data, select features on Train, evaluate on Validation.
    Parameters:
        data_dir (str): Research data directory, i.e. <project root>/data (not the project root).
        candidates_file (str): Generated candidate parquet path.
        registry_file (str): Existing factor registry CSV path.
        output_dir (str): Destination for three CSV reports.
        max_features (int): Maximum selected candidate count.
        min_coverage (float): Minimum training coverage.
        max_corr (float): Maximum pooled training rank correlation.
        alpha (float): Fixed Ridge regularization strength.
    Returns:
        None: Writes selection, daily IC and summary CSV files.
    Notes:
        Do not tune thresholds on the already-inspected Validation period.
    """
    panel, baseline, candidates = load_panel(data_dir, candidates_file)
    dates = panel.index.get_level_values("datetime")
    train = panel.loc[(dates >= TRAIN[0]) & (dates <= TRAIN[1])]
    validation = panel.loc[(dates >= VALIDATION[0]) & (dates <= VALIDATION[1])]
    registry = pd.read_csv(registry_file)
    selection = select_features(train, baseline, candidates, registry, max_features,
                                min_coverage, max_corr)
    chosen = selection.loc[selection["selected"], "factor"].tolist()
    if not chosen:
        raise ValueError("No selected factors; review training-only selection thresholds")
    daily, summary = compare_ridge(train, validation, baseline, chosen, alpha)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output / "train_factor_selection.csv", index=False)
    daily.to_csv(output / "validation_matched_daily_ic.csv")
    summary.to_csv(output / "validation_matched_summary.csv", index=False)
    print(f"Selected {len(chosen)} of {len(candidates)} candidates")
    print(summary.to_string(index=False))
