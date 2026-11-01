"""Select incremental features on Train only and evaluate matched Ridge models."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge


TRAIN = ("2015-04-01", "2020-12-31")
VALIDATION = ("2021-01-01", "2023-12-31")


def load_panel(data_root: str, candidate_file: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load frozen baseline, expansion features and valid forward labels.

    Description:
        Join unique stock-date observations and filter invalid label statuses.
    Parameters:
        data_root (str): Root of research data on D drive.
        candidate_file (str): Path to the existing candidate parquet file.
    Returns:
        tuple: Joined panel, baseline names and candidate names.
    Notes:
        Historical universe PIT status is not independently certified.
    """
    root = Path(data_root)
    files = sorted((root / "factors_preprocessed" / "raw").glob("csi300_factors_20??.parquet"))
    if not files:
        raise FileNotFoundError("No Original27 annual files found")
    baseline_frame = pd.concat([pd.read_parquet(file) for file in files]).sort_index()
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

    labels = pd.read_parquet(root / "labels" / "forward_labels.parquet")
    required = {"signal_date", "instrument", "label_5d", "label_status_5d"}
    if not required.issubset(labels):
        raise ValueError(f"Missing label columns: {required - set(labels)}")
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels = labels.set_index(["signal_date", "instrument"])
    labels.index.names = ["datetime", "instrument"]
    if not labels.index.is_unique:
        raise ValueError("Duplicate label keys")

    panel = baseline_frame.join(candidates_frame[candidates], how="left", validate="one_to_one")
    panel = panel.join(labels[["label_5d", "label_status_5d"]], how="left", validate="one_to_one")
    panel = panel.loc[panel["label_status_5d"].eq("ok") & panel["label_5d"].notna()]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel, baseline, candidates


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
        Selection never consults Validation labels. Correlation is an inexpensive
        pooled Train rank proxy, not a datewise clustering estimator.
    """
    if not train.index.is_unique or max_features < 1:
        raise ValueError("Invalid training index or max_features")
    flags = registry.set_index("factor")["excluded"].to_dict() if not registry.empty else {}
    rank_cols = baseline + candidates
    ranks = train.groupby(level="datetime")[rank_cols + ["label_5d"]].rank(pct=True)
    ic_rows = []
    for _, day in ranks.groupby(level="datetime", sort=True):
        ic_rows.append(day[candidates].corrwith(day["label_5d"]))
    ic = pd.DataFrame(ic_rows)
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
            values = ranks[name]
            correlations = ranks[reference].corrwith(values).abs()
            reason = "redundant" if correlations.max(skipna=True) >= max_corr else "selected"
        summary.loc[index, "reason"] = reason
        if reason == "selected":
            selected.append(name)
            reference.append(name)
            summary.loc[index, "selected"] = True
    return summary.sort_values(["selected", "score"], ascending=[False, False]).reset_index(drop=True)


def _rank_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert each trading-date cross-section to centered percentile ranks.

    Description:
        Apply independent same-day ranking to predictors and target.
    Parameters:
        frame (pd.DataFrame): Complete-case panel with datetime index level.
        columns (list[str]): Names to transform.
    Returns:
        pd.DataFrame: Centered percentile ranks.
    Notes:
        No future-date values enter this transformation.
    """
    return frame.groupby(level="datetime")[columns].rank(pct=True) - 0.5


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
    x_train = _rank_features(fitting, features)
    x_test = _rank_features(testing, features)
    y_train = _rank_features(fitting, ["label_5d"])["label_5d"]
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
    return daily, hac_report(daily[["baseline", "extended", "delta_ic"]])


def hac_report(daily: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Calculate mean daily IC and Newey-West inference.

    Description:
        Estimate HAC standard errors of daily mean IC values.
    Parameters:
        daily (pd.DataFrame): Daily IC series, one column per model/statistic.
        lags (int): Fixed Newey-West maximum lag.
    Returns:
        pd.DataFrame: Mean IC, ICIR, HAC t-statistic and two-sided p-value.
    Notes:
        ICIR is daily and not annualized. No multiple-testing claim is made.
    """
    rows = []
    for name in daily:
        values = daily[name].dropna().to_numpy(dtype=float)
        n = len(values)
        if n < 3:
            rows.append({"factor": name, "days": n, "mean_rank_ic": np.nan,
                         "icir": np.nan, "hac_t": np.nan, "p_value": np.nan})
            continue
        mean = values.mean()
        centered = values - mean
        long_run = centered @ centered / n
        for lag in range(1, min(lags, n - 1) + 1):
            long_run += 2 * (1 - lag / (lags + 1)) * (centered[lag:] @ centered[:-lag]) / n
        standard_error = np.sqrt(max(long_run, 0) / n)
        t_stat = mean / standard_error if standard_error > 0 else np.nan
        std = values.std(ddof=1)
        rows.append({"factor": name, "days": n, "mean_rank_ic": mean,
                     "icir": mean / std if std > 0 else np.nan, "hac_t": t_stat,
                     "p_value": 2 * norm.sf(abs(t_stat)) if np.isfinite(t_stat) else np.nan})
    return pd.DataFrame(rows)


def run_selection(data_root: str, candidates_file: str, registry_file: str,
                  output_dir: str, max_features: int = 20,
                  min_coverage: float = 0.80, max_corr: float = 0.85,
                  alpha: float = 100.0) -> None:
    """Run selection and comparison, saving concise research outputs.

    Description:
        Load existing data, select features on Train, evaluate on Validation.
    Parameters:
        data_root (str): Research data directory.
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
    panel, baseline, candidates = load_panel(data_root, candidates_file)
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
