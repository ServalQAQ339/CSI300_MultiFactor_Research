"""Cross-sectional, point-in-time factor transformations without clipping or imputation."""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessConfig:
    """Description:
        Specify a deterministic daily cross-sectional factor transformation.

    Parameters:
        method (str): One of raw, rank_uniform, rank_gaussian, or robust.
        min_valid (int): Minimum finite observations required per date and factor.
        iqr_epsilon (float): Numerical threshold below which an IQR is considered zero.

    Returns:
        PreprocessConfig: Immutable preprocessing settings.

    Notes:
        No parameters are fitted across dates. No quantile clipping or missing-value filling is performed.
    """
    method: str = "rank_gaussian"
    min_valid: int = 10
    iqr_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.method not in {"raw", "rank_uniform", "rank_gaussian", "robust"}:
            raise ValueError(f"Unsupported preprocessing method: {self.method}")
        if self.min_valid < 2 or self.iqr_epsilon <= 0:
            raise ValueError("min_valid must be >= 2 and iqr_epsilon must be positive")


def _transform_cross_section(values: pd.Series, config: PreprocessConfig) -> pd.Series:
    """Description:
        Transform one date-factor cross-section using only its finite observations.

    Parameters:
        values (pd.Series): Values for one factor on one trading date, indexed by instrument.
        config (PreprocessConfig): Transformation settings.

    Returns:
        pd.Series: Transformed values with the same index; invalid entries remain NaN.

    Notes:
        Average ranks handle ties. Rank Gaussianization uses scipy.stats.norm.ppf.
        Robust scaling uses the median and IQR; zero-IQR cross-sections return NaN.
    """
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = clean.dropna()
    if len(valid) < config.min_valid:
        return result
    if config.method == "raw":
        result.loc[valid.index] = valid
    elif config.method in {"rank_uniform", "rank_gaussian"}:
        ranks = valid.rank(method="average")
        probabilities = (ranks - 0.5) / len(valid)
        if config.method == "rank_uniform":
            result.loc[valid.index] = probabilities.to_numpy()
        else:
            from scipy.stats import norm
            result.loc[valid.index] = norm.ppf(probabilities.to_numpy())
    else:
        median = valid.median()
        q25, q75 = valid.quantile([0.25, 0.75]).to_numpy()
        iqr = q75 - q25
        if np.isfinite(iqr) and iqr > config.iqr_epsilon:
            result.loc[valid.index] = (valid - median) / iqr
    return result


def preprocess_factor_panel(
    panel: pd.DataFrame, config: PreprocessConfig, factor_columns: list[str] | None = None
) -> pd.DataFrame:
    """Description:
        Transform a multi-stock factor panel independently within each trading date.

    Parameters:
        panel (pd.DataFrame): Factor panel with MultiIndex (datetime, instrument).
        config (PreprocessConfig): Transformation settings.
        factor_columns (list[str] | None): Explicit factor names; None selects numeric columns.

    Returns:
        pd.DataFrame: Factor-only panel with identical row index and requested column order.

    Notes:
        Each date is processed in isolation; no train/validation/test observations are pooled.
        Only eligible instruments already present in the input panel are ranked.
        The caller must ensure the input factor values are available by the signal timestamp.
    """
    if not isinstance(panel.index, pd.MultiIndex) or list(panel.index.names) != [
        "datetime",
        "instrument",
    ]:
        raise ValueError("Expected MultiIndex with names ['datetime', 'instrument']")
    if panel.index.has_duplicates:
        raise ValueError("Duplicate (datetime, instrument) keys are not allowed")
    columns = (
        factor_columns
        if factor_columns is not None
        else panel.select_dtypes(include="number").columns.tolist()
    )
    if (
        not columns
        or len(set(columns)) != len(columns)
        or any(column not in panel.columns for column in columns)
    ):
        raise ValueError("factor_columns must be nonempty, unique, and present in panel")
    result = pd.DataFrame(index=panel.index, columns=columns, dtype=float)
    for _, date_panel in panel.groupby(level="datetime", sort=False):
        for column in columns:
            transformed = _transform_cross_section(date_panel[column], config)
            result.loc[date_panel.index, column] = transformed.to_numpy()
    return result


def compare_prefix_invariance(
    panel: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    config: PreprocessConfig,
    factor_columns: list[str] | None = None,
    atol: float = 1e-12,
) -> dict:
    """Description:
        Compare transformations computed on full history versus history truncated at cutoff.

    Parameters:
        panel (pd.DataFrame): Historical factor panel indexed by (datetime, instrument).
        cutoff (str | pd.Timestamp): Inclusive last available date for the prefix.
        config (PreprocessConfig): Transformation settings.
        factor_columns (list[str] | None): Columns to test.
        atol (float): Absolute numerical comparison tolerance.

    Returns:
        dict: Counts of missing-value and numerical mismatches on the truncated prefix.

    Notes:
        Tests the preprocessing algorithm, not whether source data were truly point-in-time.
    """
    full = preprocess_factor_panel(panel, config, factor_columns)
    prefix_panel = panel.loc[panel.index.get_level_values("datetime") <= pd.Timestamp(cutoff)]
    prefix = preprocess_factor_panel(prefix_panel, config, factor_columns)
    reference = full.loc[prefix.index]
    a, b = reference.to_numpy(), prefix.to_numpy()
    missing_mismatch = int(np.count_nonzero(np.isnan(a) != np.isnan(b)))
    value_mismatch = (
        int(np.count_nonzero(~np.isclose(a, b, rtol=0, atol=atol, equal_nan=True)))
        - missing_mismatch
    )
    return {
        "cutoff": str(pd.Timestamp(cutoff).date()),
        "rows": len(prefix),
        "missing_mismatch": missing_mismatch,
        "value_mismatch": value_mismatch,
    }
