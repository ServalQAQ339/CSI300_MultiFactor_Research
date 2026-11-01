"""Audit the Qlib adjusted-price input without claiming independent price validation."""
import numpy as np
import pandas as pd
from labels.label_engine import audit_prices


def summarize_price_audit(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Description: Check adjusted price and factor integrity, and flag factor changes. Parameters: prices (pd.DataFrame): dated adjusted open, close and factor values. Returns: tuple[pd.DataFrame, pd.DataFrame]: issue rows and factor-change candidates. Notes: Factor changes are candidates for review, not confirmed errors; raw price reconstruction is not independent verification."""
    audited, issues = audit_prices(prices)
    candidates = audited.loc[
        audited['factor_changed'],
        ['instrument', 'datetime', 'factor', 'raw_open_estimate', 'raw_close_estimate'],
    ].copy()

    invalid_raw = (
        ~np.isfinite(audited['raw_open_estimate']) | (audited['raw_open_estimate'] <= 0)
    ) & audited['open'].gt(0)

    if invalid_raw.any():
        extra = audited.loc[invalid_raw, ['instrument', 'datetime']].copy()
        extra['issue'] = 'invalid_recovered_raw_open'
        issues = pd.concat([issues, extra], ignore_index=True)

    return issues, candidates
