"""Synthetic tests for historical universe, cached shares, and candidate market cap."""
import pandas as pd
import pytest
from factor_neutralization.exposure_data.build_universe import extract_universe
from factor_neutralization.exposure_data.validate_shares import audit_share_cache
from factor_neutralization.exposure_data.build_market_cap import construct_candidate_exposures


def test_universe_observed_only(tmp_path, monkeypatch):
    """Description: Verify that universe extraction preserves observed rows only.

    Parameters:
        tmp_path (Path): Pytest temporary directory.
    Returns:
        None: Asserts exact observed keys and symbol formatting.
    Notes:
        Missing days must not be fabricated.
    """
    folder = tmp_path / 'data' / 'processed'
    folder.mkdir(parents=True)
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp('2015-04-01'), 'SZ000001'), (pd.Timestamp('2015-04-03'), 'SH600519')],
        names=['datetime', 'instrument'],
    )

    (folder / 'csi300_daily_2015.parquet').touch()
    monkeypatch.setattr(pd, 'read_parquet', lambda *args, **kwargs: pd.DataFrame(index=idx))
    d = extract_universe(tmp_path)
    assert len(d) == 2
    assert set(d.symbol) == {'000001', '600519'}


def test_missing_cache_report(tmp_path):
    """Description: Verify missing share histories are explicitly reported.

    Parameters:
        tmp_path (Path): Empty share cache directory.
    Returns:
        None: Asserts missing symbol status.
    Notes:
        Missing data must not be silently imputed.
    """
    _, r = audit_share_cache(tmp_path, ['000001'])
    assert r['missing_symbols'] == ['000001']
    assert r['strict_pit_certified'] is False


def test_candidate_price_requires_acknowledgement(tmp_path):
    """Description: Reject candidate price reconstruction unless explicitly authorized.

    Parameters:
        tmp_path (Path): Temporary test root.
    Returns:
        None: Asserts explicit guard.
    Notes:
        Unit and price certification remain separate from program execution.
    """
    u = pd.DataFrame(
        {'datetime': [pd.Timestamp('2015-04-01')], 'instrument': ['SZ000001'], 'symbol': ['000001']}
    )
    e = pd.DataFrame()
    with pytest.raises(ValueError, match='allow-unverified-price'):
        construct_candidate_exposures(u, e, tmp_path, 10000)
