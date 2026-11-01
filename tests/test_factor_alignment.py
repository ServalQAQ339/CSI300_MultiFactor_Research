"""Synthetic tests for factor and label alignment."""
import pandas as pd
import pytest
from factor_research.alignment import prepare_panel
from factor_research.config import FACTOR_NAMES


def test_alignment_and_purging():
    """Description: Verify exact matching, status filtering and split-end purging.

    Parameters:
        None: Constructs synthetic panels.
    Returns:
        None: Asserts expected audit outcomes.
    Notes:
        Does not access real CSI300 data.
    """
    dates = pd.to_datetime(['2020-12-30','2020-12-31'])
    factors = pd.DataFrame(
        {
            'signal_date': dates,
            'instrument': ['SH600000'] * 2,
            **{name: [1.0, 2.0] for name in FACTOR_NAMES},
        }
    )

    labels = pd.DataFrame(
        {
            'signal_date': dates,
            'instrument': ['SH600000'] * 2,
            'split': ['train'] * 2,
            'label_5d': [0.02, 0.03],
            'label_status_5d': ['ok', 'ok'],
            'entry_date_5d': pd.to_datetime(['2020-12-31', '2021-01-04']),
            'exit_date_5d': pd.to_datetime(['2021-01-08', '2021-01-11']),
        }
    )

    panel, audit = prepare_panel(factors, labels)
    assert audit['factor_matched_rows'] == 2
    assert audit['valid_labels_crossing_split_end'] == 2
    assert panel['research_label_5d'].notna().all()


def test_invalid_status():
    """Description: Verify invalid labels are retained but excluded from research.

    Parameters:
        None: Constructs synthetic observations.
    Returns:
        None: Asserts label masking.
    Notes:
        Missing labels must not be imputed.
    """
    factors = pd.DataFrame(
        {
            'signal_date': pd.to_datetime(['2021-01-04']),
            'instrument': ['SH600000'],
            **{name: [1.0] for name in FACTOR_NAMES},
        }
    )

    labels = pd.DataFrame(
        {
            'signal_date': pd.to_datetime(['2021-01-04']),
            'instrument': ['SH600000'],
            'split': ['validation'],
            'label_5d': [0.02],
            'label_status_5d': ['missing'],
            'entry_date_5d': pd.to_datetime(['2021-01-05']),
            'exit_date_5d': pd.to_datetime(['2021-01-12']),
        }
    )

    panel, audit = prepare_panel(factors, labels)
    assert audit['valid_5d_label_rows'] == 0
    assert pd.isna(panel['research_label_5d'].iloc[0])
