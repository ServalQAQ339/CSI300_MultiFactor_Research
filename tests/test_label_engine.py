"""Synthetic tests for label timing, missing observations and split leakage."""
import unittest
import numpy as np
import pandas as pd
from labels.label_engine import LabelConfig, audit_prices, build_labels


class TestLabels(unittest.TestCase):
    """Description: Check core label invariants on synthetic observations. Parameters: None. Returns: None. Notes: Tests do not validate any actual CSI300 market data."""

    def setUp(self):
        """Description: Create a synthetic exchange calendar and price panel. Parameters: None. Returns: None. Notes: All prices are invented for deterministic tests."""
        self.cal = pd.bdate_range('2020-12-21', '2021-01-15')
        self.p = pd.DataFrame({'instrument': 'SH600000', 'datetime': self.cal, 'open': np.arange(len(self.cal), dtype=float) + 10, 'close': np.arange(len(self.cal), dtype=float) + 11, 'factor': 0.1})
        self.m = pd.DataFrame({'instrument': 'SH600000', 'datetime': self.cal, 'eligible': True})
        self.cfg = LabelConfig(horizons=(1, 5, 10), research_start='2020-12-21', train_end='2020-12-31', validation_end='2021-01-15', test_end='2021-01-15')

    def test_next_open_and_holding(self):
        """Description: Verify exact H+1 exit offset. Parameters: None. Returns: None. Notes: Uses market rather than per-stock calendar."""
        r = build_labels(self.p, self.cal, self.m, self.cfg).iloc[0]
        self.assertEqual(r['entry_date_5d'], self.cal[1])
        self.assertEqual(r['exit_date_5d'], self.cal[6])
        self.assertAlmostEqual(r['label_5d'], self.p.iloc[6]['open'] / self.p.iloc[1]['open'] - 1)

    def test_missing_exit_does_not_shift(self):
        """Description: Check that a missing stock trading day is not skipped. Parameters: None. Returns: None. Notes: Missing exits remain invalid."""
        p = self.p.loc[self.p['datetime'] != self.cal[6]]
        r = build_labels(p, self.cal, self.m, self.cfg).iloc[0]
        self.assertEqual(r['label_status_5d'], 'invalid_exit_price')
        self.assertTrue(pd.isna(r['label_5d']))

    def test_split_boundary(self):
        """Description: Reject labels with exit after train endpoint. Parameters: None. Returns: None. Notes: A valid future price cannot override split rules."""
        r = build_labels(self.p, self.cal, self.m, self.cfg)
        row = r.loc[r['signal_date'] == pd.Timestamp('2020-12-30')].iloc[0]
        self.assertEqual(row['label_status_5d'], 'split_boundary')
        self.assertTrue(pd.isna(row['label_5d']))

    def test_invalid_factor(self):
        """Description: Reject zero factor at entry and flag it in price audit. Parameters: None. Returns: None. Notes: Factor is required even when adjusted open is positive."""
        p = self.p.copy()
        p.loc[p['datetime'] == self.cal[1], 'factor'] = 0
        audited, issues = audit_prices(p)
        self.assertTrue((issues['issue'] == 'invalid_factor').any())
        r = build_labels(audited, self.cal, self.m, self.cfg).iloc[0]
        self.assertEqual(r['label_status_1d'], 'invalid_entry_price')

    def test_no_membership_no_signal(self):
        """Description: Exclude ineligible signal-date securities. Parameters: None. Returns: None. Notes: Later membership changes do not forcibly close a position."""
        m = self.m.copy()
        m.loc[m['datetime'] == self.cal[0], 'eligible'] = False
        r = build_labels(self.p, self.cal, m, self.cfg)
        self.assertFalse((r['signal_date'] == self.cal[0]).any())


if __name__ == '__main__':
    unittest.main()
