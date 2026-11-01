"""Unit tests for factor adjustment continuity and missing-price classification."""
import unittest
import numpy as np
import pandas as pd
from audits.factor_adjustment_audit import analyze_factor_panel, summarize_changes


class TestFactorAdjustmentAudit(unittest.TestCase):
    """Description: Validate consecutive calendar comparisons and threshold counts. Parameters: None. Returns: None. Notes: Synthetic examples only."""

    def setUp(self):
        """Description: Construct a four-day trading calendar with a missing stock observation. Parameters: None. Returns: None. Notes: Gap must not be interpreted as a one-day change."""
        self.calendar = pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06', '2020-01-07'])
        self.panel = pd.DataFrame(
            {
                'instrument': ['A', 'A', 'A'],
                'datetime': pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-07']),
                'factor': [1.0, 1.00001, 1.2],
                'open': [10.0, np.nan, 12.0],
                'close': [10.0, np.nan, 12.0],
                'volume': [100.0, np.nan, 100.0],
            }
        )

    def test_calendar_gap_excluded(self):
        """Description: Ensure multi-day gaps are excluded. Parameters: None. Returns: None."""
        d = analyze_factor_panel(self.panel, self.calendar)
        self.assertEqual(int(d['consecutive'].sum()), 1)
        self.assertTrue(np.isnan(d.iloc[2]['relative_change']))

    def test_missing_price_factor_valid(self):
        """Description: Confirm price missingness does not imply invalid factor. Parameters: None. Returns: None."""
        d = analyze_factor_panel(self.panel, self.calendar)
        self.assertTrue(bool(d.iloc[1]['price_missing']))
        self.assertTrue(bool(d.iloc[1]['factor_valid']))

    def test_thresholds(self):
        """Description: Exclude tiny factor changes from a 0.01 percent threshold. Parameters: None. Returns: None."""
        s = summarize_changes(analyze_factor_panel(self.panel, self.calendar))
        self.assertEqual(int(s.iloc[0]['exceedances']), 0)

    def test_duplicate_rejected(self):
        """Description: Reject duplicate stock-date observations. Parameters: None. Returns: None."""
        with self.assertRaises(ValueError):
            analyze_factor_panel(pd.concat([self.panel, self.panel.iloc[[0]]]), self.calendar)


if __name__ == '__main__':
    unittest.main()
