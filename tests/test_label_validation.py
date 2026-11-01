"""Synthetic regression tests for read-only label validators."""
import unittest
import pandas as pd
from label_validation.validators import label_diagnostics, price_diagnostics


class TestLabelValidation(unittest.TestCase):
    """Description: Verify offset, return, and factor-threshold diagnostics. Parameters: None. Returns: None. Notes: Tests use synthetic data only."""

    def sample(self):
        """Description: Build a minimal valid synthetic label row. Parameters: None. Returns: tuple[pd.DataFrame,pd.DatetimeIndex]: row and market calendar. Notes: Entry and exit offsets use H+1."""
        dates = pd.bdate_range('2020-01-02', periods=20)
        row = {'instrument': 'SH600000', 'signal_date': dates[0], 'split': 'train'}
        for h in (1, 5, 10):
            row.update({f'entry_date_{h}d': dates[1], f'exit_date_{h}d': dates[h + 1], f'label_{h}d': .1, f'label_status_{h}d': 'ok', f'entry_open_adjusted_{h}d': 10., f'exit_open_adjusted_{h}d': 11.})
        return pd.DataFrame([row]), dates

    def test_valid_row(self):
        """Description: Verify a correct row produces zero violations. Parameters: None. Returns: None. Notes: Checks all horizons."""
        frame, dates = self.sample(); integrity, _, _ = label_diagnostics(frame, dates)
        self.assertEqual(int(integrity['violations'].sum()), 0)

    def test_return_mismatch(self):
        """Description: Detect a mislabeled forward return. Parameters: None. Returns: None. Notes: Uses 5-day horizon."""
        frame, dates = self.sample(); frame.loc[0, 'label_5d'] = .2
        integrity, _, _ = label_diagnostics(frame, dates)
        self.assertEqual(int(integrity.query("horizon == 5 and check == 'return_mismatch'")['violations'].iloc[0]), 1)

    def test_exit_mismatch(self):
        """Description: Detect an incorrect exit offset. Parameters: None. Returns: None. Notes: Uses 1-day horizon."""
        frame, dates = self.sample(); frame.loc[0, 'exit_date_1d'] = dates[3]
        integrity, _, _ = label_diagnostics(frame, dates)
        self.assertEqual(int(integrity.query("horizon == 1 and check == 'exit_date_mismatch'")['violations'].iloc[0]), 1)

    def test_factor_threshold(self):
        """Description: Verify factor threshold counts. Parameters: None. Returns: None. Notes: Uses synthetic factor changes."""
        issues = pd.DataFrame({'instrument': ['A'], 'datetime': ['2020-01-02'], 'issue': ['invalid_open']})
        candidates = pd.DataFrame({'instrument': ['A', 'A', 'A'], 'datetime': ['2020-01-02', '2020-01-03', '2020-01-06'], 'factor': [1., 1.000001, 1.1]})
        _, thresholds, _ = price_diagnostics(issues, candidates)
        self.assertEqual(int(thresholds.loc[thresholds['threshold'].eq(.01), 'count'].iloc[0]), 1)


if __name__ == '__main__':
    unittest.main()
