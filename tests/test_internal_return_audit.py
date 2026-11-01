"""Unit tests for internal return consistency and adjustment decomposition."""
import unittest
import pandas as pd
from audits.internal_return_audit import calculate_endpoint_checks, summarize_checks


class TestInternalReturnAudit(unittest.TestCase):
    """Description: Validate reconstruction and failure detection on synthetic prices. Parameters: None. Returns: None. Notes: Tests are independent of Qlib."""

    def setUp(self):
        """Description: Create a minimal two-endpoint synthetic dataset. Parameters: None. Returns: None. Notes: Factor rises 10 percent and raw price falls correspondingly."""
        self.prices = pd.DataFrame(
            {
                'instrument': ['A', 'A'],
                'datetime': pd.to_datetime(['2020-01-02', '2020-01-03']),
                'open': [100.0, 110.0],
                'factor': [1.0, 1.1],
            }
        )

        self.labels = pd.DataFrame(
            {
                'instrument': ['A'],
                'signal_date': ['2020-01-01'],
                'entry_date_1d': ['2020-01-02'],
                'exit_date_1d': ['2020-01-03'],
                'label_1d': [0.1],
                'label_status_1d': ['ok'],
                'entry_open_adjusted_1d': [100.0],
                'exit_open_adjusted_1d': [110.0],
            }
        )

    def test_correct_reconstruction(self):
        """Description: Verify adjusted return and factor decomposition. Parameters: None. Returns: None. Notes: Synthetic net factor increase is 10 percent."""
        d = calculate_endpoint_checks(self.labels, self.prices, 1)
        self.assertTrue(bool(d.loc[0, 'audit_pass']))
        self.assertAlmostEqual(float(d.loc[0, 'raw_price_return']), 0)
        self.assertEqual(summarize_checks(d)['1']['audit_failures'], 0)

    def test_wrong_label_fails(self):
        """Description: Detect an incorrect stored label. Parameters: None. Returns: None. Notes: Tolerance is absolute."""
        self.labels.loc[0, 'label_1d'] = 0.2
        self.assertFalse(
            bool(calculate_endpoint_checks(self.labels, self.prices, 1).loc[0, 'audit_pass'])
        )

    def test_missing_endpoint_fails(self):
        """Description: Detect absent Qlib exit key. Parameters: None. Returns: None. Notes: Missing keys must not be interpreted as zero returns."""
        d = calculate_endpoint_checks(self.labels, self.prices.iloc[:1], 1)
        self.assertEqual(summarize_checks(d)['1']['missing_qlib_endpoint_keys'], 1)
        self.assertFalse(bool(d.loc[0, 'audit_pass']))

    def test_price_mismatch_fails(self):
        """Description: Detect disagreement with stored adjusted entry price. Parameters: None. Returns: None. Notes: Independent endpoint comparison catches source mismatch."""
        self.labels.loc[0, 'entry_open_adjusted_1d'] = 99.0
        self.assertFalse(
            bool(calculate_endpoint_checks(self.labels, self.prices, 1).loc[0, 'audit_pass'])
        )


if __name__ == '__main__':
    unittest.main()
