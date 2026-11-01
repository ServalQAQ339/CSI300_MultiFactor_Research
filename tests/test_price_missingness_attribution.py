"""Description: Unit tests for Qlib price-missingness classification and join integrity."""
import unittest
import numpy as np
import pandas as pd
from audits.price_missingness_attribution import classify_price, prepare_audit, attribute_prices


class TestPriceMissingnessAttribution(unittest.TestCase):
    """Description: Validate missing-key, numeric-state, and audit-join behavior."""

    def test_classification(self):
        """Description: Verify numeric and missing-key classifications. Parameters: None. Returns: None. Notes: Synthetic cases only."""
        self.assertEqual(classify_price(None, False), 'missing_key')
        self.assertEqual(classify_price(np.nan, True), 'missing_value')
        self.assertEqual(classify_price(0, True), 'zero')
        self.assertEqual(classify_price(-2, True), 'negative')
        self.assertEqual(classify_price(np.inf, True), 'infinite')
        self.assertEqual(classify_price(3, True), 'valid')

    def test_audit_deduplication(self):
        """Description: Verify two audit issues map to one key. Parameters: None. Returns: None. Notes: Issue labels remain available."""
        audit = pd.DataFrame(
            {
                'instrument': ['A', 'A'],
                'datetime': ['2020-01-02'] * 2,
                'issue': ['invalid_open', 'invalid_close'],
            }
        )

        result = prepare_audit(audit)
        self.assertEqual(len(result), 1)
        self.assertIn('invalid_open', result.iloc[0]['audit_issues'])

    def test_unmatched_and_matched(self):
        """Description: Verify missing Qlib keys differ from matched NaN fields. Parameters: None. Returns: None. Notes: No forward fill."""
        keys = prepare_audit(
            pd.DataFrame(
                {
                    'instrument': ['A', 'B'],
                    'datetime': ['2020-01-02'] * 2,
                    'issue': ['invalid_open'] * 2,
                }
            )
        )

        quotes = pd.DataFrame(
            {
                'instrument': ['A'],
                'datetime': pd.to_datetime(['2020-01-02']),
                '$open': [np.nan],
                '$close': [3.0],
                '$volume': [0.0],
                '$factor': [1.0],
            }
        )

        result = attribute_prices(keys, quotes).set_index('instrument')
        self.assertEqual(result.loc['A', 'open_state'], 'missing_value')
        self.assertEqual(result.loc['B', 'open_state'], 'missing_key')
        self.assertEqual(result.loc['A', 'volume_state'], 'zero')

    def test_duplicate_quote_rejected(self):
        """Description: Reject duplicate stock-date Qlib observations. Parameters: None. Returns: None. Notes: Prevents silent row multiplication."""
        keys = prepare_audit(
            pd.DataFrame(
                {'instrument': ['A'], 'datetime': ['2020-01-02'], 'issue': ['invalid_open']}
            )
        )

        quotes = pd.DataFrame(
            {
                'instrument': ['A', 'A'],
                'datetime': ['2020-01-02'] * 2,
                '$open': [1, 1],
                '$close': [1, 1],
                '$volume': [1, 1],
                '$factor': [1, 1],
            }
        )

        with self.assertRaises(ValueError):
            attribute_prices(keys, quotes)


if __name__ == '__main__':
    unittest.main()
