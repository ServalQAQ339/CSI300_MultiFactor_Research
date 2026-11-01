"""Test adjusted-price validity checks on synthetic observations."""
import unittest
import numpy as np
import pandas as pd
from labels.price_audit import summarize_price_audit


class TestPriceAudit(unittest.TestCase):
    """Description: Verify invalid factor detection and factor-change reporting. Parameters: None. Returns: None. Notes: Synthetic prices are not market evidence."""

    def test_invalid_factor(self):
        """Description: Reject zero adjustment factors. Parameters: None. Returns: None. Notes: Recovered raw prices must not be considered valid."""
        frame = pd.DataFrame({'instrument': ['SH600000'], 'datetime': ['2020-01-02'], 'open': [10.0], 'close': [10.0], 'factor': [0.0]})
        issues, _ = summarize_price_audit(frame)
        self.assertIn('invalid_factor', issues['issue'].tolist())

    def test_factor_change(self):
        """Description: Flag a factor transition as an audit candidate. Parameters: None. Returns: None. Notes: A change alone is not a confirmed error."""
        frame = pd.DataFrame({'instrument': ['SH600000'] * 2, 'datetime': ['2020-01-02', '2020-01-03'], 'open': [10.0, 11.0], 'close': [10.0, 11.0], 'factor': [0.5, 0.6]})
        _, changes = summarize_price_audit(frame)
        self.assertEqual(len(changes), 1)
