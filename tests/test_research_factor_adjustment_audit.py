"""Unit tests for research-universe factor adjustment auditing."""
import unittest
import pandas as pd
from research_factor_adjustment_audit import (
    compute_changes,
    apply_membership,
    summarize_thresholds,
    label_overlap,
)


class ResearchFactorAuditTests(unittest.TestCase):
    """Check calendar, membership and label-overlap invariants."""

    def test_missing_market_day_is_not_skipped(self):
        """Ensure missing market days invalidate consecutive comparisons."""
        x = pd.DataFrame(
            {
                'instrument': ['A', 'A'],
                'datetime': ['2020-01-02', '2020-01-06'],
                'factor': [1.0, 1.2],
                'open': [1.0, 1.0],
                'close': [1.0, 1.0],
            }
        )

        c = compute_changes(x, ['2020-01-02', '2020-01-03', '2020-01-06'])
        self.assertFalse(bool(c.iloc[1]['valid_pair']))

    def test_entry_date_uses_previous_factor(self):
        """Ensure factor changes are calculated before membership filtering."""
        x = pd.DataFrame(
            {
                'instrument': ['A', 'A'],
                'datetime': ['2020-01-02', '2020-01-03'],
                'factor': [1.0, 1.1],
                'open': [1.0, 1.0],
                'close': [1.0, 1.0],
            }
        )

        c = compute_changes(x, ['2020-01-02', '2020-01-03'])
        m = pd.DataFrame(
            {
                'instrument': ['A'],
                'start': pd.to_datetime(['2020-01-03']),
                'end': pd.to_datetime(['2020-01-03']),
            }
        )

        z = apply_membership(c, m)
        self.assertEqual(len(z), 1)
        self.assertAlmostEqual(z.iloc[0]['factor_change'], 0.1)

    def test_overlapping_membership_does_not_duplicate(self):
        """Ensure overlapping intervals retain each observation once."""
        x = pd.DataFrame({'instrument': ['A'], 'datetime': pd.to_datetime(['2020-01-03'])})
        m = pd.DataFrame(
            {
                'instrument': ['A', 'A'],
                'start': pd.to_datetime(['2020-01-01', '2020-01-02']),
                'end': pd.to_datetime(['2020-01-05', '2020-01-06']),
            }
        )
        self.assertEqual(len(apply_membership(x, m)), 1)

    def test_thresholds(self):
        """Ensure threshold counts use valid comparison denominator."""
        x = pd.DataFrame(
            {
                'valid_pair': [True, True, False],
                'factor_change': [0.02, 0.00001, None],
                'price_missing': [False, True, False],
            }
        )
        d = summarize_thresholds(x)
        self.assertEqual(int(d.loc[d['threshold'].eq(0.01), 'exceedances'].iloc[0]), 1)

    def test_label_overlap(self):
        """Ensure a significant event inside a holding interval is counted."""
        events = pd.DataFrame(
            {
                'instrument': ['A'],
                'datetime': pd.to_datetime(['2020-01-06']),
                'valid_pair': [True],
                'factor_change': [0.03],
            }
        )

        labels = pd.DataFrame(
            {
                'instrument': ['A'],
                'signal_date': ['2020-01-02'],
                'entry_date_5d': ['2020-01-03'],
                'exit_date_5d': ['2020-01-10'],
                'label_5d': [0.25],
                'label_status_5d': ['ok'],
            }
        )

        out = label_overlap(events, labels)
        self.assertEqual(int(out.iloc[0]['significant_factor_events']), 1)
        self.assertTrue(bool(out.iloc[0]['extreme_return_20pct']))


if __name__ == '__main__': unittest.main()
