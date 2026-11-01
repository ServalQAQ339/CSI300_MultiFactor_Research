"""Test inclusive constituent interval expansion and overlap deduplication."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from labels.membership import expand_membership, load_membership_intervals


class TestMembership(unittest.TestCase):
    """Description: Verify historical constituent date semantics. Parameters: None. Returns: None. Notes: Uses synthetic intervals."""

    def test_inclusive_interval(self):
        """Description: Verify interval endpoints are included. Parameters: None. Returns: None. Notes: Exchange dates are explicitly supplied."""
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'csi300.txt'
            path.write_text('SH600000\t2020-01-02\t2020-01-03\n', encoding='utf-8')
            intervals = load_membership_intervals(path)
            dates = pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06'])
            result = expand_membership(intervals, dates, '2020-01-02', '2020-01-06')
            self.assertEqual(len(result), 2)

    def test_overlapping_intervals(self):
        """Description: Verify overlapping intervals cannot duplicate signal keys. Parameters: None. Returns: None. Notes: Deduplication is deterministic."""
        intervals = pd.DataFrame({'instrument': ['SH600000', 'SH600000'], 'start': pd.to_datetime(['2020-01-02', '2020-01-03']), 'end': pd.to_datetime(['2020-01-03', '2020-01-06'])})
        result = expand_membership(intervals, pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06']), '2020-01-02', '2020-01-06')
        self.assertEqual(len(result), 3)
