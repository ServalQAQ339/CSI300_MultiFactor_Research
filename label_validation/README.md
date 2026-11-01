# CSI300 Label Quality Validation v1.0

This package is **read-only** with respect to existing labels, factors, Qlib data, and price audits.

Copy `label_validation/`, `run_label_validation.py`, and `tests/test_label_validation.py` to the existing C-drive project root. Keep all existing tests.

Run from Windows CMD in the project root:

```cmd
python -m unittest discover -s tests -p "test_label_validation.py" -v
python -u run_label_validation.py
```

Default input paths:

- `D:\QuantResearch\CSI300_MultiFactor_Research\data\labels\forward_labels.parquet`
- `D:\QuantResearch\CSI300_MultiFactor_Research\data\processed\csi300_daily_YYYY.parquet`
- `D:\QuantResearch\CSI300_MultiFactor_Research\data\factors_enhanced\csi300_factors_YYYY.parquet`
- `D:\QuantResearch\CSI300_MultiFactor_Research\reports\label_engineering\*.csv`
- `D:\QuantResearch\qlib_data\cn_data\qlib_bin\instruments\csi300.txt`

Output: `D:\QuantResearch\CSI300_MultiFactor_Research\reports\label_validation\`.

Important limitations: the factor-change candidate file is not a complete daily factor series, so its relative-change distribution is **candidate-to-candidate**, not necessarily consecutive-market-day changes. No PIT certification, corporate action reconciliation, tradability or delisting validation is implied.
