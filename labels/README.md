# CSI300 Label Engineering v1.1

Run from the **C-drive Python project root** (`labels/` is part of the repository; the pipeline runs through `run_data.py`). Keep all research data on the **D drive**.

```cmd
python -m unittest discover -s tests -p "test_label_engine.py" -v
python -m unittest discover -s tests -p "test_membership.py" -v
python -m unittest discover -s tests -p "test_price_audit.py" -v
python -u run_data.py labels
```

Optional `labels` flags: `--qlib-dir`, `--data-root`, `--start`, `--end`. Defaults point to the existing Windows D-drive locations. Requires `qlib`, `pandas`, `numpy`, and a Parquet engine such as `pyarrow`.

Inputs: Qlib `instruments/csi300.txt`, Qlib full historical individual-stock prices, and existing yearly `data/processed/csi300_daily_YYYY.parquet` for membership audit.

Outputs: `data/labels/forward_labels.parquet`; `reports/label_engineering/{membership_audit.csv,price_audit.csv,factor_change_candidates.csv,label_quality.csv,label_summary.json}`.

Labels are adjusted open-to-open returns, with entry on the next market trading day's open and exit after H holding intervals at open t+H+1. Valid signal dates start 2015-04-01; train ends 2020-12-31, validation ends 2023-12-31, and test ends 2026-09-30. Cross-split exits are rejected. Missing prices are not filled. Signal membership is independent of membership at exit.

**Research caveats:** This is not a PIT-certified historical index membership dataset. The Qlib factor and adjusted prices are not independently verified against corporate actions and raw trades. No limit-up/limit-down, suspension execution, delisting payoff, liquidity, or transaction-cost model is included. All outputs are research candidates, not executable strategy returns. The annual membership audit compares processed keys only for the configured 2015-04-01 through 2026-09-30 period; missing processed files must be reviewed.

## Label validation

`labels/label_validation.py` (run with `python -u run_data.py check-labels`) is read-only with
respect to labels, factors, Qlib data and price audits. Default inputs on D:
`data/labels/forward_labels.parquet`, `data/processed/csi300_daily_YYYY.parquet`,
`data/factors_enhanced/csi300_factors_YYYY.parquet`, `reports/label_engineering/*.csv` and Qlib
`instruments/csi300.txt`. Output: `reports/label_validation/`.

Limitation: the factor-change candidate file is not a complete daily factor series, so its
relative-change distribution is candidate-to-candidate, not necessarily consecutive market days.
No PIT certification, corporate-action reconciliation, tradability or delisting validation is
implied.
