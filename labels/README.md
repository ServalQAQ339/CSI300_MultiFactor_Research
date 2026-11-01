# CSI300 Label Engineering v1.1

Run from the **C-drive Python project root** after copying `labels/`, `run_label_pipeline.py`, and the two new tests into the existing project. Keep all research data on the **D drive**.

```cmd
python -m unittest discover -s tests -p "test_label_engine.py" -v
python -m unittest discover -s tests -p "test_membership.py" -v
python -m unittest discover -s tests -p "test_price_audit.py" -v
python -u run_label_pipeline.py
```

Optional flags: `--qlib-dir`, `--data-root`, `--start`, `--end`. Defaults point to the existing Windows D-drive locations. Requires `qlib`, `pandas`, `numpy`, and a Parquet engine such as `pyarrow`.

Inputs: Qlib `instruments/csi300.txt`, Qlib full historical individual-stock prices, and existing yearly `data/processed/csi300_daily_YYYY.parquet` for membership audit.

Outputs: `data/labels/forward_labels.parquet`; `reports/label_engineering/{membership_audit.csv,price_audit.csv,factor_change_candidates.csv,label_quality.csv,label_summary.json}`.

Labels are adjusted open-to-open returns, with entry on the next market trading day's open and exit after H holding intervals at open t+H+1. Valid signal dates start 2015-04-01; train ends 2020-12-31, validation ends 2023-12-31, and test ends 2026-09-30. Cross-split exits are rejected. Missing prices are not filled. Signal membership is independent of membership at exit.

**Research caveats:** This is not a PIT-certified historical index membership dataset. The Qlib factor and adjusted prices are not independently verified against corporate actions and raw trades. No limit-up/limit-down, suspension execution, delisting payoff, liquidity, or transaction-cost model is included. All outputs are research candidates, not executable strategy returns. The annual membership audit compares processed keys only for the configured 2015-04-01 through 2026-09-30 period; missing processed files must be reviewed.
