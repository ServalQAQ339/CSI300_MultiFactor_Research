# CSI300 Label Engineering v1.0

This module produces gross adjusted open-to-open forward returns for horizons 1, 5 and 10 exchange trading days. Signal time is after close at t; entry is open t+1 and exit is open t+H+1.

## Install

```bash
pip install pandas numpy pyarrow pyqlib
```

## Membership input

Provide a CSV with columns `instrument,datetime,eligible`, e.g. `SH600000,2015-04-01,True`. It must represent **point-in-time-known** CSI300 membership for each signal date. The pipeline deliberately does not infer this file from the present-day universe. Instrument IDs must match Qlib IDs exactly. Supply a complete eligible signal-date panel; missing rows are not eligible.

## Run from project root (Windows CMD)

```cmd
python -m unittest discover -s tests -v
python run_label_pipeline.py --qlib-dir "D:\QuantResearch\qlib_data\cn_data\qlib_bin" --membership-csv "D:\QuantResearch\CSI300_MultiFactor_Research\data\csi300_membership_pit.csv" --output-dir "D:\QuantResearch\CSI300_MultiFactor_Research\data\labels"
```

Replace the membership path with the actual audited membership file location. Qlib installation and extraction interfaces may vary by local version.

## Outputs

- `forward_labels.parquet`: dated labels, adjusted entry/exit prices, split, and status codes.
- `price_audit_issues.csv`: nonpositive/missing adjusted prices and factors.
- `factor_change_candidates.csv`: changes in adjustment factors for manual corporate-action review.
- `label_summary.json`: counts and explicit unresolved warnings.

## Research limitations

- No claims of historical point-in-time correctness are made.
- Qlib normalized adjusted prices are not executable RMB quotes. Raw-price estimates are diagnostic only.
- A valid label is **not** proof of execution feasibility. Limit-up/limit-down, auction fills, slippage, liquidity, delisting and suspension exits need separate checks.
- No corporate-action cash-flow reconciliation is performed; independently verify adjusted-price return convention.
- A missing stock trading day is not skipped; fixed horizon is based on the exchange calendar.
- Split boundary is determined by exit date, not signal date alone.
- Overlap is represented by entry/exit dates; purge/embargo logic belongs to downstream validation.
- This first version loads all market instruments through Qlib and may be resource-intensive. Check available RAM and Qlib version.
