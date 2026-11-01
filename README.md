# CSI300 Multi-Factor Research

Research code for a CSI300 multi-factor stock-selection study: data extraction from Qlib,
technical factor engineering, forward-return labels, cross-sectional preprocessing, and
Rank IC / quantile factor evaluation. All outputs are descriptive research diagnostics, not
executable strategy returns.

Research window: signal dates 2015-04-01 to 2026-09-30 (2015 Q1 is factor warm-up).
Splits: train 2015-04-01 to 2020-12-31, validation 2021-01-01 to 2023-12-31,
test 2024-01-01 to 2026-09-30. Defaults live in `config.py` and `factor_research/config.py`.

## Directories

| Location | Role |
|---|---|
| `C:\Users\lyueq\Desktop\CSI300_MultiFactor_Research` | Source code only (this repository). |
| `D:\QuantResearch\CSI300_MultiFactor_Research` | Research data and reports: `data/{processed,factors,factors_enhanced,factors_preprocessed,labels,external}`, `reports/...`. Never versioned. |
| `D:\QuantResearch\qlib_data\cn_data\qlib_bin` | Local Qlib binary data (`instruments/csi300.txt` holds historical membership). |

Data paths are hard-coded in `config.py`, `factor_research/config.py` and several `run_*.py`
defaults. Most `run_*.py` scripts **write to D:**; read their docstrings before running.

## Package structure

- `config.py` — global paths, dates and splits.
- `data/` — Qlib loading, yearly OHLCV panels, data audit, extended-history loader.
- `factors/` — factor engine (27 technical factors), history backfill, preprocessing
  (rank_gaussian, robust, rank_uniform, raw), factor audits.
- `labels/` — forward open-to-open labels (1/5/10 day), historical membership, price audit.
- `label_validation/`, `preprocessing_validation/` — read-only validators for labels and preprocessing.
- `factor_research/` — alignment, daily IC / Rank IC, quintiles, HAC inference, diagnostics,
  train-only selection, consistency audit.
- `factor_registry/`, `factor_verification/` — factor definition registry and independent recomputation.
- `factor_neutralization/` — PIT source inventory and industry/market-cap exposure data (in progress).
- Root audits: `factor_adjustment_audit.py`, `research_factor_adjustment_audit.py`,
  `internal_return_audit.py`, `price_missingness_attribution.py`.
- `tests/` — pytest suite. `archive/` — superseded files kept for reference.
- `backtest/`, `portfolio/`, `models/`, `validation/` — empty placeholders.

## Pipeline execution order

Run from the project root.

1. `run_data_pipeline.py` — yearly `csi300_daily_YYYY.parquet`, data-quality audit.
2. `run_factor_pipeline.py` — baseline factors (`data/factors`); `run_factor_audit.py` checks them.
3. `run_factor_backfill.py` — enhanced factors with pre-membership history (`data/factors_enhanced`);
   `run_enhanced_audit.py` checks them.
4. `run_factor_preprocessing.py` — `data/factors_preprocessed/<method>`;
   `preprocessing_validation/run_preprocessing_validation.py` checks them.
5. `run_label_pipeline.py` — `data/labels/forward_labels.parquet`; then `run_label_validation.py`
   and the root audit scripts.
6. Rank IC research, in order: `run_factor_research.py` (v1.2) → `run_factor_diagnostics.py` (v1.3)
   → `run_factor_consistency_audit.py` (v1.4) → `run_factor_selection.py` (v1.5).
   Later steps read the v1.2 outputs.
7. `run_factor_registry.py`, `run_factor_value_verification.py` — registry and value audit.
8. `factor_neutralization/exposure_data/` — universe, share download (`akshare`), validation.

## Current status

Completed: data panels, baseline and enhanced factors, preprocessing and its validation,
labels and label validation, Rank IC research v1.2 to v1.5, factor registry, factor value verification,
corporate-action spot check (`Corporate_Action_Verification_Summary.md`).

In progress: industry / market-cap exposures for neutralization (`factor_neutralization/`).

## Known limitations and unfinished work

- No backtest, portfolio construction or model training yet (placeholder folders are empty).
- Historical index membership and exposure data are not certified point-in-time.
- Qlib adjusted prices are not independently verified; only three corporate actions were checked.
- No limit-up/down, suspension, delisting, liquidity or transaction-cost modelling.
- `factor_research/core.py:hac_mean` and `statistical_tests.py:hac_summary` are different HAC
  estimators and are used by different stages; they must not be treated as interchangeable.
- `data/process.py` is an empty file that two scripts try to import before falling back to `data.data_loader`.
- Configuration (`DATA_ROOT`, dates, `FACTOR_NAMES`) is duplicated across modules.
- `reports/internal_return_audit/return_consistency_failures.csv` on D: is about 502 MB and should be reviewed.

## Setup and validation

```bash
pip install -r requirements.txt
python -B -m pytest -q -p no:cacheprovider
```

The test suite uses synthetic data and does not read or write D:. Importing `config.py`
creates its output folders on D: if they are missing.
