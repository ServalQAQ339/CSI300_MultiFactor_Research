# CSI300 Multi-Factor Alpha Research

**A Multi-Factor Equity Selection and Machine Learning Framework for the CSI 300 Index**

Python | Microsoft Qlib | LightGBM | Ridge Regression | Factor Research | Quantitative Backtesting

## Overview

This project develops an end-to-end quantitative equity research framework for the Chinese A-share
market, focusing on the historical constituents of the CSI 300 Index.

The research integrates factor engineering, cross-sectional alpha evaluation, machine learning,
expanding-window evaluation, and systematic portfolio construction. Qlib serves as the market-data
provider; factor construction, preprocessing, signal evaluation, statistical testing and portfolio
backtesting are implemented independently in this repository.

### Research Highlights

| Research Component | Key Result |
|---|---|
| Research Period | April 2015 – September 2026 (signal dates) |
| Equity Universe | 667 historical CSI 300 constituents |
| Panel Dataset | 839,400 stock-date observations |
| Factor Engineering | 27 baseline factors + 99 candidate factors |
| Extended Feature Set | 47 features (27 baseline + 20 Train-selected candidates) |
| Prediction Models | Ridge Regression and LightGBM |
| Validation Rank IC | **0.0364** (LightGBM Original27, 2021–2023, HAC t-stat: 5.18) |
| Forward-Period Rank IC | **0.0503** (LightGBM Original27, 2024–2026, quarterly expanding-window refits) |
| Validation Long–Short Spread | **30.1% annualized arithmetic gross spread** (2021–2023, HAC t-stat: 2.59) |

*The long–short spread is a theoretical, equal-notional Top-30 minus Bottom-30 portfolio before
transaction costs and short-borrowing fees; it is not a realized or net return. The forward-period
evaluation is chronological but is not an untouched holdout: the period had been examined during
earlier research stages.*

### 项目概览

- 基于沪深300历史成分股（667只，信号日 2015-04 至 2026-09，839,400 条股票日样本）构建端到端多因子
  选股研究框架，预测目标为未来 5 个交易日的开盘到开盘收益。
- 构建 27 个基础量价因子与 99 个扩展候选因子，仅用训练集筛选 20 个增量因子；以 Rank IC、ICIR、
  Newey–West HAC 与 BH-FDR 评估因子有效性。
- 比较 Ridge 与 LightGBM：LightGBM（Original27）在 2021–2023 Validation 的日均 Rank IC 为 0.0364
  （HAC t=5.18），在 2024–2026 季度扩展窗口时间外推中为 0.0503。该区间在早期研究中被查看过，不是盲测。
- Validation 期理论 Top30−Bottom30 等名义多空价差算术年化 30.1%（HAC t=2.59），为扣费前毛收益，
  未计交易成本、融券费用及 A 股做空限制，不代表实盘或净收益。

## 1. Data Engineering

The research dataset covers historical CSI 300 constituents on each signal date rather than relying
on the current index membership.

The data pipeline supports:

- Historical constituent membership (Qlib `csi300` intervals) and yearly daily equity panels.
- Qlib adjusted OHLCV prices, with factor warm-up history from 2014.
- Forward-return labels with a five-trading-day horizon: signal after the close of day `t`, entry
  at the open of `t+1`, exit at the open of `t+6`.
- Cross-sectional factor preprocessing and normalization.
- Label integrity, data completeness and look-ahead checks on preprocessing.

The resulting dataset contains approximately 840,000 stock-date observations across 667 securities.

| Split | Signal dates | Use |
|---|---|---|
| Train | 2015-04-01 to 2020-12-31 | Factor selection and model fitting |
| Validation | 2021-01-01 to 2023-12-31 | Model comparison with frozen specifications |
| Forward evaluation | 2024-01-02 to 2026-09-30 | Quarterly expanding-window refits (11 refits) |

## 2. Alpha Factor Research

The framework develops 27 baseline factors and 99 additional candidate factors (9 from the Qlib
Alpha158 `CORR`, `VSTD` and `RSV` families, re-implemented in pandas, and 90 OHLCV-inspired factors)
across several categories:

- Price momentum and reversal.
- Volatility and trading range.
- Volume and liquidity characteristics.
- Price–volume interactions.
- Return distribution and higher-order statistics.

Factor effectiveness is evaluated using:

- Pearson Information Coefficient (IC).
- Spearman Rank Information Coefficient (Rank IC).
- Information Coefficient Information Ratio (ICIR).
- Newey–West HAC statistical inference for overlapping five-day labels.
- Benjamini–Hochberg False Discovery Rate (FDR) adjustment.
- Cross-sectional factor correlation and redundancy analysis.
- Quantile return analysis and IC autocorrelation diagnostics.

Selection uses Train data only: candidates are ranked by absolute Train Rank IC, filtered by coverage
and by redundancy against the baseline and already-selected factors, and capped at 20. These 20
factors form the extended 47-feature modeling configuration
([selection table](results/train_factor_selection.csv)).

## 3. Machine Learning Models

Two modeling approaches are implemented, both trained on cross-sectional percentile ranks of
features and of the five-day forward return:

**Ridge Regression**

A regularized linear benchmark (alpha = 100) for evaluating the predictive contribution of
cross-sectional factors.

**LightGBM**

A gradient-boosted decision-tree model designed to capture nonlinear relationships and interactions
among equity characteristics. Hyperparameters are fixed (300 trees, learning rate 0.03, 15 leaves),
with no Validation tuning or early stopping.

Models are evaluated on a chronological Train/Validation split (with the last Train dates purged to
avoid label overlap), followed by a 2024–2026 forward evaluation in which the frozen specifications
are refit every quarter on all labels that have matured before the quarter starts.

### Predictive Performance

Daily Spearman Rank IC between model scores and five-day forward returns (HAC lag 5).

| Evaluation Period | Model | Mean Rank IC | HAC t-stat |
|---|---|---:|---:|
| 2021–2023 Validation | Ridge Original27 | 0.0387 | 4.50 |
| 2021–2023 Validation | LightGBM Original27 | 0.0364 | 5.18 |
| 2021–2023 Validation | LightGBM Extended | 0.0326 | 3.78 |
| 2024–2026 Forward Evaluation | Ridge Original27 | 0.0373 | 2.30 |
| 2024–2026 Forward Evaluation | LightGBM Original27 | 0.0503 | 3.74 |
| 2024–2026 Forward Evaluation | LightGBM Extended | 0.0630 | 3.94 |

The results show positive cross-sectional ranking signals in both periods, with model performance
varying by feature configuration and market environment. In Validation, the differences between
models are not statistically significant, and the extended feature set does not improve on
Original27; the ordering of models differs between the two periods.

Sources: [Validation](results/validation_model_rank_ic.csv),
[forward evaluation](results/walk_forward_model_rank_ic.csv).

## 4. Portfolio Construction and Backtesting

Model predictions are translated into systematic equity portfolios using cross-sectional ranking.
The implemented backtest uses:

- Top-30 equity selection with equal weights.
- Five-trading-day non-overlapping rebalancing with next-open execution.
- A flat transaction cost of 10 bps per side on every rebalance.
- Comparison against the CSI 300 price index and an equal-weight proxy of the eligible universe.

### Long–Short Signal Evaluation

A theoretical portfolio takes equal-notional long positions in the 30 highest-ranked stocks and
short positions in the 30 lowest-ranked stocks, over the same 145 non-overlapping Validation
periods as the backtest.

For the LightGBM Original27 model during the 2021–2023 validation period:

| Metric | Result |
|---|---:|
| Annualized Gross Long–Short Spread (arithmetic) | **30.1%** |
| HAC t-statistic (Newey–West, lag 4) | **2.59** |
| Gross Spread CAGR | 32.7% |
| Gross Spread Maximum Drawdown | −16.1% |

These results illustrate the model's ability to separate relatively strong from relatively weak
future returns in the validation period. Most of the spread comes from the Bottom-30 leg, which
cannot be shorted directly in the China A-share market. The spread is a predictive-signal
diagnostic, not live trading performance ([details](results/README.md)).

### Gross Spread versus Net Portfolio Returns

The theoretical spread ignores transaction costs, borrowing fees and short-sale constraints, so it
measures ranking quality rather than achievable returns. The implementable long-only Top-30
portfolio is evaluated separately, net of 10 bps per side; its excess return over the CSI 300 is
not statistically significant, so a significant Rank IC does not imply a significant net portfolio
excess return. Sources: [Validation](results/validation_top30_vs_csi300.csv),
[forward evaluation](results/walk_forward_top30_vs_csi300.csv).

## 5. Research Workflow

```text
Historical CSI 300 Constituents
              |
              v
       Data Engineering
              |
              v
       Factor Generation
              |
              v
 Factor Preprocessing & Evaluation
              |
              v
     Train-only Factor Selection
              |
              v
      Ridge / LightGBM
              |
              v
   Expanding-Window Evaluation
              |
              v
     Portfolio Construction
              |
              v
     Benchmark Comparison
```

## 6. Project Structure

The framework provides four consolidated command-line entry points:

```text
CSI300_MultiFactor_Research/
|
|-- run_data.py          Data panels, labels, label checks
|-- run_factors.py       Baseline factors, warm-up backfill, preprocessing, factor registry
|-- run_research.py      Rank IC research, candidate factors, Train-only selection
|-- run_backtest.py      Models, Validation portfolios, CSI 300 comparison, forward evaluation
|-- config.py            Central dates and paths (overridable by environment variables)
|
|-- data/                Qlib extraction and yearly OHLCV panels
|-- factors/             Baseline factor engine, backfill, preprocessing, candidate factors
|-- labels/              Forward labels, CSI 300 membership, label validation
|-- factor_research/     Rank IC, HAC inference, diagnostics, Train-only selection
|-- modeling/            Ridge / LightGBM comparison
|-- portfolio/           Backtest, benchmark comparison, expanding-window evaluation
|-- audits/              Read-only data and factor consistency checks
|-- results/             Small CSV summaries of the main results
|-- tests/               Synthetic-data unit tests
|
|-- requirements.txt
|-- README.md
```

The entry points organize the major research stages, while the underlying modules implement
reusable data processing, statistical analysis, model evaluation, and portfolio research
functionality.

### Running

Data are not versioned. A local Qlib CN daily provider with the `csi300` instruments file is
required. Set the data locations with environment variables:

```powershell
$env:CSI300_DATA_ROOT = "<path to research data root>"   # holds data/ and reports/
$env:CSI300_QLIB_DIR  = "<path to Qlib cn_data provider>"
```

Without them, `config.py` uses `research_data/` inside the repository and the standard Qlib
location `~/.qlib/qlib_data/cn_data`.

The stages run in this order: `run_data.py panels`, `run_factors.py backfill` and `preprocess`,
`run_data.py labels`, `run_research.py ic`, `generate-candidates`, `analyze-candidates` and
`select`, then `run_backtest.py models`, `portfolio`, `benchmark` and `walk-forward`. Several steps
read full-history data and fit models on the full panel.

```bash
pip install -r requirements.txt
python -m pytest -q
```

The unit tests use synthetic data only and do not read Qlib data.

## 7. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python, NumPy, Pandas |
| Market Data | Microsoft Qlib |
| Machine Learning | Scikit-learn, LightGBM |
| Statistical Analysis | SciPy, Statsmodels, HAC inference, FDR |
| Data Storage | Parquet (PyArrow), CSV, JSON |
| Research Engineering | Git, Pytest, Modular Python |

## 8. Research Applications

This project demonstrates an integrated workflow for:

- Cross-sectional equity return prediction.
- Alpha factor development and evaluation.
- Machine learning model comparison.
- Systematic stock ranking and selection.
- Quantitative portfolio research.
- Statistical assessment of investment signals.

The framework is designed as an extensible research foundation for equity alpha modeling and
systematic investment strategy development.

---

**Project Type:** Independent Quantitative Finance Research  
**Market:** China A-Shares  
**Universe:** CSI 300 Historical Constituents  
**Research Period:** 2015–2026
