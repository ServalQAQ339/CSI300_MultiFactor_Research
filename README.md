# CSI300 Multi-Factor Alpha Research

**A Multi-Factor Equity Selection and Machine Learning Framework for the CSI 300 Index**

Python | Microsoft Qlib | LightGBM | Ridge Regression | Factor Research | Quantitative Backtesting

## Overview

This project develops an end-to-end quantitative equity research framework for the Chinese A-share market, focusing on the historical constituents of the CSI 300 Index.

The research integrates factor engineering, cross-sectional alpha evaluation, machine learning, expanding-window validation, and systematic portfolio construction.

Inspired by Microsoft Qlib, the framework independently implements key components of factor preprocessing, predictive signal evaluation, statistical testing, and portfolio backtesting.

### Research Highlights

| Research Component | Key Result |
|---|---|
| Research Period | January 2015 – September 2026 |
| Equity Universe | 667 historical CSI 300 constituents |
| Panel Dataset | 839,400 stock-date observations |
| Factor Engineering | 27 baseline factors + 99 candidate factors |
| Extended Feature Set | 47 features |
| Prediction Models | Ridge Regression and LightGBM |
| Validation Rank IC | **0.0364** (HAC t-stat: 5.18) |
| Forward-Period Rank IC | **0.0503** (2024–2026, LightGBM Original27) |
| Validation Long–Short Spread | **30.1% annualized gross spread** (HAC t-stat: 2.59) |

*The long–short spread represents a theoretical, equal-notional Top-30 minus Bottom-30 portfolio before transaction costs and short-borrowing fees. Forward-period evaluation is chronological but is not an untouched holdout.*

## 1. Data Engineering

The research dataset covers historical CSI 300 constituents rather than relying solely on the current index membership.

The data pipeline supports:

- Historical constituent membership and daily equity panel construction.
- OHLCV price data processing and corporate-action adjustments.
- Forward-return label engineering with a five-trading-day prediction horizon.
- Cross-sectional factor preprocessing and normalization.
- Data completeness and temporal-alignment validation.

The resulting dataset contains approximately 840,000 stock-date observations across 667 securities.

## 2. Alpha Factor Research

The framework develops 27 baseline factors and 99 additional candidate factors across several categories:

- Price momentum and reversal.
- Volatility and trading range.
- Volume and liquidity characteristics.
- Price–volume interactions.
- Return distribution and higher-order statistics.

Factor effectiveness is evaluated using:

- Pearson Information Coefficient (IC).
- Spearman Rank Information Coefficient (Rank IC).
- Information Coefficient Information Ratio (ICIR).
- Newey–West HAC statistical inference.
- False Discovery Rate (FDR) adjustment.
- Cross-sectional factor correlation analysis.
- Quantile portfolio analysis and temporal stability testing.

The factor selection pipeline identifies 20 additional features for the extended 47-feature modeling configuration.

## 3. Machine Learning Models

Two modeling approaches are implemented:

**Ridge Regression**

A regularized linear benchmark for evaluating the predictive contribution of cross-sectional factors.

**LightGBM**

A gradient-boosted decision-tree model designed to capture nonlinear relationships and interactions among equity characteristics.

Models are evaluated using chronological training and validation splits, followed by expanding-window forward-period evaluation.

### Predictive Performance

| Evaluation Period | Model | Mean Rank IC |
|---|---|---:|
| 2021–2023 Validation | Ridge Original27 | 0.0387 |
| 2021–2023 Validation | LightGBM Original27 | 0.0364 |
| 2024–2026 Forward Evaluation | LightGBM Original27 | 0.0503 |
| 2024–2026 Forward Evaluation | LightGBM Extended | 0.0630 |

The results demonstrate positive cross-sectional ranking signals across the evaluated periods, with model performance varying by feature configuration and market environment.

## 4. Portfolio Construction and Backtesting

Model predictions are translated into systematic equity portfolios using cross-sectional ranking.

The framework supports research into:

- Top-K equity selection.
- Equal-weight portfolio construction.
- Five-trading-day rebalancing.
- Quantile-based return analysis.
- Theoretical long–short portfolios.
- Transaction cost and turnover analysis.
- Benchmark-relative performance evaluation.

### Long–Short Signal Evaluation

A theoretical portfolio is constructed by taking equal-notional long positions in the 30 highest-ranked stocks and short positions in the 30 lowest-ranked stocks.

For the LightGBM Original27 model during the 2021–2023 validation period:

| Metric | Result |
|---|---:|
| Annualized Gross Long–Short Spread | **30.1%** |
| HAC t-statistic | **2.59** |
| Gross Spread CAGR | 32.7% |
| Gross Spread Maximum Drawdown | −16.1% |

These results illustrate the model's ability to differentiate between relatively strong and weak future equity returns in the validation period.

The long–short results are presented as a predictive-signal research portfolio, not as live trading performance.

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
        Factor Selection
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
     Performance Attribution
```

## 6. Project Structure

The framework provides four consolidated command-line entry points:

```text
CSI300_MultiFactor_Research/
|
|-- run_data.py
|-- run_factors.py
|-- run_research.py
|-- run_backtest.py
|
|-- data/
|-- factors/
|-- factor_research/
|-- labels/
|-- modeling/
|-- portfolio/
|-- tests/
|
|-- README.md
```

The entry points organize the major research stages, while the underlying modules implement reusable data processing, statistical analysis, model evaluation, and portfolio research functionality.

## 7. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python, NumPy, Pandas |
| Market Data | Microsoft Qlib |
| Machine Learning | Scikit-learn, LightGBM |
| Statistical Analysis | SciPy, HAC inference, FDR |
| Data Storage | Parquet, CSV, JSON |
| Research Engineering | Git, Pytest, Modular Python |

## 8. Research Applications

This project demonstrates an integrated workflow for:

- Cross-sectional equity return prediction.
- Alpha factor development and evaluation.
- Machine learning model comparison.
- Systematic stock ranking and selection.
- Quantitative portfolio research.
- Statistical assessment of investment signals.

The framework is designed as an extensible research foundation for equity alpha modeling and systematic investment strategy development.

---

**Project Type:** Independent Quantitative Finance Research  
**Market:** China A-Shares  
**Universe:** CSI 300 Historical Constituents  
**Research Period:** 2015–2026
