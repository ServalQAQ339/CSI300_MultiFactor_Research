# Result summaries

Small CSV summaries of the main research outputs. Full panels, factor files, model scores and
intermediate reports are not versioned. All results are exploratory research outputs, not live or
executable trading performance.

| File | Content | Source |
|---|---|---|
| `train_factor_selection.csv` | 99 candidate factors: Train coverage, Train Rank IC, selection decision (20 selected) | `run_research.py select` |
| `validation_model_rank_ic.csv` | Validation 2021-2023 daily Rank IC, ICIR and HAC t (lag 5) for Ridge / LightGBM, plus paired model differences | `run_backtest.py models` |
| `walk_forward_model_rank_ic.csv` | Same statistics for the 2024-2026 quarterly expanding-window evaluation (660 signal dates) | Derived, see below |
| `validation_long_short_spread.csv` | Validation theoretical Top-30 minus Bottom-30 gross spread (145 non-overlapping periods) | Derived, see below |
| `validation_top30_portfolio.csv` | Validation long-only Top-30 net returns (10 bps per side) and equal-weight proxy comparison | `run_backtest.py portfolio` |
| `validation_top30_vs_csi300.csv` | Validation long-only Top-30 net returns versus the CSI 300 price index | `run_backtest.py benchmark` |
| `walk_forward_top30_vs_csi300.csv` | 2024-2026 long-only Top-30 net returns versus the CSI 300 price index | `run_backtest.py walk-forward` |

## Conventions

- In the two Rank IC files the `factor` column holds the model name; `delta_*` rows are paired
  daily differences between models. `hac_t` uses Newey-West with lag 5.
- Portfolio returns in the `*_top30_*` files are net of 10 bps per side, annualized by compounding
  over 50.4 periods a year. The CSI 300 benchmark (`csi300_*`, `benchmark_*`) is the price index
  `SH000300`, which excludes dividends.
- `annualized_arithmetic_*` columns are the mean period value multiplied by 50.4.

## Derived summaries

Two files were computed afterwards from saved outputs, without retraining or re-running backtests.
No entrypoint in this repository writes them.

- `walk_forward_model_rank_ic.csv`: saved walk-forward model scores joined with five-day labels
  whose status is `ok`, then evaluated with `modeling.modeling.evaluate_predictions` (daily
  Spearman, at least 30 stocks per day, HAC lag 5).
- `validation_long_short_spread.csv`: on the same 145 signal dates as the Validation portfolio, the
  equal-weight mean five-day open-to-open return of the 30 highest-scored stocks minus that of the
  30 lowest-scored stocks with a valid label. The Top-30 leg matches the portfolio report's gross
  return to within 1e-8. Annualization uses 50.4 periods a year; `hac_t_nw_lag4` is a Newey-West
  (Bartlett, lag 4) t-statistic of the mean period spread.

The spread is gross and theoretical: no transaction costs, short-borrowing fees or short-sale
constraints. For LightGBM Original27, most of it comes from the Bottom-30 leg (annualized
arithmetic gross return about +7.7% for Top-30 and -22.4% for Bottom-30).
