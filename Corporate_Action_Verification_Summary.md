# CSI300 MultiFactor Research — Corporate Action Verification Summary

**Status:** Research-Ready with Documented Limitations  
**Scope:** Targeted verification of three high-risk corporate-action events; not a representative random sample.  
**Research period:** 2015-04-01 to 2026-09-30 (2015 Q1 warm-up).

## 1. Event-level findings

| Stock | Ex-date | Independent unadjusted pre-ex close | Corporate-action terms used | Theoretical factor multiplier | Qlib factor multiplier (approx.) | Assessment |
|---|---|---:|---|---:|---:|---|
| SZ002594 BYD | 2025-07-29 | CNY 337.00 | 8 bonus + 12 capitalized shares per 10; CNY 39.74 cash per 10 | 3.0358 | 3.0358 | Close agreement; exact precision and data provenance pending |
| SZ002131 Leo Group | 2017-06-13 | CNY 12.50 | 25 capitalized shares per 10; CNY 0.37 cash per 10 | 3.51039 | 3.5113 | Small residual; adjustment convention pending |
| SZ000425 XCMG | 2015-09-23 | CNY 13.81 | 20 capitalized shares per 10; zero cash provisionally assumed | 3.00000 | 3.0022 | Small unexplained residual; verify complete terms |

For the simplified no-rights case, with pre-ex unadjusted close P, per-old-share cash dividend D, bonus ratio b and capitalization ratio c:

P_ex_theoretical = (P - D) / (1 + b + c)
A_theoretical = P / P_ex_theoretical

These formulas require confirmation of event terms, adjustment convention, rounding and exceptional treatments. The ex-date market open/close need not equal the theoretical ex-rights reference price.

## 2. Data evidence and provenance

- Historical unadjusted OHLC values were obtained by the user from AkShare interfaces (BYD via Eastmoney, Leo Group and XCMG via Sina); independence of upstream vendor data from Qlib is not established.
- Corporate-action terms and dates were matched in earlier public announcement research. Exact announcement URLs and archived documents should be retained before formal publication.
- Qlib multipliers shown here are rounded figures from prior research reports. Full-precision event-by-event numerical residuals have not been computed in this summary.
- All three events are **preliminary numerical matches**, not unqualified passes.

## 3. Label engineering reconciliation

| Item | Result |
|---|---:|
| Processed historical rows | 856,188 |
| Warm-up rows | 17,096 |
| Processed research-period rows | 839,092 |
| Theoretical label signal rows | 839,400 |
| Missing Qlib stock-date keys relative to theoretical membership | 308 |
| 1D valid labels | 821,714 |
| 5D valid labels | 816,184 |
| 10D valid labels | 810,254 |
| Unique price-audit stock-date keys with missing OHLC/volume | 37,102 |

Internal return consistency: across 2,448,152 valid horizon-label observations, the maximum absolute recomputation error was about 1.10e-7. Previously flagged failures at 1e-8 all satisfy 1e-6; endpoint and decomposition checks found no other failures in the reported diagnostics. This is an internal arithmetic result, not independent verification of corporate actions.

## 4. Open limitations

1. Index constituent publication-time point-in-time availability is not established.
2. Qlib adjustment factors and unadjusted prices have not been comprehensively checked against independent primary-source data.
3. Executability at next-day open, limit-up/limit-down, suspensions, liquidity, and delisting payoffs remain unverified.
4. Three targeted cases cannot support a population error-rate confidence interval.
5. Qlib factor multipliers and theoretical ratios should be compared at full precision with documented tolerance and event-specific conventions.

## 5. Decision

**Research-Ready with Documented Limitations:** proceed to exploratory factor IC / Rank IC / ICIR and quantile analysis, while preserving the candidate label status and avoiding claims of verified implementable alpha. Use only exact stock-date intersections with valid horizon labels, and report sample counts, missingness, and split-specific results. Do not modify original label or factor datasets during exploratory analysis.
