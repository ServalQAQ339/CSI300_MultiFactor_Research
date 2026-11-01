
"""Run an exploratory CSI300 factor standardization and size-neutralization research pipeline."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def neutralize_day(frame: pd.DataFrame, factor: str, min_stocks: int = 30) -> pd.DataFrame:
    """
    Description:
        Winsorize and standardize one daily cross-sectional factor, then
        residualize it against candidate log market capitalization.

    Parameters:
        frame (pd.DataFrame): One-date stock observations containing datetime,
            instrument, the selected factor, and log_market_cap_candidate.
        factor (str): Name of the raw factor to process.
        min_stocks (int): Minimum number of valid observations required.

    Returns:
        pd.DataFrame: Date-stock keys, standardized factor values, and
            re-standardized size-neutral residuals.

    Notes:
        Winsorization uses the 1st and 99th percentiles.
        Size neutralization uses unweighted cross-sectional OLS.
        Standardization does not require valid market capitalization.
        Size-neutral values are exploratory because size inputs are uncertified.
    """
    d = frame[['datetime', 'instrument', factor, 'log_market_cap_candidate']].copy()
    d[factor] = pd.to_numeric(d[factor], errors='coerce')
    d['log_market_cap_candidate'] = pd.to_numeric(d['log_market_cap_candidate'], errors='coerce')
    d['standardized'] = np.nan
    d['size_neutral'] = np.nan

    factor_valid = np.isfinite(d[factor].to_numpy(dtype=float))
    x = d.loc[factor_valid, factor].to_numpy(dtype=float)

    if len(x) < min_stocks:
        return d[['datetime', 'instrument', 'standardized', 'size_neutral']]

    low, high = np.quantile(x, [0.01, 0.99])
    x = np.clip(x, low, high)
    sd = np.std(x)

    if sd < 1e-12:
        return d[['datetime', 'instrument', 'standardized', 'size_neutral']]

    z = (x - np.mean(x)) / sd
    d.loc[factor_valid, 'standardized'] = z

    neutral_valid = factor_valid & np.isfinite(d['log_market_cap_candidate'].to_numpy(dtype=float))

    if neutral_valid.sum() < min_stocks:
        return d[['datetime', 'instrument', 'standardized', 'size_neutral']]

    size = d.loc[neutral_valid, 'log_market_cap_candidate'].to_numpy(dtype=float)
    y = d.loc[neutral_valid, 'standardized'].to_numpy(dtype=float)

    if np.std(size) < 1e-12:
        return d[['datetime', 'instrument', 'standardized', 'size_neutral']]

    X = np.column_stack([np.ones(len(size)), size - np.mean(size)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    residual_sd = np.std(residual)

    if residual_sd > 1e-12:
        d.loc[neutral_valid, 'size_neutral'] = (residual - np.mean(residual)) / residual_sd

    return d[['datetime', 'instrument', 'standardized', 'size_neutral']]


def summarize_ic(data: pd.DataFrame, factor: str, label: str, min_stocks: int = 30) -> dict:
    """
    Description:
        Calculate daily cross-sectional Spearman Rank IC, ICIR, and
        Newey-West HAC t-statistics for one factor.

    Parameters:
        data (pd.DataFrame): Stock-date panel containing factor values,
            forward return labels, and datetime.
        factor (str): Factor column used for IC calculation.
        label (str): Forward return label column.
        min_stocks (int): Minimum paired observations required per date.

    Returns:
        dict: Number of valid IC days, mean Rank IC, daily ICIR,
            HAC t-statistic, and observation coverage.

    Notes:
        Spearman Rank IC measures cross-sectional ranking consistency.
        HAC uses five lags as a fixed exploratory setting for 5D labels.
        Daily ICIR is not annualized.
    """
    from statsmodels.api import OLS

    d = data[['datetime', factor, label]].replace([np.inf, -np.inf], np.nan)
    valid = d[[factor, label]].notna().all(axis=1)
    daily_ic = []

    for _, group in d.loc[valid].groupby('datetime', sort=True):
        if len(group) < min_stocks:
            continue

        if group[factor].nunique() <= 1 or group[label].nunique() <= 1:
            continue

        ic = group[factor].corr(group[label], method='spearman')

        if np.isfinite(ic):
            daily_ic.append(ic)

    values = np.asarray(daily_ic, dtype=float)

    result = {
        'factor': factor,
        'days': int(len(values)),
        'pair_coverage': float(valid.mean()) if len(d) else 0.0,
        'mean_rank_ic': float(values.mean()) if len(values) else None,
        'icir_daily': float(values.mean() / values.std(ddof=1)) if len(values) > 1 and values.std(ddof=1) > 0 else None,
        'hac_t': None
    }

    if len(values) > 6:
        model = OLS(values, np.ones((len(values), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        result['hac_t'] = float(model.tvalues[0])

    return result


def load_exposures(path: str) -> pd.DataFrame:
    """
    Description:
        Load candidate historical market-cap exposures and normalize join keys.

    Parameters:
        path (str): Path to the candidate market-cap Parquet file.

    Returns:
        pd.DataFrame: Unique datetime-instrument keys with candidate log size.

    Notes:
        Supports both MultiIndex and ordinary-column storage formats.
        Does not certify historical market capitalization or PIT compliance.
    """
    d = pd.read_parquet(path)

    if 'datetime' not in d.columns or 'instrument' not in d.columns:
        d = d.reset_index()

    required = ['datetime', 'instrument', 'log_market_cap_candidate']
    missing = set(required) - set(d.columns)

    if missing:
        raise ValueError(f'Missing exposure columns: {sorted(missing)}')

    d['datetime'] = pd.to_datetime(d['datetime'])
    d['instrument'] = d['instrument'].astype('string').str.upper()
    d = d[required].copy()

    if d.duplicated(['datetime', 'instrument']).any():
        raise ValueError('Duplicate exposure join keys')

    return d


def load_labels(path: str, label_column: str) -> pd.DataFrame:
    """
    Description:
        Load forward return labels and align signal_date with factor datetime.

    Parameters:
        path (str): Path to the forward_labels.parquet file.
        label_column (str): Forward return label, such as label_5d.

    Returns:
        pd.DataFrame: Unique datetime-instrument keys with forward labels.

    Notes:
        Labels are joined using signal_date rather than entry or exit dates.
        Only observations with valid label status are retained when a
        corresponding label_status column is available.
    """
    d = pd.read_parquet(path)

    if 'signal_date' not in d.columns or 'instrument' not in d.columns:
        d = d.reset_index()

    required = ['signal_date', 'instrument', label_column]
    missing = set(required) - set(d.columns)

    if missing:
        raise ValueError(f'Missing label columns: {sorted(missing)}')

    status_column = label_column.replace('label_', 'label_status_', 1)

    if status_column in d.columns:
        d.loc[d[status_column].notna() & ~d[status_column].astype('string').str.lower().isin(['valid', 'ok', 'success']), label_column] = np.nan

    d['datetime'] = pd.to_datetime(d['signal_date'])
    d['instrument'] = d['instrument'].astype('string').str.upper()
    d = d[['datetime', 'instrument', label_column]].copy()

    if d.duplicated(['datetime', 'instrument']).any():
        raise ValueError('Duplicate label join keys')

    return d


def main() -> None:
    """
    Description:
        Evaluate Original27 raw, standardized, and size-neutralized factors
        across fixed Train, Validation, and diagnostic Test periods.

    Parameters:
        None: Input paths, factor names, and output directory are supplied
            through command-line arguments.

    Returns:
        None: Writes the exploratory factor panel, Rank IC comparison JSON,
            and a tabular CSV summary to the requested output directory.

    Notes:
        Raw and neutralized factors are evaluated on identical paired samples.
        Candidate market capitalization is not strictly PIT-certified.
        The previously inspected Test period is diagnostic rather than blind.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--exposure', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--label-column', required=True)
    parser.add_argument('--factors', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    exposures = load_exposures(args.exposure)
    labels = load_labels(args.labels, args.label_column)

    splits = {
        'train': ('2015-04-01', '2020-12-31'),
        'validation': ('2021-01-01', '2023-12-31'),
        'test_diagnostic': ('2024-01-01', '2026-09-30')
    }

    panels = []
    metrics = []

    for year in range(2015, 2027):
        path = root / 'data' / 'factors_preprocessed' / 'raw' / f'csi300_factors_{year}.parquet'

        if not path.exists():
            raise FileNotFoundError(path)

        factors = pd.read_parquet(path).reset_index()
        factors['datetime'] = pd.to_datetime(factors['datetime'])
        factors['instrument'] = factors['instrument'].astype('string').str.upper()

        missing = set(args.factors) - set(factors.columns)

        if missing:
            raise ValueError(f'Missing factors in {year}: {sorted(missing)}')

        if factors.duplicated(['datetime', 'instrument']).any():
            raise ValueError(f'Duplicate factor keys in {year}')

        factors = factors[['datetime', 'instrument'] + args.factors]
        factors = factors.merge(exposures, on=['datetime', 'instrument'], how='left', validate='one_to_one')
        factors = factors.merge(labels, on=['datetime', 'instrument'], how='left', validate='one_to_one')

        for factor in args.factors:
            pieces = [neutralize_day(group, factor) for _, group in factors.groupby('datetime', sort=True)]

            transformed = pd.concat(pieces, ignore_index=True)
            transformed = transformed.rename(columns={
                'standardized': factor + '__zscore',
                'size_neutral': factor + '__size_neutral'
            })

            factors = factors.merge(transformed, on=['datetime', 'instrument'], how='left', validate='one_to_one')

        panels.append(factors)
        print(f'Processed {year}: {len(factors):,} rows', flush=True)

    result = pd.concat(panels, ignore_index=True)
    result.to_parquet(output / 'neutralized_exploratory.parquet', index=False)

    for split, (start, end) in splits.items():
        data = result.loc[result['datetime'].between(start, end)]

        for factor in args.factors:
            variants = [factor, factor + '__zscore', factor + '__size_neutral']
            common = data[variants + [args.label_column]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            paired = data.loc[common]

            for variant in variants:
                item = summarize_ic(paired, variant, args.label_column)
                item.update({
                    'split': split,
                    'base_factor': factor,
                    'variant': variant,
                    'matched_rows': int(len(paired)),
                    'total_rows': int(len(data)),
                    'sample_coverage': float(len(paired) / len(data)) if len(data) else 0.0
                })

                metrics.append(item)

    report = {
        'exploratory': True,
        'pit_certified': False,
        'test_is_blind': False,
        'label_column': args.label_column,
        'metrics': metrics
    }

    (output / 'rank_ic_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(metrics).to_csv(output / 'rank_ic_comparison.csv', index=False, encoding='utf-8-sig')

    print(json.dumps({
        'rows': len(result),
        'factor_count': len(args.factors),
        'comparison_rows': len(metrics),
        'output': str(output)
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

