"""Description: Attribute Qlib price-audit failures to missing keys and invalid numeric values.

Parameters: Configuration paths are provided through command-line arguments.
Returns: CSV diagnostics and a JSON summary; original research data is never modified.
Notes: Missing Qlib observations cannot independently establish a trading suspension or listing status.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

FIELDS = ['$open', '$close', '$volume', '$factor']


def classify_price(value, matched):
    """Description: Classify a Qlib numeric observation.

    Parameters: value (object): Numeric value or missing value; matched (bool): Whether a Qlib key exists.
    Returns: str: One of missing_key, missing_value, non_numeric, infinite, zero, negative, or valid.
    Notes: Classification does not imply a specific economic reason for missingness.
    """
    if not matched:
        return 'missing_key'
    if pd.isna(value):
        return 'missing_value'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 'non_numeric'
    if not np.isfinite(number):
        return 'infinite'
    if number == 0:
        return 'zero'
    return 'negative' if number < 0 else 'valid'


def prepare_audit(audit):
    """Description: Normalize and deduplicate the price-audit stock-date keys.

    Parameters: audit (pd.DataFrame): Audit with instrument, datetime, and issue columns.
    Returns: pd.DataFrame: Unique normalized stock-date keys with issue indicators.
    Notes: Duplicate issue records are preserved as distinct indicators, not duplicate keys.
    """
    required = {'instrument', 'datetime', 'issue'}
    if not required.issubset(audit.columns):
        raise ValueError(f'Missing audit columns: {sorted(required - set(audit.columns))}')
    data = audit.copy()
    data['datetime'] = pd.to_datetime(data['datetime'], errors='raise').dt.normalize()
    if data[['instrument', 'datetime']].isna().any().any():
        raise ValueError('Audit keys contain missing values')
    result = data.groupby(['instrument', 'datetime'], as_index=False)['issue'].agg(lambda s: '|'.join(sorted(set(s.astype(str)))))
    return result.rename(columns={'issue': 'audit_issues'})


def attribute_prices(keys, quotes):
    """Description: Left-join audit keys with Qlib quotes and classify every price field.

    Parameters: keys (pd.DataFrame): Unique audit keys; quotes (pd.DataFrame): Qlib quotes with matching keys and four fields.
    Returns: pd.DataFrame: Audited keys, raw Qlib values, join status, and per-field classifications.
    Notes: Enforces one-to-one joins to prevent accidental multiplication of observations.
    """
    required = {'instrument', 'datetime', *FIELDS}
    if not required.issubset(quotes.columns):
        raise ValueError(f'Missing quote columns: {sorted(required - set(quotes.columns))}')
    q = quotes.copy()
    q['datetime'] = pd.to_datetime(q['datetime'], errors='raise').dt.normalize()
    if q.duplicated(['instrument', 'datetime']).any():
        raise ValueError('Duplicate Qlib quote keys detected')
    result = keys.merge(q, on=['instrument', 'datetime'], how='left', validate='one_to_one', indicator=True)
    matched = result['_merge'].eq('both').to_numpy()
    for field in FIELDS:
        result[field[1:] + '_state'] = [classify_price(value, exists) for value, exists in zip(result[field], matched)]
    result['year'] = result['datetime'].dt.year
    return result


def load_qlib_quotes(stocks, start, end, qlib_dir, batch_size=80):
    """Description: Fetch Qlib prices for all audited instruments in bounded batches.

    Parameters: stocks (list[str]): Instruments; start (str): First date; end (str): Last date; qlib_dir (str): Qlib provider path; batch_size (int): Instruments per request.
    Returns: pd.DataFrame: All returned stock-date price observations.
    Notes: Qlib is used only for data extraction; no forward-filling is applied.
    """
    import qlib
    from qlib.data import D
    qlib.init(provider_uri=qlib_dir, region='cn')
    frames = []
    for index in range(0, len(stocks), batch_size):
        batch = stocks[index:index + batch_size]
        chunk = D.features(batch, FIELDS, start_time=start, end_time=end).reset_index()
        frames.append(chunk)
        print(f'Fetched {min(index + batch_size, len(stocks))}/{len(stocks)} instruments', flush=True)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['instrument', 'datetime', *FIELDS])


def run(args):
    """Description: Run read-only attribution and save aggregated diagnostics.

    Parameters: args (argparse.Namespace): Paths and fetch batch size.
    Returns: dict: JSON-serializable diagnostic summary.
    Notes: Membership intervals indicate index eligibility, not verified PIT availability or actual listing/trading status.
    """
    audit = pd.read_csv(args.audit_csv)
    keys = prepare_audit(audit)
    quotes = load_qlib_quotes(sorted(keys['instrument'].unique()), str(keys['datetime'].min().date()), str(keys['datetime'].max().date()), args.qlib_dir, args.batch_size)
    result = attribute_prices(keys, quotes)
    result['all_four_missing_key'] = result[[f[1:] + '_state' for f in FIELDS]].eq('missing_key').all(axis=1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / 'price_missingness_details.csv', index=False)
    by_year = result.groupby('year', as_index=False).agg(audit_keys=('instrument', 'size'), unmatched_qlib_keys=('all_four_missing_key', 'sum'))
    by_year.to_csv(output / 'price_missingness_by_year.csv', index=False)
    by_stock = result.groupby('instrument', as_index=False).agg(audit_keys=('datetime', 'size'), unmatched_qlib_keys=('all_four_missing_key', 'sum'), first_date=('datetime', 'min'), last_date=('datetime', 'max'))
    by_stock.sort_values('audit_keys', ascending=False).to_csv(output / 'price_missingness_by_stock.csv', index=False)
    states = pd.concat([result[['instrument', 'datetime', 'year', f[1:] + '_state']].rename(columns={f[1:] + '_state': 'state'}).assign(field=f[1:]) for f in FIELDS], ignore_index=True)
    states.groupby(['field', 'state'], as_index=False).size().rename(columns={'size': 'count'}).to_csv(output / 'price_missingness_states.csv', index=False)
    summary = {'audit_issue_rows': int(len(audit)), 'unique_audit_keys': int(len(keys)), 'qlib_unmatched_keys': int(result['all_four_missing_key'].sum()), 'qlib_matched_keys': int((result['_merge'] == 'both').sum()), 'affected_stocks': int(keys['instrument'].nunique()), 'field_states': {field[1:]: result[field[1:] + '_state'].value_counts().to_dict() for field in FIELDS}, 'limitations': ['Missing Qlib keys do not prove suspensions or delistings', 'Audit includes dates outside historical CSI300 membership unless separately filtered', 'Corporate actions and raw price correctness are not independently verified']}
    (output / 'price_missingness_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('Reports:', output)
    return summary


def main():
    """Description: Parse CLI options and execute price missingness attribution.

    Parameters: None.
    Returns: None.
    Notes: Default paths follow the existing CSI300 research project layout.
    """
    parser = argparse.ArgumentParser(description='Read-only Qlib price missingness attribution')
    parser.add_argument('--audit-csv', default=r'D:\QuantResearch\CSI300_MultiFactor_Research\reports\label_engineering\price_audit.csv')
    parser.add_argument('--qlib-dir', default=r'D:\QuantResearch\qlib_data\cn_data\qlib_bin')
    parser.add_argument('--output-dir', default=r'D:\QuantResearch\CSI300_MultiFactor_Research\reports\price_missingness_attribution')
    parser.add_argument('--batch-size', type=int, default=80)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    run(args)


if __name__ == '__main__':
    main()
