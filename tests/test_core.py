"""Unit tests for point-in-time guardrails and exposure arithmetic."""
import pandas as pd
from factor_neutralization.exposure_data.core import normalize_shares,build_share_panel,build_market_cap_panel,normalize_industry


def test_no_future_share_leakage():
    """Description: Ensure effective and announcement dates are both respected.

    Parameters: None.
    Returns: None.
    Notes: Date-only availability begins on the next calendar day.
    """
    raw=pd.DataFrame([{'证券代码':'000001','公告日期':'2015-04-07','变动日期':'2015-04-13','变动原因':'转增','总股本':100,'已流通股份':80}])
    events=normalize_shares(raw,1.0); signals=pd.DataFrame({'instrument':['SZ000001']*3,'datetime':pd.to_datetime(['2015-04-10','2015-04-13','2015-04-14'])})
    out=build_share_panel(events,signals); assert pd.isna(out.loc[0,'total_shares']); assert pd.isna(out.loc[1,'total_shares']); assert out.loc[2,'total_shares']==100


def test_periodic_report_not_backdated():
    """Description: Ensure late periodic-report disclosure is not applied at statement date.

    Parameters: None.
    Returns: None.
    Notes: Missing pre-disclosure shares remain missing.
    """
    raw=pd.DataFrame([{'证券代码':'000001','公告日期':'2015-08-14','变动日期':'2015-06-30','变动原因':'定期报告','总股本':100,'已流通股份':80}])
    out=build_share_panel(normalize_shares(raw,1.0),pd.DataFrame({'instrument':['000001']*2,'datetime':pd.to_datetime(['2015-07-01','2015-08-15'])}))
    assert pd.isna(out.loc[0,'total_shares']); assert out.loc[1,'total_shares']==100


def test_market_cap_and_industry_flags():
    """Description: Validate market-cap arithmetic and explicit industry PIT warning.

    Parameters: None.
    Returns: None.
    Notes: Price is assumed to be independently verified by the caller.
    """
    shares=pd.DataFrame({'instrument':['000001'],'datetime':pd.to_datetime(['2020-01-02']),'total_shares':[1e9],'float_shares':[8e8]})
    prices=pd.DataFrame({'instrument':['000001'],'datetime':pd.to_datetime(['2020-01-02']),'raw_close':[20.]})
    out=build_market_cap_panel(shares,prices); assert out.loc[0,'total_market_cap']==2e10
    raw=pd.DataFrame({'股票代码':[1],'计入日期':['2014-02-21'],'行业代码':['480101'],'更新日期':['2024-09-27']})
    assert normalize_industry(raw).loc[0,'pit_status']=='RETROSPECTIVE_INDUSTRY_UNVERIFIED_PIT'
