"""Run extended-history factor generation with a separate enhanced output folder."""

#%%
from config import DATA_OUTPUT_DIR, DATA_ROOT, END_DATE
from factors.factor_backfill import run_backfilled_factor_pipeline


#%%
def run_complete_backfill(overwrite: bool = False):
    """Generate enhanced factors with pre-membership historical OHLCV.

    Parameters:
        overwrite (bool): Whether to regenerate existing enhanced factor files.

    Returns:
        pd.DataFrame: Annual generation summary.

    Notes:
        The calendar begins in 2014 to support 2015 warm-up calculations.
        The original baseline factor files remain untouched.
    """
    try:
        from data.process import initialize_qlib, get_trading_calendar
    except ImportError:
        from data.data_loader import initialize_qlib, get_trading_calendar
    initialize_qlib()
    calendar = get_trading_calendar("2014-01-01", END_DATE)
    summary = run_backfilled_factor_pipeline(
        DATA_OUTPUT_DIR,
        DATA_ROOT / "data" / "factors_enhanced",
        calendar,
        2015,
        int(END_DATE[:4]),
        overwrite,
    )
    print(summary.to_string(index=False))
    return summary


#%%
if __name__ == "__main__":
    result = run_complete_backfill(overwrite=False)
