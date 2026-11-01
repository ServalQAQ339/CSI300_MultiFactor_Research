"""Run the CSI300 enhanced factor audit from the project root."""

#%%
from config import DATA_OUTPUT_DIR, DATA_ROOT, END_DATE
from factors.enhanced_audit import run_enhanced_audit


#%%
def run_complete_enhanced_audit(prefix_check: bool = True):
    """Run baseline comparison and sampled historical-prefix checks.

    Parameters:
        prefix_check (bool): Whether to query Qlib again for past-only validation.

    Returns:
        dict[str, pd.DataFrame]: Comparison tables and optional prefix diagnostics.

    Notes:
        Requires the existing baseline and enhanced yearly factor Parquet files.
        The Qlib calendar begins in 2014 to cover the 2015 warm-up period.
    """
    calendar = None

    if prefix_check:
        try:
            from data.process import initialize_qlib, get_trading_calendar
        except ImportError:
            from data.data_loader import initialize_qlib, get_trading_calendar
        initialize_qlib()
        calendar = get_trading_calendar("2014-01-01", END_DATE)

    results = run_enhanced_audit(
        DATA_OUTPUT_DIR,
        DATA_ROOT / "data" / "factors",
        DATA_ROOT / "data" / "factors_enhanced",
        DATA_ROOT / "reports" / "factor_quality_enhanced",
        calendar=calendar,
        prefix_check=prefix_check,
    )
    print("\nYEAR COMPARISON\n", results["year_comparison"].to_string(index=False))

    if prefix_check:
        print(
            "\nPREFIX MISMATCHES\n",
            results["prefix_consistency"][["missing_mismatch", "value_mismatch"]].sum().to_string(),
        )

    return results


#%%
if __name__ == "__main__":
    results = run_complete_enhanced_audit(prefix_check=True)
