"""Execute the CSI300 factor quality audit from the project root."""

from config import DATA_OUTPUT_DIR, DATA_ROOT, REPORT_OUTPUT_DIR, START_DATE, END_DATE
from factors.factor_audit import run_factor_audit


def run_complete_factor_audit():
    """Audit the generated factor panel against its original price panel.

    Parameters:
        None.

    Returns:
        dict[str, pd.DataFrame]: Data integrity, coverage, and continuity reports.

    Notes:
        The local exchange calendar is obtained from the dates available in Qlib.
        The data import path below must match the actual data module layout.
    """
    from data.data_loader import initialize_qlib, get_trading_calendar
    initialize_qlib()
    calendar = get_trading_calendar(START_DATE, END_DATE)
    reports = run_factor_audit(
        DATA_OUTPUT_DIR,
        DATA_ROOT / "data" / "factors",
        REPORT_OUTPUT_DIR / "factor_quality",
        calendar,
        int(START_DATE[:4]),
        int(END_DATE[:4]),
    )
    print("\nINTEGRITY\n", reports["integrity"].to_string(index=False))
    print("\nYEARLY COVERAGE\n", reports["year_summary"].to_string(index=False))
    print(
        "\nLOWEST COVERAGE FACTOR-YEARS\n",
        reports["factor_year"]
        .nsmallest(15, "coverage")[["year", "factor", "coverage", "nan_count", "inf_count"]]
        .to_string(index=False),
    )
    return reports


if __name__ == "__main__":
    audit_reports = run_complete_factor_audit()
