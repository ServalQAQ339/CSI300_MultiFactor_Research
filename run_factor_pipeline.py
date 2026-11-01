
from config import DATA_OUTPUT_DIR, DATA_ROOT, START_DATE, END_DATE
from data.data_loader import initialize_qlib, get_trading_calendar
from factors.factor_engine import run_factor_engine

initialize_qlib()

calendar = get_trading_calendar(START_DATE, END_DATE)

summary = run_factor_engine(
    data_dir=DATA_OUTPUT_DIR,
    output_dir=DATA_ROOT / "data" / "factors",
    calendar=calendar,
    start_year=2015,
    end_year=2026,
    overwrite=False
)

print(summary.to_string(index=False))