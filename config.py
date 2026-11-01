
"""
Central configuration for the CSI300 multi-factor stock selection project.

This module defines research dates, Qlib data paths, output directories,
model validation periods, and initial portfolio parameters.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(r"D:\QuantResearch\CSI300_MultiFactor_Research")
QLIB_DATA_DIR = Path(r"D:\QuantResearch\qlib_data\cn_data\qlib_bin")

DATA_OUTPUT_DIR = DATA_ROOT / "data" / "processed"
REPORT_OUTPUT_DIR = DATA_ROOT / "reports"
AUDIT_OUTPUT_DIR = REPORT_OUTPUT_DIR / "data_quality"

START_DATE = "2015-01-01"
END_DATE = "2026-09-30"

FACTOR_WARMUP_START = START_DATE
RESEARCH_START = "2015-04-01"

TRAIN_START = RESEARCH_START
TRAIN_END = "2020-12-31"

MARKET = "csi300"
BENCHMARK = "SH000300"

VALID_START = "2021-01-01"
VALID_END = "2023-12-31"

TEST_START = "2024-01-01"
TEST_END = END_DATE

HORIZON = 1
TOPK = 30
N_DROP = 5

for directory in [DATA_OUTPUT_DIR, REPORT_OUTPUT_DIR, AUDIT_OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)