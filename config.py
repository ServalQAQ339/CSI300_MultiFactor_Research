
"""
Central configuration for the CSI300 multi-factor stock selection project.

This module defines research dates, Qlib data paths, output directories,
model validation periods, and initial portfolio parameters. Importing it has no
filesystem side effects; each writer creates its own output folder.

Path resolution, highest priority first: the CSI300_DATA_ROOT and CSI300_QLIB_DIR environment
variables; DEFAULT_DATA_ROOT and DEFAULT_QLIB_DATA_DIR in an optional, untracked local_config.py
next to this file; otherwise <repository>/research_data and the standard Qlib location
~/.qlib/qlib_data/cn_data. CLI path defaults across the project are derived from these values.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    # Machine-specific defaults; local_config.py is git-ignored.
    from local_config import DEFAULT_DATA_ROOT, DEFAULT_QLIB_DATA_DIR
except ImportError:
    DEFAULT_DATA_ROOT = str(PROJECT_ROOT / "research_data")
    DEFAULT_QLIB_DATA_DIR = str(Path.home() / ".qlib" / "qlib_data" / "cn_data")

# Project data root (the --data-root meaning) and Qlib binary provider.
DATA_ROOT = Path(os.environ.get("CSI300_DATA_ROOT", DEFAULT_DATA_ROOT))
QLIB_DATA_DIR = Path(os.environ.get("CSI300_QLIB_DIR", DEFAULT_QLIB_DATA_DIR))

# The data subfolder (the --data-dir meaning).
DATA_DIR = DATA_ROOT / "data"
DATA_OUTPUT_DIR = DATA_DIR / "processed"
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