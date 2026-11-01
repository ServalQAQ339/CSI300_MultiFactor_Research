"""Run point-in-time preprocessing diagnostics on cached CSI300 factor panels."""

from config import DATA_ROOT
from preprocessing_validation.preprocessing_validator import run_validation

if __name__ == "__main__":
    run_validation(data_root=DATA_ROOT, research_start="2015-04-01")
