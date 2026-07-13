"""Clean and validate thesis-ready Oracle IVQR outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from simulation.dml_output import (
    CORE_IDENTIFIER_COLUMNS,
    CoreValidationSummary,
    REQUIRED_CORE_COLUMNS,
    clean_core_results_csv,
    clean_core_results_frame,
)


REQUIRED_ORACLE_COLUMNS = REQUIRED_CORE_COLUMNS
ORACLE_IDENTIFIER_COLUMNS = CORE_IDENTIFIER_COLUMNS
OracleValidationSummary = CoreValidationSummary


def clean_oracle_results_frame(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, OracleValidationSummary]:
    """Select, standardize, and strictly validate an Oracle result DataFrame."""
    return clean_core_results_frame(source, estimator="oracle")


def clean_oracle_results_csv(
    input_path: str | Path, output_path: str | Path
) -> OracleValidationSummary:
    """Clean a historical Oracle CSV without modifying its source file."""
    return clean_core_results_csv(input_path, output_path, estimator="oracle")


__all__ = [
    "ORACLE_IDENTIFIER_COLUMNS",
    "OracleValidationSummary",
    "REQUIRED_ORACLE_COLUMNS",
    "clean_oracle_results_csv",
    "clean_oracle_results_frame",
]
