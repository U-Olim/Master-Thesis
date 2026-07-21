"""Serialize and validate current DML IVQR outputs."""

from __future__ import annotations

import pandas as pd

from simulation.output_schemas import DML_OUTPUT_COLUMNS
from simulation.output_validation import clean_common_results_frame


REQUIRED_DML_COLUMNS = DML_OUTPUT_COLUMNS


def clean_dml_results_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Select, standardize, and strictly validate a DML result DataFrame."""
    return clean_common_results_frame(source, estimator="dml_ivqr")


__all__ = [
    "DML_OUTPUT_COLUMNS",
    "REQUIRED_DML_COLUMNS",
    "clean_dml_results_frame",
]
