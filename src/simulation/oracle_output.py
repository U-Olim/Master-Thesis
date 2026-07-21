"""Serialize Oracle IVQR results for CSV output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from simulation.output_schemas import (
    ORACLE_DESIGN_KEY_COLUMNS,
    ORACLE_OUTPUT_COLUMNS,
)
from simulation.output_validation import validate_component_columns

_SOURCE_ALIASES = {"covered": "cr_covers_true"}


def _source_name(columns: Any, output_column: str) -> str:
    if output_column in columns:
        return output_column
    alias = _SOURCE_ALIASES.get(output_column)
    if alias is not None and alias in columns:
        return alias
    return output_column


def serialize_oracle_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Project one expanded or current Oracle result onto the output schema."""
    sources = {
        column: _source_name(result, column) for column in ORACLE_OUTPUT_COLUMNS
    }
    missing = [
        column for column, source in sources.items() if source not in result
    ]
    if missing:
        raise ValueError(
            f"Oracle result is missing required output columns: {missing}"
        )
    return {column: result[source] for column, source in sources.items()}


def clean_oracle_results_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Project and validate expanded or current Oracle result rows."""
    if not isinstance(source, pd.DataFrame):
        raise TypeError("source must be a pandas DataFrame")
    if source.columns.duplicated().any():
        duplicates = sorted(set(source.columns[source.columns.duplicated()].astype(str)))
        raise ValueError(f"source has duplicate columns: {duplicates}")

    sources = {
        column: _source_name(source.columns, column)
        for column in ORACLE_OUTPUT_COLUMNS
    }
    missing = [
        column for column, source_column in sources.items()
        if source_column not in source.columns
    ]
    if missing:
        raise ValueError(
            f"Oracle result is missing required output columns: {missing}"
        )

    cleaned = pd.DataFrame(
        {column: source[source_column] for column, source_column in sources.items()}
    )
    if tuple(cleaned.columns) != ORACLE_OUTPUT_COLUMNS:
        raise AssertionError("Oracle output schema construction failed")
    if len(cleaned) != len(source):
        raise AssertionError("Oracle projection changed the row count")

    required_complete = [
        *ORACLE_DESIGN_KEY_COLUMNS,
        "alpha_true",
        "seed",
        "converged",
        "cr_status",
        "cr_n_blocks",
        "cr_unresolved_count",
        "iteration_warning_evaluations",
    ]
    if cleaned[required_complete].isna().any().any():
        incomplete = cleaned[required_complete].columns[
            cleaned[required_complete].isna().any()
        ].tolist()
        raise ValueError(f"Oracle required values are missing: {incomplete}")

    for column in (
        "n",
        "p",
        "rep",
        "seed",
        "cr_n_blocks",
        "cr_unresolved_count",
        "iteration_warning_evaluations",
    ):
        values = pd.to_numeric(cleaned[column], errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any():
            raise ValueError(f"Oracle {column} must be finite and numeric")
        if (values != np.floor(values)).any() or (values < 0).any():
            raise ValueError(f"Oracle {column} must be a nonnegative integer")
    if (pd.to_numeric(cleaned["n"]) < 1).any():
        raise ValueError("Oracle n must be positive")
    for column in (
        "final_alpha_evaluations",
        "refinement_depth_reached",
        "number_of_refined_intervals",
    ):
        values = pd.to_numeric(cleaned[column], errors="coerce")
        invalid = cleaned[column].notna() & values.isna()
        if invalid.any() or (~np.isfinite(values.dropna())).any():
            raise ValueError(f"Oracle {column} must be finite and numeric where present")
        if ((values.dropna() != np.floor(values.dropna())) | (values.dropna() < 0)).any():
            raise ValueError(
                f"Oracle {column} must be a nonnegative integer where present"
            )
    for column in ("pi", "tau", "alpha_true"):
        values = pd.to_numeric(cleaned[column], errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any():
            raise ValueError(f"Oracle {column} must be finite and numeric")
    alpha_hat = pd.to_numeric(cleaned["alpha_hat"], errors="coerce")
    invalid_alpha_hat = cleaned["alpha_hat"].notna() & alpha_hat.isna()
    if invalid_alpha_hat.any() or (~np.isfinite(alpha_hat.dropna())).any():
        raise ValueError("Oracle alpha_hat must be finite and numeric where present")
    tau = pd.to_numeric(cleaned["tau"], errors="coerce")
    if tau.isna().any() or ((tau <= 0) | (tau >= 1)).any():
        raise ValueError("Oracle tau must lie strictly between zero and one")
    length = pd.to_numeric(cleaned["cr_length"], errors="coerce")
    invalid_length = cleaned["cr_length"].notna() & length.isna()
    if invalid_length.any() or (~np.isfinite(length.dropna())).any():
        raise ValueError("Oracle cr_length must be finite and numeric where present")
    if (length.dropna() < 0).any():
        raise ValueError("Oracle cr_length must be nonnegative where present")
    for column in ("minimum_final_grid_spacing", "median_final_grid_spacing"):
        values = pd.to_numeric(cleaned[column], errors="coerce")
        invalid = cleaned[column].notna() & values.isna()
        if invalid.any() or (~np.isfinite(values.dropna())).any():
            raise ValueError(f"Oracle {column} must be finite and numeric where present")
        if (values.dropna() < 0).any():
            raise ValueError(f"Oracle {column} must be nonnegative where present")
    present_covered = cleaned["covered"].dropna()
    if not present_covered.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Oracle covered contains a non-Boolean value")
    for column in ("converged", "cr_disconnected", "cr_is_numerically_resolved"):
        present = cleaned[column].dropna()
        if not present.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f"Oracle {column} contains a non-Boolean value")

    validate_component_columns(cleaned)

    duplicate_count = int(
        cleaned.duplicated(list(ORACLE_DESIGN_KEY_COLUMNS), keep=False).sum()
    )
    if duplicate_count:
        raise ValueError(
            f"Oracle result contains {duplicate_count} rows with duplicate simulation "
            "keys (natural keys)"
        )
    return cleaned


__all__ = [
    "ORACLE_DESIGN_KEY_COLUMNS",
    "ORACLE_OUTPUT_COLUMNS",
    "clean_oracle_results_frame",
    "serialize_oracle_result",
]
