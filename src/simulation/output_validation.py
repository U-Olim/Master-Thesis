"""Shared normalization and validation for current simulation outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ivqr.confidence_regions import parse_cr_components, validate_cr_geometry
from simulation.output_schemas import (
    CORE_IDENTIFIER_COLUMNS,
    CR_GEOMETRY_COLUMNS,
    DML_OUTPUT_COLUMNS,
    GRID_METADATA_COLUMNS,
)


BOOLEAN_COLUMNS: tuple[str, ...] = ("covered", "converged", "cr_disconnected")


def with_neutral_grid_metadata(source: pd.DataFrame) -> pd.DataFrame:
    """Add neutral grid and confidence-region metadata to legacy rows."""
    completed = source.copy()
    defaults: dict[str, object] = {
        "grid_strategy": "not_applicable",
        "adaptive_midpoint_probe": False,
        "alpha_hat_grid": "not_applicable",
        "midpoint_intervals_considered": 0,
        "midpoint_evaluations_added": 0,
        "midpoint_unresolved_barriers": 0,
        "midpoint_probe_limit_hit": False,
        "iteration_warning_evaluations": 0,
        "rank_deficient_covariance_failures": 0,
        "refinement_limit_hit": False,
        "max_alpha_evaluations_hit": False,
        "cr_components": None,
        "cr_n_blocks": np.nan,
        "cr_disconnected": np.nan,
        "cr_status": "unavailable",
        "cr_is_numerically_resolved": np.nan,
        "cr_unresolved_count": np.nan,
        "cr_unresolved_alphas": None,
        "result_schema_version": np.nan,
    }
    for column in (
        *GRID_METADATA_COLUMNS,
        *CR_GEOMETRY_COLUMNS,
        "result_schema_version",
    ):
        if column not in completed.columns:
            completed[column] = defaults.get(column, np.nan)
    return completed


def as_boolean(series: pd.Series, column: str) -> pd.Series:
    """Return a nullable Boolean series, rejecting non-Boolean values."""
    present = series.dropna()
    valid = present.map(lambda value: isinstance(value, (bool, np.bool_)))
    if not valid.all():
        value = present.loc[~valid].iloc[0]
        raise ValueError(f"{column} contains a non-Boolean value: {value!r}")
    return series.astype("boolean")


def numeric(series: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & values.isna()
    if invalid.any():
        value = series.loc[invalid].iloc[0]
        raise ValueError(f"{column} contains a non-numeric value: {value!r}")
    return values


def validate_integer_parameter(
    frame: pd.DataFrame, column: str, *, minimum: int
) -> None:
    values = numeric(frame[column], column)
    if values.isna().any():
        raise ValueError(f"{column} must not be missing")
    present = values.dropna()
    if present.empty:
        return
    if (~np.isfinite(present)).any() or (present < minimum).any():
        raise ValueError(f"{column} must be finite and >= {minimum}")
    if (present != np.floor(present)).any():
        raise ValueError(f"{column} must be integer-valued")


def validate_parameters(frame: pd.DataFrame) -> None:
    tau_values = numeric(frame["tau"], "tau")
    if tau_values.isna().any():
        raise ValueError("tau must not be missing")
    tau = tau_values.dropna()
    if (~np.isfinite(tau)).any() or ((tau <= 0) | (tau >= 1)).any():
        raise ValueError("tau must be finite and strictly between zero and one")
    validate_integer_parameter(frame, "n", minimum=1)
    validate_integer_parameter(frame, "p", minimum=0)
    validate_integer_parameter(frame, "rep", minimum=0)
    validate_integer_parameter(frame, "seed", minimum=0)


def validate_component_columns(frame: pd.DataFrame) -> None:
    """Validate serialized components against row-level hull diagnostics."""
    lower = numeric(frame["cr_lower"], "cr_lower")
    upper = numeric(frame["cr_upper"], "cr_upper")
    length = numeric(frame["cr_length"], "cr_length")
    for index, value in frame["cr_components"].items():
        components = parse_cr_components(value)
        if components is None:
            continue
        n_blocks_value = frame.at[index, "cr_n_blocks"]
        validate_cr_geometry(
            components,
            lower=None if pd.isna(lower.at[index]) else float(lower.at[index]),
            upper=None if pd.isna(upper.at[index]) else float(upper.at[index]),
            length=None if pd.isna(length.at[index]) else float(length.at[index]),
            n_blocks=None if pd.isna(n_blocks_value) else int(float(n_blocks_value)),
            disconnected=frame.at[index, "cr_disconnected"],
        )


def validate_confidence_regions(frame: pd.DataFrame) -> None:
    lower = numeric(frame["cr_lower"], "cr_lower")
    upper = numeric(frame["cr_upper"], "cr_upper")
    length = numeric(frame["cr_length"], "cr_length")
    alpha_true = numeric(frame["alpha_true"], "alpha_true")
    missing = pd.concat([lower.isna(), upper.isna(), length.isna()], axis=1)
    partial = missing.any(axis=1) & ~missing.all(axis=1)
    if partial.any():
        raise ValueError(
            f"{int(partial.sum())} rows have partially missing confidence regions"
        )
    finite = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(length)
    if (length.loc[finite] < 0).any():
        raise ValueError("cr_length must be nonnegative where finite")
    if (upper.loc[finite] < lower.loc[finite]).any():
        raise ValueError("cr_upper must be at least cr_lower where finite")
    comparable = finite & np.isfinite(alpha_true) & frame["covered"].notna()
    expected = (lower <= alpha_true) & (alpha_true <= upper)
    actual = frame["covered"].fillna(False).astype(bool)
    mismatch = comparable & actual & ~expected
    if mismatch.any():
        raise ValueError(
            "covered=True is inconsistent with finite confidence bounds in "
            f"{int(mismatch.sum())} rows"
        )
    empty = missing.all(axis=1)
    if (empty & actual).any():
        raise ValueError("an empty confidence region cannot have covered=True")
    validate_component_columns(frame)


def assert_retained_identity(
    source: pd.DataFrame,
    cleaned: pd.DataFrame,
    columns: Sequence[str],
    source_columns: Mapping[str, str],
    boolean_columns: Sequence[str],
) -> None:
    for column in columns:
        source_column = source_columns.get(column, column)
        left = source[source_column]
        right = cleaned[column]
        if column in boolean_columns:
            left = as_boolean(left, source_column)
            right = as_boolean(right, column)
        pd.testing.assert_series_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_names=False,
            check_exact=True,
        )


def clean_common_results_frame(source: pd.DataFrame, *, estimator: str) -> pd.DataFrame:
    """Select and validate the current common diagnostic output schema."""
    if not isinstance(estimator, str) or not estimator:
        raise ValueError("estimator must be a nonempty string")
    if not isinstance(source, pd.DataFrame):
        raise TypeError("source must be a pandas DataFrame")
    if source.columns.duplicated().any():
        duplicates = sorted(set(source.columns[source.columns.duplicated()].astype(str)))
        raise ValueError(f"source has duplicate columns: {duplicates}")
    completed = with_neutral_grid_metadata(source)
    source_columns = {column: column for column in DML_OUTPUT_COLUMNS}
    source_columns["covered"] = "cr_covers_true"
    missing = sorted(
        column
        for column, source_column in source_columns.items()
        if source_column not in completed.columns
    )
    if missing:
        raise ValueError(f"source is missing required core columns: {missing}")
    cleaned = pd.DataFrame(
        {
            column: completed[source_column]
            for column, source_column in source_columns.items()
        }
    )
    for column in BOOLEAN_COLUMNS:
        cleaned[column] = as_boolean(cleaned[column], column)
    if tuple(cleaned.columns) != DML_OUTPUT_COLUMNS:
        raise AssertionError("clean common-core schema construction failed")
    if len(cleaned) != len(completed):
        raise AssertionError("common-core cleaning changed the row count")
    if cleaned["estimator"].dropna().astype(str).ne(estimator).any():
        raise ValueError(f"clean output contains an estimator other than {estimator}")
    validate_parameters(cleaned)
    validate_confidence_regions(cleaned)
    duplicate_mask = cleaned.duplicated(list(CORE_IDENTIFIER_COLUMNS), keep=False)
    duplicate_identifiers = int(
        cleaned.loc[duplicate_mask, CORE_IDENTIFIER_COLUMNS].drop_duplicates().shape[0]
    )
    if duplicate_identifiers:
        raise ValueError(
            f"{duplicate_identifiers} duplicate {estimator} identifiers affect "
            f"{int(duplicate_mask.sum())} rows; key columns are "
            f"{list(CORE_IDENTIFIER_COLUMNS)}"
        )
    assert_retained_identity(
        completed,
        cleaned,
        DML_OUTPUT_COLUMNS,
        source_columns,
        BOOLEAN_COLUMNS,
    )
    return cleaned


__all__ = [
    "BOOLEAN_COLUMNS",
    "as_boolean",
    "assert_retained_identity",
    "clean_common_results_frame",
    "numeric",
    "validate_component_columns",
    "validate_confidence_regions",
    "validate_integer_parameter",
    "validate_parameters",
    "with_neutral_grid_metadata",
]
