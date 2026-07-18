"""Build and validate current common-schema simulation outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ivqr.confidence_regions import parse_cr_components, validate_cr_geometry


GRID_METADATA_COLUMNS: tuple[str, ...] = (
    "grid_strategy",
    "adaptive_midpoint_probe",
    "alpha_hat_grid",
    "midpoint_intervals_considered",
    "midpoint_evaluations_added",
    "midpoint_unresolved_barriers",
    "midpoint_probe_limit_hit",
    "initial_alpha_grid_size",
    "final_alpha_evaluations",
    "refinement_tolerance",
    "refinement_depth_reached",
    "refinement_limit_hit",
    "max_alpha_evaluations_hit",
    "number_of_refined_intervals",
    "number_of_unresolved_refinement_barriers",
    "minimum_final_grid_spacing",
    "median_final_grid_spacing",
    "maximum_final_grid_spacing",
    "iteration_warning_evaluations",
    "rank_deficient_covariance_failures",
)
CR_GEOMETRY_COLUMNS: tuple[str, ...] = (
    "cr_components",
    "cr_n_blocks",
    "cr_disconnected",
    "cr_status",
    "cr_is_numerically_resolved",
    "cr_unresolved_count",
    "cr_unresolved_alphas",
)

REQUIRED_CORE_COLUMNS: tuple[str, ...] = (
    "dgp",
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "seed",
    "result_schema_version",
    "estimator",
    "alpha_hat",
    "alpha_true",
    "cr_lower",
    "cr_upper",
    "cr_length",
    "covered",
    "converged",
    *CR_GEOMETRY_COLUMNS,
    *GRID_METADATA_COLUMNS,
)
REQUIRED_DML_COLUMNS = REQUIRED_CORE_COLUMNS

CORE_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "dgp",
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "estimator",
)
_BOOLEAN_COLUMNS = ("covered", "converged", "cr_disconnected")
_SOURCE_COLUMNS = {"covered": "cr_covers_true"}


def with_neutral_grid_metadata(source: pd.DataFrame) -> pd.DataFrame:
    """Add neutral grid metadata to legacy inputs and DML rows."""
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


def _as_boolean(series: pd.Series, column: str) -> pd.Series:
    """Return a nullable Boolean series, rejecting non-Boolean values."""
    present = series.dropna()
    valid = present.map(lambda value: isinstance(value, (bool, np.bool_)))
    if not valid.all():
        value = present.loc[~valid].iloc[0]
        raise ValueError(f"{column} contains a non-Boolean value: {value!r}")
    return series.astype("boolean")


def _numeric(series: pd.Series, column: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & values.isna()
    if invalid.any():
        value = series.loc[invalid].iloc[0]
        raise ValueError(f"{column} contains a non-numeric value: {value!r}")
    return values


def _validate_integer_parameter(
    frame: pd.DataFrame, column: str, *, minimum: int
) -> None:
    values = _numeric(frame[column], column)
    if values.isna().any():
        raise ValueError(f"{column} must not be missing")
    present = values.dropna()
    if present.empty:
        return
    if (~np.isfinite(present)).any() or (present < minimum).any():
        raise ValueError(f"{column} must be finite and >= {minimum}")
    if (present != np.floor(present)).any():
        raise ValueError(f"{column} must be integer-valued")


def _validate_parameters(frame: pd.DataFrame) -> None:
    tau_values = _numeric(frame["tau"], "tau")
    if tau_values.isna().any():
        raise ValueError("tau must not be missing")
    tau = tau_values.dropna()
    if (~np.isfinite(tau)).any() or ((tau <= 0) | (tau >= 1)).any():
        raise ValueError("tau must be finite and strictly between zero and one")
    _validate_integer_parameter(frame, "n", minimum=1)
    _validate_integer_parameter(frame, "p", minimum=0)
    _validate_integer_parameter(frame, "rep", minimum=0)
    _validate_integer_parameter(frame, "seed", minimum=0)


def _validate_confidence_regions(frame: pd.DataFrame) -> None:
    lower = _numeric(frame["cr_lower"], "cr_lower")
    upper = _numeric(frame["cr_upper"], "cr_upper")
    length = _numeric(frame["cr_length"], "cr_length")
    alpha_true = _numeric(frame["alpha_true"], "alpha_true")

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
    # A disconnected confidence region can exclude alpha_true even when it lies
    # inside the hull [cr_lower, cr_upper].  The reverse is never consistent.
    mismatch = comparable & actual & ~expected
    if mismatch.any():
        raise ValueError(
            f"covered=True is inconsistent with finite confidence bounds in "
            f"{int(mismatch.sum())} rows"
        )

    empty = missing.all(axis=1)
    invalid_empty_coverage = empty & frame["covered"].fillna(False).astype(bool)
    if invalid_empty_coverage.any():
        raise ValueError("an empty confidence region cannot have covered=True")

    validate_component_columns(frame)


def validate_component_columns(frame: pd.DataFrame) -> None:
    """Validate serialized components against row-level hull diagnostics."""
    lower = _numeric(frame["cr_lower"], "cr_lower")
    upper = _numeric(frame["cr_upper"], "cr_upper")
    length = _numeric(frame["cr_length"], "cr_length")
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
            n_blocks=(
                None if pd.isna(n_blocks_value) else int(float(n_blocks_value))
            ),
            disconnected=frame.at[index, "cr_disconnected"],
        )


def _assert_retained_identity(source: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    for column in REQUIRED_CORE_COLUMNS:
        source_column = (
            _SOURCE_COLUMNS.get(column, column)
        )
        left = source[source_column]
        right = cleaned[column]
        if column in _BOOLEAN_COLUMNS:
            left = _as_boolean(left, source_column)
            right = _as_boolean(right, column)
        pd.testing.assert_series_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_names=False,
            check_exact=True,
        )


def clean_core_results_frame(
    source: pd.DataFrame,
    *,
    estimator: str,
) -> pd.DataFrame:
    """Select and validate the common estimator result schema."""
    if not isinstance(estimator, str) or not estimator:
        raise ValueError("estimator must be a nonempty string")
    if not isinstance(source, pd.DataFrame):
        raise TypeError("source must be a pandas DataFrame")
    if source.columns.duplicated().any():
        duplicates = sorted(set(source.columns[source.columns.duplicated()].astype(str)))
        raise ValueError(f"source has duplicate columns: {duplicates}")
    source = with_neutral_grid_metadata(source)

    source_columns = {
        column: _SOURCE_COLUMNS.get(column, column)
        for column in REQUIRED_CORE_COLUMNS
    }
    missing_columns = sorted(
        column
        for column, source_column in source_columns.items()
        if source_column not in source.columns
    )
    if missing_columns:
        raise ValueError(f"source is missing required core columns: {missing_columns}")

    cleaned = pd.DataFrame(
        {column: source[source_column] for column, source_column in source_columns.items()}
    )
    for column in _BOOLEAN_COLUMNS:
        cleaned[column] = _as_boolean(cleaned[column], column)

    if tuple(cleaned.columns) != REQUIRED_CORE_COLUMNS:
        raise AssertionError("clean common-core schema construction failed")
    if len(cleaned) != len(source):
        raise AssertionError("common-core cleaning changed the row count")
    if cleaned["estimator"].dropna().astype(str).ne(estimator).any():
        raise ValueError(
            f"clean output contains an estimator other than {estimator}"
        )

    _validate_parameters(cleaned)
    _validate_confidence_regions(cleaned)
    duplicate_mask = cleaned.duplicated(list(CORE_IDENTIFIER_COLUMNS), keep=False)
    duplicate_identifiers = int(
        cleaned.loc[duplicate_mask, CORE_IDENTIFIER_COLUMNS].drop_duplicates().shape[0]
    )
    if duplicate_identifiers:
        duplicate_rows = int(duplicate_mask.sum())
        raise ValueError(
            f"{duplicate_identifiers} duplicate {estimator} identifiers affect "
            f"{duplicate_rows} rows; key columns are "
            f"{list(CORE_IDENTIFIER_COLUMNS)}"
        )
    _assert_retained_identity(source, cleaned)

    return cleaned


def clean_dml_results_frame(
    source: pd.DataFrame,
) -> pd.DataFrame:
    """Select, standardize, and strictly validate a DML result DataFrame."""
    return clean_core_results_frame(source, estimator="dml_ivqr")


__all__ = [
    "CORE_IDENTIFIER_COLUMNS",
    "GRID_METADATA_COLUMNS",
    "CR_GEOMETRY_COLUMNS",
    "REQUIRED_CORE_COLUMNS",
    "REQUIRED_DML_COLUMNS",
    "clean_core_results_frame",
    "clean_dml_results_frame",
    "with_neutral_grid_metadata",
    "validate_component_columns",
]
