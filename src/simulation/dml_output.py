"""Clean and validate thesis-ready DML-IVQR simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_CORE_COLUMNS: tuple[str, ...] = (
    "dgp",
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "seed",
    "estimator",
    "alpha_hat",
    "alpha_true",
    "cr_lower",
    "cr_upper",
    "cr_length",
    "covered",
    "converged",
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
DML_IDENTIFIER_COLUMNS = CORE_IDENTIFIER_COLUMNS

_BOOLEAN_COLUMNS = ("covered", "converged")
_LEGACY_RENAMES = {"cr_covers_true": "covered"}


@dataclass(frozen=True)
class CoreValidationSummary:
    """Calculated validation information for one common-core cleaning operation."""

    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    removed_columns: int
    missing_values: dict[str, int]
    empty_confidence_regions: int
    duplicate_identifiers: int


DMLValidationSummary = CoreValidationSummary


def _as_boolean(series: pd.Series, column: str) -> pd.Series:
    """Return a nullable Boolean series, rejecting ambiguous values."""
    true_values = {"true", "1", "yes"}
    false_values = {"false", "0", "no"}

    def convert(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isfinite(float(value)) and float(value) in (0.0, 1.0):
                return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in true_values:
                return True
            if normalized in false_values:
                return False
        raise ValueError(f"{column} contains a non-Boolean value: {value!r}")

    return series.map(convert).astype("boolean")


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


def _validate_confidence_regions(frame: pd.DataFrame) -> int:
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
    return int(empty.sum())


def _assert_retained_identity(source: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    for column in REQUIRED_CORE_COLUMNS:
        source_column = (
            "cr_covers_true"
            if column == "covered" and "covered" not in source
            else column
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
) -> tuple[pd.DataFrame, CoreValidationSummary]:
    """Select and validate the common 15-column estimator result schema."""
    if not isinstance(estimator, str) or not estimator:
        raise ValueError("estimator must be a nonempty string")
    if not isinstance(source, pd.DataFrame):
        raise TypeError("source must be a pandas DataFrame")
    if source.columns.duplicated().any():
        duplicates = sorted(set(source.columns[source.columns.duplicated()].astype(str)))
        raise ValueError(f"source has duplicate columns: {duplicates}")

    renamed = source.copy()
    if "covered" in renamed.columns and "cr_covers_true" in renamed.columns:
        legacy = _as_boolean(renamed["cr_covers_true"], "cr_covers_true")
        current = _as_boolean(renamed["covered"], "covered")
        if not legacy.equals(current):
            raise ValueError("covered and cr_covers_true disagree")
    elif "cr_covers_true" in renamed.columns:
        renamed = renamed.rename(columns=_LEGACY_RENAMES)

    missing_columns = sorted(set(REQUIRED_CORE_COLUMNS) - set(renamed.columns))
    if missing_columns:
        raise ValueError(f"source is missing required core columns: {missing_columns}")

    cleaned = renamed.loc[:, REQUIRED_CORE_COLUMNS].copy()
    for column in _BOOLEAN_COLUMNS:
        cleaned[column] = _as_boolean(cleaned[column], column)

    if len(cleaned.columns) != 15 or tuple(cleaned.columns) != REQUIRED_CORE_COLUMNS:
        raise AssertionError("clean common-core schema construction failed")
    if len(cleaned) != len(source):
        raise AssertionError("common-core cleaning changed the row count")
    if cleaned["estimator"].dropna().astype(str).ne(estimator).any():
        raise ValueError(
            f"clean output contains an estimator other than {estimator}"
        )

    _validate_parameters(cleaned)
    empty_confidence_regions = _validate_confidence_regions(cleaned)
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

    summary = CoreValidationSummary(
        input_rows=len(source),
        output_rows=len(cleaned),
        input_columns=len(source.columns),
        output_columns=len(cleaned.columns),
        removed_columns=len(source.columns) - len(cleaned.columns),
        missing_values={
            column: int(cleaned[column].isna().sum())
            for column in REQUIRED_CORE_COLUMNS
        },
        empty_confidence_regions=empty_confidence_regions,
        duplicate_identifiers=duplicate_identifiers,
    )
    return cleaned, summary


def clean_dml_results_frame(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, DMLValidationSummary]:
    """Select, standardize, and strictly validate a DML result DataFrame."""
    return clean_core_results_frame(source, estimator="dml_ivqr")


def clean_core_results_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    estimator: str,
) -> CoreValidationSummary:
    """Clean a historical common-core CSV without modifying its source."""
    source_path = Path(input_path)
    destination = Path(output_path)
    if source_path.resolve() == destination.resolve():
        raise ValueError("input and output paths must differ")
    source = pd.read_csv(source_path, low_memory=False, float_precision="round_trip")
    cleaned, summary = clean_core_results_frame(source, estimator=estimator)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(destination, index=False)

    reread = pd.read_csv(
        destination, low_memory=False, float_precision="round_trip"
    )
    if tuple(reread.columns) != REQUIRED_CORE_COLUMNS or len(reread) != len(source):
        raise AssertionError(
            "written common-core CSV failed schema or row-count validation"
        )
    _assert_retained_identity(cleaned, reread)
    return summary


def clean_dml_results_csv(
    input_path: str | Path, output_path: str | Path
) -> DMLValidationSummary:
    """Clean a historical DML CSV without modifying the source file."""
    return clean_core_results_csv(input_path, output_path, estimator="dml_ivqr")


__all__ = [
    "CORE_IDENTIFIER_COLUMNS",
    "CoreValidationSummary",
    "DML_IDENTIFIER_COLUMNS",
    "DMLValidationSummary",
    "REQUIRED_CORE_COLUMNS",
    "REQUIRED_DML_COLUMNS",
    "clean_core_results_csv",
    "clean_core_results_frame",
    "clean_dml_results_csv",
    "clean_dml_results_frame",
]
