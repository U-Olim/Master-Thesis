"""Clean and validate thesis-ready Post-selection IVQR outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_POST_SELECTION_COLUMNS: tuple[str, ...] = (
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
    "n_selected_controls",
    "selection_lasso_multiplier",
)

POST_SELECTION_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "dgp",
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "estimator",
)

_SOURCE_COLUMNS = {
    "covered": "cr_covers_true",
    "n_selected_controls": "ps_n_selected_controls",
    "selection_lasso_multiplier": "ps_selection_lasso_multiplier",
}
_BOOLEAN_COLUMNS = ("covered", "converged")


@dataclass(frozen=True)
class PostSelectionValidationSummary:
    """Calculated validation information for one cleaning operation."""

    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    removed_columns: int
    missing_values: dict[str, int]
    empty_confidence_regions: int
    duplicate_identifiers: int
    selected_controls_min: float
    selected_controls_max: float
    selected_controls_mean: float
    selection_lasso_multipliers: tuple[float, ...]


def _as_boolean(series: pd.Series, column: str) -> pd.Series:
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
    if (~np.isfinite(values)).any() or (values < minimum).any():
        raise ValueError(f"{column} must be finite and >= {minimum}")
    if (values != np.floor(values)).any():
        raise ValueError(f"{column} must be integer-valued")


def _validate_parameters(frame: pd.DataFrame) -> None:
    tau = _numeric(frame["tau"], "tau")
    if tau.isna().any():
        raise ValueError("tau must not be missing")
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

    expected = (lower <= alpha_true) & (alpha_true <= upper)
    actual = frame["covered"].fillna(False).astype(bool)
    comparable = finite & np.isfinite(alpha_true) & frame["covered"].notna()
    # Bounds describe the hull, so false coverage inside a disconnected hull is valid.
    impossible_coverage = comparable & actual & ~expected
    if impossible_coverage.any():
        raise ValueError(
            "covered=True is inconsistent with finite confidence bounds in "
            f"{int(impossible_coverage.sum())} rows"
        )

    empty = missing.all(axis=1)
    if (empty & actual).any():
        raise ValueError("an empty confidence region cannot have covered=True")
    return int(empty.sum())


def _validate_selection_variables(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    selected = _numeric(frame["n_selected_controls"], "n_selected_controls")
    p = _numeric(frame["p"], "p")
    present_selected = selected.notna()
    invalid_selected = present_selected & (
        ~np.isfinite(selected)
        | (selected < 0)
        | (selected > p)
        | (selected != np.floor(selected))
    )
    if invalid_selected.any():
        raise ValueError("n_selected_controls must be an integer between 0 and p")

    multiplier = _numeric(
        frame["selection_lasso_multiplier"], "selection_lasso_multiplier"
    )
    present_multiplier = multiplier.notna()
    invalid_multiplier = present_multiplier & (
        ~np.isfinite(multiplier) | (multiplier <= 0)
    )
    if invalid_multiplier.any():
        raise ValueError("selection_lasso_multiplier must be finite and positive")
    return selected, multiplier


def _source_column(source: pd.DataFrame, output_column: str) -> str:
    if output_column in source.columns:
        return output_column
    return _SOURCE_COLUMNS.get(output_column, output_column)


def _assert_retained_identity(source: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    for column in REQUIRED_POST_SELECTION_COLUMNS:
        source_column = _source_column(source, column)
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


def _validate_alias_agreement(source: pd.DataFrame) -> None:
    for output_column, legacy_column in _SOURCE_COLUMNS.items():
        if output_column not in source.columns or legacy_column not in source.columns:
            continue
        if output_column in _BOOLEAN_COLUMNS:
            current = _as_boolean(source[output_column], output_column)
            legacy = _as_boolean(source[legacy_column], legacy_column)
        else:
            current = _numeric(source[output_column], output_column)
            legacy = _numeric(source[legacy_column], legacy_column)
        if not current.equals(legacy):
            raise ValueError(f"{output_column} and {legacy_column} disagree")

    if {"selected_controls", "ps_n_selected_controls"}.issubset(source.columns):
        selected = _numeric(source["selected_controls"], "selected_controls")
        diagnostic = _numeric(
            source["ps_n_selected_controls"], "ps_n_selected_controls"
        )
        comparable = selected.notna() & diagnostic.notna()
        if selected.loc[comparable].ne(diagnostic.loc[comparable]).any():
            raise ValueError(
                "selected_controls and ps_n_selected_controls disagree; "
                "the scalar selection count is ambiguous"
            )


def clean_post_selection_results_frame(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, PostSelectionValidationSummary]:
    """Select, standardize, and strictly validate a Post-selection DataFrame."""
    if not isinstance(source, pd.DataFrame):
        raise TypeError("source must be a pandas DataFrame")
    if source.columns.duplicated().any():
        duplicates = sorted(set(source.columns[source.columns.duplicated()].astype(str)))
        raise ValueError(f"source has duplicate columns: {duplicates}")
    _validate_alias_agreement(source)

    source_columns = {
        column: _source_column(source, column)
        for column in REQUIRED_POST_SELECTION_COLUMNS
    }
    missing = sorted(
        column
        for column, source_column in source_columns.items()
        if source_column not in source.columns
    )
    if missing:
        raise ValueError(f"source is missing required Post-selection columns: {missing}")

    cleaned = pd.DataFrame(
        {column: source[source_column] for column, source_column in source_columns.items()}
    )
    for column in _BOOLEAN_COLUMNS:
        cleaned[column] = _as_boolean(cleaned[column], column)

    if tuple(cleaned.columns) != REQUIRED_POST_SELECTION_COLUMNS:
        raise AssertionError("clean Post-selection schema construction failed")
    if len(cleaned) != len(source):
        raise AssertionError("Post-selection cleaning changed the row count")
    if cleaned["estimator"].dropna().astype(str).ne("post_selection_ivqr").any():
        raise ValueError("clean output contains an estimator other than post_selection_ivqr")

    _validate_parameters(cleaned)
    empty_confidence_regions = _validate_confidence_regions(cleaned)
    selected, multiplier = _validate_selection_variables(cleaned)
    duplicate_mask = cleaned.duplicated(
        list(POST_SELECTION_IDENTIFIER_COLUMNS), keep=False
    )
    duplicate_identifiers = int(
        cleaned.loc[duplicate_mask, POST_SELECTION_IDENTIFIER_COLUMNS]
        .drop_duplicates()
        .shape[0]
    )
    if duplicate_identifiers:
        raise ValueError(
            f"{duplicate_identifiers} duplicate Post-selection identifiers affect "
            f"{int(duplicate_mask.sum())} rows; key columns are "
            f"{list(POST_SELECTION_IDENTIFIER_COLUMNS)}"
        )
    _assert_retained_identity(source, cleaned)

    selected_present = selected.dropna()
    multiplier_values = tuple(sorted(float(value) for value in multiplier.dropna().unique()))
    summary = PostSelectionValidationSummary(
        input_rows=len(source),
        output_rows=len(cleaned),
        input_columns=len(source.columns),
        output_columns=len(cleaned.columns),
        removed_columns=len(source.columns) - len(cleaned.columns),
        missing_values={
            column: int(cleaned[column].isna().sum())
            for column in REQUIRED_POST_SELECTION_COLUMNS
        },
        empty_confidence_regions=empty_confidence_regions,
        duplicate_identifiers=duplicate_identifiers,
        selected_controls_min=(
            float(selected_present.min()) if not selected_present.empty else float("nan")
        ),
        selected_controls_max=(
            float(selected_present.max()) if not selected_present.empty else float("nan")
        ),
        selected_controls_mean=(
            float(selected_present.mean()) if not selected_present.empty else float("nan")
        ),
        selection_lasso_multipliers=multiplier_values,
    )
    return cleaned, summary


def clean_post_selection_results_csv(
    input_path: str | Path, output_path: str | Path
) -> PostSelectionValidationSummary:
    """Clean a historical Post-selection CSV without modifying its source."""
    source_path = Path(input_path)
    destination = Path(output_path)
    if source_path.resolve() == destination.resolve():
        raise ValueError("input and output paths must differ")
    source = pd.read_csv(source_path, low_memory=False, float_precision="round_trip")
    cleaned, summary = clean_post_selection_results_frame(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(destination, index=False)

    reread = pd.read_csv(
        destination, low_memory=False, float_precision="round_trip"
    )
    if tuple(reread.columns) != REQUIRED_POST_SELECTION_COLUMNS:
        raise AssertionError("written Post-selection CSV has the wrong schema")
    if len(reread) != len(source):
        raise AssertionError("written Post-selection CSV changed the row count")
    _assert_retained_identity(cleaned, reread)
    return summary


__all__ = [
    "POST_SELECTION_IDENTIFIER_COLUMNS",
    "PostSelectionValidationSummary",
    "REQUIRED_POST_SELECTION_COLUMNS",
    "clean_post_selection_results_csv",
    "clean_post_selection_results_frame",
]
