"""Build and validate current Post-selection IVQR outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from simulation.output_schemas import (
    POST_SELECTION_IDENTIFIER_COLUMNS,
    POST_SELECTION_OUTPUT_COLUMNS,
)
from simulation.output_validation import (
    as_boolean as _as_boolean,
    assert_retained_identity,
    numeric as _numeric,
    validate_confidence_regions as _validate_confidence_regions,
    validate_parameters as _validate_parameters,
    with_neutral_grid_metadata,
)


REQUIRED_POST_SELECTION_COLUMNS = POST_SELECTION_OUTPUT_COLUMNS

_SOURCE_COLUMNS = {
    "covered": "cr_covers_true",
    "n_selected_controls": "ps_n_selected_controls",
    "selection_lasso_multiplier": "ps_selection_lasso_multiplier",
    "selection_method": "ps_selection_method",
    "selection_target_y": "ps_selection_target_y",
    "selection_target_d": "ps_selection_target_d",
    "selection_quantile_specific": "ps_selection_quantile_specific",
    "instrument_selection_method": "ps_instrument_selection_method",
    "post_selection_inference_adjustment": (
        "ps_post_selection_inference_adjustment"
    ),
    "n_retained_instruments": "ps_n_retained_instruments",
}
_BOOLEAN_COLUMNS = (
    "covered",
    "converged",
    "cr_disconnected",
    "selection_quantile_specific",
)


def _validate_selection_variables(frame: pd.DataFrame) -> None:
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


def _source_column(output_column: str) -> str:
    return _SOURCE_COLUMNS.get(output_column, output_column)


def _assert_retained_identity(source: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    assert_retained_identity(
        source,
        cleaned,
        REQUIRED_POST_SELECTION_COLUMNS,
        {
            column: _source_column(column)
            for column in REQUIRED_POST_SELECTION_COLUMNS
        },
        _BOOLEAN_COLUMNS,
    )


def _validate_selection_count_agreement(source: pd.DataFrame) -> None:
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
) -> pd.DataFrame:
    """Select, standardize, and strictly validate a Post-selection DataFrame."""
    if not isinstance(source, pd.DataFrame):
        raise TypeError("source must be a pandas DataFrame")
    if source.columns.duplicated().any():
        duplicates = sorted(set(source.columns[source.columns.duplicated()].astype(str)))
        raise ValueError(f"source has duplicate columns: {duplicates}")
    source = with_neutral_grid_metadata(source)
    legacy_defaults: dict[str, object] = {
        "ps_selection_method": "unavailable",
        "ps_selection_target_y": "unavailable",
        "ps_selection_target_d": "unavailable",
        "ps_selection_quantile_specific": np.nan,
        "ps_instrument_selection_method": "unavailable",
        "ps_post_selection_inference_adjustment": "unavailable",
        "ps_n_retained_instruments": np.nan,
    }
    for column, default in legacy_defaults.items():
        if column not in source:
            source[column] = default
    _validate_selection_count_agreement(source)

    source_columns = {
        column: _source_column(column)
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
    _validate_confidence_regions(cleaned)
    _validate_selection_variables(cleaned)
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

    return cleaned


__all__ = [
    "POST_SELECTION_IDENTIFIER_COLUMNS",
    "REQUIRED_POST_SELECTION_COLUMNS",
    "clean_post_selection_results_frame",
]
