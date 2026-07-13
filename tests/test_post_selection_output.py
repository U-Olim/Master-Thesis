import numpy as np
import pandas as pd
import pytest

from simulation.post_selection_output import (
    REQUIRED_POST_SELECTION_COLUMNS,
    clean_post_selection_results_frame,
)


def _wide_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dgp": ["dgp1", "dgp1"],
            "n": [100, 100],
            "p": [10, 10],
            "pi": [0.5, 0.5],
            "tau": [0.5, 0.5],
            "rep": [0, 1],
            "seed": [123, 124],
            "estimator": ["post_selection_ivqr", "post_selection_ivqr"],
            "alpha_hat": [1.1, 0.9],
            "alpha_true": [1.0, 1.0],
            "cr_lower": [0.5, np.nan],
            "cr_upper": [1.5, np.nan],
            "cr_length": [1.0, np.nan],
            "cr_covers_true": [True, False],
            "converged": [True, True],
            "selected_controls": [3, 4],
            "ps_n_selected_controls": [3, 4],
            "ps_selection_lasso_multiplier": [1.8, 1.8],
            "runtime_seconds": [0.1, 0.2],
            "bias": [0.1, -0.1],
            "dml_quantile_solver": [np.nan, np.nan],
            "ps_first_stage_r2": [0.4, 0.5],
            "ps_selection_failed": [False, False],
            "message": ["ok", "empty confidence region"],
        }
    )


def test_clean_post_selection_exact_schema_and_no_extra_columns() -> None:
    cleaned = clean_post_selection_results_frame(_wide_frame())

    assert tuple(cleaned.columns) == REQUIRED_POST_SELECTION_COLUMNS
    assert len(cleaned.columns) == 17
    assert set(cleaned.columns) == set(REQUIRED_POST_SELECTION_COLUMNS)
    assert not any("runtime" in column for column in cleaned.columns)


def test_clean_post_selection_preserves_rows_and_values() -> None:
    source = _wide_frame()
    cleaned = clean_post_selection_results_frame(source)

    assert len(cleaned) == len(source)
    mappings = {
        "covered": "cr_covers_true",
        "n_selected_controls": "ps_n_selected_controls",
        "selection_lasso_multiplier": "ps_selection_lasso_multiplier",
    }
    for column in REQUIRED_POST_SELECTION_COLUMNS:
        source_column = mappings.get(column, column)
        pd.testing.assert_series_equal(
            source[source_column],
            cleaned[column],
            check_dtype=False,
            check_names=False,
        )


def test_clean_post_selection_preserves_missing_confidence_region() -> None:
    cleaned = clean_post_selection_results_frame(_wide_frame())

    assert len(cleaned) == 2
    assert cleaned.loc[1, ["cr_lower", "cr_upper", "cr_length"]].isna().all()
    assert bool(cleaned.loc[1, "covered"]) is False


def test_clean_post_selection_applies_all_required_renames() -> None:
    cleaned = clean_post_selection_results_frame(_wide_frame())

    assert cleaned["covered"].tolist() == [True, False]
    assert cleaned["n_selected_controls"].tolist() == [3, 4]
    assert cleaned["selection_lasso_multiplier"].tolist() == [1.8, 1.8]
    assert "cr_covers_true" not in cleaned
    assert "ps_n_selected_controls" not in cleaned
    assert "ps_selection_lasso_multiplier" not in cleaned


def test_clean_post_selection_rejects_duplicate_identifiers() -> None:
    source = pd.concat(
        [_wide_frame().iloc[[0]], _wide_frame().iloc[[0]]], ignore_index=True
    )

    with pytest.raises(ValueError, match="duplicate Post-selection identifiers"):
        clean_post_selection_results_frame(source)
    assert len(source) == 2


def test_clean_post_selection_rejects_ambiguous_selection_counts() -> None:
    source = _wide_frame()
    source.loc[0, "selected_controls"] = 2

    with pytest.raises(
        ValueError, match="selected_controls and ps_n_selected_controls disagree"
    ):
        clean_post_selection_results_frame(source)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("ps_n_selected_controls", -1, "between 0 and p"),
        ("ps_n_selected_controls", 11, "between 0 and p"),
        ("ps_selection_lasso_multiplier", 0, "finite and positive"),
    ],
)
def test_clean_post_selection_validates_selection_variables(
    column: str, value: float, message: str
) -> None:
    source = _wide_frame().drop(columns=["selected_controls"])
    source.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        clean_post_selection_results_frame(source)
