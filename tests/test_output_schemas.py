from types import MappingProxyType

import pytest

from simulation.output_schemas import (
    DML_OUTPUT_COLUMNS,
    INTERNAL_RESULT_COLUMNS,
    ORACLE_OUTPUT_COLUMNS,
    OUTPUT_COLUMNS_BY_ESTIMATOR,
    POST_SELECTION_OUTPUT_COLUMNS,
)
from simulation.results import RESULT_COLUMNS


EXPECTED_ORACLE_COLUMNS = (
    "dgp", "n", "p", "pi", "tau", "rep", "alpha_true", "alpha_hat",
    "covered", "cr_length", "cr_status", "cr_n_blocks", "cr_disconnected",
    "cr_components", "iteration_warning_evaluations", "seed", "cr_lower",
    "cr_upper", "converged", "cr_is_numerically_resolved",
    "cr_unresolved_count", "final_alpha_evaluations",
    "refinement_depth_reached", "number_of_refined_intervals",
    "minimum_final_grid_spacing", "median_final_grid_spacing",
)
EXPECTED_DML_COLUMNS = (
    "dgp", "n", "p", "pi", "tau", "rep", "seed", "result_schema_version",
    "estimator", "alpha_hat", "alpha_true", "cr_lower", "cr_upper",
    "cr_length", "covered", "converged", "cr_components", "cr_n_blocks",
    "cr_disconnected", "cr_status", "cr_is_numerically_resolved",
    "cr_unresolved_count", "cr_unresolved_alphas", "grid_strategy",
    "adaptive_midpoint_probe", "alpha_hat_grid", "midpoint_intervals_considered",
    "midpoint_evaluations_added", "midpoint_unresolved_barriers",
    "midpoint_probe_limit_hit", "initial_alpha_grid_size",
    "final_alpha_evaluations", "refinement_tolerance", "refinement_depth_reached",
    "refinement_limit_hit", "max_alpha_evaluations_hit",
    "number_of_refined_intervals", "number_of_unresolved_refinement_barriers",
    "minimum_final_grid_spacing", "median_final_grid_spacing",
    "maximum_final_grid_spacing", "iteration_warning_evaluations",
    "rank_deficient_covariance_failures",
)
EXPECTED_POST_SELECTION_COLUMNS = (
    *EXPECTED_DML_COLUMNS,
    "n_selected_controls", "selection_lasso_multiplier", "selection_method",
    "selection_target_y", "selection_target_d", "selection_quantile_specific",
    "instrument_selection_method", "post_selection_inference_adjustment",
    "n_retained_instruments",
)


def test_registry_has_exactly_three_read_only_entries() -> None:
    assert isinstance(OUTPUT_COLUMNS_BY_ESTIMATOR, MappingProxyType)
    assert tuple(OUTPUT_COLUMNS_BY_ESTIMATOR) == (
        "oracle",
        "post_selection",
        "dml",
    )
    with pytest.raises(TypeError):
        OUTPUT_COLUMNS_BY_ESTIMATOR["extra"] = ()  # type: ignore[index]


def test_registered_schemas_equal_pre_refactor_contracts() -> None:
    assert ORACLE_OUTPUT_COLUMNS == EXPECTED_ORACLE_COLUMNS
    assert POST_SELECTION_OUTPUT_COLUMNS == EXPECTED_POST_SELECTION_COLUMNS
    assert DML_OUTPUT_COLUMNS == EXPECTED_DML_COLUMNS
    assert OUTPUT_COLUMNS_BY_ESTIMATOR["oracle"] == EXPECTED_ORACLE_COLUMNS
    assert (
        OUTPUT_COLUMNS_BY_ESTIMATOR["post_selection"]
        == EXPECTED_POST_SELECTION_COLUMNS
    )
    assert OUTPUT_COLUMNS_BY_ESTIMATOR["dml"] == EXPECTED_DML_COLUMNS


@pytest.mark.parametrize(
    ("schema", "expected_length"),
    [
        (ORACLE_OUTPUT_COLUMNS, 26),
        (POST_SELECTION_OUTPUT_COLUMNS, 52),
        (DML_OUTPUT_COLUMNS, 43),
    ],
)
def test_registered_schema_lengths_and_uniqueness(
    schema: tuple[str, ...], expected_length: int
) -> None:
    assert len(schema) == expected_length
    assert len(schema) == len(set(schema))


def test_internal_result_schema_remains_the_150_column_contract() -> None:
    assert INTERNAL_RESULT_COLUMNS is RESULT_COLUMNS
    assert len(INTERNAL_RESULT_COLUMNS) == 150
    assert len(INTERNAL_RESULT_COLUMNS) == len(set(INTERNAL_RESULT_COLUMNS))
