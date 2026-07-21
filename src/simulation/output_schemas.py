"""Authoritative schemas for current estimator-specific simulation outputs."""

from types import MappingProxyType

from simulation.results import RESULT_COLUMNS


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

DML_OUTPUT_COLUMNS: tuple[str, ...] = (
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

POST_SELECTION_OUTPUT_COLUMNS: tuple[str, ...] = (
    *DML_OUTPUT_COLUMNS,
    "n_selected_controls",
    "selection_lasso_multiplier",
    "selection_method",
    "selection_target_y",
    "selection_target_d",
    "selection_quantile_specific",
    "instrument_selection_method",
    "post_selection_inference_adjustment",
    "n_retained_instruments",
)

ORACLE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "dgp",
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "alpha_true",
    "alpha_hat",
    "covered",
    "cr_length",
    "cr_status",
    "cr_n_blocks",
    "cr_disconnected",
    "cr_components",
    "iteration_warning_evaluations",
    "seed",
    "cr_lower",
    "cr_upper",
    "converged",
    "cr_is_numerically_resolved",
    "cr_unresolved_count",
    "final_alpha_evaluations",
    "refinement_depth_reached",
    "number_of_refined_intervals",
    "minimum_final_grid_spacing",
    "median_final_grid_spacing",
)

INTERNAL_RESULT_COLUMNS: tuple[str, ...] = RESULT_COLUMNS
CORE_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "dgp",
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "estimator",
)
POST_SELECTION_IDENTIFIER_COLUMNS: tuple[str, ...] = CORE_IDENTIFIER_COLUMNS
ORACLE_DESIGN_KEY_COLUMNS: tuple[str, ...] = ORACLE_OUTPUT_COLUMNS[:6]

OUTPUT_COLUMNS_BY_ESTIMATOR = MappingProxyType(
    {
        "oracle": ORACLE_OUTPUT_COLUMNS,
        "post_selection": POST_SELECTION_OUTPUT_COLUMNS,
        "dml": DML_OUTPUT_COLUMNS,
    }
)


__all__ = [
    "CORE_IDENTIFIER_COLUMNS",
    "CR_GEOMETRY_COLUMNS",
    "DML_OUTPUT_COLUMNS",
    "GRID_METADATA_COLUMNS",
    "INTERNAL_RESULT_COLUMNS",
    "ORACLE_DESIGN_KEY_COLUMNS",
    "ORACLE_OUTPUT_COLUMNS",
    "OUTPUT_COLUMNS_BY_ESTIMATOR",
    "POST_SELECTION_IDENTIFIER_COLUMNS",
    "POST_SELECTION_OUTPUT_COLUMNS",
]
