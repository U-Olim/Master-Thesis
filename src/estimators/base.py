"""Common estimator result objects."""

from dataclasses import dataclass
from typing import Any


RESULT_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "alpha_grid_min",
    "alpha_grid_max",
    "alpha_grid_step",
    "alpha_hat_at_lower_boundary",
    "alpha_hat_at_upper_boundary",
    "alpha_hat_at_any_boundary",
    "cr_hits_lower_boundary",
    "cr_hits_upper_boundary",
    "cr_hits_any_boundary",
    "cr_accepted_alpha_count",
    "cr_acceptance_rate",
    "cr_n_blocks",
    "cr_hull_length",
    "failed_alpha_rate",
    "min_test_stat",
    "max_test_stat",
    "test_stat_at_alpha_hat",
    "critical_value",
)

POST_SELECTION_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "ps_n_selected_controls",
    "ps_n_selected_instruments",
    "ps_n_selected_total",
    "ps_share_selected_controls",
    "ps_share_selected_instruments",
    "ps_selected_no_controls",
    "ps_selected_no_instruments",
    "ps_selected_empty_total",
    "ps_first_stage_r2",
    "ps_first_stage_adj_r2",
    "ps_first_stage_partial_r2",
    "ps_first_stage_f_stat",
    "ps_first_stage_condition_number",
    "ps_selection_method",
    "ps_lasso_alpha_controls",
    "ps_lasso_alpha_instruments",
    "ps_lasso_alpha_first_stage",
    "ps_lasso_cv_folds",
    "ps_selection_failed",
    "ps_first_stage_failed",
    "ps_rank_deficient",
    "ps_warning_code",
)


def estimation_result_diagnostic_kwargs(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Return diagnostic fields accepted by EstimationResult."""
    return {name: diagnostics[name] for name in RESULT_DIAGNOSTIC_FIELDS}


def post_selection_result_diagnostic_kwargs(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Return post-selection diagnostic fields accepted by EstimationResult."""
    return {name: diagnostics[name] for name in POST_SELECTION_DIAGNOSTIC_FIELDS}


@dataclass
class EstimationResult:
    """Standard result object returned by every estimator.

    All estimator implementations must return this object so simulation,
    aggregation, and reporting code can consume results uniformly.
    """

    estimator: str
    alpha_hat: float | None
    alpha_true: float | None
    tau: float

    converged: bool
    failed: bool
    message: str

    objective_value: float | None
    at_grid_boundary: bool
    alpha_grid_size: int | None
    failed_alpha_count: int | None

    cr_lower: float | None
    cr_upper: float | None
    cr_length: float | None
    cr_covers_true: bool | None
    cr_empty: bool
    cr_disconnected: bool | None

    selected_controls: int | None
    runtime_seconds: float

    alpha_grid_min: float | None = None
    alpha_grid_max: float | None = None
    alpha_grid_step: float | None = None
    alpha_hat_at_lower_boundary: bool | None = None
    alpha_hat_at_upper_boundary: bool | None = None
    alpha_hat_at_any_boundary: bool | None = None
    cr_hits_lower_boundary: bool | None = None
    cr_hits_upper_boundary: bool | None = None
    cr_hits_any_boundary: bool | None = None
    cr_accepted_alpha_count: int | None = None
    cr_acceptance_rate: float | None = None
    cr_n_blocks: int | None = None
    cr_hull_length: float | None = None
    failed_alpha_rate: float | None = None
    min_test_stat: float | None = None
    max_test_stat: float | None = None
    test_stat_at_alpha_hat: float | None = None
    critical_value: float | None = None

    ps_n_selected_controls: int | None = None
    ps_n_selected_instruments: int | None = None
    ps_n_selected_total: int | None = None
    ps_share_selected_controls: float | None = None
    ps_share_selected_instruments: float | None = None
    ps_selected_no_controls: bool | None = None
    ps_selected_no_instruments: bool | None = None
    ps_selected_empty_total: bool | None = None
    ps_first_stage_r2: float | None = None
    ps_first_stage_adj_r2: float | None = None
    ps_first_stage_partial_r2: float | None = None
    ps_first_stage_f_stat: float | None = None
    ps_first_stage_condition_number: float | None = None
    ps_selection_method: str | None = None
    ps_lasso_alpha_controls: float | None = None
    ps_lasso_alpha_instruments: float | None = None
    ps_lasso_alpha_first_stage: float | None = None
    ps_lasso_cv_folds: int | None = None
    ps_selection_failed: bool | None = None
    ps_first_stage_failed: bool | None = None
    ps_rank_deficient: bool | None = None
    ps_warning_code: str | None = None
