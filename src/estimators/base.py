"""Common estimator result objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.timing import RUNTIME_COLUMNS


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
    "critical_value_nominal",
    "critical_value_multiplier",
    "critical_value_adjusted",
)

POST_SELECTION_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "ps_n_selected_controls",
    "ps_n_selected_instruments",
    "ps_n_selected_total",
    "ps_share_selected_controls",
    "ps_share_selected_instruments",
    "ps_instrument_selection_method",
    "ps_n_candidate_instruments",
    "ps_n_retained_instruments",
    "ps_share_retained_instruments",
    "ps_all_instruments_retained",
    "ps_selected_no_controls",
    "ps_selected_no_instruments",
    "ps_selected_empty_total",
    "ps_first_stage_r2",
    "ps_first_stage_adj_r2",
    "ps_first_stage_partial_r2",
    "ps_first_stage_f_stat",
    "ps_first_stage_condition_number",
    "ps_selection_method",
    "ps_selection_lasso_multiplier",
    "ps_lasso_alpha_controls",
    "ps_lasso_alpha_instruments",
    "ps_lasso_alpha_first_stage",
    "ps_lasso_alpha_y_cv",
    "ps_lasso_alpha_d_cv",
    "ps_lasso_alpha_y_final",
    "ps_lasso_alpha_d_final",
    "ps_lasso_cv_folds",
    "ps_selection_failed",
    "ps_first_stage_failed",
    "ps_rank_deficient",
    "ps_warning_code",
)

RUNTIME_DIAGNOSTIC_FIELDS: tuple[str, ...] = RUNTIME_COLUMNS

DML_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "dml_quantile_penalty",
    "dml_ridge_alpha",
    "dml_quantile_solver",
    "dml_qr_fit_count",
    "dml_runtime_mean_alpha_sec",
    "dml_runtime_max_alpha_sec",
    "dml_qr_nonzero_mean",
    "dml_z_resid_var_mean",
)


def estimation_result_diagnostic_kwargs(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Return common diagnostic fields accepted by EstimationResult."""
    return {name: diagnostics[name] for name in RESULT_DIAGNOSTIC_FIELDS}


def post_selection_result_diagnostic_kwargs(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Return post-selection diagnostic fields accepted by EstimationResult."""
    return {name: diagnostics[name] for name in POST_SELECTION_DIAGNOSTIC_FIELDS}


@dataclass
class EstimationResult:
    """Standard result object returned by every estimator.

    The four retained estimators use this common object so simulation,
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
    error_type: str | None = None

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
    critical_value_nominal: float | None = None
    critical_value_multiplier: float | None = None
    critical_value_adjusted: float | None = None

    runtime_total_sec: float | None = None
    runtime_data_generation_sec: float | None = None
    runtime_estimator_sec: float | None = None
    runtime_alpha_grid_sec: float | None = None
    runtime_confidence_region_sec: float | None = None
    runtime_score_eval_sec: float | None = None
    runtime_other_sec: float | None = None
    dml_runtime_total_sec: float | None = None
    dml_runtime_crossfit_sec: float | None = None
    dml_runtime_nuisance_fit_sec: float | None = None
    dml_runtime_nuisance_predict_sec: float | None = None
    dml_runtime_alpha_loop_sec: float | None = None
    dml_runtime_score_eval_sec: float | None = None
    dml_runtime_confidence_region_sec: float | None = None
    ps_runtime_total_sec: float | None = None
    ps_runtime_selection_sec: float | None = None
    ps_runtime_first_stage_sec: float | None = None
    ps_runtime_alpha_loop_sec: float | None = None
    ps_runtime_score_eval_sec: float | None = None
    ps_runtime_confidence_region_sec: float | None = None
    ps_runtime_diagnostics_sec: float | None = None
    oracle_runtime_total_sec: float | None = None
    oracle_runtime_alpha_loop_sec: float | None = None
    oracle_runtime_score_eval_sec: float | None = None
    oracle_runtime_confidence_region_sec: float | None = None

    ps_n_selected_controls: int | None = None
    ps_n_selected_instruments: int | None = None
    ps_n_selected_total: int | None = None
    ps_share_selected_controls: float | None = None
    ps_share_selected_instruments: float | None = None
    ps_instrument_selection_method: str | None = None
    ps_n_candidate_instruments: int | None = None
    ps_n_retained_instruments: int | None = None
    ps_share_retained_instruments: float | None = None
    ps_all_instruments_retained: bool | None = None
    ps_selected_no_controls: bool | None = None
    ps_selected_no_instruments: bool | None = None
    ps_selected_empty_total: bool | None = None
    ps_first_stage_r2: float | None = None
    ps_first_stage_adj_r2: float | None = None
    ps_first_stage_partial_r2: float | None = None
    ps_first_stage_f_stat: float | None = None
    ps_first_stage_condition_number: float | None = None
    ps_selection_method: str | None = None
    ps_selection_lasso_multiplier: float | None = None
    ps_lasso_alpha_controls: float | None = None
    ps_lasso_alpha_instruments: float | None = None
    ps_lasso_alpha_first_stage: float | None = None
    ps_lasso_alpha_y_cv: float | None = None
    ps_lasso_alpha_d_cv: float | None = None
    ps_lasso_alpha_y_final: float | None = None
    ps_lasso_alpha_d_final: float | None = None
    ps_lasso_cv_folds: int | None = None
    ps_selection_failed: bool | None = None
    ps_first_stage_failed: bool | None = None
    ps_rank_deficient: bool | None = None
    ps_warning_code: str | None = None

    dml_quantile_penalty: float | None = None
    dml_ridge_alpha: float | None = None
    dml_quantile_solver: str | None = None
    dml_qr_fit_count: int | None = None
    dml_runtime_mean_alpha_sec: float | None = None
    dml_runtime_max_alpha_sec: float | None = None
    dml_qr_nonzero_mean: float | None = None
    dml_z_resid_var_mean: float | None = None

    @property
    def status(self) -> str:
        """Return the public estimator status used in simulation CSV rows."""
        return "failed" if self.failed else "ok"

    @property
    def confidence_region(self) -> dict[str, Any]:
        """Return confidence-region fields in one stable public mapping."""
        return {
            "lower": self.cr_lower,
            "upper": self.cr_upper,
            "length": self.cr_length,
            "empty": self.cr_empty,
            "covers_true": self.cr_covers_true,
            "hits_lower_boundary": self.cr_hits_lower_boundary,
            "hits_upper_boundary": self.cr_hits_upper_boundary,
            "hits_any_boundary": self.cr_hits_any_boundary,
            "n_blocks": self.cr_n_blocks,
            "disconnected": self.cr_disconnected,
            "hull_length": self.cr_hull_length,
        }

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return all standard diagnostic fields with explicit missing values."""
        names = (
            RESULT_DIAGNOSTIC_FIELDS
            + POST_SELECTION_DIAGNOSTIC_FIELDS
            + DML_DIAGNOSTIC_FIELDS
            + RUNTIME_DIAGNOSTIC_FIELDS
        )
        return {name: getattr(self, name) for name in names}
