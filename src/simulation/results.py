"""Result-row construction utilities for simulation outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from dgp.designs import Design
from estimators.base import (
    DML_DIAGNOSTIC_FIELDS,
    EstimationResult,
    POST_SELECTION_DIAGNOSTIC_FIELDS,
)
from ivqr.confidence_regions import (
    serialize_cr_components,
    summarize_alpha_grid_diagnostics,
)
from utils.timing import RUNTIME_COLUMNS, empty_runtime_columns


POST_SELECTION_WARNING_NONE = ""
MAX_ERROR_MESSAGE_LENGTH = 500
RESULT_SCHEMA_VERSION = 4

RESULT_COLUMNS: tuple[str, ...] = (
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
    "bias",
    "absolute_error",
    "squared_error",
    "status",
    "failed",
    "converged",
    "error_type",
    "error_message",
    "cr_lower",
    "cr_upper",
    "cr_length",
    "cr_components",
    "cr_empty",
    "cr_covers_true",
    "cr_hits_lower_boundary",
    "cr_hits_upper_boundary",
    "cr_hits_any_boundary",
    "cr_accepted_alpha_count",
    "cr_acceptance_rate",
    "cr_n_blocks",
    "cr_disconnected",
    "cr_hull_length",
    "hard_failure_policy",
    "cr_status",
    "cr_is_numerically_resolved",
    "cr_unresolved_count",
    "cr_unresolved_alphas",
    "coverage_status",
    "point_estimate_status",
    "usable_alpha_evaluations",
    "unresolved_alpha_evaluations",
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
    "alpha_grid_min",
    "alpha_grid_max",
    "alpha_grid_size",
    "alpha_grid_step",
    "alpha_hat_at_lower_boundary",
    "alpha_hat_at_upper_boundary",
    "alpha_hat_at_any_boundary",
    "failed_alpha_count",
    "failed_alpha_rate",
    "min_test_stat",
    "max_test_stat",
    "test_stat_at_alpha_hat",
    "critical_value",
    "critical_value_nominal",
    "critical_value_multiplier",
    "critical_value_adjusted",
    "selected_controls",
    "runtime_seconds",
    *RUNTIME_COLUMNS,
    "dml_quantile_penalty",
    "dml_ridge_alpha",
    "dml_quantile_solver",
    "dml_qr_fit_count",
    "dml_runtime_mean_alpha_sec",
    "dml_runtime_max_alpha_sec",
    "dml_qr_nonzero_mean",
    "dml_z_resid_var_mean",
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
    "ps_selection_target_y",
    "ps_selection_target_d",
    "ps_selection_quantile_specific",
    "ps_post_selection_inference_adjustment",
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
    "message",
)


def empty_post_selection_diagnostics() -> dict[str, Any]:
    """Return neutral post-selection diagnostics for non-post-selection rows."""
    return {
        "ps_n_selected_controls": None,
        "ps_n_selected_instruments": None,
        "ps_n_selected_total": None,
        "ps_share_selected_controls": None,
        "ps_share_selected_instruments": None,
        "ps_instrument_selection_method": None,
        "ps_n_candidate_instruments": None,
        "ps_n_retained_instruments": None,
        "ps_share_retained_instruments": None,
        "ps_all_instruments_retained": False,
        "ps_selected_no_controls": False,
        "ps_selected_no_instruments": False,
        "ps_selected_empty_total": False,
        "ps_first_stage_r2": None,
        "ps_first_stage_adj_r2": None,
        "ps_first_stage_partial_r2": None,
        "ps_first_stage_f_stat": None,
        "ps_first_stage_condition_number": None,
        "ps_selection_method": None,
        "ps_selection_target_y": None,
        "ps_selection_target_d": None,
        "ps_selection_quantile_specific": None,
        "ps_post_selection_inference_adjustment": None,
        "ps_selection_lasso_multiplier": None,
        "ps_lasso_alpha_controls": None,
        "ps_lasso_alpha_instruments": None,
        "ps_lasso_alpha_first_stage": None,
        "ps_lasso_alpha_y_cv": None,
        "ps_lasso_alpha_d_cv": None,
        "ps_lasso_alpha_y_final": None,
        "ps_lasso_alpha_d_final": None,
        "ps_lasso_cv_folds": None,
        "ps_selection_failed": False,
        "ps_first_stage_failed": False,
        "ps_rank_deficient": False,
        "ps_warning_code": POST_SELECTION_WARNING_NONE,
    }


def post_selection_diagnostics(result: EstimationResult) -> dict[str, Any]:
    """Return post-selection diagnostics with non-applicable defaults filled."""
    diagnostics = empty_post_selection_diagnostics()
    for name in POST_SELECTION_DIAGNOSTIC_FIELDS:
        value = getattr(result, name)
        if value is not None:
            diagnostics[name] = value
    return diagnostics


def runtime_diagnostics(result: EstimationResult) -> dict[str, Any]:
    """Return runtime diagnostics with missing stage timings filled."""
    diagnostics: dict[str, Any] = dict(empty_runtime_columns())
    for name in RUNTIME_COLUMNS:
        value = getattr(result, name)
        if value is not None:
            diagnostics[name] = value
    return diagnostics


def empty_dml_diagnostics() -> dict[str, Any]:
    """Return neutral DML diagnostics for non-DML rows."""
    return {
        "dml_quantile_penalty": None,
        "dml_ridge_alpha": None,
        "dml_quantile_solver": None,
        "dml_qr_fit_count": None,
        "dml_runtime_mean_alpha_sec": np.nan,
        "dml_runtime_max_alpha_sec": np.nan,
        "dml_qr_nonzero_mean": np.nan,
        "dml_z_resid_var_mean": np.nan,
    }


def dml_diagnostics(result: EstimationResult) -> dict[str, Any]:
    """Return DML diagnostics with non-applicable defaults filled."""
    diagnostics = empty_dml_diagnostics()
    for name in DML_DIAGNOSTIC_FIELDS:
        value = getattr(result, name)
        if value is not None:
            diagnostics[name] = value
    return diagnostics


def result_diagnostics(result: EstimationResult, alphas: np.ndarray) -> dict[str, Any]:
    """Build alpha-grid and confidence-region diagnostics for a result row."""
    failed_alpha_count = result.failed_alpha_count
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=alphas,
        accepted_mask=None,
        alpha_hat=result.alpha_hat,
        failed_alpha_count=0 if failed_alpha_count is None else failed_alpha_count,
    )
    if failed_alpha_count is None:
        diagnostics["failed_alpha_count"] = None
        diagnostics["failed_alpha_rate"] = np.nan

    for name in (
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
    ):
        diagnostics[name] = _diagnostic_value(result, name, diagnostics[name])

    diagnostics["alpha_grid_size"] = _diagnostic_value(
        result, "alpha_grid_size", diagnostics["alpha_grid_size"]
    )
    if result.grid_strategy == "adaptive":
        diagnostics["alpha_grid_step"] = np.nan
    diagnostics["failed_alpha_count"] = _diagnostic_value(
        result, "failed_alpha_count", diagnostics["failed_alpha_count"]
    )
    diagnostics["cr_lower"] = (
        result.cr_lower if result.cr_lower is not None else diagnostics["cr_lower"]
    )
    diagnostics["cr_upper"] = (
        result.cr_upper if result.cr_upper is not None else diagnostics["cr_upper"]
    )
    diagnostics["cr_length"] = (
        result.cr_length if result.cr_length is not None else diagnostics["cr_length"]
    )
    diagnostics["cr_empty"] = result.cr_empty
    diagnostics["cr_disconnected"] = (
        result.cr_disconnected
        if result.cr_disconnected is not None
        else diagnostics["cr_disconnected"]
    )
    return diagnostics


def build_simulation_result_row(
    design: Design,
    result: EstimationResult,
    alphas: np.ndarray,
    *,
    max_error_message_length: int = MAX_ERROR_MESSAGE_LENGTH,
) -> dict[str, Any]:
    """Build one standardized estimator result row."""
    bias = None
    absolute_error = None
    squared_error = None
    if result.alpha_hat is not None and result.alpha_true is not None:
        bias = result.alpha_hat - result.alpha_true
        absolute_error = abs(bias)
        squared_error = bias**2

    diagnostics = result_diagnostics(result, alphas)
    row = {
        "dgp": design.dgp,
        "n": design.n,
        "p": design.p,
        "pi": design.pi,
        "tau": design.tau,
        "rep": design.rep,
        "seed": design.seed,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "estimator": result.estimator,
        "alpha_hat": result.alpha_hat,
        "alpha_true": result.alpha_true,
        "bias": bias,
        "absolute_error": absolute_error,
        "squared_error": squared_error,
        "status": "failed" if result.failed else "ok",
        "failed": result.failed,
        "converged": result.converged,
        "error_type": (
            result.error_type
            if result.failed and result.error_type is not None
            else ("EstimatorFailure" if result.failed else None)
        ),
        "error_message": (
            result.message[:max_error_message_length] if result.failed else None
        ),
        "cr_lower": diagnostics["cr_lower"],
        "cr_upper": diagnostics["cr_upper"],
        "cr_length": diagnostics["cr_length"],
        "cr_components": (
            None
            if result.cr_components is None
            else serialize_cr_components(result.cr_components)
        ),
        "cr_empty": diagnostics["cr_empty"],
        "cr_covers_true": result.cr_covers_true,
        "cr_hits_lower_boundary": diagnostics["cr_hits_lower_boundary"],
        "cr_hits_upper_boundary": diagnostics["cr_hits_upper_boundary"],
        "cr_hits_any_boundary": diagnostics["cr_hits_any_boundary"],
        "cr_accepted_alpha_count": diagnostics["cr_accepted_alpha_count"],
        "cr_acceptance_rate": diagnostics["cr_acceptance_rate"],
        "cr_n_blocks": diagnostics["cr_n_blocks"],
        "cr_disconnected": diagnostics["cr_disconnected"],
        "cr_hull_length": diagnostics["cr_hull_length"],
        "hard_failure_policy": result.hard_failure_policy,
        "cr_status": result.cr_status,
        "cr_is_numerically_resolved": result.cr_is_numerically_resolved,
        "cr_unresolved_count": result.cr_unresolved_count,
        "cr_unresolved_alphas": result.cr_unresolved_alphas,
        "coverage_status": result.coverage_status,
        "point_estimate_status": result.point_estimate_status,
        "usable_alpha_evaluations": result.usable_alpha_evaluations,
        "unresolved_alpha_evaluations": result.unresolved_alpha_evaluations,
        "grid_strategy": result.grid_strategy,
        "adaptive_midpoint_probe": result.adaptive_midpoint_probe,
        "alpha_hat_grid": result.alpha_hat_grid,
        "midpoint_intervals_considered": result.midpoint_intervals_considered,
        "midpoint_evaluations_added": result.midpoint_evaluations_added,
        "midpoint_unresolved_barriers": result.midpoint_unresolved_barriers,
        "midpoint_probe_limit_hit": result.midpoint_probe_limit_hit,
        "initial_alpha_grid_size": result.initial_alpha_grid_size,
        "final_alpha_evaluations": result.final_alpha_evaluations,
        "refinement_tolerance": result.refinement_tolerance,
        "refinement_depth_reached": result.refinement_depth_reached,
        "refinement_limit_hit": result.refinement_limit_hit,
        "max_alpha_evaluations_hit": result.max_alpha_evaluations_hit,
        "number_of_refined_intervals": result.number_of_refined_intervals,
        "number_of_unresolved_refinement_barriers": (
            result.number_of_unresolved_refinement_barriers
        ),
        "minimum_final_grid_spacing": result.minimum_final_grid_spacing,
        "median_final_grid_spacing": result.median_final_grid_spacing,
        "maximum_final_grid_spacing": result.maximum_final_grid_spacing,
        "iteration_warning_evaluations": result.iteration_warning_evaluations,
        "rank_deficient_covariance_failures": (
            result.rank_deficient_covariance_failures
        ),
        "alpha_grid_min": diagnostics["alpha_grid_min"],
        "alpha_grid_max": diagnostics["alpha_grid_max"],
        "alpha_grid_size": diagnostics["alpha_grid_size"],
        "alpha_grid_step": diagnostics["alpha_grid_step"],
        "alpha_hat_at_lower_boundary": diagnostics["alpha_hat_at_lower_boundary"],
        "alpha_hat_at_upper_boundary": diagnostics["alpha_hat_at_upper_boundary"],
        "alpha_hat_at_any_boundary": diagnostics["alpha_hat_at_any_boundary"],
        "failed_alpha_count": diagnostics["failed_alpha_count"],
        "failed_alpha_rate": diagnostics["failed_alpha_rate"],
        "min_test_stat": diagnostics["min_test_stat"],
        "max_test_stat": diagnostics["max_test_stat"],
        "test_stat_at_alpha_hat": diagnostics["test_stat_at_alpha_hat"],
        "critical_value": diagnostics["critical_value"],
        "critical_value_nominal": diagnostics["critical_value_nominal"],
        "critical_value_multiplier": diagnostics["critical_value_multiplier"],
        "critical_value_adjusted": diagnostics["critical_value_adjusted"],
        "selected_controls": result.selected_controls,
        "runtime_seconds": result.runtime_seconds,
        **runtime_diagnostics(result),
        **dml_diagnostics(result),
        **post_selection_diagnostics(result),
        "message": result.message,
    }
    return ensure_result_schema(row)


def build_failure_result_row(
    *,
    design: Design,
    estimator: str,
    alphas: np.ndarray,
    alpha_true: float | None,
    exc: Exception,
    message: str,
    max_error_message_length: int = MAX_ERROR_MESSAGE_LENGTH,
    critical_value_multiplier: float | None = None,
) -> dict[str, Any]:
    """Build one standardized failure row for pre-estimator exceptions."""
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=alphas,
        accepted_mask=None,
        alpha_hat=None,
        failed_alpha_count=0,
    )
    diagnostics["failed_alpha_count"] = None
    diagnostics["failed_alpha_rate"] = np.nan
    if critical_value_multiplier is not None:
        diagnostics["critical_value_multiplier"] = float(critical_value_multiplier)
    is_ch_estimator = estimator in {"oracle", "post_selection_ivqr"}
    result = EstimationResult(
        estimator=estimator,
        alpha_hat=None,
        alpha_true=alpha_true,
        tau=design.tau,
        converged=False,
        failed=True,
        message=message,
        objective_value=None,
        at_grid_boundary=False,
        alpha_grid_size=int(alphas.size),
        failed_alpha_count=None,
        cr_lower=None,
        cr_upper=None,
        cr_length=None,
        cr_covers_true=None,
        cr_empty=True,
        cr_disconnected=False if is_ch_estimator else None,
        selected_controls=None,
        runtime_seconds=float("nan"),
        error_type=type(exc).__name__,
        cr_components=() if is_ch_estimator else None,
        cr_n_blocks=0 if is_ch_estimator else None,
    )
    row = build_simulation_result_row(design, result, alphas)
    row["error_message"] = str(exc)[:max_error_message_length]
    row["critical_value_multiplier"] = diagnostics["critical_value_multiplier"]
    return row


def ensure_result_schema(row: dict[str, Any]) -> dict[str, Any]:
    """Return a row containing every standard output column."""
    completed = dict(row)
    for column in RESULT_COLUMNS:
        completed.setdefault(column, None)
    return {column: completed[column] for column in RESULT_COLUMNS}


def _diagnostic_value(
    result: EstimationResult,
    name: str,
    fallback: object,
) -> object:
    value = getattr(result, name)
    return fallback if value is None else value


__all__ = [
    "MAX_ERROR_MESSAGE_LENGTH",
    "RESULT_SCHEMA_VERSION",
    "RESULT_COLUMNS",
    "build_failure_result_row",
    "build_simulation_result_row",
    "dml_diagnostics",
    "empty_dml_diagnostics",
    "empty_post_selection_diagnostics",
    "ensure_result_schema",
    "post_selection_diagnostics",
    "result_diagnostics",
    "runtime_diagnostics",
]
