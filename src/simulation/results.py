"""Shared result-row construction utilities for simulation outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from dgp.designs import Design
from estimators.base import (
    EstimationResult,
    POST_SELECTION_DIAGNOSTIC_FIELDS,
    POST_SELECTION_QUANTILE_DIAGNOSTIC_FIELDS,
    POST_SELECTION_ALIGNED_DIAGNOSTIC_FIELDS,
)
from inference.confidence_regions import (
    merge_region_and_grid_diagnostics,
    summarize_alpha_grid_diagnostics,
)
from utils.timing import (
    RUNTIME_COLUMNS,
    RuntimeDiagnosticColumns,
    empty_runtime_columns,
)


POST_SELECTION_WARNING_NONE = ""
MAX_ERROR_MESSAGE_LENGTH = 500

RESULT_COLUMNS: tuple[str, ...] = (
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
    "bias",
    "absolute_error",
    "squared_error",
    "status",
    "error_type",
    "error_message",
    "failed",
    "converged",
    "alpha_grid_min",
    "alpha_grid_max",
    "alpha_grid_size",
    "alpha_grid_step",
    "alpha_hat_at_lower_boundary",
    "alpha_hat_at_upper_boundary",
    "alpha_hat_at_any_boundary",
    "cr_lower",
    "cr_upper",
    "cr_length",
    "cr_hits_lower_boundary",
    "cr_hits_upper_boundary",
    "cr_hits_any_boundary",
    "cr_empty",
    "cr_accepted_alpha_count",
    "cr_acceptance_rate",
    "cr_n_blocks",
    "cr_disconnected",
    "cr_hull_length",
    "cr_covers_true",
    "selected_controls",
    "runtime_seconds",
    *RUNTIME_COLUMNS,
    "failed_alpha_count",
    "failed_alpha_rate",
    "min_test_stat",
    "max_test_stat",
    "test_stat_at_alpha_hat",
    "critical_value",
    "critical_value_nominal",
    "critical_value_multiplier",
    "critical_value_adjusted",
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
    "ps_lasso_alpha_controls",
    "ps_lasso_alpha_instruments",
    "ps_lasso_alpha_first_stage",
    "ps_lasso_cv_folds",
    "ps_selection_failed",
    "ps_first_stage_failed",
    "ps_rank_deficient",
    "ps_warning_code",
    "psq_selection_method",
    "psq_quantile_tau",
    "psq_quantile_alpha_selected",
    "psq_quantile_cv_folds",
    "psq_n_selected_controls_quantile_y",
    "psq_n_selected_controls_treatment_d",
    "psq_n_selected_controls_union",
    "psq_share_selected_controls_quantile_y",
    "psq_share_selected_controls_union",
    "psq_selection_failed",
    "psq_warning_code",
    "psa_selection_method",
    "psa_anchor_rule",
    "psa_alpha_anchor_count",
    "psa_alpha_anchors",
    "psa_n_selected_controls_anchor_union",
    "psa_share_selected_controls_anchor_union",
    "psa_n_selected_controls_treatment",
    "psa_n_selected_controls_final_union",
    "psa_share_selected_controls_final_union",
    "psa_anchor_selection_failed",
    "psa_n_failed_anchors",
    "psa_selected_empty_anchor_union",
    "psa_selected_empty_final",
    "psa_quantile_cv_folds",
    "psa_quantile_penalty_grid",
    "psa_selected_penalties_by_anchor",
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
        "ps_lasso_alpha_controls": None,
        "ps_lasso_alpha_instruments": None,
        "ps_lasso_alpha_first_stage": None,
        "ps_lasso_cv_folds": None,
        "ps_selection_failed": False,
        "ps_first_stage_failed": False,
        "ps_rank_deficient": False,
        "ps_warning_code": POST_SELECTION_WARNING_NONE,
    }


def empty_post_selection_quantile_diagnostics() -> dict[str, Any]:
    """Return neutral diagnostics for non-quantile post-selection rows."""
    return {
        "psq_selection_method": None,
        "psq_quantile_tau": None,
        "psq_quantile_alpha_selected": None,
        "psq_quantile_cv_folds": None,
        "psq_n_selected_controls_quantile_y": None,
        "psq_n_selected_controls_treatment_d": None,
        "psq_n_selected_controls_union": None,
        "psq_share_selected_controls_quantile_y": None,
        "psq_share_selected_controls_union": None,
        "psq_selection_failed": False,
        "psq_warning_code": POST_SELECTION_WARNING_NONE,
    }


def empty_post_selection_aligned_diagnostics() -> dict[str, Any]:
    """Return neutral diagnostics for non-IVQR-aligned post-selection rows."""
    return {
        "psa_selection_method": None,
        "psa_anchor_rule": None,
        "psa_alpha_anchor_count": None,
        "psa_alpha_anchors": None,
        "psa_n_selected_controls_anchor_union": None,
        "psa_share_selected_controls_anchor_union": None,
        "psa_n_selected_controls_treatment": None,
        "psa_n_selected_controls_final_union": None,
        "psa_share_selected_controls_final_union": None,
        "psa_anchor_selection_failed": False,
        "psa_n_failed_anchors": None,
        "psa_selected_empty_anchor_union": False,
        "psa_selected_empty_final": False,
        "psa_quantile_cv_folds": None,
        "psa_quantile_penalty_grid": None,
        "psa_selected_penalties_by_anchor": None,
    }


def empty_runtime_diagnostics() -> RuntimeDiagnosticColumns:
    """Return neutral runtime diagnostics for rows without stage timing."""
    return empty_runtime_columns()


def result_diagnostics(
    result: EstimationResult,
    alphas: np.ndarray,
) -> dict[str, Any]:
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
        result,
        "alpha_grid_size",
        diagnostics["alpha_grid_size"],
    )
    diagnostics["failed_alpha_count"] = _diagnostic_value(
        result,
        "failed_alpha_count",
        diagnostics["failed_alpha_count"],
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


def post_selection_diagnostics(result: EstimationResult) -> dict[str, Any]:
    """Return post-selection diagnostics with non-applicable defaults filled."""
    diagnostics = empty_post_selection_diagnostics()
    for name in POST_SELECTION_DIAGNOSTIC_FIELDS:
        value = getattr(result, name)
        if value is not None:
            diagnostics[name] = value
    return diagnostics


def post_selection_quantile_diagnostics(result: EstimationResult) -> dict[str, Any]:
    """Return PSQ diagnostics with non-applicable defaults filled."""
    diagnostics = empty_post_selection_quantile_diagnostics()
    for name in POST_SELECTION_QUANTILE_DIAGNOSTIC_FIELDS:
        value = getattr(result, name)
        if value is not None:
            diagnostics[name] = value
    return diagnostics


def post_selection_aligned_diagnostics(result: EstimationResult) -> dict[str, Any]:
    """Return PSA diagnostics with non-applicable defaults filled."""
    diagnostics = empty_post_selection_aligned_diagnostics()
    for name in POST_SELECTION_ALIGNED_DIAGNOSTIC_FIELDS:
        value = getattr(result, name)
        if value is not None:
            diagnostics[name] = value
    return diagnostics


def runtime_diagnostics(result: EstimationResult) -> dict[str, Any]:
    """Return runtime diagnostics with missing stage timings filled."""
    diagnostics: dict[str, Any] = dict(empty_runtime_diagnostics())
    for name in RUNTIME_COLUMNS:
        value = getattr(result, name)
        if value is not None:
            diagnostics[name] = value
    return diagnostics


def build_simulation_result_row(
    design: Design,
    result: EstimationResult,
    alphas: np.ndarray,
    *,
    max_error_message_length: int = MAX_ERROR_MESSAGE_LENGTH,
    extra: dict[str, Any] | None = None,
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
        "estimator": result.estimator,
        "alpha_hat": result.alpha_hat,
        "alpha_true": result.alpha_true,
        "bias": bias,
        "absolute_error": absolute_error,
        "squared_error": squared_error,
        "status": "failed" if result.failed else "ok",
        "error_type": (
            result.error_type
            if result.failed and result.error_type is not None
            else ("EstimatorFailure" if result.failed else None)
        ),
        "error_message": (
            result.message[:max_error_message_length] if result.failed else None
        ),
        "failed": result.failed,
        "converged": result.converged,
        "alpha_grid_min": diagnostics["alpha_grid_min"],
        "alpha_grid_max": diagnostics["alpha_grid_max"],
        "alpha_grid_size": diagnostics["alpha_grid_size"],
        "alpha_grid_step": diagnostics["alpha_grid_step"],
        "alpha_hat_at_lower_boundary": diagnostics["alpha_hat_at_lower_boundary"],
        "alpha_hat_at_upper_boundary": diagnostics["alpha_hat_at_upper_boundary"],
        "alpha_hat_at_any_boundary": diagnostics["alpha_hat_at_any_boundary"],
        "cr_lower": diagnostics["cr_lower"],
        "cr_upper": diagnostics["cr_upper"],
        "cr_length": diagnostics["cr_length"],
        "cr_hits_lower_boundary": diagnostics["cr_hits_lower_boundary"],
        "cr_hits_upper_boundary": diagnostics["cr_hits_upper_boundary"],
        "cr_hits_any_boundary": diagnostics["cr_hits_any_boundary"],
        "cr_empty": diagnostics["cr_empty"],
        "cr_accepted_alpha_count": diagnostics["cr_accepted_alpha_count"],
        "cr_acceptance_rate": diagnostics["cr_acceptance_rate"],
        "cr_n_blocks": diagnostics["cr_n_blocks"],
        "cr_disconnected": diagnostics["cr_disconnected"],
        "cr_hull_length": diagnostics["cr_hull_length"],
        "cr_covers_true": result.cr_covers_true,
        "selected_controls": result.selected_controls,
        "runtime_seconds": result.runtime_seconds,
        **runtime_diagnostics(result),
        "failed_alpha_count": diagnostics["failed_alpha_count"],
        "failed_alpha_rate": diagnostics["failed_alpha_rate"],
        "min_test_stat": diagnostics["min_test_stat"],
        "max_test_stat": diagnostics["max_test_stat"],
        "test_stat_at_alpha_hat": diagnostics["test_stat_at_alpha_hat"],
        "critical_value": diagnostics["critical_value"],
        "critical_value_nominal": diagnostics["critical_value_nominal"],
        "critical_value_multiplier": diagnostics["critical_value_multiplier"],
        "critical_value_adjusted": diagnostics["critical_value_adjusted"],
        **post_selection_diagnostics(result),
        **post_selection_quantile_diagnostics(result),
        **post_selection_aligned_diagnostics(result),
        "message": result.message,
    }
    if extra:
        row.update(extra)
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
    row = {
        "dgp": design.dgp,
        "n": design.n,
        "p": design.p,
        "pi": design.pi,
        "tau": design.tau,
        "rep": design.rep,
        "seed": design.seed,
        "estimator": estimator,
        "alpha_hat": None,
        "alpha_true": alpha_true,
        "bias": None,
        "absolute_error": None,
        "squared_error": None,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:max_error_message_length],
        "failed": True,
        "converged": False,
        "alpha_grid_min": diagnostics["alpha_grid_min"],
        "alpha_grid_max": diagnostics["alpha_grid_max"],
        "alpha_grid_size": diagnostics["alpha_grid_size"],
        "alpha_grid_step": diagnostics["alpha_grid_step"],
        "alpha_hat_at_lower_boundary": diagnostics["alpha_hat_at_lower_boundary"],
        "alpha_hat_at_upper_boundary": diagnostics["alpha_hat_at_upper_boundary"],
        "alpha_hat_at_any_boundary": diagnostics["alpha_hat_at_any_boundary"],
        "cr_lower": diagnostics["cr_lower"],
        "cr_upper": diagnostics["cr_upper"],
        "cr_length": diagnostics["cr_length"],
        "cr_hits_lower_boundary": diagnostics["cr_hits_lower_boundary"],
        "cr_hits_upper_boundary": diagnostics["cr_hits_upper_boundary"],
        "cr_hits_any_boundary": diagnostics["cr_hits_any_boundary"],
        "cr_empty": diagnostics["cr_empty"],
        "cr_accepted_alpha_count": diagnostics["cr_accepted_alpha_count"],
        "cr_acceptance_rate": diagnostics["cr_acceptance_rate"],
        "cr_n_blocks": diagnostics["cr_n_blocks"],
        "cr_disconnected": diagnostics["cr_disconnected"],
        "cr_hull_length": diagnostics["cr_hull_length"],
        "cr_covers_true": None,
        "selected_controls": None,
        "runtime_seconds": None,
        **empty_runtime_diagnostics(),
        "failed_alpha_count": diagnostics["failed_alpha_count"],
        "failed_alpha_rate": diagnostics["failed_alpha_rate"],
        "min_test_stat": diagnostics["min_test_stat"],
        "max_test_stat": diagnostics["max_test_stat"],
        "test_stat_at_alpha_hat": diagnostics["test_stat_at_alpha_hat"],
        "critical_value": diagnostics["critical_value"],
        "critical_value_nominal": diagnostics["critical_value_nominal"],
        "critical_value_multiplier": diagnostics["critical_value_multiplier"],
        "critical_value_adjusted": diagnostics["critical_value_adjusted"],
        **empty_post_selection_diagnostics(),
        **empty_post_selection_quantile_diagnostics(),
        **empty_post_selection_aligned_diagnostics(),
        "message": message,
    }
    return ensure_result_schema(row)


def ensure_result_schema(row: dict[str, Any]) -> dict[str, Any]:
    """Return a row containing every standard output column."""
    completed = dict(row)
    for column in RESULT_COLUMNS:
        completed.setdefault(column, None)
    return completed


def _diagnostic_value(
    result: EstimationResult,
    name: str,
    fallback: object,
) -> object:
    value = getattr(result, name)
    return fallback if value is None else value


__all__ = [
    "MAX_ERROR_MESSAGE_LENGTH",
    "RESULT_COLUMNS",
    "build_failure_result_row",
    "build_simulation_result_row",
    "empty_post_selection_diagnostics",
    "empty_post_selection_aligned_diagnostics",
    "empty_post_selection_quantile_diagnostics",
    "empty_runtime_diagnostics",
    "ensure_result_schema",
    "merge_region_and_grid_diagnostics",
    "post_selection_diagnostics",
    "post_selection_aligned_diagnostics",
    "post_selection_quantile_diagnostics",
    "result_diagnostics",
    "runtime_diagnostics",
]
