"""Experimental IVQR-aligned post-selection estimator."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from dgp.designs import SimData
from estimators.base import (
    EstimationResult,
    estimation_result_diagnostic_kwargs,
    post_selection_aligned_result_diagnostic_kwargs,
    post_selection_quantile_result_diagnostic_kwargs,
    post_selection_result_diagnostic_kwargs,
)
from estimators.ch_inverse_ivqr import as_2d_instruments
from estimators.post_selection_ivqr import (
    _elapsed_since,
    _validate_selection_config,
    evaluate_post_selection_alpha,
    summarize_post_selection_diagnostics,
)
from estimators.post_selection_quantile_ivqr import (
    _select_controls_treatment_lasso,
    _validate_penalty_grid,
    select_controls_quantile_y,
)
from estimators.quantile_selection import (
    build_alpha_anchors,
    format_float_sequence,
    union_selected_indices,
)
from inference.alpha_grid import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_ALPHA_STEP,
    alpha_grid,
)
from inference.confidence_regions import (
    adjust_critical_value,
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    merge_region_and_grid_diagnostics,
    sanitize_grid_statistics,
    summarize_alpha_grid_diagnostics,
    validate_critical_value_multiplier,
)
from simulation.config import (
    DEFAULT_QUANTILE_SELECTION_ALPHAS,
    DEFAULT_QUANTILE_SELECTION_CV_FOLDS,
    DEFAULT_QUANTREG_MAX_ITER,
    DEFAULT_SELECTION_COEF_TOL,
)
from simulation.results import (
    empty_post_selection_aligned_diagnostics,
    empty_post_selection_diagnostics,
    empty_post_selection_quantile_diagnostics,
)
from utils.timing import RuntimeDiagnosticColumns, estimator_runtime_columns
from utils.validation import validate_alpha_grid, validate_data_arrays, validate_tau


PSA_SELECTION_METHOD = "ivqr_aligned_quantile_l1_cv"
PSA_ANCHOR_RULE = "grid_quartiles"


@dataclass(frozen=True)
class AnchorSelectionResult:
    """Selection details for one transformed-outcome alpha anchor."""

    alpha_anchor: float
    selected_indices: np.ndarray
    penalty_selected: float | None
    failed: bool
    message: str


@dataclass(frozen=True)
class AlignedSelectionResult:
    """Combined anchor-union and treatment-selection details."""

    alpha_anchors: np.ndarray
    anchor_results: tuple[AnchorSelectionResult, ...]
    selected_anchor_union: np.ndarray
    selected_treatment: np.ndarray
    selected_final: np.ndarray
    treatment_alpha: float | None


def _selected_penalties_string(anchor_results: tuple[AnchorSelectionResult, ...]) -> str:
    entries = []
    for result in anchor_results:
        penalty = "failed" if result.failed else f"{float(result.penalty_selected):g}"
        entries.append(f"{result.alpha_anchor:g}:{penalty}")
    return ";".join(entries)


def select_controls_ivqr_aligned(
    y: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    tau: float,
    alpha_anchors: np.ndarray,
    *,
    quantile_candidate_alphas: tuple[float, ...] = DEFAULT_QUANTILE_SELECTION_ALPHAS,
    quantile_cv: int = DEFAULT_QUANTILE_SELECTION_CV_FOLDS,
    treatment_cv: int = 3,
    treatment_max_iter: int = 10000,
    coef_tol: float = DEFAULT_SELECTION_COEF_TOL,
    random_state: int | None = None,
) -> AlignedSelectionResult:
    """Select fixed controls from transformed IVQR outcomes and D~X."""
    tau = validate_tau(tau)
    y, d, x = validate_data_arrays(y, d, x)
    alpha_anchors = validate_alpha_grid(alpha_anchors)
    quantile_candidate_alphas = _validate_penalty_grid(quantile_candidate_alphas)
    _validate_selection_config(quantile_cv, max_iter=1, n=x.shape[0])

    anchor_results: list[AnchorSelectionResult] = []
    successful_selections: list[np.ndarray] = []
    for anchor in alpha_anchors:
        y_tilde = y - float(anchor) * d
        try:
            result = select_controls_quantile_y(
                y_tilde,
                x,
                tau,
                candidate_alphas=quantile_candidate_alphas,
                cv=quantile_cv,
                coef_tol=coef_tol,
                random_state=random_state,
            )
        except Exception as exc:  # noqa: BLE001 - failed anchors are diagnostics.
            anchor_results.append(
                AnchorSelectionResult(
                    alpha_anchor=float(anchor),
                    selected_indices=np.empty(0, dtype=int),
                    penalty_selected=None,
                    failed=True,
                    message=str(exc),
                )
            )
            continue
        anchor_results.append(
            AnchorSelectionResult(
                alpha_anchor=float(anchor),
                selected_indices=result.selected_indices,
                penalty_selected=result.alpha_selected,
                failed=False,
                message=result.message,
            )
        )
        successful_selections.append(result.selected_indices)

    if not successful_selections:
        raise RuntimeError("All IVQR-aligned anchor selections failed")

    selected_anchor_union = union_selected_indices(
        successful_selections,
        total=x.shape[1],
    )
    selected_treatment, treatment_alpha = _select_controls_treatment_lasso(
        d,
        x,
        random_state=random_state,
        cv=treatment_cv,
        max_iter=treatment_max_iter,
        coef_tol=coef_tol,
    )
    selected_final = union_selected_indices(
        [selected_anchor_union, selected_treatment],
        total=x.shape[1],
    )
    return AlignedSelectionResult(
        alpha_anchors=np.asarray(alpha_anchors, dtype=float),
        anchor_results=tuple(anchor_results),
        selected_anchor_union=selected_anchor_union,
        selected_treatment=selected_treatment,
        selected_final=selected_final,
        treatment_alpha=treatment_alpha,
    )


def summarize_aligned_post_selection_diagnostics(
    *,
    n_controls: int,
    alpha_anchors: np.ndarray,
    selected_anchor_union: np.ndarray | list[int] | tuple[int, ...] | None,
    selected_treatment: np.ndarray | list[int] | tuple[int, ...] | None,
    selected_final: np.ndarray | list[int] | tuple[int, ...] | None,
    anchor_results: tuple[AnchorSelectionResult, ...] = (),
    quantile_cv_folds: int | None,
    quantile_penalty_grid: tuple[float, ...],
    anchor_selection_failed: bool = False,
) -> dict[str, Any]:
    """Summarize PSA-specific diagnostics."""
    n_anchor = 0 if selected_anchor_union is None else int(np.asarray(selected_anchor_union).size)
    n_treatment = 0 if selected_treatment is None else int(np.asarray(selected_treatment).size)
    n_final = 0 if selected_final is None else int(np.asarray(selected_final).size)
    failed_anchors = sum(1 for result in anchor_results if result.failed)
    diagnostics = empty_post_selection_aligned_diagnostics()
    diagnostics.update(
        {
            "psa_selection_method": PSA_SELECTION_METHOD,
            "psa_anchor_rule": PSA_ANCHOR_RULE,
            "psa_alpha_anchor_count": int(np.asarray(alpha_anchors).size),
            "psa_alpha_anchors": format_float_sequence(alpha_anchors),
            "psa_n_selected_controls_anchor_union": n_anchor,
            "psa_share_selected_controls_anchor_union": (
                n_anchor / n_controls if n_controls > 0 else np.nan
            ),
            "psa_n_selected_controls_treatment": n_treatment,
            "psa_n_selected_controls_final_union": n_final,
            "psa_share_selected_controls_final_union": (
                n_final / n_controls if n_controls > 0 else np.nan
            ),
            "psa_anchor_selection_failed": bool(anchor_selection_failed),
            "psa_n_failed_anchors": int(failed_anchors),
            "psa_selected_empty_anchor_union": n_anchor == 0,
            "psa_selected_empty_final": n_final == 0,
            "psa_quantile_cv_folds": quantile_cv_folds,
            "psa_quantile_penalty_grid": format_float_sequence(quantile_penalty_grid),
            "psa_selected_penalties_by_anchor": _selected_penalties_string(anchor_results),
        }
    )
    return diagnostics


def _failed_result(
    data: SimData,
    tau: float,
    message: str,
    error_type: str,
    selected_controls: int | None,
    runtime_seconds: float,
    alpha_grid_size: int | None = None,
    failed_alpha_count: int | None = None,
    ps_diagnostics: dict[str, Any] | None = None,
    psa_diagnostics: dict[str, Any] | None = None,
    runtime_diagnostics: RuntimeDiagnosticColumns | None = None,
) -> EstimationResult:
    return EstimationResult(
        estimator="post_selection_ivqr_aligned",
        alpha_hat=None,
        alpha_true=data.alpha_true,
        tau=tau,
        converged=False,
        failed=True,
        message=message,
        error_type=error_type,
        objective_value=None,
        at_grid_boundary=False,
        alpha_grid_size=alpha_grid_size,
        failed_alpha_count=failed_alpha_count,
        cr_lower=None,
        cr_upper=None,
        cr_length=None,
        cr_covers_true=None,
        cr_empty=True,
        cr_disconnected=None,
        selected_controls=selected_controls,
        runtime_seconds=runtime_seconds,
        **(
            estimator_runtime_columns(
                estimator="post_selection_ivqr_aligned",
                total_sec=runtime_seconds,
            )
            if runtime_diagnostics is None
            else runtime_diagnostics
        ),
        **post_selection_result_diagnostic_kwargs(
            empty_post_selection_diagnostics()
            if ps_diagnostics is None
            else ps_diagnostics
        ),
        **post_selection_quantile_result_diagnostic_kwargs(
            empty_post_selection_quantile_diagnostics()
        ),
        **post_selection_aligned_result_diagnostic_kwargs(
            empty_post_selection_aligned_diagnostics()
            if psa_diagnostics is None
            else psa_diagnostics
        ),
    )


def estimate_post_selection_ivqr_aligned(
    data: SimData,
    tau: float,
    alphas: np.ndarray | None = None,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    alpha_step: float = DEFAULT_ALPHA_STEP,
    confidence_level: float = 0.95,
    critical_value_multiplier: float = 1.0,
    selection_random_state: int | None = 123,
    selection_cv: int = 3,
    selection_max_iter: int = 10000,
    quantile_selection_alphas: tuple[float, ...] = DEFAULT_QUANTILE_SELECTION_ALPHAS,
    quantile_selection_cv: int = DEFAULT_QUANTILE_SELECTION_CV_FOLDS,
    selection_coef_tol: float = DEFAULT_SELECTION_COEF_TOL,
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
) -> EstimationResult:
    """Estimate experimental IVQR-aligned post-selection IVQR."""
    start = perf_counter()
    tau = validate_tau(tau)
    critical_value_multiplier = validate_critical_value_multiplier(
        critical_value_multiplier
    )
    if quantreg_max_iter <= 0:
        raise ValueError("quantreg_max_iter must be positive")
    y, d, z, x = validate_data_arrays(data.y, data.d, data.x, data.z)
    z_2d = as_2d_instruments(z)
    quantile_selection_alphas = _validate_penalty_grid(quantile_selection_alphas)
    _validate_selection_config(selection_cv, selection_max_iter, x.shape[0])
    _validate_selection_config(quantile_selection_cv, max_iter=1, n=x.shape[0])
    if alphas is None:
        alphas = alpha_grid(alpha_min, alpha_max, alpha_step)
    else:
        alphas = validate_alpha_grid(alphas)
    alpha_anchors = build_alpha_anchors(float(alphas.min()), float(alphas.max()))

    anchor_selection_sec = float("nan")
    diagnostics_sec = float("nan")
    treatment_selection_sec = float("nan")
    alpha_loop_sec = float("nan")
    anchor_start: float | None = None
    treatment_start: float | None = None
    partial_anchor_results: tuple[AnchorSelectionResult, ...] = ()
    try:
        anchor_start = perf_counter()
        anchor_results: list[AnchorSelectionResult] = []
        successful_selections: list[np.ndarray] = []
        for anchor in alpha_anchors:
            y_tilde = y - float(anchor) * d
            try:
                anchor_selection = select_controls_quantile_y(
                    y_tilde,
                    x,
                    tau,
                    candidate_alphas=quantile_selection_alphas,
                    cv=quantile_selection_cv,
                    coef_tol=selection_coef_tol,
                    random_state=selection_random_state,
                )
            except Exception as anchor_exc:  # noqa: BLE001 - keep partial anchors.
                anchor_results.append(
                    AnchorSelectionResult(
                        alpha_anchor=float(anchor),
                        selected_indices=np.empty(0, dtype=int),
                        penalty_selected=None,
                        failed=True,
                        message=str(anchor_exc),
                    )
                )
                partial_anchor_results = tuple(anchor_results)
                continue
            anchor_results.append(
                AnchorSelectionResult(
                    alpha_anchor=float(anchor),
                    selected_indices=anchor_selection.selected_indices,
                    penalty_selected=anchor_selection.alpha_selected,
                    failed=False,
                    message=anchor_selection.message,
                )
            )
            partial_anchor_results = tuple(anchor_results)
            successful_selections.append(anchor_selection.selected_indices)
        if not successful_selections:
            messages = "; ".join(
                result.message for result in anchor_results if result.message
            )
            suffix = f": {messages}" if messages else ""
            raise RuntimeError(f"All IVQR-aligned anchor selections failed{suffix}")
        selected_anchor_union = union_selected_indices(
            successful_selections,
            total=x.shape[1],
        )
        anchor_selection_sec = _elapsed_since(anchor_start)
        treatment_start = perf_counter()
        selected_treatment, treatment_alpha = _select_controls_treatment_lasso(
            d,
            x,
            random_state=selection_random_state,
            cv=selection_cv,
            max_iter=selection_max_iter,
            coef_tol=selection_coef_tol,
        )
        treatment_selection_sec = _elapsed_since(treatment_start)
        selected_final = union_selected_indices(
            [selected_anchor_union, selected_treatment],
            total=x.shape[1],
        )
        selection = AlignedSelectionResult(
            alpha_anchors=alpha_anchors,
            anchor_results=tuple(anchor_results),
            selected_anchor_union=selected_anchor_union,
            selected_treatment=selected_treatment,
            selected_final=selected_final,
            treatment_alpha=treatment_alpha,
        )
    except Exception as exc:  # noqa: BLE001 - report cleanly.
        diagnostics_start = perf_counter()
        ps_diagnostics = summarize_post_selection_diagnostics(
            d=d,
            x=x,
            z=z_2d,
            selected_control_indices=None,
            selected_instrument_indices=np.arange(z_2d.shape[1]),
            lasso_cv_folds=selection_cv,
            selection_failed=True,
            selection_method="ivqr_aligned",
            warning_code="quantile_selection_failed",
        )
        psa_diagnostics = summarize_aligned_post_selection_diagnostics(
            n_controls=x.shape[1],
            alpha_anchors=alpha_anchors,
            selected_anchor_union=None,
            selected_treatment=None,
            selected_final=None,
            anchor_results=partial_anchor_results,
            quantile_cv_folds=quantile_selection_cv,
            quantile_penalty_grid=quantile_selection_alphas,
            anchor_selection_failed=True,
        )
        diagnostics_sec = _elapsed_since(diagnostics_start)
        runtime_seconds = perf_counter() - start
        return _failed_result(
            data=data,
            tau=tau,
            message=f"IVQR-aligned control selection failed: {exc}",
            error_type="quantile_selection_failed",
            selected_controls=None,
            runtime_seconds=runtime_seconds,
            ps_diagnostics=ps_diagnostics,
            psa_diagnostics=psa_diagnostics,
            runtime_diagnostics=estimator_runtime_columns(
                estimator="post_selection_ivqr_aligned",
                total_sec=runtime_seconds,
                diagnostics_sec=diagnostics_sec,
                anchor_selection_sec=_elapsed_since(anchor_start),
                treatment_selection_sec=_elapsed_since(treatment_start),
            ),
        )

    selected_indices = selection.selected_final
    diagnostics_start = perf_counter()
    ps_diagnostics = summarize_post_selection_diagnostics(
        d=d,
        x=x,
        z=z_2d,
        selected_control_indices=selected_indices,
        selected_instrument_indices=np.arange(z_2d.shape[1]),
        lasso_alpha_first_stage=selection.treatment_alpha,
        lasso_cv_folds=selection_cv,
        selection_method="ivqr_aligned",
    )
    psa_diagnostics = summarize_aligned_post_selection_diagnostics(
        n_controls=x.shape[1],
        alpha_anchors=selection.alpha_anchors,
        selected_anchor_union=selection.selected_anchor_union,
        selected_treatment=selection.selected_treatment,
        selected_final=selection.selected_final,
        anchor_results=selection.anchor_results,
        quantile_cv_folds=quantile_selection_cv,
        quantile_penalty_grid=quantile_selection_alphas,
        anchor_selection_failed=False,
    )
    diagnostics_sec = _elapsed_since(diagnostics_start)
    x_selected = (
        x[:, selected_indices]
        if selected_indices.size
        else np.empty((x.shape[0], 0))
    )
    num_qr_regressors = 1 + int(selected_indices.size) + int(z_2d.shape[1])
    if num_qr_regressors >= x.shape[0]:
        runtime_seconds = perf_counter() - start
        return _failed_result(
            data=data,
            tau=tau,
            message=(
                "IVQR-aligned post-selection infeasible: QR design dimension "
                f"is at least sample size (regressors={num_qr_regressors}, n={x.shape[0]})."
            ),
            error_type="EstimatorFailure",
            selected_controls=int(selected_indices.size),
            runtime_seconds=runtime_seconds,
            ps_diagnostics=ps_diagnostics,
            psa_diagnostics=psa_diagnostics,
            runtime_diagnostics=estimator_runtime_columns(
                estimator="post_selection_ivqr_aligned",
                total_sec=runtime_seconds,
                diagnostics_sec=diagnostics_sec,
                anchor_selection_sec=anchor_selection_sec,
                treatment_selection_sec=treatment_selection_sec,
            ),
        )

    statistics = np.empty(len(alphas), dtype=float)
    converged_flags: list[bool] = []
    alpha_loop_start = perf_counter()
    for j, alpha in enumerate(alphas):
        try:
            statistic, converged, _message = evaluate_post_selection_alpha(
                y=y,
                d=d,
                z=z,
                x_selected=x_selected,
                alpha=float(alpha),
                tau=tau,
                max_iter=quantreg_max_iter,
            )
        except Exception:
            statistic, converged = np.inf, False
        statistics[j] = statistic
        converged_flags.append(converged)
    alpha_loop_sec = perf_counter() - alpha_loop_start
    statistics, num_failed = sanitize_grid_statistics(statistics, converged_flags)
    if num_failed == len(alphas):
        runtime_seconds = perf_counter() - start
        return _failed_result(
            data=data,
            tau=tau,
            message=f"All alpha-grid evaluations failed; failed_alpha_points={num_failed}/{len(alphas)}",
            error_type="EstimatorFailure",
            selected_controls=int(selected_indices.size),
            runtime_seconds=runtime_seconds,
            alpha_grid_size=len(alphas),
            failed_alpha_count=num_failed,
            ps_diagnostics=ps_diagnostics,
            psa_diagnostics=psa_diagnostics,
            runtime_diagnostics=estimator_runtime_columns(
                estimator="post_selection_ivqr_aligned",
                total_sec=runtime_seconds,
                alpha_loop_sec=alpha_loop_sec,
                diagnostics_sec=diagnostics_sec,
                anchor_selection_sec=anchor_selection_sec,
                treatment_selection_sec=treatment_selection_sec,
            ),
        )

    confidence_region_start = perf_counter()
    alpha_hat, min_statistic, at_boundary = argmin_grid(alphas, statistics)
    critical = critical_value_chi_square(confidence_level, df=z_2d.shape[1])
    adjusted_critical = adjust_critical_value(critical, critical_value_multiplier)
    accepted_mask = statistics <= adjusted_critical
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=alphas,
        accepted_mask=accepted_mask,
        alpha_hat=alpha_hat,
        failed_alpha_count=num_failed,
        test_stats=statistics,
        critical_value=adjusted_critical,
        critical_value_nominal=critical,
        critical_value_multiplier=critical_value_multiplier,
        critical_value_adjusted=adjusted_critical,
    )
    region = invert_score_test(
        alphas=alphas,
        statistics=statistics,
        critical_value=critical,
        critical_value_multiplier=critical_value_multiplier,
        alpha_true=data.alpha_true,
        statistic_reference=0.0,
        inversion_type="absolute",
    )
    diagnostics = merge_region_and_grid_diagnostics(region, diagnostics)
    confidence_region_sec = perf_counter() - confidence_region_start
    runtime_seconds = perf_counter() - start
    return EstimationResult(
        estimator="post_selection_ivqr_aligned",
        alpha_hat=alpha_hat,
        alpha_true=data.alpha_true,
        tau=tau,
        converged=True,
        failed=False,
        message=(
            f"ok; failed_alpha_points={num_failed}/{len(alphas)}; "
            f"selected_anchor_union={selection.selected_anchor_union.size}; "
            f"selected_treatment={selection.selected_treatment.size}; "
            f"selected_final={selection.selected_final.size}"
        ),
        objective_value=min_statistic,
        at_grid_boundary=at_boundary,
        alpha_grid_size=len(alphas),
        failed_alpha_count=num_failed,
        cr_lower=diagnostics["cr_lower"],
        cr_upper=diagnostics["cr_upper"],
        cr_length=diagnostics["cr_length"],
        cr_covers_true=region.covers_true,
        cr_empty=diagnostics["cr_empty"],
        cr_disconnected=diagnostics["cr_disconnected"],
        selected_controls=int(selected_indices.size),
        runtime_seconds=runtime_seconds,
        **estimator_runtime_columns(
            estimator="post_selection_ivqr_aligned",
            total_sec=runtime_seconds,
            alpha_loop_sec=alpha_loop_sec,
            confidence_region_sec=confidence_region_sec,
            diagnostics_sec=diagnostics_sec,
            anchor_selection_sec=anchor_selection_sec,
            treatment_selection_sec=treatment_selection_sec,
        ),
        **estimation_result_diagnostic_kwargs(diagnostics),
        **post_selection_result_diagnostic_kwargs(ps_diagnostics),
        **post_selection_quantile_result_diagnostic_kwargs(
            empty_post_selection_quantile_diagnostics()
        ),
        **post_selection_aligned_result_diagnostic_kwargs(psa_diagnostics),
    )


__all__ = [
    "PSA_ANCHOR_RULE",
    "PSA_SELECTION_METHOD",
    "AlignedSelectionResult",
    "AnchorSelectionResult",
    "estimate_post_selection_ivqr_aligned",
    "select_controls_ivqr_aligned",
    "summarize_aligned_post_selection_diagnostics",
]
