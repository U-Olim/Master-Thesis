"""Experimental quantile-specific post-selection IVQR estimator."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.linear_model import LassoCV, QuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dgp.designs import SimData
from estimators.base import (
    EstimationResult,
    estimation_result_diagnostic_kwargs,
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
    empty_post_selection_diagnostics,
    empty_post_selection_quantile_diagnostics,
)
from utils.timing import RuntimeDiagnosticColumns, estimator_runtime_columns
from utils.validation import validate_alpha_grid, validate_data_arrays, validate_tau


PSQ_SELECTION_METHOD = "quantile_l1_cv"
PSQ_WARNING_NONE = ""


@dataclass(frozen=True)
class QuantileSelectionResult:
    """Outcome quantile-selection details."""

    selected_indices: np.ndarray
    alpha_selected: float | None
    cv_folds: int
    candidate_losses: dict[float, float]
    message: str


@dataclass(frozen=True)
class QuantilePostSelectionResult:
    """Combined quantile-outcome and treatment-control selection details."""

    selected_indices: np.ndarray
    selected_quantile_y: np.ndarray
    selected_treatment_d: np.ndarray
    quantile_alpha_selected: float | None
    quantile_cv_folds: int
    treatment_lasso_alpha: float | None
    message: str


def _validate_penalty_grid(alphas: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if isinstance(alphas, (str, bytes)):
        raise ValueError("quantile_selection_alphas must be a sequence of positives")
    try:
        values = tuple(float(value) for value in alphas)
    except TypeError as exc:
        raise ValueError("quantile_selection_alphas must be a sequence") from exc
    if not values:
        raise ValueError("quantile_selection_alphas must not be empty")
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("quantile_selection_alphas must contain nonnegative values")
    if len(set(values)) != len(values):
        raise ValueError("quantile_selection_alphas must not contain duplicates")
    return values


def _quantile_model(tau: float, penalty: float):
    return make_pipeline(
        StandardScaler(),
        QuantileRegressor(quantile=tau, alpha=penalty, solver="highs"),
    )


def select_controls_quantile_y(
    y: np.ndarray,
    x: np.ndarray,
    tau: float,
    *,
    candidate_alphas: tuple[float, ...] = DEFAULT_QUANTILE_SELECTION_ALPHAS,
    cv: int = DEFAULT_QUANTILE_SELECTION_CV_FOLDS,
    coef_tol: float = DEFAULT_SELECTION_COEF_TOL,
    random_state: int | None = None,
) -> QuantileSelectionResult:
    """Select controls from L1-penalized quantile regression of Y on X."""
    tau = validate_tau(tau)
    y, _d_dummy, x = validate_data_arrays(y, np.zeros_like(y, dtype=float), x)
    candidate_alphas = _validate_penalty_grid(candidate_alphas)
    _validate_selection_config(cv, max_iter=1, n=x.shape[0])
    if coef_tol < 0 or not np.isfinite(coef_tol):
        raise ValueError("coef_tol must be finite and nonnegative")

    if x.shape[1] == 0:
        return QuantileSelectionResult(
            selected_indices=np.empty(0, dtype=int),
            alpha_selected=None,
            cv_folds=cv,
            candidate_losses={},
            message="selected_y_quantile=0",
        )

    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    candidate_losses: dict[float, float] = {}
    invalid_candidates: list[float] = []
    for penalty in candidate_alphas:
        fold_losses: list[float] = []
        candidate_failed = False
        for train_idx, valid_idx in splitter.split(x):
            try:
                model = _quantile_model(tau, penalty)
                model.fit(x[train_idx], y[train_idx])
                prediction = model.predict(x[valid_idx])
                loss = mean_pinball_loss(y[valid_idx], prediction, alpha=tau)
            except Exception:
                candidate_failed = True
                break
            if not np.isfinite(loss):
                candidate_failed = True
                break
            fold_losses.append(float(loss))
        if candidate_failed or not fold_losses:
            invalid_candidates.append(penalty)
            continue
        candidate_losses[penalty] = float(np.mean(fold_losses))

    if not candidate_losses:
        invalid_text = ", ".join(f"{value:g}" for value in invalid_candidates)
        raise RuntimeError(
            "All quantile-selection fits failed"
            + (f" for candidate alpha(s): {invalid_text}" if invalid_text else "")
        )

    best_alpha = min(candidate_losses, key=lambda value: (candidate_losses[value], value))
    final_model = _quantile_model(tau, best_alpha)
    final_model.fit(x, y)
    coefficients = np.asarray(final_model.named_steps["quantileregressor"].coef_)
    selected = np.flatnonzero(np.abs(coefficients) > coef_tol).astype(int)
    return QuantileSelectionResult(
        selected_indices=selected,
        alpha_selected=float(best_alpha),
        cv_folds=cv,
        candidate_losses=candidate_losses,
        message=(
            f"selected_y_quantile={selected.size}; "
            f"quantile_alpha={best_alpha:g}"
        ),
    )


def _select_controls_treatment_lasso(
    d: np.ndarray,
    x: np.ndarray,
    *,
    random_state: int | None,
    cv: int,
    max_iter: int,
    coef_tol: float,
) -> tuple[np.ndarray, float | None]:
    _validate_selection_config(cv, max_iter, x.shape[0])
    if x.shape[1] == 0:
        return np.empty(0, dtype=int), None
    model = make_pipeline(
        StandardScaler(),
        LassoCV(cv=cv, random_state=random_state, max_iter=max_iter),
    )
    model.fit(x, d)
    lasso = model.named_steps["lassocv"]
    selected = np.flatnonzero(np.abs(np.asarray(lasso.coef_)) > coef_tol).astype(int)
    return selected, float(lasso.alpha_)


def select_controls_quantile_post_selection(
    y: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    tau: float,
    *,
    quantile_candidate_alphas: tuple[float, ...] = DEFAULT_QUANTILE_SELECTION_ALPHAS,
    quantile_cv: int = DEFAULT_QUANTILE_SELECTION_CV_FOLDS,
    treatment_cv: int = 3,
    treatment_max_iter: int = 10000,
    coef_tol: float = DEFAULT_SELECTION_COEF_TOL,
    random_state: int | None = None,
) -> QuantilePostSelectionResult:
    """Select quantile-outcome and treatment controls, then take their union."""
    tau = validate_tau(tau)
    y, d, x = validate_data_arrays(y, d, x)
    quantile = select_controls_quantile_y(
        y,
        x,
        tau,
        candidate_alphas=quantile_candidate_alphas,
        cv=quantile_cv,
        coef_tol=coef_tol,
        random_state=random_state,
    )
    selected_d, treatment_alpha = _select_controls_treatment_lasso(
        d,
        x,
        random_state=random_state,
        cv=treatment_cv,
        max_iter=treatment_max_iter,
        coef_tol=coef_tol,
    )
    selected = np.union1d(quantile.selected_indices, selected_d).astype(int)
    message = (
        f"selected_y_quantile={quantile.selected_indices.size}; "
        f"selected_d={selected_d.size}; selected_union={selected.size}; "
        f"quantile_alpha={quantile.alpha_selected}"
    )
    return QuantilePostSelectionResult(
        selected_indices=selected,
        selected_quantile_y=quantile.selected_indices,
        selected_treatment_d=selected_d,
        quantile_alpha_selected=quantile.alpha_selected,
        quantile_cv_folds=quantile.cv_folds,
        treatment_lasso_alpha=treatment_alpha,
        message=message,
    )


def summarize_quantile_post_selection_diagnostics(
    *,
    tau: float,
    n_controls: int,
    selected_quantile_y: np.ndarray | list[int] | tuple[int, ...] | None,
    selected_treatment_d: np.ndarray | list[int] | tuple[int, ...] | None,
    selected_union: np.ndarray | list[int] | tuple[int, ...] | None,
    quantile_alpha_selected: float | None,
    quantile_cv_folds: int | None,
    selection_failed: bool = False,
    warning_code: str | None = None,
) -> dict[str, Any]:
    """Summarize PSQ-specific diagnostics."""
    tau = validate_tau(tau)

    def count(values: np.ndarray | list[int] | tuple[int, ...] | None) -> int:
        if values is None:
            return 0
        return int(np.asarray(values).size)

    n_selected_y = count(selected_quantile_y)
    n_selected_d = count(selected_treatment_d)
    n_selected_union = count(selected_union)
    diagnostics = empty_post_selection_quantile_diagnostics()
    diagnostics.update(
        {
            "psq_selection_method": PSQ_SELECTION_METHOD,
            "psq_quantile_tau": tau,
            "psq_quantile_alpha_selected": (
                None
                if quantile_alpha_selected is None
                else float(quantile_alpha_selected)
            ),
            "psq_quantile_cv_folds": quantile_cv_folds,
            "psq_n_selected_controls_quantile_y": n_selected_y,
            "psq_n_selected_controls_treatment_d": n_selected_d,
            "psq_n_selected_controls_union": n_selected_union,
            "psq_share_selected_controls_quantile_y": (
                n_selected_y / n_controls if n_controls > 0 else np.nan
            ),
            "psq_share_selected_controls_union": (
                n_selected_union / n_controls if n_controls > 0 else np.nan
            ),
            "psq_selection_failed": bool(selection_failed),
            "psq_warning_code": warning_code or PSQ_WARNING_NONE,
        }
    )
    if selection_failed and not diagnostics["psq_warning_code"]:
        diagnostics["psq_warning_code"] = "quantile_selection_failed"
    return diagnostics


def _failed_result(
    data: SimData,
    tau: float,
    message: str,
    selected_controls: int | None,
    runtime_seconds: float,
    alpha_grid_size: int | None = None,
    failed_alpha_count: int | None = None,
    ps_diagnostics: dict[str, Any] | None = None,
    psq_diagnostics: dict[str, Any] | None = None,
    runtime_diagnostics: RuntimeDiagnosticColumns | None = None,
) -> EstimationResult:
    return EstimationResult(
        estimator="post_selection_quantile",
        alpha_hat=None,
        alpha_true=data.alpha_true,
        tau=tau,
        converged=False,
        failed=True,
        message=message,
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
                estimator="post_selection_quantile",
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
            if psq_diagnostics is None
            else psq_diagnostics
        ),
    )


def estimate_post_selection_quantile_ivqr(
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
    """Estimate experimental quantile-specific post-selection IVQR."""
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

    quantile_selection_sec = float("nan")
    treatment_selection_sec = float("nan")
    diagnostics_sec = float("nan")
    alpha_loop_sec = float("nan")
    confidence_region_sec = float("nan")

    quantile_start: float | None = None
    treatment_start: float | None = None
    try:
        quantile_start = perf_counter()
        quantile_details = select_controls_quantile_y(
            y,
            x,
            tau,
            candidate_alphas=quantile_selection_alphas,
            cv=quantile_selection_cv,
            coef_tol=selection_coef_tol,
            random_state=selection_random_state,
        )
        quantile_selection_sec = _elapsed_since(quantile_start)
        treatment_start = perf_counter()
        selected_d, treatment_alpha = _select_controls_treatment_lasso(
            d,
            x,
            random_state=selection_random_state,
            cv=selection_cv,
            max_iter=selection_max_iter,
            coef_tol=selection_coef_tol,
        )
        treatment_selection_sec = _elapsed_since(treatment_start)
    except Exception as exc:  # noqa: BLE001 - selection failures are reported.
        diagnostics_start = perf_counter()
        selected_instruments = np.arange(z_2d.shape[1])
        ps_diagnostics = summarize_post_selection_diagnostics(
            d=d,
            x=x,
            z=z_2d,
            selected_control_indices=None,
            selected_instrument_indices=selected_instruments,
            lasso_cv_folds=selection_cv,
            selection_failed=True,
            selection_method="quantile_specific",
            warning_code="quantile_selection_failed",
        )
        psq_diagnostics = summarize_quantile_post_selection_diagnostics(
            tau=tau,
            n_controls=x.shape[1],
            selected_quantile_y=None,
            selected_treatment_d=None,
            selected_union=None,
            quantile_alpha_selected=None,
            quantile_cv_folds=quantile_selection_cv,
            selection_failed=True,
            warning_code="quantile_selection_failed",
        )
        diagnostics_sec = _elapsed_since(diagnostics_start)
        runtime_seconds = perf_counter() - start
        return _failed_result(
            data=data,
            tau=tau,
            message=f"Quantile-specific control selection failed: {exc}",
            selected_controls=None,
            runtime_seconds=runtime_seconds,
            ps_diagnostics=ps_diagnostics,
            psq_diagnostics=psq_diagnostics,
            runtime_diagnostics=estimator_runtime_columns(
                estimator="post_selection_quantile",
                total_sec=runtime_seconds,
                diagnostics_sec=diagnostics_sec,
                quantile_selection_sec=_elapsed_since(quantile_start),
                treatment_selection_sec=_elapsed_since(treatment_start),
            ),
        )

    selected_indices = np.union1d(quantile_details.selected_indices, selected_d).astype(int)
    selected_instrument_indices = np.arange(z_2d.shape[1])
    diagnostics_start = perf_counter()
    ps_diagnostics = summarize_post_selection_diagnostics(
        d=d,
        x=x,
        z=z_2d,
        selected_control_indices=selected_indices,
        selected_instrument_indices=selected_instrument_indices,
        lasso_alpha_first_stage=treatment_alpha,
        lasso_cv_folds=selection_cv,
        selection_method="quantile_specific",
    )
    psq_diagnostics = summarize_quantile_post_selection_diagnostics(
        tau=tau,
        n_controls=x.shape[1],
        selected_quantile_y=quantile_details.selected_indices,
        selected_treatment_d=selected_d,
        selected_union=selected_indices,
        quantile_alpha_selected=quantile_details.alpha_selected,
        quantile_cv_folds=quantile_selection_cv,
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
                "Quantile post-selection IVQR infeasible: QR design dimension "
                f"is at least sample size (regressors={num_qr_regressors}, "
                f"n={x.shape[0]})."
            ),
            selected_controls=int(selected_indices.size),
            runtime_seconds=runtime_seconds,
            ps_diagnostics=ps_diagnostics,
            psq_diagnostics=psq_diagnostics,
            runtime_diagnostics=estimator_runtime_columns(
                estimator="post_selection_quantile",
                total_sec=runtime_seconds,
                diagnostics_sec=diagnostics_sec,
                quantile_selection_sec=quantile_selection_sec,
                treatment_selection_sec=treatment_selection_sec,
            ),
        )

    if alphas is None:
        alphas = alpha_grid(alpha_min, alpha_max, alpha_step)
    else:
        alphas = validate_alpha_grid(alphas)

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
            message=(
                "All alpha-grid evaluations failed; "
                f"failed_alpha_points={num_failed}/{len(alphas)}; "
                f"selected_y_quantile={quantile_details.selected_indices.size}; "
                f"selected_d={selected_d.size}; selected_union={selected_indices.size}"
            ),
            selected_controls=int(selected_indices.size),
            runtime_seconds=runtime_seconds,
            alpha_grid_size=len(alphas),
            failed_alpha_count=num_failed,
            ps_diagnostics=ps_diagnostics,
            psq_diagnostics=psq_diagnostics,
            runtime_diagnostics=estimator_runtime_columns(
                estimator="post_selection_quantile",
                total_sec=runtime_seconds,
                alpha_loop_sec=alpha_loop_sec,
                diagnostics_sec=diagnostics_sec,
                quantile_selection_sec=quantile_selection_sec,
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
        estimator="post_selection_quantile",
        alpha_hat=alpha_hat,
        alpha_true=data.alpha_true,
        tau=tau,
        converged=True,
        failed=False,
        message=(
            f"ok; failed_alpha_points={num_failed}/{len(alphas)}; "
            f"selected_y_quantile={quantile_details.selected_indices.size}; "
            f"selected_d={selected_d.size}; selected_union={selected_indices.size}"
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
            estimator="post_selection_quantile",
            total_sec=runtime_seconds,
            alpha_loop_sec=alpha_loop_sec,
            confidence_region_sec=confidence_region_sec,
            diagnostics_sec=diagnostics_sec,
            quantile_selection_sec=quantile_selection_sec,
            treatment_selection_sec=treatment_selection_sec,
        ),
        **estimation_result_diagnostic_kwargs(diagnostics),
        **post_selection_result_diagnostic_kwargs(ps_diagnostics),
        **post_selection_quantile_result_diagnostic_kwargs(psq_diagnostics),
    )


__all__ = [
    "PSQ_SELECTION_METHOD",
    "QuantilePostSelectionResult",
    "QuantileSelectionResult",
    "estimate_post_selection_quantile_ivqr",
    "select_controls_quantile_post_selection",
    "select_controls_quantile_y",
    "summarize_quantile_post_selection_diagnostics",
]
