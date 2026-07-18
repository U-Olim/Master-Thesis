"""Post-selection control-selected IVQR estimator.

This estimator first selects controls by LassoCV from the reduced-form
relations Y ~ X and D ~ X, then applies the Chernozhukov-Hansen
inverse-IVQR estimator using only the selected controls.

The selected control set is

    S_hat = S_y union S_d

and the second-stage structural equation is evaluated as

    Y = D alpha_tau + X_{S_hat}' beta_tau + U_tau

The estimator is feasible under approximate sparsity, but selection
uncertainty can cause undercoverage in finite samples. It is not the
DML-style residualized IVQR estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dgp.designs import SimData
from estimators.base import (
    EstimationResult,
    estimation_result_diagnostic_kwargs,
    post_selection_result_diagnostic_kwargs,
)
from ivqr.ch_inverse import (
    AlphaEvaluation,
    IterationWarningPolicy,
    as_2d_instruments,
    evaluate_alpha_ch_ivqr as _evaluate_alpha_ch_ivqr,
    validate_iteration_warning_policy,
)
from ivqr.alpha_grid import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_ALPHA_STEP,
    alpha_grid,
)
from ivqr.confidence_regions import (
    adjust_critical_value,
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    merge_region_and_grid_diagnostics,
    sanitize_grid_statistics,
    summarize_alpha_grid_diagnostics,
    validate_critical_value_multiplier,
)
from simulation.config import DEFAULT_QUANTREG_MAX_ITER
from simulation.results import empty_post_selection_diagnostics
from utils.timing import RuntimeDiagnosticColumns, estimator_runtime_columns
from utils.validation import (
    validate_1d_array,
    validate_2d_array,
    validate_alpha_grid,
    validate_data_arrays,
    validate_tau,
)


POST_SELECTION_WARNING_NONE = ""
RANK_DEFICIENT_CONDITION_THRESHOLD = 1e12


@dataclass(frozen=True)
class SelectionResult:
    """Control-selection details from the existing LassoCV selection step."""

    selected_indices: np.ndarray
    message: str
    lasso_alpha_controls: float | None
    lasso_alpha_first_stage: float | None
    lasso_alpha_y_cv: float | None
    lasso_alpha_d_cv: float | None
    lasso_alpha_y_final: float | None
    lasso_alpha_d_final: float | None


def _nan_or_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _elapsed_since(start: float | None) -> float:
    """Return elapsed seconds since start, or NaN if the timer was not started."""
    if start is None:
        return float("nan")
    return perf_counter() - start


def _as_selected_indices(
    selected_indices: np.ndarray | list[int] | tuple[int, ...] | None,
    total: int,
    name: str,
) -> np.ndarray:
    if selected_indices is None:
        return np.empty(0, dtype=int)
    selected = np.asarray(selected_indices)
    if selected.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if selected.size == 0:
        return np.empty(0, dtype=int)
    if not np.issubdtype(selected.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    selected = np.unique(selected.astype(int, copy=False))
    if np.any(selected < 0) or np.any(selected >= total):
        raise ValueError(f"{name} contains out-of-range indices")
    return selected


def _as_candidate_instruments(z: np.ndarray) -> np.ndarray:
    z_array = np.asarray(z, dtype=float)
    if z_array.ndim == 1:
        z_array = z_array.reshape(-1, 1)
    if z_array.ndim != 2:
        raise ValueError("z must be one- or two-dimensional")
    if z_array.shape[0] == 0:
        raise ValueError("z must contain at least one row")
    if not np.all(np.isfinite(z_array)):
        raise ValueError("z must contain only finite values")
    return z_array


def _ols_rss_and_r2(y: np.ndarray, design: np.ndarray) -> tuple[float, float, int]:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    rss = float(residuals @ residuals)
    centered = y - float(np.mean(y))
    tss = float(centered @ centered)
    r2 = float(1.0 - rss / tss) if tss > 0 else float("nan")
    rank = int(np.linalg.matrix_rank(design))
    return rss, r2, rank


def summarize_post_selection_diagnostics(
    *,
    d: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    selected_control_indices: np.ndarray | list[int] | tuple[int, ...] | None,
    selected_instrument_indices: np.ndarray | list[int] | tuple[int, ...] | None,
    lasso_alpha_controls: float | None = None,
    lasso_alpha_instruments: float | None = None,
    lasso_alpha_first_stage: float | None = None,
    selection_lasso_multiplier: float | None = 1.0,
    lasso_alpha_y_cv: float | None = None,
    lasso_alpha_d_cv: float | None = None,
    lasso_alpha_y_final: float | None = None,
    lasso_alpha_d_final: float | None = None,
    lasso_cv_folds: int | None = None,
    selection_failed: bool = False,
    selection_method: str = "lassocv_control_union",
    warning_code: str | None = None,
) -> dict[str, Any]:
    """Summarize diagnostics for the current post-selection implementation.

    The current post-selection IVQR estimator performs control selection and
    retains all excluded instruments. Instrument diagnostics therefore report
    retained instruments, not Lasso-selected instruments.
    """
    d = validate_1d_array("d", d)
    x = validate_2d_array("x", x)
    z_2d = _as_candidate_instruments(z)
    if not (len(d) == x.shape[0] == z_2d.shape[0]):
        raise ValueError("d, x, and z must have consistent row counts")

    selected_controls = _as_selected_indices(
        selected_control_indices,
        x.shape[1],
        "selected_control_indices",
    )
    retained_instruments = _as_selected_indices(
        selected_instrument_indices,
        z_2d.shape[1],
        "selected_instrument_indices",
    )

    n_controls = int(selected_controls.size)
    n_candidate_instruments = int(z_2d.shape[1])
    n_retained_instruments = int(retained_instruments.size)
    share_retained_instruments = (
        n_retained_instruments / n_candidate_instruments
        if n_candidate_instruments > 0
        else np.nan
    )
    all_instruments_retained = (
        n_candidate_instruments > 0
        and n_retained_instruments == n_candidate_instruments
    )
    n_total = n_controls + n_retained_instruments
    diagnostics = empty_post_selection_diagnostics()
    diagnostics.update(
        {
            "ps_n_selected_controls": n_controls,
            # Backward compatibility: the current estimator does not perform
            # instrument selection. These selected-instrument fields mirror
            # retained instruments. Use ps_n_retained_instruments and
            # ps_instrument_selection_method for interpretation.
            "ps_n_selected_instruments": n_retained_instruments,
            "ps_n_selected_total": n_total,
            "ps_share_selected_controls": (
                n_controls / x.shape[1] if x.shape[1] > 0 else np.nan
            ),
            "ps_share_selected_instruments": share_retained_instruments,
            "ps_instrument_selection_method": "all_instruments_retained",
            "ps_n_candidate_instruments": n_candidate_instruments,
            "ps_n_retained_instruments": n_retained_instruments,
            "ps_share_retained_instruments": share_retained_instruments,
            "ps_all_instruments_retained": all_instruments_retained,
            "ps_selected_no_controls": n_controls == 0,
            "ps_selected_no_instruments": n_retained_instruments == 0,
            "ps_selected_empty_total": n_total == 0,
            "ps_selection_method": selection_method,
            "ps_selection_lasso_multiplier": _nan_or_float(selection_lasso_multiplier),
            "ps_lasso_alpha_controls": _nan_or_float(lasso_alpha_controls),
            "ps_lasso_alpha_instruments": _nan_or_float(lasso_alpha_instruments),
            "ps_lasso_alpha_first_stage": _nan_or_float(lasso_alpha_first_stage),
            "ps_lasso_alpha_y_cv": _nan_or_float(lasso_alpha_y_cv),
            "ps_lasso_alpha_d_cv": _nan_or_float(lasso_alpha_d_cv),
            "ps_lasso_alpha_y_final": _nan_or_float(lasso_alpha_y_final),
            "ps_lasso_alpha_d_final": _nan_or_float(lasso_alpha_d_final),
            "ps_lasso_cv_folds": lasso_cv_folds,
            "ps_selection_failed": bool(selection_failed),
        }
    )

    warning = warning_code or POST_SELECTION_WARNING_NONE
    if selection_failed:
        diagnostics["ps_warning_code"] = warning or "lasso_failed"
        return diagnostics

    selected_x = x[:, selected_controls] if n_controls else np.empty((len(d), 0))
    selected_z = (
        z_2d[:, retained_instruments]
        if n_retained_instruments
        else np.empty((len(d), 0))
    )
    restricted_design = np.column_stack([np.ones(len(d)), selected_x])
    full_design = np.column_stack([restricted_design, selected_z])

    try:
        condition_number = float(np.linalg.cond(full_design))
    except Exception:
        condition_number = float("nan")
        warning = warning or "condition_number_failed"
    diagnostics["ps_first_stage_condition_number"] = condition_number

    try:
        full_rss, full_r2, full_rank = _ols_rss_and_r2(d, full_design)
        diagnostics["ps_first_stage_r2"] = full_r2
        n_obs = len(d)
        full_params = full_design.shape[1]
        df_full = n_obs - full_params
        diagnostics["ps_first_stage_adj_r2"] = (
            1.0 - (1.0 - full_r2) * (n_obs - 1) / df_full
            if df_full > 0 and np.isfinite(full_r2)
            else np.nan
        )

        rank_deficient = full_rank < full_params or (
            np.isfinite(condition_number)
            and condition_number >= RANK_DEFICIENT_CONDITION_THRESHOLD
        )
        diagnostics["ps_rank_deficient"] = bool(rank_deficient)
        if rank_deficient:
            warning = warning or "rank_deficient"

        if n_retained_instruments == 0:
            diagnostics["ps_first_stage_partial_r2"] = np.nan
            diagnostics["ps_first_stage_f_stat"] = np.nan
            warning = warning or "empty_instruments"
        else:
            restricted_rss, _restricted_r2, _restricted_rank = _ols_rss_and_r2(
                d,
                restricted_design,
            )
            diagnostics["ps_first_stage_partial_r2"] = (
                (restricted_rss - full_rss) / restricted_rss
                if restricted_rss > 0
                else np.nan
            )
            q = n_retained_instruments
            diagnostics["ps_first_stage_f_stat"] = (
                ((restricted_rss - full_rss) / q) / (full_rss / df_full)
                if q > 0 and df_full > 0 and full_rss > 0
                else np.nan
            )
    except Exception:
        diagnostics["ps_first_stage_failed"] = True
        warning = warning or "first_stage_failed"

    diagnostics["ps_warning_code"] = warning
    return diagnostics


def _validate_selection_config(cv: int, max_iter: int, n: int) -> None:
    if not isinstance(cv, int) or isinstance(cv, bool):
        raise ValueError("cv must be an integer")
    if cv < 2 or cv > n:
        raise ValueError("cv must satisfy 2 <= cv <= n")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool):
        raise ValueError("max_iter must be an integer")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")


def validate_selection_lasso_multiplier(selection_lasso_multiplier: float) -> float:
    if isinstance(selection_lasso_multiplier, bool):
        raise ValueError("selection_lasso_multiplier must be positive.")
    selection_lasso_multiplier = float(selection_lasso_multiplier)
    if not np.isfinite(selection_lasso_multiplier) or selection_lasso_multiplier <= 0:
        raise ValueError("selection_lasso_multiplier must be positive.")
    return selection_lasso_multiplier


def _failed_result(
    data: SimData,
    tau: float,
    message: str,
    selected_controls: int | None,
    runtime_seconds: float,
    alpha_grid_size: int | None = None,
    failed_alpha_count: int | None = None,
    ps_diagnostics: dict[str, Any] | None = None,
    runtime_diagnostics: RuntimeDiagnosticColumns | None = None,
) -> EstimationResult:
    if runtime_seconds < 0:
        raise ValueError("runtime_seconds must be nonnegative")
    if alpha_grid_size is not None and alpha_grid_size < 1:
        raise ValueError("alpha_grid_size must be at least 1")
    if failed_alpha_count is not None and failed_alpha_count < 0:
        raise ValueError("failed_alpha_count must be nonnegative")
    if (
        alpha_grid_size is not None
        and failed_alpha_count is not None
        and failed_alpha_count > alpha_grid_size
    ):
        raise ValueError("failed_alpha_count cannot exceed alpha_grid_size")
    return EstimationResult(
        estimator="post_selection_ivqr",
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
                estimator="post_selection_ivqr",
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
    )


def _select_controls_lasso_details(
    y: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    tau: float,
    random_state: int | None = None,
    cv: int = 5,
    max_iter: int = 10000,
    selection_lasso_multiplier: float = 1.0,
) -> SelectionResult:
    """Select controls by union of LassoCV selections for Y~X and D~X.

    The selection step is not quantile-specific; tau is validated for API
    consistency with IVQR estimators.
    """
    validate_tau(tau)
    y, d, x = validate_data_arrays(y, d, x)
    n = x.shape[0]
    if n < 2:
        raise ValueError("at least two observations are required")
    _validate_selection_config(cv, max_iter, n)
    selection_lasso_multiplier = validate_selection_lasso_multiplier(
        selection_lasso_multiplier
    )

    if x.shape[1] == 0:
        return SelectionResult(
            selected_indices=np.empty(0, dtype=int),
            message="selected_y=0; selected_d=0; selected_union=0",
            lasso_alpha_controls=None,
            lasso_alpha_first_stage=None,
            lasso_alpha_y_cv=None,
            lasso_alpha_d_cv=None,
            lasso_alpha_y_final=None,
            lasso_alpha_d_final=None,
        )

    cv_model_y = make_pipeline(
        StandardScaler(),
        LassoCV(cv=cv, random_state=random_state, max_iter=max_iter),
    )
    cv_model_d = make_pipeline(
        StandardScaler(),
        LassoCV(cv=cv, random_state=random_state, max_iter=max_iter),
    )

    cv_model_y.fit(x, y)
    cv_model_d.fit(x, d)

    alpha_y_cv = float(cv_model_y.named_steps["lassocv"].alpha_)
    alpha_d_cv = float(cv_model_d.named_steps["lassocv"].alpha_)
    alpha_y_final = selection_lasso_multiplier * alpha_y_cv
    alpha_d_final = selection_lasso_multiplier * alpha_d_cv

    final_model_y = make_pipeline(
        StandardScaler(),
        Lasso(alpha=alpha_y_final, max_iter=max_iter),
    )
    final_model_d = make_pipeline(
        StandardScaler(),
        Lasso(alpha=alpha_d_final, max_iter=max_iter),
    )
    final_model_y.fit(x, y)
    final_model_d.fit(x, d)

    coef_y = np.asarray(final_model_y.named_steps["lasso"].coef_)
    coef_d = np.asarray(final_model_d.named_steps["lasso"].coef_)
    selected_y = np.flatnonzero(np.abs(coef_y) > 1e-12)
    selected_d = np.flatnonzero(np.abs(coef_d) > 1e-12)
    selected = np.union1d(selected_y, selected_d).astype(int)
    message = (
        f"selected_y={selected_y.size}; selected_d={selected_d.size}; "
        f"selected_union={selected.size}; "
        f"selection_lasso_multiplier={selection_lasso_multiplier:g}"
    )

    return SelectionResult(
        selected_indices=selected,
        message=message,
        lasso_alpha_controls=alpha_y_final,
        lasso_alpha_first_stage=alpha_d_final,
        lasso_alpha_y_cv=alpha_y_cv,
        lasso_alpha_d_cv=alpha_d_cv,
        lasso_alpha_y_final=alpha_y_final,
        lasso_alpha_d_final=alpha_d_final,
    )


def select_controls_lasso(
    y: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    tau: float,
    random_state: int | None = None,
    cv: int = 5,
    max_iter: int = 10000,
    selection_lasso_multiplier: float = 1.0,
) -> tuple[np.ndarray, str]:
    """Select controls by union of LassoCV selections for Y~X and D~X."""
    details = _select_controls_lasso_details(
        y=y,
        d=d,
        x=x,
        tau=tau,
        random_state=random_state,
        cv=cv,
        max_iter=max_iter,
        selection_lasso_multiplier=selection_lasso_multiplier,
    )
    return details.selected_indices, details.message


def evaluate_post_selection_alpha(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    x_selected: np.ndarray,
    alpha: float,
    tau: float,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    iteration_warning_policy: IterationWarningPolicy = "use_if_valid",
) -> AlphaEvaluation:
    """Evaluate post-selection CH-IVQR by testing gamma_Z(alpha)=0.

    The production default uses valid iteration-warning fits; ``"reject"``
    remains available to reproduce the legacy rejection behavior.
    """
    y, d, z, x_selected = validate_data_arrays(y, d, x_selected, z)
    evaluation = _evaluate_alpha_ch_ivqr(
        y=y,
        d=d,
        z=z,
        x_controls=x_selected,
        alpha=alpha,
        tau=tau,
        max_iter=max_iter,
        iteration_warning_policy=iteration_warning_policy,
    )
    return evaluation


def estimate_post_selection_ivqr(
    data: SimData,
    tau: float,
    alphas: np.ndarray | None = None,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    alpha_step: float = DEFAULT_ALPHA_STEP,
    confidence_level: float = 0.95,
    critical_value_multiplier: float = 1.0,
    selection_random_state: int | None = 123,
    selection_cv: int = 5,
    selection_max_iter: int = 10000,
    selection_lasso_multiplier: float = 1.0,
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    iteration_warning_policy: IterationWarningPolicy = "use_if_valid",
) -> EstimationResult:
    """Estimate post-selection IVQR by Lasso selection and CH inverse-IVQR.

    Valid iteration-warning fits are used by default.  Pass ``"reject"`` only
    when reproducing simulations generated with the legacy warning policy.
    """
    start = perf_counter()
    selection_sec = float("nan")
    diagnostics_sec = float("nan")
    alpha_loop_sec = float("nan")
    validate_tau(tau)
    critical_value_multiplier = validate_critical_value_multiplier(
        critical_value_multiplier
    )
    selection_lasso_multiplier = validate_selection_lasso_multiplier(
        selection_lasso_multiplier
    )
    if quantreg_max_iter <= 0:
        raise ValueError("quantreg_max_iter must be positive")
    iteration_warning_policy = validate_iteration_warning_policy(
        iteration_warning_policy
    )
    y, d, z, x = validate_data_arrays(data.y, data.d, data.x, data.z)
    z_2d = as_2d_instruments(z)
    _validate_selection_config(selection_cv, selection_max_iter, x.shape[0])

    selection_start: float | None = None
    diagnostics_start: float | None = None
    try:
        selection_start = perf_counter()
        selection_details = _select_controls_lasso_details(
            y=y,
            d=d,
            x=x,
            tau=tau,
            random_state=selection_random_state,
            cv=selection_cv,
            max_iter=selection_max_iter,
            selection_lasso_multiplier=selection_lasso_multiplier,
        )
        selection_sec = _elapsed_since(selection_start)
    except Exception as exc:  # noqa: BLE001 - selection failures should be reported cleanly.
        diagnostics_start = perf_counter()
        ps_diagnostics = summarize_post_selection_diagnostics(
            d=d,
            x=x,
            z=z_2d,
            selected_control_indices=None,
            selected_instrument_indices=np.arange(z_2d.shape[1]),
            lasso_cv_folds=selection_cv,
            selection_lasso_multiplier=selection_lasso_multiplier,
            selection_failed=True,
            warning_code="lasso_failed",
        )
        diagnostics_sec = _elapsed_since(diagnostics_start)
        runtime_seconds = perf_counter() - start
        return _failed_result(
            data=data,
            tau=tau,
            message=f"Control selection failed: {exc}",
            selected_controls=None,
            runtime_seconds=runtime_seconds,
            alpha_grid_size=None,
            failed_alpha_count=None,
            ps_diagnostics=ps_diagnostics,
            runtime_diagnostics=estimator_runtime_columns(
                estimator="post_selection_ivqr",
                total_sec=runtime_seconds,
                selection_sec=_elapsed_since(selection_start),
                diagnostics_sec=diagnostics_sec,
            ),
        )

    selected_indices = selection_details.selected_indices
    selection_message = selection_details.message
    selected_instrument_indices = np.arange(z_2d.shape[1])
    diagnostics_start = perf_counter()
    ps_diagnostics = summarize_post_selection_diagnostics(
        d=d,
        x=x,
        z=z_2d,
        selected_control_indices=selected_indices,
        selected_instrument_indices=selected_instrument_indices,
        lasso_alpha_controls=selection_details.lasso_alpha_controls,
        lasso_alpha_first_stage=selection_details.lasso_alpha_first_stage,
        selection_lasso_multiplier=selection_lasso_multiplier,
        lasso_alpha_y_cv=selection_details.lasso_alpha_y_cv,
        lasso_alpha_d_cv=selection_details.lasso_alpha_d_cv,
        lasso_alpha_y_final=selection_details.lasso_alpha_y_final,
        lasso_alpha_d_final=selection_details.lasso_alpha_d_final,
        lasso_cv_folds=selection_cv,
    )
    diagnostics_sec = _elapsed_since(diagnostics_start)

    n = x.shape[0]
    if selected_indices.size == 0:
        x_selected = np.empty((n, 0))
    else:
        x_selected = x[:, selected_indices]

    num_qr_regressors = 1 + int(selected_indices.size) + int(z_2d.shape[1])
    if num_qr_regressors >= n:
        runtime_seconds = perf_counter() - start
        return _failed_result(
            data=data,
            tau=tau,
            message=(
                "Post-selection IVQR infeasible: QR design dimension is at least "
                f"sample size (regressors={num_qr_regressors}, n={n})."
            ),
            selected_controls=int(selected_indices.size),
            runtime_seconds=runtime_seconds,
            alpha_grid_size=None,
            failed_alpha_count=None,
            ps_diagnostics=ps_diagnostics,
            runtime_diagnostics=estimator_runtime_columns(
                estimator="post_selection_ivqr",
                total_sec=runtime_seconds,
                selection_sec=selection_sec,
                diagnostics_sec=diagnostics_sec,
            ),
        )

    if alphas is None:
        alphas = alpha_grid(alpha_min, alpha_max, alpha_step)
    else:
        alphas = validate_alpha_grid(alphas)

    statistics = np.empty(len(alphas), dtype=float)
    usable_flags: list[bool] = []

    alpha_loop_start = perf_counter()
    for j, alpha in enumerate(alphas):
        try:
            evaluation = evaluate_post_selection_alpha(
                y=y,
                d=d,
                z=z,
                x_selected=x_selected,
                alpha=float(alpha),
                tau=tau,
                max_iter=quantreg_max_iter,
                iteration_warning_policy=iteration_warning_policy,
            )
        except Exception:  # noqa: BLE001 - failed grid points are recorded.
            statistics[j] = np.inf
            usable_flags.append(False)
        else:
            statistics[j] = evaluation.statistic
            usable_flags.append(evaluation.usable)
    alpha_loop_sec = perf_counter() - alpha_loop_start

    statistics, num_failed = sanitize_grid_statistics(statistics, usable_flags)
    if num_failed == len(alphas):
        runtime_seconds = perf_counter() - start
        return _failed_result(
            data=data,
            tau=tau,
            message=(
                "All alpha-grid evaluations failed; "
                f"failed_alpha_points={num_failed}/{len(alphas)}; {selection_message}"
            ),
            selected_controls=int(selected_indices.size),
            runtime_seconds=runtime_seconds,
            alpha_grid_size=len(alphas),
            failed_alpha_count=num_failed,
            ps_diagnostics=ps_diagnostics,
            runtime_diagnostics=estimator_runtime_columns(
                estimator="post_selection_ivqr",
                total_sec=runtime_seconds,
                selection_sec=selection_sec,
                diagnostics_sec=diagnostics_sec,
                alpha_loop_sec=alpha_loop_sec,
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

    message = f"ok; failed_alpha_points={num_failed}/{len(alphas)}; {selection_message}"
    runtime_seconds = perf_counter() - start

    return EstimationResult(
        estimator="post_selection_ivqr",
        alpha_hat=alpha_hat,
        alpha_true=data.alpha_true,
        tau=tau,
        converged=True,
        failed=False,
        message=message,
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
        # First-stage diagnostics are computed inside the combined
        # post-selection diagnostics helper, so first-stage timing is not
        # cleanly separable and is intentionally reported as NaN.
        **estimator_runtime_columns(
            estimator="post_selection_ivqr",
            total_sec=runtime_seconds,
            selection_sec=selection_sec,
            diagnostics_sec=diagnostics_sec,
            alpha_loop_sec=alpha_loop_sec,
            confidence_region_sec=confidence_region_sec,
        ),
        **estimation_result_diagnostic_kwargs(diagnostics),
        **post_selection_result_diagnostic_kwargs(ps_diagnostics),
    )


__all__ = [
    "select_controls_lasso",
    "summarize_post_selection_diagnostics",
    "empty_post_selection_diagnostics",
    "evaluate_post_selection_alpha",
    "estimate_post_selection_ivqr",
]
