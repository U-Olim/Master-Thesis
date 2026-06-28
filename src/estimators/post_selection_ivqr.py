"""Post-selection control-selected IVQR estimator.

This estimator first selects controls by LassoCV from the reduced-form
relations Y ~ X and D ~ X, then applies the Chernozhukov-Hansen
inverse-IVQR estimator using only the selected controls.

It is a feasible Monte Carlo benchmark. It is not the DML-style IVQR estimator
and does not implement the full orthogonal double-selection IV procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dgp.designs import SimData
from estimators.base import (
    EstimationResult,
    estimation_result_diagnostic_kwargs,
    post_selection_result_diagnostic_kwargs,
)
from estimators.ch_inverse_ivqr import (
    as_2d_instruments,
    evaluate_alpha_ch_ivqr as _evaluate_alpha_ch_ivqr,
)
from inference.alpha_grid import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_ALPHA_STEP,
    alpha_grid,
)
from inference.confidence_regions import (
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    sanitize_grid_statistics,
    summarize_alpha_grid_diagnostics,
)
from simulation.config import DEFAULT_QUANTREG_MAX_ITER
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


def empty_post_selection_diagnostics() -> dict[str, Any]:
    """Return neutral post-selection diagnostics for non-post-selection rows."""
    return {
        "ps_n_selected_controls": None,
        "ps_n_selected_instruments": None,
        "ps_n_selected_total": None,
        "ps_share_selected_controls": None,
        "ps_share_selected_instruments": None,
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


def _nan_or_float(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


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
    lasso_cv_folds: int | None = None,
    selection_failed: bool = False,
    selection_method: str = "lassocv_control_union",
    warning_code: str | None = None,
) -> dict[str, Any]:
    """Summarize diagnostics for the current post-selection implementation.

    The estimator selects controls from X by unioning LassoCV fits for Y~X and
    D~X. It does not select excluded instruments; selected instruments here
    are therefore the retained excluded instruments used by the IVQR step.
    """
    d = validate_1d_array("d", d)
    x = validate_2d_array("x", x)
    z_2d = as_2d_instruments(z)
    if not (len(d) == x.shape[0] == z_2d.shape[0]):
        raise ValueError("d, x, and z must have consistent row counts")

    selected_controls = _as_selected_indices(
        selected_control_indices,
        x.shape[1],
        "selected_control_indices",
    )
    selected_instruments = _as_selected_indices(
        selected_instrument_indices,
        z_2d.shape[1],
        "selected_instrument_indices",
    )

    n_controls = int(selected_controls.size)
    n_instruments = int(selected_instruments.size)
    n_total = n_controls + n_instruments
    diagnostics = empty_post_selection_diagnostics()
    diagnostics.update(
        {
            "ps_n_selected_controls": n_controls,
            "ps_n_selected_instruments": n_instruments,
            "ps_n_selected_total": n_total,
            "ps_share_selected_controls": (
                n_controls / x.shape[1] if x.shape[1] > 0 else np.nan
            ),
            "ps_share_selected_instruments": (
                n_instruments / z_2d.shape[1] if z_2d.shape[1] > 0 else np.nan
            ),
            "ps_selected_no_controls": n_controls == 0,
            "ps_selected_no_instruments": n_instruments == 0,
            "ps_selected_empty_total": n_total == 0,
            "ps_selection_method": selection_method,
            "ps_lasso_alpha_controls": _nan_or_float(lasso_alpha_controls),
            "ps_lasso_alpha_instruments": _nan_or_float(lasso_alpha_instruments),
            "ps_lasso_alpha_first_stage": _nan_or_float(lasso_alpha_first_stage),
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
        z_2d[:, selected_instruments] if n_instruments else np.empty((len(d), 0))
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

        if n_instruments == 0:
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
            q = n_instruments
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


def _failed_result(
    data: SimData,
    tau: float,
    message: str,
    selected_controls: int | None,
    runtime_seconds: float,
    alpha_grid_size: int | None = None,
    failed_alpha_count: int | None = None,
    ps_diagnostics: dict[str, Any] | None = None,
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

    if x.shape[1] == 0:
        return SelectionResult(
            selected_indices=np.empty(0, dtype=int),
            message="selected_y=0; selected_d=0; selected_union=0",
            lasso_alpha_controls=None,
            lasso_alpha_first_stage=None,
        )

    model_y = make_pipeline(
        StandardScaler(),
        LassoCV(cv=cv, random_state=random_state, max_iter=max_iter),
    )
    model_d = make_pipeline(
        StandardScaler(),
        LassoCV(cv=cv, random_state=random_state, max_iter=max_iter),
    )

    model_y.fit(x, y)
    model_d.fit(x, d)

    coef_y = np.asarray(model_y.named_steps["lassocv"].coef_)
    coef_d = np.asarray(model_d.named_steps["lassocv"].coef_)
    selected_y = np.flatnonzero(np.abs(coef_y) > 1e-12)
    selected_d = np.flatnonzero(np.abs(coef_d) > 1e-12)
    selected = np.union1d(selected_y, selected_d).astype(int)
    message = (
        f"selected_y={selected_y.size}; selected_d={selected_d.size}; "
        f"selected_union={selected.size}"
    )

    return SelectionResult(
        selected_indices=selected,
        message=message,
        lasso_alpha_controls=float(model_y.named_steps["lassocv"].alpha_),
        lasso_alpha_first_stage=float(model_d.named_steps["lassocv"].alpha_),
    )


def select_controls_lasso(
    y: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    tau: float,
    random_state: int | None = None,
    cv: int = 5,
    max_iter: int = 10000,
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
) -> tuple[float, bool, str]:
    """Evaluate post-selection CH-IVQR by testing gamma_Z(alpha)=0."""
    y, d, z, x_selected = validate_data_arrays(y, d, x_selected, z)
    evaluation = _evaluate_alpha_ch_ivqr(
        y=y,
        d=d,
        z=z,
        x_controls=x_selected,
        alpha=alpha,
        tau=tau,
        max_iter=max_iter,
    )
    return evaluation.statistic, evaluation.converged, evaluation.message


def estimate_post_selection_ivqr(
    data: SimData,
    tau: float,
    alphas: np.ndarray | None = None,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    alpha_step: float = DEFAULT_ALPHA_STEP,
    confidence_level: float = 0.95,
    selection_random_state: int | None = 123,
    selection_cv: int = 5,
    selection_max_iter: int = 10000,
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
) -> EstimationResult:
    """Estimate post-selection IVQR by Lasso control selection and CH inverse-IVQR."""
    start = perf_counter()
    validate_tau(tau)
    if quantreg_max_iter <= 0:
        raise ValueError("quantreg_max_iter must be positive")
    y, d, z, x = validate_data_arrays(data.y, data.d, data.x, data.z)
    z_2d = as_2d_instruments(z)
    _validate_selection_config(selection_cv, selection_max_iter, x.shape[0])

    try:
        selection_details = _select_controls_lasso_details(
            y=y,
            d=d,
            x=x,
            tau=tau,
            random_state=selection_random_state,
            cv=selection_cv,
            max_iter=selection_max_iter,
        )
    except Exception as exc:  # noqa: BLE001 - selection failures should be reported cleanly.
        ps_diagnostics = summarize_post_selection_diagnostics(
            d=d,
            x=x,
            z=z_2d,
            selected_control_indices=None,
            selected_instrument_indices=np.arange(z_2d.shape[1]),
            lasso_cv_folds=selection_cv,
            selection_failed=True,
            warning_code="lasso_failed",
        )
        return _failed_result(
            data=data,
            tau=tau,
            message=f"Control selection failed: {exc}",
            selected_controls=None,
            runtime_seconds=perf_counter() - start,
            alpha_grid_size=None,
            failed_alpha_count=None,
            ps_diagnostics=ps_diagnostics,
        )

    selected_indices = selection_details.selected_indices
    selection_message = selection_details.message
    selected_instrument_indices = np.arange(z_2d.shape[1])
    ps_diagnostics = summarize_post_selection_diagnostics(
        d=d,
        x=x,
        z=z_2d,
        selected_control_indices=selected_indices,
        selected_instrument_indices=selected_instrument_indices,
        lasso_alpha_controls=selection_details.lasso_alpha_controls,
        lasso_alpha_first_stage=selection_details.lasso_alpha_first_stage,
        lasso_cv_folds=selection_cv,
    )

    n = x.shape[0]
    if selected_indices.size == 0:
        x_selected = np.empty((n, 0))
    else:
        x_selected = x[:, selected_indices]

    num_qr_regressors = 1 + int(selected_indices.size) + int(z_2d.shape[1])
    if num_qr_regressors >= n:
        return _failed_result(
            data=data,
            tau=tau,
            message=(
                "Post-selection IVQR infeasible: QR design dimension is at least "
                f"sample size (regressors={num_qr_regressors}, n={n})."
            ),
            selected_controls=int(selected_indices.size),
            runtime_seconds=perf_counter() - start,
            alpha_grid_size=None,
            failed_alpha_count=None,
            ps_diagnostics=ps_diagnostics,
        )

    if alphas is None:
        alphas = alpha_grid(alpha_min, alpha_max, alpha_step)
    else:
        alphas = validate_alpha_grid(alphas)

    statistics = np.empty(len(alphas), dtype=float)
    converged_flags: list[bool] = []

    for j, alpha in enumerate(alphas):
        try:
            statistic, converged, message = evaluate_post_selection_alpha(
                y=y,
                d=d,
                z=z,
                x_selected=x_selected,
                alpha=float(alpha),
                tau=tau,
                max_iter=quantreg_max_iter,
            )
        except Exception as exc:  # noqa: BLE001 - failed grid points are recorded.
            statistic, converged, message = np.inf, False, str(exc)
        statistics[j] = statistic
        converged_flags.append(converged)

    statistics, num_failed = sanitize_grid_statistics(statistics, converged_flags)
    if num_failed == len(alphas):
        return _failed_result(
            data=data,
            tau=tau,
            message=(
                "All alpha-grid evaluations failed; "
                f"failed_alpha_points={num_failed}/{len(alphas)}; {selection_message}"
            ),
            selected_controls=int(selected_indices.size),
            runtime_seconds=perf_counter() - start,
            alpha_grid_size=len(alphas),
            failed_alpha_count=num_failed,
            ps_diagnostics=ps_diagnostics,
        )

    alpha_hat, min_statistic, at_boundary = argmin_grid(alphas, statistics)
    critical = critical_value_chi_square(confidence_level, df=z_2d.shape[1])
    accepted_mask = statistics <= critical
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=alphas,
        accepted_mask=accepted_mask,
        alpha_hat=alpha_hat,
        failed_alpha_count=num_failed,
        test_stats=statistics,
        critical_value=critical,
    )
    region = invert_score_test(
        alphas=alphas,
        statistics=statistics,
        critical_value=critical,
        alpha_true=data.alpha_true,
        statistic_reference=0.0,
        inversion_type="absolute",
    )

    message = f"ok; failed_alpha_points={num_failed}/{len(alphas)}; {selection_message}"

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
        runtime_seconds=perf_counter() - start,
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
