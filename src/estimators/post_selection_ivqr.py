"""Post-selection IVQR estimator."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dgp.designs import SimData
from estimators.base import EstimationResult
from estimators.ch_ivqr_common import (
    as_2d_instruments,
    evaluate_alpha_ch_ivqr,
)

_evaluate_alpha_ch_ivqr = evaluate_alpha_ch_ivqr
from inference.confidence_regions import (
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    sanitize_grid_statistics,
)
from inference.moments import (
    alpha_grid,
)
from simulation.config import DEFAULT_QUANTREG_MAX_ITER
from utils.validation import (
    validate_alpha_grid,
    validate_data_arrays,
    validate_tau,
)


def _failed_result(
    data: SimData,
    tau: float,
    message: str,
    selected_controls: int | None,
    runtime_seconds: float,
    alpha_grid_size: int | None = None,
    failed_alpha_count: int | None = None,
) -> EstimationResult:
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
    validate_tau(tau)
    y, d, x = validate_data_arrays(y, d, x)

    if cv < 2:
        raise ValueError("cv must be at least 2")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

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

    return selected, message


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
    alpha_min: float = -2.0,
    alpha_max: float = 4.0,
    alpha_step: float = 0.05,
    confidence_level: float = 0.95,
    selection_random_state: int | None = 123,
    selection_cv: int = 5,
    selection_max_iter: int = 10000,
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
) -> EstimationResult:
    """Estimate post-selection IVQR by Lasso selection and weighted GMM."""
    start = perf_counter()
    validate_tau(tau)
    if quantreg_max_iter <= 0:
        raise ValueError("quantreg_max_iter must be positive")
    y, d, z, x = validate_data_arrays(data.y, data.d, data.x, data.z)
    z_2d = as_2d_instruments(z)

    try:
        selected_indices, selection_message = select_controls_lasso(
            y=y,
            d=d,
            x=x,
            tau=tau,
            random_state=selection_random_state,
            cv=selection_cv,
            max_iter=selection_max_iter,
        )
    except Exception as exc:  # noqa: BLE001 - selection failures should be reported cleanly.
        return _failed_result(
            data=data,
            tau=tau,
            message=f"Control selection failed: {exc}",
            selected_controls=None,
            runtime_seconds=perf_counter() - start,
            alpha_grid_size=None,
            failed_alpha_count=None,
        )

    n = x.shape[0]
    if selected_indices.size == 0:
        x_selected = np.empty((n, 0))
    else:
        x_selected = x[:, selected_indices]

    if selected_indices.size + 1 >= n:
        return _failed_result(
            data=data,
            tau=tau,
            message="Post-selection IVQR infeasible: selected nuisance dimension is at least sample size.",
            selected_controls=int(selected_indices.size),
            runtime_seconds=perf_counter() - start,
            alpha_grid_size=None,
            failed_alpha_count=None,
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
        )

    alpha_hat, min_statistic, at_boundary = argmin_grid(alphas, statistics)
    critical = critical_value_chi_square(confidence_level, df=z_2d.shape[1])
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
        cr_lower=region.lower,
        cr_upper=region.upper,
        cr_length=region.length,
        cr_covers_true=region.covers_true,
        cr_empty=region.empty,
        cr_disconnected=region.disconnected,
        selected_controls=int(selected_indices.size),
        runtime_seconds=perf_counter() - start,
    )
