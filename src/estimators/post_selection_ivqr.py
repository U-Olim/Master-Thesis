"""Post-selection IVQR estimator."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.quantile_regression import QuantReg

from dgp.designs import SimData
from estimators.base import EstimationResult
from estimators.full_ivqr import add_intercept
from inference.confidence_regions import (
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    sanitize_grid_statistics,
)
from inference.moments import (
    alpha_grid,
    make_instruments,
    moment_contributions,
    residuals_alpha,
    weighted_gmm_statistic,
)
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


def fit_post_selection_beta(
    y: np.ndarray,
    d: np.ndarray,
    x_selected: np.ndarray,
    alpha: float,
    tau: float,
    max_iter: int = 1000,
) -> tuple[np.ndarray, bool, str]:
    """Profile selected-control beta(alpha) with intercept by QuantReg."""
    validate_tau(tau)
    y, d, x_selected = validate_data_arrays(y, d, x_selected)
    x_design = add_intercept(x_selected)
    beta_length = x_design.shape[1]

    if beta_length >= len(y):
        return (
            np.full(beta_length, np.nan),
            False,
            "Post-selection IVQR infeasible: selected nuisance dimension is at least sample size.",
        )

    y_tilde = y - d * alpha

    try:
        result = QuantReg(y_tilde, x_design).fit(q=tau, max_iter=max_iter)
        beta_hat = np.asarray(result.params, dtype=float)
    except Exception as exc:  # noqa: BLE001 - QuantReg can fail in several ways.
        return np.full(beta_length, np.nan), False, str(exc)

    if beta_hat.shape != (beta_length,):
        return np.full(beta_length, np.nan), False, "QuantReg returned invalid coefficient shape."
    if not np.all(np.isfinite(beta_hat)):
        return beta_hat, False, "QuantReg returned non-finite coefficients."

    return beta_hat, True, "ok"


def evaluate_post_selection_alpha(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    x_selected: np.ndarray,
    alpha: float,
    tau: float,
    max_iter: int = 1000,
    gmm_ridge: float = 1e-8,
) -> tuple[float, bool, str]:
    """Evaluate the covariance-weighted post-selection IVQR objective."""
    y, d, z, x_selected = validate_data_arrays(y, d, x_selected, z)
    beta_hat, converged, message = fit_post_selection_beta(
        y=y,
        d=d,
        x_selected=x_selected,
        alpha=alpha,
        tau=tau,
        max_iter=max_iter,
    )
    if not converged:
        return np.inf, False, message

    x_design = add_intercept(x_selected)
    x_beta = x_design @ beta_hat
    residuals = residuals_alpha(y, d, x_beta, alpha)
    instruments = make_instruments(z, x_selected)
    contributions = moment_contributions(residuals, tau, instruments)
    statistic = weighted_gmm_statistic(contributions, ridge=gmm_ridge)

    return float(statistic), True, "ok"


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
    quantreg_max_iter: int = 1000,
    gmm_ridge: float = 1e-8,
) -> EstimationResult:
    """Estimate post-selection IVQR by Lasso selection and weighted GMM."""
    start = perf_counter()
    validate_tau(tau)
    y, d, z, x = validate_data_arrays(data.y, data.d, data.x, data.z)

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
                gmm_ridge=gmm_ridge,
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
    critical = critical_value_chi_square(confidence_level, df=1)
    region = invert_score_test(
        alphas=alphas,
        statistics=statistics,
        critical_value=critical,
        alpha_true=data.alpha_true,
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
