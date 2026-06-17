"""Full-control IVQR estimator."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg

from dgp.designs import SimData
from estimators.base import EstimationResult
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


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Return a design matrix with a leading intercept column."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")

    return np.column_stack([np.ones(x.shape[0]), x])


def _validate_tau(tau: float) -> None:
    if not 0 < tau < 1:
        raise ValueError("tau must satisfy 0 < tau < 1")


def _validate_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_data_arrays(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = _validate_vector(y, "y")
    d = _validate_vector(d, "d")
    z = _validate_vector(z, "z")
    x = np.asarray(x, dtype=float)

    if x.ndim != 2:
        raise ValueError("x must be two-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")
    if not (len(y) == len(d) == len(z) == x.shape[0]):
        raise ValueError("y, d, z, and x must have consistent row counts")

    return y, d, z, x


def _validate_alpha_candidates(alphas: np.ndarray) -> np.ndarray:
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1:
        raise ValueError("alphas must be one-dimensional")
    if alphas.size == 0:
        raise ValueError("alphas must be nonempty")
    if not np.all(np.isfinite(alphas)):
        raise ValueError("alphas must contain only finite values")
    if not np.all(np.diff(alphas) > 0):
        raise ValueError("alphas must be sorted strictly increasing")
    return alphas


def _failed_result(
    data: SimData,
    tau: float,
    message: str,
    runtime_seconds: float,
    alpha_grid_size: int | None = None,
    failed_alpha_count: int | None = None,
) -> EstimationResult:
    return EstimationResult(
        estimator="full_ivqr",
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
        selected_controls=None,
        runtime_seconds=runtime_seconds,
    )


def fit_profile_beta(
    y: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    alpha: float,
    tau: float,
    max_iter: int = 1000,
) -> tuple[np.ndarray, bool, str]:
    """Profile beta(alpha) by quantile regression of y - d alpha on [1, X]."""
    _validate_tau(tau)
    y = _validate_vector(y, "y")
    d = _validate_vector(d, "d")
    x_design = add_intercept(x)

    if len(y) != len(d) or len(y) != x_design.shape[0]:
        raise ValueError("y, d, and x must have consistent row counts")

    y_tilde = y - d * alpha
    beta_length = x_design.shape[1]

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


def evaluate_full_ivqr_alpha(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
    alpha: float,
    tau: float,
    max_iter: int = 1000,
    gmm_ridge: float = 1e-8,
) -> tuple[float, bool, str]:
    """Evaluate the covariance-weighted full-control IVQR objective."""
    y, d, z, x = _validate_data_arrays(y, d, z, x)

    beta_hat, converged, message = fit_profile_beta(
        y=y,
        d=d,
        x=x,
        alpha=alpha,
        tau=tau,
        max_iter=max_iter,
    )
    if not converged:
        return np.inf, False, message

    x_design = add_intercept(x)
    x_beta = x_design @ beta_hat
    residuals = residuals_alpha(y, d, x_beta, alpha)
    instruments = make_instruments(z, x)
    contributions = moment_contributions(residuals, tau, instruments)
    statistic = weighted_gmm_statistic(contributions, ridge=gmm_ridge)

    return float(statistic), True, "ok"


def estimate_full_ivqr(
    data: SimData,
    tau: float,
    alphas: np.ndarray | None = None,
    alpha_min: float = -2.0,
    alpha_max: float = 4.0,
    alpha_step: float = 0.05,
    confidence_level: float = 0.95,
    max_iter: int = 1000,
    gmm_ridge: float = 1e-8,
) -> EstimationResult:
    """Estimate full-control IVQR by weighted GMM over an alpha grid."""
    start = perf_counter()
    _validate_tau(tau)
    y, d, z, x = _validate_data_arrays(data.y, data.d, data.z, data.x)

    n, p = x.shape
    num_profile_params = p + 1
    if num_profile_params >= n:
        return _failed_result(
            data=data,
            tau=tau,
            message=(
                "Full-control IVQR infeasible: number of profiled nuisance "
                "parameters is at least sample size."
            ),
            runtime_seconds=perf_counter() - start,
            alpha_grid_size=None,
            failed_alpha_count=None,
        )

    if alphas is None:
        alphas = alpha_grid(alpha_min, alpha_max, alpha_step)
    else:
        alphas = _validate_alpha_candidates(alphas)

    statistics = np.empty(len(alphas), dtype=float)
    converged_flags: list[bool] = []

    for j, alpha in enumerate(alphas):
        try:
            statistic, converged, message = evaluate_full_ivqr_alpha(
                y=y,
                d=d,
                z=z,
                x=x,
                alpha=float(alpha),
                tau=tau,
                max_iter=max_iter,
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
                f"failed_alpha_points={num_failed}/{len(alphas)}"
            ),
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
    return EstimationResult(
        estimator="full_ivqr",
        alpha_hat=alpha_hat,
        alpha_true=data.alpha_true,
        tau=tau,
        converged=True,
        failed=False,
        message=f"ok; failed_alpha_points={num_failed}/{len(alphas)}",
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
        selected_controls=None,
        runtime_seconds=perf_counter() - start,
    )
