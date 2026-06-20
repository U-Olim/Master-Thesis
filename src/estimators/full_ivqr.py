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
from utils.validation import (
    validate_2d_array,
    validate_alpha_grid,
    validate_data_arrays,
    validate_tau,
)


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Return a design matrix with a leading intercept column."""
    x = validate_2d_array("x", x)

    return np.column_stack([np.ones(x.shape[0]), x])


def _validate_full_control_feasible(n: int, p: int) -> None:
    """Reject only mathematically infeasible full-control QR designs."""
    if p + 1 >= n:
        raise ValueError(
            "Full-control IVQR is infeasible when the number of controls plus intercept "
            "is greater than or equal to the sample size. "
            f"Received n={n}, p={p}, p+1={p + 1}."
        )


def _failed_estimation_result(
    data: SimData,
    tau: float,
    message: str,
    runtime_seconds: float,
    alpha_grid_size: int,
    failed_alpha_count: int,
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


def _fit_control_quantile_regression(
    y_alpha: np.ndarray,
    x_design: np.ndarray,
    tau: float,
    max_iter: int = 1000,
) -> tuple[np.ndarray, bool, str]:
    """Fit Q_tau(Y - D alpha | X) using the intercept-augmented full controls."""
    beta_length = x_design.shape[1]

    try:
        result = QuantReg(y_alpha, x_design).fit(q=tau, max_iter=max_iter)
        beta_hat = np.asarray(result.params, dtype=float)
    except Exception as exc:  # noqa: BLE001 - QuantReg can fail in several ways.
        return np.full(beta_length, np.nan), False, str(exc)

    if beta_hat.shape != (beta_length,):
        return np.full(beta_length, np.nan), False, "QuantReg returned invalid coefficient shape."
    if not np.all(np.isfinite(beta_hat)):
        return beta_hat, False, "QuantReg returned non-finite coefficients."

    return beta_hat, True, "ok"


def _evaluate_alpha_full_ivqr(
    y: np.ndarray,
    d: np.ndarray,
    x_design: np.ndarray,
    instruments: np.ndarray,
    alpha: float,
    tau: float,
    max_iter: int = 1000,
    gmm_ridge: float = 1e-8,
) -> tuple[float, bool, str]:
    """Evaluate the covariance-weighted full-control IVQR objective."""
    beta_hat, converged, message = _fit_control_quantile_regression(
        y_alpha=y - d * alpha,
        x_design=x_design,
        tau=tau,
        max_iter=max_iter,
    )
    if not converged:
        return np.inf, False, message

    x_beta = x_design @ beta_hat
    residuals = residuals_alpha(y, d, x_beta, alpha)
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
    validate_tau(tau)
    y, d, z, x = validate_data_arrays(data.y, data.d, data.x, data.z)

    n, p = x.shape
    _validate_full_control_feasible(n=n, p=p)
    x_design = add_intercept(x)
    instruments = make_instruments(z, x)

    if alphas is None:
        alphas = alpha_grid(alpha_min, alpha_max, alpha_step)
    else:
        alphas = validate_alpha_grid(alphas)

    statistics = np.empty(len(alphas), dtype=float)
    converged_flags: list[bool] = []

    for j, alpha in enumerate(alphas):
        try:
            statistic, converged, _message = _evaluate_alpha_full_ivqr(
                y=y,
                d=d,
                x_design=x_design,
                instruments=instruments,
                alpha=float(alpha),
                tau=tau,
                max_iter=max_iter,
                gmm_ridge=gmm_ridge,
            )
        except Exception:  # noqa: BLE001 - failed grid points are recorded.
            statistic, converged = np.inf, False
        statistics[j] = statistic
        converged_flags.append(converged)

    statistics, num_failed = sanitize_grid_statistics(statistics, converged_flags)
    if num_failed == len(alphas):
        return _failed_estimation_result(
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
