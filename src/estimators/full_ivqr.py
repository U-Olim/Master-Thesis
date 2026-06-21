"""Full-control IVQR estimator."""

from __future__ import annotations

from dataclasses import dataclass
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
)
from simulation.config import DEFAULT_QUANTREG_MAX_ITER
from utils.validation import (
    validate_1d_array,
    validate_2d_array,
    validate_alpha_grid,
    validate_data_arrays,
    validate_tau,
)


@dataclass(frozen=True)
class AlphaEvaluation:
    """CH-IVQR Wald evaluation for one structural alpha candidate."""

    statistic: float
    gamma_hat: np.ndarray
    cov_gamma: np.ndarray
    dim_z: int
    converged: bool
    message: str


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Return a design matrix with a leading intercept column."""
    x = validate_2d_array("x", x)

    return np.column_stack([np.ones(x.shape[0]), x])


def _as_2d_instruments(z: np.ndarray) -> np.ndarray:
    z_array = np.asarray(z, dtype=float)
    if z_array.ndim == 1:
        z_array = z_array.reshape(-1, 1)
    if z_array.ndim != 2:
        raise ValueError("z must be one- or two-dimensional")
    if z_array.shape[1] == 0:
        raise ValueError("z must contain at least one excluded instrument")
    if not np.all(np.isfinite(z_array)):
        raise ValueError("z must contain only finite values")
    return z_array


def _ch_ivqr_design(x_controls: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, slice]:
    x_controls = validate_2d_array("x_controls", x_controls)
    z_2d = _as_2d_instruments(z)
    if x_controls.shape[0] != z_2d.shape[0]:
        raise ValueError("x_controls and z must have the same number of rows")

    design = np.column_stack([np.ones(x_controls.shape[0]), x_controls, z_2d])
    z_start = 1 + x_controls.shape[1]
    z_stop = z_start + z_2d.shape[1]
    return design, slice(z_start, z_stop)


def _wald_statistic(gamma_hat: np.ndarray, cov_gamma: np.ndarray) -> float:
    gamma_hat = np.asarray(gamma_hat, dtype=float).reshape(-1)
    cov_gamma = np.atleast_2d(np.asarray(cov_gamma, dtype=float))
    if cov_gamma.shape != (gamma_hat.size, gamma_hat.size):
        raise ValueError("cov_gamma shape must match gamma_hat dimension")
    if not np.all(np.isfinite(gamma_hat)) or not np.all(np.isfinite(cov_gamma)):
        raise ValueError("gamma_hat and cov_gamma must be finite")

    statistic = float(gamma_hat @ np.linalg.pinv(cov_gamma) @ gamma_hat)
    if statistic < 0.0 and statistic > -1e-10:
        statistic = 0.0
    if not np.isfinite(statistic) or statistic < 0.0:
        raise ValueError("Wald statistic must be finite and nonnegative")
    return statistic


def _validate_full_control_feasible(n: int, p_controls: int, dim_z: int) -> None:
    """Reject only mathematically infeasible full-control QR designs."""
    n_regressors = 1 + p_controls + dim_z
    if n_regressors >= n:
        raise ValueError(
            "Full-control IVQR is infeasible when intercept + controls + instruments "
            "is greater than or equal to sample size. "
            f"Received n={n}, p={p_controls}, dim_z={dim_z}, "
            f"regressors={n_regressors}."
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
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
) -> tuple[np.ndarray, bool, str]:
    """Fit Q_tau(Y - D alpha | X) using the intercept-augmented full controls."""
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

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


def _evaluate_alpha_ch_ivqr(
    *,
    y: np.ndarray,
    d: np.ndarray,
    x_controls: np.ndarray,
    z: np.ndarray,
    alpha: float,
    tau: float,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
) -> AlphaEvaluation:
    """Evaluate CH-IVQR by testing the excluded-instrument coefficient.

    CH-IVQR profiles controls by running QR of Y-D*alpha on controls and
    excluded instruments. The test statistic is the Wald statistic for the
    excluded-instrument coefficient gamma(alpha)=0. Controls are not part of
    the tested moment vector.
    """
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    validate_tau(tau)
    y = validate_1d_array("y", y)
    d = validate_1d_array("d", d)
    x_controls = validate_2d_array("x_controls", x_controls)
    z_2d = _as_2d_instruments(z)
    if not (len(y) == len(d) == x_controls.shape[0] == z_2d.shape[0]):
        raise ValueError("y, d, x_controls, and z must have consistent row counts")
    design, z_block = _ch_ivqr_design(x_controls, z_2d)
    y_alpha = y - d * alpha

    try:
        result = QuantReg(y_alpha, design).fit(q=tau, max_iter=max_iter)
        params = np.asarray(result.params, dtype=float)
        cov_params = np.asarray(result.cov_params(), dtype=float)
    except Exception as exc:  # noqa: BLE001 - QuantReg can fail in several ways.
        dim_z = z_2d.shape[1]
        return AlphaEvaluation(
            statistic=np.inf,
            gamma_hat=np.full(dim_z, np.nan),
            cov_gamma=np.full((dim_z, dim_z), np.nan),
            dim_z=dim_z,
            converged=False,
            message=str(exc),
        )

    dim_z = z_2d.shape[1]
    if params.shape != (design.shape[1],):
        return AlphaEvaluation(
            statistic=np.inf,
            gamma_hat=np.full(dim_z, np.nan),
            cov_gamma=np.full((dim_z, dim_z), np.nan),
            dim_z=dim_z,
            converged=False,
            message="QuantReg returned invalid coefficient shape.",
        )
    if cov_params.shape != (design.shape[1], design.shape[1]):
        return AlphaEvaluation(
            statistic=np.inf,
            gamma_hat=np.full(dim_z, np.nan),
            cov_gamma=np.full((dim_z, dim_z), np.nan),
            dim_z=dim_z,
            converged=False,
            message="QuantReg returned invalid covariance shape.",
        )

    gamma_hat = params[z_block]
    cov_gamma = cov_params[z_block, z_block]
    try:
        # statsmodels cov_params() is the covariance estimate for gamma_hat
        # itself, so the Wald statistic is gamma' cov(gamma)^(-1) gamma.
        # Multiplying by n again would double-count sample-size scaling.
        statistic = _wald_statistic(gamma_hat, cov_gamma)
    except ValueError as exc:
        return AlphaEvaluation(
            statistic=np.inf,
            gamma_hat=np.asarray(gamma_hat, dtype=float),
            cov_gamma=np.atleast_2d(np.asarray(cov_gamma, dtype=float)),
            dim_z=dim_z,
            converged=False,
            message=str(exc),
        )

    return AlphaEvaluation(
        statistic=statistic,
        gamma_hat=np.asarray(gamma_hat, dtype=float),
        cov_gamma=np.atleast_2d(np.asarray(cov_gamma, dtype=float)),
        dim_z=dim_z,
        converged=True,
        message="ok",
    )


def _evaluate_alpha_full_ivqr(
    *,
    y: np.ndarray,
    d: np.ndarray,
    x_controls: np.ndarray,
    z: np.ndarray,
    alpha: float,
    tau: float,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
) -> tuple[float, bool, str]:
    """Backward-compatible tuple wrapper around the CH-IVQR evaluator."""
    evaluation = _evaluate_alpha_ch_ivqr(
        y=y,
        d=d,
        x_controls=x_controls,
        z=z,
        alpha=alpha,
        tau=tau,
        max_iter=max_iter,
    )
    return evaluation.statistic, evaluation.converged, evaluation.message


def estimate_full_ivqr(
    data: SimData,
    tau: float,
    alphas: np.ndarray | None = None,
    alpha_min: float = -2.0,
    alpha_max: float = 4.0,
    alpha_step: float = 0.05,
    confidence_level: float = 0.95,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    gmm_ridge: float = 1e-8,
) -> EstimationResult:
    """Estimate full-control IVQR by weighted GMM over an alpha grid."""
    start = perf_counter()
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    validate_tau(tau)
    y, d, z, x = validate_data_arrays(data.y, data.d, data.x, data.z)

    n, p = x.shape
    z_2d = _as_2d_instruments(z)
    _validate_full_control_feasible(n=n, p_controls=p, dim_z=z_2d.shape[1])

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
                x_controls=x,
                z=z_2d,
                alpha=float(alpha),
                tau=tau,
                max_iter=max_iter,
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
    critical = critical_value_chi_square(confidence_level, df=z_2d.shape[1])
    region = invert_score_test(
        alphas=alphas,
        statistics=statistics,
        critical_value=critical,
        alpha_true=data.alpha_true,
        statistic_reference=0.0,
        inversion_type="absolute",
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
