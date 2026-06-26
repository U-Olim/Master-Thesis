"""Production moment helpers for IVQR estimators."""

from __future__ import annotations

import numpy as np

from utils.validation import (
    validate_1d_array,
    validate_2d_array,
    validate_tau,
)


__all__ = [
    "quantile_score",
    "weighted_gmm_statistic",
]


def _validate_nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def quantile_score(residuals: np.ndarray, tau: float) -> np.ndarray:
    """Return the quantile score psi_tau(u) = tau - 1{u <= 0}."""
    tau = validate_tau(tau)
    residuals = validate_1d_array("residuals", residuals)
    if residuals.size == 0:
        raise ValueError("residuals must be nonempty")
    return tau - (residuals <= 0.0).astype(float)


def _moment_covariance(
    contributions: np.ndarray,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Estimate centered covariance of moment contributions with diagonal ridge."""
    contributions = validate_2d_array("contributions", contributions)
    if contributions.shape[0] < 2:
        raise ValueError("at least two moment contributions are required")
    if contributions.shape[1] == 0:
        raise ValueError("moment contributions must have at least one column")

    ridge = _validate_nonnegative_float("ridge", ridge)

    n, k = contributions.shape
    centered = contributions - contributions.mean(axis=0)
    sigma = centered.T @ centered / n
    sigma = sigma + ridge * np.eye(k)
    if not np.all(np.isfinite(sigma)):
        raise ValueError("moment covariance must be finite")
    return sigma


def weighted_gmm_statistic(
    contributions: np.ndarray,
    ridge: float = 1e-8,
    use_pinv: bool = True,
) -> float:
    """Return n * g_hat' * Sigma_hat^{-1} * g_hat."""
    contributions = validate_2d_array("contributions", contributions)
    if contributions.shape[0] < 2:
        raise ValueError("at least two moment contributions are required")
    if contributions.shape[1] == 0:
        raise ValueError("moment contributions must have at least one column")
    ridge = _validate_nonnegative_float("ridge", ridge)
    if not isinstance(use_pinv, bool):
        raise ValueError("use_pinv must be a boolean")

    n = contributions.shape[0]
    g_hat = contributions.mean(axis=0)
    sigma = _moment_covariance(contributions, ridge=ridge)

    if use_pinv:
        statistic = n * g_hat @ np.linalg.pinv(sigma) @ g_hat
    else:
        statistic = n * g_hat @ np.linalg.solve(sigma, g_hat)

    statistic = float(statistic)
    if statistic < 0.0 and statistic > -1e-10:
        statistic = 0.0
    if not np.isfinite(statistic) or statistic < 0.0:
        raise ValueError("weighted GMM statistic must be finite and nonnegative")
    return statistic
