"""Reusable IVQR moment functions.

Final estimator objectives use the covariance-weighted GMM statistic
n * g_hat(a)' * Sigma_hat(a)^(-1) * g_hat(a), with a small ridge for
numerical stability.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    src_path = Path(__file__).resolve().parents[1]
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

import numpy as np

from inference.alpha_grid import alpha_grid


__all__ = [
    "alpha_grid",
    "evaluate_grid",
    "make_instruments",
    "moment_contributions",
    "moment_covariance",
    "quantile_score",
    "residuals_alpha",
    "sample_moment",
    "score_statistic",
    "weighted_gmm_statistic",
]


def quantile_score(residuals: np.ndarray, tau: float) -> np.ndarray:
    """Return the quantile score psi_tau(u) = tau - 1{u <= 0}."""
    if not 0 < tau < 1:
        raise ValueError("tau must satisfy 0 < tau < 1")

    residuals = np.asarray(residuals)
    return tau - (residuals <= 0).astype(float)


def _validate_1d_finite(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_2d_finite(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def residuals_alpha(
    y: np.ndarray,
    d: np.ndarray,
    x_beta: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Construct residuals Y - D alpha - X beta."""
    y = np.asarray(y)
    d = np.asarray(d)
    x_beta = np.asarray(x_beta)

    if not (len(y) == len(d) == len(x_beta)):
        raise ValueError("y, d, and x_beta must have the same length")

    return y - d * alpha - x_beta


def make_instruments(
    z: np.ndarray,
    x_selected: np.ndarray | None = None,
) -> np.ndarray:
    """Build the IVQR instrument matrix Psi = (Z, X_S)."""
    z = np.asarray(z)

    if x_selected is None:
        return z.reshape(-1, 1)

    x_selected = np.asarray(x_selected)
    if x_selected.ndim == 1:
        x_selected = x_selected.reshape(-1, 1)

    if z.shape[0] != x_selected.shape[0]:
        raise ValueError("z and x_selected must have the same number of rows")

    return np.column_stack([z, x_selected])


def sample_moment(
    residuals: np.ndarray,
    tau: float,
    instruments: np.ndarray,
) -> np.ndarray:
    """Return the sample moment vector n^{-1} sum_i psi_tau(r_i) Psi_i."""
    return moment_contributions(residuals, tau, instruments).mean(axis=0)


def moment_contributions(
    residuals: np.ndarray,
    tau: float,
    instruments: np.ndarray,
) -> np.ndarray:
    """Return m_i(a) = psi_tau(residual_i) * Psi_i as an (n, k) matrix."""
    residuals = _validate_1d_finite(residuals, "residuals")
    instruments = _validate_2d_finite(instruments, "instruments")
    if residuals.shape[0] != instruments.shape[0]:
        raise ValueError("residuals and instruments must have the same number of rows")

    scores = quantile_score(residuals, tau)
    contributions = scores[:, None] * instruments
    if not np.all(np.isfinite(contributions)):
        raise ValueError("moment contributions must be finite")
    return contributions


def moment_covariance(
    contributions: np.ndarray,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Estimate centered covariance of moment contributions with ridge."""
    contributions = _validate_2d_finite(contributions, "contributions")
    if contributions.shape[0] < 2:
        raise ValueError("at least two moment contributions are required")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")

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
    contributions = _validate_2d_finite(contributions, "contributions")
    n = contributions.shape[0]
    g_hat = contributions.mean(axis=0)
    sigma = moment_covariance(contributions, ridge=ridge)

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


def score_statistic(moment_vector: np.ndarray) -> float:
    """Unweighted prototype statistic based on g'g.

    This function returns g'g for backward compatibility because it only
    receives the moment vector. For final estimator objectives, use
    weighted_gmm_statistic(), which implements n * g_hat' Sigma_hat^{-1} g_hat.
    """
    moment_vector = np.asarray(moment_vector)
    return float(np.dot(moment_vector, moment_vector))


def evaluate_grid(
    alphas: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    x_beta: np.ndarray,
    tau: float,
    instruments: np.ndarray,
) -> np.ndarray:
    """Evaluate the unweighted IVQR score statistic on an alpha grid."""
    alphas = np.asarray(alphas)
    scores = np.empty(len(alphas), dtype=float)

    for j, alpha in enumerate(alphas):
        residuals = residuals_alpha(y, d, x_beta, float(alpha))
        moment_vector = sample_moment(residuals, tau, instruments)
        scores[j] = score_statistic(moment_vector)

    return scores
