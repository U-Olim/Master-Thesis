"""Reusable IVQR and DML-IVQR moment helpers.

The module provides low-level functions for quantile scores, structural
residuals, moment contributions, sample moments, covariance estimates, and
GMM-style score statistics.

It does not estimate IVQR parameters by itself. Production estimators combine
these helpers with estimator-specific nuisance fitting, alpha-grid search, and
confidence-region inversion.
"""

from __future__ import annotations

import numpy as np

from utils.validation import (
    validate_1d_array,
    validate_2d_array,
    validate_alpha_grid,
    validate_tau,
)


__all__ = [
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


def _validate_finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _as_2d_columns(
    name: str,
    value: np.ndarray,
    n_rows: int | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)

    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional")

    if array.shape[0] == 0:
        raise ValueError(f"{name} must have at least one row")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one column")
    if n_rows is not None and array.shape[0] != n_rows:
        raise ValueError(f"{name} must have {n_rows} rows")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array


def quantile_score(residuals: np.ndarray, tau: float) -> np.ndarray:
    """Return the quantile score psi_tau(u) = tau - 1{u <= 0}."""
    tau = validate_tau(tau)
    residuals = validate_1d_array("residuals", residuals)
    if residuals.size == 0:
        raise ValueError("residuals must be nonempty")
    return tau - (residuals <= 0.0).astype(float)


def residuals_alpha(
    y: np.ndarray,
    d: np.ndarray,
    x_beta: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Construct structural residuals Y - D * alpha - X beta."""
    y = validate_1d_array("y", y)
    if y.size == 0:
        raise ValueError("y must be nonempty")
    d = validate_1d_array("d", d, length=len(y))
    x_beta = validate_1d_array("x_beta", x_beta, length=len(y))
    alpha = _validate_finite_scalar("alpha", alpha)

    residuals = y - d * alpha - x_beta
    if not np.all(np.isfinite(residuals)):
        raise ValueError("residuals must be finite")
    return residuals


def make_instruments(
    z: np.ndarray,
    x_selected: np.ndarray | None = None,
) -> np.ndarray:
    """Build the IVQR instrument matrix Psi = (Z, X_S)."""
    z_2d = _as_2d_columns("z", z)

    if x_selected is None:
        return z_2d.copy()

    x_selected_array = np.asarray(x_selected, dtype=float)
    if x_selected_array.ndim == 2 and x_selected_array.shape[1] == 0:
        if x_selected_array.shape[0] != z_2d.shape[0]:
            raise ValueError("x_selected must have matching rows")
        return z_2d.copy()

    x_selected_2d = _as_2d_columns(
        "x_selected",
        x_selected_array,
        n_rows=z_2d.shape[0],
    )

    return np.column_stack([z_2d, x_selected_2d])


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
    residuals = validate_1d_array("residuals", residuals)
    instruments = validate_2d_array("instruments", instruments)
    if instruments.shape[0] == 0:
        raise ValueError("instruments must have at least one row")
    if instruments.shape[1] == 0:
        raise ValueError("instruments must have at least one column")
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
    """Return the legacy unweighted diagnostic statistic g'g.

    This helper is retained for simple diagnostic tests and backward compatibility.
    Production estimators should use `weighted_gmm_statistic`, which implements
    n * g_hat' Sigma_hat^{-1} g_hat.
    """
    moment_vector = validate_1d_array("moment_vector", moment_vector)
    statistic = float(moment_vector @ moment_vector)
    if not np.isfinite(statistic) or statistic < 0.0:
        raise ValueError("score statistic must be finite and nonnegative")
    return statistic


def evaluate_grid(
    alphas: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    x_beta: np.ndarray,
    tau: float,
    instruments: np.ndarray,
) -> np.ndarray:
    """Evaluate the legacy unweighted IVQR diagnostic statistic on an alpha grid.

    This helper computes g(alpha)'g(alpha) for each candidate alpha. It is useful
    for simple diagnostics and tests, but production estimators should use
    covariance-weighted GMM or CH inverse-IVQR objectives.
    """
    alphas = validate_alpha_grid(alphas)
    y = validate_1d_array("y", y)
    d = validate_1d_array("d", d, length=len(y))
    x_beta = validate_1d_array("x_beta", x_beta, length=len(y))
    instruments = validate_2d_array("instruments", instruments, n_rows=len(y))
    if instruments.shape[1] == 0:
        raise ValueError("instruments must have at least one column")
    tau = validate_tau(tau)

    scores = np.empty(len(alphas), dtype=float)

    for j, alpha in enumerate(alphas):
        residuals = residuals_alpha(y, d, x_beta, float(alpha))
        moment_vector = sample_moment(residuals, tau, instruments)
        scores[j] = score_statistic(moment_vector)

    if scores.shape != alphas.shape:
        raise ValueError("grid scores must match alpha grid shape")
    if not np.all(np.isfinite(scores)):
        raise ValueError("grid scores must be finite")

    return scores
