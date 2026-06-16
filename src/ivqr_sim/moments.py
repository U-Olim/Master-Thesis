"""Reusable IVQR moment functions."""

from __future__ import annotations

import numpy as np


def quantile_score(residuals: np.ndarray, tau: float) -> np.ndarray:
    """Return the quantile score psi_tau(u) = tau - 1{u <= 0}."""
    if not 0 < tau < 1:
        raise ValueError("tau must satisfy 0 < tau < 1")

    residuals = np.asarray(residuals)
    return tau - (residuals <= 0).astype(float)


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
    residuals = np.asarray(residuals)
    instruments = np.asarray(instruments)

    if instruments.ndim == 1:
        instruments = instruments.reshape(-1, 1)

    if len(residuals) != instruments.shape[0]:
        raise ValueError("residuals and instruments must have the same number of rows")

    scores = quantile_score(residuals, tau)
    moment = scores[:, None] * instruments
    return moment.mean(axis=0)


def score_statistic(moment_vector: np.ndarray) -> float:
    """Return the unweighted quadratic score based on the Euclidean norm."""
    moment_vector = np.asarray(moment_vector)
    return float(np.dot(moment_vector, moment_vector))


def alpha_grid(
    alpha_min: float,
    alpha_max: float,
    step: float,
) -> np.ndarray:
    """Create an alpha grid with inclusive endpoint when it lies on the grid."""
    if alpha_max <= alpha_min:
        raise ValueError("alpha_max must be greater than alpha_min")
    if step <= 0:
        raise ValueError("step must be positive")

    n_steps = int(np.floor((alpha_max - alpha_min) / step))
    grid = alpha_min + step * np.arange(n_steps + 1)

    if np.isclose(grid[-1], alpha_max):
        grid[-1] = alpha_max
    elif grid[-1] < alpha_max:
        grid = np.append(grid, alpha_max)

    return grid


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
