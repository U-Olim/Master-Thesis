"""Backward-compatible old full-control IVQR API.

New production code separates the full-control benchmark into
estimators.full_control_ivqr and scripts/04_run_full_control_ivqr.py. This module
keeps the previous symbols for tests and older notebooks.
"""

from __future__ import annotations

import numpy as np
from dataclasses import replace
from statsmodels.regression.quantile_regression import QuantReg

from estimators.ch_ivqr_common import (
    AlphaEvaluation,
    add_intercept,
    as_2d_instruments as _as_2d_instruments,
    ch_ivqr_design as _ch_ivqr_design,
    wald_statistic as _wald_statistic,
)
from estimators.full_control_ivqr import estimate_full_control_ivqr
from simulation.config import DEFAULT_QUANTREG_MAX_ITER


def _fit_control_quantile_regression(
    y_alpha: np.ndarray,
    x_design: np.ndarray,
    tau: float,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
) -> tuple[np.ndarray, bool, str]:
    """Fit Q_tau(Y - D alpha | X) with an already-built design matrix."""
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    beta_length = x_design.shape[1]
    try:
        result = QuantReg(y_alpha, x_design).fit(q=tau, max_iter=max_iter)
        beta_hat = np.asarray(result.params, dtype=float)
    except Exception as exc:  # noqa: BLE001
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
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    z_2d = _as_2d_instruments(z)
    design, z_block = _ch_ivqr_design(x_controls, z_2d)
    y_alpha = np.asarray(y, dtype=float) - np.asarray(d, dtype=float) * alpha
    dim_z = z_2d.shape[1]
    try:
        result = QuantReg(y_alpha, design).fit(q=tau, max_iter=max_iter)
        params = np.asarray(result.params, dtype=float)
        cov_params = np.asarray(result.cov_params(), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return AlphaEvaluation(np.inf, np.full(dim_z, np.nan), np.full((dim_z, dim_z), np.nan), dim_z, False, str(exc))
    if params.shape != (design.shape[1],) or cov_params.shape != (design.shape[1], design.shape[1]):
        return AlphaEvaluation(np.inf, np.full(dim_z, np.nan), np.full((dim_z, dim_z), np.nan), dim_z, False, "QuantReg returned invalid shape.")
    gamma_hat = params[z_block]
    cov_gamma = cov_params[z_block, z_block]
    try:
        statistic = _wald_statistic(gamma_hat, cov_gamma)
    except ValueError as exc:
        return AlphaEvaluation(np.inf, np.asarray(gamma_hat, dtype=float), np.atleast_2d(np.asarray(cov_gamma, dtype=float)), dim_z, False, str(exc))
    return AlphaEvaluation(statistic, np.asarray(gamma_hat, dtype=float), np.atleast_2d(np.asarray(cov_gamma, dtype=float)), dim_z, True, "ok")


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


def estimate_full_ivqr(*args, **kwargs):
    """Compatibility alias returning the old estimator name full_ivqr."""
    result = estimate_full_control_ivqr(*args, **kwargs)
    return replace(result, estimator="full_ivqr")


__all__ = [
    "AlphaEvaluation",
    "QuantReg",
    "add_intercept",
    "_as_2d_instruments",
    "_ch_ivqr_design",
    "_evaluate_alpha_ch_ivqr",
    "_evaluate_alpha_full_ivqr",
    "_fit_control_quantile_regression",
    "_wald_statistic",
    "estimate_full_ivqr",
]
