"""Naive full-control IVQR benchmark estimator.

This estimator applies the Chernozhukov-Hansen inverse-IVQR procedure
using every observed control column in X. It is included as a benchmark
against oracle, post-selection, and DML-style residualized IVQR estimators.

Because it uses all controls without selection or regularization, it can be
computationally slow, numerically unstable, or infeasible when the number of
controls is large relative to the sample size.
"""

from __future__ import annotations

import numpy as np

from dgp.designs import SimData
from estimators.base import EstimationResult
from estimators.ch_inverse_ivqr import estimate_ch_ivqr_controls
from inference.confidence_regions import validate_critical_value_multiplier
from inference.alpha_grid import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_ALPHA_STEP,
)
from simulation.config import DEFAULT_QUANTREG_MAX_ITER
from utils.validation import validate_alpha_grid, validate_data_arrays, validate_tau


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _validate_alpha_grid_bounds(
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
) -> tuple[float, float, float]:
    if (
        isinstance(alpha_min, bool)
        or isinstance(alpha_max, bool)
        or isinstance(alpha_step, bool)
    ):
        raise ValueError("alpha grid bounds must be finite numeric values")
    alpha_min = float(alpha_min)
    alpha_max = float(alpha_max)
    alpha_step = float(alpha_step)
    if not (
        np.isfinite(alpha_min)
        and np.isfinite(alpha_max)
        and np.isfinite(alpha_step)
    ):
        raise ValueError("alpha grid bounds must be finite")
    if alpha_max <= alpha_min:
        raise ValueError("alpha_max must be greater than alpha_min")
    if alpha_step <= 0:
        raise ValueError("alpha_step must be positive")
    return alpha_min, alpha_max, alpha_step


def _validate_confidence_level(confidence_level: float) -> float:
    if isinstance(confidence_level, bool):
        raise ValueError(
            "confidence_level must satisfy 0 < confidence_level < 1"
        )
    confidence_level = float(confidence_level)
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must satisfy 0 < confidence_level < 1"
        )
    return confidence_level


def estimate_full_control_ivqr(
    data: SimData,
    tau: float,
    alphas: np.ndarray | None = None,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    alpha_step: float = DEFAULT_ALPHA_STEP,
    confidence_level: float = 0.95,
    critical_value_multiplier: float = 1.0,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    gmm_ridge: float = 1e-8,
) -> EstimationResult:
    """Estimate naive full-control IVQR using all columns of X.

    The estimator delegates to `estimate_ch_ivqr_controls` with
    `x_controls=data.x` and `selected_controls=p`.

    The `gmm_ridge` argument is accepted only for compatibility with the common
    estimator API. It is validated but not used by the CH inverse-IVQR Wald
    statistic, which relies on the quantile-regression covariance matrix.
    """
    tau = validate_tau(tau)
    _y, _d, _z, x = validate_data_arrays(data.y, data.d, data.x, data.z)
    max_iter = _validate_positive_int("max_iter", max_iter)
    gmm_ridge = _validate_nonnegative_float("gmm_ridge", gmm_ridge)
    confidence_level = _validate_confidence_level(confidence_level)
    critical_value_multiplier = validate_critical_value_multiplier(
        critical_value_multiplier
    )
    _ = gmm_ridge  # compatibility only; CH inverse-IVQR does not use this ridge.
    if alphas is None:
        alpha_min, alpha_max, alpha_step = _validate_alpha_grid_bounds(
            alpha_min,
            alpha_max,
            alpha_step,
        )
    else:
        alphas = validate_alpha_grid(alphas)
    return estimate_ch_ivqr_controls(
        data=data,
        tau=tau,
        x_controls=x,
        estimator_name="full_control_ivqr",
        alphas=alphas,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
        confidence_level=confidence_level,
        critical_value_multiplier=critical_value_multiplier,
        max_iter=max_iter,
        selected_controls=x.shape[1],
    )


__all__ = ["estimate_full_control_ivqr"]
