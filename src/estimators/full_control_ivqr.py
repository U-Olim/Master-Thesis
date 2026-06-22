"""Separate full-control IVQR benchmark estimator.

This estimator is intentionally excluded from the main simulation runner. It is a
naive benchmark that uses every observed control in the inverse-IVQR step.
"""

from __future__ import annotations

import numpy as np

from dgp.designs import SimData
from estimators.base import EstimationResult
from estimators.ch_ivqr_common import estimate_ch_ivqr_controls
from simulation.config import DEFAULT_QUANTREG_MAX_ITER
from utils.validation import validate_alpha_grid, validate_data_arrays, validate_tau


def estimate_full_control_ivqr(
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
    """Estimate naive full-control IVQR using all controls in X."""
    del gmm_ridge  # kept for a uniform estimator API.
    validate_tau(tau)
    _y, _d, _z, x = validate_data_arrays(data.y, data.d, data.x, data.z)
    if alphas is not None:
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
        max_iter=max_iter,
        selected_controls=x.shape[1],
    )


__all__ = ["estimate_full_control_ivqr"]
