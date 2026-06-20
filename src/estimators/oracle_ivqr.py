"""Simulation-only oracle IVQR estimator."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from dgp.designs import SimData
from estimators.base import EstimationResult
from estimators.full_ivqr import add_intercept, estimate_full_ivqr
from utils.validation import validate_1d_array, validate_2d_array


def _validate_oracle_indices(oracle_indices: Any, p: int) -> np.ndarray:
    indices = np.asarray(oracle_indices)
    if indices.ndim != 1:
        raise ValueError("oracle_indices must be one-dimensional")
    if indices.size == 0:
        raise ValueError("oracle_indices must be nonempty")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("oracle_indices must contain integers")

    indices = indices.astype(int, copy=False)
    if np.unique(indices).size != indices.size:
        raise ValueError("oracle_indices must not contain duplicates")
    if np.any(indices < 0) or np.any(indices >= p):
        raise ValueError(f"oracle_indices must be between 0 and {p - 1}")
    return np.sort(indices)


def estimate_oracle_ivqr(
    data: SimData,
    tau: float,
    oracle_indices: np.ndarray,
    alphas: np.ndarray | None = None,
    alpha_min: float = -2.0,
    alpha_max: float = 4.0,
    alpha_step: float = 0.05,
    confidence_level: float = 0.95,
    max_iter: int = 1000,
    gmm_ridge: float = 1e-8,
) -> EstimationResult:
    """Estimate infeasible oracle IVQR using the true active controls only."""
    validate_1d_array("y", data.y)
    validate_1d_array("d", data.d)
    validate_1d_array("z", data.z)
    x = validate_2d_array("x", data.x)
    indices = _validate_oracle_indices(oracle_indices, p=x.shape[1])

    oracle_data = SimData(
        y=data.y,
        d=data.d,
        z=data.z,
        x=x[:, indices],
        alpha_true=data.alpha_true,
        u=data.u,
        v=data.v,
    )
    result = estimate_full_ivqr(
        oracle_data,
        tau=tau,
        alphas=alphas,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
        confidence_level=confidence_level,
        max_iter=max_iter,
        gmm_ridge=gmm_ridge,
    )
    return replace(result, estimator="oracle_ivqr", selected_controls=int(indices.size))


__all__ = ["add_intercept", "estimate_oracle_ivqr"]
