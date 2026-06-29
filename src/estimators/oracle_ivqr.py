"""Simulation-only oracle IVQR estimator."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from dgp.designs import SimData
from estimators.base import EstimationResult
from estimators.ch_inverse_ivqr import estimate_ch_ivqr_controls
from utils.validation import validate_1d_array, validate_2d_array, validate_data_arrays


_CH_COMPATIBILITY_KWARGS = {
    "alpha_min",
    "alpha_max",
    "alpha_step",
    "confidence_level",
    "critical_value_multiplier",
    "max_iter",
}


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
    y: SimData | np.ndarray,
    d: np.ndarray | None = None,
    x: np.ndarray | None = None,
    z: np.ndarray | None = None,
    tau: float | None = None,
    alpha_candidates: Sequence[float] | np.ndarray | None = None,
    oracle_indices: Sequence[int] | np.ndarray | None = None,
    alpha_true: float | None = None,
    gmm_ridge: float | None = None,
    **kwargs: Any,
) -> EstimationResult:
    """Estimate infeasible oracle IVQR using the true active controls only."""
    if alpha_candidates is None and "alphas" in kwargs:
        alpha_candidates = kwargs.pop("alphas")

    # CH inverse-IVQR does not use GMM ridge regularization.
    # This argument is accepted only for simulation-runner API compatibility.
    _ = gmm_ridge

    unknown = sorted(set(kwargs) - _CH_COMPATIBILITY_KWARGS)
    if unknown:
        names = ", ".join(unknown)
        raise TypeError(f"Unknown oracle IVQR keyword argument(s): {names}")

    if isinstance(y, SimData):
        if tau is None:
            raise ValueError("tau is required")
        if oracle_indices is None:
            raise ValueError("oracle_indices is required")
        data = y
        validate_1d_array("y", data.y)
        validate_1d_array("d", data.d)
        validate_1d_array("z", data.z)
        x_validated = validate_2d_array("x", data.x)
    else:
        if d is None or x is None or z is None:
            raise ValueError("d, x, and z are required when y is an array")
        if tau is None:
            raise ValueError("tau is required")
        if oracle_indices is None:
            raise ValueError("oracle_indices is required")
        y_validated, d_validated, z_validated, x_validated = validate_data_arrays(
            y, d, x, z
        )
        data = SimData(
            y=y_validated,
            d=d_validated,
            z=z_validated,
            x=x_validated,
            alpha_true=alpha_true,
        )

    indices = _validate_oracle_indices(oracle_indices, p=x_validated.shape[1])

    oracle_data = SimData(
        y=data.y,
        d=data.d,
        z=data.z,
        x=x_validated[:, indices],
        alpha_true=data.alpha_true,
        u=data.u,
        v=data.v,
    )
    alphas = (
        None
        if alpha_candidates is None
        else validate_1d_array("alphas", alpha_candidates)
    )
    result = estimate_ch_ivqr_controls(
        oracle_data,
        tau=tau,
        x_controls=oracle_data.x,
        estimator_name="oracle",
        alphas=alphas,
        selected_controls=int(indices.size),
        **kwargs,
    )
    return replace(result, estimator="oracle", selected_controls=int(indices.size))


__all__ = ["estimate_oracle_ivqr"]
