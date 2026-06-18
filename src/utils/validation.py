"""Reusable validation helpers for arrays and simulation parameters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

import numpy as np


def validate_tau(tau: float) -> float:
    """Validate and return a quantile level in the open unit interval."""
    tau = float(tau)
    if not 0 < tau < 1:
        raise ValueError("tau must satisfy 0 < tau < 1")
    return tau


def check_finite(name: str, array: np.ndarray) -> np.ndarray:
    """Validate that an array contains only finite values."""
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_1d_array(
    name: str,
    value: Any,
    length: int | None = None,
) -> np.ndarray:
    """Return a finite one-dimensional float array."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    check_finite(name, array)
    if length is not None and len(array) != length:
        raise ValueError(f"{name} must have length {length}")
    return array


def validate_2d_array(
    name: str,
    value: Any,
    n_rows: int | None = None,
) -> np.ndarray:
    """Return a finite two-dimensional float array."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    check_finite(name, array)
    if n_rows is not None and array.shape[0] != n_rows:
        raise ValueError(f"{name} must have {n_rows} rows")
    return array


def validate_alpha_grid(alpha_candidates: Any) -> np.ndarray:
    """Return a finite, nonempty, strictly increasing alpha grid."""
    alphas = validate_1d_array("alphas", alpha_candidates)
    if alphas.size == 0:
        raise ValueError("alphas must be nonempty")
    if not np.all(np.diff(alphas) > 0):
        raise ValueError("alphas must be sorted strictly increasing")
    return alphas


@overload
def validate_data_arrays(
    y: Any,
    d: Any,
    x: Any,
    z: None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def validate_data_arrays(
    y: Any,
    d: Any,
    x: Any,
    z: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def validate_data_arrays(
    y: Any,
    d: Any,
    x: Any,
    z: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Validate outcome, treatment, controls, and optionally instrument arrays."""
    y_array = validate_1d_array("y", y)
    d_array = validate_1d_array("d", d)
    x_array = validate_2d_array("x", x)

    if z is None:
        if not (len(y_array) == len(d_array) == x_array.shape[0]):
            raise ValueError("y, d, and x must have consistent row counts")
        return y_array, d_array, x_array

    z_array = validate_1d_array("z", z)
    if not (len(y_array) == len(d_array) == len(z_array) == x_array.shape[0]):
        raise ValueError("y, d, z, and x must have consistent row counts")
    return y_array, d_array, z_array, x_array


def validate_positive_int(name: str, value: int) -> int:
    """Validate and return a positive integer."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_k_folds(k_folds: int, n: int) -> int:
    """Validate K-fold count for a sample of size n."""
    if n <= 1:
        raise ValueError("n must be greater than 1")
    if k_folds < 2 or k_folds > n:
        raise ValueError("k_folds must satisfy 2 <= k_folds <= n")
    return k_folds


def validate_nonempty_sequence(name: str, value: Sequence[Any]) -> Sequence[Any]:
    """Validate that a sequence is nonempty."""
    if len(value) == 0:
        raise ValueError(f"{name} must be nonempty")
    return value
