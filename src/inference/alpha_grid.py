"""Alpha-grid construction helpers."""

from __future__ import annotations

import numpy as np


DEFAULT_ALPHA_MIN: float = -1.0
DEFAULT_ALPHA_MAX: float = 3.0
DEFAULT_ALPHA_STEP: float = 0.2


__all__ = [
    "DEFAULT_ALPHA_MIN",
    "DEFAULT_ALPHA_MAX",
    "DEFAULT_ALPHA_STEP",
    "alpha_grid",
    "default_alpha_grid",
]


def _validate_grid_scalar(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def alpha_grid(
    alpha_min: float,
    alpha_max: float,
    step: float,
) -> np.ndarray:
    """Create a finite, strictly increasing alpha grid with an inclusive endpoint.

    The grid starts at `alpha_min`, advances by `step`, and always includes
    `alpha_max` as the final candidate. If the step does not divide the interval
    exactly, the last interval is shorter.
    """
    alpha_min = _validate_grid_scalar("alpha_min", alpha_min)
    alpha_max = _validate_grid_scalar("alpha_max", alpha_max)
    step = _validate_grid_scalar("step", step)

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

    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("alpha grid must be nonempty and one-dimensional")
    if not np.all(np.isfinite(grid)):
        raise ValueError("alpha grid must be finite")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("alpha grid must be strictly increasing")

    return grid


def default_alpha_grid() -> np.ndarray:
    """Return the project-default direct-estimator fallback alpha grid."""
    return alpha_grid(DEFAULT_ALPHA_MIN, DEFAULT_ALPHA_MAX, DEFAULT_ALPHA_STEP)
