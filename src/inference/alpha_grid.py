"""Alpha-grid construction helpers."""

from __future__ import annotations

import numpy as np


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
