"""Small shared helpers for quantile-L1 selection experiments."""

from __future__ import annotations

import numpy as np


def build_alpha_anchors(alpha_min: float, alpha_max: float) -> np.ndarray:
    """Return data-independent quartile anchors for an alpha search interval."""
    if isinstance(alpha_min, bool) or isinstance(alpha_max, bool):
        raise ValueError("alpha_min and alpha_max must be finite")
    alpha_min = float(alpha_min)
    alpha_max = float(alpha_max)
    if not np.isfinite(alpha_min) or not np.isfinite(alpha_max):
        raise ValueError("alpha_min and alpha_max must be finite")
    if alpha_max <= alpha_min:
        raise ValueError("alpha_max must exceed alpha_min")
    width = alpha_max - alpha_min
    return np.array(
        [
            alpha_min + 0.25 * width,
            alpha_min + 0.50 * width,
            alpha_min + 0.75 * width,
        ],
        dtype=float,
    )


def union_selected_indices(
    selections: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    total: int,
) -> np.ndarray:
    """Return sorted unique selected indices from several one-dimensional arrays."""
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        raise ValueError("total must be a nonnegative integer")
    if not selections:
        return np.empty(0, dtype=int)

    normalized: list[np.ndarray] = []
    for selected in selections:
        values = np.asarray(selected)
        if values.ndim != 1:
            raise ValueError("selected indices must be one-dimensional")
        if values.size == 0:
            continue
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("selected indices must contain integers")
        values = values.astype(int, copy=False)
        if np.any(values < 0) or np.any(values >= total):
            raise ValueError("selected indices contain out-of-range values")
        normalized.append(values)
    if not normalized:
        return np.empty(0, dtype=int)
    return np.unique(np.concatenate(normalized)).astype(int)


def format_float_sequence(values: np.ndarray | tuple[float, ...] | list[float]) -> str:
    """Return compact semicolon-separated floats for CSV diagnostics."""
    return ";".join(f"{float(value):g}" for value in values)


__all__ = [
    "build_alpha_anchors",
    "format_float_sequence",
    "union_selected_indices",
]
