"""Inference routines for IVQR simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import chi2


FAILED_ALPHA_STATISTIC = 1e12


__all__ = [
    "FAILED_ALPHA_STATISTIC",
    "ConfidenceRegion",
    "invert_score_test",
    "is_disconnected_region",
    "critical_value_chi_square",
    "sanitize_grid_statistics",
    "argmin_grid",
    "summarize_region",
]


@dataclass(frozen=True)
class ConfidenceRegion:
    """Grid-based confidence region from score-test inversion."""

    lower: float | None
    upper: float | None
    length: float
    hull_length: float
    blocks: tuple[tuple[float, float], ...]
    accepted_alphas: tuple[float, ...]
    n_blocks: int
    empty: bool
    disconnected: bool
    covers_true: bool | None
    selected_grid: np.ndarray
    critical_value: float
    statistic_reference: float

    @property
    def region_length(self) -> float:
        """Total accepted-region length across all connected blocks."""
        return self.length

    @property
    def is_empty(self) -> bool:
        """Backward-compatible explicit empty-region diagnostic."""
        return self.empty

    @property
    def is_disconnected(self) -> bool:
        """Backward-compatible explicit disconnected-region diagnostic."""
        return self.disconnected


def _as_1d_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_strictly_increasing(alphas: np.ndarray) -> None:
    if not np.all(np.diff(alphas) > 0):
        raise ValueError("alpha grid must be sorted strictly increasing")


def _validate_grid_and_statistics(
    alphas: np.ndarray,
    statistics: np.ndarray,
    *,
    sort_alphas: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    alphas = _as_1d_array(alphas, "alpha grid")
    statistics = _as_1d_array(statistics, "statistics")

    if alphas.size != statistics.size:
        raise ValueError("alpha grid and statistics must have equal length")

    if sort_alphas:
        order = np.argsort(alphas)
        alphas = alphas[order]
        statistics = statistics[order]

    _validate_strictly_increasing(alphas)
    return alphas, statistics


def _validate_critical_value(critical_value: float) -> float:
    if isinstance(critical_value, bool):
        raise ValueError("critical value must be positive and finite")
    critical_value = float(critical_value)
    if not np.isfinite(critical_value) or critical_value <= 0:
        raise ValueError("critical value must be positive and finite")
    return critical_value


def _validate_statistic_reference(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("statistic_reference must be finite when provided")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("statistic_reference must be finite when provided")
    return value


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied


def _accepted_blocks(
    alpha_candidates: np.ndarray,
    accepted: np.ndarray,
) -> tuple[tuple[float, float], ...]:
    """Return connected blocks of accepted neighboring alpha-grid points."""
    alpha_candidates = _as_1d_array(alpha_candidates, "alpha candidates")
    _validate_strictly_increasing(alpha_candidates)

    accepted_array = np.asarray(accepted)
    if accepted_array.dtype != np.bool_:
        raise ValueError("accepted mask must be boolean")
    accepted = accepted_array
    if accepted.ndim != 1:
        raise ValueError("accepted mask must be one-dimensional")
    if accepted.size != alpha_candidates.size:
        raise ValueError("accepted mask and alpha candidates must have equal length")
    if not np.any(accepted):
        return ()

    accepted_indices = np.flatnonzero(accepted)
    split_points = np.flatnonzero(np.diff(accepted_indices) > 1) + 1
    index_blocks = np.split(accepted_indices, split_points)

    return tuple(
        (float(alpha_candidates[block[0]]), float(alpha_candidates[block[-1]]))
        for block in index_blocks
    )


def _covers_alpha(
    blocks: tuple[tuple[float, float], ...],
    alpha_true: float | None,
) -> bool | None:
    """Return whether alpha_true belongs to at least one accepted block."""
    if alpha_true is None:
        return None
    alpha_true = float(alpha_true)
    if not np.isfinite(alpha_true):
        raise ValueError("alpha_true must be finite when provided")
    if not blocks:
        return False

    # Coverage is computed using membership in accepted grid blocks, not the
    # convex hull of accepted alphas. This matters for weak-IV settings where
    # inverted confidence regions may be disconnected.
    return any(lower <= alpha_true <= upper for lower, upper in blocks)


def _interpolate_crossing(
    alpha_left: float,
    value_left: float,
    alpha_right: float,
    value_right: float,
    critical_value: float,
) -> float:
    if value_left == value_right:
        return float(alpha_left)
    weight = (critical_value - value_left) / (value_right - value_left)
    return float(alpha_left + weight * (alpha_right - alpha_left))


def _accepted_blocks_interpolated(
    alpha_candidates: np.ndarray,
    statistic_values: np.ndarray,
    critical_value: float,
    accepted: np.ndarray,
) -> tuple[tuple[float, float], ...]:
    alpha_candidates = _as_1d_array(alpha_candidates, "alpha candidates")
    statistic_values = _as_1d_array(statistic_values, "statistic values")
    _validate_strictly_increasing(alpha_candidates)

    accepted_array = np.asarray(accepted)
    if accepted_array.dtype != np.bool_:
        raise ValueError("accepted mask must be boolean")
    accepted = accepted_array
    if accepted.ndim != 1:
        raise ValueError("accepted mask must be one-dimensional")
    if not (
        accepted.size == alpha_candidates.size == statistic_values.size
    ):
        raise ValueError(
            "accepted mask, alpha candidates, and statistic values must have equal length"
        )
    if not np.any(accepted):
        return ()

    accepted_indices = np.flatnonzero(accepted)
    split_points = np.flatnonzero(np.diff(accepted_indices) > 1) + 1
    index_blocks = np.split(accepted_indices, split_points)

    blocks: list[tuple[float, float]] = []
    for block in index_blocks:
        start = int(block[0])
        end = int(block[-1])

        lower = float(alpha_candidates[start])
        if start > 0:
            lower = _interpolate_crossing(
                float(alpha_candidates[start - 1]),
                float(statistic_values[start - 1]),
                float(alpha_candidates[start]),
                float(statistic_values[start]),
                critical_value,
            )

        upper = float(alpha_candidates[end])
        if end < alpha_candidates.size - 1:
            upper = _interpolate_crossing(
                float(alpha_candidates[end]),
                float(statistic_values[end]),
                float(alpha_candidates[end + 1]),
                float(statistic_values[end + 1]),
                critical_value,
            )

        blocks.append((lower, upper))

    return tuple(blocks)


def invert_score_test(
    alphas: np.ndarray,
    statistics: np.ndarray,
    critical_value: float,
    alpha_true: float | None = None,
    statistic_reference: float | None = None,
    inversion_type: Literal["absolute", "qlr"] = "absolute",
) -> ConfidenceRegion:
    """Invert a grid-evaluated score test into a confidence region.

    In this project, confidence regions are formed by inverting the scalar
    alpha objective over a grid. Failed alpha evaluations should be represented
    by large finite statistics before calling this function.
    """
    alphas, statistics = _validate_grid_and_statistics(
        alphas,
        statistics,
        sort_alphas=True,
    )
    critical_value = _validate_critical_value(critical_value)
    if inversion_type not in {"absolute", "qlr"}:
        raise ValueError("inversion_type must be 'absolute' or 'qlr'")
    if inversion_type == "absolute":
        statistic_reference = 0.0
    elif statistic_reference is None:
        statistic_reference = float(np.min(statistics))
    statistic_reference = _validate_statistic_reference(statistic_reference)
    if (
        inversion_type == "qlr"
        and statistic_reference > float(np.min(statistics)) + 1e-10
    ):
        raise ValueError(
            "statistic_reference cannot exceed the minimum statistic for qlr inversion"
        )

    statistic_values = statistics - statistic_reference
    accepted_mask = statistic_values <= critical_value
    accepted = alphas[accepted_mask]
    blocks = _accepted_blocks_interpolated(
        alphas,
        statistic_values,
        critical_value,
        accepted_mask,
    )
    covers_true = _covers_alpha(blocks, alpha_true)

    if accepted.size == 0:
        return ConfidenceRegion(
            lower=None,
            upper=None,
            length=0.0,
            hull_length=0.0,
            blocks=(),
            accepted_alphas=(),
            n_blocks=0,
            empty=True,
            disconnected=False,
            covers_true=covers_true,
            selected_grid=_readonly_copy(accepted),
            critical_value=critical_value,
            statistic_reference=statistic_reference,
        )

    lower = float(min(block_lower for block_lower, _block_upper in blocks))
    upper = float(max(block_upper for _block_lower, block_upper in blocks))
    region_length = float(sum(block_upper - block_lower for block_lower, block_upper in blocks))
    hull_length = upper - lower

    return ConfidenceRegion(
        lower=lower,
        upper=upper,
        length=region_length,
        hull_length=hull_length,
        blocks=blocks,
        accepted_alphas=tuple(float(alpha) for alpha in accepted),
        n_blocks=len(blocks),
        empty=False,
        disconnected=len(blocks) > 1,
        covers_true=covers_true,
        selected_grid=_readonly_copy(accepted),
        critical_value=critical_value,
        statistic_reference=statistic_reference,
    )


def is_disconnected_region(
    accepted_grid: np.ndarray,
    full_grid: np.ndarray,
) -> bool:
    """Return True if accepted grid points form separated blocks."""
    full_grid = _as_1d_array(full_grid, "full grid")
    _validate_strictly_increasing(full_grid)

    accepted_grid = np.asarray(accepted_grid, dtype=float)
    if accepted_grid.ndim != 1:
        raise ValueError("accepted grid must be one-dimensional")
    if accepted_grid.size == 0:
        return False
    if not np.all(np.isfinite(accepted_grid)):
        raise ValueError("accepted grid must contain only finite values")
    if accepted_grid.size == 1:
        return False

    indices = np.searchsorted(full_grid, accepted_grid)
    if np.any(indices >= full_grid.size) or not np.all(full_grid[indices] == accepted_grid):
        raise ValueError("accepted grid points must be contained in the full grid")

    indices = np.sort(indices)
    return bool(np.any(np.diff(indices) > 1))


def critical_value_chi_square(
    level: float = 0.95,
    df: int = 1,
) -> float:
    """Return a chi-square critical value.

    The default df=1 is used for scalar alpha score inversion. For fully
    overidentified GMM J-tests, the relevant degrees of freedom may differ and
    should be supplied explicitly. In this project, confidence regions are
    formed by inverting the scalar alpha objective over a grid.
    """
    if isinstance(level, bool):
        raise ValueError("level must satisfy 0 < level < 1")
    level = float(level)
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must satisfy 0 < level < 1")
    if not isinstance(df, int) or isinstance(df, bool):
        raise ValueError("df must be an integer")
    if df < 1:
        raise ValueError("df must be at least 1")

    return float(chi2.ppf(level, df=df))


def sanitize_grid_statistics(
    statistics: np.ndarray,
    converged: np.ndarray | list[bool],
    failed_value: float = FAILED_ALPHA_STATISTIC,
) -> tuple[np.ndarray, int]:
    """Replace failed or non-finite alpha-grid statistics with a finite value."""
    statistics = np.asarray(statistics, dtype=float)
    converged_array = np.asarray(converged)
    if converged_array.dtype != np.bool_:
        raise ValueError("converged must be boolean")
    converged = converged_array

    if statistics.ndim != 1:
        raise ValueError("statistics must be one-dimensional")
    if converged.ndim != 1:
        raise ValueError("converged must be one-dimensional")
    if statistics.size != converged.size:
        raise ValueError("statistics and converged must have equal length")
    if statistics.size == 0:
        raise ValueError("statistics must be nonempty")
    if isinstance(failed_value, bool):
        raise ValueError("failed_value must be positive and finite")
    failed_value = float(failed_value)
    if not np.isfinite(failed_value) or failed_value <= 0:
        raise ValueError("failed_value must be positive and finite")

    sanitized = statistics.copy()
    failed_mask = (~converged) | (~np.isfinite(sanitized))
    sanitized[failed_mask] = failed_value
    if not np.all(np.isfinite(sanitized)):
        raise ValueError("sanitized statistics must be finite")

    return sanitized, int(failed_mask.sum())


def argmin_grid(
    alphas: np.ndarray,
    statistics: np.ndarray,
) -> tuple[float, float, bool]:
    """Return the grid minimizer, minimum statistic, and boundary flag."""
    alphas, statistics = _validate_grid_and_statistics(alphas, statistics)

    min_index = int(np.argmin(statistics))
    alpha_hat = float(alphas[min_index])
    min_statistic = float(statistics[min_index])
    at_boundary = min_index == 0 or min_index == alphas.size - 1

    return alpha_hat, min_statistic, bool(at_boundary)


def summarize_region(region: ConfidenceRegion) -> dict[str, object]:
    """Return ConfidenceRegion fields in EstimationResult-compatible names."""
    return {
        "cr_lower": region.lower,
        "cr_upper": region.upper,
        "cr_length": region.length,
        "cr_empty": region.empty,
        "cr_disconnected": region.disconnected,
        "cr_covers_true": region.covers_true,
    }
