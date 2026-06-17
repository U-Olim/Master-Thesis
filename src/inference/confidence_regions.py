"""Inference routines for IVQR simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


FAILED_ALPHA_STATISTIC = 1e12


@dataclass(frozen=True)
class ConfidenceRegion:
    """Grid-based confidence region from score-test inversion."""

    lower: float | None
    upper: float | None
    length: float | None
    empty: bool
    disconnected: bool
    covers_true: bool | None
    selected_grid: np.ndarray


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
) -> tuple[np.ndarray, np.ndarray]:
    alphas = _as_1d_array(alphas, "alpha grid")
    statistics = _as_1d_array(statistics, "statistics")

    if alphas.size != statistics.size:
        raise ValueError("alpha grid and statistics must have equal length")

    _validate_strictly_increasing(alphas)
    return alphas, statistics


def _validate_critical_value(critical_value: float) -> float:
    critical_value = float(critical_value)
    if not np.isfinite(critical_value) or critical_value <= 0:
        raise ValueError("critical value must be positive and finite")
    return critical_value


def invert_score_test(
    alphas: np.ndarray,
    statistics: np.ndarray,
    critical_value: float,
    alpha_true: float | None = None,
) -> ConfidenceRegion:
    """Invert a grid-evaluated score test into a confidence region.

    In this project, confidence regions are formed by inverting the scalar
    alpha objective over a grid. Failed alpha evaluations should be represented
    by large finite statistics before calling this function.
    """
    alphas, statistics = _validate_grid_and_statistics(alphas, statistics)
    critical_value = _validate_critical_value(critical_value)

    accepted = alphas[statistics <= critical_value]

    if accepted.size == 0:
        covers_true = False if alpha_true is not None else None
        return ConfidenceRegion(
            lower=None,
            upper=None,
            length=None,
            empty=True,
            disconnected=False,
            covers_true=covers_true,
            selected_grid=accepted,
        )

    lower = float(accepted.min())
    upper = float(accepted.max())
    covers_true = None
    if alpha_true is not None:
        covers_true = lower <= alpha_true <= upper

    return ConfidenceRegion(
        lower=lower,
        upper=upper,
        length=upper - lower,
        empty=False,
        disconnected=is_disconnected_region(accepted, alphas),
        covers_true=covers_true,
        selected_grid=accepted,
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
    if not 0 < level < 1:
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
    converged = np.asarray(converged, dtype=bool)

    if statistics.ndim != 1:
        raise ValueError("statistics must be one-dimensional")
    if converged.ndim != 1:
        raise ValueError("converged must be one-dimensional")
    if statistics.size != converged.size:
        raise ValueError("statistics and converged must have equal length")
    if statistics.size == 0:
        raise ValueError("statistics must be nonempty")
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
