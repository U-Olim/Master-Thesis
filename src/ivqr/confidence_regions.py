"""Grid-based confidence-region inversion for IVQR simulation outputs.

The project convention is

    CR_tau = {a in A : T_n(a) <= c_tau}.

Boundary-hit diagnostics are true only when an accepted grid point touches an
endpoint of A. Disconnected regions mean more than one accepted block on the
alpha grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import chi2


FAILED_ALPHA_STATISTIC = 1e12
ConfidenceRegionStatus = Literal[
    "valid",
    "partially_unresolved",
    "fully_unresolved",
    "empty_valid",
    "full_grid_valid",
]
CoverageStatus = Literal["covered", "not_covered", "coverage_unresolved", "unknown"]
PointEstimateStatus = Literal["resolved", "potentially_unresolved", "fully_unresolved"]


__all__ = [
    "FAILED_ALPHA_STATISTIC",
    "AlphaGridMasks",
    "ConfidenceRegion",
    "PointEstimateResult",
    "adjust_critical_value",
    "invert_score_test",
    "critical_value_chi_square",
    "validate_critical_value_multiplier",
    "sanitize_grid_statistics",
    "classify_alpha_grid",
    "argmin_grid",
    "argmin_grid_usable",
    "summarize_alpha_grid_diagnostics",
    "merge_region_and_grid_diagnostics",
]


@dataclass(frozen=True)
class AlphaGridMasks:
    """Disjoint inferential states for each alpha-grid evaluation."""

    usable: np.ndarray
    accepted: np.ndarray
    rejected: np.ndarray
    unresolved: np.ndarray


@dataclass(frozen=True)
class PointEstimateResult:
    """Grid argmin restricted to finite usable alpha evaluations."""

    alpha_hat: float | None
    statistic: float | None
    at_boundary: bool
    status: PointEstimateStatus


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
    critical_value_nominal: float
    critical_value_multiplier: float
    critical_value_adjusted: float
    statistic_reference: float
    status: ConfidenceRegionStatus
    is_numerically_resolved: bool
    unresolved_count: int
    unresolved_alphas: tuple[float, ...]
    usable_count: int
    rejected_count: int
    full_grid_accepted: bool
    coverage_status: CoverageStatus

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


def validate_critical_value_multiplier(multiplier: float) -> float:
    """Validate the sensitivity multiplier applied to CR critical values."""
    if isinstance(multiplier, bool):
        raise ValueError("critical_value_multiplier must be positive and finite")
    multiplier = float(multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("critical_value_multiplier must be positive and finite")
    return multiplier


def adjust_critical_value(
    critical_value: float,
    critical_value_multiplier: float = 1.0,
) -> float:
    """Return the critical value used for CR inversion after sensitivity scaling."""
    critical_value = _validate_critical_value(critical_value)
    multiplier = validate_critical_value_multiplier(critical_value_multiplier)
    return float(critical_value * multiplier)


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


def _readonly_bool_copy(values: np.ndarray) -> np.ndarray:
    copied = np.array(values, dtype=bool, copy=True)
    copied.setflags(write=False)
    return copied


def classify_alpha_grid(
    statistics: np.ndarray,
    usable: np.ndarray | list[bool],
    critical_value: float,
) -> AlphaGridMasks:
    """Partition alpha evaluations into accepted, rejected, and unresolved.

    A failed evaluation is unresolved regardless of its numeric placeholder.
    The returned masks are pairwise disjoint and exhaustive.
    """
    statistics = np.asarray(statistics, dtype=float)
    usable_array = np.asarray(usable)
    critical_value = _validate_critical_value(critical_value)
    if statistics.ndim != 1 or statistics.size == 0:
        raise ValueError("statistics must be a nonempty one-dimensional array")
    if usable_array.dtype != np.bool_ or usable_array.ndim != 1:
        raise ValueError("usable must be a one-dimensional boolean array")
    if statistics.size != usable_array.size:
        raise ValueError("statistics and usable must have equal length")
    usable_mask = usable_array & np.isfinite(statistics)
    unresolved = ~usable_mask
    accepted = usable_mask & (statistics <= critical_value)
    rejected = usable_mask & (statistics > critical_value)
    if not np.all(accepted | rejected | unresolved):
        raise RuntimeError("alpha inference masks must be exhaustive")
    overlap = (accepted & rejected) | (accepted & unresolved) | (rejected & unresolved)
    if np.any(overlap):
        raise RuntimeError("alpha inference masks must be pairwise disjoint")
    return AlphaGridMasks(
        usable=_readonly_bool_copy(usable_mask),
        accepted=_readonly_bool_copy(accepted),
        rejected=_readonly_bool_copy(rejected),
        unresolved=_readonly_bool_copy(unresolved),
    )


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
    usable: np.ndarray,
) -> tuple[tuple[float, float], ...]:
    alpha_candidates = _as_1d_array(alpha_candidates, "alpha candidates")
    statistic_values = np.asarray(statistic_values, dtype=float)
    _validate_strictly_increasing(alpha_candidates)

    accepted_array = np.asarray(accepted)
    if accepted_array.dtype != np.bool_:
        raise ValueError("accepted mask must be boolean")
    accepted = accepted_array
    usable_array = np.asarray(usable)
    if usable_array.dtype != np.bool_ or usable_array.ndim != 1:
        raise ValueError("usable mask must be one-dimensional and boolean")
    usable = usable_array
    if accepted.ndim != 1:
        raise ValueError("accepted mask must be one-dimensional")
    if not (
        accepted.size == usable.size == alpha_candidates.size == statistic_values.size
    ):
        raise ValueError(
            "accepted mask, alpha candidates, and statistic values must have equal length"
        )
    if not np.all(np.isfinite(statistic_values[usable])):
        raise ValueError("usable statistic values must be finite")
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
        if start > 0 and usable[start - 1]:
            lower = _interpolate_crossing(
                float(alpha_candidates[start - 1]),
                float(statistic_values[start - 1]),
                float(alpha_candidates[start]),
                float(statistic_values[start]),
                critical_value,
            )

        upper = float(alpha_candidates[end])
        if end < alpha_candidates.size - 1 and usable[end + 1]:
            upper = _interpolate_crossing(
                float(alpha_candidates[end]),
                float(statistic_values[end]),
                float(alpha_candidates[end + 1]),
                float(statistic_values[end + 1]),
                critical_value,
            )

        blocks.append((lower, upper))

    return tuple(blocks)


def _coverage_from_grid(
    alphas: np.ndarray,
    blocks: tuple[tuple[float, float], ...],
    masks: AlphaGridMasks,
    alpha_true: float | None,
) -> tuple[bool | None, CoverageStatus]:
    """Return tri-state coverage using only the truth's resolved neighborhood."""
    if alpha_true is None:
        return None, "unknown"
    alpha_true = float(alpha_true)
    if not np.isfinite(alpha_true):
        raise ValueError("alpha_true must be finite when provided")
    if any(lower <= alpha_true <= upper for lower, upper in blocks):
        return True, "covered"
    exact = np.flatnonzero(np.isclose(alphas, alpha_true, rtol=0.0, atol=1e-12))
    if exact.size:
        index = int(exact[0])
        if masks.unresolved[index]:
            return None, "coverage_unresolved"
        return False, "not_covered"
    if alpha_true < alphas[0] or alpha_true > alphas[-1]:
        return False, "not_covered"
    right = int(np.searchsorted(alphas, alpha_true, side="right"))
    left = right - 1
    if masks.unresolved[left] or masks.unresolved[right]:
        return None, "coverage_unresolved"
    return False, "not_covered"


def invert_score_test(
    alphas: np.ndarray,
    statistics: np.ndarray,
    critical_value: float,
    alpha_true: float | None = None,
    statistic_reference: float | None = None,
    inversion_type: str = "absolute",
    critical_value_multiplier: float = 1.0,
    usable: np.ndarray | list[bool] | None = None,
) -> ConfidenceRegion:
    """Invert a grid-evaluated score test into a confidence region.

    Unusable evaluations are unresolved: they are neither rejected nor used
    for boundary interpolation. Omitting ``usable`` preserves the all-usable
    behavior for callers whose supplied statistics are already valid.
    """
    alphas = _as_1d_array(alphas, "alpha grid")
    statistics = np.asarray(statistics, dtype=float)
    if statistics.ndim != 1 or statistics.size != alphas.size:
        raise ValueError("alpha grid and statistics must be matching vectors")
    usable_array = (
        np.ones(alphas.size, dtype=bool) if usable is None else np.asarray(usable)
    )
    if usable_array.dtype != np.bool_ or usable_array.shape != alphas.shape:
        raise ValueError("usable must be a boolean vector matching the alpha grid")
    order = np.argsort(alphas)
    alphas = alphas[order]
    statistics = statistics[order]
    usable_array = usable_array[order]
    _validate_strictly_increasing(alphas)
    critical_value_nominal = _validate_critical_value(critical_value)
    critical_value_multiplier = validate_critical_value_multiplier(
        critical_value_multiplier
    )
    critical_value_adjusted = adjust_critical_value(
        critical_value_nominal,
        critical_value_multiplier,
    )
    if inversion_type != "absolute":
        raise ValueError("Only absolute confidence-region inversion is supported.")
    statistic_reference = (
        0.0
        if statistic_reference is None
        else _validate_statistic_reference(statistic_reference)
    )

    statistic_values = statistics - statistic_reference
    masks = classify_alpha_grid(statistic_values, usable_array, critical_value_adjusted)
    accepted = alphas[masks.accepted]
    blocks = _accepted_blocks_interpolated(
        alphas,
        statistic_values,
        critical_value_adjusted,
        masks.accepted,
        masks.usable,
    )
    covers_true, coverage_status = _coverage_from_grid(
        alphas, blocks, masks, alpha_true
    )
    unresolved_alphas = tuple(float(value) for value in alphas[masks.unresolved])
    unresolved_count = len(unresolved_alphas)
    usable_count = int(np.sum(masks.usable))
    rejected_count = int(np.sum(masks.rejected))
    fully_resolved = unresolved_count == 0
    full_grid_accepted = fully_resolved and accepted.size == alphas.size
    if usable_count == 0:
        status: ConfidenceRegionStatus = "fully_unresolved"
    elif unresolved_count:
        status = "partially_unresolved"
    elif accepted.size == 0:
        status = "empty_valid"
    elif full_grid_accepted:
        status = "full_grid_valid"
    else:
        status = "valid"

    shared = {
        "covers_true": covers_true,
        "selected_grid": _readonly_copy(accepted),
        "critical_value": critical_value_adjusted,
        "critical_value_nominal": critical_value_nominal,
        "critical_value_multiplier": critical_value_multiplier,
        "critical_value_adjusted": critical_value_adjusted,
        "statistic_reference": statistic_reference,
        "status": status,
        "is_numerically_resolved": fully_resolved,
        "unresolved_count": unresolved_count,
        "unresolved_alphas": unresolved_alphas,
        "usable_count": usable_count,
        "rejected_count": rejected_count,
        "full_grid_accepted": full_grid_accepted,
        "coverage_status": coverage_status,
    }

    if accepted.size == 0:
        return ConfidenceRegion(
            lower=None,
            upper=None,
            length=float("nan"),
            hull_length=float("nan"),
            blocks=(),
            accepted_alphas=(),
            n_blocks=0,
            empty=True,
            disconnected=False,
            **shared,
        )

    lower = float(min(block_lower for block_lower, _block_upper in blocks))
    upper = float(max(block_upper for _block_lower, block_upper in blocks))
    region_length = float(
        sum(block_upper - block_lower for block_lower, block_upper in blocks)
    )
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
        **shared,
    )


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


def argmin_grid_usable(
    alphas: np.ndarray,
    statistics: np.ndarray,
    usable: np.ndarray | list[bool],
) -> PointEstimateResult:
    """Minimize over usable finite statistics without selecting failures."""
    alphas = _as_1d_array(alphas, "alpha grid")
    statistics = np.asarray(statistics, dtype=float)
    usable_array = np.asarray(usable)
    if statistics.ndim != 1 or statistics.shape != alphas.shape:
        raise ValueError("statistics must match the alpha grid")
    if usable_array.dtype != np.bool_ or usable_array.shape != alphas.shape:
        raise ValueError("usable must be a boolean vector matching the alpha grid")
    _validate_strictly_increasing(alphas)
    valid = usable_array & np.isfinite(statistics)
    if not np.any(valid):
        return PointEstimateResult(None, None, False, "fully_unresolved")
    valid_indices = np.flatnonzero(valid)
    index = int(valid_indices[int(np.argmin(statistics[valid]))])
    status: PointEstimateStatus = (
        "resolved" if np.all(valid) else "potentially_unresolved"
    )
    return PointEstimateResult(
        alpha_hat=float(alphas[index]),
        statistic=float(statistics[index]),
        at_boundary=index == 0 or index == alphas.size - 1,
        status=status,
    )


def _nan() -> float:
    return float("nan")


def _optional_float(value: float | None) -> float:
    if value is None:
        return _nan()
    value = float(value)
    return value if np.isfinite(value) else _nan()


def _count_accepted_blocks(accepted_mask: np.ndarray) -> int:
    accepted_indices = np.flatnonzero(accepted_mask)
    if accepted_indices.size == 0:
        return 0
    return int(np.sum(np.diff(accepted_indices) > 1) + 1)


def summarize_alpha_grid_diagnostics(
    alpha_grid: np.ndarray,
    accepted_mask: np.ndarray | None,
    alpha_hat: float | None,
    failed_alpha_count: int = 0,
    test_stats: np.ndarray | None = None,
    critical_value: float | None = None,
    critical_value_nominal: float | None = None,
    critical_value_multiplier: float | None = None,
    critical_value_adjusted: float | None = None,
) -> dict[str, Any]:
    """Summarize alpha-grid, confidence-region, and boundary diagnostics."""
    alphas = _as_1d_array(alpha_grid, "alpha grid")
    _validate_strictly_increasing(alphas)
    alpha_grid_size = int(alphas.size)
    alpha_grid_min = float(alphas[0])
    alpha_grid_max = float(alphas[-1])
    alpha_grid_step = (
        (alpha_grid_max - alpha_grid_min) / (alpha_grid_size - 1)
        if alpha_grid_size > 1
        else _nan()
    )

    if failed_alpha_count < 0:
        raise ValueError("failed_alpha_count must be nonnegative")
    if failed_alpha_count > alpha_grid_size:
        raise ValueError("failed_alpha_count must not exceed alpha grid size")

    alpha_hat_value = _optional_float(alpha_hat)
    alpha_hat_at_lower = bool(
        np.isfinite(alpha_hat_value) and np.isclose(alpha_hat_value, alpha_grid_min)
    )
    alpha_hat_at_upper = bool(
        np.isfinite(alpha_hat_value) and np.isclose(alpha_hat_value, alpha_grid_max)
    )

    if accepted_mask is None:
        accepted = np.zeros(alpha_grid_size, dtype=bool)
    else:
        accepted = np.asarray(accepted_mask)
        if accepted.dtype != np.bool_:
            raise ValueError("accepted_mask must be boolean")
        if accepted.ndim != 1:
            raise ValueError("accepted_mask must be one-dimensional")
        if accepted.size != alpha_grid_size:
            raise ValueError("accepted_mask and alpha grid must have equal length")

    accepted_count = int(np.sum(accepted))
    cr_empty = accepted_count == 0
    cr_n_blocks = _count_accepted_blocks(accepted)
    accepted_alphas = alphas[accepted]
    if cr_empty:
        cr_lower = _nan()
        cr_upper = _nan()
        cr_length = _nan()
        cr_hull_length = _nan()
    else:
        cr_lower = float(accepted_alphas[0])
        cr_upper = float(accepted_alphas[-1])
        cr_length = cr_upper - cr_lower
        cr_hull_length = cr_length

    cr_hits_lower = bool(accepted[0])
    cr_hits_upper = bool(accepted[-1])

    min_test_stat = _nan()
    max_test_stat = _nan()
    test_stat_at_alpha_hat = _nan()
    if test_stats is not None:
        stats = np.asarray(test_stats, dtype=float)
        if stats.ndim != 1 or stats.size != alpha_grid_size:
            raise ValueError("test_stats and alpha grid must have equal length")
        finite_stats = stats[np.isfinite(stats)]
        if finite_stats.size:
            min_test_stat = float(np.min(finite_stats))
            max_test_stat = float(np.max(finite_stats))
        if np.isfinite(alpha_hat_value):
            matches = np.flatnonzero(np.isclose(alphas, alpha_hat_value))
            if matches.size > 0 and np.isfinite(stats[int(matches[0])]):
                test_stat_at_alpha_hat = float(stats[int(matches[0])])

    if critical_value_nominal is None:
        critical_value_nominal = critical_value
    if critical_value_multiplier is None:
        critical_value_multiplier = 1.0 if critical_value_nominal is not None else None
    if critical_value_adjusted is None:
        critical_value_adjusted = critical_value
    if critical_value is None:
        critical_value = critical_value_adjusted

    return {
        "alpha_grid_min": alpha_grid_min,
        "alpha_grid_max": alpha_grid_max,
        "alpha_grid_size": alpha_grid_size,
        "alpha_grid_step": alpha_grid_step,
        "alpha_hat_at_lower_boundary": alpha_hat_at_lower,
        "alpha_hat_at_upper_boundary": alpha_hat_at_upper,
        "alpha_hat_at_any_boundary": alpha_hat_at_lower or alpha_hat_at_upper,
        "cr_lower": cr_lower,
        "cr_upper": cr_upper,
        "cr_length": cr_length,
        "cr_hits_lower_boundary": cr_hits_lower,
        "cr_hits_upper_boundary": cr_hits_upper,
        "cr_hits_any_boundary": cr_hits_lower or cr_hits_upper,
        "cr_empty": cr_empty,
        "cr_accepted_alpha_count": accepted_count,
        "cr_acceptance_rate": accepted_count / alpha_grid_size,
        "cr_n_blocks": cr_n_blocks,
        "cr_disconnected": cr_n_blocks > 1,
        "cr_hull_length": cr_hull_length,
        "failed_alpha_count": int(failed_alpha_count),
        "failed_alpha_rate": failed_alpha_count / alpha_grid_size,
        "min_test_stat": min_test_stat,
        "max_test_stat": max_test_stat,
        "test_stat_at_alpha_hat": test_stat_at_alpha_hat,
        "critical_value": _optional_float(critical_value),
        "critical_value_nominal": _optional_float(critical_value_nominal),
        "critical_value_multiplier": _optional_float(critical_value_multiplier),
        "critical_value_adjusted": _optional_float(critical_value_adjusted),
    }


def merge_region_and_grid_diagnostics(
    region: ConfidenceRegion,
    grid_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Return grid diagnostics with authoritative region geometry merged in."""
    diagnostics = dict(grid_diagnostics)
    # Confidence-region geometry is taken from ConfidenceRegion because it may
    # include interpolation and disconnected blocks. Grid diagnostics are used
    # only for accepted-point counts and boundary-hit flags.
    diagnostics.update(
        {
            "cr_lower": _optional_float(region.lower),
            "cr_upper": _optional_float(region.upper),
            "cr_length": float(region.length),
            "cr_hull_length": float(region.hull_length),
            "cr_empty": bool(region.empty),
            "cr_n_blocks": int(region.n_blocks),
            "cr_disconnected": bool(region.disconnected),
        }
    )
    return diagnostics
