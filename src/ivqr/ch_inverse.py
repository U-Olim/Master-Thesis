"""Common Chernozhukov-Hansen inverse-IVQR grid routines.

This module contains low-dimensional CH-IVQR machinery. The same inversion
logic is shared by the Oracle and post-selection IVQR estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import json
from time import perf_counter
from typing import Literal, cast
import warnings

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from dgp.designs import SimData
from estimators.base import EstimationResult, estimation_result_diagnostic_kwargs
from ivqr.alpha_grid import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_ALPHA_STEP,
    alpha_grid,
)
from ivqr.confidence_regions import (
    adjust_critical_value,
    argmin_grid_usable,
    classify_alpha_grid,
    critical_value_chi_square,
    invert_score_test,
    merge_region_and_grid_diagnostics,
    sanitize_grid_statistics,
    summarize_alpha_grid_diagnostics,
    validate_critical_value_multiplier,
)
from simulation.config import DEFAULT_QUANTREG_MAX_ITER
from utils.timing import RuntimeDiagnosticColumns, estimator_runtime_columns
from utils.validation import (
    validate_1d_array,
    validate_2d_array,
    validate_alpha_grid,
    validate_data_arrays,
    validate_tau,
)


IterationWarningPolicy = Literal["reject", "use_if_valid"]
AlphaInferenceStatus = Literal["valid", "unresolved"]
HardFailurePolicy = Literal["unresolved", "legacy_reject"]
GridStrategy = Literal["fixed", "adaptive"]
AlphaHatGrid = Literal["initial", "all_evaluated"]
ITERATION_WARNING_POLICIES: tuple[IterationWarningPolicy, ...] = (
    "use_if_valid",
    "reject",
)
_COVARIANCE_TOLERANCE = 1e-10
HARD_FAILURE_POLICIES: tuple[HardFailurePolicy, ...] = (
    "unresolved",
    "legacy_reject",
)
GRID_STRATEGIES: tuple[GridStrategy, ...] = ("fixed", "adaptive")
DEFAULT_REFINEMENT_TOLERANCE = 0.025
DEFAULT_MAX_REFINEMENT_DEPTH = 10
DEFAULT_MAX_ALPHA_EVALUATIONS = 201
DEFAULT_ADAPTIVE_MIDPOINT_PROBE = True
DEFAULT_ALPHA_HAT_GRID: AlphaHatGrid = "initial"
ALPHA_HAT_GRIDS: tuple[AlphaHatGrid, ...] = ("initial", "all_evaluated")


@dataclass(frozen=True)
class AlphaEvaluation:
    """Evaluation of one structural alpha candidate in inverse IVQR.

    ``converged`` records optimizer convergence, while ``usable`` records the
    separate statistical judgment that the returned Wald statistic is valid
    for inference.  An iteration-limit fit can therefore be usable without
    being classified as converged.
    """

    statistic: float
    gamma_hat: np.ndarray
    cov_gamma: np.ndarray
    dim_z: int
    converged: bool
    usable: bool
    warning_type: str | None
    failure_reason: str | None
    message: str
    covariance_rank: int | None = None
    covariance_condition_number: float | None = None

    @property
    def inferential_status(self) -> AlphaInferenceStatus:
        """Return validity for inference without encoding acceptance."""
        return "valid" if self.usable else "unresolved"


@dataclass(frozen=True)
class AlphaGridEvaluation:
    """Cached, sorted evaluations and adaptive-refinement metadata."""

    alphas: np.ndarray
    evaluations: tuple[AlphaEvaluation, ...]
    grid_strategy: GridStrategy
    initial_alpha_grid_size: int
    initial_alphas: np.ndarray
    adaptive_midpoint_probe: bool
    alpha_hat_grid: AlphaHatGrid
    midpoint_intervals_considered: int
    midpoint_evaluations_added: int
    midpoint_unresolved_barriers: int
    midpoint_probe_limit_hit: bool
    refinement_tolerance: float
    refinement_depth_reached: int
    refinement_limit_hit: bool
    max_alpha_evaluations_hit: bool
    number_of_refined_intervals: int
    number_of_unresolved_refinement_barriers: int

    @property
    def final_alpha_evaluations(self) -> int:
        return len(self.evaluations)

    @property
    def spacings(self) -> tuple[float | None, float | None, float | None]:
        differences = np.diff(self.alphas)
        if differences.size == 0:
            return None, None, None
        return (
            float(np.min(differences)),
            float(np.median(differences)),
            float(np.max(differences)),
        )


def validate_iteration_warning_policy(
    policy: IterationWarningPolicy | str,
) -> IterationWarningPolicy:
    """Validate and return the QuantReg iteration-warning policy."""
    if policy not in ITERATION_WARNING_POLICIES:
        choices = ", ".join(ITERATION_WARNING_POLICIES)
        raise ValueError(f"iteration_warning_policy must be one of: {choices}")
    return cast(IterationWarningPolicy, policy)


def validate_hard_failure_policy(
    policy: HardFailurePolicy | str,
) -> HardFailurePolicy:
    """Validate the treatment of unusable alpha evaluations."""
    if policy not in HARD_FAILURE_POLICIES:
        choices = ", ".join(HARD_FAILURE_POLICIES)
        raise ValueError(f"hard_failure_policy must be one of: {choices}")
    return cast(HardFailurePolicy, policy)


def validate_grid_strategy(strategy: GridStrategy | str) -> GridStrategy:
    """Validate the production CH alpha-grid strategy."""
    if strategy not in GRID_STRATEGIES:
        choices = ", ".join(GRID_STRATEGIES)
        raise ValueError(f"grid_strategy must be one of: {choices}")
    return cast(GridStrategy, strategy)


def validate_alpha_hat_grid(value: AlphaHatGrid | str) -> AlphaHatGrid:
    """Validate the grid eligible for CH point estimation."""
    if value not in ALPHA_HAT_GRIDS:
        choices = ", ".join(ALPHA_HAT_GRIDS)
        raise ValueError(f"alpha_hat_grid must be one of: {choices}")
    return cast(AlphaHatGrid, value)


def evaluate_ch_alpha_grid(
    initial_alphas: np.ndarray,
    evaluate_alpha: Callable[[float], AlphaEvaluation],
    *,
    critical_value: float,
    grid_strategy: GridStrategy = "adaptive",
    refinement_tolerance: float = DEFAULT_REFINEMENT_TOLERANCE,
    max_refinement_depth: int = DEFAULT_MAX_REFINEMENT_DEPTH,
    max_alpha_evaluations: int = DEFAULT_MAX_ALPHA_EVALUATIONS,
    adaptive_midpoint_probe: bool = DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    alpha_hat_grid: AlphaHatGrid = DEFAULT_ALPHA_HAT_GRID,
) -> AlphaGridEvaluation:
    """Evaluate a midpoint-assisted adaptive CH boundary grid.

    Adaptive mode optionally probes each usable initial interval once, then
    bisects valid accepted/rejected transitions widest-first. It remains a
    boundary-resolution method and does not guarantee discovery of every
    disconnected component. Every exact alpha value is fitted at most once.
    """
    strategy = validate_grid_strategy(grid_strategy)
    alpha_hat_grid = validate_alpha_hat_grid(alpha_hat_grid)
    if not isinstance(adaptive_midpoint_probe, bool):
        raise ValueError("adaptive_midpoint_probe must be boolean")
    raw_alphas = np.asarray(initial_alphas, dtype=float)
    if raw_alphas.ndim != 1 or raw_alphas.size == 0:
        raise ValueError("initial_alphas must be a nonempty vector")
    if not np.all(np.isfinite(raw_alphas)):
        raise ValueError("initial_alphas must contain only finite values")
    alphas = np.unique(raw_alphas)
    initial_alphas = alphas.copy()
    initial_alphas.setflags(write=False)
    if not np.isfinite(critical_value) or critical_value < 0:
        raise ValueError("critical_value must be finite and nonnegative")
    if not np.isfinite(refinement_tolerance) or refinement_tolerance <= 0:
        raise ValueError("refinement_tolerance must be positive and finite")
    if (
        not isinstance(max_refinement_depth, int)
        or isinstance(max_refinement_depth, bool)
        or max_refinement_depth < 0
    ):
        raise ValueError("max_refinement_depth must be a nonnegative integer")
    if (
        not isinstance(max_alpha_evaluations, int)
        or isinstance(max_alpha_evaluations, bool)
        or max_alpha_evaluations < 1
    ):
        raise ValueError("max_alpha_evaluations must be a positive integer")
    if strategy == "adaptive" and alphas.size > max_alpha_evaluations:
        raise ValueError("max_alpha_evaluations must cover the unique initial grid")

    cache: dict[float, AlphaEvaluation] = {}

    def cached_evaluate(alpha: float) -> AlphaEvaluation:
        key = float(alpha)
        if key not in cache:
            cache[key] = evaluate_alpha(key)
        return cache[key]

    for alpha in alphas:
        cached_evaluate(float(alpha))

    midpoint_intervals_considered = 0
    midpoint_evaluations_added = 0
    midpoint_unresolved_barriers = 0
    midpoint_probe_limit_hit = False
    effective_midpoint_probe = strategy == "adaptive" and adaptive_midpoint_probe
    if effective_midpoint_probe:
        for left, right in zip(initial_alphas[:-1], initial_alphas[1:], strict=True):
            left_eval, right_eval = cache[float(left)], cache[float(right)]
            if not (
                left_eval.usable
                and right_eval.usable
                and np.isfinite(left_eval.statistic)
                and np.isfinite(right_eval.statistic)
            ):
                continue
            midpoint_intervals_considered += 1
            if len(cache) >= max_alpha_evaluations:
                midpoint_probe_limit_hit = True
                break
            midpoint = float(left + (right - left) / 2.0)
            if midpoint in cache:
                continue
            evaluation = cached_evaluate(midpoint)
            midpoint_evaluations_added += 1
            if not evaluation.usable or not np.isfinite(evaluation.statistic):
                midpoint_unresolved_barriers += 1

    points_after_probe = np.array(sorted(cache), dtype=float)
    depth_by_interval = {
        (float(left), float(right)): 0
        for left, right in zip(points_after_probe[:-1], points_after_probe[1:], strict=True)
    }
    depth_reached = 0
    depth_limit_hit = False
    evaluation_limit_hit = False
    refined_intervals = 0
    unresolved_barriers = 0

    if strategy == "adaptive":
        while True:
            points = np.array(sorted(cache), dtype=float)
            transitions: list[tuple[float, float, int]] = []
            for left, right in zip(points[:-1], points[1:], strict=True):
                left_key, right_key = float(left), float(right)
                left_eval, right_eval = cache[left_key], cache[right_key]
                if not (
                    left_eval.usable
                    and right_eval.usable
                    and np.isfinite(left_eval.statistic)
                    and np.isfinite(right_eval.statistic)
                ):
                    continue
                changes = (left_eval.statistic <= critical_value) != (
                    right_eval.statistic <= critical_value
                )
                if not changes or right_key - left_key <= refinement_tolerance + 1e-12:
                    continue
                depth = depth_by_interval.get((left_key, right_key), 0)
                if depth >= max_refinement_depth:
                    depth_limit_hit = True
                    continue
                transitions.append((left_key, right_key, depth))
            if not transitions:
                break
            if len(cache) >= max_alpha_evaluations:
                evaluation_limit_hit = True
                break
            # Deterministic fairness: widest transition first, then lower alpha,
            # then shallower depth. Recompute after every midpoint evaluation.
            left, right, depth = min(
                transitions,
                key=lambda interval: (
                    -(interval[1] - interval[0]),
                    interval[0],
                    interval[2],
                ),
            )
            midpoint = float(left + (right - left) / 2.0)
            if midpoint in cache:
                depth_by_interval.pop((left, right), None)
                continue
            evaluation = cached_evaluate(midpoint)
            refined_intervals += 1
            child_depth = depth + 1
            depth_reached = max(depth_reached, child_depth)
            depth_by_interval.pop((left, right), None)
            depth_by_interval[(left, midpoint)] = child_depth
            depth_by_interval[(midpoint, right)] = child_depth
            if not evaluation.usable or not np.isfinite(evaluation.statistic):
                unresolved_barriers += 1

    final_alphas = np.array(sorted(cache), dtype=float)
    final_alphas.setflags(write=False)
    return AlphaGridEvaluation(
        alphas=final_alphas,
        evaluations=tuple(cache[float(alpha)] for alpha in final_alphas),
        grid_strategy=strategy,
        initial_alpha_grid_size=int(alphas.size),
        initial_alphas=initial_alphas,
        adaptive_midpoint_probe=effective_midpoint_probe,
        alpha_hat_grid=alpha_hat_grid,
        midpoint_intervals_considered=midpoint_intervals_considered,
        midpoint_evaluations_added=midpoint_evaluations_added,
        midpoint_unresolved_barriers=midpoint_unresolved_barriers,
        midpoint_probe_limit_hit=midpoint_probe_limit_hit,
        refinement_tolerance=float(refinement_tolerance),
        refinement_depth_reached=depth_reached,
        refinement_limit_hit=(
            depth_limit_hit or evaluation_limit_hit or midpoint_probe_limit_hit
        ),
        max_alpha_evaluations_hit=evaluation_limit_hit or midpoint_probe_limit_hit,
        number_of_refined_intervals=refined_intervals,
        number_of_unresolved_refinement_barriers=unresolved_barriers,
    )


def grid_metadata_kwargs(grid: AlphaGridEvaluation) -> dict[str, object]:
    minimum, median, maximum = grid.spacings
    return {
        "grid_strategy": grid.grid_strategy,
        "initial_alpha_grid_size": grid.initial_alpha_grid_size,
        "adaptive_midpoint_probe": grid.adaptive_midpoint_probe,
        "alpha_hat_grid": grid.alpha_hat_grid,
        "midpoint_intervals_considered": grid.midpoint_intervals_considered,
        "midpoint_evaluations_added": grid.midpoint_evaluations_added,
        "midpoint_unresolved_barriers": grid.midpoint_unresolved_barriers,
        "midpoint_probe_limit_hit": grid.midpoint_probe_limit_hit,
        "final_alpha_evaluations": grid.final_alpha_evaluations,
        "refinement_tolerance": grid.refinement_tolerance,
        "refinement_depth_reached": grid.refinement_depth_reached,
        "refinement_limit_hit": grid.refinement_limit_hit,
        "max_alpha_evaluations_hit": grid.max_alpha_evaluations_hit,
        "number_of_refined_intervals": grid.number_of_refined_intervals,
        "number_of_unresolved_refinement_barriers": (
            grid.number_of_unresolved_refinement_barriers
        ),
        "minimum_final_grid_spacing": minimum,
        "median_final_grid_spacing": median,
        "maximum_final_grid_spacing": maximum,
        "iteration_warning_evaluations": sum(
            evaluation.warning_type == "iteration_limit"
            for evaluation in grid.evaluations
        ),
        "rank_deficient_covariance_failures": sum(
            evaluation.failure_reason == "rank_deficient_instrument_covariance"
            for evaluation in grid.evaluations
        ),
    }


def default_grid_metadata(
    *,
    grid_strategy: GridStrategy,
    adaptive_midpoint_probe: bool = DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    alpha_hat_grid: AlphaHatGrid = DEFAULT_ALPHA_HAT_GRID,
) -> dict[str, object]:
    """Return internally consistent metadata before grid evaluation exists."""
    strategy = validate_grid_strategy(grid_strategy)
    point_grid = validate_alpha_hat_grid(alpha_hat_grid)
    return {
        "grid_strategy": strategy,
        "initial_alpha_grid_size": None,
        "final_alpha_evaluations": None,
        "adaptive_midpoint_probe": strategy == "adaptive" and adaptive_midpoint_probe,
        "alpha_hat_grid": point_grid,
        "midpoint_intervals_considered": 0,
        "midpoint_evaluations_added": 0,
        "midpoint_unresolved_barriers": 0,
        "midpoint_probe_limit_hit": False,
        "refinement_tolerance": None,
        "refinement_depth_reached": 0,
        "refinement_limit_hit": False,
        "max_alpha_evaluations_hit": False,
        "number_of_refined_intervals": 0,
        "number_of_unresolved_refinement_barriers": 0,
        "minimum_final_grid_spacing": None,
        "median_final_grid_spacing": None,
        "maximum_final_grid_spacing": None,
        "iteration_warning_evaluations": 0,
        "rank_deficient_covariance_failures": 0,
    }


def _failed_alpha_evaluation(
    *,
    dim_z: int,
    converged: bool,
    warning_type: str | None,
    failure_reason: str,
    gamma_hat: np.ndarray | None = None,
    cov_gamma: np.ndarray | None = None,
    covariance_rank: int | None = None,
    covariance_condition_number: float | None = None,
) -> AlphaEvaluation:
    return AlphaEvaluation(
        statistic=np.inf,
        gamma_hat=(
            np.full(dim_z, np.nan)
            if gamma_hat is None
            else np.asarray(gamma_hat, dtype=float)
        ),
        cov_gamma=(
            np.full((dim_z, dim_z), np.nan)
            if cov_gamma is None
            else np.atleast_2d(np.asarray(cov_gamma, dtype=float))
        ),
        dim_z=dim_z,
        converged=converged,
        usable=False,
        warning_type=warning_type,
        failure_reason=failure_reason,
        message=failure_reason,
        covariance_rank=covariance_rank,
        covariance_condition_number=covariance_condition_number,
    )


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Return a design matrix with a leading intercept column."""
    x = validate_2d_array("x", x)
    if x.shape[0] == 0:
        raise ValueError("x must contain at least one row")
    return np.column_stack([np.ones(x.shape[0]), x])


def as_2d_instruments(z: np.ndarray) -> np.ndarray:
    """Validate excluded instruments and return a two-dimensional array."""
    z_array = np.asarray(z, dtype=float)
    if z_array.ndim == 1:
        z_array = z_array.reshape(-1, 1)
    if z_array.ndim != 2:
        raise ValueError("z must be one- or two-dimensional")
    if z_array.shape[0] == 0:
        raise ValueError("z must contain at least one row")
    if z_array.shape[1] == 0:
        raise ValueError("z must contain at least one excluded instrument")
    if not np.all(np.isfinite(z_array)):
        raise ValueError("z must contain only finite values")
    return z_array


def ch_ivqr_design(x_controls: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, slice]:
    """Build the profiled QR design [1, selected controls, excluded instruments]."""
    x_controls = validate_2d_array("x_controls", x_controls)
    z_2d = as_2d_instruments(z)
    if x_controls.shape[0] != z_2d.shape[0]:
        raise ValueError("x_controls and z must have the same number of rows")

    design = np.column_stack([np.ones(x_controls.shape[0]), x_controls, z_2d])
    z_start = 1 + x_controls.shape[1]
    z_stop = z_start + z_2d.shape[1]
    return design, slice(z_start, z_stop)


def wald_statistic(gamma_hat: np.ndarray, cov_gamma: np.ndarray) -> float:
    """Return the Wald statistic for the excluded-instrument coefficients.

    CH writes the statistic as ``W_n(a) = n * gamma_hat' A_hat gamma_hat``.
    Statsmodels ``cov_params()`` estimates ``Var(gamma_hat)``, which already
    contains the inverse sample-size scaling. Therefore
    ``gamma_hat' Var(gamma_hat)^(-1) gamma_hat`` is the CH Wald statistic;
    multiplying by ``n`` again would double-count the sample-size scaling.
    """
    gamma_hat = np.asarray(gamma_hat, dtype=float).reshape(-1)
    cov_gamma = np.atleast_2d(np.asarray(cov_gamma, dtype=float))
    if cov_gamma.shape != (gamma_hat.size, gamma_hat.size):
        raise ValueError("cov_gamma shape must match gamma_hat dimension")
    if not np.all(np.isfinite(gamma_hat)) or not np.all(np.isfinite(cov_gamma)):
        raise ValueError("gamma_hat and cov_gamma must be finite")

    statistic = float(gamma_hat @ np.linalg.pinv(cov_gamma) @ gamma_hat)
    if statistic < 0.0 and statistic > -1e-10:
        statistic = 0.0
    if not np.isfinite(statistic) or statistic < 0.0:
        raise ValueError("Wald statistic must be finite and nonnegative")
    return statistic


def evaluate_alpha_ch_ivqr(
    *,
    y: np.ndarray,
    d: np.ndarray,
    x_controls: np.ndarray,
    z: np.ndarray,
    alpha: float,
    tau: float,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    iteration_warning_policy: IterationWarningPolicy = "use_if_valid",
) -> AlphaEvaluation:
    """Evaluate a CH inverse-IVQR statistic at one structural alpha.

    For a candidate alpha, run the tau-quantile regression of Y - D*alpha on
    included controls and excluded instruments. At the true alpha, the IVQR
    restriction implies that the excluded-instrument coefficient should be zero.

    The production policy ``"use_if_valid"`` retains an iteration-limit fit
    only when all inferential validity checks pass.  Pass ``"reject"`` to
    reproduce the legacy behavior that rejects every iteration-limit fit.
    """
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    iteration_warning_policy = validate_iteration_warning_policy(
        iteration_warning_policy
    )
    validate_tau(tau)
    alpha = float(alpha)
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    y = validate_1d_array("y", y)
    d = validate_1d_array("d", d)
    x_controls = validate_2d_array("x_controls", x_controls)
    z_2d = as_2d_instruments(z)
    if not (len(y) == len(d) == x_controls.shape[0] == z_2d.shape[0]):
        raise ValueError("y, d, x_controls, and z must have consistent row counts")

    design, z_block = ch_ivqr_design(x_controls, z_2d)
    y_alpha = y - d * alpha
    dim_z = z_2d.shape[1]

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IterationLimitWarning)
            result = QuantReg(y_alpha, design).fit(q=tau, max_iter=max_iter)
        iteration_limit_warning = any(
            issubclass(warning.category, IterationLimitWarning) for warning in caught
        )
        if iteration_limit_warning and iteration_warning_policy == "reject":
            return _failed_alpha_evaluation(
                dim_z=dim_z,
                converged=False,
                warning_type="iteration_limit",
                failure_reason="QuantReg reached iteration limit",
            )
    except Exception as exc:  # noqa: BLE001 - QuantReg can fail in several ways.
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=False,
            warning_type=None,
            failure_reason=f"quantreg_fit_failed: {type(exc).__name__}: {exc}",
        )

    warning_type = "iteration_limit" if iteration_limit_warning else None
    converged = not iteration_limit_warning
    try:
        params = np.asarray(result.params, dtype=float)
    except Exception as exc:  # noqa: BLE001 - malformed result objects vary.
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason=f"params_unavailable: {type(exc).__name__}: {exc}",
        )
    if params.shape != (design.shape[1],):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="invalid_parameter_dimension",
        )
    if not np.all(np.isfinite(params)):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="nonfinite_parameters",
        )
    try:
        cov_params = np.asarray(result.cov_params(), dtype=float)
    except Exception as exc:  # noqa: BLE001 - covariance failures vary.
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason=f"covariance_unavailable: {type(exc).__name__}: {exc}",
        )
    if cov_params.shape != (design.shape[1], design.shape[1]):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="invalid_covariance_dimension",
        )
    if not np.all(np.isfinite(cov_params)):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="nonfinite_covariance",
        )

    gamma_hat = params[z_block]
    cov_gamma = cov_params[z_block, z_block]
    if gamma_hat.shape != (dim_z,):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="invalid_instrument_coefficient_dimension",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
        )
    if cov_gamma.shape != (dim_z, dim_z):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="invalid_instrument_covariance_dimension",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
        )
    try:
        covariance_rank = int(np.linalg.matrix_rank(cov_gamma))
        covariance_condition_number = float(np.linalg.cond(cov_gamma))
    except np.linalg.LinAlgError as exc:
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason=f"invalid_instrument_covariance: {exc}",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
        )
    if dim_z > 1 and covariance_rank < dim_z:
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="rank_deficient_instrument_covariance",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
            covariance_rank=covariance_rank,
            covariance_condition_number=covariance_condition_number,
        )
    if not np.allclose(cov_gamma, cov_gamma.T, rtol=1e-8, atol=_COVARIANCE_TOLERANCE):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="nonsymmetric_instrument_covariance",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
            covariance_rank=covariance_rank,
            covariance_condition_number=covariance_condition_number,
        )
    try:
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(cov_gamma)))
    except np.linalg.LinAlgError as exc:
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason=f"invalid_instrument_covariance: {exc}",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
            covariance_rank=covariance_rank,
            covariance_condition_number=covariance_condition_number,
        )
    if minimum_eigenvalue < -_COVARIANCE_TOLERANCE:
        reason = (
            "negative_instrument_variance"
            if dim_z == 1
            else "indefinite_instrument_covariance"
        )
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason=reason,
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
            covariance_rank=covariance_rank,
            covariance_condition_number=covariance_condition_number,
        )
    if dim_z == 1 and float(cov_gamma[0, 0]) <= _COVARIANCE_TOLERANCE:
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="zero_instrument_variance",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
            covariance_rank=covariance_rank,
            covariance_condition_number=covariance_condition_number,
        )
    try:
        statistic = wald_statistic(gamma_hat, cov_gamma)
    except ValueError as exc:
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason=f"invalid_wald_statistic: {exc}",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
            covariance_rank=covariance_rank,
            covariance_condition_number=covariance_condition_number,
        )

    return AlphaEvaluation(
        statistic=statistic,
        gamma_hat=np.asarray(gamma_hat, dtype=float),
        cov_gamma=np.atleast_2d(np.asarray(cov_gamma, dtype=float)),
        dim_z=dim_z,
        converged=converged,
        usable=True,
        warning_type=warning_type,
        failure_reason=None,
        message="ok" if converged else "usable despite QuantReg iteration limit",
        covariance_rank=covariance_rank,
        covariance_condition_number=covariance_condition_number,
    )


def failed_ch_ivqr_result(
    *,
    data: SimData,
    tau: float,
    estimator: str,
    message: str,
    runtime_seconds: float,
    alpha_grid_size: int | None,
    failed_alpha_count: int | None,
    selected_controls: int | None = None,
    runtime_diagnostics: RuntimeDiagnosticColumns | None = None,
    hard_failure_policy: HardFailurePolicy = "unresolved",
    usable_alpha_evaluations: int | None = None,
    unresolved_alpha_evaluations: int | None = None,
    cr_unresolved_alphas: str = "[]",
    grid_strategy: GridStrategy = "adaptive",
    grid_evaluation: AlphaGridEvaluation | None = None,
    adaptive_midpoint_probe: bool = DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    alpha_hat_grid: AlphaHatGrid = DEFAULT_ALPHA_HAT_GRID,
) -> EstimationResult:
    """Create a standard failed result for any CH-IVQR-style estimator."""
    if not np.isfinite(runtime_seconds) or runtime_seconds < 0:
        raise ValueError("runtime_seconds must be finite and nonnegative")
    if alpha_grid_size is not None and alpha_grid_size < 1:
        raise ValueError("alpha_grid_size must be at least 1 when provided")
    if failed_alpha_count is not None and failed_alpha_count < 0:
        raise ValueError("failed_alpha_count must be nonnegative when provided")
    if (
        alpha_grid_size is not None
        and failed_alpha_count is not None
        and failed_alpha_count > alpha_grid_size
    ):
        raise ValueError("failed_alpha_count must not exceed alpha_grid_size")
    grid_metadata = (
        default_grid_metadata(
            grid_strategy=grid_strategy,
            adaptive_midpoint_probe=adaptive_midpoint_probe,
            alpha_hat_grid=alpha_hat_grid,
        )
        if grid_evaluation is None
        else grid_metadata_kwargs(grid_evaluation)
    )
    return EstimationResult(
        estimator=estimator,
        alpha_hat=None,
        alpha_true=data.alpha_true,
        tau=tau,
        converged=False,
        failed=True,
        message=message,
        objective_value=None,
        at_grid_boundary=False,
        alpha_grid_size=alpha_grid_size,
        failed_alpha_count=failed_alpha_count,
        cr_lower=None,
        cr_upper=None,
        cr_length=None,
        cr_covers_true=None,
        cr_empty=True,
        cr_disconnected=False,
        selected_controls=selected_controls,
        runtime_seconds=runtime_seconds,
        cr_components=(),
        cr_n_blocks=0,
        hard_failure_policy=hard_failure_policy,
        cr_status=(
            "fully_unresolved"
            if unresolved_alpha_evaluations and usable_alpha_evaluations == 0
            else "not_applicable"
        ),
        cr_is_numerically_resolved=(False if unresolved_alpha_evaluations else None),
        cr_unresolved_count=unresolved_alpha_evaluations or 0,
        cr_unresolved_alphas=cr_unresolved_alphas,
        coverage_status=(
            "coverage_unresolved" if unresolved_alpha_evaluations else "unknown"
        ),
        point_estimate_status=(
            "fully_unresolved"
            if unresolved_alpha_evaluations and usable_alpha_evaluations == 0
            else "not_applicable"
        ),
        usable_alpha_evaluations=usable_alpha_evaluations,
        unresolved_alpha_evaluations=unresolved_alpha_evaluations,
        **grid_metadata,
        **(
            estimator_runtime_columns(estimator=estimator, total_sec=runtime_seconds)
            if runtime_diagnostics is None
            else runtime_diagnostics
        ),
    )


def estimate_ch_ivqr_controls(
    data: SimData,
    tau: float,
    x_controls: np.ndarray,
    estimator_name: str,
    alphas: np.ndarray | None = None,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    alpha_step: float = DEFAULT_ALPHA_STEP,
    confidence_level: float = 0.95,
    critical_value_multiplier: float = 1.0,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    iteration_warning_policy: IterationWarningPolicy = "use_if_valid",
    hard_failure_policy: HardFailurePolicy = "unresolved",
    grid_strategy: GridStrategy = "adaptive",
    refinement_tolerance: float = DEFAULT_REFINEMENT_TOLERANCE,
    max_refinement_depth: int = DEFAULT_MAX_REFINEMENT_DEPTH,
    max_alpha_evaluations: int = DEFAULT_MAX_ALPHA_EVALUATIONS,
    adaptive_midpoint_probe: bool = DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    alpha_hat_grid: AlphaHatGrid = DEFAULT_ALPHA_HAT_GRID,
    selected_controls: int | None = None,
) -> EstimationResult:
    """Estimate alpha by CH inverse-IVQR using a supplied control matrix.

    Valid iteration-limit fits are used in normal production operation.  Set
    ``iteration_warning_policy="reject"`` for legacy-result reproducibility.
    Genuine unusable fits are unresolved by default; set
    ``hard_failure_policy="legacy_reject"`` only to reproduce sentinel-based
    rejection from historical simulations. Production uses
    ``grid_strategy="adaptive"`` with a 0.025 boundary tolerance, depth limit
    10, and at most 201 alpha evaluations. ``"fixed"`` reproduces the legacy
    initial-grid estimator. Midpoint probing improves detection of hidden
    components without guaranteeing discovery of every disconnected region.
    Point estimation uses usable initial-grid evaluations by default; set
    ``alpha_hat_grid="all_evaluated"`` for the earlier adaptive behavior.
    """
    start = perf_counter()
    alpha_loop_sec = float("nan")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    iteration_warning_policy = validate_iteration_warning_policy(
        iteration_warning_policy
    )
    hard_failure_policy = validate_hard_failure_policy(hard_failure_policy)
    grid_strategy = validate_grid_strategy(grid_strategy)
    alpha_hat_grid = validate_alpha_hat_grid(alpha_hat_grid)
    critical_value_multiplier = validate_critical_value_multiplier(
        critical_value_multiplier
    )
    validate_tau(tau)
    y, d, z, original_x = validate_data_arrays(data.y, data.d, data.x, data.z)
    # original_x is validated only to ensure the SimData object is well-formed.
    # CH inverse-IVQR estimation uses the supplied x_controls matrix.
    _ = original_x
    x_controls = validate_2d_array("x_controls", x_controls)
    if x_controls.shape[0] != len(y):
        raise ValueError("x_controls must have the same number of rows as y")

    z_2d = as_2d_instruments(z)
    n_regressors = 1 + x_controls.shape[1] + z_2d.shape[1]
    if n_regressors >= len(y):
        return failed_ch_ivqr_result(
            data=data,
            tau=tau,
            estimator=estimator_name,
            message=(
                "CH-IVQR infeasible: intercept + controls + instruments must be "
                f"less than n. n={len(y)}, regressors={n_regressors}."
            ),
            runtime_seconds=perf_counter() - start,
            alpha_grid_size=None,
            failed_alpha_count=None,
            selected_controls=selected_controls,
            grid_strategy=grid_strategy,
            adaptive_midpoint_probe=adaptive_midpoint_probe,
            alpha_hat_grid=alpha_hat_grid,
        )

    if alphas is None:
        alphas = alpha_grid(alpha_min, alpha_max, alpha_step)
    else:
        alphas = validate_alpha_grid(alphas)

    critical = critical_value_chi_square(confidence_level, df=z_2d.shape[1])
    adjusted_critical = adjust_critical_value(critical, critical_value_multiplier)
    alpha_loop_start = perf_counter()
    grid_evaluation = evaluate_ch_alpha_grid(
        alphas,
        lambda alpha: evaluate_alpha_ch_ivqr(
            y=y,
            d=d,
            x_controls=x_controls,
            z=z_2d,
            alpha=alpha,
            tau=tau,
            max_iter=max_iter,
            iteration_warning_policy=iteration_warning_policy,
        ),
        critical_value=adjusted_critical,
        grid_strategy=grid_strategy,
        refinement_tolerance=refinement_tolerance,
        max_refinement_depth=max_refinement_depth,
        max_alpha_evaluations=max_alpha_evaluations,
        adaptive_midpoint_probe=adaptive_midpoint_probe,
        alpha_hat_grid=alpha_hat_grid,
    )
    alpha_loop_sec = perf_counter() - alpha_loop_start

    alphas = grid_evaluation.alphas
    statistics = np.array(
        [evaluation.statistic for evaluation in grid_evaluation.evaluations],
        dtype=float,
    )
    usable_flags = [evaluation.usable for evaluation in grid_evaluation.evaluations]

    usable = np.asarray(usable_flags, dtype=bool) & np.isfinite(statistics)
    num_failed = int(np.sum(~usable))
    inference_statistics = statistics.copy()
    inference_usable = usable.copy()
    if hard_failure_policy == "legacy_reject":
        inference_statistics, _ = sanitize_grid_statistics(statistics, usable)
        inference_usable = np.ones(len(alphas), dtype=bool)
    if num_failed == len(alphas):
        runtime_seconds = perf_counter() - start
        return failed_ch_ivqr_result(
            data=data,
            tau=tau,
            estimator=estimator_name,
            message=(
                "All alpha-grid evaluations failed; "
                f"failed_alpha_points={num_failed}/{len(alphas)}"
            ),
            runtime_seconds=runtime_seconds,
            alpha_grid_size=len(alphas),
            failed_alpha_count=num_failed,
            selected_controls=selected_controls,
            hard_failure_policy=hard_failure_policy,
            usable_alpha_evaluations=0,
            unresolved_alpha_evaluations=num_failed,
            cr_unresolved_alphas=json.dumps([float(value) for value in alphas]),
            runtime_diagnostics=estimator_runtime_columns(
                estimator=estimator_name,
                total_sec=runtime_seconds,
                alpha_loop_sec=alpha_loop_sec,
            ),
            grid_strategy=grid_strategy,
            grid_evaluation=grid_evaluation,
            adaptive_midpoint_probe=adaptive_midpoint_probe,
            alpha_hat_grid=alpha_hat_grid,
        )

    confidence_region_start = perf_counter()
    if alpha_hat_grid == "initial":
        point_indices = np.searchsorted(alphas, grid_evaluation.initial_alphas)
        point_alphas = alphas[point_indices]
        point_statistics = statistics[point_indices]
        point_usable = usable[point_indices]
    else:
        point_alphas = alphas
        point_statistics = statistics
        point_usable = usable
    point_estimate = argmin_grid_usable(
        point_alphas, point_statistics, point_usable
    )
    alpha_hat = point_estimate.alpha_hat
    min_statistic = point_estimate.statistic
    at_boundary = point_estimate.at_boundary
    masks = classify_alpha_grid(
        inference_statistics, inference_usable, adjusted_critical
    )
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=alphas,
        accepted_mask=masks.accepted,
        alpha_hat=alpha_hat,
        failed_alpha_count=num_failed,
        test_stats=np.where(inference_usable, inference_statistics, np.nan),
        critical_value=adjusted_critical,
        critical_value_nominal=critical,
        critical_value_multiplier=critical_value_multiplier,
        critical_value_adjusted=adjusted_critical,
    )
    region = invert_score_test(
        alphas=alphas,
        statistics=inference_statistics,
        critical_value=critical,
        critical_value_multiplier=critical_value_multiplier,
        alpha_true=data.alpha_true,
        statistic_reference=0.0,
        inversion_type="absolute",
        usable=inference_usable,
    )
    diagnostics = merge_region_and_grid_diagnostics(region, diagnostics)
    if grid_strategy == "adaptive":
        diagnostics["alpha_grid_step"] = None
    confidence_region_sec = perf_counter() - confidence_region_start
    runtime_seconds = perf_counter() - start

    return EstimationResult(
        estimator=estimator_name,
        alpha_hat=alpha_hat,
        alpha_true=data.alpha_true,
        tau=tau,
        converged=True,
        failed=False,
        message=f"ok; failed_alpha_points={num_failed}/{len(alphas)}",
        objective_value=min_statistic,
        at_grid_boundary=at_boundary,
        alpha_grid_size=len(alphas),
        failed_alpha_count=num_failed,
        cr_lower=diagnostics["cr_lower"],
        cr_upper=diagnostics["cr_upper"],
        cr_length=diagnostics["cr_length"],
        cr_covers_true=region.covers_true,
        cr_empty=diagnostics["cr_empty"],
        cr_disconnected=diagnostics["cr_disconnected"],
        cr_components=region.components,
        selected_controls=selected_controls,
        runtime_seconds=runtime_seconds,
        hard_failure_policy=hard_failure_policy,
        cr_status=region.status,
        cr_is_numerically_resolved=region.is_numerically_resolved,
        cr_unresolved_count=region.unresolved_count,
        cr_unresolved_alphas=json.dumps(region.unresolved_alphas),
        coverage_status=region.coverage_status,
        point_estimate_status=point_estimate.status,
        usable_alpha_evaluations=int(np.sum(usable)),
        unresolved_alpha_evaluations=num_failed,
        **grid_metadata_kwargs(grid_evaluation),
        **estimator_runtime_columns(
            estimator=estimator_name,
            total_sec=runtime_seconds,
            alpha_loop_sec=alpha_loop_sec,
            confidence_region_sec=confidence_region_sec,
        ),
        **estimation_result_diagnostic_kwargs(diagnostics),
    )


__all__ = [
    "AlphaInferenceStatus",
    "AlphaEvaluation",
    "AlphaGridEvaluation",
    "AlphaHatGrid",
    "ALPHA_HAT_GRIDS",
    "GridStrategy",
    "GRID_STRATEGIES",
    "DEFAULT_REFINEMENT_TOLERANCE",
    "DEFAULT_MAX_REFINEMENT_DEPTH",
    "DEFAULT_MAX_ALPHA_EVALUATIONS",
    "DEFAULT_ADAPTIVE_MIDPOINT_PROBE",
    "DEFAULT_ALPHA_HAT_GRID",
    "HardFailurePolicy",
    "HARD_FAILURE_POLICIES",
    "IterationWarningPolicy",
    "ITERATION_WARNING_POLICIES",
    "add_intercept",
    "as_2d_instruments",
    "ch_ivqr_design",
    "wald_statistic",
    "evaluate_alpha_ch_ivqr",
    "evaluate_ch_alpha_grid",
    "failed_ch_ivqr_result",
    "grid_metadata_kwargs",
    "default_grid_metadata",
    "estimate_ch_ivqr_controls",
    "validate_iteration_warning_policy",
    "validate_hard_failure_policy",
    "validate_grid_strategy",
    "validate_alpha_hat_grid",
]
