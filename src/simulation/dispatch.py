"""Estimator-name normalization and estimator dispatch."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import contextmanager
from typing import cast, get_args
import warnings

import numpy as np
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from dgp.designs import Design, SimData
from dgp.true_parameters import true_active_control_indices
from estimators.base import EstimationResult
from estimators.dml import QuantileSolver, estimate_dml_ivqr
from estimators.oracle import estimate_oracle_ivqr
from estimators.post_selection import (
    estimate_post_selection_ivqr,
)
from simulation.config import DEFAULT_ESTIMATORS, ESTIMATORS


VALID_ESTIMATORS: tuple[str, ...] = ESTIMATORS
DEFAULT_SIMULATION_ESTIMATORS: tuple[str, ...] = DEFAULT_ESTIMATORS
ESTIMATOR_OUTPUT_NAMES: dict[str, str] = {
    "oracle": "oracle",
    "post_selection": "post_selection_ivqr",
    "dml": "dml_ivqr",
}
ESTIMATOR_ALIASES: dict[str, str] = {
    "oracle": "oracle",
    "oracle_ivqr": "oracle",
    "post_selection": "post_selection",
    "post_selection_ivqr": "post_selection",
    "post-selection": "post_selection",
    "post-selection-ivqr": "post_selection",
    "post_selection-ivqr": "post_selection",
    "dml": "dml",
    "dml_ivqr": "dml",
    "dml-ivqr": "dml",
}
MULTI_ESTIMATOR_REMOVAL_MESSAGE = """Multi-estimator full mode has been removed. Run each estimator separately with:
  scenarios/run_oracle_ivqr.py
  scenarios/run_post_selection_ivqr.py
  scenarios/run_dml_ivqr.py"""


@contextmanager
def quantreg_iteration_warning_filter(show_warnings: bool = False):
    if show_warnings:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=IterationLimitWarning)
        warnings.filterwarnings(
            "ignore", message=r"Maximum number of iterations reached.*"
        )
        yield


def normalize_estimator_names(
    raw_estimators: Sequence[str] | None,
) -> tuple[str, ...]:
    if raw_estimators is None:
        return DEFAULT_SIMULATION_ESTIMATORS
    if isinstance(raw_estimators, str):
        raise ValueError("estimators must be a sequence of estimator names")
    normalized: list[str] = []
    invalid: list[str] = []
    for raw in raw_estimators:
        if not isinstance(raw, str) or not raw.strip():
            invalid.append(str(raw))
            continue
        token = raw.strip().lower().replace(" ", "_")
        canonical = ESTIMATOR_ALIASES.get(token)
        if canonical is None:
            invalid.append(raw)
            continue
        if canonical not in normalized:
            normalized.append(canonical)
    if invalid:
        valid = ", ".join(VALID_ESTIMATORS)
        aliases = ", ".join(sorted(ESTIMATOR_ALIASES))
        raise ValueError(
            f"Unknown estimator name(s): {invalid}. Valid estimators: {valid}. "
            f"Supported aliases: {aliases}."
        )
    if not normalized:
        raise ValueError("estimators must contain at least one estimator name")
    return tuple(normalized)


def validate_estimators(estimators: Sequence[str]) -> tuple[str, ...]:
    estimators = normalize_estimator_names(estimators)
    invalid = sorted(set(estimators) - set(VALID_ESTIMATORS))
    if invalid:
        valid = ", ".join(VALID_ESTIMATORS)
        raise ValueError(f"Unknown estimator(s): {invalid}. Valid estimators: {valid}")
    if len(estimators) != 1:
        raise ValueError(MULTI_ESTIMATOR_REMOVAL_MESSAGE)
    return estimators


def validate_oracle_support(
    design: Design, supplied_indices: np.ndarray
) -> np.ndarray:
    supplied = np.asarray(supplied_indices)
    design_id = (
        f"dgp={design.dgp}, n={design.n}, p={design.p}, pi={design.pi}, "
        f"tau={design.tau}, rep={design.rep}, seed={design.seed}"
    )
    if supplied.ndim != 1 or not np.issubdtype(supplied.dtype, np.integer):
        raise ValueError(
            "Oracle support must be a one-dimensional integer vector; "
            f"{design_id}"
        )
    supplied = supplied.astype(int, copy=False)
    if np.unique(supplied).size != supplied.size:
        raise ValueError(f"Oracle support contains duplicate indices; {design_id}")
    expected = true_active_control_indices(design.dgp, design.p)
    supplied_sorted = np.sort(supplied)
    if not np.array_equal(supplied_sorted, expected):
        raise ValueError(
            f"Oracle support mismatch; {design_id}; expected={expected.tolist()}, "
            f"supplied={supplied_sorted.tolist()}"
        )
    return supplied_sorted


def _validate_dml_quantile_solver(solver: str) -> QuantileSolver:
    if solver not in get_args(QuantileSolver):
        raise ValueError(f"Unknown quantile solver: {solver}")
    return cast(QuantileSolver, solver)


def run_estimator(
    estimator_name: str,
    data: SimData,
    design: Design,
    alphas: np.ndarray,
    *,
    quantreg_max_iter: int,
    dml_k_folds: int,
    dml_quantile_penalty: float,
    dml_ridge_alpha: float,
    dml_quantile_solver: str,
    gmm_ridge: float,
    critical_value_multiplier: float,
    selection_lasso_multiplier: float,
    grid_strategy: str,
    refinement_tolerance: float,
    max_refinement_depth: int,
    max_alpha_evaluations: int,
    iteration_warning_policy: str,
    hard_failure_policy: str,
    adaptive_midpoint_probe: bool,
    alpha_hat_grid: str,
    oracle_estimator: Callable[..., EstimationResult] | None = None,
    post_selection_estimator: Callable[..., EstimationResult] | None = None,
    dml_estimator: Callable[..., EstimationResult] | None = None,
) -> EstimationResult:
    oracle_estimator = oracle_estimator or estimate_oracle_ivqr
    post_selection_estimator = (
        post_selection_estimator or estimate_post_selection_ivqr
    )
    dml_estimator = dml_estimator or estimate_dml_ivqr
    random_state = int(design.seed % (2**32 - 1))
    if estimator_name == "oracle":
        oracle_indices = validate_oracle_support(
            design, true_active_control_indices(design.dgp, design.p)
        )
        return oracle_estimator(
            data, tau=design.tau, alphas=alphas, oracle_indices=oracle_indices,
            max_iter=quantreg_max_iter, gmm_ridge=gmm_ridge,
            critical_value_multiplier=critical_value_multiplier,
            grid_strategy=grid_strategy, refinement_tolerance=refinement_tolerance,
            max_refinement_depth=max_refinement_depth,
            max_alpha_evaluations=max_alpha_evaluations,
            iteration_warning_policy=iteration_warning_policy,
            hard_failure_policy=hard_failure_policy,
            adaptive_midpoint_probe=adaptive_midpoint_probe,
            alpha_hat_grid=alpha_hat_grid,
        )
    if estimator_name == "post_selection":
        return post_selection_estimator(
            data, tau=design.tau, alphas=alphas, selection_cv=3,
            selection_max_iter=10000, quantreg_max_iter=quantreg_max_iter,
            selection_random_state=random_state,
            selection_lasso_multiplier=selection_lasso_multiplier,
            critical_value_multiplier=critical_value_multiplier,
            grid_strategy=grid_strategy, refinement_tolerance=refinement_tolerance,
            max_refinement_depth=max_refinement_depth,
            max_alpha_evaluations=max_alpha_evaluations,
            iteration_warning_policy=iteration_warning_policy,
            hard_failure_policy=hard_failure_policy,
            adaptive_midpoint_probe=adaptive_midpoint_probe,
            alpha_hat_grid=alpha_hat_grid,
        )
    if estimator_name == "dml":
        return dml_estimator(
            data, tau=design.tau, alphas=alphas, k_folds=dml_k_folds,
            fold_random_state=random_state,
            quantile_penalty=dml_quantile_penalty, ridge_alpha=dml_ridge_alpha,
            quantile_solver=_validate_dml_quantile_solver(dml_quantile_solver),
            gmm_ridge=gmm_ridge,
            critical_value_multiplier=critical_value_multiplier,
        )
    raise ValueError(f"Unknown estimator: {estimator_name}")


__all__ = [
    "DEFAULT_SIMULATION_ESTIMATORS", "ESTIMATOR_ALIASES",
    "ESTIMATOR_OUTPUT_NAMES", "MULTI_ESTIMATOR_REMOVAL_MESSAGE",
    "VALID_ESTIMATORS", "normalize_estimator_names",
    "quantreg_iteration_warning_filter", "run_estimator",
    "validate_estimators", "validate_oracle_support",
]
