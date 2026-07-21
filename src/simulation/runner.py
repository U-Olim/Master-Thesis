"""Stable compatibility facade for simulation execution infrastructure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dgp.designs import Design
from estimators.dml import estimate_dml_ivqr
from estimators.oracle import estimate_oracle_ivqr
from estimators.post_selection import estimate_post_selection_ivqr
from simulation.config import (
    DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_ALPHA_HAT_GRID,
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_BASE_SEED,
    DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_DML_QUANTILE_PENALTY,
    DEFAULT_DML_QUANTILE_SOLVER,
    DEFAULT_DML_RIDGE_ALPHA,
    DEFAULT_GRID_STRATEGY,
    DEFAULT_HARD_FAILURE_POLICY,
    DEFAULT_ITERATION_WARNING_POLICY,
    DEFAULT_MAX_ALPHA_EVALUATIONS,
    DEFAULT_MAX_REFINEMENT_DEPTH,
    DEFAULT_QUANTREG_MAX_ITER,
    DEFAULT_REFINEMENT_TOLERANCE,
)
from simulation.designs import DESIGN_KEY_COLUMNS, make_simulation_grid
from simulation.dispatch import (
    DEFAULT_SIMULATION_ESTIMATORS,
    ESTIMATOR_ALIASES,
    ESTIMATOR_OUTPUT_NAMES,
    MULTI_ESTIMATOR_REMOVAL_MESSAGE,
    VALID_ESTIMATORS,
    normalize_estimator_names,
    quantreg_iteration_warning_filter,
    validate_oracle_support,
)
from simulation.execution import run_simulation_batch
from simulation.execution import run_simulation_design as _run_simulation_design
from simulation.resume import filter_completed_designs
from simulation.results import RESULT_COLUMNS
from simulation.seeds import SEED_RULE_TEXT, make_design_seed


VALID_DGPS: tuple[str, ...] = (
    "dgp1",
    "dgp2",
    "dgp3",
)


def run_simulation_design(
    design: Design,
    alphas: np.ndarray,
    estimators: tuple[str, ...] = DEFAULT_SIMULATION_ESTIMATORS,
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    dml_k_folds: int = DEFAULT_DML_K_FOLDS,
    dml_quantile_penalty: float = DEFAULT_DML_QUANTILE_PENALTY,
    dml_ridge_alpha: float = DEFAULT_DML_RIDGE_ALPHA,
    dml_quantile_solver: str = DEFAULT_DML_QUANTILE_SOLVER,
    gmm_ridge: float = 1e-8,
    critical_value_multiplier: float = DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    selection_lasso_multiplier: float = 1.0,
    show_quantreg_warnings: bool = False,
    grid_strategy: str = DEFAULT_GRID_STRATEGY,
    refinement_tolerance: float = DEFAULT_REFINEMENT_TOLERANCE,
    max_refinement_depth: int = DEFAULT_MAX_REFINEMENT_DEPTH,
    max_alpha_evaluations: int = DEFAULT_MAX_ALPHA_EVALUATIONS,
    iteration_warning_policy: str = DEFAULT_ITERATION_WARNING_POLICY,
    hard_failure_policy: str = DEFAULT_HARD_FAILURE_POLICY,
    adaptive_midpoint_probe: bool = DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    alpha_hat_grid: str = DEFAULT_ALPHA_HAT_GRID,
) -> list[dict[str, object]]:
    """Compatibility wrapper around the execution-module implementation."""
    return _run_simulation_design(
        design,
        alphas,
        estimators=estimators,
        quantreg_max_iter=quantreg_max_iter,
        dml_k_folds=dml_k_folds,
        dml_quantile_penalty=dml_quantile_penalty,
        dml_ridge_alpha=dml_ridge_alpha,
        dml_quantile_solver=dml_quantile_solver,
        gmm_ridge=gmm_ridge,
        critical_value_multiplier=critical_value_multiplier,
        selection_lasso_multiplier=selection_lasso_multiplier,
        show_quantreg_warnings=show_quantreg_warnings,
        grid_strategy=grid_strategy,
        refinement_tolerance=refinement_tolerance,
        max_refinement_depth=max_refinement_depth,
        max_alpha_evaluations=max_alpha_evaluations,
        iteration_warning_policy=iteration_warning_policy,
        hard_failure_policy=hard_failure_policy,
        adaptive_midpoint_probe=adaptive_midpoint_probe,
        alpha_hat_grid=alpha_hat_grid,
        _oracle_estimator=estimate_oracle_ivqr,
        _post_selection_estimator=estimate_post_selection_ivqr,
        _dml_estimator=estimate_dml_ivqr,
    )


def run_single_replication(*args, **kwargs) -> list[dict[str, object]]:
    """Backward-compatible alias for run_simulation_design."""
    return run_simulation_design(*args, **kwargs)


def run_small_simulation(
    dgp: str = "dgp1",
    n: int = 80,
    p: int = 5,
    pi: float = 1.0,
    tau: float = 0.5,
    reps: int = 1,
    base_seed: int = DEFAULT_BASE_SEED,
    alphas: np.ndarray | None = None,
    estimators: tuple[str, ...] = DEFAULT_SIMULATION_ESTIMATORS,
    alpha_grid_size: int = DEFAULT_ALPHA_GRID_SIZE,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    **kwargs,
) -> pd.DataFrame:
    """Run a small simulation and return estimator-level rows."""
    if alphas is None:
        alphas = np.linspace(alpha_min, alpha_max, alpha_grid_size)
    designs = make_simulation_grid(
        dgps=(dgp,),
        n_values=(n,),
        p_values=(p,),
        pi_values=(pi,),
        taus=(tau,),
        reps=reps,
        base_seed=base_seed,
    )
    return run_simulation_batch(
        designs, alphas, estimators=estimators, n_jobs=1, **kwargs
    )


__all__ = [
    "DEFAULT_SIMULATION_ESTIMATORS",
    "DESIGN_KEY_COLUMNS",
    "ESTIMATOR_ALIASES",
    "ESTIMATOR_OUTPUT_NAMES",
    "MULTI_ESTIMATOR_REMOVAL_MESSAGE",
    "RESULT_COLUMNS",
    "SEED_RULE_TEXT",
    "VALID_DGPS",
    "VALID_ESTIMATORS",
    "filter_completed_designs",
    "make_design_seed",
    "make_simulation_grid",
    "normalize_estimator_names",
    "quantreg_iteration_warning_filter",
    "run_simulation_batch",
    "run_simulation_design",
    "run_single_replication",
    "run_small_simulation",
    "validate_oracle_support",
]
