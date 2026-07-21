"""Single-design, serial, and parallel simulation execution."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dgp.designs import Design
from dgp.generators import generate_data
from dgp.true_parameters import true_alpha
from estimators.base import EstimationResult
from estimators.post_selection import validate_selection_lasso_multiplier
from ivqr.ch_inverse import (
    validate_alpha_hat_grid,
    validate_grid_strategy,
    validate_hard_failure_policy,
    validate_iteration_warning_policy,
)
from simulation.config import (
    DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    DEFAULT_ALPHA_HAT_GRID,
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
    DEFAULT_N_JOBS,
    DEFAULT_QUANTREG_MAX_ITER,
    DEFAULT_REFINEMENT_TOLERANCE,
)
from simulation.designs import DESIGN_KEY_COLUMNS, validate_design
from simulation.dispatch import (
    DEFAULT_SIMULATION_ESTIMATORS,
    ESTIMATOR_OUTPUT_NAMES,
    quantreg_iteration_warning_filter,
    run_estimator,
    validate_estimators,
)
from simulation.persistence import persist_results_frame, prepare_results_frame
from simulation.results import (
    MAX_ERROR_MESSAGE_LENGTH,
    RESULT_COLUMNS,
    build_failure_result_row,
    build_simulation_result_row,
)
from simulation.seeds import (
    validate_nonnegative_float,
    validate_positive_float,
)
from utils.validation import validate_alpha_grid, validate_positive_int


RESULT_SORT_COLUMNS: tuple[str, ...] = (*DESIGN_KEY_COLUMNS, "estimator")


@dataclass(frozen=True)
class WorkerArgs:
    design: Design
    alphas: np.ndarray
    estimators: tuple[str, ...]
    quantreg_max_iter: int
    dml_k_folds: int
    dml_quantile_penalty: float
    dml_ridge_alpha: float
    dml_quantile_solver: str
    gmm_ridge: float
    critical_value_multiplier: float
    selection_lasso_multiplier: float
    show_quantreg_warnings: bool
    grid_strategy: str
    refinement_tolerance: float
    max_refinement_depth: int
    max_alpha_evaluations: int
    iteration_warning_policy: str
    hard_failure_policy: str
    adaptive_midpoint_probe: bool
    alpha_hat_grid: str


def _result_to_row(
    design: Design,
    result: EstimationResult,
    alphas: np.ndarray,
    critical_value_multiplier: float,
) -> dict[str, object]:
    row = build_simulation_result_row(design, result, alphas)
    if pd.isna(row["critical_value_multiplier"]):
        row["critical_value_multiplier"] = critical_value_multiplier
    return row


def _failure_row_for_estimator(
    design: Design,
    estimator: str,
    alphas: np.ndarray,
    exc: Exception,
    critical_value_multiplier: float,
) -> dict[str, object]:
    try:
        alpha_true = true_alpha(design.tau, design.dgp)
    except Exception:
        alpha_true = None
    message = (
        f"Unexpected estimator error: {type(exc).__name__}: "
        f"{str(exc)[:MAX_ERROR_MESSAGE_LENGTH]}"
    )
    return build_failure_result_row(
        design=design,
        estimator=ESTIMATOR_OUTPUT_NAMES[estimator],
        alphas=alphas,
        alpha_true=alpha_true,
        exc=exc,
        message=message,
        critical_value_multiplier=critical_value_multiplier,
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
    *,
    _oracle_estimator: Callable[..., EstimationResult] | None = None,
    _post_selection_estimator: Callable[..., EstimationResult] | None = None,
    _dml_estimator: Callable[..., EstimationResult] | None = None,
) -> list[dict[str, object]]:
    """Generate one dataset and run exactly one requested estimator on it."""
    design = validate_design(design)
    estimators = validate_estimators(estimators)
    quantreg_max_iter = validate_positive_int("quantreg_max_iter", quantreg_max_iter)
    dml_k_folds = validate_positive_int("dml_k_folds", dml_k_folds)
    if dml_k_folds < 2 or dml_k_folds > design.n:
        raise ValueError("dml_k_folds must satisfy 2 <= dml_k_folds <= n")
    dml_quantile_penalty = validate_nonnegative_float(
        "dml_quantile_penalty", dml_quantile_penalty
    )
    dml_ridge_alpha = validate_nonnegative_float(
        "dml_ridge_alpha", dml_ridge_alpha
    )
    dml_quantile_solver = str(dml_quantile_solver)
    gmm_ridge = validate_nonnegative_float("gmm_ridge", gmm_ridge)
    critical_value_multiplier = validate_positive_float(
        "critical_value_multiplier", critical_value_multiplier
    )
    selection_lasso_multiplier = validate_selection_lasso_multiplier(
        selection_lasso_multiplier
    )
    alphas = validate_alpha_grid(alphas)
    grid_strategy = validate_grid_strategy(grid_strategy)
    iteration_warning_policy = validate_iteration_warning_policy(
        iteration_warning_policy
    )
    hard_failure_policy = validate_hard_failure_policy(hard_failure_policy)
    alpha_hat_grid = validate_alpha_hat_grid(alpha_hat_grid)

    data = generate_data(design)
    rows: list[dict[str, object]] = []
    for estimator_name in estimators:
        try:
            with quantreg_iteration_warning_filter(show_quantreg_warnings):
                result = run_estimator(
                    estimator_name,
                    data,
                    design,
                    alphas,
                    quantreg_max_iter=quantreg_max_iter,
                    dml_k_folds=dml_k_folds,
                    dml_quantile_penalty=dml_quantile_penalty,
                    dml_ridge_alpha=dml_ridge_alpha,
                    dml_quantile_solver=dml_quantile_solver,
                    gmm_ridge=gmm_ridge,
                    critical_value_multiplier=critical_value_multiplier,
                    selection_lasso_multiplier=selection_lasso_multiplier,
                    grid_strategy=grid_strategy,
                    refinement_tolerance=refinement_tolerance,
                    max_refinement_depth=max_refinement_depth,
                    max_alpha_evaluations=max_alpha_evaluations,
                    iteration_warning_policy=iteration_warning_policy,
                    hard_failure_policy=hard_failure_policy,
                    adaptive_midpoint_probe=adaptive_midpoint_probe,
                    alpha_hat_grid=alpha_hat_grid,
                    oracle_estimator=_oracle_estimator,
                    post_selection_estimator=_post_selection_estimator,
                    dml_estimator=_dml_estimator,
                )
            rows.append(
                _result_to_row(
                    design, result, alphas, critical_value_multiplier
                )
            )
        except Exception as exc:  # noqa: BLE001 - record failed replications.
            rows.append(
                _failure_row_for_estimator(
                    design,
                    estimator_name,
                    alphas,
                    exc,
                    critical_value_multiplier,
                )
            )
    return rows


def run_single_replication(*args, **kwargs) -> list[dict[str, object]]:
    """Backward-compatible alias for run_simulation_design."""
    return run_simulation_design(*args, **kwargs)


def _run_worker(args: WorkerArgs) -> list[dict[str, object]]:
    return run_simulation_design(
        args.design,
        args.alphas,
        estimators=args.estimators,
        quantreg_max_iter=args.quantreg_max_iter,
        dml_k_folds=args.dml_k_folds,
        dml_quantile_penalty=args.dml_quantile_penalty,
        dml_ridge_alpha=args.dml_ridge_alpha,
        dml_quantile_solver=args.dml_quantile_solver,
        gmm_ridge=args.gmm_ridge,
        critical_value_multiplier=args.critical_value_multiplier,
        selection_lasso_multiplier=args.selection_lasso_multiplier,
        show_quantreg_warnings=args.show_quantreg_warnings,
        grid_strategy=args.grid_strategy,
        refinement_tolerance=args.refinement_tolerance,
        max_refinement_depth=args.max_refinement_depth,
        max_alpha_evaluations=args.max_alpha_evaluations,
        iteration_warning_policy=args.iteration_warning_policy,
        hard_failure_policy=args.hard_failure_policy,
        adaptive_midpoint_probe=args.adaptive_midpoint_probe,
        alpha_hat_grid=args.alpha_hat_grid,
    )


def _row_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row[column] for column in RESULT_SORT_COLUMNS)


def run_simulation_batch(
    designs: list[Design],
    alphas: np.ndarray,
    estimators: tuple[str, ...] = DEFAULT_SIMULATION_ESTIMATORS,
    output_path: str | Path | None = None,
    append: bool = False,
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    dml_k_folds: int = DEFAULT_DML_K_FOLDS,
    dml_quantile_penalty: float = DEFAULT_DML_QUANTILE_PENALTY,
    dml_ridge_alpha: float = DEFAULT_DML_RIDGE_ALPHA,
    dml_quantile_solver: str = DEFAULT_DML_QUANTILE_SOLVER,
    gmm_ridge: float = 1e-8,
    critical_value_multiplier: float = DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    selection_lasso_multiplier: float = 1.0,
    n_jobs: int = DEFAULT_N_JOBS,
    show_quantreg_warnings: bool = False,
    grid_strategy: str = DEFAULT_GRID_STRATEGY,
    refinement_tolerance: float = DEFAULT_REFINEMENT_TOLERANCE,
    max_refinement_depth: int = DEFAULT_MAX_REFINEMENT_DEPTH,
    max_alpha_evaluations: int = DEFAULT_MAX_ALPHA_EVALUATIONS,
    iteration_warning_policy: str = DEFAULT_ITERATION_WARNING_POLICY,
    hard_failure_policy: str = DEFAULT_HARD_FAILURE_POLICY,
    adaptive_midpoint_probe: bool = DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    alpha_hat_grid: str = DEFAULT_ALPHA_HAT_GRID,
) -> pd.DataFrame:
    """Run a batch of designs and optionally persist projected rows to CSV."""
    designs = [validate_design(design) for design in designs]
    estimators = validate_estimators(estimators)
    alphas = validate_alpha_grid(alphas)
    n_jobs = validate_positive_int("n_jobs", n_jobs)
    selection_lasso_multiplier = validate_selection_lasso_multiplier(
        selection_lasso_multiplier
    )
    dml_quantile_penalty = validate_nonnegative_float(
        "dml_quantile_penalty", dml_quantile_penalty
    )
    dml_ridge_alpha = validate_nonnegative_float(
        "dml_ridge_alpha", dml_ridge_alpha
    )
    dml_quantile_solver = str(dml_quantile_solver)
    worker_args = [
        WorkerArgs(
            design, alphas, estimators, quantreg_max_iter, dml_k_folds,
            dml_quantile_penalty, dml_ridge_alpha, dml_quantile_solver,
            gmm_ridge, critical_value_multiplier, selection_lasso_multiplier,
            show_quantreg_warnings, grid_strategy, refinement_tolerance,
            max_refinement_depth, max_alpha_evaluations,
            iteration_warning_policy, hard_failure_policy,
            adaptive_midpoint_probe, alpha_hat_grid,
        )
        for design in designs
    ]
    rows: list[dict[str, object]] = []
    if n_jobs == 1 or len(worker_args) <= 1:
        for args in worker_args:
            rows.extend(_run_worker(args))
    else:
        with ProcessPoolExecutor(
            max_workers=min(n_jobs, len(worker_args))
        ) as executor:
            futures = {
                executor.submit(_run_worker, args): args for args in worker_args
            }
            for future in as_completed(futures):
                rows.extend(future.result())
        rows.sort(key=_row_sort_key)
    results = prepare_results_frame(
        pd.DataFrame(rows, columns=RESULT_COLUMNS), estimators
    )
    if output_path is not None:
        persist_results_frame(
            results, output_path, append=append, estimators=estimators
        )
    return results


__all__ = [
    "RESULT_SORT_COLUMNS",
    "run_simulation_batch",
    "run_simulation_design",
    "run_single_replication",
]
