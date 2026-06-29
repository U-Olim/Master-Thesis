"""Batch execution and resume helpers for simulation runs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dgp.designs import Design
from simulation._validation import (
    design_key,
    parse_explicit_bool,
    row_design_key,
    validate_alpha_candidates,
    validate_bool,
    validate_designs,
    validate_estimators,
    validate_k_folds_for_designs,
    validate_nonnegative_float,
    validate_optional_nonnegative_int,
    validate_positive_float,
    validate_output_file_path,
    validate_positive_int,
)
from simulation.config import DEFAULT_CRITICAL_VALUE_MULTIPLIER, DEFAULT_N_JOBS
from simulation.runner import (
    DEFAULT_SIMULATION_ESTIMATORS,
    DESIGN_KEY_COLUMNS,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_QUANTREG_MAX_ITER,
    ESTIMATOR_OUTPUT_NAMES,
    RESULT_COLUMNS,
    failure_rows_for_design,
    run_single_replication,
)


__all__ = [
    "SimulationWorkerArgs",
    "completed_design_keys",
    "filter_completed_designs",
    "observed_design_keys",
    "run_simulation_batch",
]


@dataclass(frozen=True)
class SimulationWorkerArgs:
    design: Design
    alphas: np.ndarray
    estimators: tuple[str, ...]
    quantreg_max_iter: int
    selection_cv: int
    selection_max_iter: int
    dml_k_folds: int
    dml_quantile_penalty: float
    dml_ridge_alpha: float
    dml_fold_random_state: int | None
    gmm_ridge: float
    critical_value_multiplier: float
    show_quantreg_warnings: bool


def _run_design_worker(args: SimulationWorkerArgs) -> list[dict[str, object]]:
    """Run one independent simulation design in a worker process."""
    return run_single_replication(
        args.design,
        args.alphas,
        estimators=args.estimators,
        quantreg_max_iter=args.quantreg_max_iter,
        selection_cv=args.selection_cv,
        selection_max_iter=args.selection_max_iter,
        dml_k_folds=args.dml_k_folds,
        dml_quantile_penalty=args.dml_quantile_penalty,
        dml_ridge_alpha=args.dml_ridge_alpha,
        dml_fold_random_state=args.dml_fold_random_state,
        gmm_ridge=args.gmm_ridge,
        critical_value_multiplier=args.critical_value_multiplier,
        show_quantreg_warnings=args.show_quantreg_warnings,
    )


def _as_bool(value: object) -> bool:
    return parse_explicit_bool(value)


def _design_key(design: Design) -> tuple[object, ...]:
    return design_key(design)


def _row_design_key(row: pd.Series) -> tuple[object, ...]:
    return row_design_key(row)


def _row_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["dgp"],
        row["n"],
        row["p"],
        row["pi"],
        row["tau"],
        row["rep"],
        row["seed"],
        row["estimator"],
    )


def run_simulation_batch(
    designs: list[Design],
    alphas: np.ndarray,
    estimators: tuple[str, ...] = DEFAULT_SIMULATION_ESTIMATORS,
    output_path: str | Path | None = None,
    append: bool = False,
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    selection_cv: int = 3,
    selection_max_iter: int = 10000,
    dml_k_folds: int = DEFAULT_DML_K_FOLDS,
    dml_quantile_penalty: float = 0.01,
    dml_ridge_alpha: float = 1.0,
    dml_fold_random_state: int | None = None,
    gmm_ridge: float = 1e-8,
    critical_value_multiplier: float = DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    n_jobs: int = DEFAULT_N_JOBS,
    show_quantreg_warnings: bool = False,
) -> pd.DataFrame:
    """Run a batch of simulation designs and optionally persist it to CSV."""
    designs = validate_designs(designs)
    estimators = validate_estimators(estimators)
    n_jobs = validate_positive_int("n_jobs", n_jobs)
    quantreg_max_iter = validate_positive_int("quantreg_max_iter", quantreg_max_iter)
    selection_cv = validate_positive_int("selection_cv", selection_cv)
    selection_max_iter = validate_positive_int("selection_max_iter", selection_max_iter)
    dml_k_folds = validate_k_folds_for_designs(dml_k_folds, designs)
    dml_quantile_penalty = validate_nonnegative_float(
        "dml_quantile_penalty", dml_quantile_penalty
    )
    dml_ridge_alpha = validate_nonnegative_float("dml_ridge_alpha", dml_ridge_alpha)
    gmm_ridge = validate_nonnegative_float("gmm_ridge", gmm_ridge)
    critical_value_multiplier = validate_positive_float(
        "critical_value_multiplier",
        critical_value_multiplier,
    )
    append = validate_bool("append", append)
    show_quantreg_warnings = validate_bool(
        "show_quantreg_warnings", show_quantreg_warnings
    )
    dml_fold_random_state = validate_optional_nonnegative_int(
        "dml_fold_random_state", dml_fold_random_state
    )
    alphas = validate_alpha_candidates(alphas)
    path = (
        validate_output_file_path(output_path)
        if output_path is not None
        else None
    )

    rows: list[dict[str, object]] = []
    worker_args = [
        SimulationWorkerArgs(
            design=design,
            alphas=alphas,
            estimators=estimators,
            quantreg_max_iter=quantreg_max_iter,
            selection_cv=selection_cv,
            selection_max_iter=selection_max_iter,
            dml_k_folds=dml_k_folds,
            dml_quantile_penalty=dml_quantile_penalty,
            dml_ridge_alpha=dml_ridge_alpha,
            dml_fold_random_state=dml_fold_random_state,
            gmm_ridge=gmm_ridge,
            critical_value_multiplier=critical_value_multiplier,
            show_quantreg_warnings=show_quantreg_warnings,
        )
        for design in designs
    ]

    if n_jobs == 1 or len(worker_args) <= 1:
        for args in worker_args:
            try:
                rows.extend(_run_design_worker(args))
            except Exception as exc:
                rows.extend(
                    failure_rows_for_design(
                        args.design,
                        estimators,
                        alphas,
                        exc,
                        critical_value_multiplier=critical_value_multiplier,
                    )
                )
    else:
        max_workers = min(n_jobs, len(worker_args))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_design_worker, args): args.design
                for args in worker_args
            }
            for future in as_completed(futures):
                design = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:
                    rows.extend(
                        failure_rows_for_design(
                            design,
                            estimators,
                            alphas,
                            exc,
                            critical_value_multiplier=critical_value_multiplier,
                        )
                    )

    if n_jobs > 1:
        rows.sort(key=_row_sort_key)
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not (append and path.exists())
        results.to_csv(
            path, mode="a" if append else "w", header=write_header, index=False
        )
    return results


def observed_design_keys(results_path: str | Path) -> set[tuple[object, ...]]:
    """Return design keys with at least one persisted result row."""
    path = Path(results_path)
    if not path.exists():
        return set()
    if path.is_dir():
        raise ValueError("results_path must be a file")

    try:
        existing = pd.read_csv(path, usecols=list(DESIGN_KEY_COLUMNS))
    except pd.errors.EmptyDataError as exc:
        raise ValueError("results CSV is empty or malformed") from exc
    except ValueError as exc:
        raise ValueError("results CSV is missing required design-key columns") from exc

    return {_row_design_key(row) for _, row in existing.drop_duplicates().iterrows()}


def completed_design_keys(results_path: str | Path) -> set[tuple[object, ...]]:
    """Deprecated alias for observed_design_keys."""
    return observed_design_keys(results_path)


def filter_completed_designs(
    designs: list[Design],
    results_path: str | Path,
    estimators: tuple[str, ...],
    rerun_failed: bool = False,
) -> list[Design]:
    """Return designs that do not yet have all requested estimator rows."""
    designs = validate_designs(designs)
    estimators = validate_estimators(estimators)
    rerun_failed = validate_bool("rerun_failed", rerun_failed)
    path = Path(results_path)
    if not path.exists():
        return designs
    if path.is_dir():
        raise ValueError("results_path must be a file")

    required_columns = list(DESIGN_KEY_COLUMNS) + ["estimator"]
    if rerun_failed:
        required_columns += ["failed"]
    try:
        existing = pd.read_csv(path, usecols=required_columns)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("results CSV is empty or malformed") from exc
    except ValueError as exc:
        raise ValueError("results CSV is missing required resume columns") from exc

    expected_estimators = {
        ESTIMATOR_OUTPUT_NAMES[estimator] for estimator in estimators
    }
    completed: dict[tuple[object, ...], set[str]] = {}
    for _, row in existing.iterrows():
        if rerun_failed and _as_bool(row["failed"]):
            continue
        key = _row_design_key(row)
        completed.setdefault(key, set()).add(str(row["estimator"]))

    return [
        design
        for design in designs
        if not expected_estimators.issubset(completed.get(_design_key(design), set()))
    ]
