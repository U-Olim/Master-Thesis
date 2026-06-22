"""Batch execution and resume helpers for simulation runs."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dgp.designs import Design
from simulation.config import DEFAULT_N_JOBS
from simulation.runner import (
    DESIGN_KEY_COLUMNS,
    ESTIMATOR_OUTPUT_NAMES,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_QUANTREG_MAX_ITER,
    RESULT_COLUMNS,
    _failure_rows_for_design,
    _validate_estimators,
    run_single_replication,
)


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
        show_quantreg_warnings=args.show_quantreg_warnings,
    )


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _design_key(design: Design) -> tuple[object, ...]:
    return (
        design.dgp,
        design.n,
        design.p,
        design.pi,
        design.tau,
        design.rep,
        design.seed,
    )


def _row_design_key(row: pd.Series) -> tuple[object, ...]:
    return (
        row["dgp"],
        int(row["n"]),
        int(row["p"]),
        float(row["pi"]),
        float(row["tau"]),
        int(row["rep"]),
        int(row["seed"]),
    )


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
    estimators: tuple[str, ...] = ("oracle", "post_selection", "dml"),
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
    n_jobs: int = DEFAULT_N_JOBS,
    show_quantreg_warnings: bool = False,
) -> pd.DataFrame:
    """Run a batch of simulation designs and optionally persist it to CSV."""
    _validate_estimators(estimators)
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1")
    if quantreg_max_iter <= 0:
        raise ValueError("quantreg_max_iter must be positive")
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("alphas must be a nonempty one-dimensional array")

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
                    _failure_rows_for_design(args.design, estimators, alphas, exc)
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
                    rows.extend(_failure_rows_for_design(design, estimators, alphas, exc))

    if n_jobs > 1:
        rows.sort(key=_row_sort_key)
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if output_path is not None:
        path = Path(output_path)
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

    try:
        existing = pd.read_csv(path, usecols=DESIGN_KEY_COLUMNS)
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
    _validate_estimators(estimators)
    path = Path(results_path)
    if not path.exists():
        return designs

    required_columns = DESIGN_KEY_COLUMNS + ["estimator"]
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
