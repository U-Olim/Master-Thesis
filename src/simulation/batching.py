"""Batch execution and resume helpers for simulation runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dgp.designs import Design
from simulation.runner import (
    DESIGN_KEY_COLUMNS,
    ESTIMATOR_OUTPUT_NAMES,
    RESULT_COLUMNS,
    _failure_rows_for_design,
    _validate_estimators,
    run_single_replication,
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


def run_simulation_batch(
    designs: list[Design],
    alphas: np.ndarray,
    estimators: tuple[str, ...] = ("full", "post_selection", "dml"),
    output_path: str | Path | None = None,
    append: bool = False,
    quantreg_max_iter: int = 500,
    selection_cv: int = 3,
    selection_max_iter: int = 10000,
    dml_k_folds: int = 5,
    dml_quantile_penalty: float = 0.01,
    dml_ridge_alpha: float = 1.0,
    dml_fold_random_state: int | None = 123,
    gmm_ridge: float = 1e-8,
) -> pd.DataFrame:
    """Run a batch of simulation designs and optionally persist it to CSV."""
    _validate_estimators(estimators)
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("alphas must be a nonempty one-dimensional array")

    rows: list[dict[str, object]] = []
    for design in designs:
        try:
            rows.extend(
                run_single_replication(
                    design,
                    alphas,
                    estimators=estimators,
                    quantreg_max_iter=quantreg_max_iter,
                    selection_cv=selection_cv,
                    selection_max_iter=selection_max_iter,
                    dml_k_folds=dml_k_folds,
                    dml_quantile_penalty=dml_quantile_penalty,
                    dml_ridge_alpha=dml_ridge_alpha,
                    dml_fold_random_state=dml_fold_random_state,
                    gmm_ridge=gmm_ridge,
                )
            )
        except Exception as exc:
            rows.extend(_failure_rows_for_design(design, estimators, alphas, exc))

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
