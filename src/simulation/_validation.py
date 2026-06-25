"""Shared validation helpers for simulation runners and batching."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dgp.designs import Design
from simulation.config import DGPS, MAIN_ESTIMATORS
from utils.validation import validate_alpha_grid


def validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_nonnegative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def validate_nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def validate_finite_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_probability_quantile(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must satisfy 0 < {name} < 1")
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must satisfy 0 < {name} < 1")
    return value


def validate_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def validate_optional_nonnegative_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return validate_nonnegative_int(name, value)


def validate_alpha_candidates(alphas: Any) -> np.ndarray:
    return validate_alpha_grid(alphas)


def validate_estimators(
    estimators: Sequence[str],
    valid_estimators: Sequence[str] = MAIN_ESTIMATORS,
) -> tuple[str, ...]:
    if isinstance(estimators, str):
        raise ValueError("estimators must be a sequence of estimator names")
    try:
        estimators = tuple(estimators)
    except TypeError as exc:
        raise ValueError("estimators must be a sequence of estimator names") from exc

    if len(estimators) == 0:
        raise ValueError("estimators must contain at least one estimator name")
    if any(not isinstance(estimator, str) or not estimator for estimator in estimators):
        raise ValueError("estimators must contain nonempty strings")
    if len(set(estimators)) != len(estimators):
        raise ValueError("estimators must not contain duplicates")

    invalid = sorted(set(estimators) - set(valid_estimators))
    if invalid:
        valid = ", ".join(valid_estimators)
        raise ValueError(f"Unknown estimator(s): {invalid}. Valid estimators: {valid}")

    return estimators


def validate_dgps(
    dgps: Sequence[str],
    valid_dgps: Sequence[str] = DGPS,
) -> tuple[str, ...]:
    if isinstance(dgps, str):
        raise ValueError("dgps must be a sequence of DGP names")
    try:
        dgps = tuple(dgps)
    except TypeError as exc:
        raise ValueError("dgps must be a sequence of DGP names") from exc

    if len(dgps) == 0:
        raise ValueError("dgps must be nonempty")
    if any(not isinstance(dgp, str) or not dgp for dgp in dgps):
        raise ValueError("dgps must contain nonempty strings")
    if len(set(dgps)) != len(dgps):
        raise ValueError("dgps must not contain duplicates")

    invalid = sorted(set(dgps) - set(valid_dgps))
    if invalid:
        raise ValueError(f"Unknown DGP(s): {invalid}. Valid DGPs: {tuple(valid_dgps)}")

    return dgps


def validate_unique_sequence(name: str, values: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty sequence")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty sequence") from exc
    if not values:
        raise ValueError(f"{name} must be nonempty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def validate_design(design: Design) -> Design:
    if not isinstance(design, Design):
        raise ValueError("design must be a Design object")

    validate_dgps((design.dgp,))
    validate_positive_int("n", design.n)
    validate_positive_int("p", design.p)
    validate_nonnegative_float("pi", design.pi)
    validate_probability_quantile("tau", design.tau)
    validate_nonnegative_int("rep", design.rep)
    validate_nonnegative_int("seed", design.seed)
    return design


def validate_designs(designs: Iterable[Design]) -> list[Design]:
    if isinstance(designs, (str, bytes)):
        raise ValueError("designs must be an iterable of Design objects")
    try:
        designs = list(designs)
    except TypeError as exc:
        raise ValueError("designs must be an iterable of Design objects") from exc

    for design in designs:
        validate_design(design)

    return designs


def validate_k_folds_for_designs(k_folds: int, designs: Sequence[Design]) -> int:
    k_folds = validate_positive_int("dml_k_folds", k_folds)
    if k_folds < 2:
        raise ValueError("dml_k_folds must be at least 2")
    if designs:
        min_n = min(design.n for design in designs)
        if k_folds > min_n:
            raise ValueError("dml_k_folds must not exceed the smallest design n")
    return k_folds


def validate_output_file_path(path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise ValueError("output_path must be a file path")
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise ValueError("output_path parent must be a directory")
    return output_path


def parse_explicit_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            return False
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return False


def _parse_integer_valued(value: object, name: str) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"{name} must be numeric")
    if not np.isfinite(float(numeric)):
        raise ValueError(f"{name} must be finite")
    if float(numeric) != int(numeric):
        raise ValueError(f"{name} must be integer-valued")
    return int(numeric)


def _parse_finite_float(value: object, name: str) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not np.isfinite(float(numeric)):
        raise ValueError(f"{name} must be finite")
    return float(numeric)


def design_key(design: Design) -> tuple[object, ...]:
    validate_design(design)
    return (
        design.dgp,
        design.n,
        design.p,
        design.pi,
        design.tau,
        design.rep,
        design.seed,
    )


def row_design_key(row: pd.Series) -> tuple[object, ...]:
    try:
        return (
            str(row["dgp"]),
            _parse_integer_valued(row["n"], "n"),
            _parse_integer_valued(row["p"], "p"),
            _parse_finite_float(row["pi"], "pi"),
            _parse_finite_float(row["tau"], "tau"),
            _parse_integer_valued(row["rep"], "rep"),
            _parse_integer_valued(row["seed"], "seed"),
        )
    except Exception as exc:
        raise ValueError("results CSV contains invalid design-key values") from exc


__all__ = [
    "design_key",
    "parse_explicit_bool",
    "row_design_key",
    "validate_alpha_candidates",
    "validate_bool",
    "validate_design",
    "validate_designs",
    "validate_dgps",
    "validate_estimators",
    "validate_finite_float",
    "validate_k_folds_for_designs",
    "validate_nonnegative_float",
    "validate_nonnegative_int",
    "validate_optional_nonnegative_int",
    "validate_output_file_path",
    "validate_positive_int",
    "validate_probability_quantile",
    "validate_unique_sequence",
]
