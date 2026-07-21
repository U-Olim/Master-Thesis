"""Deterministic simulation-design construction and keys."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import pandas as pd

from dgp.designs import Design
from simulation.config import DEFAULT_BASE_SEED, DGPS
from simulation.seeds import (
    make_design_seed,
    validate_nonnegative_float,
    validate_nonnegative_int,
)
from utils.validation import validate_positive_int, validate_tau


_T = TypeVar("_T")
VALID_DGPS: tuple[str, ...] = DGPS
DESIGN_KEY_COLUMNS: tuple[str, ...] = (
    "dgp",
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "seed",
)


def _validate_unique_sequence(name: str, values: Sequence[_T]) -> tuple[_T, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    try:
        values_tuple = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc
    if not values_tuple:
        raise ValueError(f"{name} must not be empty")
    seen: list[_T] = []
    for value in values_tuple:
        if value in seen:
            raise ValueError(f"{name} must not contain duplicates")
        seen.append(value)
    return values_tuple


def _validate_dgp_sequence(name: str, values: Sequence[str]) -> tuple[str, ...]:
    dgps = tuple(str(dgp) for dgp in _validate_unique_sequence(name, values))
    invalid_dgps = sorted(set(dgps) - set(VALID_DGPS))
    if invalid_dgps:
        raise ValueError(f"Unknown DGP(s): {invalid_dgps}")
    return dgps


def _validate_positive_int_sequence(
    name: str, values: Sequence[int]
) -> tuple[int, ...]:
    return tuple(
        validate_positive_int(name, int(value))
        for value in _validate_unique_sequence(name, values)
    )


def _validate_nonnegative_float_sequence(
    name: str, values: Sequence[float]
) -> tuple[float, ...]:
    return tuple(
        validate_nonnegative_float(name, float(value))
        for value in _validate_unique_sequence(name, values)
    )


def _validate_tau_sequence(name: str, values: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        validate_tau(float(value))
        for value in _validate_unique_sequence(name, values)
    )


def validate_design(design: Design) -> Design:
    if not isinstance(design, Design):
        raise ValueError("design must be a Design object")
    if design.dgp not in VALID_DGPS:
        raise ValueError(f"Unknown DGP: {design.dgp}")
    validate_positive_int("n", design.n)
    validate_positive_int("p", design.p)
    validate_nonnegative_float("pi", design.pi)
    validate_tau(design.tau)
    validate_nonnegative_int("rep", design.rep)
    validate_nonnegative_int("seed", design.seed)
    return design


def design_key(design: Design) -> tuple[object, ...]:
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
    return (
        str(row["dgp"]),
        int(row["n"]),
        int(row["p"]),
        float(row["pi"]),
        float(row["tau"]),
        int(row["rep"]),
        int(row["seed"]),
    )


def make_simulation_grid(
    dgps: tuple[str, ...],
    n_values: tuple[int, ...],
    p_values: tuple[int, ...],
    pi_values: tuple[float, ...],
    taus: tuple[float, ...],
    reps: int,
    base_seed: int = DEFAULT_BASE_SEED,
    rep_start: int = 0,
    rep_end: int | None = None,
) -> list[Design]:
    """Create the deterministic full Monte Carlo design grid."""
    dgps = _validate_dgp_sequence("dgps", dgps)
    n_values = _validate_positive_int_sequence("n_values", n_values)
    p_values = _validate_positive_int_sequence("p_values", p_values)
    pi_values = _validate_nonnegative_float_sequence("pi_values", pi_values)
    taus = _validate_tau_sequence("taus", taus)
    reps = validate_positive_int("reps", reps)
    base_seed = validate_nonnegative_int("base_seed", base_seed)
    rep_start = validate_nonnegative_int("rep_start", rep_start)
    rep_end = (
        reps - 1
        if rep_end is None
        else validate_nonnegative_int("rep_end", rep_end)
    )
    if rep_end < rep_start:
        raise ValueError("rep_end must be greater than or equal to rep_start")
    if rep_end >= reps:
        raise ValueError("rep_end must be less than reps")

    designs: list[Design] = []
    seeds: set[int] = set()
    for dgp in dgps:
        for n in n_values:
            for p in p_values:
                for pi in pi_values:
                    for tau in taus:
                        for rep in range(rep_start, rep_end + 1):
                            seed = make_design_seed(
                                base_seed=base_seed,
                                dgp=dgp,
                                n=n,
                                p=p,
                                pi=pi,
                                tau=tau,
                                rep=rep,
                            )
                            designs.append(Design(dgp, n, p, pi, tau, rep, seed))
                            seeds.add(seed)
    if len(seeds) != len(designs):
        raise ValueError("generated seeds are not unique")
    return designs


__all__ = ["DESIGN_KEY_COLUMNS", "make_simulation_grid"]
