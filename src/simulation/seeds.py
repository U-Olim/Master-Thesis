"""Deterministic simulation seed derivation."""

from __future__ import annotations

import hashlib

import numpy as np

from simulation.config import DGPS
from utils.validation import validate_positive_int, validate_tau


VALID_DGPS: tuple[str, ...] = DGPS
SEED_RULE_TEXT = (
    "seed = sha256(base_seed, dgp, n, p, pi, tau, rep), "
    "independent of estimator and execution order"
)


def validate_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_positive_float(name: str, value: float) -> float:
    value = validate_float(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_nonnegative_float(name: str, value: float) -> float:
    value = validate_float(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def validate_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def make_design_seed(
    *,
    base_seed: int,
    dgp: str,
    n: int,
    p: int,
    pi: float,
    tau: float,
    rep: int,
) -> int:
    """Return a stable design-cell seed independent of estimator execution."""
    base_seed = validate_nonnegative_int("base_seed", base_seed)
    dgp = str(dgp).lower()
    if dgp not in VALID_DGPS:
        raise ValueError(f"Unknown DGP: {dgp}")
    n = validate_positive_int("n", int(n))
    p = validate_positive_int("p", int(p))
    pi = validate_nonnegative_float("pi", float(pi))
    tau = validate_tau(float(tau))
    rep = validate_nonnegative_int("rep", rep)

    key = f"{base_seed}|{dgp}|{n}|{p}|{pi:.12g}|{tau:.12g}|{rep}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**63 - 1)
    return seed if seed != 0 else 1


__all__ = ["SEED_RULE_TEXT", "make_design_seed"]
