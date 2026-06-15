"""Simulation design identifiers.

The ``Design`` dataclass uniquely identifies one Monte Carlo cell and
replication.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Design:
    dgp: str
    n: int
    p: int
    pi: float
    tau: float
    rep: int
    seed: int
