"""Design and data containers for IVQR simulations."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Design:
    """Unique identifier for one Monte Carlo cell and replication."""

    dgp: str
    n: int
    p: int
    pi: float
    tau: float
    rep: int
    seed: int


@dataclass
class SimData:
    """Container for one simulated IVQR dataset."""

    y: np.ndarray
    d: np.ndarray
    z: np.ndarray
    x: np.ndarray
    alpha_true: float
    u: np.ndarray | None = None
    v: np.ndarray | None = None
