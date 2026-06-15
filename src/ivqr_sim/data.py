"""Data objects for simulated IVQR Monte Carlo samples."""

from dataclasses import dataclass

import numpy as np


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
