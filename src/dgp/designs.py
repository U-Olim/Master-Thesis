"""Design and data containers for IVQR simulations.

The simulation compares DGP1, DGP2, and DGP3. DGP1 is the sparse Gaussian
baseline, DGP2 is a denser sparse Gaussian design, and DGP3 keeps sparse
controls but uses heavy-tailed structural shocks.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Design:
    """Unique identifier for one Monte Carlo cell and replication.

    `n` is sample size, `p` is the number of controls, `pi` is excluded
    instrument strength, `tau` is the quantile, `rep` is the replication index,
    and `seed` makes the simulated dataset reproducible.
    """

    dgp: str
    n: int
    p: int
    pi: float
    tau: float
    rep: int
    seed: int


@dataclass
class SimData:
    """Container for one simulated IVQR dataset.

    `alpha_true` is the target structural coefficient for the requested DGP and
    quantile. Optional `u` and `v` store structural and first-stage shocks for
    diagnostics.
    """

    y: np.ndarray
    d: np.ndarray
    z: np.ndarray
    x: np.ndarray
    alpha_true: float | None
    u: np.ndarray | None = None
    v: np.ndarray | None = None

