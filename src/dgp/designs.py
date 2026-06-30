"""Design and data containers for IVQR simulations.

The simulation compares DGP1, DGP2, and DGP3. DGP1 is the baseline sparse
Gaussian design with 5 active controls, DGP2 is the denser sparse
selection-stress design with 10 active controls, and DGP3 is the heavy-tail
sparse robustness design with 5 active controls.
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

    Shapes are `y: (n,)`, `d: (n,)`, `z: (n,)` for the current single
    excluded instrument, and `x: (n, p)`. The true active controls used by the
    oracle benchmark are obtained from the DGP-specific coefficient support.
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

