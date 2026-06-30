"""True structural coefficients and oracle supports for the Monte Carlo design.

DGP1 and DGP2 use Gaussian structural shocks, while DGP3 uses a
variance-normalized Student-t shock. The returned `alpha_true` is the target
structural quantile coefficient alpha_tau used to evaluate bias, RMSE, and
coverage.
"""

from math import sqrt

import numpy as np
from scipy.stats import norm, t


VALID_DGPS = {"dgp1", "dgp2", "dgp3"}


def _normalize_dgp(dgp: str) -> str:
    """Return a validated lowercase DGP name."""

    if not isinstance(dgp, str):
        raise ValueError("dgp must be a string.")

    normalized = dgp.lower()
    if normalized not in VALID_DGPS:
        raise ValueError(f"Unknown DGP: {dgp}")
    return normalized


def true_alpha(tau: float, dgp: str, df: int = 5) -> float:
    """Return the true structural quantile treatment effect.

    For DGP1 and DGP2, the structural shock is standard Gaussian, so

        alpha_0(tau) = 1 + Phi^{-1}(tau).

    For DGP3, the structural shock is a variance-normalized Student-t random
    variable, so

        alpha_0(tau)
            = 1 + sqrt((df - 2) / df) * t_df^{-1}(tau).
    """

    if tau <= 0.0 or tau >= 1.0:
        raise ValueError("tau must be strictly between 0 and 1.")

    normalized_dgp = _normalize_dgp(dgp)

    if normalized_dgp in {"dgp1", "dgp2"}:
        return float(1.0 + norm.ppf(tau))

    if df <= 2:
        raise ValueError("df must be greater than 2 for dgp3.")

    scale = sqrt((df - 2) / df)
    return float(1.0 + scale * t.ppf(tau, df=df))


def _active_control_count(dgp: str) -> int:
    normalized_dgp = _normalize_dgp(dgp)
    return 10 if normalized_dgp == "dgp2" else 5


def get_oracle_control_indices(
    dgp_name: str,
    p: int,
    tol: float = 1e-12,
) -> np.ndarray:
    """Return simulation-only oracle controls with nonzero true beta or gamma.

    The oracle support is infeasible in real applications because it uses DGP
    knowledge. It is the union of controls with nonzero structural or
    first-stage coefficients.
    """
    if p <= 0:
        raise ValueError("p must be positive.")

    required = _active_control_count(dgp_name)
    if p < required:
        raise ValueError(
            f"{dgp_name} requires p >= {required} because it uses "
            f"{required} active controls; "
            f"received p={p}."
        )

    beta, gamma = true_sparse_coefficients(dgp_name, p)
    active = np.flatnonzero((np.abs(beta) > tol) | (np.abs(gamma) > tol))
    return np.asarray(active, dtype=int)


def get_oracle_control_count(dgp_name: str, p: int) -> int:
    """Return the number of simulation-only oracle controls."""
    return int(get_oracle_control_indices(dgp_name, p).size)


def true_sparse_coefficients(dgp: str, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Return true exact-sparse outcome and first-stage coefficient vectors.

    DGP2 is the denser exact-sparse selection-stress design with 10 active
    controls. DGP1 and DGP3 have 5 active controls.
    """
    normalized_dgp = _normalize_dgp(dgp)
    if p <= 0:
        raise ValueError("p must be positive.")

    beta = np.zeros(p)
    gamma = np.zeros(p)

    if normalized_dgp == "dgp2":
        s = min(10, p)
        indices = np.arange(1, s + 1, dtype=float)
        beta[:s] = 0.5 / np.sqrt(indices)
        gamma[:s] = 0.4 / np.sqrt(indices)
    else:
        s = min(5, p)
        indices = np.arange(1, s + 1)
        values = 0.5 / indices
        beta[:s] = values
        gamma[:s] = values
    return beta, gamma

