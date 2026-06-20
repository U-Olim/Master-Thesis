"""Data-generating processes for the IVQR simulation study."""

from math import sqrt

import numpy as np
from scipy.stats import norm, t

from dgp.designs import Design, SimData
from dgp.true_parameters import _normalize_dgp, true_alpha, true_sparse_coefficients
from simulation.config import DF_T, RHO_UV, RHO_X


def make_covariance_matrix(p: int, rho_x: float) -> np.ndarray:
    """Return the AR(1) covariance matrix for high-dimensional controls."""

    if p <= 0:
        raise ValueError("p must be positive.")
    if abs(rho_x) >= 1:
        raise ValueError("abs(rho_x) must be strictly less than 1.")

    indices = np.arange(p)
    return rho_x ** np.abs(indices[:, None] - indices[None, :])


def generate_x(
    n: int,
    p: int,
    rho_x: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Gaussian controls with AR(1) covariance."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if p <= 0:
        raise ValueError("p must be positive.")

    covariance = make_covariance_matrix(p, rho_x)
    return rng.multivariate_normal(mean=np.zeros(p), cov=covariance, size=n)


def generate_coefficients(dgp: str, p: int) -> dict[str, np.ndarray]:
    """Generate sparse outcome and first-stage coefficient vectors."""
    beta, gamma = true_sparse_coefficients(dgp, p)
    return {"beta": beta, "gamma": gamma}


def generate_errors(
    dgp: str,
    n: int,
    rho_uv: float,
    df: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate correlated structural and first-stage disturbances."""

    normalized_dgp = _normalize_dgp(dgp)
    if n <= 0:
        raise ValueError("n must be positive.")
    if abs(rho_uv) >= 1:
        raise ValueError("abs(rho_uv) must be strictly less than 1.")
    if normalized_dgp == "dgp3" and df <= 2:
        raise ValueError("df must be greater than 2 for dgp3.")

    covariance = np.array([[1.0, rho_uv], [rho_uv, 1.0]])
    errors = rng.multivariate_normal(mean=np.zeros(2), cov=covariance, size=n)
    e1 = errors[:, 0]
    e2 = errors[:, 1]

    if normalized_dgp in {"dgp1", "dgp2"}:
        return e1, e2

    eps = np.finfo(float).eps
    u_uniform = np.clip(norm.cdf(e1), eps, 1.0 - eps)
    v_uniform = np.clip(norm.cdf(e2), eps, 1.0 - eps)
    scale = sqrt((df - 2) / df)
    u = scale * t.ppf(u_uniform, df=df)
    v = scale * t.ppf(v_uniform, df=df)
    return u, v


def generate_data(design: Design) -> SimData:
    """Generate one simulated dataset for a Monte Carlo design."""

    if design.n <= 0:
        raise ValueError("design.n must be positive.")
    if design.p <= 0:
        raise ValueError("design.p must be positive.")
    if design.pi < 0:
        raise ValueError("design.pi must be nonnegative.")

    rng = np.random.default_rng(design.seed)
    x = generate_x(design.n, design.p, RHO_X, rng)
    z = rng.normal(size=design.n)
    u, v = generate_errors(design.dgp, design.n, RHO_UV, DF_T, rng)
    coefficients = generate_coefficients(design.dgp, design.p)
    beta = coefficients["beta"]
    gamma = coefficients["gamma"]

    d_latent = design.pi * z + x @ gamma + v
    d = (d_latent > 0).astype(int)
    alpha = true_alpha(design.tau, design.dgp, df=DF_T)
    # Project_structure.pdf DGP outcome:
    # Y_i = 1 + X_i' beta + D_i(1 + u_i),
    # which implies alpha_0(tau) = 1 + F_u^{-1}(tau).
    y = 1.0 + x @ beta + d * (1.0 + u)

    return SimData(y=y, d=d, z=z, x=x, alpha_true=alpha, u=u, v=v)
