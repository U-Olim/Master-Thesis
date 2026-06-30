"""Data-generating processes for the IVQR simulation study.

The common structural design is

    X_i ~ N(0, Sigma),  Sigma_jk = rho_x ** |j - k|,
    Z_i ~ N(0, 1),
    D_i = 1{pi Z_i + X_i' gamma + V_i > 0},
    Y_i(d) = 1 + X_i' beta + U_i + d(1 + U_i),
    Y_i = Y_i(D_i),
    alpha_0(tau) = 1 + Q_U(tau).

DGP1 is the baseline exact-sparse Gaussian IVQR design. It combines sparse
high-dimensional controls with Gaussian U and V shocks. Positive correlation
between U and V induces endogeneity, while pi controls excluded-instrument
strength.

DGP2 is the denser exact-sparse Gaussian IVQR design with slower coefficient
decay. Its 20 active controls make control selection harder than in DGP1, but
all coefficients outside the active set remain exactly zero.

DGP3 is the heavy-tailed Gaussian-copula IVQR design with scaled Student-t
structural shocks. It uses the same sparse coefficient support as DGP1 while
giving U and V heavy-tailed, variance-normalized Student-t marginals. The
parameter rho_uv is the latent Gaussian-copula correlation and need not equal
the Pearson correlation of the transformed shocks. With the project's
quartile and median targets, DGP3 is a heavy-tail robustness design rather
than an extreme-quantile design.
"""

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
    """Generate exact-sparse outcome and first-stage coefficient vectors.

    DGP1 and DGP3 have 10 active controls. DGP2 is the denser exact-sparse
    design and has 20 active controls, subject to the available dimension.
    """
    beta, gamma = true_sparse_coefficients(dgp, p)
    return {"beta": beta, "gamma": gamma}


def generate_errors(
    dgp: str,
    n: int,
    rho_uv: float,
    df: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate structural shock U and first-stage shock V.

    DGP1 and DGP2 use correlated standard-Gaussian shocks. DGP3 maps
    correlated Gaussian draws through Student-t quantile functions, producing
    heavy-tailed, variance-normalized marginals with Gaussian-copula
    dependence.
    """

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


def _treatment_from_latent_index(
    x: np.ndarray,
    z: np.ndarray,
    gamma: np.ndarray,
    v: np.ndarray,
    pi: float,
) -> np.ndarray:
    """Return D = 1{pi Z + X gamma + V > 0}."""
    latent_index = pi * z + x @ gamma + v
    return (latent_index > 0).astype(int)


def _structural_outcome(
    x: np.ndarray,
    beta: np.ndarray,
    u: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    """Return Y = 1 + X beta + U + D(1 + U)."""
    return 1.0 + x @ beta + u + d * (1.0 + u)


def generate_data(design: Design) -> SimData:
    """Generate one dataset from the documented structural IVQR design."""

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

    d = _treatment_from_latent_index(x, z, gamma, v, design.pi)
    alpha = true_alpha(design.tau, design.dgp, df=DF_T)
    # IVQR-compatible structural quantile outcome:
    # Y_i(d) = 1 + X_i' beta + u_i + d(1 + u_i).
    # Therefore q_d(tau, X_i) = 1 + X_i' beta + Q_u(tau)
    # + d(1 + Q_u(tau)), and alpha_0(tau) = 1 + Q_u(tau).
    y = _structural_outcome(x, beta, u, d)

    return SimData(y=y, d=d, z=z, x=x, alpha_true=alpha, u=u, v=v)

