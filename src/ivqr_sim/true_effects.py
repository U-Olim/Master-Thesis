"""True structural quantile treatment effects for the Monte Carlo design."""

from math import sqrt

from scipy.stats import norm, t


VALID_DGPS = {"dgp1", "dgp2", "dgp3"}


def true_alpha(tau: float, dgp: str, df: int = 5) -> float:
    """Return the true structural quantile treatment effect alpha_0(tau)."""

    if tau <= 0.0 or tau >= 1.0:
        raise ValueError("tau must be strictly between 0 and 1.")

    normalized_dgp = dgp.lower()
    if normalized_dgp not in VALID_DGPS:
        raise ValueError(f"Unknown DGP: {dgp}")

    if normalized_dgp in {"dgp1", "dgp2"}:
        return float(1.0 + norm.ppf(tau))

    if df <= 2:
        raise ValueError("df must be greater than 2 for dgp3.")

    scale = sqrt((df - 2) / df)
    return float(1.0 + scale * t.ppf(tau, df=df))
