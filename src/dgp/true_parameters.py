"""True structural quantile treatment effects for the Monte Carlo design."""

from math import sqrt

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
    """Return the true structural quantile treatment effect alpha_0(tau)."""

    if tau <= 0.0 or tau >= 1.0:
        raise ValueError("tau must be strictly between 0 and 1.")

    normalized_dgp = _normalize_dgp(dgp)

    if normalized_dgp in {"dgp1", "dgp2"}:
        return float(1.0 + norm.ppf(tau))

    if df <= 2:
        raise ValueError("df must be greater than 2 for dgp3.")

    scale = sqrt((df - 2) / df)
    return float(1.0 + scale * t.ppf(tau, df=df))
