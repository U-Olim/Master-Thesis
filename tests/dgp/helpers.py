from __future__ import annotations

from dgp.designs import Design


def design(
    *,
    dgp: str = "dgp1",
    n: int = 100,
    p: int = 20,
    pi: float = 0.5,
    tau: float = 0.5,
    rep: int = 0,
    seed: int = 123,
) -> Design:
    return Design(dgp=dgp, n=n, p=p, pi=pi, tau=tau, rep=rep, seed=seed)
