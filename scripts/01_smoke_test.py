"""Lightweight installed-package smoke test for IVQR estimators."""

from __future__ import annotations

import numpy as np

from ivqr_sim.dgp import generate_data
from ivqr_sim.estimators.dml_ivqr import estimate_dml_ivqr
from ivqr_sim.estimators.full_ivqr import estimate_full_ivqr
from ivqr_sim.estimators.post_selection_ivqr import estimate_post_selection_ivqr
from ivqr_sim.simulation.design import Design


def _format_status(name: str, result: object) -> str:
    return (
        f"{name}: "
        f"alpha_hat={result.alpha_hat}, "
        f"alpha_true={result.alpha_true}, "
        f"failed={result.failed}, "
        f"converged={result.converged}, "
        f"cr_empty={result.cr_empty}, "
        f"failed_alpha={result.failed_alpha_count}/{result.alpha_grid_size}"
    )


def main() -> None:
    design = Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alphas = np.linspace(0.0, 2.0, 5)

    results = [
        ("full_ivqr", estimate_full_ivqr(data, tau=0.5, alphas=alphas)),
        (
            "post_selection_ivqr",
            estimate_post_selection_ivqr(data, tau=0.5, alphas=alphas, selection_cv=3),
        ),
        (
            "dml_ivqr",
            estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=3),
        ),
    ]

    for name, result in results:
        print(_format_status(name, result))


if __name__ == "__main__":
    main()
