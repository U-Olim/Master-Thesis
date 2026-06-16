"""Pilot Monte Carlo runner utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ivqr_sim.dgp import generate_data
from ivqr_sim.estimators.base import EstimationResult
from ivqr_sim.estimators.dml_ivqr import estimate_dml_ivqr
from ivqr_sim.estimators.full_ivqr import estimate_full_ivqr
from ivqr_sim.estimators.post_selection_ivqr import estimate_post_selection_ivqr
from ivqr_sim.simulation.design import Design


EstimatorFn = Callable[..., EstimationResult]
PILOT_QUANTREG_MAX_ITER = 5


def _result_to_row(design: Design, result: EstimationResult) -> dict[str, object]:
    bias = None
    if result.alpha_hat is not None:
        bias = result.alpha_hat - result.alpha_true

    return {
        "dgp": design.dgp,
        "n": design.n,
        "p": design.p,
        "pi": design.pi,
        "tau": design.tau,
        "rep": design.rep,
        "seed": design.seed,
        "estimator": result.estimator,
        "alpha_hat": result.alpha_hat,
        "alpha_true": result.alpha_true,
        "bias": bias,
        "failed": result.failed,
        "converged": result.converged,
        "cr_lower": result.cr_lower,
        "cr_upper": result.cr_upper,
        "cr_length": result.cr_length,
        "cr_empty": result.cr_empty,
        "cr_disconnected": result.cr_disconnected,
        "cr_covers_true": result.cr_covers_true,
        "selected_controls": result.selected_controls,
        "runtime_seconds": result.runtime_seconds,
        "failed_alpha_count": result.failed_alpha_count,
        "alpha_grid_size": result.alpha_grid_size,
        "message": result.message,
    }


def run_single_replication(
    design: Design,
    alphas: np.ndarray,
    estimators: tuple[str, ...] = ("full", "post_selection", "dml"),
) -> list[dict[str, object]]:
    """Generate one dataset and run requested estimators on it."""
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("alphas must be a nonempty one-dimensional array")

    data = generate_data(design)
    estimator_map: dict[str, EstimatorFn] = {
        "full": estimate_full_ivqr,
        "post_selection": estimate_post_selection_ivqr,
        "dml": estimate_dml_ivqr,
    }

    rows: list[dict[str, object]] = []
    for estimator_name in estimators:
        if estimator_name not in estimator_map:
            raise ValueError(f"Unknown estimator: {estimator_name}")

        estimator = estimator_map[estimator_name]
        if estimator_name == "post_selection":
            result = estimator(
                data,
                tau=design.tau,
                alphas=alphas,
                selection_cv=3,
                quantreg_max_iter=PILOT_QUANTREG_MAX_ITER,
            )
        elif estimator_name == "dml":
            result = estimator(data, tau=design.tau, alphas=alphas, k_folds=3)
        else:
            result = estimator(
                data,
                tau=design.tau,
                alphas=alphas,
                max_iter=PILOT_QUANTREG_MAX_ITER,
            )
        rows.append(_result_to_row(design, result))

    return rows


def run_pilot_simulation(
    dgp: str = "dgp1",
    n: int = 250,
    p: int = 200,
    pi: float = 1.0,
    tau: float = 0.5,
    reps: int = 10,
    base_seed: int = 12345,
    alphas: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run a small pilot simulation and return raw estimator-level rows."""
    if reps <= 0:
        raise ValueError("reps must be positive")
    if alphas is None:
        alphas = np.linspace(-1.0, 3.0, 17)
    else:
        alphas = np.asarray(alphas, dtype=float)

    rows: list[dict[str, object]] = []
    for rep in range(reps):
        design = Design(
            dgp=dgp,
            n=n,
            p=p,
            pi=pi,
            tau=tau,
            rep=rep,
            seed=base_seed + rep,
        )
        rows.extend(run_single_replication(design, alphas))

    return pd.DataFrame(rows)
