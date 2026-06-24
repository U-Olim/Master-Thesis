"""Tests for simulation result rows and tables."""

import numpy as np

from dgp.designs import Design
from simulation.runner import run_single_replication, run_small_simulation


REQUIRED_KEYS = {
    "dgp",
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "seed",
    "estimator",
    "alpha_hat",
    "alpha_true",
    "bias",
    "absolute_error",
    "squared_error",
    "status",
    "error_type",
    "error_message",
    "failed",
    "converged",
    "cr_lower",
    "cr_upper",
    "cr_length",
    "cr_empty",
    "cr_disconnected",
    "cr_covers_true",
    "selected_controls",
    "runtime_seconds",
    "failed_alpha_count",
    "alpha_grid_size",
    "message",
}


def test_run_single_replication_returns_one_row_per_estimator() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    alphas = np.linspace(0.0, 2.0, 5)

    rows = run_single_replication(design, alphas)

    assert isinstance(rows, list)
    assert len(rows) == 3
    alpha_true = rows[0]["alpha_true"]

    for row in rows:
        assert REQUIRED_KEYS.issubset(row.keys())
        assert row["rep"] == 0
        assert row["seed"] == 123
        assert row["tau"] == 0.5
        assert row["alpha_true"] == alpha_true


def test_run_small_simulation_returns_dataframe() -> None:
    alphas = np.linspace(0.0, 2.0, 5)

    results = run_small_simulation(reps=2, n=80, p=5, alphas=alphas)

    assert len(results) == 6
    assert set(results["estimator"]) == {"oracle", "post_selection_ivqr", "dml_ivqr"}
    assert REQUIRED_KEYS.issubset(results.columns)
    assert {"cr_disconnected", "failed_alpha_count", "alpha_grid_size"}.issubset(
        results.columns
    )


def test_run_small_simulation_bias_logic() -> None:
    results = run_small_simulation(reps=1, n=80, p=5, alphas=np.linspace(0.0, 2.0, 5))

    for row in results.to_dict("records"):
        if row["alpha_hat"] is not None and not np.isnan(row["alpha_hat"]):
            assert row["bias"] == row["alpha_hat"] - row["alpha_true"]


