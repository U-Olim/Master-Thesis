from pathlib import Path

import numpy as np

from ivqr_sim.simulation.design import Design
from ivqr_sim.simulation.runner import run_pilot_simulation, run_single_replication


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


def test_run_pilot_simulation_returns_dataframe() -> None:
    alphas = np.linspace(0.0, 2.0, 5)

    results = run_pilot_simulation(reps=2, n=80, p=5, alphas=alphas)

    assert len(results) == 6
    assert set(results["estimator"]) == {"full_ivqr", "post_selection_ivqr", "dml_ivqr"}
    assert REQUIRED_KEYS.issubset(results.columns)
    assert {"cr_disconnected", "failed_alpha_count", "alpha_grid_size"}.issubset(
        results.columns
    )


def test_run_pilot_simulation_default_grid_has_17_points() -> None:
    results = run_pilot_simulation(
        dgp="dgp1",
        n=80,
        p=5,
        pi=1.0,
        tau=0.5,
        reps=1,
        base_seed=123,
        alphas=None,
    )

    assert results["alpha_grid_size"].dropna().unique().tolist() == [17]


def test_run_pilot_simulation_bias_logic() -> None:
    results = run_pilot_simulation(reps=1, n=80, p=5, alphas=np.linspace(0.0, 2.0, 5))

    for row in results.to_dict("records"):
        if row["alpha_hat"] is not None and not np.isnan(row["alpha_hat"]):
            assert row["bias"] == row["alpha_hat"] - row["alpha_true"]


def test_run_pilot_simulation_does_not_write_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    run_pilot_simulation(reps=1, n=80, p=5, alphas=np.linspace(0.0, 2.0, 5))

    assert not Path("results/raw/pilot_results.csv").exists()
