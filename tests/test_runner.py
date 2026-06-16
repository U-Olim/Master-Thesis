from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ivqr_sim.estimators.base import EstimationResult
from ivqr_sim.simulation.design import Design
from ivqr_sim.simulation import runner as runner_module
from ivqr_sim.simulation.runner import (
    completed_design_keys,
    filter_completed_designs,
    make_simulation_grid,
    observed_design_keys,
    run_pilot_simulation,
    run_simulation_batch,
    run_single_replication,
)


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


def test_run_pilot_simulation_default_grid_has_9_points() -> None:
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

    assert results["alpha_grid_size"].dropna().unique().tolist() == [9]


def test_run_pilot_simulation_explicit_grid_size_has_17_points() -> None:
    results = run_pilot_simulation(
        dgp="dgp1",
        n=80,
        p=5,
        pi=1.0,
        tau=0.5,
        reps=1,
        base_seed=123,
        alphas=None,
        alpha_grid_size=17,
    )

    assert results["alpha_grid_size"].dropna().unique().tolist() == [17]


def test_run_pilot_simulation_estimator_subset() -> None:
    results = run_pilot_simulation(
        reps=2,
        n=80,
        p=5,
        alphas=np.linspace(0.0, 2.0, 5),
        estimators=("post_selection",),
    )

    assert len(results) == 2
    assert set(results["estimator"]) == {"post_selection_ivqr"}


def test_run_pilot_simulation_invalid_estimator_raises() -> None:
    with pytest.raises(ValueError, match="Unknown estimator"):
        run_pilot_simulation(
            reps=1,
            n=80,
            p=5,
            alphas=np.linspace(0.0, 2.0, 5),
            estimators=("bad_estimator",),
        )


def test_quantreg_max_iter_is_passed_to_full_estimator(monkeypatch) -> None:
    captured: dict[str, int] = {}

    def fake_full_estimator(data, tau, alphas, max_iter, gmm_ridge):
        captured["max_iter"] = max_iter
        return EstimationResult(
            estimator="full_ivqr",
            alpha_hat=1.0,
            alpha_true=data.alpha_true,
            tau=tau,
            converged=True,
            failed=False,
            message="ok",
            objective_value=0.0,
            at_grid_boundary=False,
            alpha_grid_size=len(alphas),
            failed_alpha_count=0,
            cr_lower=None,
            cr_upper=None,
            cr_length=None,
            cr_covers_true=None,
            cr_empty=True,
            cr_disconnected=False,
            selected_controls=None,
            runtime_seconds=0.0,
        )

    monkeypatch.setattr(runner_module, "estimate_full_ivqr", fake_full_estimator)
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)

    rows = run_single_replication(
        design,
        np.linspace(0.0, 2.0, 5),
        estimators=("full",),
        quantreg_max_iter=123,
    )

    assert len(rows) == 1
    assert captured["max_iter"] == 123


def test_run_pilot_simulation_bias_logic() -> None:
    results = run_pilot_simulation(reps=1, n=80, p=5, alphas=np.linspace(0.0, 2.0, 5))

    for row in results.to_dict("records"):
        if row["alpha_hat"] is not None and not np.isnan(row["alpha_hat"]):
            assert row["bias"] == row["alpha_hat"] - row["alpha_true"]


def test_run_pilot_simulation_does_not_write_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    run_pilot_simulation(reps=1, n=80, p=5, alphas=np.linspace(0.0, 2.0, 5))

    assert not Path("results/raw/pilot_results.csv").exists()


def test_make_simulation_grid_size_and_unique_seeds() -> None:
    designs = make_simulation_grid(
        dgps=("dgp1",),
        n_values=(80,),
        p_values=(5,),
        pi_values=(1.0, 0.5),
        taus=(0.5,),
        reps=3,
    )

    assert len(designs) == 6
    assert len({design.seed for design in designs}) == 6


def test_make_simulation_grid_accepts_all_valid_dgps() -> None:
    designs = make_simulation_grid(
        dgps=("dgp1", "dgp2", "dgp3"),
        n_values=(80,),
        p_values=(5,),
        pi_values=(1.0,),
        taus=(0.5,),
        reps=1,
    )

    assert [design.dgp for design in designs] == ["dgp1", "dgp2", "dgp3"]


def test_make_simulation_grid_invalid_dgp_raises() -> None:
    with pytest.raises(ValueError, match="Unknown DGP"):
        make_simulation_grid(
            dgps=("dgp1", "bad_dgp"),
            n_values=(80,),
            p_values=(5,),
            pi_values=(1.0,),
            taus=(0.5,),
            reps=1,
        )


def test_make_simulation_grid_loop_order_is_deterministic() -> None:
    designs = make_simulation_grid(
        dgps=("dgp1", "dgp2"),
        n_values=(80,),
        p_values=(5,),
        pi_values=(1.0, 0.5),
        taus=(0.25, 0.5),
        reps=2,
        base_seed=100,
    )

    assert designs[0] == Design("dgp1", 80, 5, 1.0, 0.25, rep=0, seed=100)
    assert designs[-1] == Design(
        "dgp2",
        80,
        5,
        0.5,
        0.5,
        rep=1,
        seed=10_000_000 + 10_000 + 1_000 + 1 + 100,
    )


def test_run_single_replication_catches_unexpected_estimator_exception(
    monkeypatch,
) -> None:
    def broken_full_estimator(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_module, "estimate_full_ivqr", broken_full_estimator)
    design = Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123)

    rows = run_single_replication(
        design,
        np.linspace(0.0, 2.0, 5),
        estimators=("full", "post_selection"),
    )

    assert len(rows) == 2
    by_estimator = {row["estimator"]: row for row in rows}
    assert by_estimator["full_ivqr"]["failed"] is True
    assert by_estimator["full_ivqr"]["converged"] is False
    assert "Unexpected estimator error: RuntimeError: boom" in by_estimator["full_ivqr"][
        "message"
    ]
    assert "post_selection_ivqr" in by_estimator
    assert by_estimator["post_selection_ivqr"]["message"] != by_estimator["full_ivqr"][
        "message"
    ]


def test_run_simulation_batch_returns_expected_rows() -> None:
    designs = [
        Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123),
        Design("dgp1", 80, 5, 1.0, 0.5, rep=1, seed=124),
    ]

    results = run_simulation_batch(
        designs,
        np.linspace(0.0, 2.0, 5),
        estimators=("post_selection", "dml"),
    )

    assert len(results) == 4
    assert set(results["estimator"]) == {"post_selection_ivqr", "dml_ivqr"}
    assert REQUIRED_KEYS.issubset(results.columns)


def test_run_simulation_batch_writes_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "batch.csv"
    designs = [Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123)]

    run_simulation_batch(
        designs,
        np.linspace(0.0, 2.0, 5),
        estimators=("post_selection",),
        output_path=output_path,
    )

    written = pd.read_csv(output_path)
    assert len(written) == 1
    assert written.loc[0, "estimator"] == "post_selection_ivqr"


def test_filter_completed_designs_removes_fully_completed_design(tmp_path: Path) -> None:
    designs = [
        Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123),
        Design("dgp1", 80, 5, 1.0, 0.5, rep=1, seed=124),
    ]
    output_path = tmp_path / "existing.csv"
    pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": 80,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
                "estimator": "post_selection_ivqr",
            },
            {
                "dgp": "dgp1",
                "n": 80,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
                "estimator": "dml_ivqr",
            },
        ]
    ).to_csv(output_path, index=False)

    pending = filter_completed_designs(
        designs,
        output_path,
        estimators=("post_selection", "dml"),
    )

    assert pending == [designs[1]]


def test_filter_completed_designs_keeps_partially_completed_design(tmp_path: Path) -> None:
    design = Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123)
    output_path = tmp_path / "existing.csv"
    pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": 80,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
                "estimator": "post_selection_ivqr",
            }
        ]
    ).to_csv(output_path, index=False)

    pending = filter_completed_designs(
        [design],
        output_path,
        estimators=("post_selection", "dml"),
    )

    assert pending == [design]


def test_observed_design_keys_reads_existing_results(tmp_path: Path) -> None:
    output_path = tmp_path / "existing.csv"
    pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": 80,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
            }
        ]
    ).to_csv(output_path, index=False)

    assert observed_design_keys(output_path) == {("dgp1", 80, 5, 1.0, 0.5, 0, 123)}


def test_completed_design_keys_deprecated_alias_still_works(tmp_path: Path) -> None:
    output_path = tmp_path / "existing.csv"
    pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": 80,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
            }
        ]
    ).to_csv(output_path, index=False)

    assert completed_design_keys(output_path) == observed_design_keys(output_path)


def test_filter_completed_designs_treats_failed_rows_as_completed_by_default(
    tmp_path: Path,
) -> None:
    design = Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123)
    output_path = tmp_path / "existing.csv"
    pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": 80,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
                "estimator": "post_selection_ivqr",
                "failed": True,
            }
        ]
    ).to_csv(output_path, index=False)

    pending = filter_completed_designs(
        [design],
        output_path,
        estimators=("post_selection",),
        rerun_failed=False,
    )

    assert pending == []


def test_filter_completed_designs_rerun_failed_keeps_failed_design(
    tmp_path: Path,
) -> None:
    design = Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123)
    output_path = tmp_path / "existing.csv"
    pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": 80,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
                "estimator": "post_selection_ivqr",
                "failed": "True",
            }
        ]
    ).to_csv(output_path, index=False)

    pending = filter_completed_designs(
        [design],
        output_path,
        estimators=("post_selection",),
        rerun_failed=True,
    )

    assert pending == [design]


def test_filter_completed_designs_rerun_failed_requires_all_successes(
    tmp_path: Path,
) -> None:
    design = Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123)
    output_path = tmp_path / "existing.csv"
    pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": 80,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
                "estimator": "post_selection_ivqr",
                "failed": False,
            },
            {
                "dgp": "dgp1",
                "n": 80,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
                "estimator": "dml_ivqr",
                "failed": True,
            },
        ]
    ).to_csv(output_path, index=False)

    pending = filter_completed_designs(
        [design],
        output_path,
        estimators=("post_selection", "dml"),
        rerun_failed=True,
    )

    assert pending == [design]


def test_filter_completed_designs_rerun_failed_missing_failed_column_raises(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "missing_failed.csv"
    pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": 80,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
                "estimator": "post_selection_ivqr",
            }
        ]
    ).to_csv(output_path, index=False)

    with pytest.raises(ValueError, match="missing required resume columns"):
        filter_completed_designs(
            [Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123)],
            output_path,
            estimators=("post_selection",),
            rerun_failed=True,
        )


def test_filter_completed_designs_malformed_csv_raises(tmp_path: Path) -> None:
    output_path = tmp_path / "bad.csv"
    pd.DataFrame([{"dgp": "dgp1", "n": 80}]).to_csv(output_path, index=False)

    with pytest.raises(ValueError, match="missing required resume columns"):
        filter_completed_designs(
            [Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123)],
            output_path,
            estimators=("post_selection", "dml"),
        )
