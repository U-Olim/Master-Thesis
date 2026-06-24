"""Tests for simulation runner execution and dispatch."""

import numpy as np
import pytest

from dgp.designs import Design
from dgp.true_parameters import get_oracle_control_indices
from estimators.base import EstimationResult
from simulation import runner as runner_module
from simulation.runner import (
    make_simulation_grid,
    run_single_replication,
    run_small_simulation,
)


def test_run_small_simulation_estimator_subset() -> None:
    results = run_small_simulation(
        reps=2,
        n=80,
        p=5,
        alphas=np.linspace(0.0, 2.0, 5),
        estimators=("post_selection",),
    )

    assert len(results) == 2
    assert set(results["estimator"]) == {"post_selection_ivqr"}


def test_run_small_simulation_invalid_estimator_raises() -> None:
    with pytest.raises(ValueError, match="Unknown estimator"):
        run_small_simulation(
            reps=1,
            n=80,
            p=5,
            alphas=np.linspace(0.0, 2.0, 5),
            estimators=("bad_estimator",),
        )


@pytest.mark.parametrize(
    ("dgp", "expected_support_size"),
    [("dgp1", 10), ("dgp2", 20), ("dgp3", 10)],
)
def test_runner_passes_dgp_oracle_support_to_oracle_estimator(
    monkeypatch,
    dgp: str,
    expected_support_size: int,
) -> None:
    captured: dict[str, object] = {}

    def fake_oracle_estimator(data, tau, alphas, oracle_indices, max_iter, gmm_ridge):
        captured["max_iter"] = max_iter
        captured["oracle_indices"] = np.asarray(oracle_indices)
        return EstimationResult(
            estimator="oracle",
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

    monkeypatch.setattr(runner_module, "estimate_oracle_ivqr", fake_oracle_estimator)
    design = Design(dgp, n=80, p=20, pi=1.0, tau=0.5, rep=0, seed=123)

    rows = run_single_replication(
        design,
        np.linspace(0.0, 2.0, 5),
        estimators=("oracle",),
        quantreg_max_iter=123,
    )

    assert len(rows) == 1
    assert captured["max_iter"] == 123
    expected_indices = get_oracle_control_indices(dgp, design.p)
    np.testing.assert_array_equal(captured["oracle_indices"], expected_indices)
    assert len(expected_indices) == expected_support_size


def test_dml_k_folds_is_passed_to_dml_estimator(monkeypatch) -> None:
    captured: dict[str, int] = {}

    def fake_dml_estimator(data, tau, alphas, k_folds, **kwargs):
        captured["k_folds"] = k_folds
        return EstimationResult(
            estimator="dml_ivqr",
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

    monkeypatch.setattr(runner_module, "estimate_dml_ivqr", fake_dml_estimator)
    rows = run_single_replication(
        Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123),
        np.linspace(0.0, 2.0, 5),
        estimators=("dml",),
        dml_k_folds=5,
    )

    assert len(rows) == 1
    assert captured["k_folds"] == 5


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
    def broken_oracle_estimator(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_module, "estimate_oracle_ivqr", broken_oracle_estimator)
    design = Design("dgp1", 80, 20, 1.0, 0.5, rep=0, seed=123)

    rows = run_single_replication(
        design,
        np.linspace(0.0, 2.0, 5),
        estimators=("oracle", "post_selection"),
    )

    assert len(rows) == 2
    by_estimator = {row["estimator"]: row for row in rows}
    assert by_estimator["oracle"]["failed"] is True
    assert by_estimator["oracle"]["status"] == "failed"
    assert by_estimator["oracle"]["error_type"] == "RuntimeError"
    assert by_estimator["oracle"]["error_message"] == "boom"
    assert by_estimator["oracle"]["converged"] is False
    oracle_message = by_estimator["oracle"]["message"]
    assert isinstance(oracle_message, str)
    assert "Unexpected estimator error: RuntimeError: boom" in oracle_message
    assert "post_selection_ivqr" in by_estimator
    assert by_estimator["post_selection_ivqr"]["message"] != by_estimator["oracle"][
        "message"
    ]


def test_run_single_replication_rejects_full_control_in_main_runner() -> None:
    design = Design("dgp1", 80, 20, 1.0, 0.5, rep=0, seed=123)

    with pytest.raises(ValueError, match="Unknown estimator"):
        run_single_replication(
            design,
            np.linspace(0.0, 2.0, 5),
            estimators=("full",),
        )


def test_run_single_replication_accepts_oracle_estimator() -> None:
    design = Design("dgp1", 200, 50, 1.0, 0.5, rep=0, seed=123)

    rows = run_single_replication(
        design,
        np.linspace(0.0, 2.0, 3),
        estimators=("oracle",),
    )

    assert len(rows) == 1
    assert rows[0]["estimator"] == "oracle"
    assert rows[0]["selected_controls"] == 10
    assert rows[0]["status"] in {"ok", "failed"}


