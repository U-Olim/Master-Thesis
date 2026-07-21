from __future__ import annotations

import inspect

import numpy as np
import pytest

from dgp.designs import Design, SimData
import simulation.execution as execution
import simulation.runner as runner
from simulation import designs, dispatch, resume, seeds


REFERENCE_SEED = 8248586238594842791
ALPHAS = np.array([-1.0, 1.0, 3.0])


def test_seed_owner_and_runner_facade_are_exact() -> None:
    kwargs = {
        "base_seed": 12345,
        "dgp": "dgp1",
        "n": 100,
        "p": 20,
        "pi": 0.5,
        "tau": 0.5,
        "rep": 0,
    }
    assert seeds.make_design_seed(**kwargs) == REFERENCE_SEED
    assert runner.make_design_seed is seeds.make_design_seed
    assert runner.make_design_seed(**kwargs) == REFERENCE_SEED


def test_production_grid_order_count_blocks_and_seed_uniqueness() -> None:
    grid = designs.make_simulation_grid(
        dgps=("dgp1", "dgp2", "dgp3"),
        n_values=(500, 1000),
        p_values=(200, 500),
        pi_values=(1.0, 0.5, 0.25, 0.10),
        taus=(0.25, 0.50, 0.75),
        reps=500,
        base_seed=12345,
    )
    assert len(grid) == 144 * 500 == 72_000
    assert grid[0] == Design(
        "dgp1", 500, 200, 1.0, 0.25, 0,
        seeds.make_design_seed(
            base_seed=12345, dgp="dgp1", n=500, p=200,
            pi=1.0, tau=0.25, rep=0,
        ),
    )
    assert grid[-1] == Design(
        "dgp3", 1000, 500, 0.10, 0.75, 499,
        seeds.make_design_seed(
            base_seed=12345, dgp="dgp3", n=1000, p=500,
            pi=0.10, tau=0.75, rep=499,
        ),
    )
    assert len({item.seed for item in grid}) == len(grid)
    block = designs.make_simulation_grid(
        dgps=("dgp1",), n_values=(40,), p_values=(5,),
        pi_values=(0.5,), taus=(0.5,), reps=10,
        rep_start=4, rep_end=6,
    )
    assert [item.rep for item in block] == [4, 5, 6]


def test_one_design_generates_one_dgp_and_dispatches_exact_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = Design("dgp1", 40, 5, 0.5, 0.5, 2, 123)
    data = SimData(
        y=np.zeros(40), d=np.zeros(40), z=np.zeros(40),
        x=np.zeros((40, 5)), alpha_true=1.0,
        u=np.zeros(40), v=np.zeros(40),
    )
    calls = {"dgp": 0, "dispatch": 0}
    captured: dict[str, object] = {}

    def fake_generate(received: Design) -> SimData:
        calls["dgp"] += 1
        assert received is design
        return data

    def fake_dispatch(name, received_data, received_design, alphas, **kwargs):
        calls["dispatch"] += 1
        captured.update(name=name, data=received_data, design=received_design)
        captured.update(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(execution, "generate_data", fake_generate)
    monkeypatch.setattr(execution, "run_estimator", fake_dispatch)
    rows = execution.run_simulation_design(
        design, ALPHAS, estimators=("dml",), dml_k_folds=4,
        dml_quantile_penalty=0.03, dml_ridge_alpha=2.0,
        dml_quantile_solver="highs-ds", critical_value_multiplier=1.25,
    )
    assert calls == {"dgp": 1, "dispatch": 1}
    assert captured["name"] == "dml"
    assert captured["data"] is data
    assert captured["design"] is design
    assert captured["dml_k_folds"] == 4
    assert captured["dml_quantile_penalty"] == 0.03
    assert captured["dml_ridge_alpha"] == 2.0
    assert captured["dml_quantile_solver"] == "highs-ds"
    assert captured["critical_value_multiplier"] == 1.25
    assert len(rows) == 1
    assert rows[0]["failed"] is True


def test_runner_facade_and_execution_validation_are_equivalent() -> None:
    design = Design("dgp1", 40, 5, 0.5, 0.5, 0, 123)
    with pytest.raises(ValueError) as facade_error:
        runner.run_simulation_design(
            design, ALPHAS, estimators=("dml",), dml_k_folds=1
        )
    with pytest.raises(ValueError) as owner_error:
        execution.run_simulation_design(
            design, ALPHAS, estimators=("dml",), dml_k_folds=1
        )
    assert str(facade_error.value) == str(owner_error.value)
    facade_parameters = tuple(inspect.signature(runner.run_simulation_design).parameters)
    owner_parameters = tuple(
        name
        for name in inspect.signature(execution.run_simulation_design).parameters
        if not name.startswith("_")
    )
    assert facade_parameters == owner_parameters


def test_public_facade_reexports_owning_functions() -> None:
    assert runner.make_simulation_grid is designs.make_simulation_grid
    assert runner.run_simulation_batch is execution.run_simulation_batch
    assert runner.filter_completed_designs is resume.filter_completed_designs
    assert runner.normalize_estimator_names is dispatch.normalize_estimator_names
    assert runner.validate_oracle_support is dispatch.validate_oracle_support


def test_run_single_replication_compatibility_alias() -> None:
    design = Design("dgp1", 40, 5, 0.5, 0.5, 0, 123)
    with pytest.raises(ValueError) as alias_error:
        runner.run_single_replication(
            design, ALPHAS, estimators=("dml",), dml_k_folds=1
        )
    with pytest.raises(ValueError) as design_error:
        runner.run_simulation_design(
            design, ALPHAS, estimators=("dml",), dml_k_folds=1
        )
    assert str(alias_error.value) == str(design_error.value)
