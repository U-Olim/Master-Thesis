"""Tests for simulation runner execution and dispatch."""

import numpy as np
import pytest

from dgp.designs import Design
from dgp.true_parameters import get_oracle_control_indices
from estimators.base import EstimationResult
from simulation import runner as runner_module
from simulation._validation import (
    row_design_key,
    validate_design,
    validate_designs,
    validate_dgps,
    validate_estimators,
    validate_k_folds_for_designs,
    validate_output_file_path,
)
from simulation.runner import (
    DESIGN_KEY_COLUMNS,
    RESULT_COLUMNS,
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


def test_run_small_simulation_quantile_post_selection_subset() -> None:
    results = run_small_simulation(
        reps=1,
        n=50,
        p=4,
        alphas=np.linspace(0.0, 2.0, 3),
        estimators=("post_selection_quantile",),
        selection_cv=2,
    )

    assert len(results) == 1
    assert set(results["estimator"]) == {"post_selection_quantile"}
    assert results.iloc[0]["ps_selection_method"] == "quantile_specific"
    assert results.iloc[0]["psq_selection_method"] == "quantile_l1_cv"


def test_run_small_simulation_aligned_post_selection_subset() -> None:
    results = run_small_simulation(
        reps=1,
        n=50,
        p=4,
        alphas=np.linspace(0.0, 2.0, 3),
        estimators=("post_selection_ivqr_aligned",),
        selection_cv=2,
    )

    assert len(results) == 1
    assert set(results["estimator"]) == {"post_selection_ivqr_aligned"}
    assert results.iloc[0]["ps_selection_method"] == "ivqr_aligned"
    assert results.iloc[0]["psa_selection_method"] == "ivqr_aligned_quantile_l1_cv"


def test_runner_schema_constants_are_immutable_tuples() -> None:
    assert isinstance(RESULT_COLUMNS, tuple)
    assert isinstance(DESIGN_KEY_COLUMNS, tuple)


def test_runner_public_api_excludes_private_validation_helpers() -> None:
    assert "run_small_simulation" in runner_module.__all__
    assert "RESULT_COLUMNS" in runner_module.__all__
    assert "_validate_estimators" not in runner_module.__all__


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

    def fake_oracle_estimator(
        data,
        tau,
        alphas,
        oracle_indices,
        max_iter,
        gmm_ridge,
        critical_value_multiplier,
    ):
        captured["max_iter"] = max_iter
        captured["oracle_indices"] = np.asarray(oracle_indices)
        captured["critical_value_multiplier"] = critical_value_multiplier
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
    assert captured["critical_value_multiplier"] == pytest.approx(1.0)
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


def test_runner_passes_design_seed_to_post_selection_random_state(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post_selection_estimator(data, tau, alphas, **kwargs):
        captured.update(kwargs)
        return EstimationResult(
            estimator="post_selection_ivqr",
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
            selected_controls=1,
            runtime_seconds=0.0,
        )

    monkeypatch.setattr(
        runner_module,
        "estimate_post_selection_ivqr",
        fake_post_selection_estimator,
    )
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=98765)

    rows = run_single_replication(
        design,
        np.linspace(0.0, 2.0, 5),
        estimators=("post_selection",),
    )

    assert len(rows) == 1
    assert captured["selection_random_state"] == design.seed


def test_runner_passes_design_seed_to_quantile_post_selection_random_state(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_quantile_post_selection_estimator(data, tau, alphas, **kwargs):
        captured.update(kwargs)
        return EstimationResult(
            estimator="post_selection_quantile",
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
            selected_controls=1,
            runtime_seconds=0.0,
        )

    monkeypatch.setattr(
        runner_module,
        "estimate_post_selection_quantile_ivqr",
        fake_quantile_post_selection_estimator,
    )
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=98765)

    rows = run_single_replication(
        design,
        np.linspace(0.0, 2.0, 5),
        estimators=("post_selection_quantile",),
    )

    assert len(rows) == 1
    assert captured["selection_random_state"] == design.seed


def test_runner_passes_design_seed_to_aligned_post_selection_random_state(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_aligned_post_selection_estimator(data, tau, alphas, **kwargs):
        captured.update(kwargs)
        return EstimationResult(
            estimator="post_selection_ivqr_aligned",
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
            selected_controls=1,
            runtime_seconds=0.0,
        )

    monkeypatch.setattr(
        runner_module,
        "estimate_post_selection_ivqr_aligned",
        fake_aligned_post_selection_estimator,
    )
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=98765)

    rows = run_single_replication(
        design,
        np.linspace(0.0, 2.0, 5),
        estimators=("post_selection_ivqr_aligned",),
    )

    assert len(rows) == 1
    assert captured["selection_random_state"] == design.seed
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


@pytest.mark.parametrize(
    "estimators",
    [
        "oracle",
        ("oracle", "oracle"),
        ("oracle", 1),
        (),
    ],
)
def test_validate_estimators_rejects_invalid_sequences(estimators) -> None:
    with pytest.raises(ValueError):
        validate_estimators(estimators)


@pytest.mark.parametrize("dgps", ["dgp1", ("dgp1", "dgp1"), ()])
def test_validate_dgps_rejects_invalid_sequences(dgps) -> None:
    with pytest.raises(ValueError):
        validate_dgps(dgps)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_values": (True,)},
        {"p_values": (1.5,)},
        {"pi_values": (True,)},
        {"pi_values": (np.inf,)},
        {"taus": (True,)},
        {"taus": (1.0,)},
        {"reps": True},
        {"reps": 1000},
        {"base_seed": True},
        {"n_values": (80, 80)},
    ],
)
def test_make_simulation_grid_rejects_invalid_design_inputs(overrides) -> None:
    arguments = {
        "dgps": ("dgp1",),
        "n_values": (80,),
        "p_values": (5,),
        "pi_values": (1.0,),
        "taus": (0.5,),
        "reps": 1,
        "base_seed": 123,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError):
        make_simulation_grid(**arguments)


def test_make_simulation_grid_accepts_500_reps_with_existing_seed_schedule() -> None:
    designs = make_simulation_grid(
        dgps=("dgp1",),
        n_values=(80,),
        p_values=(5,),
        pi_values=(1.0,),
        taus=(0.5,),
        reps=500,
        base_seed=123,
    )

    assert len(designs) == 500
    assert designs[0].seed == 123
    assert designs[-1].seed == 622


def test_make_simulation_grid_rejects_negative_base_seed() -> None:
    with pytest.raises(ValueError, match="base_seed must be nonnegative"):
        make_simulation_grid(
            dgps=("dgp1",),
            n_values=(100,),
            p_values=(10,),
            pi_values=(1.0,),
            taus=(0.5,),
            reps=1,
            base_seed=-1,
        )


def test_run_small_simulation_rejects_negative_base_seed() -> None:
    with pytest.raises(ValueError, match="base_seed must be nonnegative"):
        run_small_simulation(
            dgp="dgp1",
            n=80,
            p=5,
            tau=0.5,
            reps=1,
            base_seed=-1,
            alphas=np.linspace(0.0, 2.0, 3),
        )


def test_make_simulation_grid_preserves_valid_seed_schedule() -> None:
    designs = make_simulation_grid(
        dgps=("dgp1",),
        n_values=(100,),
        p_values=(10,),
        pi_values=(1.0,),
        taus=(0.5,),
        reps=2,
        base_seed=123,
    )

    assert [design.seed for design in designs] == [123, 124]


@pytest.mark.parametrize(
    "alphas",
    [
        np.array([]),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([1.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([[0.0, 1.0]]),
    ],
)
def test_run_small_simulation_rejects_invalid_explicit_alphas(alphas) -> None:
    with pytest.raises(ValueError):
        run_small_simulation(reps=1, n=80, p=5, alphas=alphas)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantreg_max_iter": True},
        {"selection_cv": 0},
        {"selection_max_iter": 1.5},
        {"dml_k_folds": True},
        {"dml_quantile_penalty": -1.0},
        {"dml_ridge_alpha": np.inf},
        {"gmm_ridge": True},
        {"show_quantreg_warnings": 1},
        {"dml_fold_random_state": True},
    ],
)
def test_run_single_replication_rejects_invalid_runtime_parameters(kwargs) -> None:
    design = Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123)

    with pytest.raises(ValueError):
        run_single_replication(
            design,
            np.linspace(0.0, 2.0, 5),
            **kwargs,
        )


@pytest.mark.parametrize(
    "design",
    [
        object(),
        Design("bad", 80, 5, 1.0, 0.5, 0, 123),
        Design("dgp1", 0, 5, 1.0, 0.5, 0, 123),
        Design("dgp1", 80, 0, 1.0, 0.5, 0, 123),
        Design("dgp1", 80, 5, -0.1, 0.5, 0, 123),
        Design("dgp1", 80, 5, 1.0, 1.0, 0, 123),
        Design("dgp1", 80, 5, 1.0, 0.5, -1, 123),
        Design("dgp1", 80, 5, 1.0, 0.5, 0, -1),
    ],
)
def test_validate_design_rejects_invalid_designs(design) -> None:
    with pytest.raises(ValueError):
        validate_design(design)


@pytest.mark.parametrize(
    "designs",
    [
        "design",
        1,
        [object()],
        [Design("dgp1", 0, 5, 1.0, 0.5, 0, 123)],
    ],
)
def test_validate_designs_rejects_invalid_inputs(designs) -> None:
    with pytest.raises(ValueError):
        validate_designs(designs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n", 80.5),
        ("p", 5.2),
        ("rep", 0.5),
        ("seed", 123.4),
        ("pi", np.inf),
        ("tau", np.nan),
        ("n", "invalid"),
    ],
)
def test_row_design_key_rejects_invalid_values(field, value) -> None:
    row = {
        "dgp": "dgp1",
        "n": 80,
        "p": 5,
        "pi": 1.0,
        "tau": 0.5,
        "rep": 0,
        "seed": 123,
    }
    row[field] = value

    with pytest.raises(ValueError, match="invalid design-key values"):
        row_design_key(runner_module.pd.Series(row))


def test_validate_k_folds_for_designs_rejects_invalid_values() -> None:
    designs = [
        Design("dgp1", 80, 5, 1.0, 0.5, 0, 123),
        Design("dgp1", 40, 5, 1.0, 0.5, 1, 124),
    ]

    with pytest.raises(ValueError, match="at least 2"):
        validate_k_folds_for_designs(1, designs)
    with pytest.raises(ValueError, match="integer"):
        validate_k_folds_for_designs(True, designs)
    with pytest.raises(ValueError, match="smallest design n"):
        validate_k_folds_for_designs(41, designs)


def test_validate_output_file_path_rejects_invalid_paths(tmp_path) -> None:
    with pytest.raises(ValueError, match="file path"):
        validate_output_file_path(tmp_path)

    parent_file = tmp_path / "parent"
    parent_file.write_text("not a directory")
    with pytest.raises(ValueError, match="parent must be a directory"):
        validate_output_file_path(parent_file / "results.csv")


def test_run_single_replication_validates_design_before_data_generation(
    monkeypatch,
) -> None:
    generated = False

    def fake_generate_data(design):
        nonlocal generated
        generated = True

    monkeypatch.setattr(runner_module, "generate_data", fake_generate_data)

    with pytest.raises(ValueError):
        run_single_replication(
            Design("bad", 80, 5, 1.0, 0.5, 0, 123),
            np.linspace(0.0, 2.0, 5),
        )
    assert generated is False


@pytest.mark.parametrize("dml_k_folds", [1, 81])
def test_run_single_replication_rejects_infeasible_dml_folds(dml_k_folds) -> None:
    with pytest.raises(ValueError, match="dml_k_folds"):
        run_single_replication(
            Design("dgp1", 80, 5, 1.0, 0.5, 0, 123),
            np.linspace(0.0, 2.0, 5),
            dml_k_folds=dml_k_folds,
        )


def test_run_small_simulation_rejects_infeasible_dml_folds() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        run_small_simulation(
            n=80,
            p=5,
            reps=1,
            alphas=np.linspace(0.0, 2.0, 5),
            dml_k_folds=1,
        )


