# Consolidated tests for the thematic project structure.

import importlib.util
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from estimators.base import EstimationResult
from dgp.designs import Design
from dgp.true_parameters import get_oracle_control_indices
from simulation import runner as runner_module
from simulation.batching import (
    completed_design_keys,
    filter_completed_designs,
    observed_design_keys,
    run_simulation_batch,
)
from simulation.chunking import select_design_chunk
from simulation.runner import (
    DEFAULT_DML_K_FOLDS,
    DEFAULT_QUANTREG_MAX_ITER,
    DEFAULT_SIMULATION_ESTIMATORS,
    VALID_ESTIMATORS,
    make_simulation_grid,
    quantreg_iteration_warning_filter,
    run_small_simulation,
    run_single_replication,
)
from simulation.config import DEFAULT_N_JOBS

FULL_SIMULATION_SCRIPT = Path(__file__).resolve().parents[1] / "scenarios" / "main_simulation.py"
spec = importlib.util.spec_from_file_location("full_simulation_cli", FULL_SIMULATION_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {FULL_SIMULATION_SCRIPT}")
full_simulation_cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = full_simulation_cli
spec.loader.exec_module(full_simulation_cli)
full_simulation_main = full_simulation_cli.main

FULL_CONTROL_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scenarios" / "full_control_ivqr.py"
)
full_control_spec = importlib.util.spec_from_file_location(
    "full_control_cli", FULL_CONTROL_SCRIPT
)
if full_control_spec is None or full_control_spec.loader is None:
    raise ImportError(f"Could not load {FULL_CONTROL_SCRIPT}")
full_control_cli = importlib.util.module_from_spec(full_control_spec)
sys.modules[full_control_spec.name] = full_control_cli
full_control_spec.loader.exec_module(full_control_cli)
full_control_main = full_control_cli.main


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


STABLE_ROW_SORT_COLUMNS = ["dgp", "n", "p", "pi", "tau", "rep", "seed", "estimator"]


def _sort_result_rows(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(STABLE_ROW_SORT_COLUMNS).reset_index(drop=True)


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


def test_run_small_simulation_default_grid_has_9_points() -> None:
    results = run_small_simulation(
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


def test_run_small_simulation_explicit_grid_size_has_17_points() -> None:
    results = run_small_simulation(
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


def test_valid_estimators_includes_oracle() -> None:
    assert "oracle" in VALID_ESTIMATORS


def test_default_simulation_estimators_are_main_estimators() -> None:
    assert DEFAULT_SIMULATION_ESTIMATORS == ("oracle", "post_selection", "dml")


def test_default_dml_k_folds_is_three() -> None:
    assert DEFAULT_DML_K_FOLDS == 3


def test_default_n_jobs_is_six() -> None:
    assert DEFAULT_N_JOBS == 6


def test_default_quantreg_max_iter_is_1000() -> None:
    assert DEFAULT_QUANTREG_MAX_ITER == 1000


def test_quantreg_iteration_warning_filter_suppresses_by_default() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with quantreg_iteration_warning_filter(show_warnings=False):
            warnings.warn("iteration limit", IterationLimitWarning)

    assert caught == []


def test_quantreg_iteration_warning_filter_can_show_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with quantreg_iteration_warning_filter(show_warnings=True):
            warnings.warn("iteration limit", IterationLimitWarning)

    assert len(caught) == 1
    assert issubclass(caught[0].category, IterationLimitWarning)


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


def test_run_small_simulation_bias_logic() -> None:
    results = run_small_simulation(reps=1, n=80, p=5, alphas=np.linspace(0.0, 2.0, 5))

    for row in results.to_dict("records"):
        if row["alpha_hat"] is not None and not np.isnan(row["alpha_hat"]):
            assert row["bias"] == row["alpha_hat"] - row["alpha_true"]


def test_run_small_simulation_does_not_write_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    run_small_simulation(reps=1, n=80, p=5, alphas=np.linspace(0.0, 2.0, 5))

    assert not Path("results/raw/small_simulation_results.csv").exists()


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


def test_run_simulation_batch_returns_expected_rows() -> None:
    designs = [
        Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123),
        Design("dgp1", 80, 5, 1.0, 0.5, rep=1, seed=124),
    ]

    results = run_simulation_batch(
        designs,
        np.linspace(0.0, 2.0, 5),
        estimators=("post_selection", "dml"),
        n_jobs=1,
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
        n_jobs=1,
    )

    written = pd.read_csv(output_path)
    assert len(written) == 1
    assert written.loc[0, "estimator"] == "post_selection_ivqr"


def test_run_simulation_batch_parallel_writes_valid_csv(tmp_path: Path) -> None:
    output_path = tmp_path / "parallel_batch.csv"
    designs = [
        Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123),
        Design("dgp1", 80, 5, 1.0, 0.5, rep=1, seed=124),
    ]

    results = run_simulation_batch(
        designs,
        np.linspace(0.0, 2.0, 5),
        estimators=("post_selection",),
        output_path=output_path,
        n_jobs=2,
    )

    written = pd.read_csv(output_path)
    assert len(results) == 2
    assert len(written) == 2
    assert set(written["rep"]) == {0, 1}
    assert set(written["estimator"]) == {"post_selection_ivqr"}


def test_run_simulation_batch_serial_and_parallel_are_equivalent() -> None:
    designs = [
        Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123),
        Design("dgp1", 80, 5, 1.0, 0.5, rep=1, seed=124),
    ]
    alphas = np.linspace(0.0, 2.0, 5)

    serial = _sort_result_rows(
        run_simulation_batch(
            designs,
            alphas,
            estimators=("post_selection",),
            n_jobs=1,
        )
    )
    parallel = _sort_result_rows(
        run_simulation_batch(
            designs,
            alphas,
            estimators=("post_selection",),
            n_jobs=2,
        )
    )

    assert len(serial) == len(parallel)
    pd.testing.assert_frame_equal(
        serial[STABLE_ROW_SORT_COLUMNS + ["status"]],
        parallel[STABLE_ROW_SORT_COLUMNS + ["status"]],
    )
    np.testing.assert_allclose(
        serial["alpha_hat"].to_numpy(dtype=float),
        parallel["alpha_hat"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_run_simulation_batch_rejects_invalid_n_jobs() -> None:
    with pytest.raises(ValueError, match="n_jobs must be at least 1"):
        run_simulation_batch(
            [Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123)],
            np.linspace(0.0, 2.0, 5),
            estimators=("post_selection",),
            n_jobs=0,
        )


def test_parallel_resume_filters_completed_designs_without_duplicates(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "resume_parallel.csv"
    designs = [
        Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123),
        Design("dgp1", 80, 5, 1.0, 0.5, rep=1, seed=124),
    ]
    alphas = np.linspace(0.0, 2.0, 5)

    run_simulation_batch(
        [designs[0]],
        alphas,
        estimators=("post_selection",),
        output_path=output_path,
        n_jobs=1,
    )
    pending = filter_completed_designs(
        designs,
        output_path,
        estimators=("post_selection",),
    )
    run_simulation_batch(
        pending,
        alphas,
        estimators=("post_selection",),
        output_path=output_path,
        append=True,
        n_jobs=2,
    )

    written = pd.read_csv(output_path)
    assert len(written) == 2
    assert not written.duplicated(STABLE_ROW_SORT_COLUMNS).any()
    assert set(written["rep"]) == {0, 1}


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


def test_select_design_chunk_partitions_designs() -> None:
    designs = [
        Design("dgp1", 80, 5, 1.0, 0.5, rep=rep, seed=123 + rep)
        for rep in range(10)
    ]

    chunk_0 = select_design_chunk(designs, chunk_index=0, num_chunks=2)
    chunk_1 = select_design_chunk(designs, chunk_index=1, num_chunks=2)

    assert set(chunk_0).isdisjoint(chunk_1)
    assert sorted(chunk_0 + chunk_1, key=lambda design: design.rep) == designs


def test_full_simulation_dry_run_does_not_write_output_csv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_path = tmp_path / "dry_run.csv"
    manifest_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "reports" / "summary.csv"
    tables_dir = tmp_path / "reports" / "tables"
    figures_dir = tmp_path / "reports" / "figures"
    monkeypatch.setattr(
        "sys.argv",
        [
            "main_simulation.py",
            "--dry-run",
            "--dgps",
            "dgp1",
            "--n-values",
            "80",
            "--p-values",
            "10",
            "--pi-values",
            "1.0",
            "--taus",
            "0.5",
            "--reps",
            "1",
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--summary-output",
            str(summary_path),
            "--tables-dir",
            str(tables_dir),
            "--figures-dir",
            str(figures_dir),
        ],
    )

    full_simulation_main()

    assert not output_path.exists()
    assert not manifest_path.exists()
    assert not summary_path.exists()
    assert not tables_dir.exists()
    assert not figures_dir.exists()


def test_full_simulation_fast_mode_defaults_exclude_full_control() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="fast",
        estimators=None,
        dgps=None,
        n_values=None,
        p_values=None,
        pi_values=None,
        taus=None,
        reps=None,
        alpha_grid_size=None,
        output=None,
        summary_output=None,
        tables_dir=None,
        figures_dir=None,
    )

    full_simulation_cli._apply_mode_defaults(args)

    assert args.estimators == ["oracle", "post_selection", "dml"]
    assert "full" not in args.estimators
    assert args.dgps == ["dgp1", "dgp2", "dgp3"]
    assert args.n_values == [500, 1000]
    assert args.p_values == [200, 500]
    assert args.pi_values == [1.0, 0.5, 0.25, 0.10]
    assert args.taus == [0.25, 0.5, 0.75]
    assert args.reps == 10
    assert args.alpha_grid_size == 9
    assert args.output == Path("results/raw/fast_mode_results.csv")
    assert args.summary_output == Path("results/summary/main_simulation_summary.csv")
    assert args.tables_dir == Path("results/tables/main")
    assert args.figures_dir == Path("results/figures/main")


def test_full_simulation_full_mode_defaults_use_500_reps() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="full",
        estimators=None,
        dgps=None,
        n_values=None,
        p_values=None,
        pi_values=None,
        taus=None,
        reps=None,
        alpha_grid_size=None,
        output=None,
        summary_output=None,
        tables_dir=None,
        figures_dir=None,
    )

    full_simulation_cli._apply_mode_defaults(args)

    assert args.estimators == ["oracle", "post_selection", "dml"]
    assert args.dgps == ["dgp1", "dgp2", "dgp3"]
    assert args.n_values == [500, 1000]
    assert args.p_values == [200, 500]
    assert args.pi_values == [1.0, 0.5, 0.25, 0.10]
    assert args.taus == [0.25, 0.5, 0.75]
    assert args.reps == 500
    assert args.alpha_grid_size == 9
    assert args.output == Path("results/raw/full_mode_results.csv")
    assert args.summary_output == Path("results/summary/main_simulation_summary.csv")


def test_main_simulation_mode_outputs_are_separate() -> None:
    assert full_simulation_cli._default_output_for_mode("fast") == Path(
        "results/raw/fast_mode_results.csv"
    )
    assert full_simulation_cli._default_output_for_mode("full") == Path(
        "results/raw/full_mode_results.csv"
    )
    assert full_simulation_cli._default_output_for_mode(
        "fast"
    ) != full_simulation_cli._default_output_for_mode("full")


@pytest.mark.parametrize(
    ("mode", "expected_output"),
    [
        ("fast", Path("results/raw/fast_mode_results.csv")),
        ("full", Path("results/raw/full_mode_results.csv")),
    ],
)
def test_main_simulation_dry_run_reports_mode_output(
    mode: str,
    expected_output: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--mode", mode, "--dry-run"],
    )

    full_simulation_main()

    assert f"Output: {expected_output}" in capsys.readouterr().out


def test_full_control_dry_run_reports_default_output(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["full_control_ivqr.py", "--dry-run"],
    )

    full_control_main()

    expected_output = Path("results/raw/full_control_ivqr_results.csv")
    assert f"Output: {expected_output}" in capsys.readouterr().out


def test_full_simulation_mode_defaults_respect_explicit_overrides() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="fast",
        estimators=["oracle"],
        dgps=["dgp1"],
        n_values=[500],
        p_values=[200],
        pi_values=None,
        taus=None,
        reps=10,
        alpha_grid_size=3,
        output="custom.csv",
        summary_output="custom_summary.csv",
        tables_dir="custom_tables",
        figures_dir="custom_figures",
    )

    full_simulation_cli._apply_mode_defaults(args)

    assert args.estimators == ["oracle"]
    assert args.dgps == ["dgp1"]
    assert args.n_values == [500]
    assert args.p_values == [200]
    assert args.pi_values == [1.0, 0.5, 0.25, 0.10]
    assert args.taus == [0.25, 0.5, 0.75]
    assert args.reps == 10
    assert args.alpha_grid_size == 3
    assert args.output == Path("custom.csv")
    assert args.summary_output == "custom_summary.csv"
    assert args.tables_dir == "custom_tables"
    assert args.figures_dir == "custom_figures"


def test_full_simulation_main_mode_respects_alpha_grid_override() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="fast",
        estimators=None,
        dgps=None,
        n_values=None,
        p_values=None,
        pi_values=None,
        taus=None,
        reps=None,
        alpha_grid_size=13,
        output=None,
        summary_output=None,
        tables_dir=None,
        figures_dir=None,
    )

    full_simulation_cli._apply_mode_defaults(args)

    assert args.estimators == ["oracle", "post_selection", "dml"]
    assert args.alpha_grid_size == 13


def test_full_simulation_parser_dml_k_folds_default_and_override(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py"])
    args = full_simulation_cli._parse_args()
    assert args.dml_k_folds == 3

    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--mode", "full", "--dml-k-folds", "5"],
    )
    args = full_simulation_cli._parse_args()
    assert args.dml_k_folds == 5


def test_full_simulation_parser_n_jobs_default_and_override(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py"])
    args = full_simulation_cli._parse_args()
    assert args.n_jobs == 6

    for n_jobs in (1, 4, 6):
        monkeypatch.setattr(
            "sys.argv",
            ["main_simulation.py", "--mode", "full", "--n-jobs", str(n_jobs)],
        )
        args = full_simulation_cli._parse_args()
        assert args.n_jobs == n_jobs


def test_full_simulation_parser_quantreg_max_iter_default_and_override(
    monkeypatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["main_simulation.py"])
    args = full_simulation_cli._parse_args()
    assert args.quantreg_max_iter == 1000
    assert args.show_quantreg_warnings is False

    monkeypatch.setattr(
        "sys.argv",
        [
            "main_simulation.py",
            "--mode",
            "full",
            "--quantreg-max-iter",
            "2000",
            "--show-quantreg-warnings",
        ],
    )
    args = full_simulation_cli._parse_args()
    assert args.quantreg_max_iter == 2000
    assert args.show_quantreg_warnings is True


def test_full_simulation_rejects_invalid_n_jobs(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--dry-run", "--n-jobs", "0"],
    )
    with pytest.raises(ValueError, match="--n-jobs must be at least 1"):
        full_simulation_main()

    monkeypatch.setattr(
        "sys.argv",
        ["main_simulation.py", "--dry-run", "--n-jobs", "-1"],
    )
    with pytest.raises(ValueError, match="--n-jobs must be at least 1"):
        full_simulation_main()


def test_full_simulation_rejects_invalid_quantreg_max_iter(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "main_simulation.py",
            "--dry-run",
            "--quantreg-max-iter",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="--quantreg-max-iter must be at least 1"):
        full_simulation_main()


def test_main_runner_validation_rejects_full_control() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="fast",
        estimators=["full"],
        dgps=["dgp1"],
        n_values=[500],
        p_values=[100],
        pi_values=[1.0],
        taus=[0.5],
        reps=1,
        alpha_grid_size=3,
        output="manual.csv",
        summary_output=None,
        tables_dir=None,
        figures_dir=None,
        batch_size=1,
        n_jobs=1,
        dml_k_folds=3,
        quantreg_max_iter=1000,
        alpha_min=-1.0,
        alpha_max=3.0,
        chunk_index=None,
        num_chunks=None,
        max_designs=None,
    )

    with pytest.raises(ValueError, match="Main runner only allows"):
        full_simulation_cli._validate_args(args)


def test_manual_oracle_uses_single_scenario_defaults() -> None:
    args = full_simulation_cli.argparse.Namespace(
        mode="fast",
        estimators=["oracle"],
        dgps=None,
        n_values=[100],
        p_values=[20],
        pi_values=None,
        taus=None,
        reps=1,
        alpha_grid_size=3,
        output="oracle.csv",
        summary_output=None,
        tables_dir=None,
        figures_dir=None,
    )

    full_simulation_cli._apply_mode_defaults(args)

    assert args.estimators == ["oracle"]
    assert args.dgps == ["dgp1", "dgp2", "dgp3"]
    assert args.pi_values == [1.0, 0.5, 0.25, 0.10]
    assert args.taus == [0.25, 0.5, 0.75]
