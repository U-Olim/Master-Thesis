"""Tests for simulation batching, resume, and output planning."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dgp.designs import Design
from simulation.batching import (
    completed_design_keys,
    filter_completed_designs,
    observed_design_keys,
    run_simulation_batch,
)
from simulation.chunking import select_design_chunk
from simulation.runner import run_small_simulation


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


import importlib.util
import sys

FULL_SIMULATION_SCRIPT = Path(__file__).resolve().parents[2] / "scenarios" / "main_simulation.py"
spec = importlib.util.spec_from_file_location("full_simulation_cli", FULL_SIMULATION_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {FULL_SIMULATION_SCRIPT}")
full_simulation_cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = full_simulation_cli
spec.loader.exec_module(full_simulation_cli)
full_simulation_main = full_simulation_cli.main

FULL_CONTROL_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scenarios" / "full_control_ivqr.py"
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


STABLE_ROW_SORT_COLUMNS = ["dgp", "n", "p", "pi", "tau", "rep", "seed", "estimator"]


def _sort_result_rows(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(STABLE_ROW_SORT_COLUMNS).reset_index(drop=True)


def test_run_small_simulation_does_not_write_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    run_small_simulation(reps=1, n=80, p=5, alphas=np.linspace(0.0, 2.0, 5))

    assert not Path("results/raw/small_simulation_results.csv").exists()


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


