"""Tests for simulation batching, resume, and output planning."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dgp.designs import Design
import simulation.batching as batching_module
import simulation.chunking as chunking_module
from simulation.batching import (
    _as_bool,
    completed_design_keys,
    filter_completed_designs,
    observed_design_keys,
    run_simulation_batch,
)
from simulation.chunking import select_design_chunk, validate_chunk_args
from simulation.runner import run_small_simulation
from tests.helpers import (
    SIMULATION_RESULT_REQUIRED_KEYS,
    load_full_control_cli,
    load_main_simulation_cli,
)


full_simulation_cli = load_main_simulation_cli()
full_simulation_main = full_simulation_cli.main

full_control_cli = load_full_control_cli()
full_control_main = full_control_cli.main


STABLE_ROW_SORT_COLUMNS = ["dgp", "n", "p", "pi", "tau", "rep", "seed", "estimator"]


def _sort_result_rows(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(STABLE_ROW_SORT_COLUMNS).reset_index(drop=True)


def test_run_small_simulation_does_not_write_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    run_small_simulation(reps=1, n=80, p=5, alphas=np.linspace(0.0, 2.0, 5))

    assert not Path("results/raw/small_simulation_results.csv").exists()


def test_batching_and_chunking_public_apis_exclude_private_helpers() -> None:
    assert "run_simulation_batch" in batching_module.__all__
    assert "_as_bool" not in batching_module.__all__
    assert chunking_module.__all__ == ["select_design_chunk", "validate_chunk_args"]


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
    assert SIMULATION_RESULT_REQUIRED_KEYS.issubset(results.columns)


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


@pytest.mark.slow
@pytest.mark.filterwarnings(
    r"ignore:This process .* is multi-threaded, use of fork\(\) may lead to deadlocks in the child:DeprecationWarning"
)
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


@pytest.mark.slow
@pytest.mark.filterwarnings(
    r"ignore:This process .* is multi-threaded, use of fork\(\) may lead to deadlocks in the child:DeprecationWarning"
)
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
    with pytest.raises(ValueError, match="n_jobs must be positive"):
        run_simulation_batch(
            [Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123)],
            np.linspace(0.0, 2.0, 5),
            estimators=("post_selection",),
            n_jobs=0,
        )


@pytest.mark.slow
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


def test_filter_completed_designs_removes_fully_completed_design(
    tmp_path: Path,
) -> None:
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


def test_filter_completed_designs_keeps_partially_completed_design(
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
        Design("dgp1", 80, 5, 1.0, 0.5, rep=rep, seed=123 + rep) for rep in range(10)
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


@pytest.mark.parametrize("n_jobs", [True, 0, 1.5])
def test_run_simulation_batch_rejects_strictly_invalid_n_jobs(n_jobs) -> None:
    with pytest.raises(ValueError):
        run_simulation_batch(
            [Design("dgp1", 80, 5, 1.0, 0.5, rep=0, seed=123)],
            np.linspace(0.0, 2.0, 5),
            estimators=("post_selection",),
            n_jobs=n_jobs,
        )


def test_run_simulation_batch_rejects_nonboolean_append() -> None:
    with pytest.raises(ValueError, match="append must be a boolean"):
        run_simulation_batch(
            [],
            np.linspace(0.0, 2.0, 5),
            append="yes",
        )


@pytest.mark.parametrize(
    "alphas",
    [
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([1.0, 0.0]),
        np.array([0.0, 0.0]),
    ],
)
def test_run_simulation_batch_rejects_invalid_alphas(alphas) -> None:
    with pytest.raises(ValueError):
        run_simulation_batch([], alphas)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (1.0, True),
        (0, False),
        (0.0, False),
        (2, False),
        (-1, False),
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("maybe", False),
    ],
)
def test_as_bool_parses_only_explicit_boolean_values(value, expected) -> None:
    assert _as_bool(value) is expected


def test_observed_design_keys_rejects_invalid_key_values(tmp_path: Path) -> None:
    output_path = tmp_path / "invalid_keys.csv"
    pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": "invalid",
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
            }
        ]
    ).to_csv(output_path, index=False)

    with pytest.raises(
        ValueError,
        match="results CSV contains invalid design-key values",
    ):
        observed_design_keys(output_path)


@pytest.mark.parametrize(
    ("chunk_index", "num_chunks"),
    [
        (True, 2),
        (0, True),
        (1.5, 2),
        (0, 2.5),
        ("0", 2),
        (0, "2"),
    ],
)
def test_validate_chunk_args_rejects_bool_and_noninteger_values(
    chunk_index,
    num_chunks,
) -> None:
    with pytest.raises(ValueError):
        validate_chunk_args(chunk_index, num_chunks)


def test_run_simulation_batch_rejects_invalid_design_without_output() -> None:
    with pytest.raises(ValueError, match="design must be a Design object"):
        run_simulation_batch([object()], np.linspace(0.0, 2.0, 5), n_jobs=1)


@pytest.mark.parametrize("dml_k_folds", [1, 81])
def test_run_simulation_batch_rejects_infeasible_dml_folds(dml_k_folds) -> None:
    with pytest.raises(ValueError, match="dml_k_folds"):
        run_simulation_batch(
            [Design("dgp1", 80, 5, 1.0, 0.5, 0, 123)],
            np.linspace(0.0, 2.0, 5),
            dml_k_folds=dml_k_folds,
            n_jobs=1,
        )


def test_run_simulation_batch_rejects_invalid_output_paths(tmp_path: Path) -> None:
    design = Design("dgp1", 80, 5, 1.0, 0.5, 0, 123)

    with pytest.raises(ValueError, match="file path"):
        run_simulation_batch(
            [design],
            np.linspace(0.0, 2.0, 5),
            estimators=("post_selection",),
            output_path=tmp_path,
            n_jobs=1,
        )

    parent_file = tmp_path / "parent"
    parent_file.write_text("not a directory")
    with pytest.raises(ValueError, match="parent must be a directory"):
        run_simulation_batch(
            [design],
            np.linspace(0.0, 2.0, 5),
            estimators=("post_selection",),
            output_path=parent_file / "results.csv",
            n_jobs=1,
        )


def test_resume_helpers_reject_directory_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="results_path must be a file"):
        observed_design_keys(tmp_path)
    with pytest.raises(ValueError, match="results_path must be a file"):
        filter_completed_designs(
            [],
            tmp_path,
            estimators=("post_selection",),
        )


def test_filter_completed_designs_rejects_nonboolean_rerun_failed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="rerun_failed must be a boolean"):
        filter_completed_designs(
            [],
            tmp_path / "missing.csv",
            estimators=("post_selection",),
            rerun_failed="yes",
        )


def test_resume_key_parsing_rejects_decimal_integer_fields(tmp_path: Path) -> None:
    output_path = tmp_path / "invalid_keys.csv"
    pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": 80.5,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
            }
        ]
    ).to_csv(output_path, index=False)

    with pytest.raises(ValueError, match="invalid design-key values"):
        observed_design_keys(output_path)


def test_select_design_chunk_rejects_invalid_design_iterables() -> None:
    with pytest.raises(ValueError, match="iterable"):
        select_design_chunk("abc", 0, 1)
    with pytest.raises(ValueError, match="iterable"):
        select_design_chunk(1, 0, 1)


def test_select_design_chunk_accepts_generator() -> None:
    assert select_design_chunk((value for value in range(5)), 1, 2) == [1, 3]
