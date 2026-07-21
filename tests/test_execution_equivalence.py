from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import simulation.runner as simulation_runner
from simulation.oracle_output import ORACLE_OUTPUT_COLUMNS
from simulation.results import RESULT_COLUMNS, RESULT_SCHEMA_VERSION
from simulation.runner import (
    filter_completed_designs,
    make_simulation_grid,
    run_simulation_batch,
)
from utils.timing import RUNTIME_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGE_SCRIPT = PROJECT_ROOT / "scripts" / "merge_simulation_blocks.py"
MERGE_SPEC = importlib.util.spec_from_file_location(
    "characterization_merge_simulation_blocks", MERGE_SCRIPT
)
assert MERGE_SPEC is not None and MERGE_SPEC.loader is not None
MERGE_MODULE = importlib.util.module_from_spec(MERGE_SPEC)
sys.modules[MERGE_SPEC.name] = MERGE_MODULE
MERGE_SPEC.loader.exec_module(MERGE_MODULE)
merge_simulation_blocks = MERGE_MODULE.merge_simulation_blocks

SMALL_ALPHAS = np.array([-1.0, 1.0, 3.0])
SORT_COLUMNS = ["dgp", "n", "p", "pi", "tau", "rep", "seed", "estimator"]
NONDETERMINISTIC_COLUMNS = {"runtime_seconds", *RUNTIME_COLUMNS}


def _designs(*, reps: int, rep_start: int = 0, rep_end: int | None = None):
    return make_simulation_grid(
        dgps=("dgp1",),
        n_values=(40,),
        p_values=(5,),
        pi_values=(0.5,),
        taus=(0.5,),
        reps=reps,
        base_seed=12345,
        rep_start=rep_start,
        rep_end=rep_end,
    )


def _canonical_internal(frame: pd.DataFrame) -> pd.DataFrame:
    retained = [
        column
        for column in frame.columns
        if column not in NONDETERMINISTIC_COLUMNS and "runtime" not in column
    ]
    return (
        frame.sort_values(SORT_COLUMNS, kind="mergesort")
        .reset_index(drop=True)
        .loc[:, retained]
    )


def _canonical_oracle(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["dgp", "n", "p", "pi", "tau", "rep"], kind="mergesort"
    ).reset_index(drop=True)


def _assert_oracle_science_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        _canonical_oracle(left),
        _canonical_oracle(right),
        check_dtype=False,
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )


def test_parallel_execution_matches_serial_execution() -> None:
    designs = _designs(reps=2)
    kwargs = {
        "estimators": ("oracle", "post_selection", "dml"),
        "quantreg_max_iter": 1000,
        "dml_k_folds": 2,
        "dml_quantile_penalty": 0.01,
    }
    serial = run_simulation_batch(designs, SMALL_ALPHAS, n_jobs=1, **kwargs)
    parallel = run_simulation_batch(designs, SMALL_ALPHAS, n_jobs=2, **kwargs)

    assert len(serial) == len(parallel) == 6
    expected_parallel_order = parallel.sort_values(
        SORT_COLUMNS, kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(parallel, expected_parallel_order, check_exact=True)
    pd.testing.assert_frame_equal(
        _canonical_internal(serial),
        _canonical_internal(parallel),
        check_dtype=False,
        check_exact=True,
    )


def test_block_merge_matches_uninterrupted_run(tmp_path: Path) -> None:
    full_designs = _designs(reps=4)
    uninterrupted = run_simulation_batch(
        full_designs,
        SMALL_ALPHAS,
        estimators=("oracle",),
        n_jobs=1,
    )

    first_path = tmp_path / "oracle_block_000_001.csv"
    second_path = tmp_path / "oracle_block_002_003.csv"
    run_simulation_batch(
        _designs(reps=4, rep_start=0, rep_end=1),
        SMALL_ALPHAS,
        estimators=("oracle",),
        output_path=first_path,
        n_jobs=1,
    )
    run_simulation_batch(
        _designs(reps=4, rep_start=2, rep_end=3),
        SMALL_ALPHAS,
        estimators=("oracle",),
        output_path=second_path,
        n_jobs=1,
    )
    output = tmp_path / "merged.csv"
    merge_simulation_blocks(
        inputs=[first_path, second_path],
        output=output,
        manifest=tmp_path / "merged_manifest.json",
        estimator="oracle",
        expected_schema_version=RESULT_SCHEMA_VERSION,
        expected_design_cells=1,
        expected_reps_per_cell=4,
        base_seed=12345,
        project_root=tmp_path,
    )
    merged = pd.read_csv(output)

    assert tuple(merged.columns) == ORACLE_OUTPUT_COLUMNS
    assert merged["rep"].tolist() == [0, 1, 2, 3]
    assert not merged.duplicated(["dgp", "n", "p", "pi", "tau", "rep"]).any()
    _assert_oracle_science_equal(merged, uninterrupted)


def test_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    designs = _designs(reps=4)
    clean = run_simulation_batch(
        designs,
        SMALL_ALPHAS,
        estimators=("oracle",),
        n_jobs=1,
    )
    output = tmp_path / "resumed.csv"
    run_simulation_batch(
        designs[:2],
        SMALL_ALPHAS,
        estimators=("oracle",),
        output_path=output,
        n_jobs=1,
    )
    pending = filter_completed_designs(designs, output, estimators=("oracle",))
    assert [design.rep for design in pending] == [2, 3]
    run_simulation_batch(
        pending,
        SMALL_ALPHAS,
        estimators=("oracle",),
        output_path=output,
        append=True,
        n_jobs=1,
    )
    resumed = pd.read_csv(output)

    assert resumed["rep"].tolist() == [0, 1, 2, 3]
    assert not resumed.duplicated(["dgp", "n", "p", "pi", "tau", "rep"]).any()
    _assert_oracle_science_equal(resumed, clean)


def test_full_mode_uses_union_schema_and_one_shared_dataset(monkeypatch) -> None:
    calls = 0
    production_generate_data = simulation_runner.generate_data

    def counted_generate_data(design):
        nonlocal calls
        calls += 1
        return production_generate_data(design)

    monkeypatch.setattr(simulation_runner, "generate_data", counted_generate_data)
    output = run_simulation_batch(
        _designs(reps=1),
        SMALL_ALPHAS,
        estimators=("oracle", "post_selection", "dml"),
        dml_k_folds=2,
        n_jobs=1,
    )

    assert calls == 1
    assert len(output) == 3
    assert tuple(output.columns) == RESULT_COLUMNS
    assert len(output.columns) == 150
    assert output["estimator"].tolist() == [
        "oracle",
        "post_selection_ivqr",
        "dml_ivqr",
    ]
    assert "bias" in output.columns
    assert "runtime_seconds" in output.columns
    assert tuple(output.columns) != ORACLE_OUTPUT_COLUMNS
