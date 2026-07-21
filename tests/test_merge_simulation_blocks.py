from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from simulation.oracle_output import ORACLE_OUTPUT_COLUMNS

SCRIPT = Path(__file__).parents[1] / "scripts" / "merge_simulation_blocks.py"
SPEC = importlib.util.spec_from_file_location("merge_simulation_blocks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
merge_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = merge_module
SPEC.loader.exec_module(merge_module)
MergeValidationError = merge_module.MergeValidationError
merge_simulation_blocks = merge_module.merge_simulation_blocks


def _frame(reps: range) -> pd.DataFrame:
    rows = []
    for dgp in ("dgp1", "dgp2"):
        for rep in reps:
            rows.append(
                {
                    "dgp": dgp,
                    "n": 500,
                    "p": 200,
                    "pi": 1.0,
                    "tau": 0.5,
                    "rep": rep,
                    "seed": 10_000 + rep,
                    "result_schema_version": 4,
                    "estimator": "oracle",
                    "alpha_hat": 0.5,
                    "alpha_true": 0.5,
                    "cr_lower": 0.0,
                    "cr_upper": 1.0,
                    "cr_length": 1.0,
                    "covered": True,
                    "converged": True,
                    "cr_components": "[[0.0,1.0]]",
                    "cr_n_blocks": 1,
                    "cr_disconnected": False,
                    "cr_status": "valid",
                    "cr_is_numerically_resolved": True,
                    "cr_unresolved_count": 0,
                    "cr_unresolved_alphas": "[]",
                    "final_alpha_evaluations": 31,
                    "refinement_depth_reached": 4,
                    "number_of_refined_intervals": 5,
                    "minimum_final_grid_spacing": 0.01,
                    "median_final_grid_spacing": 0.05,
                    "grid_strategy": "adaptive",
                    "adaptive_midpoint_probe": True,
                    "alpha_hat_grid": "initial",
                    "midpoint_intervals_considered": 10,
                    "midpoint_evaluations_added": 5,
                    "midpoint_unresolved_barriers": 0,
                    "refinement_tolerance": 0.025,
                    "initial_alpha_grid_size": 21,
                    "refinement_limit_hit": False,
                    "midpoint_probe_limit_hit": False,
                    "max_alpha_evaluations_hit": False,
                    "number_of_unresolved_refinement_barriers": 0,
                    "maximum_final_grid_spacing": 0.1,
                    "rank_deficient_covariance_failures": 0,
                    "iteration_warning_evaluations": 0,
                }
            )
    return pd.DataFrame(rows)


def _inputs(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / "result_block_000_001.csv", tmp_path / "result_block_002_003.csv"]
    _frame(range(0, 2)).to_csv(paths[0], index=False)
    _frame(range(2, 4)).to_csv(paths[1], index=False)
    return paths


def _merge(tmp_path: Path, inputs: list[Path], **overrides):
    arguments = {
        "inputs": inputs,
        "output": tmp_path / "merged.csv",
        "manifest": tmp_path / "merged_manifest.json",
        "estimator": "oracle",
        "expected_schema_version": 4,
        "expected_design_cells": 2,
        "expected_reps_per_cell": 4,
        "project_root": tmp_path,
    }
    arguments.update(overrides)
    return merge_simulation_blocks(**arguments)


def test_successful_merge(tmp_path: Path) -> None:
    payload = _merge(tmp_path, _inputs(tmp_path))
    assert payload["final_row_count"] == 8
    assert payload["design_cells"] == 2
    merged = pd.read_csv(tmp_path / "merged.csv")
    assert len(merged) == 8
    assert tuple(merged.columns) == ORACLE_OUTPUT_COLUMNS
    assert len(merged.columns) == 26


def test_duplicate_replication_key_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    frame = pd.read_csv(inputs[1])
    frame.loc[0, "dgp"] = "dgp1"
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    frame.to_csv(inputs[1], index=False)
    with pytest.raises(MergeValidationError, match="duplicate simulation keys"):
        _merge(tmp_path, inputs)


def test_missing_replication_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    frame = pd.read_csv(inputs[1])
    frame = frame.loc[~((frame["dgp"] == "dgp2") & (frame["rep"] == 3))]
    frame.to_csv(inputs[1], index=False)
    with pytest.raises(MergeValidationError, match="missing design-cell replications|row count"):
        _merge(tmp_path, inputs)


def test_historical_schema_version_is_ignored_after_oracle_projection(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    frame = pd.read_csv(inputs[1])
    frame["result_schema_version"] = 3
    frame.to_csv(inputs[1], index=False)
    _merge(tmp_path, inputs)
    assert tuple(pd.read_csv(tmp_path / "merged.csv").columns) == ORACLE_OUTPUT_COLUMNS


def test_historical_column_order_is_ignored_after_oracle_projection(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    frame = pd.read_csv(inputs[1])
    frame = frame[[*frame.columns[1:], frame.columns[0]]]
    frame.to_csv(inputs[1], index=False)
    _merge(tmp_path, inputs)
    assert tuple(pd.read_csv(tmp_path / "merged.csv").columns) == ORACLE_OUTPUT_COLUMNS


def test_wrong_estimator_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    frame = pd.read_csv(inputs[1])
    frame["estimator"] = "dml"
    frame.to_csv(inputs[1], index=False)
    with pytest.raises(MergeValidationError, match="estimator mismatch"):
        _merge(tmp_path, inputs)


def test_malformed_retained_cr_components_are_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    frame = pd.read_csv(inputs[1])
    frame.loc[0, "cr_components"] = "not-json"
    frame.to_csv(inputs[1], index=False)
    with pytest.raises(MergeValidationError, match="component"):
        _merge(tmp_path, inputs)


def test_inconsistent_retained_cr_n_blocks_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    frame = pd.read_csv(inputs[1])
    frame.loc[0, "cr_n_blocks"] = 2
    frame.to_csv(inputs[1], index=False)
    with pytest.raises(MergeValidationError, match="cr_n_blocks"):
        _merge(tmp_path, inputs)


def test_missing_oracle_analysis_column_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    frame = pd.read_csv(inputs[1]).drop(columns="cr_length")
    frame.to_csv(inputs[1], index=False)
    with pytest.raises(MergeValidationError, match="missing required output columns.*cr_length"):
        _merge(tmp_path, inputs)


def test_current_and_historical_oracle_blocks_can_be_merged(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    current = pd.read_csv(inputs[1]).rename(columns={"cr_covers_true": "covered"})
    current = current.loc[:, list(ORACLE_OUTPUT_COLUMNS)]
    current.to_csv(inputs[1], index=False)
    _merge(tmp_path, inputs)
    assert tuple(pd.read_csv(tmp_path / "merged.csv").columns) == ORACLE_OUTPUT_COLUMNS


def test_incorrect_total_row_count_is_rejected(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(MergeValidationError, match="expected 3 design cells"):
        _merge(tmp_path, inputs, expected_design_cells=3)


def test_failed_validation_preserves_existing_output(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    output = tmp_path / "merged.csv"
    output.write_bytes(b"existing-output\n")
    frame = pd.read_csv(inputs[1])
    frame = frame.drop(columns="cr_length")
    frame.to_csv(inputs[1], index=False)
    with pytest.raises(MergeValidationError):
        _merge(tmp_path, inputs, output=output)
    assert output.read_bytes() == b"existing-output\n"
    assert not list(tmp_path.glob(".merged.csv.*.tmp"))


def test_output_is_sorted_deterministically(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    for path in inputs:
        pd.read_csv(path).sample(frac=1.0, random_state=7).to_csv(path, index=False)
    _merge(tmp_path, inputs)
    merged = pd.read_csv(tmp_path / "merged.csv")
    expected = merged.sort_values(
        ["dgp", "n", "p", "pi", "tau", "rep"], kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(merged, expected)


def test_manifest_and_checksums_are_created(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    payload = _merge(tmp_path, inputs)
    manifest = json.loads((tmp_path / "merged_manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((tmp_path / "merged.csv").read_bytes()).hexdigest()
    assert manifest == payload
    assert manifest["final_csv_sha256"] == digest
    assert [item["sha256"] for item in manifest["input_files"]] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
    ]
    assert manifest["validation_summary"]["atomic_temporary_file"] == "passed"
