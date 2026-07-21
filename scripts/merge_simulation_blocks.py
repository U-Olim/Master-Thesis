"""Merge completed simulation CSV blocks with strict validation and provenance."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from analysis.data import DESIGN_COLUMNS, IDENTIFIER_COLUMNS, sha256_file
from simulation.dml_output import validate_component_columns
from simulation.oracle_output import (
    ORACLE_DESIGN_KEY_COLUMNS,
    ORACLE_OUTPUT_COLUMNS,
    clean_oracle_results_frame,
)


SORT_COLUMNS = ["dgp", "n", "p", "pi", "tau", "rep", "estimator"]
DEFAULT_CORRECTED_CH_CONFIG: dict[str, object] = {
    "grid_strategy": "adaptive",
    "adaptive_midpoint_probe": True,
    "alpha_hat_grid": "initial",
    "iteration_warning_policy": "use_if_valid",
    "hard_failure_policy": "unresolved",
    "refinement_tolerance": 0.025,
    "max_refinement_depth": 10,
    "max_alpha_evaluations": 201,
}
_BLOCK_RANGE = re.compile(r"block_(\d+)_(\d+)(?:\.csv)?$")


class MergeValidationError(ValueError):
    """Raised when an input or merged result violates the merge contract."""


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _dtype_family(dtype: Any) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return "text"
    return str(dtype)


def _validate_constant(frame: pd.DataFrame, column: str, expected: object) -> None:
    if column not in frame.columns:
        return
    observed = frame[column].drop_duplicates().tolist()
    if len(observed) != 1:
        raise MergeValidationError(f"{column} is not constant: {observed!r}")
    actual = observed[0]
    if isinstance(expected, float):
        matches = bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
    else:
        matches = actual == expected
    if not matches:
        raise MergeValidationError(
            f"{column} mismatch: expected {expected!r}, observed {actual!r}"
        )


def _parse_block_range(path: Path) -> tuple[int, int]:
    match = _BLOCK_RANGE.search(path.name)
    if match is None:
        raise MergeValidationError(
            f"cannot infer replication range from input filename: {path.name}"
        )
    start, end = (int(match.group(1)), int(match.group(2)))
    if start > end:
        raise MergeValidationError(f"invalid replication range in {path.name}")
    return start, end


def _read_and_validate_inputs(
    inputs: Sequence[Path],
    *,
    estimator: str,
    expected_schema_version: int,
    corrected_ch_config: dict[str, object],
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    if not inputs:
        raise MergeValidationError("at least one input file is required")
    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, object]] = []
    reference_columns: list[str] | None = None
    reference_families: dict[str, str] | None = None

    for path in inputs:
        if not path.is_file():
            raise MergeValidationError(f"input file does not exist: {path}")
        if path.stat().st_size == 0:
            raise MergeValidationError(f"input file is empty: {path}")
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise MergeValidationError(f"input file is empty: {path}") from exc
        if frame.empty:
            raise MergeValidationError(f"input file has no data rows: {path}")

        if estimator == "oracle":
            if "estimator" in frame:
                _validate_constant(frame, "estimator", estimator)
            try:
                frame = clean_oracle_results_frame(frame)
            except (TypeError, ValueError) as exc:
                raise MergeValidationError(
                    f"Oracle input validation failed for {path}: {exc}"
                ) from exc

        columns = list(frame.columns)
        if len(columns) != len(set(columns)):
            raise MergeValidationError(f"input has duplicate column names: {path}")
        if reference_columns is None:
            reference_columns = columns
            reference_families = {
                column: _dtype_family(frame[column].dtype) for column in columns
            }
        elif columns != reference_columns:
            raise MergeValidationError(
                f"column names or order differ in {path}; expected {reference_columns}, "
                f"observed {columns}"
            )
        else:
            families = {column: _dtype_family(frame[column].dtype) for column in columns}
            incompatible = {
                column: (reference_families[column], families[column])
                for column in columns
                if families[column] != reference_families[column]
            }
            if incompatible:
                raise MergeValidationError(f"incompatible dtypes in {path}: {incompatible}")

        required = (
            set(ORACLE_OUTPUT_COLUMNS)
            if estimator == "oracle"
            else set(IDENTIFIER_COLUMNS) | {
                "result_schema_version",
                "cr_components",
                "cr_n_blocks",
                "cr_disconnected",
                "cr_lower",
                "cr_upper",
                "cr_length",
            }
        )
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise MergeValidationError(f"required columns missing from {path}: {missing}")
        if estimator != "oracle":
            _validate_constant(frame, "result_schema_version", expected_schema_version)
            _validate_constant(frame, "estimator", estimator)
            for column, expected in corrected_ch_config.items():
                _validate_constant(frame, column, expected)

        expected_start, expected_end = _parse_block_range(path)
        observed_reps = set(pd.to_numeric(frame["rep"], errors="raise").astype(int))
        expected_reps = set(range(expected_start, expected_end + 1))
        if observed_reps != expected_reps:
            raise MergeValidationError(
                f"replication range mismatch in {path}: expected "
                f"{expected_start}-{expected_end}, observed "
                f"{min(observed_reps)}-{max(observed_reps)} with "
                f"{len(expected_reps - observed_reps)} missing values"
            )

        frames.append(frame)
        metadata.append(
            {
                "path": path,
                "row_count": len(frame),
                "replication_range": [expected_start, expected_end],
            }
        )
    return frames, metadata


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    def true_count(column: str) -> int:
        if column not in frame:
            return 0
        return int(frame[column].fillna(False).astype(bool).sum())

    if "failed" in frame:
        estimator_failures = true_count("failed")
    elif "converged" in frame:
        estimator_failures = int((~frame["converged"].fillna(False).astype(bool)).sum())
    else:
        estimator_failures = 0
    statuses = frame.get("cr_status", pd.Series(index=frame.index, dtype="object"))
    unresolved_cr = int(
        statuses.isin(["partially_unresolved", "fully_unresolved"]).sum()
    )
    if "cr_is_numerically_resolved" in frame:
        unresolved_cr = max(
            unresolved_cr,
            int((frame["cr_is_numerically_resolved"] == False).sum()),  # noqa: E712
        )
    if "coverage_status" in frame:
        unresolved_coverage = int(
            frame["coverage_status"].eq("coverage_unresolved").sum()
        )
    else:
        unresolved_coverage = int(frame.get("covered", pd.Series(dtype=float)).isna().sum())
    rank_column = pd.to_numeric(
        frame.get("rank_deficient_covariance_failures", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0)
    return {
        "estimator_failures": estimator_failures,
        "unresolved_cr": unresolved_cr,
        "partially_unresolved_cr": int(statuses.eq("partially_unresolved").sum()),
        "unresolved_coverage": unresolved_coverage,
        "refinement_limit_hits": true_count("refinement_limit_hit"),
        "midpoint_probe_limit_hits": true_count("midpoint_probe_limit_hit"),
        "maximum_evaluation_limit_hits": true_count("max_alpha_evaluations_hit"),
        "rank_deficient_covariance_failures": int(rank_column.sum()),
        "rows_with_rank_deficient_covariance_failures": int((rank_column > 0).sum()),
    }


def _validate_merged_frame(
    frame: pd.DataFrame,
    *,
    estimator: str,
    expected_schema_version: int,
    expected_design_cells: int,
    expected_reps_per_cell: int,
    expected_columns: Sequence[str],
    require_sorted: bool,
) -> dict[str, object]:
    if list(frame.columns) != list(expected_columns):
        raise MergeValidationError("merged output columns differ from input columns")
    key_columns = (
        list(ORACLE_DESIGN_KEY_COLUMNS)
        if estimator == "oracle"
        else list(IDENTIFIER_COLUMNS)
    )
    for column in key_columns:
        if column not in frame:
            raise MergeValidationError(f"canonical key column is missing: {column}")
    missing_key_count = int(frame[key_columns].isna().any(axis=1).sum())
    if missing_key_count:
        raise MergeValidationError(f"found {missing_key_count} rows with missing key values")
    duplicate_count = int(frame.duplicated(key_columns, keep="first").sum())
    if duplicate_count:
        raise MergeValidationError(f"found {duplicate_count} duplicate simulation keys")
    if estimator != "oracle":
        _validate_constant(frame, "result_schema_version", expected_schema_version)
        _validate_constant(frame, "estimator", estimator)

    design_columns = (
        ["dgp", "n", "p", "pi", "tau"]
        if estimator == "oracle" else list(DESIGN_COLUMNS)
    )
    design_cells = frame.groupby(design_columns, dropna=False, sort=False)
    observed_cell_count = int(design_cells.ngroups)
    if observed_cell_count != expected_design_cells:
        raise MergeValidationError(
            f"expected {expected_design_cells} design cells, observed {observed_cell_count}"
        )
    expected_total = expected_design_cells * expected_reps_per_cell
    if len(frame) != expected_total:
        raise MergeValidationError(
            f"incorrect total row count: expected {expected_total}, observed {len(frame)}"
        )
    rep_min = int(frame["rep"].min())
    rep_max = int(frame["rep"].max())
    expected_reps = set(range(rep_min, rep_max + 1))
    if len(expected_reps) != expected_reps_per_cell:
        raise MergeValidationError(
            f"replication span {rep_min}-{rep_max} does not contain "
            f"{expected_reps_per_cell} replications"
        )
    missing_replication_count = 0
    bad_sizes: list[tuple[object, int]] = []
    for design, group in design_cells:
        reps = set(pd.to_numeric(group["rep"], errors="raise").astype(int))
        missing_replication_count += len(expected_reps - reps)
        if len(group) != expected_reps_per_cell:
            bad_sizes.append((design, len(group)))
    if missing_replication_count:
        raise MergeValidationError(
            f"found {missing_replication_count} missing design-cell replications"
        )
    if bad_sizes:
        raise MergeValidationError(f"design cells have incorrect row counts: {bad_sizes[:5]}")

    if estimator != "oracle":
        try:
            validate_component_columns(frame)
        except (TypeError, ValueError) as exc:
            raise MergeValidationError(
                f"confidence-region component validation failed: {exc}"
            ) from exc
    if require_sorted:
        sort_columns = (
            list(ORACLE_DESIGN_KEY_COLUMNS) if estimator == "oracle" else SORT_COLUMNS
        )
        expected_order = frame.sort_values(sort_columns, kind="mergesort").index
        if not expected_order.equals(pd.RangeIndex(len(frame))):
            raise MergeValidationError("merged output is not in canonical deterministic order")
    sizes = design_cells.size()
    return {
        "final_row_count": len(frame),
        "design_cells": observed_cell_count,
        "replications_per_design_cell": expected_reps_per_cell,
        "replication_range": [rep_min, rep_max],
        "rows_per_design_cell": {"minimum": int(sizes.min()), "maximum": int(sizes.max())},
        "duplicate_count": duplicate_count,
        "missing_key_count": missing_key_count,
        "missing_replication_count": missing_replication_count,
        "cr_component_validation": (
            "passed"
        ),
        "status_counts": _status_counts(frame),
        "sort_columns": (
            list(ORACLE_DESIGN_KEY_COLUMNS) if estimator == "oracle" else SORT_COLUMNS
        ),
    }


def _git_metadata(project_root: Path) -> dict[str, object | None]:
    def git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-c", f"safe.directory={project_root.as_posix()}", "-C", str(project_root), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    dirty_text = git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": None if dirty_text is None else bool(dirty_text),
    }


def merge_simulation_blocks(
    *,
    inputs: Sequence[str | Path],
    output: str | Path,
    manifest: str | Path,
    estimator: str,
    expected_schema_version: int,
    expected_design_cells: int,
    expected_reps_per_cell: int,
    base_seed: int = 12345,
    corrected_ch_config: dict[str, object] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate, merge, atomically publish, and describe simulation blocks."""
    input_paths = [Path(path) for path in inputs]
    output_path = Path(output)
    manifest_path = Path(manifest)
    root = Path(project_root) if project_root is not None else Path.cwd()
    config = dict(DEFAULT_CORRECTED_CH_CONFIG)
    if corrected_ch_config is not None:
        config.update(corrected_ch_config)
    frames, input_metadata = _read_and_validate_inputs(
        input_paths,
        estimator=estimator,
        expected_schema_version=expected_schema_version,
        corrected_ch_config=config,
    )
    merged = pd.concat(frames, ignore_index=True)
    sort_columns = (
        list(ORACLE_DESIGN_KEY_COLUMNS) if estimator == "oracle" else SORT_COLUMNS
    )
    merged = merged.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    summary = _validate_merged_frame(
        merged,
        estimator=estimator,
        expected_schema_version=expected_schema_version,
        expected_design_cells=expected_design_cells,
        expected_reps_per_cell=expected_reps_per_cell,
        expected_columns=frames[0].columns,
        require_sorted=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", delete=False,
            dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            merged.to_csv(handle, index=False, lineterminator="\n")
        written = pd.read_csv(temporary_path)
        written_summary = _validate_merged_frame(
            written,
            estimator=estimator,
            expected_schema_version=expected_schema_version,
            expected_design_cells=expected_design_cells,
            expected_reps_per_cell=expected_reps_per_cell,
            expected_columns=frames[0].columns,
            require_sorted=True,
        )
        if written_summary != summary:
            raise MergeValidationError("temporary-file validation summary changed after serialization")
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    input_records = []
    for item in input_metadata:
        path = item["path"]
        assert isinstance(path, Path)
        input_records.append(
            {
                "path": _relative_or_absolute(path, root),
                "row_count": item["row_count"],
                "replication_range": item["replication_range"],
                "sha256": sha256_file(path),
            }
        )
    payload: dict[str, object] = {
        "result_schema_version": expected_schema_version,
        "estimator": estimator,
        "input_files": input_records,
        "input_row_counts": {item["path"]: item["row_count"] for item in input_records},
        "input_replication_ranges": {
            item["path"]: item["replication_range"] for item in input_records
        },
        "final_output_path": _relative_or_absolute(output_path, root),
        **summary,
        "base_seed": base_seed,
        "corrected_ch_configuration": {**config, "base_seed": base_seed},
        **_git_metadata(root),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "final_csv_sha256": sha256_file(output_path),
        "validation_summary": {
            "inputs": "passed",
            "schema_and_estimator": "passed",
            "replications": "passed",
            "unique_keys": "passed",
            "confidence_region_components": (
                "passed"
            ),
            "atomic_temporary_file": "passed",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", delete=False,
            dir=manifest_path.parent, prefix=f".{manifest_path.name}.", suffix=".tmp",
        ) as handle:
            manifest_temp = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        json.loads(manifest_temp.read_text(encoding="utf-8"))
        os.replace(manifest_temp, manifest_path)
        manifest_temp = None
    finally:
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--estimator", required=True)
    parser.add_argument("--expected-schema-version", required=True, type=int)
    parser.add_argument("--expected-design-cells", required=True, type=int)
    parser.add_argument("--expected-reps-per-cell", required=True, type=int)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = merge_simulation_blocks(
        inputs=args.inputs,
        output=args.output,
        manifest=args.manifest,
        estimator=args.estimator,
        expected_schema_version=args.expected_schema_version,
        expected_design_cells=args.expected_design_cells,
        expected_reps_per_cell=args.expected_reps_per_cell,
        base_seed=args.base_seed,
        project_root=args.project_root,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
