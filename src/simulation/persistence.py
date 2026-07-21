"""Estimator-specific result projection and CSV persistence."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd

from simulation.dml_output import clean_dml_results_frame
from simulation.oracle_output import clean_oracle_results_frame
from simulation.output_schemas import ORACLE_OUTPUT_COLUMNS
from simulation.post_selection_output import clean_post_selection_results_frame
from simulation.results import RESULT_SCHEMA_VERSION


def prepare_results_frame(
    results: pd.DataFrame, estimators: tuple[str, ...]
) -> pd.DataFrame:
    if estimators == ("dml",):
        return clean_dml_results_frame(results)
    if estimators == ("oracle",):
        return clean_oracle_results_frame(results)
    if estimators == ("post_selection",):
        return clean_post_selection_results_frame(results)
    return results


def _project_historical_oracle(path: Path) -> None:
    existing = clean_oracle_results_frame(pd.read_csv(path))
    if tuple(pd.read_csv(path, nrows=0).columns) == ORACLE_OUTPUT_COLUMNS:
        return
    warnings.warn(
        f"Projecting historical Oracle output to the current output schema "
        f"before resume: {path}",
        UserWarning,
        stacklevel=4,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            temporary_path = Path(handle.name)
            existing.to_csv(handle, index=False)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def persist_results_frame(
    results: pd.DataFrame,
    output_path: str | Path,
    *,
    append: bool,
    estimators: tuple[str, ...],
) -> None:
    path = Path(output_path)
    if path.exists() and path.is_dir():
        raise ValueError("output_path must be a file path")
    if append and path.exists() and path.stat().st_size > 0:
        if estimators == ("oracle",):
            _project_historical_oracle(path)
        existing_header = pd.read_csv(path, nrows=0)
        existing_columns = tuple(existing_header.columns)
        if estimators != ("oracle",) and "result_schema_version" not in existing_columns:
            raise ValueError(
                "cannot append results with a different schema: existing file "
                "is legacy/unversioned"
            )
        if existing_columns != tuple(results.columns):
            raise ValueError(
                "cannot append results with a different schema: "
                f"existing file has {len(existing_columns)} columns, "
                f"new rows have {len(results.columns)}"
            )
        existing_versions = (
            np.array([RESULT_SCHEMA_VERSION])
            if estimators == ("oracle",)
            else pd.read_csv(path, usecols=["result_schema_version"])[
                "result_schema_version"
            ]
            .dropna()
            .unique()
        )
        if (
            len(existing_versions) != 1
            or int(existing_versions[0]) != RESULT_SCHEMA_VERSION
        ):
            raise ValueError(
                "cannot append results with an incompatible result schema "
                f"version; expected {RESULT_SCHEMA_VERSION}, observed "
                f"{existing_versions.tolist()}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        path,
        mode="a" if append else "w",
        header=not (append and path.exists()),
        index=False,
    )


__all__ = ["persist_results_frame", "prepare_results_frame"]
