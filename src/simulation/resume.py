"""Completion-key detection and resume filtering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dgp.designs import Design
from simulation.designs import DESIGN_KEY_COLUMNS, design_key, row_design_key
from simulation.dispatch import ESTIMATOR_OUTPUT_NAMES, validate_estimators
from simulation.output_schemas import ORACLE_DESIGN_KEY_COLUMNS


def _completed_successes(
    existing: pd.DataFrame, rerun_failed: bool
) -> pd.DataFrame:
    if not rerun_failed or "failed" not in existing.columns:
        return existing
    failed = existing["failed"].astype(str).str.lower().isin({"true", "1", "yes"})
    return existing.loc[~failed]


def filter_completed_designs(
    designs: list[Design],
    results_path: str | Path,
    estimators: tuple[str, ...],
    rerun_failed: bool = False,
) -> list[Design]:
    """Return designs that do not yet have all requested estimator rows."""
    path = Path(results_path)
    if not path.exists():
        return designs
    oracle_only = validate_estimators(estimators) == ("oracle",)
    header = tuple(pd.read_csv(path, nrows=0).columns)
    if oracle_only:
        required = list(ORACLE_DESIGN_KEY_COLUMNS)
        if "estimator" in header:
            required.append("estimator")
    else:
        required = list(DESIGN_KEY_COLUMNS) + ["estimator"]
    if rerun_failed and (not oracle_only or "failed" in header):
        required.append("failed")
    existing = _completed_successes(
        pd.read_csv(path, usecols=required), rerun_failed
    )
    expected = {
        ESTIMATOR_OUTPUT_NAMES[name] for name in validate_estimators(estimators)
    }
    completed: dict[tuple[object, ...], set[str]] = {}
    for _, row in existing.iterrows():
        if oracle_only:
            key = tuple(row[column] for column in ORACLE_DESIGN_KEY_COLUMNS)
            output_name = str(row["estimator"]) if "estimator" in row else "oracle"
        else:
            key = row_design_key(row)
            output_name = str(row["estimator"])
        completed.setdefault(key, set()).add(output_name)
    return [
        design
        for design in designs
        if not expected.issubset(
            completed.get(
                tuple(
                    getattr(design, column)
                    for column in ORACLE_DESIGN_KEY_COLUMNS
                )
                if oracle_only
                else design_key(design),
                set(),
            )
        )
    ]


__all__ = ["filter_completed_designs"]
