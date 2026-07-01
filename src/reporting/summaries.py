"""Aggregation of raw Monte Carlo results into scenario-level summaries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from ivqr.metrics import summarize_group, validate_metric_input


GROUP_COLUMNS: tuple[str, ...] = ("dgp", "n", "p", "pi", "tau", "estimator")

RAW_UNIQUE_COLUMNS: tuple[str, ...] = (
    "dgp",
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "seed",
    "estimator",
)

SUMMARY_METRIC_COLUMNS: tuple[str, ...] = (
    "replications",
    "valid_estimates",
    "bias",
    "median_bias",
    "rmse",
    "mae",
    "coverage",
    "coverage_valid_only",
    "avg_cr_length",
    "avg_cr_length_valid_only",
    "avg_cr_hull_length",
    "failure_rate",
    "non_convergence_rate",
    "cr_empty_rate",
    "cr_disconnected_rate",
    "boundary_rate",
    "alpha_hat_boundary_rate",
    "cr_boundary_hit_rate",
    "mean_failed_alpha_count",
    "mean_failed_alpha_rate",
    "mean_selected_controls",
    "critical_value_multiplier",
    "ps_selection_lasso_multiplier",
    "mean_critical_value_adjusted",
    "mean_runtime_seconds",
    "mean_runtime_total_sec",
    "median_runtime_total_sec",
    "mean_runtime_alpha_grid_sec",
    "mean_runtime_confidence_region_sec",
    "mean_dml_runtime_crossfit_sec",
    "mean_ps_runtime_selection_sec",
)

__all__ = [
    "GROUP_COLUMNS",
    "RAW_UNIQUE_COLUMNS",
    "SUMMARY_METRIC_COLUMNS",
    "aggregate_results",
    "aggregate_results_file",
    "compare_result_files",
    "incomplete_groups",
    "load_raw_results",
    "save_summary",
    "validate_no_duplicate_raw_rows",
]


def _assert_dataframe(name: str, value: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if value.columns.duplicated().any():
        duplicated = sorted(set(value.columns[value.columns.duplicated()].astype(str)))
        raise ValueError(f"{name} has duplicate columns: {duplicated}")
    return value


def _validate_expected_replications(
    expected_replications: int | None,
) -> int | None:
    if expected_replications is None:
        return None
    if not isinstance(expected_replications, int) or isinstance(
        expected_replications, bool
    ):
        raise ValueError("expected_replications must be a positive integer or None")
    if expected_replications <= 0:
        raise ValueError("expected_replications must be positive")
    return expected_replications


def _validate_output_path(path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise ValueError("output path must be a file path")
    if output_path.parent.exists() and not output_path.parent.is_dir():
        raise ValueError("output path parent must be a directory")
    return output_path


def _validate_group_columns(df: pd.DataFrame) -> None:
    _assert_dataframe("raw results", df)
    missing = sorted(set(GROUP_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"raw results are missing required group columns: {missing}")


def load_raw_results(path: str | Path) -> pd.DataFrame:
    """Load raw simulation results from CSV and validate required columns."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not csv_path.is_file():
        raise ValueError("raw results path must be a file")

    raw = pd.read_csv(csv_path)
    validate_metric_input(raw)
    _validate_group_columns(raw)
    return raw


def validate_no_duplicate_raw_rows(raw: pd.DataFrame) -> None:
    """Reject duplicate estimator-replication rows when the full raw key exists."""
    _assert_dataframe("raw results", raw)
    if not set(RAW_UNIQUE_COLUMNS).issubset(raw.columns):
        return

    duplicates = raw.duplicated(list(RAW_UNIQUE_COLUMNS), keep=False)
    if duplicates.any():
        duplicate_count = int(duplicates.sum())
        raise ValueError(
            "duplicate raw rows detected: "
            f"{duplicate_count} rows share key columns {list(RAW_UNIQUE_COLUMNS)}"
        )


def _observed_replications(group: pd.DataFrame) -> int:
    if "rep" not in group.columns:
        return int(len(group))

    raw_reps = group["rep"]
    converted = pd.to_numeric(raw_reps, errors="coerce")
    bad_nonmissing = raw_reps.notna() & converted.isna()
    if bad_nonmissing.any():
        raise ValueError("rep values must be numeric")

    reps = converted.dropna()
    if reps.empty:
        return 0
    if not np.all(np.isfinite(reps)):
        raise ValueError("rep values must be finite")
    if not np.all(reps >= 0):
        raise ValueError("rep values must be nonnegative")
    if not np.all(reps == np.floor(reps)):
        raise ValueError("rep values must be integer-valued")
    return int(reps.nunique())


def aggregate_results(
    raw: pd.DataFrame,
    expected_replications: int | None = None,
) -> pd.DataFrame:
    """Aggregate raw estimator-level rows by scenario and estimator."""
    expected_replications = _validate_expected_replications(expected_replications)
    validate_metric_input(raw)
    _validate_group_columns(raw)
    validate_no_duplicate_raw_rows(raw)

    rows: list[dict[str, object]] = []
    for key, group in raw.groupby(list(GROUP_COLUMNS), dropna=False, sort=True):
        row = cast(
            dict[str, object],
            dict(zip(GROUP_COLUMNS, key, strict=True)),
        )
        row.update(summarize_group(group))

        observed_replications = _observed_replications(group)
        row["expected_replications"] = expected_replications
        row["observed_replications"] = observed_replications
        row["completion_rate"] = (
            observed_replications / expected_replications
            if expected_replications is not None
            else None
        )
        rows.append(row)

    columns = list(GROUP_COLUMNS + SUMMARY_METRIC_COLUMNS) + [
        "expected_replications",
        "observed_replications",
        "completion_rate",
    ]
    summary = pd.DataFrame(rows, columns=columns)
    if summary.empty:
        return summary
    return summary.sort_values(
        list(GROUP_COLUMNS),
        kind="mergesort",
    ).reset_index(drop=True)


def save_summary(summary: pd.DataFrame, path: str | Path) -> None:
    """Save an aggregated summary CSV."""
    _assert_dataframe("summary", summary)
    output_path = _validate_output_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def aggregate_results_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    expected_replications: int | None = None,
) -> pd.DataFrame:
    """Load, aggregate, optionally save, and return simulation summary metrics."""
    summary = aggregate_results(
        load_raw_results(input_path),
        expected_replications=expected_replications,
    )
    if output_path is not None:
        save_summary(summary, output_path)
    return summary


def compare_result_files(
    result_paths: Sequence[str | Path],
    labels: Sequence[str],
    expected_replications: int | None = None,
) -> pd.DataFrame:
    """Aggregate and label multiple raw result files for sensitivity comparisons."""
    if isinstance(result_paths, (str, Path)):
        raise ValueError("result_paths must be a sequence of paths")
    if isinstance(labels, str):
        raise ValueError("labels must be a sequence of strings")
    paths = list(result_paths)
    labels = list(labels)
    if not paths:
        raise ValueError("at least one result path is required")
    if len(paths) != len(labels):
        raise ValueError("result_paths and labels must have the same length")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("labels must contain nonempty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must not contain duplicates")

    summaries = []
    for path, label in zip(paths, labels, strict=True):
        summary = aggregate_results_file(
            path,
            expected_replications=expected_replications,
        )
        summary.insert(0, "run_label", label)
        summary.insert(1, "source_file", str(path))
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def incomplete_groups(summary: pd.DataFrame) -> pd.DataFrame:
    """Return summary rows with completion_rate below one."""
    _assert_dataframe("summary", summary)
    if "completion_rate" not in summary.columns:
        return summary.iloc[0:0].copy()

    completion = pd.to_numeric(summary["completion_rate"], errors="coerce")
    if completion.dropna().empty:
        return summary.iloc[0:0].copy()
    return summary.loc[completion < 1].copy()

