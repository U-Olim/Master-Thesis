"""Aggregation of raw Monte Carlo results into scenario-level summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ivqr_sim.metrics import summarize_group, validate_metric_input


GROUP_COLUMNS = ["dgp", "n", "p", "pi", "tau", "estimator"]
RAW_UNIQUE_COLUMNS = ["dgp", "n", "p", "pi", "tau", "rep", "seed", "estimator"]
SUMMARY_METRIC_COLUMNS = [
    "replications",
    "valid_estimates",
    "bias",
    "median_bias",
    "rmse",
    "mae",
    "coverage",
    "coverage_valid_only",
    "avg_cr_length",
    "failure_rate",
    "non_convergence_rate",
    "cr_empty_rate",
    "cr_disconnected_rate",
    "boundary_rate",
    "mean_failed_alpha_count",
    "mean_selected_controls",
    "mean_runtime_seconds",
]


def _validate_group_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(GROUP_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"raw results are missing required group columns: {missing}")


def load_raw_results(path: str | Path) -> pd.DataFrame:
    """Load raw simulation results from CSV and validate required columns."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    raw = pd.read_csv(csv_path)
    validate_metric_input(raw)
    _validate_group_columns(raw)
    return raw


def validate_no_duplicate_raw_rows(raw: pd.DataFrame) -> None:
    """Reject duplicate estimator-replication rows when the full raw key exists."""
    if not set(RAW_UNIQUE_COLUMNS).issubset(raw.columns):
        return

    duplicates = raw.duplicated(RAW_UNIQUE_COLUMNS, keep=False)
    if duplicates.any():
        duplicate_count = int(duplicates.sum())
        raise ValueError(
            "duplicate raw rows detected: "
            f"{duplicate_count} rows share key columns {RAW_UNIQUE_COLUMNS}"
        )


def aggregate_results(
    raw: pd.DataFrame,
    expected_replications: int | None = None,
) -> pd.DataFrame:
    """Aggregate raw estimator-level rows by scenario and estimator."""
    validate_metric_input(raw)
    _validate_group_columns(raw)
    validate_no_duplicate_raw_rows(raw)

    rows: list[dict[str, object]] = []
    for key, group in raw.groupby(GROUP_COLUMNS, dropna=False):
        row = dict(zip(GROUP_COLUMNS, key, strict=True))
        row.update(summarize_group(group))

        observed_replications = (
            int(group["rep"].nunique(dropna=True)) if "rep" in group.columns else int(len(group))
        )
        row["expected_replications"] = expected_replications
        row["observed_replications"] = observed_replications
        if expected_replications is not None and expected_replications > 0:
            row["completion_rate"] = observed_replications / expected_replications
        else:
            row["completion_rate"] = None
        rows.append(row)

    columns = GROUP_COLUMNS + SUMMARY_METRIC_COLUMNS + [
        "expected_replications",
        "observed_replications",
        "completion_rate",
    ]
    summary = pd.DataFrame(rows, columns=columns)
    if summary.empty:
        return summary
    return summary.sort_values(GROUP_COLUMNS).reset_index(drop=True)


def save_summary(summary: pd.DataFrame, path: str | Path) -> None:
    """Save an aggregated summary CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def aggregate_results_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    expected_replications: int | None = None,
) -> pd.DataFrame:
    """Load, aggregate, optionally save, and return simulation summary metrics."""
    raw = load_raw_results(input_path)
    summary = aggregate_results(raw, expected_replications=expected_replications)
    if output_path is not None:
        save_summary(summary, output_path)
    return summary


def incomplete_groups(summary: pd.DataFrame) -> pd.DataFrame:
    """Return summary rows with completion_rate below one."""
    if "completion_rate" not in summary.columns:
        return summary.iloc[0:0].copy()

    completion = pd.to_numeric(summary["completion_rate"], errors="coerce")
    if completion.dropna().empty:
        return summary.iloc[0:0].copy()
    return summary.loc[completion < 1].copy()
