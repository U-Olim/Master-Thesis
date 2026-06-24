"""Table builders for Monte Carlo summaries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
import re

import pandas as pd


TABLE_GROUP_COLUMNS: tuple[str, ...] = ("dgp", "n", "p", "pi", "tau")
ESTIMATOR_COLUMN: str = "estimator"
ESTIMATOR_ORDER: tuple[str, ...] = (
    "oracle",
    "oracle_ivqr",
    "post_selection_ivqr",
    "dml_ivqr",
    "full_control_ivqr",
)
ESTIMATOR_LABELS = MappingProxyType(
    {
        "oracle_ivqr": "Oracle IVQR",
        "oracle": "Oracle IVQR",
        "post_selection_ivqr": "Post-selection IVQR",
        "dml_ivqr": "DML-IVQR",
        "full_control_ivqr": "Full-control IVQR",
    }
)
ESTIMATOR_COLUMN_NAMES = MappingProxyType(
    {
        "oracle": "oracle",
        "oracle_ivqr": "oracle_ivqr",
        "post_selection_ivqr": "post_selection",
        "dml_ivqr": "dml",
        "full_control_ivqr": "full_control",
    }
)
CORE_METRICS: tuple[str, ...] = (
    "bias",
    "median_bias",
    "rmse",
    "mae",
    "coverage",
    "avg_cr_length",
    "failure_rate",
    "non_convergence_rate",
    "cr_empty_rate",
    "cr_disconnected_rate",
    "mean_runtime_seconds",
)
DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "replications",
    "valid_estimates",
    "expected_replications",
    "observed_replications",
    "completion_rate",
    "failure_rate",
    "non_convergence_rate",
    "cr_empty_rate",
    "cr_disconnected_rate",
    "avg_cr_length_valid_only",
    "boundary_rate",
    "mean_failed_alpha_count",
    "mean_selected_controls",
    "mean_runtime_seconds",
)
WIDE_TABLE_METRICS = MappingProxyType(
    {
        "bias": "bias_wide.csv",
        "rmse": "rmse_wide.csv",
        "mae": "mae_wide.csv",
        "coverage": "coverage_wide.csv",
        "avg_cr_length": "cr_length_wide.csv",
        "mean_runtime_seconds": "runtime_wide.csv",
        "failure_rate": "failure_rate_wide.csv",
    }
)

__all__ = [
    "CORE_METRICS",
    "DIAGNOSTIC_COLUMNS",
    "ESTIMATOR_COLUMN",
    "ESTIMATOR_COLUMN_NAMES",
    "ESTIMATOR_LABELS",
    "ESTIMATOR_ORDER",
    "TABLE_GROUP_COLUMNS",
    "WIDE_TABLE_METRICS",
    "add_estimator_labels",
    "filter_summary",
    "load_summary",
    "make_comparison_table",
    "make_diagnostic_table",
    "make_wide_metric_table",
    "write_tables",
]


def _safe_column_name(value: object) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return name or "unknown"


def _wide_metric_column_name(estimator: object, metric: str) -> str:
    estimator_name = str(estimator)
    prefix = ESTIMATOR_COLUMN_NAMES.get(
        estimator_name,
        _safe_column_name(estimator_name),
    )
    return f"{prefix}_{metric}"


def _flatten_columns(columns: pd.Index) -> list[object]:
    if not isinstance(columns, pd.MultiIndex):
        return list(columns)
    flattened: list[object] = []
    for column in columns.to_flat_index():
        parts = [str(part) for part in column if part not in ("", None)]
        flattened.append("_".join(parts))
    return flattened


def _assert_unique_columns(df: pd.DataFrame) -> None:
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(
            f"table contains duplicate columns before rounding: {duplicated}"
        )


def _validate_summary_columns(summary: pd.DataFrame) -> None:
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame")
    if summary.columns.duplicated().any():
        duplicated = sorted(set(summary.columns[summary.columns.duplicated()].astype(str)))
        raise ValueError(f"summary has duplicate columns: {duplicated}")
    if summary.empty:
        raise ValueError("summary must not be empty")

    required = set(TABLE_GROUP_COLUMNS + (ESTIMATOR_COLUMN,))
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary is missing required columns: {missing}")


def _known_metric_columns(summary: pd.DataFrame) -> list[str]:
    known = set(CORE_METRICS + DIAGNOSTIC_COLUMNS + ("avg_cr_length",))
    return [column for column in summary.columns if column in known]


def _validate_metrics(
    summary: pd.DataFrame,
    metrics: Sequence[str],
) -> list[str]:
    if isinstance(metrics, str):
        raise ValueError("metrics must be a sequence of metric names")
    metrics = list(metrics)
    if not metrics:
        raise ValueError("at least one metric column is required")
    if any(not isinstance(metric, str) or not metric for metric in metrics):
        raise ValueError("metrics must contain nonempty strings")
    if len(set(metrics)) != len(metrics):
        raise ValueError("metrics must not contain duplicates")
    missing = [metric for metric in metrics if metric not in summary.columns]
    if missing:
        raise ValueError(f"summary is missing requested metric columns: {missing}")
    return metrics


def _validate_round_digits(round_digits: int | None) -> int | None:
    if round_digits is None:
        return None
    if not isinstance(round_digits, int) or isinstance(round_digits, bool):
        raise ValueError("round_digits must be an integer or None")
    if round_digits < 0:
        raise ValueError("round_digits must be nonnegative")
    return round_digits


def _round_numeric(
    df: pd.DataFrame,
    columns: Sequence[str],
    round_digits: int | None,
) -> pd.DataFrame:
    round_digits = _validate_round_digits(round_digits)
    for column in columns:
        if column in df.columns and isinstance(df[column], pd.DataFrame):
            raise ValueError(
                f"Column {column!r} is duplicated in table; "
                "fix wide-table column construction."
            )
    _assert_unique_columns(df)
    if round_digits is None:
        return df.copy()

    rounded = df.copy()
    for column in columns:
        if column not in rounded.columns:
            continue
        if isinstance(rounded[column], pd.DataFrame):
            raise ValueError(
                f"Column {column!r} is duplicated in table; "
                "fix wide-table column construction."
            )
        rounded[column] = pd.to_numeric(
            rounded[column], errors="coerce"
        ).round(round_digits)
    _assert_unique_columns(rounded)
    return rounded


def _validate_index_columns(
    summary: pd.DataFrame,
    index_columns: Sequence[str] | None,
) -> list[str]:
    if index_columns is None:
        return list(TABLE_GROUP_COLUMNS)
    if isinstance(index_columns, str):
        raise ValueError("index_columns must be a sequence of column names")
    index_columns = list(index_columns)
    if not index_columns:
        raise ValueError("index_columns must not be empty")
    if any(not isinstance(column, str) or not column for column in index_columns):
        raise ValueError("index_columns must contain nonempty strings")
    if len(set(index_columns)) != len(index_columns):
        raise ValueError("index_columns must not contain duplicates")
    missing = [column for column in index_columns if column not in summary.columns]
    if missing:
        raise ValueError(f"summary is missing requested index columns: {missing}")
    return index_columns


def load_summary(path: str | Path) -> pd.DataFrame:
    """Load an aggregated summary CSV and validate table-generation columns."""
    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = pd.read_csv(summary_path)
    _validate_summary_columns(summary)
    metric_columns = _known_metric_columns(summary)
    if not metric_columns:
        raise ValueError("summary must contain at least one metric column")
    if not any(
        pd.to_numeric(summary[column], errors="coerce").notna().any()
        for column in metric_columns
    ):
        raise ValueError("summary metric columns must contain at least one numeric value")
    return summary


def add_estimator_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add display labels and sort by scenario plus canonical estimator order."""
    _validate_summary_columns(df)
    labeled = df.copy()
    estimators = labeled[ESTIMATOR_COLUMN].astype(str)
    labeled["estimator_label"] = estimators.map(ESTIMATOR_LABELS).fillna(estimators)

    unknown = sorted(
        estimator for estimator in estimators.unique() if estimator not in ESTIMATOR_ORDER
    )
    categories = list(ESTIMATOR_ORDER) + unknown
    labeled[ESTIMATOR_COLUMN] = pd.Categorical(
        estimators,
        categories=categories,
        ordered=True,
    )
    return labeled.sort_values(
        list(TABLE_GROUP_COLUMNS) + [ESTIMATOR_COLUMN]
    ).reset_index(drop=True)


def filter_summary(
    summary: pd.DataFrame,
    dgp: str | None = None,
    n: int | None = None,
    p: int | None = None,
    pi: float | None = None,
    tau: float | None = None,
    estimators: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Filter an aggregated summary by scenario values and estimator names."""
    _validate_summary_columns(summary)
    if estimators is not None:
        if isinstance(estimators, str):
            raise ValueError("estimators must be a sequence of estimator names")
        estimators = tuple(estimators)
        if not estimators:
            raise ValueError("estimators must not be empty")
        if any(
            not isinstance(estimator, str) or not estimator
            for estimator in estimators
        ):
            raise ValueError("estimators must contain nonempty strings")
        if len(set(estimators)) != len(estimators):
            raise ValueError("estimators must not contain duplicates")

    filtered = summary.copy()
    filters = {"dgp": dgp, "n": n, "p": p, "pi": pi, "tau": tau}
    for column, value in filters.items():
        if value is not None:
            filtered = filtered.loc[filtered[column] == value]
    if estimators is not None:
        filtered = filtered.loc[filtered[ESTIMATOR_COLUMN].isin(estimators)]
    return filtered.copy()


def make_wide_metric_table(
    summary: pd.DataFrame,
    metric: str,
    index_columns: Sequence[str] | None = None,
    round_digits: int | None = 4,
) -> pd.DataFrame:
    """Create a wide scenario-by-estimator table for one metric."""
    _validate_summary_columns(summary)
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a nonempty string")
    if metric not in summary.columns:
        raise ValueError(f"metric column not found in summary: {metric}")
    _validate_metrics(summary, [metric])
    index_columns = _validate_index_columns(summary, index_columns)
    round_digits = _validate_round_digits(round_digits)

    duplicates = summary.duplicated(index_columns + [ESTIMATOR_COLUMN], keep=False)
    if duplicates.any():
        raise ValueError("duplicate scenario-estimator rows cannot be pivoted")

    labeled = add_estimator_labels(summary)
    labeled[metric] = pd.to_numeric(labeled[metric], errors="coerce")
    if labeled[metric].notna().sum() == 0:
        raise ValueError(f"metric column {metric!r} has no numeric values")
    wide = labeled.pivot(
        index=index_columns,
        columns=ESTIMATOR_COLUMN,
        values=metric,
    )
    wide.columns = _flatten_columns(wide.columns)

    existing_ordered = [
        estimator for estimator in ESTIMATOR_ORDER if estimator in wide.columns
    ]
    remaining = sorted(
        str(estimator)
        for estimator in wide.columns
        if estimator not in set(existing_ordered)
    )
    wide = wide[existing_ordered + remaining]
    wide = wide.rename(
        columns={
            estimator: _wide_metric_column_name(estimator, metric)
            for estimator in wide.columns
        }
    ).reset_index()

    metric_columns = [column for column in wide.columns if column not in index_columns]
    _assert_unique_columns(wide)
    return _round_numeric(wide, metric_columns, round_digits)


def make_comparison_table(
    summary: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    round_digits: int | None = 4,
) -> pd.DataFrame:
    """Create a long-format master result table with core metrics."""
    _validate_summary_columns(summary)
    selected_metrics = list(CORE_METRICS) if metrics is None else metrics
    selected_metrics = _validate_metrics(summary, selected_metrics)
    round_digits = _validate_round_digits(round_digits)

    table = add_estimator_labels(summary)
    columns = (
        list(TABLE_GROUP_COLUMNS)
        + [ESTIMATOR_COLUMN, "estimator_label"]
        + selected_metrics
    )
    table = table[columns].copy()
    for metric in selected_metrics:
        table[metric] = pd.to_numeric(table[metric], errors="coerce")
    if not any(table[metric].notna().any() for metric in selected_metrics):
        raise ValueError("comparison metrics have no numeric values")
    return _round_numeric(table, selected_metrics, round_digits)


def make_diagnostic_table(
    summary: pd.DataFrame,
    round_digits: int | None = 4,
) -> pd.DataFrame:
    """Create a long-format diagnostic table for completion and failure checks."""
    _validate_summary_columns(summary)
    round_digits = _validate_round_digits(round_digits)
    table = add_estimator_labels(summary)
    available_diagnostics = [
        column for column in DIAGNOSTIC_COLUMNS if column in table.columns
    ]
    columns = (
        list(TABLE_GROUP_COLUMNS)
        + [ESTIMATOR_COLUMN, "estimator_label"]
        + available_diagnostics
    )
    table = table[columns].copy()
    for diagnostic in available_diagnostics:
        table[diagnostic] = pd.to_numeric(table[diagnostic], errors="coerce")
    return _round_numeric(table, available_diagnostics, round_digits)


def write_tables(
    summary: pd.DataFrame,
    output_dir: str | Path,
    round_digits: int | None = 4,
) -> dict[str, Path]:
    """Write standard thesis-ready CSV tables and return their paths."""
    _validate_summary_columns(summary)
    round_digits = _validate_round_digits(round_digits)
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("output_dir must be a directory path")
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_metrics = [
        metric for metric in CORE_METRICS if metric in summary.columns
    ]
    if not comparison_metrics:
        raise ValueError("summary does not contain any core metrics for comparison table")

    written: dict[str, Path] = {}
    comparison_path = output_path / "comparison_table.csv"
    make_comparison_table(
        summary,
        metrics=comparison_metrics,
        round_digits=round_digits,
    ).to_csv(comparison_path, index=False)
    written["comparison"] = comparison_path

    diagnostic_path = output_path / "diagnostic_table.csv"
    make_diagnostic_table(summary, round_digits=round_digits).to_csv(
        diagnostic_path,
        index=False,
    )
    written["diagnostic"] = diagnostic_path

    for metric, filename in WIDE_TABLE_METRICS.items():
        if metric not in summary.columns:
            continue
        table = make_wide_metric_table(summary, metric, round_digits=round_digits)
        path = output_path / filename
        table.to_csv(path, index=False)
        key = filename.removesuffix("_wide.csv").removesuffix(".csv")
        written[key] = path
    return written
