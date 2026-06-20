"""Table builders for Monte Carlo summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TABLE_GROUP_COLUMNS = ["dgp", "n", "p", "pi", "tau"]
ESTIMATOR_COLUMN = "estimator"
ESTIMATOR_ORDER = [
    "oracle",
    "oracle_ivqr",
    "full_ivqr",
    "post_selection_ivqr",
    "dml_ivqr",
]
ESTIMATOR_LABELS = {
    "oracle_ivqr": "Oracle IVQR",
    "oracle": "Oracle IVQR",
    "full_ivqr": "Full-control IVQR",
    "post_selection_ivqr": "Post-selection IVQR",
    "dml_ivqr": "DML-IVQR",
}
CORE_METRICS = [
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
]
DIAGNOSTIC_COLUMNS = [
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
]
WIDE_TABLE_METRICS = {
    "bias": "bias_wide.csv",
    "rmse": "rmse_wide.csv",
    "mae": "mae_wide.csv",
    "coverage": "coverage_wide.csv",
    "avg_cr_length": "cr_length_wide.csv",
    "mean_runtime_seconds": "runtime_wide.csv",
    "failure_rate": "failure_rate_wide.csv",
}


def _validate_summary_columns(summary: pd.DataFrame) -> None:
    required = set(TABLE_GROUP_COLUMNS + [ESTIMATOR_COLUMN])
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary is missing required columns: {missing}")


def _known_metric_columns(summary: pd.DataFrame) -> list[str]:
    known = set(CORE_METRICS + DIAGNOSTIC_COLUMNS + ["avg_cr_length"])
    return [column for column in summary.columns if column in known]


def _validate_metrics(summary: pd.DataFrame, metrics: list[str]) -> None:
    missing = [metric for metric in metrics if metric not in summary.columns]
    if missing:
        raise ValueError(f"summary is missing requested metric columns: {missing}")


def _round_numeric(
    df: pd.DataFrame, columns: list[str], round_digits: int | None
) -> pd.DataFrame:
    if round_digits is None:
        return df
    rounded = df.copy()
    for column in columns:
        if column in rounded.columns:
            values = pd.to_numeric(rounded[column], errors="coerce")
            rounded[column] = values.round(round_digits)
    return rounded


def load_summary(path: str | Path) -> pd.DataFrame:
    """Load an aggregated summary CSV and validate table-generation columns."""
    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = pd.read_csv(summary_path)
    _validate_summary_columns(summary)
    if not _known_metric_columns(summary):
        raise ValueError("summary must contain at least one metric column")
    return summary


def add_estimator_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add display labels and sort by scenario plus canonical estimator order."""
    labeled = df.copy()
    _validate_summary_columns(labeled)
    labeled["estimator_label"] = (
        labeled[ESTIMATOR_COLUMN]
        .map(ESTIMATOR_LABELS)
        .fillna(labeled[ESTIMATOR_COLUMN])
    )

    estimators = labeled[ESTIMATOR_COLUMN].astype(str)
    unknown = sorted(
        estimator
        for estimator in estimators.unique()
        if estimator not in ESTIMATOR_ORDER
    )
    categories = ESTIMATOR_ORDER + unknown
    labeled[ESTIMATOR_COLUMN] = pd.Categorical(
        estimators,
        categories=categories,
        ordered=True,
    )
    return labeled.sort_values(TABLE_GROUP_COLUMNS + [ESTIMATOR_COLUMN]).reset_index(
        drop=True
    )


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
    filtered = summary.copy()
    filters = {
        "dgp": dgp,
        "n": n,
        "p": p,
        "pi": pi,
        "tau": tau,
    }
    for column, value in filters.items():
        if value is not None:
            filtered = filtered.loc[filtered[column] == value]
    if estimators is not None:
        filtered = filtered.loc[filtered[ESTIMATOR_COLUMN].isin(estimators)]
    return filtered.copy()


def make_wide_metric_table(
    summary: pd.DataFrame,
    metric: str,
    index_columns: list[str] | None = None,
    round_digits: int | None = 4,
) -> pd.DataFrame:
    """Create a wide scenario-by-estimator table for one metric."""
    _validate_summary_columns(summary)
    if metric not in summary.columns:
        raise ValueError(f"metric column not found in summary: {metric}")

    index_columns = TABLE_GROUP_COLUMNS if index_columns is None else index_columns
    duplicate_columns = index_columns + [ESTIMATOR_COLUMN]
    duplicates = summary.duplicated(duplicate_columns, keep=False)
    if duplicates.any():
        raise ValueError("duplicate scenario-estimator rows cannot be pivoted")

    labeled = add_estimator_labels(summary)
    wide = labeled.pivot(index=index_columns, columns="estimator_label", values=metric)

    ordered_labels = [ESTIMATOR_LABELS[estimator] for estimator in ESTIMATOR_ORDER]
    existing_ordered = [label for label in ordered_labels if label in wide.columns]
    remaining = [label for label in wide.columns if label not in existing_ordered]
    wide = wide[existing_ordered + remaining]
    wide = wide.reset_index()

    metric_columns = [column for column in wide.columns if column not in index_columns]
    return _round_numeric(wide, metric_columns, round_digits)


def make_comparison_table(
    summary: pd.DataFrame,
    metrics: list[str] | None = None,
    round_digits: int | None = 4,
) -> pd.DataFrame:
    """Create a long-format master result table with core metrics."""
    _validate_summary_columns(summary)
    metrics = CORE_METRICS if metrics is None else metrics
    _validate_metrics(summary, metrics)

    table = add_estimator_labels(summary)
    columns = TABLE_GROUP_COLUMNS + [ESTIMATOR_COLUMN, "estimator_label"] + metrics
    table = table[columns].copy()
    return _round_numeric(table, metrics, round_digits)


def make_diagnostic_table(
    summary: pd.DataFrame,
    round_digits: int | None = 4,
) -> pd.DataFrame:
    """Create a long-format diagnostic table for completion and failure checks."""
    _validate_summary_columns(summary)
    table = add_estimator_labels(summary)
    available_diagnostics = [
        column for column in DIAGNOSTIC_COLUMNS if column in table.columns
    ]
    columns = (
        TABLE_GROUP_COLUMNS
        + [ESTIMATOR_COLUMN, "estimator_label"]
        + available_diagnostics
    )
    table = table[columns].copy()
    return _round_numeric(table, available_diagnostics, round_digits)


def write_tables(
    summary: pd.DataFrame,
    output_dir: str | Path,
    round_digits: int = 4,
) -> dict[str, Path]:
    """Write standard thesis-ready CSV tables and return their paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    comparison_path = output_path / "comparison_table.csv"
    comparison_metrics = [
        metric for metric in CORE_METRICS if metric in summary.columns
    ]
    make_comparison_table(
        summary,
        metrics=comparison_metrics,
        round_digits=round_digits,
    ).to_csv(comparison_path, index=False)
    written["comparison"] = comparison_path

    diagnostic_path = output_path / "diagnostic_table.csv"
    make_diagnostic_table(summary, round_digits=round_digits).to_csv(
        diagnostic_path, index=False
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
