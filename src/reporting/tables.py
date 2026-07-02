"""Compact automatic tables for Monte Carlo reports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd


ESTIMATOR_COLUMN: str = "estimator"
ESTIMATOR_ORDER: tuple[str, ...] = (
    "oracle",
    "post_selection_ivqr",
    "full_control_ivqr",
    "dml_ivqr",
)
ESTIMATOR_LABELS = MappingProxyType(
    {
        "oracle": "Oracle IVQR",
        "post_selection_ivqr": "Post-selection IVQR",
        "full_control_ivqr": "Full-control IVQR",
        "dml_ivqr": "DML-style IVQR",
    }
)

MAIN_SUMMARY_COLUMNS: tuple[str, ...] = (
    "estimator",
    "rows",
    "coverage",
    "mean_ci_length",
    "mean_runtime_sec",
    "median_runtime_sec",
    "p95_runtime_sec",
    "failed_share",
    "empty_ci_share",
    "boundary_share",
)
COVERAGE_BY_PI_COLUMNS: tuple[str, ...] = (
    "estimator",
    "pi",
    "coverage",
    "mean_ci_length",
    "boundary_share",
    "mean_runtime_sec",
    "rows",
)
RUNTIME_SUMMARY_COLUMNS: tuple[str, ...] = (
    "estimator",
    "rows",
    "mean_runtime_sec",
    "median_runtime_sec",
    "p90_runtime_sec",
    "p95_runtime_sec",
    "max_runtime_sec",
    "total_runtime_sec",
)
COVERAGE_BY_TAU_COLUMNS: tuple[str, ...] = (
    "estimator",
    "tau",
    "coverage",
    "mean_ci_length",
    "rows",
)

__all__ = [
    "COVERAGE_BY_PI_COLUMNS",
    "COVERAGE_BY_TAU_COLUMNS",
    "ESTIMATOR_COLUMN",
    "ESTIMATOR_LABELS",
    "ESTIMATOR_ORDER",
    "MAIN_SUMMARY_COLUMNS",
    "RUNTIME_SUMMARY_COLUMNS",
    "add_estimator_labels",
    "build_coverage_by_pi",
    "build_coverage_by_tau",
    "build_main_summary",
    "build_runtime_summary",
    "infer_reporting_columns",
    "load_summary",
    "write_tables",
]


_RUNTIME_CANDIDATES: tuple[str, ...] = (
    "runtime_sec",
    "elapsed_sec",
    "total_runtime_sec",
    "runtime_seconds",
    "mean_runtime_seconds",
    "mean_runtime_total_sec",
    "median_runtime_total_sec",
)
_CI_LENGTH_CANDIDATES: tuple[str, ...] = (
    "ci_length",
    "cr_length",
    "avg_cr_length",
    "mean_ci_length",
)
_COVERAGE_CANDIDATES: tuple[str, ...] = (
    "covers",
    "cr_covers_true",
    "coverage",
)
_EMPTY_CANDIDATES: tuple[str, ...] = (
    "empty_ci",
    "empty_cr",
    "cr_empty",
    "cr_empty_rate",
)
_BOUNDARY_CANDIDATES: tuple[str, ...] = (
    "boundary_hit",
    "cr_hits_any_boundary",
    "cr_boundary_hit",
    "cr_boundary_hit_rate",
    "alpha_hat_at_any_boundary",
    "boundary_rate",
)
_FAILED_RATE_CANDIDATES: tuple[str, ...] = (
    "failure_rate",
    "failed_alpha_rate",
    "mean_failed_alpha_rate",
)


def _assert_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("report data must be a pandas DataFrame")
    if df.columns.duplicated().any():
        duplicated = sorted(set(df.columns[df.columns.duplicated()].astype(str)))
        raise ValueError(f"report data has duplicate columns: {duplicated}")
    if df.empty:
        raise ValueError("report data must not be empty")
    if ESTIMATOR_COLUMN not in df.columns:
        raise ValueError("report data must contain an estimator column")
    return df


def _first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def infer_reporting_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Return the source columns used for compact automatic reporting."""
    _assert_frame(df)
    return {
        "coverage": _first_existing(df, _COVERAGE_CANDIDATES),
        "runtime": _first_existing(df, _RUNTIME_CANDIDATES),
        "ci_length": _first_existing(df, _CI_LENGTH_CANDIDATES),
        "empty_ci": _first_existing(df, _EMPTY_CANDIDATES),
        "boundary": _first_existing(df, _BOUNDARY_CANDIDATES),
        "failed_rate": _first_existing(df, _FAILED_RATE_CANDIDATES),
    }


def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def _rate_from_columns(df: pd.DataFrame, primary: str | None) -> pd.Series | None:
    if primary is not None:
        return _to_numeric(df[primary])
    return None


def _failed_series(df: pd.DataFrame, columns: dict[str, str | None]) -> pd.Series | None:
    if "status" in df.columns:
        return df["status"].astype(str).str.lower().ne("ok").astype(float)
    if "failed" in df.columns:
        return _to_numeric(df["failed"])
    failed_rate = columns["failed_rate"]
    if failed_rate is not None:
        return _to_numeric(df[failed_rate])
    if "failed_alpha_count" in df.columns:
        return (_to_numeric(df["failed_alpha_count"]) > 0).astype(float)
    return None


def _boundary_series(df: pd.DataFrame, columns: dict[str, str | None]) -> pd.Series | None:
    boundary = columns["boundary"]
    if boundary is not None:
        return _to_numeric(df[boundary])
    lower = "lower_boundary_hit" if "lower_boundary_hit" in df.columns else None
    upper = "upper_boundary_hit" if "upper_boundary_hit" in df.columns else None
    if lower is None and upper is None:
        return None
    pieces = [
        _to_numeric(df[column]).fillna(0).astype(bool)
        for column in (lower, upper)
        if column is not None
    ]
    return pd.concat(pieces, axis=1).any(axis=1).astype(float)


def _normalized_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = _assert_frame(df).copy()
    columns = infer_reporting_columns(df)
    out = pd.DataFrame(index=df.index)
    out["estimator"] = df[ESTIMATOR_COLUMN].astype(str)
    if "pi" in df.columns:
        out["pi"] = pd.to_numeric(df["pi"], errors="coerce")
    if "tau" in df.columns:
        out["tau"] = pd.to_numeric(df["tau"], errors="coerce")

    weight_column = _first_existing(df, ("rows", "replications", "observed_replications"))
    out["_rows_weight"] = (
        _to_numeric(df[weight_column]).fillna(0)
        if weight_column is not None
        else pd.Series(1.0, index=df.index)
    )
    out.loc[out["_rows_weight"] <= 0, "_rows_weight"] = 1.0

    for name, source in (
        ("coverage", columns["coverage"]),
        ("runtime_sec", columns["runtime"]),
        ("ci_length", columns["ci_length"]),
        ("empty_ci", columns["empty_ci"]),
    ):
        out[name] = (
            _to_numeric(df[source])
            if source is not None
            else pd.Series(np.nan, index=df.index)
        )
    failed = _failed_series(df, columns)
    boundary = _boundary_series(df, columns)
    out["failed"] = failed if failed is not None else pd.Series(np.nan, index=df.index)
    out["boundary"] = (
        boundary if boundary is not None else pd.Series(np.nan, index=df.index)
    )
    return out


def _weighted_mean(group: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(group[column], errors="coerce")
    valid = values.notna()
    if not valid.any():
        return float("nan")
    weights = pd.to_numeric(group.loc[valid, "_rows_weight"], errors="coerce").fillna(1)
    if float(weights.sum()) <= 0:
        return float(values.loc[valid].mean())
    return float(np.average(values.loc[valid], weights=weights))


def _quantile(group: pd.DataFrame, column: str, q: float) -> float:
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return float(values.quantile(q)) if not values.empty else float("nan")


def _sum_rows(group: pd.DataFrame) -> int:
    return int(pd.to_numeric(group["_rows_weight"], errors="coerce").fillna(1).sum())


def _grouped_report(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    data = _normalized_metrics(df)
    rows: list[dict[str, object]] = []
    for key, group in data.groupby(group_columns, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row: dict[str, object] = dict(zip(group_columns, key_tuple, strict=True))
        row["rows"] = _sum_rows(group)
        row["coverage"] = _weighted_mean(group, "coverage")
        row["mean_ci_length"] = _weighted_mean(group, "ci_length")
        row["mean_runtime_sec"] = _weighted_mean(group, "runtime_sec")
        row["median_runtime_sec"] = _quantile(group, "runtime_sec", 0.5)
        row["p90_runtime_sec"] = _quantile(group, "runtime_sec", 0.9)
        row["p95_runtime_sec"] = _quantile(group, "runtime_sec", 0.95)
        row["max_runtime_sec"] = _quantile(group, "runtime_sec", 1.0)
        row["total_runtime_sec"] = float(
            (
                pd.to_numeric(group["runtime_sec"], errors="coerce")
                * pd.to_numeric(group["_rows_weight"], errors="coerce").fillna(1)
            ).sum(min_count=1)
        )
        row["failed_share"] = _weighted_mean(group, "failed")
        row["empty_ci_share"] = _weighted_mean(group, "empty_ci")
        row["boundary_share"] = _weighted_mean(group, "boundary")
        rows.append(row)
    return pd.DataFrame(rows)


def _sort_estimators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or ESTIMATOR_COLUMN not in df.columns:
        return df
    unknown = sorted(
        estimator for estimator in df[ESTIMATOR_COLUMN].astype(str).unique()
        if estimator not in ESTIMATOR_ORDER
    )
    ordered = list(ESTIMATOR_ORDER) + unknown
    sorted_df = df.copy()
    sorted_df[ESTIMATOR_COLUMN] = pd.Categorical(
        sorted_df[ESTIMATOR_COLUMN].astype(str),
        categories=ordered,
        ordered=True,
    )
    sort_columns = [ESTIMATOR_COLUMN]
    for optional in ("pi", "tau"):
        if optional in sorted_df.columns:
            sort_columns.append(optional)
    return sorted_df.sort_values(sort_columns).assign(
        estimator=lambda frame: frame[ESTIMATOR_COLUMN].astype(str)
    ).reset_index(drop=True)


def _round(df: pd.DataFrame, round_digits: int | None) -> pd.DataFrame:
    if round_digits is None:
        return df
    rounded = df.copy()
    for column in rounded.columns:
        if column == ESTIMATOR_COLUMN:
            continue
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].round(round_digits)
    return rounded


def build_main_summary(
    df: pd.DataFrame,
    round_digits: int | None = 4,
) -> pd.DataFrame:
    """Build one-row-per-estimator automatic summary table."""
    table = _grouped_report(df, [ESTIMATOR_COLUMN])
    return _round(_sort_estimators(table[list(MAIN_SUMMARY_COLUMNS)]), round_digits)


def build_coverage_by_pi(
    df: pd.DataFrame,
    round_digits: int | None = 4,
) -> pd.DataFrame | None:
    """Build estimator-by-pi coverage table, or None when pi is unavailable."""
    if "pi" not in df.columns:
        return None
    table = _grouped_report(df, [ESTIMATOR_COLUMN, "pi"])
    return _round(_sort_estimators(table[list(COVERAGE_BY_PI_COLUMNS)]), round_digits)


def build_runtime_summary(
    df: pd.DataFrame,
    round_digits: int | None = 4,
) -> pd.DataFrame | None:
    """Build one-row-per-estimator runtime table, or None when runtime is unavailable."""
    if infer_reporting_columns(df)["runtime"] is None:
        return None
    table = _grouped_report(df, [ESTIMATOR_COLUMN])
    return _round(_sort_estimators(table[list(RUNTIME_SUMMARY_COLUMNS)]), round_digits)


def build_coverage_by_tau(
    df: pd.DataFrame,
    round_digits: int | None = 4,
) -> pd.DataFrame | None:
    """Build estimator-by-tau coverage table, or None when tau is unavailable."""
    if "tau" not in df.columns:
        return None
    table = _grouped_report(df, [ESTIMATOR_COLUMN, "tau"])
    return _round(_sort_estimators(table[list(COVERAGE_BY_TAU_COLUMNS)]), round_digits)


def load_summary(path: str | Path) -> pd.DataFrame:
    """Load a CSV suitable for compact report generation."""
    summary_path = Path(path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    return _assert_frame(pd.read_csv(summary_path))


def add_estimator_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add readable estimator labels without changing row granularity."""
    _assert_frame(df)
    labeled = df.copy()
    estimators = labeled[ESTIMATOR_COLUMN].astype(str)
    labeled["estimator_label"] = estimators.map(ESTIMATOR_LABELS).fillna(estimators)
    return labeled


def write_tables(
    summary: pd.DataFrame,
    output_dir: str | Path,
    round_digits: int | None = 4,
) -> dict[str, Path]:
    """Write compact automatic CSV tables and return their paths."""
    _assert_frame(summary)
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("output_dir must be a directory path")
    output_path.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    table_specs = (
        ("main_summary", "main_summary.csv", build_main_summary(summary, round_digits)),
        ("coverage_by_pi", "coverage_by_pi.csv", build_coverage_by_pi(summary, round_digits)),
        ("runtime_summary", "runtime_summary.csv", build_runtime_summary(summary, round_digits)),
        ("coverage_by_tau", "coverage_by_tau.csv", build_coverage_by_tau(summary, round_digits)),
    )
    for key, filename, table in table_specs:
        if table is None:
            continue
        path = output_path / filename
        table.to_csv(path, index=False)
        written[key] = path
    return written
