"""Compact automatic figures for Monte Carlo reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from reporting.tables import (
    ESTIMATOR_COLUMN,
    ESTIMATOR_LABELS,
    build_coverage_by_pi,
    build_main_summary,
    build_runtime_summary,
    infer_reporting_columns,
)


__all__ = [
    "make_coverage_by_pi_figure",
    "make_coverage_overall_figure",
    "make_runtime_by_estimator_figure",
    "make_weak_iv_diagnostic_figure",
    "write_figures",
]


def _validate_report_frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame")
    if df.columns.duplicated().any():
        duplicated = sorted(set(df.columns[df.columns.duplicated()].astype(str)))
        raise ValueError(f"summary has duplicate columns: {duplicated}")
    if df.empty:
        raise ValueError("cannot plot an empty summary")
    if ESTIMATOR_COLUMN not in df.columns:
        raise ValueError("summary must contain an estimator column")
    return df


def _label(estimator: object) -> str:
    text = str(estimator)
    return ESTIMATOR_LABELS.get(text, text)


def _finish(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)
    return path


def _has_numeric(table: pd.DataFrame, column: str) -> bool:
    return bool(pd.to_numeric(table[column], errors="coerce").notna().any())


def _should_use_log_scale(values: pd.Series) -> bool:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    clean = clean.loc[clean > 0]
    if clean.empty:
        return False
    min_value = float(clean.min())
    max_value = float(clean.max())
    if min_value <= 0:
        return False
    return bool(max_value / min_value > 10)


def make_coverage_by_pi_figure(summary: pd.DataFrame, output_path: str | Path) -> Path | None:
    """Write coverage-by-pi line plot, or return None when unavailable."""
    _validate_report_frame(summary)
    if "pi" not in summary.columns:
        return None
    pi_values = pd.to_numeric(summary["pi"], errors="coerce").dropna().unique()
    if len(pi_values) < 2:
        return None
    table = build_coverage_by_pi(summary, round_digits=None)
    if table is None or table.empty or not _has_numeric(table, "coverage"):
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    for estimator, group in table.groupby("estimator", sort=False):
        plot_data = group.sort_values("pi")
        ax.plot(
            plot_data["pi"],
            plot_data["coverage"],
            marker="o",
            linewidth=1.8,
            label=_label(estimator),
        )
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("pi")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Estimator", fontsize=8)
    return _finish(fig, Path(output_path))


def make_runtime_by_estimator_figure(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path | None:
    """Write median-runtime bar chart, or return None when unavailable."""
    _validate_report_frame(summary)
    if infer_reporting_columns(summary)["runtime"] is None:
        return None
    table = build_runtime_summary(summary, round_digits=None)
    if table is None or table.empty or not _has_numeric(table, "median_runtime_sec"):
        return None
    table = table.loc[pd.to_numeric(table["median_runtime_sec"], errors="coerce").notna()]
    if table.empty:
        return None

    labels = [_label(estimator) for estimator in table["estimator"]]
    values = pd.to_numeric(table["median_runtime_sec"], errors="coerce")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    if _should_use_log_scale(values):
        ax.set_yscale("log")
    ax.set_xlabel("Estimator")
    ax.set_ylabel("Median runtime (sec)")
    if max(len(label) for label in labels) > 14:
        ax.tick_params(axis="x", labelrotation=25)
    return _finish(fig, Path(output_path))


def make_coverage_overall_figure(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path | None:
    """Write overall coverage bar chart for multi-estimator runs."""
    _validate_report_frame(summary)
    if summary[ESTIMATOR_COLUMN].nunique(dropna=True) < 2:
        return None
    table = build_main_summary(summary, round_digits=None)
    if table.empty or not _has_numeric(table, "coverage"):
        return None
    table = table.loc[pd.to_numeric(table["coverage"], errors="coerce").notna()]
    if len(table) < 2:
        return None

    labels = [_label(estimator) for estimator in table["estimator"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, pd.to_numeric(table["coverage"], errors="coerce"))
    ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Estimator")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)
    if max(len(label) for label in labels) > 14:
        ax.tick_params(axis="x", labelrotation=25)
    return _finish(fig, Path(output_path))


def make_weak_iv_diagnostic_figure(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path | None:
    """Write boundary-share-by-pi line plot, or return None when unavailable."""
    _validate_report_frame(summary)
    if "pi" not in summary.columns:
        return None
    if infer_reporting_columns(summary)["boundary"] is None:
        return None
    pi_values = pd.to_numeric(summary["pi"], errors="coerce").dropna().unique()
    if len(pi_values) < 2:
        return None
    table = build_coverage_by_pi(summary, round_digits=None)
    if table is None or table.empty or not _has_numeric(table, "boundary_share"):
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    for estimator, group in table.groupby("estimator", sort=False):
        plot_data = group.sort_values("pi")
        ax.plot(
            plot_data["pi"],
            plot_data["boundary_share"],
            marker="o",
            linewidth=1.8,
            label=_label(estimator),
        )
    ax.set_xlabel("pi")
    ax.set_ylabel("Boundary share")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Estimator", fontsize=8)
    return _finish(fig, Path(output_path))


def write_figures(summary: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Generate compact automatic figures and return written paths."""
    _validate_report_frame(summary)
    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise ValueError("output_dir must be a directory path")
    output.mkdir(parents=True, exist_ok=True)

    figure_specs = (
        ("coverage_by_pi", "coverage_by_pi.png", make_coverage_by_pi_figure),
        ("runtime_by_estimator", "runtime_by_estimator.png", make_runtime_by_estimator_figure),
        ("coverage_overall", "coverage_overall.png", make_coverage_overall_figure),
        ("weak_iv_diagnostic", "weak_iv_diagnostic.png", make_weak_iv_diagnostic_figure),
    )
    written: dict[str, Path] = {}
    for key, filename, builder in figure_specs:
        path = builder(summary, output / filename)
        if path is not None:
            written[key] = path
    return written
