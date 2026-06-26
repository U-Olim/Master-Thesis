"""Figure generation for Monte Carlo diagnostics and results."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from reporting.tables import ESTIMATOR_LABELS


FigureMetricSpec: TypeAlias = str | tuple[str, str]

DEFAULT_FIGURE_METRICS: Mapping[str, FigureMetricSpec] = MappingProxyType(
    {
        "bias": ("fig_bias.png", "Bias"),
        "rmse": ("fig_rmse.png", "RMSE"),
        "coverage": ("fig_coverage.png", "Coverage"),
        "avg_cr_length": ("fig_cr_length.png", "Average confidence-region length"),
        "failure_rate": ("fig_failure_rate.png", "Failure rate"),
    }
)

__all__ = [
    "DEFAULT_FIGURE_METRICS",
    "FigureMetricSpec",
    "make_metric_figure",
    "write_figures",
]


def _validate_summary(summary: pd.DataFrame, metrics: list[str]) -> None:
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame")
    if summary.columns.duplicated().any():
        duplicated = sorted(set(summary.columns[summary.columns.duplicated()].astype(str)))
        raise ValueError(f"summary has duplicate columns: {duplicated}")
    if summary.empty:
        raise ValueError("cannot plot an empty summary")
    if not metrics:
        raise ValueError("at least one figure metric is required")
    if any(not isinstance(metric, str) or not metric for metric in metrics):
        raise ValueError("figure metrics must be nonempty strings")

    required = {"dgp", "n", "p", "pi", "tau", "estimator", *metrics}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary is missing required figure columns: {missing}")

    for metric in metrics:
        numeric = pd.to_numeric(summary[metric], errors="coerce")
        if numeric.notna().sum() == 0:
            raise ValueError(f"metric column '{metric}' has no numeric values")


def _format_int_label(value: object, name: str) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"{name} must be numeric for scenario labels")
    if float(numeric) != int(numeric):
        raise ValueError(f"{name} must be integer-valued for scenario labels")
    return str(int(numeric))


def _format_numeric_label(value: object, name: str) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError(f"{name} must be numeric for scenario labels")
    return f"{float(numeric):g}"


def _scenario_label(row: pd.Series) -> str:
    return (
        f"{row['dgp']} | "
        f"n={_format_int_label(row['n'], 'n')} | "
        f"p={_format_int_label(row['p'], 'p')} | "
        f"pi={_format_numeric_label(row['pi'], 'pi')} | "
        f"tau={_format_numeric_label(row['tau'], 'tau')}"
    )


def _ordered_estimator_columns(columns: pd.Index) -> list[str]:
    preferred = [
        ESTIMATOR_LABELS.get("oracle", "Oracle IVQR"),
        ESTIMATOR_LABELS.get("post_selection_ivqr", "Post-selection IVQR"),
        ESTIMATOR_LABELS.get("dml_ivqr", "DML-style IVQR"),
        ESTIMATOR_LABELS.get("full_control_ivqr", "Full-control IVQR"),
    ]
    existing = [label for label in preferred if label in columns]
    remaining = sorted(str(label) for label in columns if label not in set(existing))
    return existing + remaining


def make_metric_figure(
    summary: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Write one grouped bar chart for a scenario-level metric."""
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a nonempty string")
    path = Path(output_path)
    if path.name == "":
        raise ValueError("output_path must include a file name")

    _validate_summary(summary, [metric])
    data = summary.copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.loc[data[metric].notna()].copy()
    if data.empty:
        raise ValueError(f"metric column '{metric}' has no numeric values")
    data["scenario"] = data.apply(_scenario_label, axis=1)
    data["estimator_label"] = (
        data["estimator"].map(ESTIMATOR_LABELS).fillna(data["estimator"])
    )

    pivot = data.pivot_table(
        index="scenario",
        columns="estimator_label",
        values=metric,
        aggfunc="mean",
        observed=False,
    )
    if pivot.empty:
        raise ValueError("cannot plot an empty summary")
    pivot = pivot.loc[:, _ordered_estimator_columns(pivot.columns)]

    fig_width = max(9.0, min(24.0, 0.45 * len(pivot.index) + 6.0))
    ax = cast(Axes, pivot.plot(kind="bar", figsize=(fig_width, 5.5)))
    fig = cast(Figure, ax.get_figure())
    ax.set_title(title or metric)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(title or metric)
    ax.legend(title="Estimator", loc="best")
    ax.tick_params(axis="x", labelrotation=70)
    plt.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=200)
    finally:
        plt.close(fig)
    return path


def _validate_figure_metrics(
    metrics: Mapping[str, FigureMetricSpec],
) -> dict[str, tuple[str, str]]:
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    if not metrics:
        raise ValueError("metrics must not be empty")

    normalized: dict[str, tuple[str, str]] = {}
    for metric, spec in metrics.items():
        if not isinstance(metric, str) or not metric:
            raise ValueError("figure metric names must be nonempty strings")
        if isinstance(spec, str):
            if not spec:
                raise ValueError("figure metric title must be nonempty")
            normalized[metric] = (f"fig_{metric}.png", spec)
            continue
        if (
            isinstance(spec, tuple)
            and len(spec) == 2
            and all(isinstance(item, str) and item for item in spec)
        ):
            filename, title = spec
            normalized[metric] = (filename, title)
            continue
        raise ValueError(
            "figure metric specs must be a title string or a (filename, title) tuple"
        )
    return normalized


def write_figures(
    summary: pd.DataFrame,
    output_dir: str | Path,
    metrics: dict[str, FigureMetricSpec] | None = None,
) -> dict[str, Path]:
    """Generate standard Monte Carlo figures and return written paths."""
    selected_metrics = DEFAULT_FIGURE_METRICS if metrics is None else metrics
    normalized = _validate_figure_metrics(selected_metrics)

    output = Path(output_dir)
    written: dict[str, Path] = {}
    for metric, (filename, title) in normalized.items():
        if metric not in summary.columns:
            continue
        path = output / filename
        written[metric] = make_metric_figure(summary, metric, path, title=title)
    return written
