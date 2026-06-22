"""Figure generation for Monte Carlo diagnostics and results."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reporting.tables import ESTIMATOR_LABELS

DEFAULT_FIGURE_METRICS = {
    "bias": ("fig_bias.png", "Bias"),
    "rmse": ("fig_rmse.png", "RMSE"),
    "coverage": ("fig_coverage.png", "Coverage"),
    "avg_cr_length": ("fig_cr_length.png", "Average confidence-region length"),
    "failure_rate": ("fig_failure_rate.png", "Failure rate"),
}
FigureMetricSpec: TypeAlias = str | tuple[str, str]


def _validate_summary(summary: pd.DataFrame, metrics: list[str]) -> None:
    required = {"dgp", "n", "p", "pi", "tau", "estimator", *metrics}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary is missing required figure columns: {missing}")


def _scenario_label(row: pd.Series) -> str:
    return (
        f"{row['dgp']} | n={int(row['n'])} | p={int(row['p'])} | "
        f"pi={row['pi']} | tau={row['tau']}"
    )


def make_metric_figure(
    summary: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Write one grouped bar chart for a scenario-level metric."""
    _validate_summary(summary, [metric])
    data = summary.copy()
    data["scenario"] = data.apply(_scenario_label, axis=1)
    data["estimator_label"] = data["estimator"].map(ESTIMATOR_LABELS).fillna(data["estimator"])
    data[metric] = pd.to_numeric(data[metric], errors="coerce")

    pivot = data.pivot_table(
        index="scenario",
        columns="estimator_label",
        values=metric,
        aggfunc="mean",
        observed=False,
    )
    if pivot.empty:
        raise ValueError("cannot plot an empty summary")

    fig_width = max(9.0, min(24.0, 0.45 * len(pivot.index) + 6.0))
    ax = pivot.plot(kind="bar", figsize=(fig_width, 5.5))
    fig = ax.get_figure()
    ax.set_title(title or metric)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(metric)
    ax.legend(title="Estimator", loc="best")
    ax.tick_params(axis="x", labelrotation=70)
    plt.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_figures(
    summary: pd.DataFrame,
    output_dir: str | Path,
    metrics: dict[str, FigureMetricSpec] | None = None,
) -> dict[str, Path]:
    """Generate standard Monte Carlo figures and return written paths."""
    metrics = DEFAULT_FIGURE_METRICS if metrics is None else metrics
    output = Path(output_dir)
    written: dict[str, Path] = {}
    for metric, spec in metrics.items():
        if metric not in summary.columns:
            continue
        if isinstance(spec, tuple):
            filename, title = spec
        else:
            filename, title = f"fig_{metric}.png", spec
        path = output / filename
        written[metric] = make_metric_figure(summary, metric, path, title=title)
    return written
