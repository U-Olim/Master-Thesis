"""Clean, single-panel figures for the final Monte Carlo results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from analysis.tables import ESTIMATOR_NAMES, ESTIMATOR_ORDER, summarize_performance


COLORS = {
    "oracle": "#1f77b4",
    "post_selection": "#d95f02",
    "dml": "#2a9d8f",
}
MARKERS = {"oracle": "o", "post_selection": "s", "dml": "^"}


def _save_figure(figure: plt.Figure, output_dir: str | Path, name: str) -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination / f"{name}.pdf", destination / f"{name}.png"]
    figure.tight_layout()
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], dpi=300, bbox_inches="tight")
    plt.close(figure)
    return paths


def _estimator_line_figure(
    summary: pd.DataFrame,
    *,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    output_dir: str | Path,
    name: str,
    reference: float | None = None,
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for estimator in ESTIMATOR_ORDER:
        values = summary.loc[summary["estimator"].eq(estimator)].sort_values(x)
        axis.plot(
            values[x],
            values[y],
            color=COLORS[estimator],
            marker=MARKERS[estimator],
            linewidth=1.8,
            markersize=5,
            label=ESTIMATOR_NAMES[estimator],
        )
    if reference is not None:
        axis.axhline(reference, color="0.35", linestyle="--", linewidth=1, label="Nominal 95%")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="0.88", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    return _save_figure(figure, output_dir, name)


def plot_coverage_vs_strength(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    summary = summarize_performance(results, ["pi", "estimator"])
    return _estimator_line_figure(
        summary,
        x="pi",
        y="coverage",
        xlabel="Instrument strength ($\\pi$)",
        ylabel="Coverage probability",
        output_dir=output_dir,
        name="coverage_vs_strength",
        reference=0.95,
    )


def plot_rmse_vs_strength(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    summary = summarize_performance(results, ["pi", "estimator"])
    return _estimator_line_figure(
        summary,
        x="pi",
        y="rmse",
        xlabel="Instrument strength ($\\pi$)",
        ylabel="RMSE",
        output_dir=output_dir,
        name="rmse_vs_strength",
    )


def plot_cr_length_vs_strength(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    summary = summarize_performance(results, ["pi", "estimator"])
    return _estimator_line_figure(
        summary,
        x="pi",
        y="average_cr_length",
        xlabel="Instrument strength ($\\pi$)",
        ylabel="Average confidence-region length",
        output_dir=output_dir,
        name="cr_length_vs_strength",
    )


def plot_coverage_by_quantile(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    summary = summarize_performance(results, ["tau", "estimator"])
    return _estimator_line_figure(
        summary,
        x="tau",
        y="coverage",
        xlabel="Quantile ($\\tau$)",
        ylabel="Coverage probability",
        output_dir=output_dir,
        name="coverage_by_quantile",
        reference=0.95,
    )


def plot_rmse_by_quantile(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    summary = summarize_performance(results, ["tau", "estimator"])
    return _estimator_line_figure(
        summary,
        x="tau",
        y="rmse",
        xlabel="Quantile ($\\tau$)",
        ylabel="RMSE",
        output_dir=output_dir,
        name="rmse_by_quantile",
    )


def plot_rmse_by_dgp(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    summary = summarize_performance(results, ["dgp", "estimator"])
    return _estimator_line_figure(
        summary,
        x="dgp",
        y="rmse",
        xlabel="Data-generating process",
        ylabel="RMSE",
        output_dir=output_dir,
        name="rmse_by_dgp",
    )


def plot_selected_controls(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    selected = results.loc[results["estimator"].eq("post_selection")]
    if selected.empty or "n_selected_controls" not in selected:
        raise ValueError("Post-selection diagnostics are unavailable")
    summary = selected.groupby("pi", sort=True)["n_selected_controls"].mean()
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(summary.index, summary.values, color=COLORS["post_selection"], marker="s", linewidth=1.8)
    axis.set_xlabel("Instrument strength ($\\pi$)")
    axis.set_ylabel("Mean selected controls")
    axis.grid(axis="y", color="0.88", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    return _save_figure(figure, output_dir, "selected_controls_vs_strength")


def write_all_figures(results: pd.DataFrame, output_dir: str | Path) -> dict[str, list[Path]]:
    """Write the final figures in PDF and high-resolution PNG formats."""
    functions = {
        "coverage_vs_strength": plot_coverage_vs_strength,
        "rmse_vs_strength": plot_rmse_vs_strength,
        "cr_length_vs_strength": plot_cr_length_vs_strength,
        "coverage_by_quantile": plot_coverage_by_quantile,
        "rmse_by_quantile": plot_rmse_by_quantile,
        "rmse_by_dgp": plot_rmse_by_dgp,
        "selected_controls_vs_strength": plot_selected_controls,
    }
    return {name: function(results, output_dir) for name, function in functions.items()}
