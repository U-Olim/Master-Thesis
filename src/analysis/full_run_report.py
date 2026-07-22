"""Publication assets for the validated R=500 full-run IVQR report.

This module treats the CSV summaries produced by ``scripts/report_full_run.py``
as authoritative.  It never recomputes coverage from replication-level data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from dgp.true_parameters import get_oracle_control_count, true_alpha
from simulation.config import (
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    DEFAULT_REFINEMENT_TOLERANCE,
    DF_T,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPOSITORY_ROOT / "results" / "report_full_run"
DEFAULT_ASSET_DIR = DEFAULT_REPORT_DIR / "report_assets"
ESTIMATOR_ORDER = ["DML-IVQR", "Oracle IVQR", "Post-selection IVQR"]
ESTIMATOR_DISPLAY_LABELS = {
    "DML-IVQR": "DML-style IVQR",
    "Oracle IVQR": "Oracle IVQR",
    "Post-selection IVQR": "Mean-Lasso Post-selection IVQR",
}
ESTIMATOR_LEGEND_LABELS = {
    **ESTIMATOR_DISPLAY_LABELS,
    "Post-selection IVQR": "Mean-Lasso PS-IVQR",
}
ESTIMATOR_COLORS = {
    "DML-IVQR": "#009E73",
    "Oracle IVQR": "#0072B2",
    "Post-selection IVQR": "#D55E00",
}
ESTIMATOR_MARKERS = {
    "DML-IVQR": "o",
    "Oracle IVQR": "s",
    "Post-selection IVQR": "^",
}
TABLE_FILES = {
    "overall": "table_01_overall.csv",
    "quantile": "table_02_by_quantile.csv",
    "strength": "table_03_by_strength.csv",
    "n_p": "table_04_by_n_p.csv",
    "cell": "table_05_by_design_cell.csv",
    "worst": "table_06_worst_cells.csv",
    "diagnostics": "table_07_diagnostics.csv",
}
COMMON_METRIC_COLUMNS = {
    "estimator_label",
    "replications",
    "resolved_replications",
    "unresolved_replications",
    "bias",
    "mae",
    "rmse",
    "estimate_sd",
    "empirical_coverage",
    "coverage_mcse",
    "mean_cr_length",
    "median_cr_length",
    "full_grid_rate",
    "empty_region_rate",
    "disconnected_region_rate",
    "unresolved_rate",
    "boundary_estimate_rate",
    "iteration_warning_rate",
    "rank_failure_rate",
    "refinement_limit_rate",
}
REQUIRED_COLUMNS = {
    "overall": COMMON_METRIC_COLUMNS,
    "quantile": COMMON_METRIC_COLUMNS | {"tau"},
    "strength": COMMON_METRIC_COLUMNS | {"pi"},
    "n_p": COMMON_METRIC_COLUMNS | {"n", "p"},
    "cell": COMMON_METRIC_COLUMNS | {"dgp", "n", "p", "pi", "tau"},
    "worst": COMMON_METRIC_COLUMNS | {"dgp", "n", "p", "pi", "tau"},
    "diagnostics": {
        "estimator_label",
        "cr_status_standardized",
        "replications",
        "resolved_replications",
        "covered_resolved",
        "uncovered_resolved",
        "unresolved_replications",
        "empirical_coverage",
        "mean_cr_length",
        "disconnected_region_rate",
        "iteration_warning_rate",
        "rank_failure_rate",
        "refinement_limit_rate",
    },
}


class ReportInputError(ValueError):
    """Raised when validated report inputs are missing or inconsistent."""


def validate_required_columns(
    frame: pd.DataFrame, required: set[str], source_name: str
) -> None:
    """Raise a readable error when an authoritative summary schema changed."""
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ReportInputError(f"{source_name}: missing required columns: {missing}")


def order_estimators(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable DML, Oracle, Post-selection ordering."""
    labels = set(frame["estimator_label"].dropna().astype(str).unique())
    unknown = sorted(labels.difference(ESTIMATOR_ORDER))
    if unknown:
        raise ReportInputError(f"Unknown estimator labels: {unknown}")
    ordered = frame.copy()
    ordered["_estimator_order"] = ordered["estimator_label"].map(
        {label: index for index, label in enumerate(ESTIMATOR_ORDER)}
    )
    secondary = [
        column
        for column in ("tau", "pi", "n", "p", "dgp")
        if column in ordered.columns
    ]
    ordered = ordered.sort_values(
        ["_estimator_order", *secondary], kind="stable", na_position="last"
    )
    return ordered.drop(columns="_estimator_order").reset_index(drop=True)


def display_estimator_label(label: str, *, short: bool = False) -> str:
    """Map validated source identifiers to scientifically precise display labels."""
    mapping = ESTIMATOR_LEGEND_LABELS if short else ESTIMATOR_DISPLAY_LABELS
    try:
        return mapping[label]
    except KeyError as exc:
        raise ReportInputError(f"Unknown estimator label: {label}") from exc


def format_percentage(value: object, digits: int = 2) -> str:
    """Format a probability as a percentage while preserving missingness."""
    if pd.isna(value):
        return "NA"
    return f"{100.0 * float(value):.{digits}f}%"


def format_number(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def format_integer(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{int(value):,}"


def load_validated_tables(report_dir: Path = DEFAULT_REPORT_DIR) -> dict[str, pd.DataFrame]:
    """Load and cross-check the authoritative full-run summary tables."""
    source_dir = report_dir.expanduser().resolve()
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in TABLE_FILES.items():
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Validated report input does not exist: {path}")
        frame = pd.read_csv(path, low_memory=False)
        validate_required_columns(frame, REQUIRED_COLUMNS[name], filename)
        tables[name] = order_estimators(frame)

    validation_path = source_dir / "validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"Validated report input does not exist: {validation_path}"
        )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("reconciliation", {}).get("all_row_totals_match", False):
        raise ReportInputError("validation.json reports unreconciled row totals")
    if not validation.get("reconciliation", {}).get("resolved_totals_match", False):
        raise ReportInputError("validation.json reports unreconciled resolved totals")

    overall = tables["overall"]
    if overall["estimator_label"].tolist() != ESTIMATOR_ORDER:
        raise ReportInputError("Overall table must contain exactly one row per estimator")
    expected_rows = int(validation["reconciliation"]["overall_replications"])
    if int(overall["replications"].sum()) != expected_rows:
        raise ReportInputError("Overall table conflicts with validation.json row totals")
    expected_resolved = int(
        validation["reconciliation"]["overall_resolved_replications"]
    )
    if int(overall["resolved_replications"].sum()) != expected_resolved:
        raise ReportInputError(
            "Overall table conflicts with validation.json resolved totals"
        )
    tables["validation"] = validation  # type: ignore[assignment]
    return tables


def prepare_overall_display(overall: pd.DataFrame) -> pd.DataFrame:
    """Create the concise main table without changing any metric."""
    ordered = order_estimators(overall)
    return pd.DataFrame(
        {
            "Estimator": ordered["estimator_label"].map(display_estimator_label),
            "Coverage": ordered["empirical_coverage"].map(format_percentage),
            "Bias": ordered["bias"].map(format_number),
            "MAE": ordered["mae"].map(format_number),
            "RMSE": ordered["rmse"].map(format_number),
            "Estimate SD": ordered["estimate_sd"].map(format_number),
            "Mean CR length": ordered["mean_cr_length"].map(format_number),
            "Median CR length": ordered["median_cr_length"].map(format_number),
            "Full grid": ordered["full_grid_rate"].map(format_percentage),
            "Unresolved": ordered["unresolved_rate"].map(format_percentage),
        }
    )


def prepare_diagnostics_display(overall: pd.DataFrame) -> pd.DataFrame:
    """Preserve DML's unavailable diagnostics as explicit ``NA`` strings."""
    ordered = order_estimators(overall)
    return pd.DataFrame(
        {
            "Estimator": ordered["estimator_label"].map(display_estimator_label),
            "Unresolved": ordered["unresolved_rate"].map(format_percentage),
            "Empty": ordered["empty_region_rate"].map(format_percentage),
            "Disconnected": ordered["disconnected_region_rate"].map(
                format_percentage
            ),
            "Boundary estimate": ordered["boundary_estimate_rate"].map(
                format_percentage
            ),
            "Iteration warning": ordered["iteration_warning_rate"].map(
                format_percentage
            ),
            "Rank failure": ordered["rank_failure_rate"].map(format_percentage),
            "Refinement limit": ordered["refinement_limit_rate"].map(
                format_percentage
            ),
        }
    )


def compute_coverage_dispersion(cell: pd.DataFrame) -> pd.DataFrame:
    """Summarize validated design-cell coverage without changing its denominator."""
    validate_required_columns(
        cell,
        {"estimator_label", "dgp", "n", "p", "pi", "tau", "empirical_coverage"},
        TABLE_FILES["cell"],
    )
    duplicate_keys = ["estimator_label", "dgp", "n", "p", "pi", "tau"]
    if cell.duplicated(duplicate_keys).any():
        raise ReportInputError("Validated design-cell table contains duplicate cells")

    rows: list[dict[str, object]] = []
    for label in ESTIMATOR_ORDER:
        values = pd.to_numeric(
            cell.loc[cell["estimator_label"].eq(label), "empirical_coverage"],
            errors="raise",
        ).to_numpy(dtype=float)
        if values.size == 0:
            raise ReportInputError(f"Design-cell table is missing estimator {label}")
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ReportInputError(f"Invalid design-cell coverage for {label}")
        quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
        rows.append(
            {
                "estimator_label": display_estimator_label(label),
                "design_cells": int(values.size),
                "minimum_coverage": float(values.min()),
                "p10_coverage": float(quantiles[0]),
                "p25_coverage": float(quantiles[1]),
                "median_coverage": float(quantiles[2]),
                "p75_coverage": float(quantiles[3]),
                "p90_coverage": float(quantiles[4]),
                "maximum_coverage": float(values.max()),
                "share_below_90": float(np.mean(values < 0.90)),
                "share_below_92_5": float(np.mean(values < 0.925)),
                "share_below_95": float(np.mean(values < 0.95)),
                "share_at_or_above_95": float(np.mean(values >= 0.95)),
            }
        )
    return pd.DataFrame(rows)


def prepare_coverage_dispersion_display(dispersion: pd.DataFrame) -> pd.DataFrame:
    """Format the design-cell dispersion summary for the PDF report."""
    return pd.DataFrame(
        {
            "Estimator": dispersion["estimator_label"],
            "Cells": dispersion["design_cells"].astype(int),
            "Min": dispersion["minimum_coverage"].map(format_percentage),
            "P10": dispersion["p10_coverage"].map(format_percentage),
            "P25": dispersion["p25_coverage"].map(format_percentage),
            "Median": dispersion["median_coverage"].map(format_percentage),
            "P75": dispersion["p75_coverage"].map(format_percentage),
            "P90": dispersion["p90_coverage"].map(format_percentage),
            "Max": dispersion["maximum_coverage"].map(format_percentage),
            "Below 90%": dispersion["share_below_90"].map(format_percentage),
            "Below 92.5%": dispersion["share_below_92_5"].map(format_percentage),
            "Below 95%": dispersion["share_below_95"].map(format_percentage),
            "At least 95%": dispersion["share_at_or_above_95"].map(
                format_percentage
            ),
        }
    )


def _format_display_metric(value: object, kind: str) -> str:
    if kind == "percentage":
        return format_percentage(value)
    if kind == "number":
        return format_number(value)
    if kind == "integer":
        return format_integer(value)
    raise ValueError(f"Unknown display metric kind: {kind}")


def _transposed_metric_display(
    frame: pd.DataFrame,
    *,
    metrics: Sequence[tuple[str, str, str]],
    first_column: str,
    group_columns: Sequence[str] = (),
    group_label: Any | None = None,
) -> pd.DataFrame:
    """Put estimator names in columns without altering validated metric values."""
    ordered = order_estimators(frame)
    if group_columns:
        groups = (
            ordered[list(group_columns)]
            .drop_duplicates()
            .sort_values(list(group_columns), kind="stable")
            .itertuples(index=False, name=None)
        )
    else:
        groups = [tuple()]

    rows: list[dict[str, str]] = []
    for group in groups:
        subset = ordered
        for column, value in zip(group_columns, group, strict=True):
            subset = subset.loc[subset[column].eq(value)]
        if set(subset["estimator_label"]) != set(ESTIMATOR_ORDER):
            raise ReportInputError(
                f"Comparison group {group!r} does not contain every estimator"
            )
        prefix = "" if group_label is None else f"{group_label(group)}: "
        for measure, source_column, kind in metrics:
            row = {first_column: prefix + measure}
            for estimator in ESTIMATOR_ORDER:
                values = subset.loc[
                    subset["estimator_label"].eq(estimator), source_column
                ]
                if len(values) != 1:
                    raise ReportInputError(
                        f"Comparison group {group!r} does not uniquely identify "
                        f"{estimator}"
                    )
                row[display_estimator_label(estimator)] = _format_display_metric(
                    values.iloc[0], kind
                )
            rows.append(row)
    return pd.DataFrame(rows)


def prepare_overall_transposed_display(overall: pd.DataFrame) -> pd.DataFrame:
    return _transposed_metric_display(
        overall,
        first_column="Measure",
        metrics=[
            ("Coverage", "empirical_coverage", "percentage"),
            ("Bias", "bias", "number"),
            ("MAE", "mae", "number"),
            ("RMSE", "rmse", "number"),
            ("Estimate SD", "estimate_sd", "number"),
            ("Mean CR length", "mean_cr_length", "number"),
            ("Median CR length", "median_cr_length", "number"),
            ("Full-grid rate", "full_grid_rate", "percentage"),
            ("Unresolved rate", "unresolved_rate", "percentage"),
        ],
    )


def prepare_quantile_transposed_display(quantile: pd.DataFrame) -> pd.DataFrame:
    return _transposed_metric_display(
        quantile,
        first_column="Quantile and measure",
        group_columns=("tau",),
        group_label=lambda group: rf"\(\tau={group[0]:.2f}\)",
        metrics=[
            ("Coverage", "empirical_coverage", "percentage"),
            ("Bias", "bias", "number"),
            ("MAE", "mae", "number"),
            ("RMSE", "rmse", "number"),
            ("Mean CR length", "mean_cr_length", "number"),
            ("Full-grid rate", "full_grid_rate", "percentage"),
            ("Unresolved rate", "unresolved_rate", "percentage"),
        ],
    )


def prepare_strength_transposed_display(strength: pd.DataFrame) -> pd.DataFrame:
    return _transposed_metric_display(
        strength,
        first_column="Strength and measure",
        group_columns=("pi",),
        group_label=lambda group: rf"\(\pi={group[0]:.2f}\)",
        metrics=[
            ("Coverage", "empirical_coverage", "percentage"),
            ("Bias", "bias", "number"),
            ("RMSE", "rmse", "number"),
            ("Mean CR length", "mean_cr_length", "number"),
            ("Full-grid rate", "full_grid_rate", "percentage"),
            ("Unresolved rate", "unresolved_rate", "percentage"),
        ],
    )


def prepare_n_p_transposed_display(n_p: pd.DataFrame) -> pd.DataFrame:
    return _transposed_metric_display(
        n_p,
        first_column="Design and measure",
        group_columns=("n", "p"),
        group_label=lambda group: f"n={int(group[0])}, p={int(group[1])}",
        metrics=[
            ("Coverage", "empirical_coverage", "percentage"),
            ("Bias", "bias", "number"),
            ("RMSE", "rmse", "number"),
            ("Mean CR length", "mean_cr_length", "number"),
            ("Full-grid rate", "full_grid_rate", "percentage"),
        ],
    )


def prepare_diagnostics_transposed_display(overall: pd.DataFrame) -> pd.DataFrame:
    return _transposed_metric_display(
        overall,
        first_column="Diagnostic",
        metrics=[
            ("Resolved replications", "resolved_replications", "integer"),
            ("Unresolved rate", "unresolved_rate", "percentage"),
            ("Empty-region rate", "empty_region_rate", "percentage"),
            ("Disconnected-region rate", "disconnected_region_rate", "percentage"),
            ("Boundary-estimate rate", "boundary_estimate_rate", "percentage"),
            ("Iteration-warning rate", "iteration_warning_rate", "percentage"),
            ("Rank-failure rate", "rank_failure_rate", "percentage"),
            ("Refinement-limit rate", "refinement_limit_rate", "percentage"),
        ],
    )


def prepare_coverage_dispersion_transposed_display(
    dispersion: pd.DataFrame,
) -> pd.DataFrame:
    raw_labels = {value: key for key, value in ESTIMATOR_DISPLAY_LABELS.items()}
    source = dispersion.copy()
    source["estimator_label"] = source["estimator_label"].map(raw_labels)
    if source["estimator_label"].isna().any():
        raise ReportInputError("Coverage-dispersion table has unknown estimator labels")
    return _transposed_metric_display(
        source,
        first_column="Coverage-dispersion measure",
        metrics=[
            ("Number of design cells", "design_cells", "integer"),
            ("Minimum coverage", "minimum_coverage", "percentage"),
            ("10th percentile", "p10_coverage", "percentage"),
            ("25th percentile", "p25_coverage", "percentage"),
            ("Median", "median_coverage", "percentage"),
            ("75th percentile", "p75_coverage", "percentage"),
            ("90th percentile", "p90_coverage", "percentage"),
            ("Maximum", "maximum_coverage", "percentage"),
            ("Share below 90%", "share_below_90", "percentage"),
            ("Share below 92.5%", "share_below_92_5", "percentage"),
            ("Share below 95%", "share_below_95", "percentage"),
            ("Share at or above 95%", "share_at_or_above_95", "percentage"),
        ],
    )


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _tabular_tex(
    frame: pd.DataFrame,
    *,
    align: str,
    caption: str,
    label: str,
    font_size: str = r"\small",
    row_breaks: Sequence[int] = (),
) -> str:
    headers = " & ".join(_latex_escape(column) for column in frame.columns)
    rows: list[str] = []
    for index, row in enumerate(frame.itertuples(index=False, name=None)):
        if index in row_breaks:
            rows.extend([r"\addlinespace[0.35em]", r"\midrule", r"\addlinespace[0.2em]"])
        rows.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[H]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{font_size}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{align}}}
\\toprule
{headers} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    return path


def _save_figure(figure: Figure, output_path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)
    return output_path


def _style_axis(axis: Axes, *, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="0.88", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)


def _line_figure(
    summary: pd.DataFrame,
    *,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    coverage: bool = False,
    percentage: bool = False,
) -> Path:
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    for label in ESTIMATOR_ORDER:
        values = summary.loc[summary["estimator_label"].eq(label)].sort_values(x)
        if values.empty:
            raise ReportInputError(f"Figure data are missing estimator {label}")
        x_values = values[x].to_numpy(dtype=float)
        y_values = values[y].to_numpy(dtype=float)
        if coverage:
            errors = 1.96 * values["coverage_mcse"].to_numpy(dtype=float)
            axis.errorbar(
                x_values,
                y_values,
                yerr=errors,
                color=ESTIMATOR_COLORS[label],
                marker=ESTIMATOR_MARKERS[label],
                linewidth=1.7,
                markersize=5,
                capsize=2.5,
                label=display_estimator_label(label, short=True),
            )
        else:
            axis.plot(
                x_values,
                y_values,
                color=ESTIMATOR_COLORS[label],
                marker=ESTIMATOR_MARKERS[label],
                linewidth=1.7,
                markersize=5,
                label=display_estimator_label(label, short=True),
            )
    if coverage:
        axis.axhline(0.95, color="0.25", linestyle="--", linewidth=1.1)
        percentage = True
    if percentage:
        axis.set_ylim(0.00, 1.00)
        axis.yaxis.set_major_formatter(lambda value, _position: f"{100 * value:.0f}%")
    _style_axis(axis, xlabel=xlabel, ylabel=ylabel)
    axis.legend(frameon=False, fontsize=8.5, ncol=1)
    return _save_figure(figure, output_path)


def _coverage_length_tradeoff(summary: pd.DataFrame, output_path: Path) -> Path:
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    for label in ESTIMATOR_ORDER:
        values = summary.loc[summary["estimator_label"].eq(label)].sort_values(
            "tau"
        )
        if values.empty:
            raise ReportInputError(f"Trade-off data are missing estimator {label}")
        x_values = values["mean_cr_length"].to_numpy(dtype=float)
        y_values = values["empirical_coverage"].to_numpy(dtype=float)
        errors = 1.96 * values["coverage_mcse"].to_numpy(dtype=float)
        axis.errorbar(
            x_values,
            y_values,
            yerr=errors,
            color=ESTIMATOR_COLORS[label],
            marker=ESTIMATOR_MARKERS[label],
            linewidth=1.4,
            markersize=6,
            capsize=3,
            label=display_estimator_label(label, short=True),
        )
    axis.axhline(0.95, color="0.25", linestyle="--", linewidth=1.1)
    axis.set_ylim(0.00, 1.00)
    axis.yaxis.set_major_formatter(lambda value, _position: f"{100 * value:.0f}%")
    _style_axis(
        axis,
        xlabel="Mean confidence-region length",
        ylabel="Resolved-replication coverage",
    )
    axis.legend(frameon=False, fontsize=8.5)
    return _save_figure(figure, output_path)


def _boundary_estimate_figure(overall: pd.DataFrame, output_path: Path) -> Path:
    ordered = order_estimators(overall)
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    positions = np.arange(len(ordered))
    values = ordered["boundary_estimate_rate"].to_numpy(dtype=float)
    colors = [ESTIMATOR_COLORS[label] for label in ordered["estimator_label"]]
    axis.bar(positions, values, color=colors, width=0.62)
    axis.set_xticks(
        positions,
        [display_estimator_label(label, short=True) for label in ordered["estimator_label"]],
    )
    axis.set_ylim(0.00, 1.00)
    axis.yaxis.set_major_formatter(lambda value, _position: f"{100 * value:.0f}%")
    _style_axis(
        axis,
        xlabel="Estimator",
        ylabel="Boundary-estimate rate",
    )
    return _save_figure(figure, output_path)


def _coverage_by_n_p_figure(n_p: pd.DataFrame, output_path: Path) -> Path:
    combinations = (
        n_p[["n", "p"]]
        .drop_duplicates()
        .sort_values(["n", "p"], kind="stable")
        .reset_index(drop=True)
    )
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    positions = np.arange(len(combinations))
    for label in ESTIMATOR_ORDER:
        values = n_p.loc[n_p["estimator_label"].eq(label)].sort_values(
            ["n", "p"], kind="stable"
        )
        if len(values) != len(combinations):
            raise ReportInputError(f"n,p figure is missing estimator {label}")
        axis.errorbar(
            positions,
            values["empirical_coverage"],
            yerr=1.96 * values["coverage_mcse"],
            color=ESTIMATOR_COLORS[label],
            marker=ESTIMATOR_MARKERS[label],
            linewidth=1.7,
            markersize=5,
            capsize=2.5,
            label=display_estimator_label(label, short=True),
        )
    axis.axhline(0.95, color="0.25", linestyle="--", linewidth=1.1)
    axis.set_xticks(
        positions,
        [
            f"n={int(row.n)}\np={int(row.p)}"
            for row in combinations.itertuples(index=False)
        ],
    )
    axis.set_ylim(0.00, 1.00)
    axis.yaxis.set_major_formatter(lambda value, _position: f"{100 * value:.0f}%")
    _style_axis(
        axis,
        xlabel="Sample size and control dimension",
        ylabel="Resolved-replication coverage",
    )
    axis.legend(frameon=False, fontsize=8.5)
    return _save_figure(figure, output_path)


def _selected_controls_figure(cell: pd.DataFrame, output_path: Path) -> Path:
    post = cell.loc[cell["estimator_label"].eq("Post-selection IVQR")].copy()
    selected = pd.to_numeric(post["mean_selected_controls"], errors="coerce")
    if selected.isna().all():
        raise ReportInputError("Selected-control diagnostics are unavailable")
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    scatter = axis.scatter(
        selected,
        post["empirical_coverage"],
        c=post["tau"],
        cmap="viridis",
        alpha=0.65,
        s=24,
        edgecolors="none",
    )
    axis.axhline(0.95, color="0.25", linestyle="--", linewidth=1.1)
    axis.set_ylim(0.00, 1.00)
    axis.yaxis.set_major_formatter(lambda value, _position: f"{100 * value:.0f}%")
    _style_axis(
        axis,
        xlabel="Mean selected controls in design cell",
        ylabel="Resolved-replication coverage",
    )
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label(r"Quantile $\tau$")
    return _save_figure(figure, output_path)


def _design_table(validation: Mapping[str, Any]) -> pd.DataFrame:
    panel = validation["panel"]["DML-IVQR"]
    values = panel["design_values"]
    dgp_values = [str(value) for value in values["dgp"]]
    p_reference = int(values["p"][0])
    support = ", ".join(
        f"{dgp.upper()}: {get_oracle_control_count(dgp, p_reference)} active controls"
        for dgp in dgp_values
    )
    truth_parts: list[str] = []
    for dgp in dgp_values:
        truths = [true_alpha(float(tau), dgp, df=DF_T) for tau in values["tau"]]
        rendered = ", ".join(f"{truth:.4f}" for truth in truths)
        truth_parts.append(f"{dgp.upper()}: ({rendered})")
    grid_step = (DEFAULT_ALPHA_MAX - DEFAULT_ALPHA_MIN) / (
        DEFAULT_ALPHA_GRID_SIZE - 1
    )
    return pd.DataFrame(
        [
            ("DGPs", ", ".join(dgp.upper() for dgp in dgp_values)),
            ("Active support", support),
            ("Sample sizes", ", ".join(map(str, values["n"]))),
            ("Control dimensions", ", ".join(map(str, values["p"]))),
            ("Instrument-strength index pi", ", ".join(map(str, values["pi"]))),
            ("Quantiles tau", ", ".join(map(str, values["tau"]))),
            ("Replications per cell", str(panel["replications_per_design_min"])),
            ("Design cells", str(panel["design_cells"])),
            (
                "True alpha by tau=(0.25, 0.50, 0.75)",
                "; ".join(truth_parts),
            ),
            (
                "Initial alpha grid",
                f"[{DEFAULT_ALPHA_MIN:g}, {DEFAULT_ALPHA_MAX:g}], "
                f"{DEFAULT_ALPHA_GRID_SIZE} points, spacing {grid_step:g}",
            ),
            (
                "CR refinement",
                f"Adaptive boundary refinement; tolerance {DEFAULT_REFINEMENT_TOLERANCE:g}",
            ),
            ("Nominal confidence level", "95%"),
            (
                "Critical-value multiplier",
                f"{DEFAULT_CRITICAL_VALUE_MULTIPLIER:g}",
            ),
            (
                "Estimators",
                "DML-style residualized IVQR; true-support Oracle IVQR; "
                "Mean-Lasso Post-selection IVQR",
            ),
        ],
        columns=["Component", "Validated/configured value"],
    )


def _worst_display(worst: pd.DataFrame) -> pd.DataFrame:
    lowest = worst.sort_values(
        ["empirical_coverage", "estimator_label", "dgp", "n", "p", "pi", "tau"],
        kind="stable",
    ).head(10)
    return pd.DataFrame(
        {
            "Estimator": lowest["estimator_label"].map(display_estimator_label),
            "DGP": lowest["dgp"].str.upper(),
            "n": lowest["n"].astype(int).astype(str),
            "p": lowest["p"].astype(int).astype(str),
            "pi": lowest["pi"].map(lambda value: f"{value:g}"),
            "tau": lowest["tau"].map(lambda value: f"{value:g}"),
            "Resolved": lowest["resolved_replications"].astype(int).astype(str),
            "Coverage": lowest["empirical_coverage"].map(format_percentage),
            "MCSE": lowest["coverage_mcse"].map(format_percentage),
            "Bias": lowest["bias"].map(format_number),
            "RMSE": lowest["rmse"].map(format_number),
            "Mean CR length": lowest["mean_cr_length"].map(format_number),
            "Full grid": lowest["full_grid_rate"].map(format_percentage),
            "Unresolved": lowest["unresolved_rate"].map(format_percentage),
        }
    )


def _worst_table_tex(display: pd.DataFrame) -> str:
    design_columns = [
        "Estimator", "DGP", "n", "p", "pi", "tau", "Resolved", "Coverage", "MCSE"
    ]
    performance_columns = [
        "Estimator", "DGP", "n", "p", "pi", "tau", "Bias", "RMSE",
        "Mean CR length", "Full grid", "Unresolved",
    ]

    panel_a = _tabular_tex(
        display[design_columns],
        align="llrrrrrrr",
        caption="Ten lowest-coverage design cells: design and coverage",
        label="tbl-worst-cells-a",
        font_size=r"\footnotesize",
    )
    panel_b = _tabular_tex(
        display[performance_columns],
        align="llrrrrrrrrr",
        caption="Ten lowest-coverage design cells: performance and diagnostics",
        label="tbl-worst-cells-b",
        font_size=r"\footnotesize",
    )
    return panel_a + "\n" + panel_b


def _long_design_cell_tex(cell: pd.DataFrame) -> str:
    ordered = order_estimators(cell)
    display = pd.DataFrame(
        {
            "Estimator": ordered["estimator_label"].map(
                lambda label: display_estimator_label(label, short=True)
            ),
            "DGP": ordered["dgp"].str.upper(),
            "n": ordered["n"].astype(int),
            "p": ordered["p"].astype(int),
            "pi": ordered["pi"].map(lambda value: f"{value:g}"),
            "tau": ordered["tau"].map(lambda value: f"{value:g}"),
            "R-res.": ordered["resolved_replications"].astype(int),
            "Coverage": ordered["empirical_coverage"].map(format_percentage),
            "Bias": ordered["bias"].map(format_number),
            "RMSE": ordered["rmse"].map(format_number),
            "Mean CR": ordered["mean_cr_length"].map(format_number),
            "Full grid": ordered["full_grid_rate"].map(format_percentage),
            "Unresolved": ordered["unresolved_rate"].map(format_percentage),
        }
    )
    header = " & ".join(_latex_escape(column) for column in display.columns)
    rows = "\n".join(
        " & ".join(_latex_escape(value) for value in row) + r" \\"
        for row in display.itertuples(index=False, name=None)
    )
    return f"""\\begin{{landscape}}
\\begingroup
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\renewcommand{{\\arraystretch}}{{0.88}}
\\begin{{longtable}}{{llrrrrrrrrrrr}}
\\caption{{Complete results by estimator and design cell}}\\label{{tbl-all-cells}}\\\\
\\toprule
{header} \\\\
\\midrule
\\endfirsthead
\\multicolumn{{13}}{{l}}{{\\footnotesize Table \\thetable\\ (continued)}}\\\\
\\toprule
{header} \\\\
\\midrule
\\endhead
\\midrule
\\multicolumn{{13}}{{r}}{{\\footnotesize Continued on next page}}\\\\
\\endfoot
\\bottomrule
\\endlastfoot
{rows}
\\end{{longtable}}
\\endgroup
\\end{{landscape}}
"""


def _n_p_display(n_p: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Estimator": n_p["estimator_label"].map(display_estimator_label),
            "n": n_p["n"].astype(int),
            "p": n_p["p"].astype(int),
            "Coverage": n_p["empirical_coverage"].map(format_percentage),
            "Bias": n_p["bias"].map(format_number),
            "RMSE": n_p["rmse"].map(format_number),
            "Mean CR length": n_p["mean_cr_length"].map(format_number),
            "Full grid": n_p["full_grid_rate"].map(format_percentage),
            "Unresolved": n_p["unresolved_rate"].map(format_percentage),
        }
    )


def _status_display(diagnostics: pd.DataFrame) -> pd.DataFrame:
    display = diagnostics.copy()
    return pd.DataFrame(
        {
            "Estimator": display["estimator_label"].map(display_estimator_label),
            "Status": display["cr_status_standardized"],
            "Rows": display["replications"].astype(int),
            "Resolved": display["resolved_replications"].astype(int),
            "Covered": display["covered_resolved"].astype(int),
            "Uncovered": display["uncovered_resolved"].astype(int),
            "Unresolved": display["unresolved_replications"].astype(int),
            "Coverage": display["empirical_coverage"].map(format_percentage),
        }
    )


def _narrative_assets(tables: Mapping[str, Any], output_dir: Path) -> list[Path]:
    overall = tables["overall"].set_index("estimator_label")
    quantile = tables["quantile"]
    strength = tables["strength"]
    worst = tables["worst"].sort_values("empirical_coverage", kind="stable")
    dispersion = compute_coverage_dispersion(tables["cell"]).set_index(
        "estimator_label"
    )
    dml = overall.loc["DML-IVQR"]
    oracle = overall.loc["Oracle IVQR"]
    post = overall.loc["Post-selection IVQR"]
    post_upper = quantile.loc[
        quantile["estimator_label"].eq("Post-selection IVQR")
        & np.isclose(quantile["tau"], 0.75)
    ].iloc[0]
    weakest = strength.loc[np.isclose(strength["pi"], 0.1)]
    strongest = strength.loc[np.isclose(strength["pi"], 1.0)]
    weak_lengths = weakest.set_index("estimator_label")["mean_cr_length"]
    strong_lengths = strongest.set_index("estimator_label")["mean_cr_length"]
    worst_row = worst.iloc[0]
    dml_dispersion = dispersion.loc[ESTIMATOR_DISPLAY_LABELS["DML-IVQR"]]
    oracle_dispersion = dispersion.loc[ESTIMATOR_DISPLAY_LABELS["Oracle IVQR"]]
    post_dispersion = dispersion.loc[
        ESTIMATOR_DISPLAY_LABELS["Post-selection IVQR"]
    ]
    panel = tables["validation"]["panel"]["DML-IVQR"]
    design_values = panel["design_values"]
    rows_per_estimator = int(overall["replications"].iloc[0])
    rows_overall = int(overall["replications"].sum())

    texts = {
        "design_summary.md": f"""The validated design contains {len(design_values['dgp'])} DGPs,
{len(design_values['n'])} sample sizes, {len(design_values['p'])} control dimensions,
{len(design_values['pi'])} instrument-strength settings, and
{len(design_values['tau'])} quantiles. Their Cartesian product gives
{int(panel['design_cells'])} design cells with
{int(panel['replications_per_design_min'])} replications per cell,
{rows_per_estimator:,} replications per estimator, and {rows_overall:,} rows overall.
""",
        "headline_values.md": f"""The completed full run contains 500 replications in each of 144 design cells.
The implemented DML-style IVQR procedure attains **{format_percentage(dml['empirical_coverage'])}** coverage on
resolved replications, compared with **{format_percentage(oracle['empirical_coverage'])}**
for Oracle IVQR and **{format_percentage(post['empirical_coverage'])}** for
Mean-Lasso Post-selection IVQR. The DML-style procedure's stronger calibration comes with a mean confidence-region
length of {format_number(dml['mean_cr_length'])} and a
{format_percentage(dml['full_grid_rate'])} full-grid rate. The central concern is
Mean-Lasso Post-selection IVQR: its coverage falls to **{format_percentage(post_upper['empirical_coverage'])}**
at $\\tau=0.75$, and the worst reported cell covers only
**{format_percentage(worst_row['empirical_coverage'], 1)}**.
""",
        "overall_findings.md": f"""The implemented DML-style IVQR procedure has the highest overall coverage
({format_percentage(dml['empirical_coverage'])}), but also the longest mean region
({format_number(dml['mean_cr_length'])}). Oracle IVQR has the smallest MAE
({format_number(oracle['mae'])}) and RMSE ({format_number(oracle['rmse'])}), making it
the strongest point-estimation benchmark. Mean-Lasso Post-selection IVQR produces a slightly shorter
mean region than Oracle ({format_number(post['mean_cr_length'])} versus
{format_number(oracle['mean_cr_length'])}) but covers only
{format_percentage(post['empirical_coverage'])}. The shorter set is therefore not
evidence of superior inference.
""",
        "coverage_dispersion_findings.md": f"""Overall averages conceal meaningful design-level heterogeneity.
The implemented DML-style IVQR procedure has a lower-tail (10th-percentile) cell
coverage of {format_percentage(dml_dispersion['p10_coverage'])}, and
{format_percentage(dml_dispersion['share_below_90'])} of its cells fall below 90%.
Oracle IVQR shows mild finite-sample undercoverage in its lower tail, with a
10th percentile of {format_percentage(oracle_dispersion['p10_coverage'])}.
Mean-Lasso Post-selection IVQR has both frequent and severe undercoverage:
{format_percentage(post_dispersion['share_below_90'])} of cells fall below 90%,
and its minimum is {format_percentage(post_dispersion['minimum_coverage'], 1)}.
Average calibration must therefore be assessed alongside lower-tail cell performance.
""",
        "quantile_findings.md": f"""The main instability is concentrated in the
upper quantile. Mean-Lasso Post-selection IVQR coverage is
{format_percentage(post_upper['empirical_coverage'])} at $\\tau=0.75$, with bias
{format_number(post_upper['bias'])}; its coverage is
{format_percentage(quantile.loc[(quantile['estimator_label'].eq('Post-selection IVQR')) & np.isclose(quantile['tau'], 0.25), 'empirical_coverage'].iloc[0])}
at $\\tau=0.25$. Oracle also weakens at $\\tau=0.75$ but less severely, while DML
remains close to nominal at each reported quantile. The coincident negative upper-
quantile bias is consistent with, but does not by itself prove, a selection mechanism.
""",
        "strength_findings.md": f"""All estimators become more informative as the
instrument-strength index increases. For $\\pi=0.1$, mean CR lengths are
{format_number(weak_lengths['DML-IVQR'])} (DML-style),
{format_number(weak_lengths['Oracle IVQR'])} (Oracle), and
{format_number(weak_lengths['Post-selection IVQR'])} (Mean-Lasso post-selection). At $\\pi=1$,
the corresponding lengths fall to {format_number(strong_lengths['DML-IVQR'])},
{format_number(strong_lengths['Oracle IVQR'])}, and
{format_number(strong_lengths['Post-selection IVQR'])}. This is the expected
informativeness response to stronger identification. Mean-Lasso post-selection undercoverage,
however, persists—and is most visible at $\\pi=1$—so weak identification is not its
only plausible source.
""",
        "worst_findings.md": f"""The lowest-coverage cell is
{_latex_escape(display_estimator_label(str(worst_row['estimator_label'])))},
{str(worst_row['dgp']).upper()}, $n={int(worst_row['n'])}$,
$p={int(worst_row['p'])}$, $\\pi={worst_row['pi']:g}$, and
$\\tau={worst_row['tau']:g}$, with coverage
{format_percentage(worst_row['empirical_coverage'], 1)} and MCSE
{format_percentage(worst_row['coverage_mcse'], 2)}. A 95% Monte Carlo interval is
approximately [{format_percentage(worst_row['coverage_mc95_lower'], 1)},
{format_percentage(worst_row['coverage_mc95_upper'], 1)}], far below 95%. Sampling
noise from 500 replications cannot plausibly explain that gap.
""",
        "diagnostic_findings.md": f"""The resolved denominator excludes 43 DML rows
({format_percentage(dml['unresolved_rate'], 3)}) and two Mean-Lasso post-selection rows
({format_percentage(post['unresolved_rate'], 3)}); Oracle has no unresolved rows.
DML's legacy file contains no explicit CR status, component, iteration-warning,
rank-failure, or refinement metadata. Those entries are intentionally reported as
**NA**, not zero. Oracle and Mean-Lasso post-selection record zero rank failures and zero
refinement-limit hits in these validated summaries.
""",
    }
    return [_write_text(output_dir / name, text) for name, text in texts.items()]


def generate_report_assets(
    report_dir: Path = DEFAULT_REPORT_DIR,
    output_dir: Path = DEFAULT_ASSET_DIR,
) -> list[Path]:
    """Generate deterministic tables, vector figures, and data-derived prose."""
    tables = load_validated_tables(report_dir)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "savefig.transparent": False,
        }
    )

    overall_display = prepare_overall_transposed_display(tables["overall"])
    diagnostics_display = prepare_diagnostics_transposed_display(tables["overall"])
    coverage_dispersion = compute_coverage_dispersion(tables["cell"])
    coverage_dispersion_display = prepare_coverage_dispersion_transposed_display(
        coverage_dispersion
    )
    quantile_display = prepare_quantile_transposed_display(tables["quantile"])
    strength_display = prepare_strength_transposed_display(tables["strength"])
    n_p_display = prepare_n_p_transposed_display(tables["n_p"])

    csv_displays = {
        "table_overall.csv": overall_display,
        "table_overall_transposed.csv": overall_display,
        "table_coverage_dispersion.csv": coverage_dispersion_display,
        "table_coverage_dispersion_transposed.csv": coverage_dispersion_display,
        "table_by_quantile.csv": quantile_display,
        "table_by_quantile_transposed.csv": quantile_display,
        "table_by_strength.csv": strength_display,
        "table_by_strength_transposed.csv": strength_display,
        "table_by_n_p.csv": n_p_display,
        "table_by_n_p_transposed.csv": n_p_display,
        "table_worst_cells.csv": _worst_display(tables["worst"]),
        "table_diagnostics.csv": diagnostics_display,
        "table_diagnostics_transposed.csv": diagnostics_display,
    }
    outputs: list[Path] = []
    for filename, frame in csv_displays.items():
        path = destination / filename
        frame.to_csv(path, index=False, lineterminator="\n")
        outputs.append(path)

    comparison_align = (
        r"p{0.31\textwidth}p{0.20\textwidth}"
        r"p{0.16\textwidth}p{0.25\textwidth}"
    )
    overall_tex = _tabular_tex(
            overall_display,
            align=comparison_align,
            caption="Overall finite-sample performance",
            label="tbl-overall-performance",
        )
    dispersion_tex = _tabular_tex(
            coverage_dispersion_display,
            align=comparison_align,
            caption="Distribution of resolved-replication coverage across design cells",
            label="tbl-coverage-dispersion",
            font_size=r"\footnotesize",
        )
    quantile_tex = _tabular_tex(
            quantile_display,
            align=comparison_align,
            caption="Performance by structural quantile",
            label="tbl-quantile",
            font_size=r"\footnotesize",
            row_breaks=(7, 14),
        )
    strength_tex = _tabular_tex(
            strength_display,
            align=comparison_align,
            caption="Coverage and informativeness by instrument-strength index",
            label="tbl-strength",
            font_size=r"\footnotesize",
            row_breaks=(6, 12, 18),
        )
    diagnostics_tex = _tabular_tex(
            diagnostics_display,
            align=comparison_align,
            caption="Numerical and confidence-region diagnostics",
            label="tbl-diagnostics",
        )
    n_p_tex = _tabular_tex(
            n_p_display,
            align=comparison_align,
            caption="Performance by sample size and control dimension",
            label="tbl-n-p",
            font_size=r"\footnotesize",
            row_breaks=(5, 10, 15),
        )
    tex_assets = {
        "table_overall.tex": overall_tex,
        "table_overall_transposed.tex": overall_tex,
        "table_coverage_dispersion.tex": dispersion_tex,
        "table_coverage_dispersion_transposed.tex": dispersion_tex,
        "table_by_quantile.tex": quantile_tex,
        "table_by_quantile_transposed.tex": quantile_tex,
        "table_by_strength.tex": strength_tex,
        "table_by_strength_transposed.tex": strength_tex,
        "table_worst_cells.tex": _worst_table_tex(_worst_display(tables["worst"])),
        "table_diagnostics.tex": diagnostics_tex,
        "table_diagnostics_transposed.tex": diagnostics_tex,
        "table_design.tex": _tabular_tex(
            _design_table(tables["validation"]),
            align=r"p{0.28\textwidth}p{0.66\textwidth}",
            caption="Validated Monte Carlo design",
            label="tbl-design",
        ),
        "table_by_n_p.tex": n_p_tex,
        "table_by_n_p_transposed.tex": n_p_tex,
        "table_status_detail.tex": _tabular_tex(
            _status_display(tables["diagnostics"]),
            align="llrrrrrr",
            caption="Detailed standardized confidence-region statuses",
            label="tbl-status-detail",
            font_size=r"\footnotesize",
        ),
        "table_design_cells.tex": _long_design_cell_tex(tables["cell"]),
    }
    for filename, content in tex_assets.items():
        outputs.append(_write_text(destination / filename, content))

    figure_specs = [
        (
            "figure_coverage_by_quantile.pdf",
            lambda path: _line_figure(
                tables["quantile"],
                x="tau",
                y="empirical_coverage",
                xlabel=r"Structural quantile $\tau$",
                ylabel="Resolved-replication coverage",
                output_path=path,
                coverage=True,
            ),
        ),
        (
            "figure_coverage_by_strength.pdf",
            lambda path: _line_figure(
                tables["strength"],
                x="pi",
                y="empirical_coverage",
                xlabel=r"Instrument-strength index $\pi$",
                ylabel="Resolved-replication coverage",
                output_path=path,
                coverage=True,
            ),
        ),
        (
            "figure_bias_by_quantile.pdf",
            lambda path: _line_figure(
                tables["quantile"],
                x="tau",
                y="bias",
                xlabel=r"Structural quantile $\tau$",
                ylabel="Bias",
                output_path=path,
            ),
        ),
        (
            "figure_rmse_by_quantile.pdf",
            lambda path: _line_figure(
                tables["quantile"],
                x="tau",
                y="rmse",
                xlabel=r"Structural quantile $\tau$",
                ylabel="RMSE",
                output_path=path,
            ),
        ),
        (
            "figure_cr_length_by_strength.pdf",
            lambda path: _line_figure(
                tables["strength"],
                x="pi",
                y="mean_cr_length",
                xlabel=r"Instrument-strength index $\pi$",
                ylabel="Mean confidence-region length",
                output_path=path,
            ),
        ),
        (
            "figure_full_grid_by_strength.pdf",
            lambda path: _line_figure(
                tables["strength"],
                x="pi",
                y="full_grid_rate",
                xlabel=r"Instrument-strength index $\pi$",
                ylabel="Full-grid rate",
                output_path=path,
                percentage=True,
            ),
        ),
        (
            "figure_coverage_length_tradeoff.pdf",
            lambda path: _coverage_length_tradeoff(tables["quantile"], path),
        ),
        (
            "figure_boundary_estimates.pdf",
            lambda path: _boundary_estimate_figure(tables["overall"], path),
        ),
        (
            "figure_coverage_by_n_p.pdf",
            lambda path: _coverage_by_n_p_figure(tables["n_p"], path),
        ),
        (
            "figure_post_selection_controls_vs_coverage.pdf",
            lambda path: _selected_controls_figure(tables["cell"], path),
        ),
    ]
    for filename, generator in figure_specs:
        outputs.append(generator(destination / filename))

    outputs.extend(_narrative_assets(tables, destination))
    manifest = {
        "source_directory": str(report_dir.expanduser().resolve()),
        "coverage_source": "validated summary CSVs; no replication-level recomputation",
        "estimator_order": ESTIMATOR_ORDER,
        "generated_files": sorted(path.name for path in outputs),
    }
    manifest_path = destination / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    outputs.append(manifest_path)
    return sorted(outputs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ASSET_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = generate_report_assets(args.report_dir, args.output_dir)
    print(f"Generated {len(outputs)} report assets in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
