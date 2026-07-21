"""Build the deterministic thesis-facing package from validated Phase 1/2 outputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from analysis.r500_thesis_package import (  # noqa: E402
    CSV_MISSING,
    ESTIMATOR_LABELS,
    ESTIMATOR_ORDER,
    FIGURE_FAMILIES,
    FLOAT_FORMAT,
    PAIR_ORIENTATION,
    SOURCE_CONTRACTS,
    TABLE_FAMILIES,
    aggregate_performance,
    build_design_table,
    build_overall_table,
    build_tradeoff_table,
    build_warning_exception_table,
    build_weakest_table,
    consistency_row,
    dataframe_to_latex,
    finding,
    json_ready,
    load_sources,
    require_consistency,
    sha256_file,
    sourced_sentence,
    validate_findings,
    validate_historical_multiplier_outputs,
    validate_report_provenance,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "thesis" / "r500"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
COLORS = {"oracle": "#1f77b4", "post_selection": "#d95f02", "dml": "#2a9d8f"}
MARKERS = {"oracle": "o", "post_selection": "s", "dml": "^"}
FIGURE_SIZE = (7.0, 4.6)
THREE_PANEL_SIZE = (8.0, 8.5)


FAMILY_SOURCES: dict[str, list[str]] = {
    "table_01_simulation_design": [
        "results/validation/r500_audit/structural_validation.json",
        "results/validation/r500_audit/scenario_summary.csv",
    ],
    "table_02_overall_estimator_performance": [
        "results/validation/r500_audit/estimator_summary.csv",
        "results/validation/r500_phase2/estimator_classification.csv",
    ],
    "table_03_estimator_tradeoffs": [
        "results/validation/r500_phase2/paired_estimator_summary.csv"
    ],
    "table_04_performance_by_quantile": [
        "results/validation/r500_audit/scenario_summary.csv",
        "results/validation/r500_phase2/coverage_uncertainty.csv",
    ],
    "table_05_performance_by_instrument_strength": [
        "results/validation/r500_audit/scenario_summary.csv",
        "results/validation/r500_phase2/coverage_uncertainty.csv",
    ],
    "table_06_weakest_scenarios": [
        "results/validation/r500_phase2/tables/worst_coverage_scenarios.csv",
        "results/validation/r500_phase2/coverage_uncertainty.csv",
        "results/validation/r500_phase2/warning_scenario_summary.csv",
        "results/validation/r500_phase2/exception_scenario_summary.csv",
    ],
    "table_07_warning_exception_diagnostics": [
        "results/validation/r500_phase2/warning_summary.csv",
        "results/validation/r500_phase2/exception_rows.csv",
    ],
    "figure_01_overall_estimator_tradeoff": [
        "results/validation/r500_audit/estimator_summary.csv",
        "results/validation/r500_phase2/estimator_classification.csv",
    ],
    "figure_02_coverage_by_quantile": [
        "results/validation/r500_audit/scenario_summary.csv",
        "results/validation/r500_phase2/coverage_uncertainty.csv",
    ],
    "figure_03_coverage_by_instrument_strength": [
        "results/validation/r500_audit/scenario_summary.csv",
        "results/validation/r500_phase2/coverage_uncertainty.csv",
    ],
    "figure_04_rmse_by_instrument_strength": [
        "results/validation/r500_audit/scenario_summary.csv"
    ],
    "figure_05_cr_length_by_instrument_strength": [
        "results/validation/r500_audit/scenario_summary.csv"
    ],
    "figure_06_paired_estimator_differences": [
        "results/validation/r500_phase2/paired_estimator_summary.csv"
    ],
    "figure_07_weak_scenario_structure": [
        "results/validation/r500_phase2/coverage_uncertainty.csv"
    ],
    "figure_08_warning_exception_diagnostics": [
        "results/validation/r500_phase2/warning_summary.csv",
        "results/validation/r500_phase2/exception_rows.csv",
    ],
}
FAMILY_SOURCE_COLUMNS: dict[str, dict[str, list[str]]] = {
    "table_01_simulation_design": {
        "results/validation/r500_audit/structural_validation.json": [
            "structure.*.rows", "structure.*.unique_replications", "structure.*.design_cells"
        ],
        "results/validation/r500_audit/scenario_summary.csv": ["dgp", "n", "p", "pi", "tau"],
    },
    "table_02_overall_estimator_performance": {
        "results/validation/r500_audit/estimator_summary.csv": [
            "conditional_coverage", "coverage_denominator", "bias", "rmse",
            "mean_cr_length", "empty_valid_rate", "unresolved_rate", "iteration_warning_rate",
        ],
        "results/validation/r500_phase2/estimator_classification.csv": [
            "overall_wilson95_lower", "overall_wilson95_upper", "coverage_gap",
            "classification", "diagnostic_confidence",
        ],
    },
    "table_03_estimator_tradeoffs": {
        "results/validation/r500_phase2/paired_estimator_summary.csv": [
            "estimator_a", "estimator_b", "metric", "mean_paired_difference",
            "paired_ci95_lower", "paired_ci95_upper", "valid_paired_denominator",
        ]
    },
    "table_04_performance_by_quantile": {
        "results/validation/r500_audit/scenario_summary.csv": [
            "estimator", "tau", "observations", "bias", "rmse", "mean_cr_length",
            "coverage_denominator", "missing_estimate_rate", "missing_cr_rate",
        ],
        "results/validation/r500_phase2/coverage_uncertainty.csv": [
            "coverage_successes", "coverage_denominator"
        ],
    },
    "table_05_performance_by_instrument_strength": {
        "results/validation/r500_audit/scenario_summary.csv": [
            "estimator", "pi", "observations", "bias", "rmse", "mean_cr_length",
            "coverage_denominator", "missing_estimate_rate", "missing_cr_rate",
        ],
        "results/validation/r500_phase2/coverage_uncertainty.csv": [
            "coverage_successes", "coverage_denominator"
        ],
    },
    "table_06_weakest_scenarios": {
        "results/validation/r500_phase2/tables/worst_coverage_scenarios.csv": [
            "estimator", "dgp", "n", "p", "pi", "tau", "conditional_coverage",
            "coverage_denominator", "coverage_gap", "rmse", "mean_cr_length",
        ],
        "results/validation/r500_phase2/coverage_uncertainty.csv": [
            "wilson95_lower", "wilson95_upper"
        ],
        "results/validation/r500_phase2/warning_scenario_summary.csv": [
            "warning_category", "warning_frequency"
        ],
        "results/validation/r500_phase2/exception_scenario_summary.csv": [
            "exception_type", "exception_count"
        ],
    },
    "table_07_warning_exception_diagnostics": {
        "results/validation/r500_phase2/warning_summary.csv": [
            "warning_frequency", "warning_event_count", "coverage_affected_valid",
            "coverage_without_warning_valid", "rmse", "rmse_without_warning",
        ],
        "results/validation/r500_phase2/exception_rows.csv": [
            "estimator_name", "exception_type"
        ],
    },
}
for figure_family, table_family in {
    "figure_01_overall_estimator_tradeoff": "table_02_overall_estimator_performance",
    "figure_02_coverage_by_quantile": "table_04_performance_by_quantile",
    "figure_03_coverage_by_instrument_strength": "table_05_performance_by_instrument_strength",
    "figure_04_rmse_by_instrument_strength": "table_05_performance_by_instrument_strength",
    "figure_05_cr_length_by_instrument_strength": "table_05_performance_by_instrument_strength",
    "figure_06_paired_estimator_differences": "table_03_estimator_tradeoffs",
    "figure_07_weak_scenario_structure": "table_06_weakest_scenarios",
    "figure_08_warning_exception_diagnostics": "table_07_warning_exception_diagnostics",
}.items():
    if figure_family != "figure_07_weak_scenario_structure":
        FAMILY_SOURCES[figure_family] = FAMILY_SOURCES[table_family]
        FAMILY_SOURCE_COLUMNS[figure_family] = FAMILY_SOURCE_COLUMNS[table_family]
FAMILY_SOURCE_COLUMNS["figure_07_weak_scenario_structure"] = {
    "results/validation/r500_phase2/coverage_uncertainty.csv": [
        "estimator", "dgp", "n", "p", "pi", "tau", "coverage_gap"
    ]
}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(
        path, index=False, na_rep=CSV_MISSING, float_format=FLOAT_FORMAT, lineterminator="\n"
    )


def _write_json(payload: object, path: Path) -> None:
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_table(frame: pd.DataFrame, family: str) -> list[Path]:
    csv_path = TABLE_DIR / f"{family}.csv"
    tex_path = TABLE_DIR / f"{family}.tex"
    _write_csv(frame, csv_path)
    tex_path.write_text(dataframe_to_latex(frame), encoding="utf-8", newline="\n")
    return [csv_path, tex_path]


def _base_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.set_ylabel(ylabel, fontsize=10)
    axis.tick_params(labelsize=9)
    axis.grid(axis="y", color="0.82", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def _save_figure(figure: plt.Figure, family: str, *, tight: bool = True) -> list[Path]:
    pdf = FIGURE_DIR / f"{family}.pdf"
    png = FIGURE_DIR / f"{family}.png"
    if tight:
        figure.tight_layout()
    figure.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Creator": "R500 thesis package", "CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        png,
        dpi=240,
        bbox_inches="tight",
        metadata={"Software": "R500 thesis package"},
    )
    plt.close(figure)
    return [pdf, png]


def _overall_figure(table: pd.DataFrame) -> list[Path]:
    figure, axes = plt.subplots(3, 1, figsize=THREE_PANEL_SIZE)
    x = np.arange(len(table))
    labels = [ESTIMATOR_LABELS[value] for value in table["estimator"]]
    metrics = (
        ("conditional_coverage", "Coverage", 0.95),
        ("rmse", "RMSE", None),
        ("mean_cr_length", "Mean CR length", None),
    )
    for axis, (metric, label, reference) in zip(axes, metrics, strict=True):
        axis.bar(x, table[metric], color=[COLORS[value] for value in table["estimator"]])
        if reference is not None:
            axis.axhline(reference, color="0.25", linestyle="--", linewidth=1)
        axis.set_xticks(x, labels)
        _base_axis(axis, label)
    return _save_figure(figure, "figure_01_overall_estimator_tradeoff")


def _coverage_lines(table: pd.DataFrame, dimension: str, family: str) -> list[Path]:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for estimator in ESTIMATOR_ORDER:
        values = table[table["estimator"].eq(estimator)].sort_values(dimension)
        x = values[dimension].to_numpy(float)
        y = values["conditional_coverage"].to_numpy(float)
        axis.errorbar(
            x,
            y,
            yerr=np.vstack([
                y - values["wilson95_lower"].to_numpy(float),
                values["wilson95_upper"].to_numpy(float) - y,
            ]),
            marker=MARKERS[estimator],
            linewidth=1.6,
            capsize=2.5,
            color=COLORS[estimator],
            label=ESTIMATOR_LABELS[estimator],
        )
    axis.axhline(0.95, color="0.25", linestyle="--", linewidth=1, label="Nominal 0.95")
    axis.set_xlabel("Quantile" if dimension == "tau" else "Instrument strength")
    _base_axis(axis, "Conditional coverage")
    axis.legend(frameon=False, fontsize=8)
    return _save_figure(figure, family)


def _metric_lines(table: pd.DataFrame, metric: str, ylabel: str, family: str) -> list[Path]:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    for estimator in ESTIMATOR_ORDER:
        values = table[table["estimator"].eq(estimator)].sort_values("pi")
        axis.plot(
            values["pi"], values[metric], marker=MARKERS[estimator], linewidth=1.6,
            color=COLORS[estimator], label=ESTIMATOR_LABELS[estimator],
        )
    axis.set_xlabel("Instrument strength")
    _base_axis(axis, ylabel)
    axis.legend(frameon=False, fontsize=8)
    return _save_figure(figure, family)


def _paired_figure(table: pd.DataFrame) -> list[Path]:
    figure, axes = plt.subplots(3, 1, figsize=THREE_PANEL_SIZE)
    labels = [
        f"{ESTIMATOR_LABELS[a]}\n- {ESTIMATOR_LABELS[b]}"
        for a, b in zip(table["estimator_a"], table["estimator_b"], strict=True)
    ]
    x = np.arange(len(table))
    specs = (
        ("coverage", "Paired coverage difference"),
        ("squared_error", "Paired squared-error difference"),
        ("cr_length", "Paired CR-length difference"),
    )
    for axis, (prefix, ylabel) in zip(axes, specs, strict=True):
        means = table[f"paired_{prefix}_difference"].to_numpy(float)
        axis.errorbar(
            x,
            means,
            yerr=np.vstack([
                means - table[f"paired_{prefix}_ci95_lower"].to_numpy(float),
                table[f"paired_{prefix}_ci95_upper"].to_numpy(float) - means,
            ]),
            fmt="o",
            color="#4c4c4c",
            capsize=4,
        )
        axis.axhline(0, color="0.2", linestyle="--", linewidth=1)
        axis.set_xticks(x, labels, fontsize=8)
        _base_axis(axis, ylabel)
    figure.suptitle(f"Difference orientation: {PAIR_ORIENTATION}", fontsize=10)
    return _save_figure(figure, "figure_06_paired_estimator_differences")


def _weak_heatmap(uncertainty: pd.DataFrame) -> list[Path]:
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 7.6), sharex=True, sharey=True)
    for axis, estimator in zip(axes, ESTIMATOR_ORDER, strict=True):
        values = uncertainty[uncertainty["estimator"].eq(estimator)].copy()
        values["row"] = (
            values["dgp"].astype(str)
            + " | n=" + values["n"].astype(str)
            + " | p=" + values["p"].astype(str)
            + " | tau=" + values["tau"].map(lambda value: f"{value:.2f}")
        )
        pivot = values.pivot(index="row", columns="pi", values="coverage_gap").sort_index()
        image = axis.imshow(
            pivot.to_numpy(float).T,
            aspect="auto",
            interpolation="none",
            cmap="Greys",
            vmin=-0.12,
            vmax=0.03,
        )
        axis.set_yticks(range(len(pivot.columns)), [f"pi={value:g}" for value in pivot.columns])
        axis.set_title(ESTIMATOR_LABELS[estimator], loc="left", fontsize=9)
    tick_positions = np.linspace(0, len(pivot.index) - 1, 6, dtype=int)
    axes[-1].set_xticks(tick_positions, [pivot.index[index] for index in tick_positions], rotation=25, ha="right", fontsize=7)
    figure.colorbar(image, ax=axes, label="Coverage gap from 0.95", shrink=0.8)
    figure.subplots_adjust(left=0.09, right=0.88, bottom=0.20, hspace=0.28)
    return _save_figure(figure, "figure_07_weak_scenario_structure", tight=False)


def _diagnostic_figure(table: pd.DataFrame) -> list[Path]:
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    x = np.arange(len(table))
    labels = [ESTIMATOR_LABELS[value] for value in table["estimator"]]
    warning_values = table["warning_row_frequency"].to_numpy(float)
    available = np.isfinite(warning_values)
    axes[0].bar(x[available], warning_values[available], color=[COLORS[value] for value in table.loc[available, "estimator"]])
    for position in x[~available]:
        axes[0].text(position, 0.02, "Unavailable", rotation=90, ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x, labels, rotation=15)
    _base_axis(axes[0], "Warning-row frequency")
    exception_frequency = table["validated_exception_count"] / 72000
    axes[1].bar(x, exception_frequency, color=[COLORS[value] for value in table["estimator"]])
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].text(2, exception_frequency.iloc[2], "legacy geometry\nmissingness", ha="center", va="bottom", fontsize=7)
    _base_axis(axes[1], "Validated exception frequency")
    return _save_figure(figure, "figure_08_warning_exception_diagnostics")


def _build_findings(
    tables: dict[str, pd.DataFrame], hashes: dict[str, str]
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    design = tables["table_01_simulation_design"].iloc[0]
    findings.append(finding(
        "simulation_design", "design",
        {
            key: design[key]
            for key in (
                "replications_per_design_cell", "design_cells", "rows_per_estimator",
                "estimator_count", "total_estimator_rows",
            )
        },
        "results/validation/r500_audit/structural_validation.json",
        hashes["results/validation/r500_audit/structural_validation.json"], {},
        design["total_estimator_rows"], "descriptive",
        "The design contains 144 cells and 500 replications per cell.",
        "The design proves external validity.",
    ))
    overall = tables["table_02_overall_estimator_performance"]
    for row in overall.itertuples(index=False):
        findings.append(finding(
            f"overall_{row.estimator}", "overall_performance",
            {
                "coverage": row.conditional_coverage,
                "wilson_lower": row.overall_wilson95_lower,
                "wilson_upper": row.overall_wilson95_upper,
                "coverage_gap": row.coverage_gap,
                "rmse": row.rmse,
                "mean_cr_length": row.mean_cr_length,
                "classification": row.classification,
                "diagnostic_confidence": row.diagnostic_confidence,
            },
            "results/validation/r500_phase2/estimator_classification.csv",
            hashes["results/validation/r500_phase2/estimator_classification.csv"],
            {"estimator": row.estimator}, row.coverage_denominator, "descriptive",
            f"{ESTIMATOR_LABELS[row.estimator]} had conditional coverage {row.conditional_coverage:.4f}.",
            "The estimator is universally superior.",
        ))
        findings.append(finding(
            f"bias_{row.estimator}", "overall_bias", {"bias": row.bias},
            "results/validation/r500_audit/estimator_summary.csv",
            hashes["results/validation/r500_audit/estimator_summary.csv"],
            {"estimator": row.estimator}, row.coverage_denominator, "descriptive",
            f"{ESTIMATOR_LABELS[row.estimator]} had mean bias {row.bias:.4f}.",
            "The observed bias guarantees unbiasedness in other designs.",
        ))
    tradeoffs = tables["table_03_estimator_tradeoffs"]
    for row in tradeoffs.itertuples(index=False):
        findings.append(finding(
            f"paired_{row.estimator_a}_vs_{row.estimator_b}", "paired_comparison",
            {
                "coverage_difference": row.paired_coverage_difference,
                "coverage_ci_lower": row.paired_coverage_ci95_lower,
                "coverage_ci_upper": row.paired_coverage_ci95_upper,
                "squared_error_difference": row.paired_squared_error_difference,
                "cr_length_difference": row.paired_cr_length_difference,
                "orientation": PAIR_ORIENTATION,
            },
            "results/validation/r500_phase2/paired_estimator_summary.csv",
            hashes["results/validation/r500_phase2/paired_estimator_summary.csv"],
            {"estimator_a": row.estimator_a, "estimator_b": row.estimator_b},
            row.paired_coverage_denominator, "statistically_supported",
            f"{ESTIMATOR_LABELS[row.estimator_a]} had a paired coverage difference of {row.paired_coverage_difference:+.4f} relative to {ESTIMATOR_LABELS[row.estimator_b]}.",
            "The paired difference identifies a causal mechanism.",
        ))
    for dimension, family in (("tau", "table_04_performance_by_quantile"), ("pi", "table_05_performance_by_instrument_strength")):
        table = tables[family]
        for estimator in ESTIMATOR_ORDER:
            group = table[table["estimator"].eq(estimator)]
            weakest = group.loc[group["conditional_coverage"].idxmin()]
            findings.append(finding(
                f"{dimension}_pattern_{estimator}", f"performance_by_{dimension}",
                {
                    dimension: weakest[dimension],
                    "lowest_coverage": weakest["conditional_coverage"],
                    "rmse": weakest["rmse"],
                    "mean_cr_length": weakest["mean_cr_length"],
                },
                "results/validation/r500_audit/scenario_summary.csv",
                hashes["results/validation/r500_audit/scenario_summary.csv"],
                {"estimator": estimator, "aggregation": dimension},
                int(weakest["coverage_denominator"]), "descriptive",
                f"{ESTIMATOR_LABELS[estimator]}'s lowest aggregated coverage by {dimension} occurred at {weakest[dimension]}.",
                "The design dimension caused the coverage pattern.",
            ))
    for row in tables["table_06_weakest_scenarios"].itertuples(index=False):
        findings.append(finding(
            f"weak_{row.estimator}_{int(row.rank)}", "weak_scenario",
            {
                "rank": row.rank, "dgp": row.dgp, "n": row.n, "p": row.p,
                "pi": row.pi, "tau": row.tau, "coverage": row.conditional_coverage,
                "coverage_gap": row.coverage_gap, "rmse": row.rmse,
                "mean_cr_length": row.mean_cr_length,
            },
            "results/validation/r500_phase2/tables/worst_coverage_scenarios.csv",
            hashes["results/validation/r500_phase2/tables/worst_coverage_scenarios.csv"],
            {key: getattr(row, key) for key in ("estimator", "dgp", "n", "p", "pi", "tau")},
            row.coverage_denominator, "descriptive",
            f"This was rank {int(row.rank)} among the weakest coverage scenarios for {ESTIMATOR_LABELS[row.estimator]}.",
            "The scenario disproves asymptotic validity.",
        ))
        findings.append(finding(
            f"weak_interval_{row.estimator}_{int(row.rank)}", "weak_scenario_uncertainty",
            {"wilson_lower": row.wilson95_lower, "wilson_upper": row.wilson95_upper},
            "results/validation/r500_phase2/coverage_uncertainty.csv",
            hashes["results/validation/r500_phase2/coverage_uncertainty.csv"],
            {key: getattr(row, key) for key in ("estimator", "dgp", "n", "p", "pi", "tau")},
            row.coverage_denominator, "statistically_supported",
            f"The Wilson interval for this ranked scenario was [{row.wilson95_lower:.4f}, {row.wilson95_upper:.4f}].",
            "The interval proves failure of the estimator in all samples.",
        ))
    diagnostic = tables["table_07_warning_exception_diagnostics"]
    for row in diagnostic.itertuples(index=False):
        source = (
            "results/validation/r500_phase2/warning_summary.csv"
            if row.estimator != "dml"
            else "results/validation/r500_phase2/exception_rows.csv"
        )
        findings.append(finding(
            f"diagnostic_{row.estimator}", "warning_exception",
            {
                "warning_frequency": row.warning_row_frequency,
                "coverage_with_warnings": row.coverage_with_warnings,
                "coverage_without_warnings": row.coverage_without_warnings,
                "exception_count": row.validated_exception_count,
                "empty_count": row.empty_confidence_regions,
                "unresolved_count": row.unresolved_rows,
                "missing_legacy_geometry": row.missing_legacy_geometry,
            }, source, hashes[source], {"estimator": row.estimator}, 72000,
            "limitation" if row.estimator == "dml" else "descriptive",
            (
                "DML warning and numerical-resolution diagnostics are unavailable."
                if row.estimator == "dml"
                else f"Warnings were associated with the reported outcomes for {ESTIMATOR_LABELS[row.estimator]}."
            ),
            "Warnings caused the observed outcomes.",
            "partially unavailable" if row.estimator == "dml" else "available",
        ))
    findings.append(finding(
        "historical_multiplier", "calibration",
        {"historical_post_selection_multiplier": 1.0, "future_value_not_analyzed": 1.8},
        "results/validation/r500_phase2/phase2_report.md",
        hashes["results/validation/r500_phase2/phase2_report.md"],
        {"estimator": "post_selection", "selection_multiplier_values": "1:72000"},
        72000, "limitation",
        "Historical Post-selection results use multiplier 1.0; no conclusion about 1.8 follows.",
        "Historical results use multiplier 1.8.",
    ))
    validate_findings(findings)
    return findings


def _build_report(findings: list[dict[str, object]]) -> str:
    by_id = {item["id"]: item for item in findings}
    design = by_id["simulation_design"]["values"]
    lines = ["# Technical empirical-results report", ""]
    sections: list[tuple[str, list[str]]] = [
        ("1. Simulation design", [sourced_sentence(
            f"The validated design contains {design['design_cells']} cells, {design['replications_per_design_cell']} replications per cell, {design['rows_per_estimator']:,} rows per estimator, and {design['total_estimator_rows']:,} estimator rows overall.",
            ["simulation_design"],
        )]),
        ("2. Validation basis", [sourced_sentence(
            "Phase 1 supplies structural provenance and Phase 2 supplies scientific diagnostics; this package introduces no new inferential procedure.",
            ["simulation_design"],
        )]),
        ("3. Overall estimator performance", [
            sourced_sentence(
                "; ".join(
                    f"{ESTIMATOR_LABELS[name]} coverage {by_id[f'overall_{name}']['values']['coverage']:.4f}, RMSE {by_id[f'overall_{name}']['values']['rmse']:.4f}, mean CR length {by_id[f'overall_{name}']['values']['mean_cr_length']:.4f}"
                    for name in ESTIMATOR_ORDER
                ) + ".",
                [f"overall_{name}" for name in ESTIMATOR_ORDER],
            )
        ]),
        ("4. Coverage", [sourced_sentence(
            "DML-style IVQR was closest to nominal coverage; Oracle and Post-selection coverage were lower.",
            [f"overall_{name}" for name in ESTIMATOR_ORDER],
        )]),
        ("5. Bias and RMSE", [sourced_sentence(
            f"Bias was {by_id['bias_oracle']['values']['bias']:.4f} for Oracle, {by_id['bias_post_selection']['values']['bias']:.4f} for Post-selection, and {by_id['bias_dml']['values']['bias']:.4f} for DML; corresponding RMSE values were {by_id['overall_oracle']['values']['rmse']:.4f}, {by_id['overall_post_selection']['values']['rmse']:.4f}, and {by_id['overall_dml']['values']['rmse']:.4f}.",
            [*[f"bias_{name}" for name in ESTIMATOR_ORDER], *[f"overall_{name}" for name in ESTIMATOR_ORDER]],
        )]),
        ("6. Confidence-region length", [sourced_sentence(
            f"Mean CR length was {by_id['overall_oracle']['values']['mean_cr_length']:.4f} for Oracle, {by_id['overall_post_selection']['values']['mean_cr_length']:.4f} for Post-selection, and {by_id['overall_dml']['values']['mean_cr_length']:.4f} for DML.",
            [f"overall_{name}" for name in ESTIMATOR_ORDER],
        )]),
        ("7. Paired estimator comparisons", [
            sourced_sentence(item["permitted_wording"], [item["id"]])
            for item in findings if item["topic"] == "paired_comparison"
        ]),
        ("8. Quantile patterns", [
            sourced_sentence(item["permitted_wording"], [item["id"]])
            for item in findings if item["topic"] == "performance_by_tau"
        ]),
        ("9. Instrument-strength patterns", [
            sourced_sentence(item["permitted_wording"], [item["id"]])
            for item in findings if item["topic"] == "performance_by_pi"
        ]),
        ("10. Weak scenarios", [sourced_sentence(
            "The five weakest coverage scenarios per estimator are selected mechanically from the Phase 2 ranking and reported with their validated uncertainty intervals.",
            [item["id"] for item in findings if item["topic"] == "weak_scenario"],
        )]),
        ("11. Warning diagnostics", [
            sourced_sentence(by_id[f"diagnostic_{name}"]["permitted_wording"], [f"diagnostic_{name}"])
            for name in ("oracle", "post_selection")
        ]),
        ("12. Empty and unresolved regions", [sourced_sentence(
            f"Validated exception counts were {by_id['diagnostic_oracle']['values']['exception_count']} for Oracle, {by_id['diagnostic_post_selection']['values']['exception_count']} for Post-selection, and {by_id['diagnostic_dml']['values']['exception_count']} DML legacy missing-geometry rows.",
            [f"diagnostic_{name}" for name in ESTIMATOR_ORDER],
        )]),
        ("13. Estimator classifications", [sourced_sentence(
            "; ".join(
                f"{ESTIMATOR_LABELS[name]}: {by_id[f'overall_{name}']['values']['classification']}"
                for name in ESTIMATOR_ORDER
            ) + ".",
            [f"overall_{name}" for name in ESTIMATOR_ORDER],
        )]),
        ("14. Limitations", [
            sourced_sentence(by_id["historical_multiplier"]["permitted_wording"], ["historical_multiplier"]),
            sourced_sentence(
                "The historical DML schema has 15 columns; unavailable warning, unresolved, and rich geometry diagnostics are not zeros.",
                ["diagnostic_dml"],
            ),
        ]),
        ("15. Defensible conclusions", [sourced_sentence(
            "DML's higher coverage coincided with longer regions and higher RMSE; Oracle offered the lowest RMSE; Post-selection displayed the largest undercoverage. Coverage, error, and interval length must be interpreted jointly.",
            [f"overall_{name}" for name in ESTIMATOR_ORDER],
        )]),
    ]
    for heading, paragraphs in sections:
        lines.extend([f"## {heading}", ""])
        for paragraph in paragraphs:
            lines.extend([paragraph, ""])
    report = "\n".join(lines)
    validate_report_provenance(report, set(by_id))
    return report


def _build_consistency_checks(
    tables: dict[str, pd.DataFrame], sources: dict[str, object]
) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    design = tables["table_01_simulation_design"].iloc[0]
    structural = sources["structural_validation"]["structure"]["oracle"]
    source_design = "results/validation/r500_audit/structural_validation.json"
    for metric, expected in (
        ("design_cells", structural["design_cells"]),
        ("replications_per_design_cell", structural["unique_replications"]),
        ("rows_per_estimator", structural["rows"]),
        ("total_estimator_rows", structural["rows"] * 3),
    ):
        checks.append(consistency_row(f"design_{metric}", metric, source_design, expected, design[metric]))
    summary = sources["estimator_summary"].set_index("estimator")
    classification = sources["estimator_classification"].set_index("estimator")
    overall = tables["table_02_overall_estimator_performance"].set_index("estimator")
    for estimator in ESTIMATOR_ORDER:
        for metric in ("conditional_coverage", "coverage_denominator", "bias", "rmse", "mean_cr_length", "iteration_warning_rate"):
            checks.append(consistency_row(
                f"overall_{estimator}_{metric}", metric,
                "results/validation/r500_audit/estimator_summary.csv",
                summary.loc[estimator, metric], overall.loc[estimator, metric], 1e-12,
            ))
        for metric in ("overall_wilson95_lower", "overall_wilson95_upper", "coverage_gap", "classification", "diagnostic_confidence"):
            checks.append(consistency_row(
                f"classification_{estimator}_{metric}", metric,
                "results/validation/r500_phase2/estimator_classification.csv",
                classification.loc[estimator, metric], overall.loc[estimator, metric], 1e-12,
            ))
    paired_source = sources["paired_estimator_summary"]
    tradeoffs = tables["table_03_estimator_tradeoffs"]
    for row in tradeoffs.itertuples(index=False):
        group = paired_source[
            paired_source["estimator_a"].eq(row.estimator_a)
            & paired_source["estimator_b"].eq(row.estimator_b)
        ].set_index("metric")
        for metric, prefix in (("coverage", "coverage"), ("squared_error", "squared_error"), ("cr_length", "cr_length")):
            for source_column, output_suffix in (
                ("mean_paired_difference", "difference"),
                ("paired_ci95_lower", "ci95_lower"),
                ("paired_ci95_upper", "ci95_upper"),
                ("valid_paired_denominator", "denominator"),
            ):
                checks.append(consistency_row(
                    f"paired_{row.estimator_a}_{row.estimator_b}_{metric}_{output_suffix}",
                    f"{metric}_{output_suffix}",
                    "results/validation/r500_phase2/paired_estimator_summary.csv",
                    group.loc[metric, source_column],
                    getattr(row, f"paired_{prefix}_{output_suffix}"), 1e-12,
                ))
    diagnostic = tables["table_07_warning_exception_diagnostics"].set_index("estimator")
    warning = sources["warning_summary"]
    exceptions = sources["exception_rows"]
    for estimator in ("oracle", "post_selection"):
        expected = warning[
            warning["estimator"].eq(estimator)
            & warning["warning_category"].eq("iteration_warning")
        ].iloc[0]
        checks.append(consistency_row(
            f"warning_{estimator}", "warning_frequency",
            "results/validation/r500_phase2/warning_summary.csv",
            expected.warning_frequency, diagnostic.loc[estimator, "warning_row_frequency"], 1e-12,
        ))
        checks.append(consistency_row(
            f"warning_events_{estimator}", "warning_event_count",
            "results/validation/r500_phase2/warning_summary.csv",
            expected.warning_event_count, diagnostic.loc[estimator, "warning_event_count"],
        ))
    for estimator in ESTIMATOR_ORDER:
        count = int(exceptions["estimator_name"].eq(estimator).sum())
        checks.append(consistency_row(
            f"exceptions_{estimator}", "validated_exception_count",
            "results/validation/r500_phase2/exception_rows.csv",
            count, diagnostic.loc[estimator, "validated_exception_count"],
        ))
    checks.append(consistency_row(
        "historical_multiplier", "selection_lasso_multiplier",
        "results/validation/r500_audit/estimator_summary.csv",
        "1:72000", summary.loc["post_selection", "selection_multiplier_values"],
    ))
    for metric in ("warning_row_frequency", "empty_confidence_regions", "unresolved_rows"):
        checks.append(consistency_row(
            f"dml_unavailable_{metric}", metric,
            "results/validation/r500_phase2/exception_rows.csv",
            np.nan, diagnostic.loc["dml", metric],
        ))
    weakest = tables["table_06_weakest_scenarios"]
    worst_source = sources["worst_coverage_scenarios"]
    uncertainty_source = sources["coverage_uncertainty"].set_index(
        ["estimator", "dgp", "n", "p", "pi", "tau"]
    )
    for estimator in ESTIMATOR_ORDER:
        expected = worst_source[worst_source["estimator"].eq(estimator)].head(5)
        actual = weakest[weakest["estimator"].eq(estimator)]
        for expected_row, actual_row in zip(expected.itertuples(index=False), actual.itertuples(index=False), strict=True):
            rank = int(actual_row.rank)
            for metric in (
                "conditional_coverage", "coverage_denominator", "coverage_gap", "rmse",
                "mean_cr_length",
            ):
                checks.append(consistency_row(
                    f"weak_{estimator}_{rank}_{metric}", metric,
                    "results/validation/r500_phase2/tables/worst_coverage_scenarios.csv",
                    getattr(expected_row, metric), getattr(actual_row, metric), 1e-12,
                ))
            key = (
                estimator, actual_row.dgp, actual_row.n, actual_row.p, actual_row.pi,
                actual_row.tau,
            )
            for metric in ("wilson95_lower", "wilson95_upper"):
                checks.append(consistency_row(
                    f"weak_{estimator}_{rank}_{metric}", metric,
                    "results/validation/r500_phase2/coverage_uncertainty.csv",
                    uncertainty_source.loc[key, metric], getattr(actual_row, metric), 1e-12,
                ))
    expected_order = pd.concat(
        [weakest[weakest["estimator"].eq(estimator)] for estimator in ESTIMATOR_ORDER],
        ignore_index=True,
    )
    checks.append(consistency_row(
        "scenario_order", "scenario_order",
        "results/validation/r500_phase2/tables/worst_coverage_scenarios.csv",
        True,
        weakest.reset_index(drop=True).equals(expected_order.reset_index(drop=True)),
    ))
    result = pd.DataFrame(checks)
    require_consistency(result)
    return result


def _manifest_entry(path: Path, family: str, output_type: str, table: pd.DataFrame | None = None) -> dict[str, object]:
    sources = FAMILY_SOURCES.get(family, list(SOURCE_CONTRACTS))
    return {
        "output_filename": path.relative_to(OUTPUT_DIR).as_posix(),
        "output_family": family,
        "output_type": output_type,
        "authoritative_sources": sources,
        "source_sha256": {source: SOURCE_CONTRACTS[source]["sha256"] for source in sources},
        "source_columns": FAMILY_SOURCE_COLUMNS.get(
            family, {source: ["finding-level provenance"] for source in sources}
        ),
        "output_columns_or_figure_series": [] if table is None else list(table.columns),
        "filters": "family-specific deterministic selection recorded in source inventory",
        "aggregation_level": "thesis presentation",
        "metric_denominator": "actual validated denominator; N/A where diagnostic unavailable",
        "estimator_ordering": list(ESTIMATOR_ORDER),
        "scenario_ordering": ["dgp", "n", "p", "pi", "tau"],
        "paired_difference_orientation": PAIR_ORIENTATION,
        "missing_value_policy": {"csv": "NA", "json": None, "latex": "N/A", "figure": "Unavailable label or omitted mark"},
        "decimal_precision": 4,
        "float_formatting_rule": FLOAT_FORMAT,
        "latex_formatting_rule": "Explicit escaping; four decimals; N/A for missing values",
        "figure_dimensions": None if output_type not in {"pdf", "png"} else "fixed by family; no secondary axes",
        "generation_timestamp_policy": "omitted",
        "generated_output_sha256": sha256_file(path),
    }


def main() -> int:
    sources = load_sources(PROJECT_ROOT)
    source_hashes = {relative: str(contract["sha256"]) for relative, contract in SOURCE_CONTRACTS.items()}
    structural = sources["structural_validation"]
    estimator_summary = sources["estimator_summary"]
    scenarios = sources["scenario_summary"]
    uncertainty = sources["coverage_uncertainty"]
    classification = sources["estimator_classification"]
    paired = sources["paired_estimator_summary"]
    worst = sources["worst_coverage_scenarios"]
    warning_summary = sources["warning_summary"]
    warning_scenarios = sources["warning_scenario_summary"]
    exception_rows = sources["exception_rows"]
    exception_scenarios = sources["exception_scenario_summary"]

    tables = {
        "table_01_simulation_design": build_design_table(structural, scenarios),
        "table_02_overall_estimator_performance": build_overall_table(estimator_summary, classification),
        "table_03_estimator_tradeoffs": build_tradeoff_table(paired),
        "table_04_performance_by_quantile": aggregate_performance(scenarios, uncertainty, "tau"),
        "table_05_performance_by_instrument_strength": aggregate_performance(scenarios, uncertainty, "pi"),
        "table_06_weakest_scenarios": build_weakest_table(
            worst, uncertainty, scenarios, warning_scenarios, exception_scenarios
        ),
        "table_07_warning_exception_diagnostics": build_warning_exception_table(
            warning_summary, exception_rows
        ),
    }
    if tuple(tables) != TABLE_FAMILIES:
        raise AssertionError("Phase 3 must define exactly seven ordered table families")

    checks = _build_consistency_checks(tables, sources)
    findings = _build_findings(tables, source_hashes)
    report = _build_report(findings)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[tuple[Path, str, str, pd.DataFrame | None]] = []
    for family, table in tables.items():
        for path in _write_table(table, family):
            generated.append((path, family, path.suffix.lstrip("."), table))

    figure_paths = {
        "figure_01_overall_estimator_tradeoff": _overall_figure(tables["table_02_overall_estimator_performance"]),
        "figure_02_coverage_by_quantile": _coverage_lines(tables["table_04_performance_by_quantile"], "tau", "figure_02_coverage_by_quantile"),
        "figure_03_coverage_by_instrument_strength": _coverage_lines(tables["table_05_performance_by_instrument_strength"], "pi", "figure_03_coverage_by_instrument_strength"),
        "figure_04_rmse_by_instrument_strength": _metric_lines(tables["table_05_performance_by_instrument_strength"], "rmse", "RMSE", "figure_04_rmse_by_instrument_strength"),
        "figure_05_cr_length_by_instrument_strength": _metric_lines(tables["table_05_performance_by_instrument_strength"], "mean_cr_length", "Mean CR length", "figure_05_cr_length_by_instrument_strength"),
        "figure_06_paired_estimator_differences": _paired_figure(tables["table_03_estimator_tradeoffs"]),
        "figure_07_weak_scenario_structure": _weak_heatmap(uncertainty),
        "figure_08_warning_exception_diagnostics": _diagnostic_figure(tables["table_07_warning_exception_diagnostics"]),
    }
    if tuple(figure_paths) != FIGURE_FAMILIES:
        raise AssertionError("Phase 3 must define exactly eight ordered figure families")
    figure_source_tables = {
        "figure_01_overall_estimator_tradeoff": tables["table_02_overall_estimator_performance"],
        "figure_02_coverage_by_quantile": tables["table_04_performance_by_quantile"],
        "figure_03_coverage_by_instrument_strength": tables["table_05_performance_by_instrument_strength"],
        "figure_04_rmse_by_instrument_strength": tables["table_05_performance_by_instrument_strength"][["estimator", "pi", "rmse"]],
        "figure_05_cr_length_by_instrument_strength": tables["table_05_performance_by_instrument_strength"][["estimator", "pi", "mean_cr_length"]],
        "figure_06_paired_estimator_differences": tables["table_03_estimator_tradeoffs"],
        "figure_07_weak_scenario_structure": uncertainty[["estimator", "dgp", "n", "p", "pi", "tau", "coverage_gap"]],
        "figure_08_warning_exception_diagnostics": tables["table_07_warning_exception_diagnostics"],
    }
    for family, paths in figure_paths.items():
        generated.extend(
            (path, family, path.suffix.lstrip("."), figure_source_tables[family])
            for path in paths
        )

    findings_path = OUTPUT_DIR / "thesis_findings.json"
    _write_json({"findings": findings}, findings_path)
    generated.append((findings_path, "thesis_findings", "json", None))
    report_path = OUTPUT_DIR / "empirical_results_report.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    generated.append((report_path, "empirical_results_report", "markdown", None))
    checks_path = OUTPUT_DIR / "consistency_checks.csv"
    _write_csv(checks, checks_path)
    generated.append((checks_path, "consistency_audit", "csv", checks))
    consistency_report = (
        "# Phase 3 consistency report\n\n"
        f"All {len(checks)} checks passed. No substantive discrepancy was found.\n"
    )
    consistency_report_path = OUTPUT_DIR / "consistency_report.md"
    consistency_report_path.write_text(
        consistency_report, encoding="utf-8", newline="\n"
    )
    generated.append((consistency_report_path, "consistency_audit", "markdown", None))

    entries = [
        _manifest_entry(path, family, output_type, table)
        for path, family, output_type, table in generated
    ]
    entries.append({
        "output_filename": "thesis_output_manifest.json",
        "output_family": "manifest",
        "output_type": "json",
        "authoritative_sources": list(SOURCE_CONTRACTS),
        "source_sha256": source_hashes,
        "source_columns": [],
        "filters": "all generated outputs",
        "aggregation_level": "provenance index",
        "metric_denominator": "not applicable",
        "estimator_ordering": list(ESTIMATOR_ORDER),
        "scenario_ordering": ["dgp", "n", "p", "pi", "tau"],
        "paired_difference_orientation": PAIR_ORIENTATION,
        "missing_value_policy": {"json": None},
        "decimal_precision": 4,
        "float_formatting_rule": FLOAT_FORMAT,
        "latex_formatting_rule": "explicit escaping and N/A",
        "figure_dimensions": None,
        "generation_timestamp_policy": "omitted",
        "generated_output_sha256": None,
        "self_hash_policy": "null because embedding a file's own SHA-256 is self-referential",
    })
    manifest = {
        "schema_version": 1,
        "table_family_count": len(TABLE_FAMILIES),
        "figure_family_count": len(FIGURE_FAMILIES),
        "generation_timestamp_policy": "omitted",
        "historical_post_selection_multiplier": 1.0,
        "future_multiplier_not_analyzed": 1.8,
        "dml_diagnostic_availability": (
            "Historical 15-column DML data lack warning, resolution, component, and rich geometry diagnostics."
        ),
        "outputs": entries,
    }
    _write_json(manifest, OUTPUT_DIR / "thesis_output_manifest.json")
    validate_historical_multiplier_outputs(OUTPUT_DIR)
    print(f"Built {len(TABLE_FAMILIES)} table families and {len(FIGURE_FAMILIES)} figure families")
    print(f"All {len(checks)} consistency checks passed")
    print(f"Wrote thesis package to {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
