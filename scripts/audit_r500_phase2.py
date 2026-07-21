"""Generate Phase 2 scientific diagnostics for immutable historical R=500 results."""

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

from analysis.data import (  # noqa: E402
    RAW_MANIFEST_PATH,
    RAW_RESULT_FILES,
    load_all_results,
    verify_raw_manifest,
)
from analysis.r500_audit import (  # noqa: E402
    artifact_metadata,
    harmonize_frames,
    summarize_estimator,
    summarize_scenarios,
    validate_alignment,
    validate_cr_components_frame,
    validate_result_values,
    validate_structure,
)
from analysis.r500_phase2 import (  # noqa: E402
    classify_estimators,
    coverage_uncertainty,
    exception_diagnostics,
    paired_comparisons,
    warning_summaries,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "validation" / "r500_phase2"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
ESTIMATOR_LABELS = {
    "oracle": "Oracle IVQR",
    "post_selection": "Post-selection IVQR",
    "dml": "DML-style IVQR",
}
COLORS = {"oracle": "#1f77b4", "post_selection": "#d95f02", "dml": "#2a9d8f"}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")


def _write_json(payload: object, path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_table(frame: pd.DataFrame, name: str) -> None:
    _write_csv(frame, TABLE_DIR / f"{name}.csv")
    def latex_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            text = f"{float(value):.4f}"
        else:
            text = str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        return "".join(replacements.get(character, character) for character in text)

    columns = [latex_value(column) for column in frame.columns]
    lines = [
        rf"\begin{{tabular}}{{{'l' * len(columns)}}}",
        r"\hline",
        " & ".join(columns) + r" \\",
        r"\hline",
    ]
    lines.extend(
        " & ".join(latex_value(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    latex = "\n".join(lines)
    (TABLE_DIR / f"{name}.tex").write_text(latex, encoding="utf-8", newline="\n")


def _save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / f"{name}.pdf",
        bbox_inches="tight",
        metadata={"Creator": "R500 Phase 2 audit", "CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        FIGURE_DIR / f"{name}.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "R500 Phase 2 audit"},
    )
    plt.close(figure)


def _base_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="0.88", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def _coverage_figure(coverage: pd.DataFrame) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 8.5), sharex=True, sharey=True)
    for axis, estimator in zip(axes, ESTIMATOR_LABELS, strict=True):
        values = coverage[coverage["estimator"].eq(estimator)].reset_index(drop=True)
        x = np.arange(1, len(values) + 1)
        y = values["coverage"].to_numpy(float)
        lower = y - values["wilson95_lower"].to_numpy(float)
        upper = values["wilson95_upper"].to_numpy(float) - y
        axis.errorbar(
            x, y, yerr=np.vstack([lower, upper]), fmt=".", markersize=2.5,
            linewidth=0.45, elinewidth=0.4, color=COLORS[estimator], alpha=0.8,
        )
        axis.axhline(0.95, color="0.3", linestyle="--", linewidth=0.9)
        axis.set_title(ESTIMATOR_LABELS[estimator], loc="left", fontsize=10)
        _base_axis(axis, "Coverage")
    axes[-1].set_xlabel("Scenario index in natural-key order")
    _save_figure(figure, "scenario_coverage_with_wilson_intervals")


def _strength_figure(scenarios: pd.DataFrame, metric: str, ylabel: str, name: str) -> None:
    summary = scenarios.groupby(["estimator", "pi"], sort=True)[metric].mean().reset_index()
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    for estimator in ESTIMATOR_LABELS:
        values = summary[summary["estimator"].eq(estimator)]
        axis.plot(
            values["pi"], values[metric], marker="o", linewidth=1.7,
            color=COLORS[estimator], label=ESTIMATOR_LABELS[estimator],
        )
    axis.set_xlabel("Instrument strength (pi)")
    _base_axis(axis, ylabel)
    axis.legend(frameon=False)
    _save_figure(figure, name)


def _paired_coverage_figure(paired_scenarios: pd.DataFrame) -> None:
    values = paired_scenarios[paired_scenarios["metric"].eq("coverage")]
    pairs = values[["estimator_a", "estimator_b"]].drop_duplicates().itertuples(index=False)
    figure, axes = plt.subplots(3, 1, figsize=(9.0, 8.2), sharex=True, sharey=True)
    for axis, pair in zip(axes, pairs, strict=True):
        group = values[
            values["estimator_a"].eq(pair.estimator_a)
            & values["estimator_b"].eq(pair.estimator_b)
        ].reset_index(drop=True)
        x = np.arange(1, len(group) + 1)
        y = group["mean_paired_difference"].to_numpy(float)
        axis.errorbar(
            x, y,
            yerr=np.vstack([
                y - group["paired_ci95_lower"].to_numpy(float),
                group["paired_ci95_upper"].to_numpy(float) - y,
            ]),
            fmt=".", markersize=2.5, linewidth=0.4, elinewidth=0.4, color="#5b3c88",
        )
        axis.axhline(0, color="0.3", linestyle="--", linewidth=0.9)
        axis.set_title(
            f"{ESTIMATOR_LABELS[pair.estimator_a]} minus "
            f"{ESTIMATOR_LABELS[pair.estimator_b]}",
            loc="left", fontsize=10,
        )
        _base_axis(axis, "Paired coverage difference")
    axes[-1].set_xlabel("Scenario index in natural-key order")
    _save_figure(figure, "paired_coverage_differences")


def _warning_figure(warnings: pd.DataFrame) -> None:
    pivot = warnings.pivot(
        index="warning_category", columns="estimator", values="warning_frequency"
    ).fillna(0)
    x = np.arange(len(pivot))
    figure, axis = plt.subplots(figsize=(9.0, 4.6))
    width = 0.36
    for offset, estimator in zip((-width / 2, width / 2), ("oracle", "post_selection"), strict=True):
        axis.bar(
            x + offset, pivot.get(estimator, pd.Series(0, index=pivot.index)),
            width=width, label=ESTIMATOR_LABELS[estimator], color=COLORS[estimator],
        )
    axis.set_xticks(x, [value.replace("_", "\n") for value in pivot.index], fontsize=8)
    _base_axis(axis, "Affected-row frequency")
    axis.legend(frameon=False)
    _save_figure(figure, "warning_frequency_by_category")


def _exception_figure(
    exception_rows: pd.DataFrame, frames: dict[str, pd.DataFrame]
) -> None:
    dimensions = ("n", "p", "pi", "tau")
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    for axis, dimension in zip(axes.flat, dimensions, strict=True):
        for estimator in ESTIMATOR_LABELS:
            numerator = exception_rows[
                exception_rows["estimator_name"].eq(estimator)
            ].groupby(dimension).size()
            denominator = frames[estimator].groupby(dimension).size()
            frequency = numerator.reindex(denominator.index, fill_value=0) / denominator
            axis.plot(
                frequency.index.astype(float), frequency.to_numpy(float), marker="o",
                linewidth=1.5, color=COLORS[estimator], label=ESTIMATOR_LABELS[estimator],
            )
        axis.set_xlabel(dimension)
        _base_axis(axis, "Exception frequency")
    axes[0, 0].legend(frameon=False, fontsize=8)
    _save_figure(figure, "exception_frequency_by_design_dimension")


def _fmt(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "unavailable"
    return f"{float(value):.{digits}f}"


def _build_report(
    metadata: dict[str, object],
    warnings: pd.DataFrame,
    exceptions: pd.DataFrame,
    exception_scenarios: pd.DataFrame,
    paired: pd.DataFrame,
    classifications: pd.DataFrame,
    scenarios: pd.DataFrame,
) -> str:
    iteration = warnings[warnings["warning_category"].eq("iteration_warning")].set_index("estimator")
    pair_coverage = paired[paired["metric"].eq("coverage")]
    pair_lengths = paired[paired["metric"].eq("cr_length")]
    top_exception = exception_scenarios.sort_values(
        ["exception_count", "estimator"], ascending=[False, True], kind="mergesort"
    ).head(3)
    post_worst = scenarios[
        scenarios["estimator"].eq("post_selection")
    ].nsmallest(2, "coverage_gap")
    lines = [
        "# Phase 2 deep scientific diagnostics of historical R=500 results",
        "",
        "## 1. Scope and restrictions",
        "",
        "Observed fact: this analysis reads only the three immutable historical artifacts. "
        "No simulation, source-data rewrite, estimator change, or inference change is part "
        "of Phase 2. Phase 1 definitions and outputs remain intact.",
        "",
        "## 2. Source artifacts and hashes",
        "",
        "; ".join(
            f"{name}: `{values['sha256']}` ({values['size_bytes']:,} bytes)"
            for name, values in metadata.items()
        ) + ".",
        "",
        "## 3. Warning analysis",
        "",
        "Observed fact: no textual warning reasons are stored. The taxonomy is multi-label: "
        "affected-row counts and summed warning-event counts are both reported, and a row may "
        "belong to several categories.",
        "",
        "; ".join(
            f"{estimator} iteration-warning prevalence {_fmt(row.warning_frequency)} "
            f"({int(row.warning_count):,} rows; {int(row.warning_event_count):,} events), "
            f"affected valid coverage {_fmt(row.coverage_affected_valid)} versus "
            f"{_fmt(row.coverage_without_warning_valid)} without, affected RMSE "
            f"{_fmt(row.rmse)} versus {_fmt(row.rmse_without_warning)}, and affected mean "
            f"CR length {_fmt(row.mean_cr_length)} versus "
            f"{_fmt(row.mean_cr_length_without_warning)}"
            for estimator, row in iteration.iterrows()
        ) + ". Statistical association is not a causal warning effect.",
        "",
        "## 4. Empty and unresolved confidence regions",
        "",
        f"Observed fact: {len(exceptions)} exceptional rows were inspected: "
        + ", ".join(
            f"{name}={count}"
            for name, count in exceptions["estimator_name"].value_counts().sort_index().items()
        )
        + ". Oracle/Post-selection empty sets have valid empty components and no reversed "
        "bounds; unresolved rows remain separate from noncoverage; DML missing geometry is "
        "classified as legacy missingness, not numerical non-resolution.",
        "",
        "Largest scenario-level exception cells: "
        + "; ".join(
            f"{row.estimator}/{row.dgp}/n={row.n:g}/p={row.p:g}/pi={row.pi:g}/tau={row.tau:g}: "
            f"{int(row.exception_count)} {row.exception_type}"
            for row in top_exception.itertuples(index=False)
        ) + ".",
        "",
        "## 5. Paired estimator comparisons",
        "",
        "All comparisons use identical design-replication keys and seeds. Differences are "
        "estimator A minus estimator B. Row-level paired uncertainty is reported overall; "
        "scenario files separately aggregate the 500 paired replications per design cell.",
        "",
        "; ".join(
            f"{row.estimator_a} minus {row.estimator_b}: coverage "
            f"{row.mean_paired_difference:+.4f} (95% CI "
            f"[{row.paired_ci95_lower:.4f}, {row.paired_ci95_upper:.4f}], "
            f"m={int(row.valid_paired_denominator):,})"
            for row in pair_coverage.itertuples(index=False)
        ) + ".",
        "",
        "; ".join(
            f"{row.estimator_a} minus {row.estimator_b}: mean CR-length difference "
            f"{row.mean_paired_difference:+.4f}"
            for row in pair_lengths.itertuples(index=False)
        ) + ".",
        "",
        "## 6. Coverage-uncertainty robustness",
        "",
        "Both bounded Wald and Wilson 95% intervals use the actual conditional denominator. "
        "Wilson inclusion of 0.95 is reported scenario by scenario and in formal overall "
        "classification; no excluded row is restored to an expected denominator.",
        "",
        "## 7. Formal classification rules",
        "",
        "The non-compensatory rules are recorded in `classification_rules.json`; no weighted "
        "composite score is used. Results: "
        + "; ".join(
            f"{row.estimator}: **{row.classification}** (diagnostic confidence "
            f"{row.diagnostic_confidence}; reasons {row.machine_readable_reasons})"
            for row in classifications.itertuples(index=False)
        ) + ".",
        "",
        "## 8. Estimator-specific findings",
        "",
        "Oracle remains the infeasible precision benchmark but its overall Wilson interval "
        "does not include 0.95. Post-selection has the largest negative overall coverage gap. "
        "DML is closest to nominal coverage but has limited diagnostic confidence because its "
        "historical schema lacks warning, component, and resolution fields.",
        "",
        "## 9. Cross-estimator conclusions",
        "",
        "Statistical evidence: paired comparisons preserve the Phase 1 trade-off. DML covers "
        "more often than Oracle/Post-selection while using longer regions; this is an empirical "
        "coverage-length trade-off, not proof that interval length causes coverage. Oracle and "
        "Post-selection have closer error and length performance, but Post-selection covers less.",
        "",
        "## 10. Limitations",
        "",
        "Warning messages and failure reasons are absent; DML diagnostic metadata is absent; "
        "grid endpoints are not stored row-wise; and paired row-level intervals do not model "
        "cross-scenario heterogeneity. Associations cannot establish causal mechanisms.",
        "",
        "## 11. Thesis implications",
        "",
        "The two worst Post-selection scenarios are "
        + "; ".join(
            f"{row.dgp}/n={row.n:g}/p={row.p:g}/pi={row.pi:g}/tau={row.tau:g} "
            f"(coverage {_fmt(row.conditional_coverage)})"
            for row in post_worst.itertuples(index=False)
        )
        + ". Stored evidence identifies concentration at the upper quantile and high dimension, "
        "with scenario warning rates and mean selected-control counts of "
        + "; ".join(
            f"{_fmt(row.iteration_warning_rate)}/{_fmt(row.mean_selected_controls)}"
            for row in post_worst.itertuples(index=False)
        )
        + ", respectively. One warning rate is above and one below the overall Post-selection "
        "warning prevalence, while selected-control complexity also differs. The stored fields "
        "therefore do not identify why the misses occurred. They must be presented as finite-"
        "sample weaknesses, not assigned a causal selection or warning explanation.",
        "",
        "## 12. Exact reproduction command",
        "",
        "```powershell",
        "pixi run audit_r500_phase2",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    verify_raw_manifest()
    load_all_results(expected_replications=500)
    frames = {name: pd.read_csv(path) for name, path in RAW_RESULT_FILES.items()}
    for name, frame in frames.items():
        validate_structure(frame, name)
        validate_result_values(frame, name)
    validate_alignment(frames)
    for name in ("oracle", "post_selection"):
        validate_cr_components_frame(frames[name], name)

    harmonized = harmonize_frames(frames)
    estimator_summary = summarize_estimator(harmonized)
    scenario_summary = summarize_scenarios(harmonized)
    warnings, warning_scenarios, taxonomy = warning_summaries(frames)
    exceptions, exception_scenarios = exception_diagnostics(frames)
    paired, paired_scenarios, discordance = paired_comparisons(frames)
    uncertainty = coverage_uncertainty(scenario_summary)
    classifications, rules = classify_estimators(
        estimator_summary, scenario_summary, frames
    )
    metadata = {
        name: artifact_metadata(path, display_root=PROJECT_ROOT)
        for name, path in RAW_RESULT_FILES.items()
    }
    metadata["manifest"] = artifact_metadata(
        RAW_MANIFEST_PATH, display_root=PROJECT_ROOT
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    products = {
        "warning_summary.csv": warnings,
        "warning_scenario_summary.csv": warning_scenarios,
        "exception_rows.csv": exceptions,
        "exception_scenario_summary.csv": exception_scenarios,
        "paired_estimator_summary.csv": paired,
        "paired_scenario_summary.csv": paired_scenarios,
        "coverage_discordance.csv": discordance,
        "coverage_uncertainty.csv": uncertainty,
        "estimator_classification.csv": classifications,
    }
    for name, frame in products.items():
        _write_csv(frame, OUTPUT_DIR / name)
    _write_json(taxonomy, OUTPUT_DIR / "warning_taxonomy.json")
    _write_json(rules, OUTPUT_DIR / "classification_rules.json")

    overall_table = estimator_summary.merge(
        classifications, on="estimator", suffixes=("", "_class")
    )[[
        "estimator", "observations", "conditional_coverage", "coverage_denominator",
        "bias", "rmse", "mean_cr_length", "unresolved_rows", "empty_valid_rate",
        "classification", "diagnostic_confidence", "machine_readable_reasons",
    ]]
    worst_table = scenario_summary.sort_values(
        ["estimator", "coverage_gap", "dgp", "n", "p", "pi", "tau"],
        kind="mergesort",
    ).groupby("estimator", sort=True).head(10)[[
        "estimator", "dgp", "n", "p", "pi", "tau", "conditional_coverage",
        "coverage_denominator", "coverage_gap", "coverage_mcse", "coverage_mc95_lower",
        "coverage_mc95_upper", "rmse", "mean_cr_length",
    ]]
    exception_table = exceptions.groupby(
        ["estimator_name", "exception_type", "cause_classification"], sort=True
    ).size().rename("count").reset_index()
    uncertainty_table = classifications[[
        "estimator", "classification", "diagnostic_confidence", "conditional_coverage",
        "coverage_denominator", "overall_wilson95_lower", "overall_wilson95_upper",
        "wilson_includes_nominal", "machine_readable_reasons",
    ]]
    tables = {
        "overall_estimator_comparison": overall_table,
        "worst_coverage_scenarios": worst_table,
        "warning_decomposition": warnings,
        "empty_unresolved_summary": exception_table,
        "paired_estimator_comparison": paired,
        "coverage_uncertainty_and_classification": uncertainty_table,
    }
    for name, frame in tables.items():
        _write_table(frame, name)

    _coverage_figure(uncertainty)
    _strength_figure(scenario_summary, "rmse", "Scenario-mean RMSE", "rmse_by_instrument_strength")
    _strength_figure(
        scenario_summary, "mean_cr_length", "Scenario-mean CR length",
        "cr_length_by_instrument_strength",
    )
    _paired_coverage_figure(paired_scenarios)
    _warning_figure(warnings)
    _exception_figure(exceptions, frames)

    report = _build_report(
        metadata,
        warnings,
        exceptions,
        exception_scenarios,
        paired,
        classifications,
        scenario_summary,
    )
    (OUTPUT_DIR / "phase2_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    print(
        f"Validated 3 immutable artifacts and wrote Phase 2 diagnostics to "
        f"{OUTPUT_DIR.relative_to(PROJECT_ROOT)}"
    )
    print(classifications[["estimator", "classification", "diagnostic_confidence"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
