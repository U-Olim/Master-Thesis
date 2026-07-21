"""Audit the immutable historical R=500 Oracle, Post-selection, and DML results."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


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
    NOMINAL_COVERAGE,
    artifact_metadata,
    comparison_checks,
    defensibility,
    harmonize_frames,
    monotonicity_checks,
    summarize_estimator,
    summarize_scenarios,
    suspicious_patterns,
    validate_alignment,
    validate_cr_components_frame,
    validate_result_values,
    validate_structure,
    worst_scenarios,
)


OUTPUT_DIR = PROJECT_ROOT / "results" / "validation" / "r500_audit"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")


def _fmt(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "unavailable"
    return f"{float(value):.{digits}f}"


def _pattern_text(monotonicity: pd.DataFrame, comparison: str) -> str:
    selected = monotonicity[monotonicity.comparison == comparison]
    rates = selected.groupby(["estimator", "metric"], sort=True)[
        "follows_expectation"
    ].mean()
    return "; ".join(
        f"{estimator} {metric}: {float(rate):.1%} follow the stated direction"
        for (estimator, metric), rate in rates.items()
    ) + "."


def _factor_text(scenarios: pd.DataFrame, factor: str) -> str:
    rows = []
    for estimator, group in scenarios.groupby("estimator", sort=True):
        means = group.groupby(factor, sort=True)[
            ["conditional_coverage", "rmse", "mean_cr_length"]
        ].mean()
        lowest = means["conditional_coverage"].idxmin()
        largest_rmse = means["rmse"].idxmax()
        rows.append(
            f"{estimator}: lowest mean scenario coverage at {factor}={lowest} "
            f"({_fmt(means.loc[lowest, 'conditional_coverage'])}); largest mean RMSE "
            f"at {factor}={largest_rmse} ({_fmt(means.loc[largest_rmse, 'rmse'])})"
        )
    return "; ".join(rows) + "."


def _artifact_text(name: str, values: dict[str, object]) -> str:
    dimensions = (
        "non-tabular JSON"
        if values["row_count"] is None
        else f"{values['row_count']:,} rows, {values['column_count']} columns"
    )
    return (
        f"{name}: SHA-256 `{values['sha256']}`, {values['size_bytes']:,} bytes, "
        f"{dimensions}, modified {values['modified_utc']}"
    )


def _worst_coverage_text(worst: pd.DataFrame) -> str:
    selected = worst[(worst["metric"] == "coverage_gap") & (worst["rank"] <= 3)]
    rows = []
    for estimator, group in selected.groupby("estimator", sort=True):
        cells = ", ".join(
            f"{row.dgp}/n={row.n:g}/p={row.p:g}/pi={row.pi:g}/tau={row.tau:g}: "
            f"gap {row.value:+.3f} (m={int(row.coverage_denominator)})"
            for row in group.itertuples(index=False)
        )
        rows.append(f"{estimator}: {cells}")
    return "; ".join(rows) + "."


def build_report(
    estimator_summary: pd.DataFrame,
    scenarios: pd.DataFrame,
    worst: pd.DataFrame,
    monotonicity: pd.DataFrame,
    suspicious: pd.DataFrame,
    classifications: dict[str, str],
    structural: dict[str, object],
    metadata: dict[str, object],
) -> str:
    by_estimator = estimator_summary.set_index("estimator")
    lines = [
        "# Joint scientific audit of immutable R=500 results",
        "",
        "## 1. Executive conclusion",
        "",
        "All three files satisfy the recorded structural and cross-estimator alignment "
        "contracts. They are suitable for thesis analysis subject to the estimator-specific "
        "coverage and finite-sample caveats below. Oracle is an infeasible benchmark; "
        "Post-selection reflects estimated-support costs; DML uses a different inferential "
        "construction and has fewer historical diagnostics.",
        "",
        "## 2. Artifact integrity",
        "",
        "Manifest checksum, byte-size, ordered-column, and dimension validation passed. "
        + "; ".join(
            _artifact_text(name, metadata[name])
            for name in ("oracle", "post_selection", "dml", "manifest")
        ) + ".",
        "",
        "## 3. Design completeness",
        "",
        f"Each estimator contains {structural['oracle']['rows']:,} rows, "
        f"{structural['oracle']['design_cells']} design cells, 500 replications numbered "
        "0-499, and 144 rows per replication. Natural keys, seeds, and `alpha_true` agree "
        "exactly across estimators. Oracle is in natural-key order; historical Post-selection "
        "and DML file order is deterministic but not lexicographic natural-key order. Audit "
        "products are stably sorted; raw files were not reordered.",
        "",
        "## 4. Metric definitions",
        "",
        "Bias and RMSE use finite point-estimation errors. Confidence-region length uses "
        "finite historical `cr_length` values. Unsupported diagnostics remain missing, not zero.",
        "",
        "## 5. Coverage denominator policy",
        "",
        "Unconditional coverage uses all nonmissing Boolean `covered` observations. "
        "Conditional coverage additionally requires convergence, finite CR geometry, and, "
        "for Oracle/Post-selection, historical numerical resolution. DML has no resolution "
        "field, so its denominator requires convergence and finite legacy CR fields. "
        "Unresolved observations are reported separately and never coded as noncoverage.",
        "",
        f"Nominal coverage is {NOMINAL_COVERAGE:.0%}, confirmed by production estimator "
        "`confidence_level` defaults. Scenario Monte Carlo standard errors use "
        "sqrt(p(1-p)/m), with bounded Wald 95% intervals and the actual denominator. A "
        "scenario is within Monte Carlo uncertainty when its interval contains 95%; otherwise "
        "a negative gap of at least 10 percentage points is severe, a smaller negative gap "
        "is moderate, and a positive statistically distinguishable gap is overcoverage.",
    ]
    for number, estimator, title in (
        (6, "oracle", "Oracle results"),
        (7, "post_selection", "Post-selection results"),
        (8, "dml", "DML results"),
    ):
        row = by_estimator.loc[estimator]
        severe = int(
            scenarios.loc[scenarios.estimator == estimator, "coverage_assessment"]
            .eq("severe undercoverage")
            .sum()
        )
        lines.extend([
            "",
            f"## {number}. {title}",
            "",
            f"Classification: **{classifications[estimator]}**. Conditional coverage is "
            f"{_fmt(row.conditional_coverage)} (m={int(row.coverage_denominator):,}); "
            f"unconditional coverage is {_fmt(row.unconditional_coverage)} "
            f"(m={int(row.unconditional_coverage_denominator):,}); bias is {_fmt(row.bias)}; "
            f"RMSE is {_fmt(row.rmse)}; mean CR length is {_fmt(row.mean_cr_length)}; "
            f"convergence is {_fmt(row.convergence_rate)}. The conditional rule excludes "
            f"{int(row.conditional_excluded_rows):,} rows. There are {severe} severe-"
            "undercoverage scenarios under the disclosed rule.",
        ])
        if estimator == "oracle":
            lines.append(
                f"Full-grid frequency is {_fmt(row.full_grid_rate)}, numerical resolution "
                f"is {_fmt(row.numerically_resolved_rate)}, unresolved rows are "
                f"{int(row.unresolved_rows)}, and warning frequency is "
                f"{_fmt(row.iteration_warning_rate)}. True-support knowledge does not remove "
                "weak-instrument finite-sample uncertainty."
            )
        elif estimator == "post_selection":
            lines.append(
                f"Mean selected controls are {_fmt(row.mean_selected_controls)} (median "
                f"{_fmt(row.median_selected_controls)}; range "
                f"{int(row.min_selected_controls)}-{int(row.max_selected_controls)}). "
                f"Unresolved rows are {int(row.unresolved_rows)}, full-grid frequency is "
                f"{_fmt(row.full_grid_rate)}, and warning frequency is "
                f"{_fmt(row.iteration_warning_rate)}. The multiplier record is "
                f"`{row.selection_multiplier_values}`: these historical results use 1.0, "
                "not the future-run value 1.8; no counterfactual claim is made."
            )
        else:
            lines.append(
                "The 15-column DML artifact contains no CR components, resolution status, "
                "block count, warning, or nuisance/cross-fitting diagnostics. Those properties "
                f"cannot be audited or reported as zero. Its {int(row.conditional_excluded_rows)} "
                "finite-CR exclusions are not called unresolved because no resolution field exists."
            )

    flagged = suspicious[suspicious["count"] > 0]
    lines.extend([
        "",
        "## 9. Cross-estimator comparison",
        "",
        "Pairwise scenario differences are in `comparison_checks.csv`. Coverage, RMSE, and "
        "interval length must be interpreted jointly: lower RMSE does not establish superior "
        "inference, and wider intervals may purchase coverage. Oracle is a benchmark, not a "
        "feasible competitor; Post-selection differences reflect estimated-support costs; "
        "DML diagnostic differences are not directly comparable where fields are absent.",
        "",
        "## 10. Weak-instrument patterns",
        "",
        _pattern_text(monotonicity, "instrument_strength"),
        "",
        "## 11. Sample-size patterns",
        "",
        _pattern_text(monotonicity, "sample_size"),
        "",
        "## 12. Dimension patterns",
        "",
        _pattern_text(monotonicity, "dimension"),
        "",
        "## 13. Quantile patterns",
        "",
        _factor_text(scenarios, "tau"),
        "",
        "## 14. DGP patterns",
        "",
        _factor_text(scenarios, "dgp") + " DGP1 is the five-control Gaussian baseline, "
        "DGP2 is the denser ten-control Gaussian selection-stress design, and DGP3 is the "
        "five-control heavy-tail robustness design. These are descriptive associations, "
        "not causal explanations.",
        "",
        "## 15. Worst scenarios",
        "",
        f"`worst_scenarios.csv` contains {len(worst)} metric-labelled ranked rows: ten per "
        "estimator for coverage, RMSE, CR length, non-convergence, and resolution where "
        "available. Three worst coverage gaps per estimator: " + _worst_coverage_text(worst),
        "",
        "## 16. Suspicious findings",
        "",
        (
            "No suspicious-pattern checks were flagged."
            if flagged.empty
            else "; ".join(
                f"{row.estimator}/{row.check}: {int(row.count)}"
                for row in flagged.itertuples(index=False)
            ) + "."
        ),
        "",
        "The row-order flags document historical file order, not corruption. Component-based "
        "coverage and geometry validation passed for Oracle and Post-selection. DML has no "
        "components, so false coverage inside its hull is not treated as inconsistent because "
        "the accepted set may be disconnected.",
        "",
        "## 17. Thesis defensibility",
        "",
        "The rule is: strong requires overall conditional coverage >=93%, convergence >=99%, "
        "and no more than 10% severe-undercoverage scenarios; acceptable with caveats requires "
        "coverage >=85% and convergence >=95%; problematic requires coverage >=70% and "
        "convergence >=80%; otherwise not defensible. Scenario weaknesses must still be "
        "disclosed. The classifications describe these artifacts, not universal method validity.",
        "",
        "## 18. Required caveats",
        "",
        "Oracle is infeasible; weak instruments can generate broad or full-grid regions; "
        "Post-selection uses multiplier 1.0 and estimated support; DML's historical schema "
        "prevents claims beyond its 15 fields; 500 replications leave measurable Monte Carlo "
        "uncertainty. Difficult-scenario weakness is not evidence of corruption or, by itself, "
        "implementation failure.",
        "",
        "## 19. Limitations of historical schemas",
        "",
        "This audit does not prove or disprove asymptotic validity or identify causal reasons "
        "for every pattern. Oracle's 43-column and DML's 15-column historical schemas differ "
        "from current future-output schemas and were validated as historical contracts rather "
        "than rewritten.",
        "",
        "## 20. Recommendation for thesis tables and figures",
        "",
        "Later thesis outputs should show conditional coverage with denominators and Monte "
        "Carlo uncertainty, RMSE and CR length together, weak-instrument panels, ranked caveat "
        "scenarios, and explicit schema/calibration notes. This audit does not create final "
        "publication tables or figures.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    verify_raw_manifest()
    load_all_results(expected_replications=500)
    frames = {name: pd.read_csv(path) for name, path in RAW_RESULT_FILES.items()}
    structural = {
        name: validate_structure(frame, name) for name, frame in frames.items()
    }
    for name, frame in frames.items():
        validate_result_values(frame, name)
    alignment = validate_alignment(frames)
    component_results = {
        name: validate_cr_components_frame(frames[name], name)
        for name in ("oracle", "post_selection")
    }
    harmonized = harmonize_frames(frames)
    estimator_summary = summarize_estimator(harmonized)
    scenario_summary = summarize_scenarios(harmonized)
    if len(estimator_summary) != 3:
        raise ValueError("Estimator summary must contain exactly three rows")
    if len(scenario_summary) != 432:
        raise ValueError("Scenario summary must contain exactly 432 rows")
    worst = worst_scenarios(scenario_summary)
    monotonicity = monotonicity_checks(scenario_summary)
    comparisons = comparison_checks(scenario_summary)
    suspicious = suspicious_patterns(frames, scenario_summary)
    classifications = defensibility(estimator_summary, scenario_summary)
    multiplier_counts = (
        frames["post_selection"]["selection_lasso_multiplier"]
        .value_counts(dropna=False)
        .sort_index()
    )
    if list(multiplier_counts.index) != [1.0]:
        raise ValueError("Historical Post-selection multiplier must be exactly 1.0")
    metadata = {
        name: artifact_metadata(path, display_root=PROJECT_ROOT)
        for name, path in RAW_RESULT_FILES.items()
    }
    metadata["manifest"] = artifact_metadata(
        RAW_MANIFEST_PATH, display_root=PROJECT_ROOT
    )
    validation = {
        "nominal_coverage": NOMINAL_COVERAGE,
        "nominal_source": "Estimator confidence_level defaults in production code",
        "historical_post_selection_multiplier": {"1.0": int(multiplier_counts.iloc[0])},
        "artifacts": metadata,
        "structure": structural,
        "alignment": alignment,
        "component_validation": component_results,
        "estimator_summary_rows": len(estimator_summary),
        "scenario_summary_rows": len(scenario_summary),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "structural_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(estimator_summary, OUTPUT_DIR / "estimator_summary.csv")
    _write_csv(scenario_summary, OUTPUT_DIR / "scenario_summary.csv")
    _write_csv(worst, OUTPUT_DIR / "worst_scenarios.csv")
    _write_csv(monotonicity, OUTPUT_DIR / "monotonicity_checks.csv")
    _write_csv(comparisons, OUTPUT_DIR / "comparison_checks.csv")
    _write_csv(suspicious, OUTPUT_DIR / "diagnostic_summary.csv")
    report = build_report(
        estimator_summary,
        scenario_summary,
        worst,
        monotonicity,
        suspicious,
        classifications,
        structural,
        metadata,
    )
    (OUTPUT_DIR / "audit_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    relative = OUTPUT_DIR.relative_to(PROJECT_ROOT)
    print(f"Validated 3 artifacts and wrote audit outputs to {relative}")
    columns = ["estimator", "conditional_coverage", "rmse", "mean_cr_length"]
    print(estimator_summary[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
