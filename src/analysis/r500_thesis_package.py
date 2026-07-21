"""Deterministic Phase 3 presentation layer for validated R=500 audit outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from analysis.r500_phase2 import wilson_interval


ESTIMATOR_ORDER = ("oracle", "post_selection", "dml")
ESTIMATOR_LABELS = {
    "oracle": "Oracle IVQR",
    "post_selection": "Post-selection IVQR",
    "dml": "DML-style IVQR",
}
SCENARIO_ORDER = ("dgp", "n", "p", "pi", "tau")
PAIR_ORIENTATION = "estimator A - estimator B"
CSV_MISSING = "NA"
LATEX_MISSING = "N/A"
FLOAT_FORMAT = "%.10g"
TABLE_FAMILIES = (
    "table_01_simulation_design",
    "table_02_overall_estimator_performance",
    "table_03_estimator_tradeoffs",
    "table_04_performance_by_quantile",
    "table_05_performance_by_instrument_strength",
    "table_06_weakest_scenarios",
    "table_07_warning_exception_diagnostics",
)
FIGURE_FAMILIES = (
    "figure_01_overall_estimator_tradeoff",
    "figure_02_coverage_by_quantile",
    "figure_03_coverage_by_instrument_strength",
    "figure_04_rmse_by_instrument_strength",
    "figure_05_cr_length_by_instrument_strength",
    "figure_06_paired_estimator_differences",
    "figure_07_weak_scenario_structure",
    "figure_08_warning_exception_diagnostics",
)

SOURCE_CONTRACTS: dict[str, dict[str, object]] = {
    "results/validation/r500_audit/structural_validation.json": {
        "sha256": "080bfb57216f7ad9e83848e5efac7fb6208ee0e37b55396cc2323edb03e2f26a"
    },
    "results/validation/r500_audit/estimator_summary.csv": {
        "sha256": "cfe33284157974b792f7d55fb1ca63a2c527bcb8923b84371653f9641df5be07",
        "rows": 3,
    },
    "results/validation/r500_audit/scenario_summary.csv": {
        "sha256": "fda31b8f55a395ab4e757bbcc527eb05161d08d64a413eb5d361fef95fd9f770",
        "rows": 432,
    },
    "results/validation/r500_phase2/coverage_uncertainty.csv": {
        "sha256": "7a7316d2d88fd1b56b7a6bee1aea862f43dc9265c116af4262d755bab4334b57",
        "rows": 432,
    },
    "results/validation/r500_phase2/estimator_classification.csv": {
        "sha256": "72962dd8948a9073e2ddbbc1cddd3c884b5c27f5f312e9d42323b26b666d139b",
        "rows": 3,
    },
    "results/validation/r500_phase2/paired_estimator_summary.csv": {
        "sha256": "ffc42d05fefb003b1ba5452ab2c97944d6f2a994594c8c5221cdceb11c54d601",
        "rows": 12,
    },
    "results/validation/r500_phase2/tables/worst_coverage_scenarios.csv": {
        "sha256": "76561edadadaeb5fcd732d0edafd789743bec1b84075da4ac6c3129d0f61fded",
        "rows": 30,
    },
    "results/validation/r500_phase2/warning_summary.csv": {
        "sha256": "28c8b6919a33ec6723d8851f426b7b99d1841384fdaa7b0910bb9fcc61975ad8",
        "rows": 14,
    },
    "results/validation/r500_phase2/warning_scenario_summary.csv": {
        "sha256": "0acd9a0cfc494ad5061148e739f74dc135a9dd8f95432b52d019527ecbb7e6aa",
        "rows": 2016,
    },
    "results/validation/r500_phase2/exception_rows.csv": {
        "sha256": "e197a3cee9b76689772ef0f932936e56033db45e7fe716d28f4428a67185f030",
        "rows": 99,
    },
    "results/validation/r500_phase2/exception_scenario_summary.csv": {
        "sha256": "5f142e906ca4ddf13c7a9b7d6281dcded00b356a104337bbf6ef7771f3cbdc1e",
        "rows": 66,
    },
    "results/validation/r500_phase2/phase2_report.md": {
        "sha256": "ddd4a77f38dc6cd84003f51208155e45b6d4cd2a7a4b2b7bb5445b90c483ca40"
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_contracts(project_root: str | Path) -> dict[str, str]:
    """Fail when a required committed Phase 1/2 source differs from its contract."""
    root = Path(project_root)
    hashes: dict[str, str] = {}
    for relative, contract in SOURCE_CONTRACTS.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required authoritative source is missing: {relative}")
        actual_hash = sha256_file(path)
        if actual_hash != contract["sha256"]:
            raise ValueError(f"Authoritative source hash mismatch: {relative}")
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
            if len(frame) != contract["rows"]:
                raise ValueError(f"Authoritative source row-count mismatch: {relative}")
        hashes[relative] = actual_hash
    return hashes


def load_sources(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root)
    validate_source_contracts(root)
    sources: dict[str, object] = {}
    for relative in SOURCE_CONTRACTS:
        path = root / relative
        key = path.stem
        if key in sources:
            key = f"{path.parent.name}_{key}"
        if path.suffix == ".json":
            sources[key] = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".csv":
            sources[key] = pd.read_csv(path)
        else:
            sources[key] = path.read_text(encoding="utf-8")
    return sources


def estimator_sort(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["estimator"] = pd.Categorical(
        result["estimator"], categories=ESTIMATOR_ORDER, ordered=True
    )
    result = result.sort_values("estimator", kind="mergesort").reset_index(drop=True)
    result["estimator"] = result["estimator"].astype(object)
    return result


def scenario_sort(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in SCENARIO_ORDER if column in frame]
    prefix = [column for column in ("estimator", "rank") if column in frame]
    result = frame.copy()
    if "estimator" in result:
        result["estimator"] = pd.Categorical(
            result["estimator"], categories=ESTIMATOR_ORDER, ordered=True
        )
    result = result.sort_values([*prefix, *columns], kind="mergesort").reset_index(drop=True)
    if "estimator" in result:
        result["estimator"] = result["estimator"].astype(object)
    return result


def build_design_table(
    structural: Mapping[str, object], scenarios: pd.DataFrame
) -> pd.DataFrame:
    structure = structural["structure"]
    oracle = structure["oracle"]
    value_text = {
        "dgp": "; ".join(sorted(scenarios["dgp"].astype(str).unique())),
        "n": "; ".join(str(int(value)) for value in sorted(scenarios["n"].unique())),
        "p": "; ".join(str(int(value)) for value in sorted(scenarios["p"].unique())),
        "pi": "; ".join(f"{value:.2f}" for value in sorted(scenarios["pi"].unique())),
        "tau": "; ".join(f"{value:.2f}" for value in sorted(scenarios["tau"].unique())),
    }
    return pd.DataFrame([{
        "dgp_values": value_text["dgp"],
        "sample_sizes": value_text["n"],
        "dimensionalities": value_text["p"],
        "instrument_strengths": value_text["pi"],
        "quantiles": value_text["tau"],
        "replications_per_design_cell": int(oracle["unique_replications"]),
        "design_cells": int(oracle["design_cells"]),
        "rows_per_estimator": int(oracle["rows"]),
        "estimator_count": len(ESTIMATOR_ORDER),
        "total_estimator_rows": int(sum(structure[name]["rows"] for name in ESTIMATOR_ORDER)),
    }])


def build_overall_table(summary: pd.DataFrame, classification: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "estimator", "conditional_coverage", "overall_wilson95_lower",
        "overall_wilson95_upper", "coverage_denominator", "coverage_gap", "bias",
        "rmse", "mean_cr_length", "empty_valid_rate", "unresolved_rate",
        "iteration_warning_rate", "classification", "diagnostic_confidence",
    ]
    result = summary.merge(
        classification[[
            "estimator", "overall_wilson95_lower", "overall_wilson95_upper",
            "coverage_gap", "classification", "diagnostic_confidence",
        ]],
        on="estimator",
        validate="one_to_one",
    )
    return estimator_sort(result.loc[:, columns])


def build_tradeoff_table(paired: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    pair_order = (("oracle", "post_selection"), ("oracle", "dml"), ("post_selection", "dml"))
    for estimator_a, estimator_b in pair_order:
        selected = paired[
            paired["estimator_a"].eq(estimator_a)
            & paired["estimator_b"].eq(estimator_b)
        ]
        if selected.empty:
            continue
        group = selected.set_index("metric")
        record: dict[str, object] = {
            "estimator_a": estimator_a,
            "estimator_b": estimator_b,
            "difference_orientation": PAIR_ORIENTATION,
        }
        for metric, prefix in (
            ("coverage", "coverage"),
            ("squared_error", "squared_error"),
            ("cr_length", "cr_length"),
        ):
            row = group.loc[metric]
            record.update({
                f"paired_{prefix}_difference": row["mean_paired_difference"],
                f"paired_{prefix}_ci95_lower": row["paired_ci95_lower"],
                f"paired_{prefix}_ci95_upper": row["paired_ci95_upper"],
                f"paired_{prefix}_denominator": int(row["valid_paired_denominator"]),
            })
        records.append(record)
    return pd.DataFrame(records)


def aggregate_performance(
    scenarios: pd.DataFrame,
    uncertainty: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:
    """Aggregate validated scenario summaries using their actual metric denominators."""
    merged = scenarios.merge(
        uncertainty[[*SCENARIO_ORDER, "estimator", "coverage_successes"]],
        on=["estimator", *SCENARIO_ORDER],
        validate="one_to_one",
    )
    records: list[dict[str, object]] = []
    for key, group in merged.groupby(["estimator", dimension], sort=True):
        estimator, value = key
        coverage_denominator = int(group["coverage_denominator"].sum())
        successes = int(group["coverage_successes"].sum())
        coverage = successes / coverage_denominator
        wilson_lower, wilson_upper = wilson_interval(successes, coverage_denominator)
        point_weights = group["observations"] * (1 - group["missing_estimate_rate"])
        length_weights = group["observations"] * (1 - group["missing_cr_rate"])
        records.append({
            "estimator": estimator,
            dimension: value,
            "conditional_coverage": coverage,
            "wilson95_lower": wilson_lower,
            "wilson95_upper": wilson_upper,
            "coverage_denominator": coverage_denominator,
            "bias": float(np.average(group["bias"], weights=point_weights)),
            "rmse": float(np.sqrt(np.average(group["rmse"] ** 2, weights=point_weights))),
            "mean_cr_length": float(
                np.average(group["mean_cr_length"], weights=length_weights)
            ),
        })
    return estimator_sort(pd.DataFrame(records))


def build_weakest_table(
    worst: pd.DataFrame,
    uncertainty: pd.DataFrame,
    scenarios: pd.DataFrame,
    warning_scenarios: pd.DataFrame,
    exception_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["estimator", *SCENARIO_ORDER]
    parts = []
    for estimator in ESTIMATOR_ORDER:
        part = worst[worst["estimator"].eq(estimator)].head(5).copy()
        part["rank"] = range(1, len(part) + 1)
        parts.append(part)
    selected = pd.concat(parts, ignore_index=True)
    selected = selected.merge(
        uncertainty[[*keys, "wilson95_lower", "wilson95_upper"]],
        on=keys,
        validate="one_to_one",
    )
    selected = selected.merge(
        scenarios[[*keys, "iteration_warning_rate", "empty_valid_rate", "unresolved_rate"]],
        on=keys,
        validate="one_to_one",
    )
    iteration = warning_scenarios[
        warning_scenarios["warning_category"].eq("iteration_warning")
    ][[*keys, "warning_frequency"]]
    selected = selected.drop(columns="iteration_warning_rate").merge(
        iteration, on=keys, how="left", validate="one_to_one"
    )
    exception_counts = exception_scenarios.pivot_table(
        index=keys,
        columns="exception_type",
        values="exception_count",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    selected = selected.merge(exception_counts, on=keys, how="left", validate="one_to_one")
    for column in ("empty_cr", "unresolved_cr"):
        if column not in selected:
            selected[column] = 0
    selected[["empty_cr", "unresolved_cr"]] = selected[
        ["empty_cr", "unresolved_cr"]
    ].fillna(0)
    dml = selected["estimator"].eq("dml")
    selected.loc[dml, ["warning_frequency", "empty_cr", "unresolved_cr"]] = np.nan
    selected["ranking_basis"] = "Phase 2 ascending conditional-coverage gap"
    columns = [
        "estimator", "rank", *SCENARIO_ORDER, "conditional_coverage",
        "wilson95_lower", "wilson95_upper", "coverage_denominator", "coverage_gap",
        "rmse", "mean_cr_length", "warning_frequency", "empty_cr", "unresolved_cr",
        "ranking_basis",
    ]
    return scenario_sort(selected.loc[:, columns])


def build_warning_exception_table(
    warning_summary: pd.DataFrame, exception_rows: pd.DataFrame
) -> pd.DataFrame:
    warning = warning_summary[
        warning_summary["warning_category"].eq("iteration_warning")
    ].set_index("estimator")
    records: list[dict[str, object]] = []
    for estimator in ESTIMATOR_ORDER:
        observed = exception_rows[exception_rows["estimator_name"].eq(estimator)]
        if estimator == "dml":
            warning_values = {name: np.nan for name in (
                "warning_frequency", "warning_event_count", "coverage_affected_valid",
                "coverage_without_warning_valid", "rmse", "rmse_without_warning",
            )}
            empty = unresolved = np.nan
            missing_geometry = int(observed["exception_type"].eq("missing_legacy_geometry").sum())
        else:
            warning_values = warning.loc[estimator].to_dict()
            empty = int(observed["exception_type"].eq("empty_cr").sum())
            unresolved = int(observed["exception_type"].eq("unresolved_cr").sum())
            missing_geometry = np.nan
        records.append({
            "estimator": estimator,
            "warning_row_frequency": warning_values["warning_frequency"],
            "warning_event_count": warning_values["warning_event_count"],
            "coverage_with_warnings": warning_values["coverage_affected_valid"],
            "coverage_without_warnings": warning_values["coverage_without_warning_valid"],
            "rmse_with_warnings": warning_values["rmse"],
            "rmse_without_warnings": warning_values["rmse_without_warning"],
            "empty_confidence_regions": empty,
            "unresolved_rows": unresolved,
            "missing_legacy_geometry": missing_geometry,
            "validated_exception_count": len(observed),
        })
    return estimator_sort(pd.DataFrame(records))


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def format_value(value: object, *, latex: bool = False) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return LATEX_MISSING if latex else CSV_MISSING
    if isinstance(value, (float, np.floating)):
        text = f"{float(value):.4f}"
    elif isinstance(value, (bool, np.bool_)):
        text = "Yes" if bool(value) else "No"
    else:
        text = str(value)
    return latex_escape(text) if latex else text


def dataframe_to_latex(frame: pd.DataFrame) -> str:
    columns = [latex_escape(column) for column in frame.columns]
    lines = [
        rf"\begin{{tabular}}{{{'l' * len(columns)}}}",
        r"\hline",
        " & ".join(columns) + r" \\",
        r"\hline",
    ]
    lines.extend(
        " & ".join(format_value(value, latex=True) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def finding(
    finding_id: str,
    topic: str,
    values: Mapping[str, object],
    source: str,
    source_hash: str,
    source_filter: Mapping[str, object],
    denominator: object,
    strength: str,
    permitted: str,
    prohibited: str,
    availability: str = "available",
) -> dict[str, object]:
    if strength not in {"descriptive", "statistically_supported", "limitation"}:
        raise ValueError(f"Invalid interpretation strength: {strength}")
    return json_ready({
        "id": finding_id,
        "topic": topic,
        "values": values,
        "authoritative_source": source,
        "source_hash": source_hash,
        "source_row_or_filter": source_filter,
        "denominator": denominator,
        "interpretation_strength": strength,
        "permitted_wording": permitted,
        "prohibited_overinterpretation": prohibited,
        "availability": availability,
    })


PROHIBITED_WORDS = re.compile(r"\b(caused|proved|eliminated|guaranteed)\b", re.IGNORECASE)
HISTORICAL_18 = re.compile(
    r"(?:historical multiplier (?:is|=)|historical results use multiplier|"
    r"historical post-selection results use multiplier)\s*1\.8",
    re.IGNORECASE,
)


def validate_findings(findings: Sequence[Mapping[str, object]]) -> None:
    identifiers = [str(item["id"]) for item in findings]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Thesis finding IDs must be unique")
    for item in findings:
        source = str(item["authoritative_source"])
        if source not in SOURCE_CONTRACTS:
            raise ValueError(f"Finding has unrecognized authoritative source: {source}")
        if item["source_hash"] != SOURCE_CONTRACTS[source]["sha256"]:
            raise ValueError(f"Finding source hash mismatch: {item['id']}")
        if PROHIBITED_WORDS.search(str(item["permitted_wording"])):
            raise ValueError(f"Finding uses prohibited causal wording: {item['id']}")


def sourced_sentence(text: str, finding_ids: Sequence[str]) -> str:
    if not finding_ids:
        raise ValueError("Every report sentence must cite at least one finding")
    return f"{text} <!-- findings:{','.join(finding_ids)} -->"


def validate_report_provenance(report: str, finding_ids: set[str]) -> None:
    citation = re.compile(r"<!-- findings:([^>]+) -->")
    for line in report.splitlines():
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        if any(character.isdigit() for character in line):
            match = citation.search(line)
            if match is None:
                raise ValueError(f"Numerical report line lacks finding provenance: {line}")
            cited = {value.strip() for value in match.group(1).split(",")}
            if not cited.issubset(finding_ids):
                raise ValueError(f"Report cites unknown finding IDs: {sorted(cited - finding_ids)}")


def validate_historical_multiplier_outputs(output_dir: str | Path) -> None:
    """Reject any thesis text that attributes multiplier 1.8 to historical results."""
    for path in sorted(Path(output_dir).rglob("*")):
        if path.suffix.lower() not in {".md", ".csv", ".tex", ".json"}:
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            payload = json.loads(text)

            def permitted_text(value: object, key: str = "") -> list[str]:
                if key in {"prohibited_overinterpretation", "future_multiplier_not_analyzed"}:
                    return []
                if isinstance(value, dict):
                    return [
                        item
                        for child_key, child in value.items()
                        for item in permitted_text(child, str(child_key))
                    ]
                if isinstance(value, list):
                    return [item for child in value for item in permitted_text(child, key)]
                return [value] if isinstance(value, str) else []

            text = "\n".join(permitted_text(payload))
        if HISTORICAL_18.search(text):
            raise ValueError(f"Historical multiplier 1.8 misattribution in {path.name}")


def consistency_row(
    check_id: str,
    metric: str,
    source: str,
    expected: object,
    actual: object,
    tolerance: float = 0.0,
) -> dict[str, object]:
    if pd.isna(expected) and pd.isna(actual):
        difference = 0.0
        passed = True
    elif isinstance(expected, (int, float, np.number)) and isinstance(
        actual, (int, float, np.number)
    ):
        difference = abs(float(expected) - float(actual))
        passed = difference <= tolerance
    else:
        difference = 0.0 if str(expected) == str(actual) else np.nan
        passed = str(expected) == str(actual)
    return {
        "check_id": check_id,
        "metric": metric,
        "authoritative_source": source,
        "expected_value": expected,
        "thesis_output": actual,
        "absolute_difference": difference,
        "tolerance": tolerance,
        "status": "passed" if passed else "failed",
    }


def require_consistency(checks: pd.DataFrame) -> None:
    failed = checks[~checks["status"].eq("passed")]
    if not failed.empty:
        identifiers = ", ".join(failed["check_id"].astype(str))
        raise ValueError(f"Substantive Phase 3 consistency checks failed: {identifiers}")


__all__ = [
    "CSV_MISSING", "ESTIMATOR_LABELS", "ESTIMATOR_ORDER", "FIGURE_FAMILIES",
    "FLOAT_FORMAT", "LATEX_MISSING", "PAIR_ORIENTATION", "SCENARIO_ORDER",
    "SOURCE_CONTRACTS", "TABLE_FAMILIES", "aggregate_performance",
    "build_design_table", "build_overall_table", "build_tradeoff_table",
    "build_warning_exception_table", "build_weakest_table", "consistency_row",
    "dataframe_to_latex", "estimator_sort", "finding", "format_value", "json_ready",
    "latex_escape", "load_sources", "require_consistency", "scenario_sort",
    "sha256_file", "sourced_sentence", "validate_findings",
    "validate_historical_multiplier_outputs", "validate_report_provenance",
    "validate_source_contracts",
]
