"""Deep deterministic diagnostics for the immutable historical R=500 results."""

from __future__ import annotations

from collections.abc import Mapping
import json
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from analysis.r500_audit import (
    ESTIMATORS,
    NATURAL_KEY,
    NOMINAL_COVERAGE,
    SCENARIO_KEY,
)
from ivqr.confidence_regions import parse_cr_components


PAIR_ORDER = (
    ("oracle", "post_selection"),
    ("oracle", "dml"),
    ("post_selection", "dml"),
)
WARNING_DEFINITIONS: dict[str, dict[str, str]] = {
    "iteration_warning": {
        "column": "iteration_warning_evaluations",
        "kind": "count",
        "description": "One or more stored alpha-evaluation iteration warnings.",
    },
    "rank_deficient_covariance": {
        "column": "rank_deficient_covariance_failures",
        "kind": "count",
        "description": "One or more stored rank-deficient covariance failures.",
    },
    "midpoint_unresolved_barrier": {
        "column": "midpoint_unresolved_barriers",
        "kind": "count",
        "description": "One or more stored unresolved midpoint barriers.",
    },
    "refinement_unresolved_barrier": {
        "column": "number_of_unresolved_refinement_barriers",
        "kind": "count",
        "description": "One or more stored unresolved refinement barriers.",
    },
    "midpoint_probe_limit": {
        "column": "midpoint_probe_limit_hit",
        "kind": "boolean",
        "description": "The stored midpoint-probe limit indicator is true.",
    },
    "refinement_limit": {
        "column": "refinement_limit_hit",
        "kind": "boolean",
        "description": "The stored refinement-depth limit indicator is true.",
    },
    "maximum_alpha_evaluations": {
        "column": "max_alpha_evaluations_hit",
        "kind": "boolean",
        "description": "The stored maximum-alpha-evaluations indicator is true.",
    },
}
EXCEPTION_OPTIONAL_COLUMNS = (
    "cr_lower",
    "cr_upper",
    "cr_length",
    "cr_status",
    "cr_components",
    "cr_n_blocks",
    "cr_disconnected",
    "cr_unresolved_count",
    "cr_unresolved_alphas",
    "grid_strategy",
    "adaptive_midpoint_probe",
    "alpha_hat_grid",
    "initial_alpha_grid_size",
    "final_alpha_evaluations",
    "midpoint_intervals_considered",
    "midpoint_evaluations_added",
    "midpoint_unresolved_barriers",
    "midpoint_probe_limit_hit",
    "refinement_depth_reached",
    "refinement_limit_hit",
    "max_alpha_evaluations_hit",
    "number_of_refined_intervals",
    "number_of_unresolved_refinement_barriers",
    "iteration_warning_evaluations",
    "rank_deficient_covariance_failures",
    "n_selected_controls",
    "selection_lasso_multiplier",
    "selection_method",
    "n_retained_instruments",
)


def valid_coverage_mask(frame: pd.DataFrame, estimator: str) -> pd.Series:
    complete = np.isfinite(frame[["cr_lower", "cr_upper", "cr_length"]]).all(axis=1)
    mask = frame["converged"].eq(True) & complete & frame["covered"].notna()
    if estimator in {"oracle", "post_selection"}:
        mask &= frame["cr_is_numerically_resolved"].eq(True)
    return mask


def warning_membership(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return multi-label warning membership without inventing unavailable reasons."""
    records: list[pd.DataFrame] = []
    available: list[str] = []
    unavailable: list[str] = []
    for category, definition in WARNING_DEFINITIONS.items():
        column = definition["column"]
        if column not in frame:
            unavailable.append(category)
            continue
        available.append(category)
        if definition["kind"] == "count":
            events = pd.to_numeric(frame[column], errors="coerce").fillna(0)
        else:
            events = frame[column].eq(True).astype(int)
        affected = events.gt(0)
        if affected.any():
            selected = frame.loc[affected, NATURAL_KEY + ["seed"]].copy()
            selected["warning_category"] = category
            selected["warning_events"] = events.loc[affected].astype(int).to_numpy()
            records.append(selected)
    columns = NATURAL_KEY + ["seed", "warning_category", "warning_events"]
    membership = (
        pd.concat(records, ignore_index=True).loc[:, columns]
        if records
        else pd.DataFrame(columns=columns)
    )
    membership = membership.sort_values(
        ["warning_category", *NATURAL_KEY], kind="mergesort"
    ).reset_index(drop=True)
    metadata = {
        "available_categories": available,
        "unavailable_categories": unavailable,
        "textual_warning_reason_available": False,
        "counting_rule": (
            "A row belongs to every category whose stored Boolean is true or stored "
            "count is positive. warning_count counts affected rows; "
            "warning_event_count sums stored counts, so categories may overlap."
        ),
    }
    return membership, metadata


def _warning_classification(row: Mapping[str, object]) -> tuple[str, str]:
    count = int(row["warning_count"])
    if count == 0:
        return "unknown from stored information", "No affected historical rows."
    if int(row["unresolved_count"]) > 0:
        return "consequential", "At least one affected row is numerically unresolved."
    if int(row["empty_cr_count"]) > 0:
        return "potentially consequential", "At least one affected row has an empty CR."
    category = str(row["warning_category"])
    semantic_hazards = {
        "rank_deficient_covariance",
        "midpoint_unresolved_barrier",
        "refinement_unresolved_barrier",
        "midpoint_probe_limit",
        "refinement_limit",
        "maximum_alpha_evaluations",
    }
    affected = float(row["coverage_affected_valid"])
    unaffected = float(row["coverage_without_warning_valid"])
    if category in semantic_hazards or (
        np.isfinite(affected) and np.isfinite(unaffected) and affected < unaffected - 0.01
    ):
        return (
            "potentially consequential",
            "Stored category signals a numerical barrier/limit or affected coverage is "
            "more than one percentage point lower.",
        )
    return (
        "clearly benign",
        "No empty/unresolved rows and no material adverse coverage association under the rule.",
    )


def _warning_row(
    frame: pd.DataFrame,
    estimator: str,
    category: str,
    mask: pd.Series,
    events: pd.Series,
) -> dict[str, object]:
    valid = valid_coverage_mask(frame, estimator)
    affected_valid = mask & valid
    unaffected_valid = ~mask & valid
    errors = frame["alpha_hat"] - frame["alpha_true"]
    finite_error = mask & np.isfinite(errors)
    finite_error_without = ~mask & np.isfinite(errors)
    finite_length = mask & np.isfinite(frame["cr_length"])
    finite_length_without = ~mask & np.isfinite(frame["cr_length"])
    unresolved = (
        ~frame["cr_is_numerically_resolved"].eq(True)
        if "cr_is_numerically_resolved" in frame
        else pd.Series(False, index=frame.index)
    )
    row: dict[str, object] = {
        "estimator": estimator,
        "warning_category": category,
        "warning_count": int(mask.sum()),
        "warning_frequency": float(mask.mean()),
        "warning_event_count": int(events.loc[mask].sum()),
        "warning_events_per_row": float(events.sum() / len(frame)),
        "affected_replications": int(frame.loc[mask, "rep"].nunique()),
        "affected_design_scenarios": int(
            frame.loc[mask].groupby(NATURAL_KEY[:-1]).ngroups
        ),
        "coverage_affected_valid": (
            float(frame.loc[affected_valid, "covered"].astype(float).mean())
            if affected_valid.any()
            else np.nan
        ),
        "coverage_affected_denominator": int(affected_valid.sum()),
        "coverage_without_warning_valid": (
            float(frame.loc[unaffected_valid, "covered"].astype(float).mean())
            if unaffected_valid.any()
            else np.nan
        ),
        "coverage_without_warning_denominator": int(unaffected_valid.sum()),
        "bias": float(errors.loc[finite_error].mean()) if finite_error.any() else np.nan,
        "rmse": (
            float(np.sqrt(np.mean(errors.loc[finite_error] ** 2)))
            if finite_error.any()
            else np.nan
        ),
        "mean_cr_length": (
            float(frame.loc[finite_length, "cr_length"].mean())
            if finite_length.any()
            else np.nan
        ),
        "bias_without_warning": (
            float(errors.loc[finite_error_without].mean())
            if finite_error_without.any()
            else np.nan
        ),
        "rmse_without_warning": (
            float(np.sqrt(np.mean(errors.loc[finite_error_without] ** 2)))
            if finite_error_without.any()
            else np.nan
        ),
        "mean_cr_length_without_warning": (
            float(frame.loc[finite_length_without, "cr_length"].mean())
            if finite_length_without.any()
            else np.nan
        ),
        "empty_cr_count": int((mask & frame.get("cr_status", "").eq("empty_valid")).sum()),
        "unresolved_count": int((mask & unresolved).sum()),
    }
    classification, reason = _warning_classification(row)
    row["classification"] = classification
    row["classification_reason"] = reason
    return row


def warning_summaries(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Summarize stored warning counters overall and by design scenario."""
    overall: list[dict[str, object]] = []
    scenarios: list[dict[str, object]] = []
    availability: dict[str, object] = {}
    for estimator in ("oracle", "post_selection"):
        frame = frames[estimator]
        _, metadata = warning_membership(frame)
        availability[estimator] = metadata
        for category, definition in WARNING_DEFINITIONS.items():
            column = definition["column"]
            if column not in frame:
                continue
            events = (
                pd.to_numeric(frame[column], errors="coerce").fillna(0)
                if definition["kind"] == "count"
                else frame[column].eq(True).astype(int)
            )
            mask = events.gt(0)
            overall.append(_warning_row(frame, estimator, category, mask, events))
            for key, group in frame.groupby(NATURAL_KEY[:-1], sort=True):
                group_events = events.loc[group.index]
                row = _warning_row(
                    group,
                    estimator,
                    category,
                    group_events.gt(0),
                    group_events,
                )
                row.update(dict(zip(NATURAL_KEY[:-1], key, strict=True)))
                scenarios.append(row)
    overall_frame = pd.DataFrame(overall).sort_values(
        ["estimator", "warning_category"], kind="mergesort"
    ).reset_index(drop=True)
    scenario_frame = pd.DataFrame(scenarios).sort_values(
        ["estimator", "warning_category", *NATURAL_KEY[:-1]], kind="mergesort"
    ).reset_index(drop=True)
    taxonomy = {
        "definitions": WARNING_DEFINITIONS,
        "availability": availability,
        "classification_rules": {
            "consequential": "At least one affected row is numerically unresolved.",
            "potentially consequential": (
                "At least one affected row is empty; the category is a stored numerical "
                "barrier/limit/failure; or affected valid coverage is >0.01 below unaffected."
            ),
            "clearly benign": (
                "Affected rows have no empty/unresolved result and do not meet the adverse "
                "coverage or semantic-hazard rules."
            ),
            "unknown from stored information": "No affected rows or no stored category.",
        },
        "limitation": (
            "No textual warning reason is stored. Categories describe counters/flags, not "
            "the underlying solver message or causal mechanism."
        ),
    }
    return overall_frame, scenario_frame, taxonomy


def _invalid_geometry(row: pd.Series) -> bool:
    values = [row.get("cr_lower"), row.get("cr_upper"), row.get("cr_length")]
    finite = all(pd.notna(value) and np.isfinite(float(value)) for value in values)
    if finite and (float(values[0]) > float(values[1]) or float(values[2]) < 0):
        return True
    if "cr_components" in row and pd.notna(row.get("cr_components")):
        try:
            parse_cr_components(row.get("cr_components"))
        except ValueError:
            return True
    return False


def classify_exception(row: pd.Series, estimator: str) -> tuple[str, str]:
    """Conservatively classify an exceptional historical CR row."""
    if _invalid_geometry(row):
        return "invalid_geometry", "Stored bounds, length, or components are invalid."
    if estimator == "dml":
        complete = all(
            pd.notna(row.get(column)) and np.isfinite(float(row.get(column)))
            for column in ("cr_lower", "cr_upper", "cr_length")
        )
        if not complete:
            return (
                "missing_legacy_geometry",
                "One or more legacy DML CR geometry fields are missing; no resolution field exists.",
            )
    resolved_value = row.get("cr_is_numerically_resolved")
    unresolved_value = row.get("cr_unresolved_count", 0)
    unresolved_count = 0 if pd.isna(unresolved_value) else int(unresolved_value)
    if (pd.notna(resolved_value) and not bool(resolved_value)) or unresolved_count > 0:
        return (
            "numerical_non_resolution",
            "Stored resolution is false or the unresolved-alpha count is positive.",
        )
    if row.get("cr_status") == "empty_valid":
        components = parse_cr_components(row.get("cr_components"))
        if components == ():
            return (
                "complete_rejection_across_evaluated_grid",
                "The stored valid empty accepted set has zero components.",
            )
        return "unknown", "The stored status is empty but components are unavailable."
    return "unknown", "Stored evidence does not identify a supported cause."


def exception_diagnostics(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row- and scenario-level empty, unresolved, and missing-geometry diagnostics."""
    output: list[pd.DataFrame] = []
    identifiers = NATURAL_KEY + [
        "seed",
        "alpha_true",
        "alpha_hat",
        "covered",
        "converged",
    ]
    for estimator in ESTIMATORS:
        frame = frames[estimator]
        finite_cr = np.isfinite(frame[["cr_lower", "cr_upper", "cr_length"]]).all(axis=1)
        if estimator == "dml":
            mask = ~finite_cr
            exception_type = pd.Series("missing_legacy_geometry", index=frame.index)
        else:
            empty = frame["cr_status"].eq("empty_valid")
            unresolved = ~frame["cr_is_numerically_resolved"].eq(True)
            mask = empty | unresolved
            exception_type = pd.Series("empty_cr", index=frame.index)
            exception_type.loc[unresolved] = "unresolved_cr"
        selected = frame.loc[mask].copy()
        selected.insert(0, "estimator_name", estimator)
        selected["exception_type"] = exception_type.loc[mask].to_numpy()
        causes = [classify_exception(row, estimator) for _, row in selected.iterrows()]
        selected["cause_classification"] = [value[0] for value in causes]
        selected["cause_evidence"] = [value[1] for value in causes]
        selected["grid_lower_boundary"] = np.nan
        selected["grid_upper_boundary"] = np.nan
        selected["touches_grid_lower"] = pd.NA
        selected["touches_grid_upper"] = pd.NA
        selected["invalid_geometry"] = selected.apply(_invalid_geometry, axis=1)
        for column in EXCEPTION_OPTIONAL_COLUMNS:
            if column not in selected:
                selected[column] = np.nan
        ordered = [
            "estimator_name",
            *identifiers,
            "exception_type",
            "cause_classification",
            "cause_evidence",
            *EXCEPTION_OPTIONAL_COLUMNS,
            "grid_lower_boundary",
            "grid_upper_boundary",
            "touches_grid_lower",
            "touches_grid_upper",
            "invalid_geometry",
        ]
        output.append(selected.loc[:, ordered])
    rows = pd.concat(output, ignore_index=True).sort_values(
        ["estimator_name", *NATURAL_KEY], kind="mergesort"
    ).reset_index(drop=True)
    if rows["invalid_geometry"].any():
        raise ValueError("Exceptional rows contain invalid confidence-region geometry")
    summary_rows: list[dict[str, object]] = []
    scenario_columns = NATURAL_KEY[:-1]
    totals = {
        estimator: frame.groupby(scenario_columns, sort=True).size()
        for estimator, frame in frames.items()
    }
    for key, group in rows.groupby(
        ["estimator_name", *scenario_columns, "exception_type", "cause_classification"],
        sort=True,
    ):
        estimator, *rest = key
        scenario = tuple(rest[: len(scenario_columns)])
        count = len(group)
        summary_rows.append({
            "estimator": estimator,
            **dict(zip(scenario_columns, scenario, strict=True)),
            "exception_type": rest[-2],
            "cause_classification": rest[-1],
            "exception_count": count,
            "scenario_replications": int(totals[estimator].loc[scenario]),
            "exception_frequency": float(count / totals[estimator].loc[scenario]),
            "affected_replications": int(group["rep"].nunique()),
            "covered_true_count": int(group["covered"].eq(True).sum()),
            "converged_count": int(group["converged"].eq(True).sum()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["estimator", *scenario_columns, "exception_type"], kind="mergesort"
    ).reset_index(drop=True)
    return rows, summary


def _paired_frame(
    frames: Mapping[str, pd.DataFrame], estimator_a: str, estimator_b: str
) -> pd.DataFrame:
    left = frames[estimator_a].copy()
    right = frames[estimator_b].copy()
    for name, frame in ((estimator_a, left), (estimator_b, right)):
        if frame.duplicated(NATURAL_KEY).any():
            raise ValueError(f"{name} contains duplicate natural keys")
    left["valid_coverage"] = valid_coverage_mask(left, estimator_a)
    right["valid_coverage"] = valid_coverage_mask(right, estimator_b)
    columns = NATURAL_KEY + [
        "seed",
        "alpha_true",
        "alpha_hat",
        "covered",
        "cr_length",
        "valid_coverage",
    ]
    merged = left[columns].merge(
        right[columns],
        on=NATURAL_KEY,
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"Paired-key mismatch for {estimator_a} and {estimator_b}")
    if merged["seed_a"].ne(merged["seed_b"]).any():
        raise ValueError(f"Seed conflict for {estimator_a} and {estimator_b}")
    if not np.isclose(
        merged["alpha_true_a"], merged["alpha_true_b"], rtol=0, atol=1e-12
    ).all():
        raise ValueError(f"alpha_true conflict for {estimator_a} and {estimator_b}")
    return merged.drop(columns="_merge").sort_values(NATURAL_KEY, kind="mergesort").reset_index(drop=True)


def _paired_statistic(
    group: pd.DataFrame,
    metric: str,
    estimator_a: str,
    estimator_b: str,
) -> dict[str, object]:
    if metric == "coverage":
        valid = group["valid_coverage_a"].eq(True) & group["valid_coverage_b"].eq(True)
        a = group.loc[valid, "covered_a"].astype(bool)
        b = group.loc[valid, "covered_b"].astype(bool)
        differences = a.astype(float) - b.astype(float)
        both = int((a & b).sum())
        only_a = int((a & ~b).sum())
        only_b = int((~a & b).sum())
        neither = int((~a & ~b).sum())
        favorable = differences.gt(0)
    else:
        if metric == "absolute_error":
            a_values = (group["alpha_hat_a"] - group["alpha_true_a"]).abs()
            b_values = (group["alpha_hat_b"] - group["alpha_true_b"]).abs()
        elif metric == "squared_error":
            a_values = (group["alpha_hat_a"] - group["alpha_true_a"]) ** 2
            b_values = (group["alpha_hat_b"] - group["alpha_true_b"]) ** 2
        elif metric == "cr_length":
            a_values = group["cr_length_a"]
            b_values = group["cr_length_b"]
        else:
            raise ValueError(f"Unknown paired metric: {metric}")
        valid = np.isfinite(a_values) & np.isfinite(b_values)
        differences = a_values.loc[valid] - b_values.loc[valid]
        both = only_a = only_b = neither = np.nan
        favorable = differences.lt(0)
    denominator = len(differences)
    mean = float(differences.mean()) if denominator else np.nan
    median = float(differences.median()) if denominator else np.nan
    se = float(differences.std(ddof=1) / sqrt(denominator)) if denominator > 1 else np.nan
    lower = mean - 1.96 * se if np.isfinite(se) else np.nan
    upper = mean + 1.96 * se if np.isfinite(se) else np.nan
    return {
        "estimator_a": estimator_a,
        "estimator_b": estimator_b,
        "metric": metric,
        "difference_orientation": "estimator_a - estimator_b",
        "valid_paired_denominator": denominator,
        "mean_paired_difference": mean,
        "median_paired_difference": median,
        "paired_standard_error": se,
        "paired_ci95_lower": lower,
        "paired_ci95_upper": upper,
        "a_better_count": int(favorable.sum()),
        "a_better_proportion": float(favorable.mean()) if denominator else np.nan,
        "tie_count": int(differences.eq(0).sum()),
        "both_cover": both,
        "only_a_covers": only_a,
        "only_b_covers": only_b,
        "neither_covers": neither,
    }


def paired_comparisons(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute overall and scenario paired differences with explicit A-minus-B orientation."""
    overall: list[dict[str, object]] = []
    scenarios: list[dict[str, object]] = []
    discordance: list[dict[str, object]] = []
    metrics = ("coverage", "absolute_error", "squared_error", "cr_length")
    for estimator_a, estimator_b in PAIR_ORDER:
        paired = _paired_frame(frames, estimator_a, estimator_b)
        for metric in metrics:
            row = _paired_statistic(paired, metric, estimator_a, estimator_b)
            overall.append(row)
            if metric == "coverage":
                discordance.append(_discordance_row(row, "overall", {}))
        for key, group in paired.groupby(NATURAL_KEY[:-1], sort=True):
            design = dict(zip(NATURAL_KEY[:-1], key, strict=True))
            for metric in metrics:
                row = _paired_statistic(group, metric, estimator_a, estimator_b)
                scenarios.append({**design, **row})
                if metric == "coverage":
                    discordance.append(_discordance_row(row, "scenario", design))
    overall_frame = pd.DataFrame(overall).sort_values(
        ["estimator_a", "estimator_b", "metric"], kind="mergesort"
    ).reset_index(drop=True)
    scenario_frame = pd.DataFrame(scenarios).sort_values(
        ["estimator_a", "estimator_b", "metric", *NATURAL_KEY[:-1]],
        kind="mergesort",
    ).reset_index(drop=True)
    discordance_frame = pd.DataFrame(discordance).sort_values(
        ["scope", "estimator_a", "estimator_b", *NATURAL_KEY[:-1]],
        kind="mergesort",
        na_position="first",
    ).reset_index(drop=True)
    return overall_frame, scenario_frame, discordance_frame


def _discordance_row(
    paired: Mapping[str, object], scope: str, design: Mapping[str, object]
) -> dict[str, object]:
    only_a = int(paired["only_a_covers"])
    only_b = int(paired["only_b_covers"])
    discordant = only_a + only_b
    statistic = (
        float((max(abs(only_a - only_b) - 1, 0) ** 2) / discordant)
        if discordant
        else 0.0
    )
    pvalue = (
        float(binomtest(min(only_a, only_b), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "scope": scope,
        **{column: design.get(column, np.nan) for column in NATURAL_KEY[:-1]},
        "estimator_a": paired["estimator_a"],
        "estimator_b": paired["estimator_b"],
        "difference_orientation": "coverage_a - coverage_b",
        "valid_paired_denominator": paired["valid_paired_denominator"],
        "both_cover": paired["both_cover"],
        "only_a_covers": only_a,
        "only_b_covers": only_b,
        "neither_covers": paired["neither_covers"],
        "paired_coverage_difference": paired["mean_paired_difference"],
        "paired_ci95_lower": paired["paired_ci95_lower"],
        "paired_ci95_upper": paired["paired_ci95_upper"],
        "discordant_pairs": discordant,
        "mcnemar_continuity_statistic": statistic,
        "exact_binomial_pvalue": pvalue,
    }


def wilson_interval(successes: int, denominator: int, z: float = 1.96) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if denominator <= 0 or successes < 0 or successes > denominator:
        return np.nan, np.nan
    p = successes / denominator
    denominator_term = 1 + z**2 / denominator
    center = (p + z**2 / (2 * denominator)) / denominator_term
    half = z * sqrt(p * (1 - p) / denominator + z**2 / (4 * denominator**2))
    half /= denominator_term
    return max(0.0, center - half), min(1.0, center + half)


def coverage_uncertainty(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Add Wilson robustness intervals to Phase 1 conditional coverage estimates."""
    rows: list[dict[str, object]] = []
    for row in scenarios.itertuples(index=False):
        denominator = int(row.coverage_denominator)
        successes = int(round(float(row.conditional_coverage) * denominator))
        wilson_lower, wilson_upper = wilson_interval(successes, denominator)
        rows.append({
            **{column: getattr(row, column) for column in SCENARIO_KEY},
            "coverage": float(row.conditional_coverage),
            "coverage_successes": successes,
            "coverage_denominator": denominator,
            "coverage_mcse": float(row.coverage_mcse),
            "bounded_wald95_lower": float(row.coverage_mc95_lower),
            "bounded_wald95_upper": float(row.coverage_mc95_upper),
            "wilson95_lower": wilson_lower,
            "wilson95_upper": wilson_upper,
            "nominal_coverage": NOMINAL_COVERAGE,
            "coverage_gap": float(row.conditional_coverage) - NOMINAL_COVERAGE,
            "wilson_includes_nominal": wilson_lower <= NOMINAL_COVERAGE <= wilson_upper,
        })
    return pd.DataFrame(rows).sort_values(SCENARIO_KEY, kind="mergesort").reset_index(drop=True)


def classification_rules() -> dict[str, object]:
    return {
        "nominal_coverage": NOMINAL_COVERAGE,
        "concerning": (
            "overall coverage gap <= -0.02, severe-undercoverage share >= 0.05, "
            "unresolved rate > 0.01, or empty-CR rate > 0.01"
        ),
        "strong": (
            "absolute overall coverage gap <= 0.01; overall Wilson interval includes 0.95; "
            "zero severe scenarios; unresolved and empty rates <= 0.001; relative RMSE <= 1.15; "
            "relative mean CR length <= 1.20; and rich CR diagnostics"
        ),
        "acceptable with caveats": "neither concerning nor fully strong",
        "insufficiently diagnosable": (
            "coverage itself or the fields needed to apply performance rules are unavailable"
        ),
        "severe_scenario": "coverage gap <= -0.10",
        "diagnostic_confidence": {
            "high": "CR status, numerical resolution, components, blocks, and warnings stored",
            "limited": "one or more rich CR diagnostic families absent",
        },
        "no_composite_score": True,
    }


def classify_estimators(
    estimator_summary: pd.DataFrame,
    scenarios: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply disclosed non-compensatory classification rules."""
    minimum_rmse = float(estimator_summary["rmse"].min())
    minimum_length = float(estimator_summary["mean_cr_length"].min())
    rows: list[dict[str, object]] = []
    rich_fields = {
        "cr_status",
        "cr_components",
        "cr_n_blocks",
        "cr_is_numerically_resolved",
        "iteration_warning_evaluations",
    }
    for summary in estimator_summary.itertuples(index=False):
        estimator = summary.estimator
        estimator_scenarios = scenarios[scenarios["estimator"].eq(estimator)]
        severe_count = int(estimator_scenarios["coverage_gap"].le(-0.10).sum())
        severe_share = severe_count / len(estimator_scenarios)
        successes = int(round(summary.conditional_coverage * summary.coverage_denominator))
        wilson_lower, wilson_upper = wilson_interval(successes, int(summary.coverage_denominator))
        gap = float(summary.conditional_coverage - NOMINAL_COVERAGE)
        unresolved_rate = (
            float(summary.unresolved_rate) if pd.notna(summary.unresolved_rate) else np.nan
        )
        empty_rate = (
            float(summary.empty_valid_rate) if pd.notna(summary.empty_valid_rate) else np.nan
        )
        relative_rmse = float(summary.rmse / minimum_rmse)
        relative_length = float(summary.mean_cr_length / minimum_length)
        confidence = "high" if rich_fields.issubset(frames[estimator].columns) else "limited"
        reasons: list[str] = []
        if not np.isfinite(summary.conditional_coverage):
            classification = "insufficiently diagnosable"
            reasons.append("Conditional coverage is unavailable.")
        else:
            concerning_reasons = []
            if gap <= -0.02:
                concerning_reasons.append(f"overall coverage gap {gap:.4f} <= -0.02")
            if severe_share >= 0.05:
                concerning_reasons.append(f"severe-scenario share {severe_share:.4f} >= 0.05")
            if np.isfinite(unresolved_rate) and unresolved_rate > 0.01:
                concerning_reasons.append(f"unresolved rate {unresolved_rate:.4f} > 0.01")
            if np.isfinite(empty_rate) and empty_rate > 0.01:
                concerning_reasons.append(f"empty-CR rate {empty_rate:.4f} > 0.01")
            strong_conditions = {
                "absolute coverage gap <= 0.01": abs(gap) <= 0.01,
                "Wilson interval includes 0.95": wilson_lower <= 0.95 <= wilson_upper,
                "zero severe scenarios": severe_count == 0,
                "unresolved rate <= 0.001 or unavailable": (
                    not np.isfinite(unresolved_rate) or unresolved_rate <= 0.001
                ),
                "empty rate <= 0.001 or unavailable": (
                    not np.isfinite(empty_rate) or empty_rate <= 0.001
                ),
                "relative RMSE <= 1.15": relative_rmse <= 1.15,
                "relative CR length <= 1.20": relative_length <= 1.20,
                "rich diagnostic schema": confidence == "high",
            }
            if concerning_reasons:
                classification = "concerning"
                reasons.extend(concerning_reasons)
            elif all(strong_conditions.values()):
                classification = "strong"
                reasons.append("All disclosed strong-class conditions are met.")
            else:
                classification = "acceptable with caveats"
                reasons.extend(
                    f"failed: {name}"
                    for name, passed in strong_conditions.items()
                    if not passed
                )
        rows.append({
            "estimator": estimator,
            "classification": classification,
            "diagnostic_confidence": confidence,
            "machine_readable_reasons": json.dumps(reasons, separators=(",", ":")),
            "conditional_coverage": summary.conditional_coverage,
            "coverage_denominator": int(summary.coverage_denominator),
            "coverage_gap": gap,
            "overall_wilson95_lower": wilson_lower,
            "overall_wilson95_upper": wilson_upper,
            "wilson_includes_nominal": wilson_lower <= 0.95 <= wilson_upper,
            "severe_scenario_count": severe_count,
            "severe_scenario_share": severe_share,
            "unresolved_rate": unresolved_rate,
            "empty_cr_rate": empty_rate,
            "rmse": summary.rmse,
            "relative_rmse": relative_rmse,
            "mean_cr_length": summary.mean_cr_length,
            "relative_mean_cr_length": relative_length,
        })
    return (
        pd.DataFrame(rows).sort_values("estimator", kind="mergesort").reset_index(drop=True),
        classification_rules(),
    )


__all__ = [
    "EXCEPTION_OPTIONAL_COLUMNS",
    "PAIR_ORDER",
    "WARNING_DEFINITIONS",
    "classify_estimators",
    "classify_exception",
    "classification_rules",
    "coverage_uncertainty",
    "exception_diagnostics",
    "paired_comparisons",
    "valid_coverage_mask",
    "warning_membership",
    "warning_summaries",
    "wilson_interval",
]
