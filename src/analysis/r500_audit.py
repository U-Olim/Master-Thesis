"""Deterministic scientific audit helpers for immutable historical R=500 results."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ivqr.confidence_regions import parse_cr_components
from simulation.output_validation import validate_component_columns


ESTIMATORS = ("oracle", "post_selection", "dml")
NATURAL_KEY = ["dgp", "n", "p", "pi", "tau", "rep"]
SCENARIO_KEY = ["estimator", "dgp", "n", "p", "pi", "tau"]
EXPECTED_VALUES: dict[str, tuple[object, ...]] = {
    "dgp": ("dgp1", "dgp2", "dgp3"),
    "n": (500, 1000),
    "p": (200, 500),
    "pi": (0.10, 0.25, 0.50, 1.0),
    "tau": (0.25, 0.50, 0.75),
}
NOMINAL_COVERAGE = 0.95
ALPHA_GRID_MIN = -1.0
ALPHA_GRID_MAX = 3.0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_metadata(
    path: str | Path, *, display_root: str | Path | None = None
) -> dict[str, object]:
    source = Path(path)
    stat = source.stat()
    if source.suffix.lower() == ".csv":
        header = pd.read_csv(source, nrows=0)
        with source.open("r", encoding="utf-8") as handle:
            rows: int | None = sum(1 for _ in handle) - 1
        column_count: int | None = len(header.columns)
        columns = list(header.columns)
    else:
        rows = None
        column_count = None
        columns = []
    display_path = source
    if display_root is not None:
        display_path = source.resolve().relative_to(Path(display_root).resolve())
    return {
        "path": display_path.as_posix(),
        "sha256": sha256_file(source),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "row_count": rows,
        "column_count": column_count,
        "columns": columns,
    }


def validate_structure(
    frame: pd.DataFrame,
    estimator: str,
    *,
    expected_replications: int = 500,
    expected_values: Mapping[str, tuple[object, ...]] = EXPECTED_VALUES,
) -> dict[str, object]:
    missing = set(NATURAL_KEY + ["seed", "alpha_true"]).difference(frame.columns)
    if missing:
        raise ValueError(f"{estimator} is missing structural columns: {sorted(missing)}")
    if frame.duplicated(NATURAL_KEY).any():
        raise ValueError(f"{estimator} contains duplicate natural keys")
    if frame.duplicated().any():
        raise ValueError(f"{estimator} contains duplicated rows")
    for column, expected in expected_values.items():
        observed = tuple(sorted(frame[column].dropna().unique().tolist()))
        if observed != tuple(sorted(expected)):
            raise ValueError(
                f"{estimator} has unexpected {column} values: {observed}; "
                f"expected {tuple(sorted(expected))}"
            )
    scenario_counts = frame.groupby(NATURAL_KEY[:-1], sort=False)["rep"].agg(
        ["size", "nunique", "min", "max"]
    )
    expected_max = expected_replications - 1
    incomplete = scenario_counts[
        (scenario_counts["size"] != expected_replications)
        | (scenario_counts["nunique"] != expected_replications)
        | (scenario_counts["min"] != 0)
        | (scenario_counts["max"] != expected_max)
    ]
    if not incomplete.empty:
        raise ValueError(f"{estimator} has {len(incomplete)} incomplete design cells")
    expected_rows = int(np.prod([len(values) for values in expected_values.values()]))
    expected_rows *= expected_replications
    if len(frame) != expected_rows:
        raise ValueError(
            f"{estimator} row count is {len(frame)}, expected {expected_rows}"
        )
    canonical = frame.sort_values(NATURAL_KEY, kind="mergesort").index
    sorted_naturally = canonical.equals(pd.RangeIndex(len(frame)))
    labels = set(frame.get("estimator", pd.Series(dtype=object)).dropna().astype(str))
    expected_labels = {
        "oracle": {"oracle"},
        "post_selection": {"post_selection_ivqr"},
        "dml": {"dml_ivqr"},
    }
    if labels and labels != expected_labels[estimator]:
        raise ValueError(f"{estimator} has unexpected estimator labels: {labels}")
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "unique_replications": int(frame["rep"].nunique()),
        "replication_min": int(frame["rep"].min()),
        "replication_max": int(frame["rep"].max()),
        "rows_per_replication": sorted(
            int(value) for value in frame.groupby("rep").size().unique()
        ),
        "design_cells": int(frame.groupby(NATURAL_KEY[:-1]).ngroups),
        "duplicate_natural_keys": 0,
        "naturally_sorted": sorted_naturally,
    }


def validate_alignment(frames: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    key_sets = {
        estimator: set(frame[NATURAL_KEY].itertuples(index=False, name=None))
        for estimator, frame in frames.items()
    }
    reference = key_sets["oracle"]
    differences: dict[str, int] = {}
    for estimator in ESTIMATORS:
        differences[f"only_in_{estimator}"] = len(key_sets[estimator] - set.union(
            *(key_sets[other] for other in ESTIMATORS if other != estimator)
        ))
        if key_sets[estimator] != reference:
            raise ValueError(f"Cross-estimator natural-key mismatch for {estimator}")
    merged = frames["oracle"][NATURAL_KEY + ["seed", "alpha_true"]].copy()
    merged = merged.rename(columns={"seed": "seed_oracle", "alpha_true": "alpha_oracle"})
    for estimator in ("post_selection", "dml"):
        selected = frames[estimator][NATURAL_KEY + ["seed", "alpha_true"]].rename(
            columns={"seed": f"seed_{estimator}", "alpha_true": f"alpha_{estimator}"}
        )
        merged = merged.merge(selected, on=NATURAL_KEY, validate="one_to_one")
    seed_conflicts = (
        (merged["seed_oracle"] != merged["seed_post_selection"])
        | (merged["seed_oracle"] != merged["seed_dml"])
    )
    alpha_conflicts = (
        ~np.isclose(merged["alpha_oracle"], merged["alpha_post_selection"], rtol=0, atol=1e-12)
        | ~np.isclose(merged["alpha_oracle"], merged["alpha_dml"], rtol=0, atol=1e-12)
    )
    if seed_conflicts.any():
        raise ValueError(f"Seed mismatch in {int(seed_conflicts.sum())} aligned rows")
    if alpha_conflicts.any():
        raise ValueError(f"alpha_true mismatch in {int(alpha_conflicts.sum())} aligned rows")
    return {
        "aligned_keys": len(reference),
        **differences,
        "seed_conflicts": 0,
        "alpha_true_conflicts": 0,
    }


def validate_cr_components_frame(frame: pd.DataFrame, estimator: str) -> dict[str, int]:
    if "cr_components" not in frame:
        return {"rows_checked": 0, "coverage_component_mismatches": 0}
    validate_component_columns(frame)
    mismatches = 0
    rows_checked = 0
    for row in frame.itertuples(index=False):
        components = parse_cr_components(getattr(row, "cr_components"))
        if components is None:
            continue
        rows_checked += 1
        alpha_true = float(getattr(row, "alpha_true"))
        expected = any(lower <= alpha_true <= upper for lower, upper in components)
        resolved = bool(getattr(row, "cr_is_numerically_resolved"))
        actual = bool(getattr(row, "covered"))
        if resolved and expected != actual:
            mismatches += 1
    if mismatches:
        raise ValueError(
            f"{estimator} has {mismatches} component-based coverage mismatches"
        )
    return {"rows_checked": rows_checked, "coverage_component_mismatches": 0}


def validate_result_values(frame: pd.DataFrame, estimator: str) -> None:
    """Reject impossible common result values before descriptive analysis."""
    required = {"cr_lower", "cr_upper", "cr_length", "covered"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{estimator} is missing result columns: {sorted(missing)}")
    finite = np.isfinite(frame[["cr_lower", "cr_upper", "cr_length"]]).all(axis=1)
    if (finite & frame["cr_length"].lt(0)).any():
        raise ValueError(f"{estimator} contains negative CR lengths")
    if (finite & frame["cr_lower"].gt(frame["cr_upper"])).any():
        raise ValueError(f"{estimator} contains reversed CR bounds")
    present = frame["covered"].dropna()
    if not present.isin([True, False]).all():
        raise ValueError(f"{estimator} contains non-Boolean coverage values")
    covered_outside_hull = (
        finite
        & frame["covered"].eq(True)
        & ~frame["cr_lower"].le(frame["alpha_true"])
        | finite
        & frame["covered"].eq(True)
        & ~frame["alpha_true"].le(frame["cr_upper"])
    )
    if covered_outside_hull.any():
        raise ValueError(f"{estimator} has covered=True outside finite CR hulls")


def harmonize_frames(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    optional = (
        "cr_status", "cr_n_blocks", "cr_disconnected",
        "cr_is_numerically_resolved", "cr_unresolved_count",
        "iteration_warning_evaluations", "n_selected_controls",
        "selection_lasso_multiplier", "n_retained_instruments", "selection_method",
    )
    harmonized: list[pd.DataFrame] = []
    for estimator in ESTIMATORS:
        frame = frames[estimator].copy()
        frame["estimator"] = estimator
        for column in optional:
            if column not in frame:
                frame[column] = np.nan
        frame["error"] = frame["alpha_hat"] - frame["alpha_true"]
        frame["absolute_error"] = frame["error"].abs()
        frame["squared_error"] = frame["error"] ** 2
        frame["finite_alpha_hat"] = np.isfinite(frame["alpha_hat"])
        frame["finite_cr_length"] = np.isfinite(frame["cr_length"])
        complete_cr = np.isfinite(frame[["cr_lower", "cr_upper", "cr_length"]]).all(axis=1)
        if estimator in {"oracle", "post_selection"}:
            resolved = frame["cr_is_numerically_resolved"].eq(True)
        else:
            resolved = pd.Series(True, index=frame.index)
        frame["valid_coverage_observation"] = (
            frame["converged"].eq(True)
            & complete_cr
            & resolved
            & frame["covered"].notna()
        )
        harmonized.append(frame)
    result = pd.concat(harmonized, ignore_index=True, sort=False)
    return result.sort_values([*SCENARIO_KEY, "rep"], kind="mergesort").reset_index(drop=True)


def monte_carlo_interval(coverage: float, denominator: int) -> tuple[float, float, float]:
    if denominator <= 0 or not np.isfinite(coverage):
        return np.nan, np.nan, np.nan
    se = float(np.sqrt(coverage * (1.0 - coverage) / denominator))
    return se, max(0.0, coverage - 1.96 * se), min(1.0, coverage + 1.96 * se)


def classify_coverage(coverage: float, lower: float, upper: float) -> str:
    if not np.isfinite(coverage):
        return "no valid denominator"
    if lower <= NOMINAL_COVERAGE <= upper:
        return "within Monte Carlo uncertainty"
    gap = coverage - NOMINAL_COVERAGE
    if gap <= -0.10:
        return "severe undercoverage"
    if gap < 0:
        return "moderate undercoverage"
    return "overcoverage"


def _summary_row(group: pd.DataFrame) -> dict[str, object]:
    errors = group.loc[np.isfinite(group["error"]), "error"]
    lengths = group.loc[group["finite_cr_length"], "cr_length"]
    covered_present = group["covered"].notna()
    conditional = group["valid_coverage_observation"]
    unconditional_coverage = (
        float(group.loc[covered_present, "covered"].astype(float).mean())
        if covered_present.any() else np.nan
    )
    conditional_coverage = (
        float(group.loc[conditional, "covered"].astype(float).mean())
        if conditional.any() else np.nan
    )
    row: dict[str, object] = {
        "observations": len(group),
        "converged_count": int(group["converged"].eq(True).sum()),
        "convergence_rate": float(group["converged"].eq(True).mean()),
        "nonconverged_count": int(group["converged"].ne(True).sum()),
        "nonconvergence_rate": float(group["converged"].ne(True).mean()),
        "bias": float(errors.mean()) if len(errors) else np.nan,
        "median_bias": float(errors.median()) if len(errors) else np.nan,
        "absolute_bias": float(abs(errors.mean())) if len(errors) else np.nan,
        "median_absolute_error": float(errors.abs().median()) if len(errors) else np.nan,
        "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
        "error_std": float(errors.std(ddof=1)) if len(errors) > 1 else np.nan,
        "unconditional_coverage": unconditional_coverage,
        "unconditional_coverage_denominator": int(covered_present.sum()),
        "conditional_coverage": conditional_coverage,
        "coverage_denominator": int(conditional.sum()),
        "conditional_excluded_rows": int(len(group) - conditional.sum()),
        "missing_coverage_rows": int(group["covered"].isna().sum()),
        "mean_cr_length": float(lengths.mean()) if len(lengths) else np.nan,
        "median_cr_length": float(lengths.median()) if len(lengths) else np.nan,
        "cr_length_std": float(lengths.std(ddof=1)) if len(lengths) > 1 else np.nan,
        "min_cr_length": float(lengths.min()) if len(lengths) else np.nan,
        "max_cr_length": float(lengths.max()) if len(lengths) else np.nan,
        "missing_estimate_rate": float((~group["finite_alpha_hat"]).mean()),
        "missing_cr_rate": float((~group["finite_cr_length"]).mean()),
    }
    if group["cr_is_numerically_resolved"].notna().any():
        resolved = group["cr_is_numerically_resolved"].eq(True)
        row.update(
            unresolved_rows=int((~resolved).sum()),
            numerically_resolved_rate=float(resolved.mean()),
            unresolved_rate=float((~resolved).mean()),
            full_grid_rate=float(group["cr_status"].eq("full_grid_valid").mean()),
            empty_valid_rate=float(group["cr_status"].eq("empty_valid").mean()),
            disconnected_rate=float(group["cr_disconnected"].eq(True).mean()),
            mean_cr_blocks=float(pd.to_numeric(group["cr_n_blocks"]).mean()),
            iteration_warning_rate=float(
                pd.to_numeric(group["iteration_warning_evaluations"]).gt(0).mean()
            ),
            mean_warning_evaluations=float(
                pd.to_numeric(group["iteration_warning_evaluations"]).mean()
            ),
        )
    else:
        row["unresolved_rows"] = np.nan
        for column in (
            "numerically_resolved_rate", "unresolved_rate", "full_grid_rate",
            "empty_valid_rate", "disconnected_rate", "mean_cr_blocks",
            "iteration_warning_rate", "mean_warning_evaluations",
        ):
            row[column] = np.nan
    if group["n_selected_controls"].notna().any():
        selected = pd.to_numeric(group["n_selected_controls"])
        retained = pd.to_numeric(group["n_retained_instruments"])
        row.update(
            mean_selected_controls=float(selected.mean()),
            median_selected_controls=float(selected.median()),
            min_selected_controls=float(selected.min()),
            max_selected_controls=float(selected.max()),
            mean_retained_instruments=float(retained.mean()),
        )
    else:
        for column in (
            "mean_selected_controls", "median_selected_controls",
            "min_selected_controls", "max_selected_controls",
            "mean_retained_instruments",
        ):
            row[column] = np.nan
    return row


def summarize_estimator(harmonized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for estimator, group in harmonized.groupby("estimator", sort=True):
        row = {"estimator": estimator, **_summary_row(group)}
        multipliers = group["selection_lasso_multiplier"].dropna().value_counts().sort_index()
        row["selection_multiplier_values"] = ";".join(
            f"{float(value):g}:{int(count)}" for value, count in multipliers.items()
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("estimator").reset_index(drop=True)


def summarize_scenarios(harmonized: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in harmonized.groupby(SCENARIO_KEY, sort=True):
        row = dict(zip(SCENARIO_KEY, key, strict=True))
        row.update(_summary_row(group))
        row["replications"] = len(group)
        se, lower, upper = monte_carlo_interval(
            float(row["conditional_coverage"]), int(row["coverage_denominator"])
        )
        row.update(
            coverage_mcse=se,
            coverage_mc95_lower=lower,
            coverage_mc95_upper=upper,
            coverage_gap=float(row["conditional_coverage"]) - NOMINAL_COVERAGE,
            coverage_assessment=classify_coverage(
                float(row["conditional_coverage"]), lower, upper
            ),
        )
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(SCENARIO_KEY, kind="mergesort")
    return result.reset_index(drop=True)


def worst_scenarios(scenarios: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    specifications = (
        ("coverage_gap", True),
        ("rmse", False),
        ("mean_cr_length", False),
        ("nonconvergence_rate", False),
        ("unresolved_rate", False),
    )
    rows: list[pd.DataFrame] = []
    for estimator, group in scenarios.groupby("estimator", sort=True):
        for metric, ascending in specifications:
            available = group.dropna(subset=[metric]).sort_values(
                [metric, "dgp", "n", "p", "pi", "tau"],
                ascending=[ascending, True, True, True, True, True],
                kind="mergesort",
            ).head(top_n).copy()
            available["metric"] = metric
            available["value"] = available[metric]
            available["rank"] = range(1, len(available) + 1)
            rows.append(available)
    result = pd.concat(rows, ignore_index=True)
    columns = [
        "estimator", "dgp", "n", "p", "pi", "tau", "metric", "value",
        "rank", "coverage_denominator", "coverage_mcse",
    ]
    return result.loc[:, columns]


def monotonicity_checks(scenarios: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = (
        ("sample_size", "n", 500, 1000, ["estimator", "dgp", "p", "pi", "tau"]),
        ("dimension", "p", 200, 500, ["estimator", "dgp", "n", "pi", "tau"]),
    )
    metrics = (
        "rmse",
        "absolute_bias",
        "mean_cr_length",
        "conditional_coverage",
        "nonconvergence_rate",
        "unresolved_rate",
        "full_grid_rate",
    )
    for comparison, column, lower_value, upper_value, keys in specs:
        lower = scenarios[scenarios[column] == lower_value]
        upper = scenarios[scenarios[column] == upper_value]
        merged = lower.merge(upper, on=keys, suffixes=("_lower", "_upper"), validate="one_to_one")
        for row in merged.itertuples(index=False):
            base = {key: getattr(row, key) for key in keys}
            for metric in metrics:
                low = getattr(row, f"{metric}_lower")
                high = getattr(row, f"{metric}_upper")
                if not np.isfinite(low) or not np.isfinite(high):
                    continue
                expected = "increase" if comparison == "dimension" and metric != "conditional_coverage" else "decrease"
                if metric == "conditional_coverage":
                    expected = "increase" if comparison == "sample_size" else "decrease"
                delta = high - low
                follows = (delta >= 0) if expected == "increase" else (delta <= 0)
                rows.append({
                    "comparison": comparison, **base, "metric": metric,
                    "lower_value": lower_value, "upper_value": upper_value,
                    "lower_metric": low, "upper_metric": high, "delta": delta,
                    "expected_direction": expected, "follows_expectation": follows,
                })
    strength_keys = ["estimator", "dgp", "n", "p", "tau"]
    for key, group in scenarios.groupby(strength_keys, sort=True):
        ordered = group.sort_values("pi")
        for left, right in zip(ordered.iloc[:-1].itertuples(), ordered.iloc[1:].itertuples(), strict=True):
            for metric in metrics:
                expected = "increase" if metric == "conditional_coverage" else "decrease"
                low, high = getattr(left, metric), getattr(right, metric)
                if not np.isfinite(low) or not np.isfinite(high):
                    continue
                delta = high - low
                follows = (delta >= 0) if expected == "increase" else (delta <= 0)
                rows.append({
                    "comparison": "instrument_strength",
                    **dict(zip(strength_keys, key, strict=True)),
                    "metric": metric, "lower_value": left.pi,
                    "upper_value": right.pi, "lower_metric": low,
                    "upper_metric": high, "delta": delta,
                    "expected_direction": expected, "follows_expectation": follows,
                })
    dimension = scenarios.pivot_table(
        index=["estimator", "dgp", "n", "pi", "tau"],
        columns="p",
        values="mean_selected_controls",
        dropna=True,
    )
    if 200 in dimension and 500 in dimension:
        for key, row in dimension.dropna(subset=[200, 500]).iterrows():
            low, high = float(row[200]), float(row[500])
            rows.append({
                "comparison": "dimension",
                **dict(zip(["estimator", "dgp", "n", "pi", "tau"], key, strict=True)),
                "metric": "mean_selected_controls",
                "lower_value": 200,
                "upper_value": 500,
                "lower_metric": low,
                "upper_metric": high,
                "delta": high - low,
                "expected_direction": "increase",
                "follows_expectation": high >= low,
            })
    return pd.DataFrame(rows).sort_values(
        ["comparison", "estimator", "metric", "dgp"], kind="mergesort"
    ).reset_index(drop=True)


def comparison_checks(scenarios: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "conditional_coverage", "absolute_bias", "rmse", "mean_cr_length",
        "convergence_rate", "unresolved_rate",
    )
    rows = []
    pairs = (("oracle", "post_selection"), ("oracle", "dml"), ("post_selection", "dml"))
    keys = SCENARIO_KEY[1:]
    for left_name, right_name in pairs:
        left = scenarios[scenarios.estimator == left_name]
        right = scenarios[scenarios.estimator == right_name]
        merged = left.merge(right, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
        for row in merged.itertuples(index=False):
            for metric in metrics:
                left_value = getattr(row, f"{metric}_left")
                right_value = getattr(row, f"{metric}_right")
                rows.append({
                    **{key: getattr(row, key) for key in keys},
                    "left_estimator": left_name, "right_estimator": right_name,
                    "metric": metric, "left_value": left_value,
                    "right_value": right_value, "difference_right_minus_left": right_value - left_value,
                })
    return pd.DataFrame(rows).sort_values(
        ["left_estimator", "right_estimator", "metric", *keys], kind="mergesort"
    ).reset_index(drop=True)


def suspicious_patterns(frames: Mapping[str, pd.DataFrame], scenarios: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    for estimator, frame in frames.items():
        finite_cr = frame[["cr_lower", "cr_upper", "cr_length"]].notna().all(axis=1)
        covered_invalid = frame["covered"].notna() & ~frame["covered"].isin([True, False])
        hull_expected = (
            frame["cr_lower"].le(frame["alpha_true"])
            & frame["alpha_true"].le(frame["cr_upper"])
        )
        connected = frame.get("cr_n_blocks", pd.Series(np.nan, index=frame.index)).eq(1)
        covered = frame["covered"].eq(True)
        not_covered = frame["covered"].eq(False)
        coverage_mismatch = (
            finite_cr
            & ((covered & ~hull_expected) | (connected & not_covered & hull_expected))
        )
        if "cr_disconnected" in frame and "cr_n_blocks" in frame:
            disconnected_mismatch = (
                frame["cr_disconnected"].notna()
                & frame["cr_n_blocks"].notna()
                & frame["cr_disconnected"].astype(bool).ne(frame["cr_n_blocks"].gt(1))
            )
        else:
            disconnected_mismatch = pd.Series(False, index=frame.index)
        naturally_sorted = frame.sort_values(NATURAL_KEY, kind="mergesort").index.equals(
            pd.RangeIndex(len(frame))
        )
        definitions = {
            "impossible_coverage": int(covered_invalid.sum()),
            "negative_cr_length": int((frame["cr_length"] < 0).sum()),
            "reversed_cr_bounds": int((frame["cr_lower"] > frame["cr_upper"]).sum()),
            "connected_coverage_inconsistency": int(coverage_mismatch.sum()),
            "alpha_hat_outside_search_grid": int(((frame["alpha_hat"] < ALPHA_GRID_MIN) | (frame["alpha_hat"] > ALPHA_GRID_MAX)).sum()),
            "missing_estimate_when_converged": int((frame["converged"].eq(True) & ~np.isfinite(frame["alpha_hat"])).sum()),
            "finite_estimate_when_not_converged": int((frame["converged"].ne(True) & np.isfinite(frame["alpha_hat"])).sum()),
            "zero_length_nonempty_cr": int((finite_cr & frame["cr_length"].eq(0)).sum()),
            "disconnected_block_count_inconsistency": int(disconnected_mismatch.sum()),
            "extreme_absolute_error_over_grid_width": int(((frame["alpha_hat"] - frame["alpha_true"]).abs() > (ALPHA_GRID_MAX - ALPHA_GRID_MIN)).sum()),
            "not_natural_key_sorted": int(not naturally_sorted),
        }
        if "selection_lasso_multiplier" in frame:
            definitions["unexpected_selection_multiplier_variation"] = int(
                frame["selection_lasso_multiplier"].dropna().nunique() != 1
            )
        for check, count in definitions.items():
            checks.append({"estimator": estimator, "check": check, "count": count, "status": "flagged" if count else "passed"})
    no_denominator = scenarios["coverage_denominator"].eq(0).groupby(scenarios["estimator"]).sum()
    all_full = scenarios["full_grid_rate"].eq(1.0).groupby(scenarios["estimator"]).sum()
    for estimator in ESTIMATORS:
        checks.extend([
            {"estimator": estimator, "check": "scenarios_without_coverage_denominator", "count": int(no_denominator.get(estimator, 0)), "status": "flagged" if no_denominator.get(estimator, 0) else "passed"},
            {"estimator": estimator, "check": "scenarios_all_full_grid", "count": int(all_full.get(estimator, 0)), "status": "flagged" if all_full.get(estimator, 0) else "passed"},
        ])
    return pd.DataFrame(checks).sort_values(["estimator", "check"]).reset_index(drop=True)


def defensibility(summary: pd.DataFrame, scenarios: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in summary.itertuples(index=False):
        estimator_scenarios = scenarios[scenarios.estimator == row.estimator]
        severe_rate = float(estimator_scenarios.coverage_assessment.eq("severe undercoverage").mean())
        if row.conditional_coverage >= 0.93 and row.convergence_rate >= 0.99 and severe_rate <= 0.10:
            label = "strong"
        elif row.conditional_coverage >= 0.85 and row.convergence_rate >= 0.95:
            label = "acceptable with caveats"
        elif row.conditional_coverage >= 0.70 and row.convergence_rate >= 0.80:
            label = "problematic"
        else:
            label = "not defensible"
        result[row.estimator] = label
    return result


__all__ = [
    "EXPECTED_VALUES", "NATURAL_KEY", "NOMINAL_COVERAGE", "SCENARIO_KEY",
    "artifact_metadata", "classify_coverage", "comparison_checks", "defensibility",
    "harmonize_frames", "monotonicity_checks", "monte_carlo_interval",
    "summarize_estimator", "summarize_scenarios", "suspicious_patterns",
    "validate_alignment", "validate_cr_components_frame", "validate_structure",
    "validate_result_values", "worst_scenarios",
]
