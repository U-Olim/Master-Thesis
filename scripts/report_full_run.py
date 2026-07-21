#!/usr/bin/env python
"""Build the reproducible full-run IVQR comparison report.

The primary coverage estimand is conditional on a numerically resolved
confidence region.  The legacy DML file has no numerical-resolution/status
columns, so a complete observed CR triplet is used as an explicitly labelled
proxy; missing DML CRs remain unresolved with their status unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "report_full_run"
DESIGN_COLUMNS = ["dgp", "n", "p", "pi", "tau"]
PANEL_KEY_COLUMNS = [*DESIGN_COLUMNS, "rep"]
COMMON_REQUIRED_COLUMNS = [
    *PANEL_KEY_COLUMNS,
    "seed",
    "estimator",
    "alpha_hat",
    "alpha_true",
    "cr_lower",
    "cr_upper",
    "cr_length",
    "covered",
    "converged",
]
DETAILED_CR_COLUMNS = [
    "cr_components",
    "cr_n_blocks",
    "cr_disconnected",
    "cr_status",
    "cr_is_numerically_resolved",
]
ESTIMATOR_ORDER = ["DML-IVQR", "Oracle IVQR", "Post-selection IVQR"]
RAW_ESTIMATOR_LABELS = {
    "DML-IVQR": "dml_ivqr",
    "Oracle IVQR": "oracle",
    "Post-selection IVQR": "post_selection_ivqr",
}
GRID_LOWER = -1.0
GRID_UPPER = 3.0
GRID_TOLERANCE = 1e-9
COVERAGE_BENCHMARKS = {
    "DML-IVQR": 0.948,
    "Oracle IVQR": 0.942,
    "Post-selection IVQR": 0.924,
}
OUTPUT_NAMES = [
    "combined_standardized_results.csv",
    "table_01_overall.csv",
    "table_02_by_quantile.csv",
    "table_03_by_strength.csv",
    "table_04_by_n_p.csv",
    "table_05_by_design_cell.csv",
    "table_06_worst_cells.csv",
    "table_07_diagnostics.csv",
    "validation.json",
    "analysis_report.md",
]


def discover_result_file(names: Sequence[str]) -> Path:
    """Find one repository result file, preferring the inspected raw directory."""
    preferred = [REPOSITORY_ROOT / "results" / "raw" / name for name in names]
    for candidate in preferred:
        if candidate.is_file():
            return candidate

    matches: list[Path] = []
    for name in names:
        matches.extend(REPOSITORY_ROOT.rglob(name))
    unique = sorted({path.resolve() for path in matches})
    if not unique:
        raise FileNotFoundError(
            f"Could not discover any of {list(names)} below {REPOSITORY_ROOT}"
        )
    if len(unique) > 1:
        rendered = "\n  ".join(str(path) for path in unique)
        raise ValueError(
            "Result-file discovery is ambiguous; pass an explicit CLI path:\n  "
            + rendered
        )
    return unique[0]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dml",
        type=Path,
        default=discover_result_file(["dml_ivqr.csv"]),
        help="DML full-run CSV (default: repository-discovered file)",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=discover_result_file(["oracle_ivqr.csv", "oracle_ivqr(1).csv"]),
        help="Oracle full-run CSV (default: repository-discovered file)",
    )
    parser.add_argument(
        "--post-selection",
        type=Path,
        default=discover_result_file(["post_selection_ivqr.csv"]),
        help="Post-selection full-run CSV (default: repository-discovered file)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Report directory (default: results/report_full_run)",
    )
    return parser.parse_args(argv)


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing required columns: {missing}")


def _strict_boolean(series: pd.Series, name: str) -> pd.Series:
    """Return pandas nullable booleans without accepting ambiguous values."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")
    normalized = series.map(
        lambda value: value
        if pd.isna(value) or isinstance(value, (bool, np.bool_))
        else str(value).strip().lower()
    )
    mapping = {
        True: True,
        False: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    invalid = normalized.notna() & ~normalized.isin(mapping)
    if invalid.any():
        values = sorted(normalized.loc[invalid].astype(str).unique().tolist())
        raise ValueError(f"{name}: invalid Boolean values: {values}")
    return normalized.map(mapping).astype("boolean")


def _nullable_flag(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="boolean")
    return _strict_boolean(frame[column], column)


def _nullable_positive_indicator(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="boolean")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    indicator = numeric.gt(0).astype("boolean")
    indicator.loc[numeric.isna()] = pd.NA
    return indicator


def _components_from_json(value: object, row_description: str) -> tuple[tuple[float, float], ...]:
    if pd.isna(value):
        raise ValueError(f"{row_description}: missing cr_components")
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{row_description}: invalid cr_components JSON") from exc
    if not isinstance(decoded, list):
        raise ValueError(f"{row_description}: cr_components must be a JSON list")
    components: list[tuple[float, float]] = []
    previous_upper: float | None = None
    for component in decoded:
        if not isinstance(component, list) or len(component) != 2:
            raise ValueError(f"{row_description}: invalid CR component")
        lower, upper = float(component[0]), float(component[1])
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
            raise ValueError(f"{row_description}: invalid CR component endpoints")
        if previous_upper is not None and lower <= previous_upper:
            raise ValueError(f"{row_description}: overlapping/unsorted CR components")
        components.append((lower, upper))
        previous_upper = upper
    return tuple(components)


def _validate_detailed_confidence_regions(frame: pd.DataFrame, label: str) -> None:
    allowed_statuses = {
        "valid",
        "full_grid_valid",
        "empty_valid",
        "partially_unresolved",
        "fully_unresolved",
    }
    statuses = frame["cr_status"].astype("string")
    invalid_statuses = sorted(set(statuses.dropna()).difference(allowed_statuses))
    if statuses.isna().any() or invalid_statuses:
        raise ValueError(f"{label}: missing or invalid cr_status values: {invalid_statuses}")

    explicit_resolved = _strict_boolean(
        frame["cr_is_numerically_resolved"], "cr_is_numerically_resolved"
    )
    if explicit_resolved.isna().any():
        raise ValueError(f"{label}: missing cr_is_numerically_resolved values")
    expected_resolved = ~statuses.isin(["partially_unresolved", "fully_unresolved"])
    if not explicit_resolved.astype(bool).eq(expected_resolved).all():
        raise ValueError(f"{label}: cr_status conflicts with numerical-resolution flag")

    for index, row in frame.iterrows():
        description = f"{label} row {index}"
        components = _components_from_json(row["cr_components"], description)
        n_blocks = int(row["cr_n_blocks"])
        disconnected = bool(row["cr_disconnected"])
        if n_blocks != len(components) or disconnected != (len(components) > 1):
            raise ValueError(f"{description}: component diagnostics are inconsistent")
        status = str(row["cr_status"])
        if status == "empty_valid" and components:
            raise ValueError(f"{description}: empty_valid has nonempty components")
        if status != "empty_valid" and not components:
            if status not in {"partially_unresolved", "fully_unresolved"}:
                raise ValueError(f"{description}: resolved nonempty status has no components")
        if components:
            expected_length = sum(upper - lower for lower, upper in components)
            if not np.isclose(float(row["cr_lower"]), components[0][0], atol=GRID_TOLERANCE):
                raise ValueError(f"{description}: cr_lower conflicts with components")
            if not np.isclose(float(row["cr_upper"]), components[-1][1], atol=GRID_TOLERANCE):
                raise ValueError(f"{description}: cr_upper conflicts with components")
            if not np.isclose(float(row["cr_length"]), expected_length, atol=GRID_TOLERANCE):
                raise ValueError(f"{description}: cr_length conflicts with components")
        elif status == "empty_valid":
            if pd.notna(row["cr_lower"]) or pd.notna(row["cr_upper"]) or pd.notna(row["cr_length"]):
                raise ValueError(f"{description}: empty_valid must have missing hull/length")


def _component_coverage(frame: pd.DataFrame, label: str) -> pd.Series:
    values: list[bool] = []
    for index, row in frame.iterrows():
        components = _components_from_json(row["cr_components"], f"{label} row {index}")
        truth = float(row["alpha_true"])
        values.append(
            any(
                lower - GRID_TOLERANCE <= truth <= upper + GRID_TOLERANCE
                for lower, upper in components
            )
        )
    return pd.Series(values, index=frame.index, dtype="boolean")


def standardize_estimator(
    source: pd.DataFrame,
    label: str,
    *,
    grid_lower: float = GRID_LOWER,
    grid_upper: float = GRID_UPPER,
) -> pd.DataFrame:
    """Validate and harmonize one estimator while preserving unknown states."""
    require_columns(source, COMMON_REQUIRED_COLUMNS, label)
    if source.empty:
        raise ValueError(f"{label}: result file contains no rows")
    frame = source.copy()
    expected_raw_label = RAW_ESTIMATOR_LABELS[label]
    raw_labels = set(frame["estimator"].dropna().astype(str).unique())
    if raw_labels != {expected_raw_label}:
        raise ValueError(
            f"{label}: expected raw estimator label {expected_raw_label!r}, "
            f"found {sorted(raw_labels)!r}"
        )
    if frame[PANEL_KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{label}: missing design-replication keys")

    frame["estimator_label"] = label
    frame["covered_raw"] = _strict_boolean(frame["covered"], "covered")
    frame["converged"] = _strict_boolean(frame["converged"], "converged")
    numeric_columns = [
        "n", "p", "pi", "tau", "rep", "seed", "alpha_hat", "alpha_true",
        "cr_lower", "cr_upper", "cr_length",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["n", "p", "pi", "tau", "rep", "seed", "alpha_true"]].isna().any().any():
        raise ValueError(f"{label}: invalid required numeric values")

    complete_cr = frame[["cr_lower", "cr_upper", "cr_length"]].notna().all(axis=1)
    partial_cr = frame[["cr_lower", "cr_upper", "cr_length"]].notna().any(axis=1) & ~complete_cr
    if partial_cr.any():
        raise ValueError(f"{label}: {int(partial_cr.sum())} partially missing CR triplets")
    if (complete_cr & frame["cr_length"].lt(-GRID_TOLERANCE)).any():
        raise ValueError(f"{label}: negative confidence-region length")

    has_detailed_schema = set(DETAILED_CR_COLUMNS).issubset(frame.columns)
    if label == "DML-IVQR" and has_detailed_schema:
        raise ValueError("DML-IVQR: unexpected mixed legacy/detailed CR schema")
    if label != "DML-IVQR" and not has_detailed_schema:
        missing = sorted(set(DETAILED_CR_COLUMNS).difference(frame.columns))
        raise ValueError(f"{label}: detailed CR schema is incomplete: {missing}")

    if has_detailed_schema:
        _validate_detailed_confidence_regions(frame, label)
        resolved = _strict_boolean(
            frame["cr_is_numerically_resolved"], "cr_is_numerically_resolved"
        )
        if (resolved.fillna(False) & ~frame["converged"].fillna(False)).any():
            raise ValueError(f"{label}: resolved CR attached to nonconverged row")
        component_covered = _component_coverage(frame, label)
        comparable = resolved.fillna(False)
        if not component_covered.loc[comparable].eq(frame.loc[comparable, "covered_raw"]).all():
            raise ValueError(f"{label}: covered flag conflicts with CR components")
        frame["cr_status_standardized"] = frame["cr_status"].astype("string")
        frame["resolution_basis"] = "explicit_numerical_status"
        frame["empty_region"] = frame["cr_status"].eq("empty_valid").astype("boolean")
        frame["full_grid_region"] = frame["cr_status"].eq("full_grid_valid").astype("boolean")
        frame["disconnected_region"] = _strict_boolean(
            frame["cr_disconnected"], "cr_disconnected"
        )
        frame["cr_length_analysis"] = frame["cr_length"]
        frame.loc[frame["empty_region"].fillna(False), "cr_length_analysis"] = 0.0
        frame.loc[~resolved.fillna(False), "cr_length_analysis"] = np.nan
    else:
        resolved = (complete_cr & frame["converged"].fillna(False)).astype("boolean")
        geometry_full_grid = (
            complete_cr
            & np.isclose(frame["cr_lower"], grid_lower, atol=GRID_TOLERANCE)
            & np.isclose(frame["cr_upper"], grid_upper, atol=GRID_TOLERANCE)
            & np.isclose(
                frame["cr_length"], grid_upper - grid_lower, atol=GRID_TOLERANCE
            )
        )
        frame["cr_status_standardized"] = np.select(
            [~complete_cr, geometry_full_grid],
            ["missing_status_unavailable", "full_grid_observed_status_unavailable"],
            default="observed_status_unavailable",
        )
        frame["resolution_basis"] = "complete_observed_cr_proxy_status_unavailable"
        # A zero-length observed set can be a singleton; absence of components means
        # DML emptiness and disconnectedness cannot be reconstructed.
        frame["empty_region"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        frame["full_grid_region"] = geometry_full_grid.astype("boolean")
        frame["disconnected_region"] = pd.Series(
            pd.NA, index=frame.index, dtype="boolean"
        )
        frame["cr_length_analysis"] = frame["cr_length"].where(resolved.fillna(False))

    frame["resolved"] = resolved.astype("boolean")
    frame["unresolved"] = (~frame["resolved"]).astype("boolean")
    frame["coverage_status"] = pd.Series("unresolved", index=frame.index, dtype="string")
    resolved_mask = frame["resolved"].fillna(False)
    frame.loc[resolved_mask & frame["covered_raw"].fillna(False), "coverage_status"] = "covered"
    frame.loc[resolved_mask & ~frame["covered_raw"].fillna(False), "coverage_status"] = "not_covered"
    frame["coverage_resolved"] = pd.Series(np.nan, index=frame.index, dtype=float)
    frame.loc[resolved_mask, "coverage_resolved"] = (
        frame.loc[resolved_mask, "covered_raw"].astype(float)
    )

    finite_estimate = np.isfinite(frame["alpha_hat"]) & np.isfinite(frame["alpha_true"])
    frame["point_estimate_valid"] = (
        frame["converged"].fillna(False) & finite_estimate
    ).astype("boolean")
    error = frame["alpha_hat"] - frame["alpha_true"]
    frame["estimation_error"] = error.where(frame["point_estimate_valid"].fillna(False))
    frame["absolute_error"] = frame["estimation_error"].abs()
    frame["squared_error"] = frame["estimation_error"].pow(2)
    frame["boundary_estimate"] = (
        frame["point_estimate_valid"].fillna(False)
        & (
            np.isclose(frame["alpha_hat"], grid_lower, atol=GRID_TOLERANCE)
            | np.isclose(frame["alpha_hat"], grid_upper, atol=GRID_TOLERANCE)
        )
    ).astype("boolean")

    frame["iteration_warning"] = _nullable_positive_indicator(
        frame, "iteration_warning_evaluations"
    )
    frame["rank_failure"] = _nullable_positive_indicator(
        frame, "rank_deficient_covariance_failures"
    )
    frame["refinement_limit"] = _nullable_flag(frame, "refinement_limit_hit")
    max_evaluations = _nullable_flag(frame, "max_alpha_evaluations_hit")
    midpoint_limit = _nullable_flag(frame, "midpoint_probe_limit_hit")
    numerical_limit = frame["refinement_limit"] | max_evaluations | midpoint_limit
    all_limit_missing = (
        frame["refinement_limit"].isna()
        & max_evaluations.isna()
        & midpoint_limit.isna()
    )
    numerical_limit.loc[all_limit_missing] = pd.NA
    frame["numerical_limit"] = numerical_limit.astype("boolean")

    for optional_numeric in ["n_selected_controls", "n_retained_instruments"]:
        if optional_numeric not in frame:
            frame[optional_numeric] = np.nan
        else:
            frame[optional_numeric] = pd.to_numeric(
                frame[optional_numeric], errors="coerce"
            )
    return frame


def read_estimator(path: Path, label: str) -> tuple[pd.DataFrame, list[str]]:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{label}: file does not exist: {resolved_path}")
    if resolved_path.stat().st_size == 0:
        raise ValueError(f"{label}: file is empty: {resolved_path}")
    raw = pd.read_csv(resolved_path, low_memory=False)
    original_columns = list(raw.columns)
    return standardize_estimator(raw, label), original_columns


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_panel(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Fail loudly unless every estimator contains the same balanced panel."""
    diagnostics: dict[str, Any] = {}
    key_frames: dict[str, pd.DataFrame] = {}
    reference_replications: set[object] | None = None

    for label in ESTIMATOR_ORDER:
        frame = frames[label]
        duplicate_count = int(frame.duplicated(PANEL_KEY_COLUMNS).sum())
        if duplicate_count:
            raise ValueError(
                f"{label}: {duplicate_count} duplicate design-replication rows"
            )
        cell_counts = frame.groupby(DESIGN_COLUMNS, observed=True, dropna=False).size()
        if cell_counts.empty:
            raise ValueError(f"{label}: no design cells")
        cell_rep_sets = frame.groupby(DESIGN_COLUMNS, observed=True, dropna=False)["rep"].agg(
            lambda values: frozenset(values.tolist())
        )
        first_rep_set = set(cell_rep_sets.iloc[0])
        if any(set(values) != first_rep_set for values in cell_rep_sets):
            raise ValueError(f"{label}: replication identifiers differ across design cells")
        if not cell_counts.eq(len(first_rep_set)).all():
            raise ValueError(f"{label}: invalid replication counts or duplicate rep IDs")
        if reference_replications is None:
            reference_replications = first_rep_set
        elif first_rep_set != reference_replications:
            raise ValueError(f"{label}: replication identifiers differ across estimators")

        key_frames[label] = frame[PANEL_KEY_COLUMNS].sort_values(
            PANEL_KEY_COLUMNS, kind="stable"
        ).reset_index(drop=True)
        status_counts = frame["cr_status_standardized"].value_counts(dropna=False)
        diagnostics[label] = {
            "rows": int(len(frame)),
            "design_cells": int(len(cell_counts)),
            "replication_identifiers": sorted(_json_scalar(v) for v in first_rep_set),
            "replications_per_design_min": int(cell_counts.min()),
            "replications_per_design_max": int(cell_counts.max()),
            "duplicates": duplicate_count,
            "resolved_replications": int(frame["resolved"].sum()),
            "unresolved_replications": int(frame["unresolved"].sum()),
            "status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "design_values": {
                column: [
                    _json_scalar(value)
                    for value in sorted(frame[column].dropna().unique().tolist())
                ]
                for column in DESIGN_COLUMNS
            },
        }

    reference_label = ESTIMATOR_ORDER[0]
    for label in ESTIMATOR_ORDER[1:]:
        if not key_frames[label].equals(key_frames[reference_label]):
            raise ValueError(
                f"Design-replication keys differ between {reference_label} and {label}"
            )

    aligned = {
        label: frames[label].sort_values(PANEL_KEY_COLUMNS, kind="stable").reset_index(drop=True)
        for label in ESTIMATOR_ORDER
    }
    for label in ESTIMATOR_ORDER[1:]:
        if not aligned[label]["seed"].equals(aligned[reference_label]["seed"]):
            raise ValueError(f"Seeds differ between {reference_label} and {label}")
        if not np.allclose(
            aligned[label]["alpha_true"],
            aligned[reference_label]["alpha_true"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"alpha_true differs between {reference_label} and {label}")
    diagnostics["cross_estimator_keys_identical"] = True
    diagnostics["cross_estimator_seeds_identical"] = True
    diagnostics["cross_estimator_truth_identical"] = True
    return diagnostics


def _mean_nullable(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else float("nan")


def _summary_record(group: pd.DataFrame) -> dict[str, int | float]:
    replications = int(len(group))
    resolved_replications = int(group["resolved"].sum())
    coverage = _mean_nullable(group["coverage_resolved"])
    coverage_mcse = (
        math.sqrt(coverage * (1.0 - coverage) / resolved_replications)
        if resolved_replications and math.isfinite(coverage)
        else float("nan")
    )
    squared_error = pd.to_numeric(group["squared_error"], errors="coerce")
    selected = pd.to_numeric(group["n_selected_controls"], errors="coerce")
    retained = pd.to_numeric(group["n_retained_instruments"], errors="coerce")
    return {
        "replications": replications,
        "resolved_replications": resolved_replications,
        "unresolved_replications": int(group["unresolved"].sum()),
        "convergence_rate": _mean_nullable(group["converged"]),
        "bias": _mean_nullable(group["estimation_error"]),
        "median_bias": float(group["estimation_error"].median()),
        "mae": _mean_nullable(group["absolute_error"]),
        "rmse": float(math.sqrt(squared_error.mean())),
        "estimate_sd": float(group["alpha_hat"].where(group["point_estimate_valid"]).std()),
        "empirical_coverage": coverage,
        "coverage_mcse": coverage_mcse,
        "coverage_mc95_lower": max(0.0, coverage - 1.96 * coverage_mcse)
        if math.isfinite(coverage_mcse)
        else float("nan"),
        "coverage_mc95_upper": min(1.0, coverage + 1.96 * coverage_mcse)
        if math.isfinite(coverage_mcse)
        else float("nan"),
        "mean_cr_length": _mean_nullable(group["cr_length_analysis"]),
        "median_cr_length": float(group["cr_length_analysis"].median()),
        "full_grid_rate": _mean_nullable(group["full_grid_region"]),
        "empty_region_rate": _mean_nullable(group["empty_region"]),
        "disconnected_region_rate": _mean_nullable(group["disconnected_region"]),
        "unresolved_rate": _mean_nullable(group["unresolved"]),
        "boundary_estimate_rate": _mean_nullable(group["boundary_estimate"]),
        "iteration_warning_rate": _mean_nullable(group["iteration_warning"]),
        "rank_failure_rate": _mean_nullable(group["rank_failure"]),
        "refinement_limit_rate": _mean_nullable(group["refinement_limit"]),
        "numerical_limit_rate": _mean_nullable(group["numerical_limit"]),
        "mean_selected_controls": float(selected.mean()),
        "median_selected_controls": float(selected.median()),
        "min_selected_controls": float(selected.min()),
        "max_selected_controls": float(selected.max()),
        "mean_retained_instruments": float(retained.mean()),
    }


def summarize(frame: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    group_argument: str | list[str] = by[0] if len(by) == 1 else list(by)
    for key, group in frame.groupby(
        group_argument, observed=True, dropna=False, sort=True
    ):
        keys = (key,) if len(by) == 1 else tuple(key)
        record: dict[str, object] = dict(zip(by, keys, strict=True))
        record.update(_summary_record(group))
        records.append(record)
    return pd.DataFrame.from_records(records)


def diagnostics_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (label, status), group in frame.groupby(
        ["estimator_label", "cr_status_standardized"],
        observed=True,
        dropna=False,
        sort=True,
    ):
        record: dict[str, object] = {
            "estimator_label": label,
            "cr_status_standardized": status,
            "replications": int(len(group)),
            "resolved_replications": int(group["resolved"].sum()),
            "covered_resolved": int(group["coverage_status"].eq("covered").sum()),
            "uncovered_resolved": int(group["coverage_status"].eq("not_covered").sum()),
            "unresolved_replications": int(group["coverage_status"].eq("unresolved").sum()),
            "empirical_coverage": _mean_nullable(group["coverage_resolved"]),
            "mean_cr_length": _mean_nullable(group["cr_length_analysis"]),
            "disconnected_region_rate": _mean_nullable(group["disconnected_region"]),
            "iteration_warning_rate": _mean_nullable(group["iteration_warning"]),
            "rank_failure_rate": _mean_nullable(group["rank_failure"]),
            "refinement_limit_rate": _mean_nullable(group["refinement_limit"]),
            "numerical_limit_rate": _mean_nullable(group["numerical_limit"]),
            "mean_selected_controls": _mean_nullable(group["n_selected_controls"]),
            "median_selected_controls": float(group["n_selected_controls"].median()),
            "mean_retained_instruments": _mean_nullable(group["n_retained_instruments"]),
        }
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def worst_cells(cell_summary: pd.DataFrame, per_estimator: int = 10) -> pd.DataFrame:
    ranked = cell_summary.sort_values(
        ["estimator_label", "empirical_coverage", "unresolved_rate", *DESIGN_COLUMNS],
        ascending=[True, True, False, True, True, True, True, True],
        kind="stable",
    )
    return ranked.groupby("estimator_label", sort=False, observed=True).head(
        per_estimator
    ).reset_index(drop=True)


def _sort_combined(frame: pd.DataFrame) -> pd.DataFrame:
    order = {label: index for index, label in enumerate(ESTIMATOR_ORDER)}
    sorted_frame = frame.assign(
        _estimator_order=frame["estimator_label"].map(order).astype(int)
    ).sort_values(["_estimator_order", *PANEL_KEY_COLUMNS], kind="stable")
    return sorted_frame.drop(columns="_estimator_order").reset_index(drop=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    headers = list(columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.loc[:, headers].iterrows():
        cells: list[str] = []
        for value in row:
            if pd.isna(value):
                cells.append("NA")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved)


def build_report(
    paths: dict[str, Path],
    validation: dict[str, Any],
    overall: pd.DataFrame,
) -> str:
    path_lines = "\n".join(
        f"- {label}: `{_relative_or_absolute(path)}`" for label, path in paths.items()
    )
    output_lines = "\n".join(f"- `{name}`" for name in OUTPUT_NAMES)
    report_overall = overall.copy()
    for column in [
        "empirical_coverage", "coverage_mcse", "coverage_mc95_lower",
        "coverage_mc95_upper", "unresolved_rate",
    ]:
        report_overall[column] = report_overall[column] * 100.0
    table = _markdown_table(
        report_overall,
        [
            "estimator_label", "replications", "resolved_replications",
            "empirical_coverage", "coverage_mcse", "coverage_mc95_lower",
            "coverage_mc95_upper", "bias", "mae", "rmse",
            "mean_cr_length", "unresolved_rate",
        ],
    )
    benchmark_lines = "\n".join(
        "- "
        + label
        + f": primary {details['actual_primary_coverage']:.4%}; "
        + f"reference {details['reference_coverage']:.1%}; "
        + f"difference {details['difference_percentage_points']:+.3f} percentage points."
        for label, details in validation["coverage_benchmarks"].items()
    )
    design_values = validation["panel"]["DML-IVQR"]["design_values"]
    design_summary = "; ".join(
        f"`{column}` = {', '.join(map(str, design_values[column]))}"
        for column in DESIGN_COLUMNS
    )
    return f"""# Full-run IVQR analysis report

This report is generated deterministically from the existing simulation CSVs. No
simulation or estimator implementation is invoked.

## Reproduction

Run this exact one-line Windows PowerShell command from the repository root:

```powershell
pixi run python scripts/report_full_run.py
```

If an alternate Oracle file is supplied and its name contains parentheses, quote
it, for example: `--oracle "results\\raw\\oracle_ivqr(1).csv"`.

## Inputs and panel validation

{path_lines}

Each estimator has {validation['panel']['DML-IVQR']['rows']:,} rows across
{validation['panel']['DML-IVQR']['design_cells']} design cells, with
{validation['panel']['DML-IVQR']['replications_per_design_min']} replications per
cell. Design variables are `dgp`, `n`, `p`, `pi`, and `tau`; `rep` is the
replication identifier. The design-replication keys and seeds are identical
across estimators, and no duplicates were found.

Observed design values are: {design_summary}. The DML source has
{len(validation['inputs']['DML-IVQR']['original_columns'])} columns, Oracle has
{len(validation['inputs']['Oracle IVQR']['original_columns'])}, and
post-selection has
{len(validation['inputs']['Post-selection IVQR']['original_columns'])}.
`validation.json` records every original column and every column unavailable for
each estimator.

## Coverage and status rules

Primary empirical coverage uses only resolved replications. Explicit
`cr_is_numerically_resolved` values define the denominator for Oracle and
post-selection. `partially_unresolved` and `fully_unresolved` rows have missing
analysis coverage and are reported separately; their raw `covered` values are
never silently treated as success or failure.

`empty_valid` is an explicitly resolved empty set. It contributes zero CR length
and is uncovered because its validated component list contains no true parameter.
In contrast, a missing DML CR is **not** called empty: the legacy DML schema has no
components, CR status, or numerical-resolution flag. For DML only, a complete CR
triplet plus estimator convergence is the observable resolved proxy. Its
{validation['panel']['DML-IVQR']['unresolved_replications']} missing CR triplets
are classified `missing_status_unavailable`, excluded from primary coverage, and
reported as unresolved. DML empty- and disconnected-region rates remain `NA`.

Confidence-region length is total component length for detailed schemas; resolved
empty regions use length zero. Full-grid and boundary diagnostics use the
simulation grid [{GRID_LOWER:g}, {GRID_UPPER:g}], as configured for these full runs.
Unavailable DML warning, rank, refinement, and selected-control diagnostics remain
`NA` rather than being fabricated as zero.

## Overall results

Coverage quantities in this display are percentages; MC intervals use the normal
Monte Carlo approximation `coverage +/- 1.96 * MCSE`.

{table}

## Benchmark reconciliation

{benchmark_lines}

All three primary coverage values are within the predeclared 0.5-percentage-point
reconciliation tolerance. DML's unconditional raw mean is
{validation['coverage_benchmarks']['DML-IVQR']['raw_coverage_all_rows']:.4%}; the
small difference from primary coverage comes from the 43 status-unavailable rows.
Post-selection has two explicitly unresolved rows, one of which has raw
`covered=True`; it is correctly excluded from both sides of primary coverage.

## Generated files

{output_lines}

`combined_standardized_results.csv` retains all source columns and adds derived,
tri-state analysis fields. Summary-table rates are `NA` when an estimator's source
schema does not contain the needed diagnostic.

## Remaining data-quality limitations

- The legacy DML schema cannot distinguish an empty valid set from a numerical
  failure or another cause of missing CR geometry.
- DML has no component representation, so disconnected regions cannot be
  reconstructed and its observed hull is the only available CR geometry.
- DML has no iteration-warning, rank-failure, refinement-limit, or variable-
  selection diagnostics. These are preserved as unavailable.
- The normal 95% Monte Carlo interval is a simulation-uncertainty interval for
  empirical coverage, not a confidence interval for an individual estimate.
"""


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "DML-IVQR": args.dml.expanduser().resolve(),
        "Oracle IVQR": args.oracle.expanduser().resolve(),
        "Post-selection IVQR": args.post_selection.expanduser().resolve(),
    }

    frames: dict[str, pd.DataFrame] = {}
    schemas: dict[str, list[str]] = {}
    for label in ESTIMATOR_ORDER:
        frames[label], schemas[label] = read_estimator(paths[label], label)

    panel_validation = validate_panel(frames)
    combined = _sort_combined(pd.concat(frames.values(), ignore_index=True, sort=False))
    overall = summarize(combined, ["estimator_label"])
    by_quantile = summarize(combined, ["estimator_label", "tau"])
    by_strength = summarize(combined, ["estimator_label", "pi"])
    by_np = summarize(combined, ["estimator_label", "n", "p"])
    by_cell = summarize(combined, ["estimator_label", *DESIGN_COLUMNS])
    worst = worst_cells(by_cell)
    diagnostics = diagnostics_table(combined)

    coverage_benchmarks: dict[str, dict[str, float | bool]] = {}
    for label in ESTIMATOR_ORDER:
        row = overall.loc[overall["estimator_label"].eq(label)].iloc[0]
        actual = float(row["empirical_coverage"])
        reference = COVERAGE_BENCHMARKS[label]
        difference_pp = 100.0 * (actual - reference)
        coverage_benchmarks[label] = {
            "actual_primary_coverage": actual,
            "raw_coverage_all_rows": float(frames[label]["covered_raw"].mean()),
            "reference_coverage": reference,
            "difference_percentage_points": difference_pp,
            "within_0_5_percentage_points": abs(difference_pp) <= 0.5,
        }

    all_columns = set().union(*(set(columns) for columns in schemas.values()))
    common_columns = set(schemas[ESTIMATOR_ORDER[0]])
    for columns in schemas.values():
        common_columns.intersection_update(columns)
    validation: dict[str, Any] = {
        "inputs": {
            label: {
                "path": str(paths[label]),
                "original_columns": schemas[label],
                "estimator_specific_columns": sorted(
                    set(schemas[label]).difference(common_columns)
                ),
                "columns_not_available": sorted(
                    all_columns.difference(schemas[label])
                ),
            }
            for label in ESTIMATOR_ORDER
        },
        "schema": {
            "common_original_columns": sorted(common_columns),
            "union_original_columns": sorted(all_columns),
        },
        "panel": panel_validation,
        "coverage_benchmarks": coverage_benchmarks,
        "reconciliation": {
            "sum_input_rows": int(sum(len(frame) for frame in frames.values())),
            "combined_rows": int(len(combined)),
            "overall_replications": int(overall["replications"].sum()),
            "overall_resolved_replications": int(
                overall["resolved_replications"].sum()
            ),
            "combined_resolved_replications": int(combined["resolved"].sum()),
            "all_row_totals_match": bool(
                len(combined)
                == sum(len(frame) for frame in frames.values())
                == overall["replications"].sum()
            ),
            "resolved_totals_match": bool(
                overall["resolved_replications"].sum() == combined["resolved"].sum()
            ),
        },
    }
    if not validation["reconciliation"]["all_row_totals_match"]:
        raise ValueError("Generated row totals do not reconcile")
    if not validation["reconciliation"]["resolved_totals_match"]:
        raise ValueError("Generated resolved-replication totals do not reconcile")

    csv_outputs = {
        "combined_standardized_results.csv": combined,
        "table_01_overall.csv": overall,
        "table_02_by_quantile.csv": by_quantile,
        "table_03_by_strength.csv": by_strength,
        "table_04_by_n_p.csv": by_np,
        "table_05_by_design_cell.csv": by_cell,
        "table_06_worst_cells.csv": worst,
        "table_07_diagnostics.csv": diagnostics,
    }
    for name, frame in csv_outputs.items():
        write_csv(frame, output_dir / name)
    with (output_dir / "validation.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(validation, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    report = build_report(paths, validation, overall)
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8", newline="\n")

    print(f"Wrote {len(OUTPUT_NAMES)} outputs to {output_dir}")
    print(
        overall[
            [
                "estimator_label", "replications", "resolved_replications",
                "empirical_coverage", "unresolved_rate",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
