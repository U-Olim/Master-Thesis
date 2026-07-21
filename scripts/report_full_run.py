#!/usr/bin/env python
"""
Prepare standardized analysis datasets and core Monte Carlo tables
for the full IVQR estimator comparison.

Usage
-----
python scripts/report_full_run.py \
    --dml results/dml_ivqr.csv \
    --oracle results/oracle_ivqr.csv \
    --post-selection results/post_selection_ivqr.csv \
    --output-dir results/report_full_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DESIGN_KEYS = ["dgp", "n", "p", "pi", "tau", "rep"]
GROUP_KEYS = ["dgp", "n", "p", "pi", "tau"]
GRID_LOWER = -1.0
GRID_UPPER = 3.0
GRID_TOL = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dml", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--post-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


def read_estimator(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    common_required = [
        *DESIGN_KEYS,
        "seed",
        "alpha_hat",
        "alpha_true",
        "cr_lower",
        "cr_upper",
        "cr_length",
        "covered",
        "converged",
    ]
    require_columns(df, common_required, label)

    df = df.copy()
    df["estimator_label"] = label

    # Schema harmonization.
    defaults = {
        "cr_components": pd.NA,
        "cr_n_blocks": pd.NA,
        "cr_disconnected": pd.NA,
        "cr_status": pd.NA,
        "cr_is_numerically_resolved": pd.NA,
        "cr_unresolved_count": pd.NA,
        "iteration_warning_evaluations": pd.NA,
        "rank_deficient_covariance_failures": pd.NA,
        "refinement_limit_hit": pd.NA,
        "max_alpha_evaluations_hit": pd.NA,
        "n_selected_controls": pd.NA,
        "n_retained_instruments": pd.NA,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    # DML has no explicit CR status. Infer only what is observable.
    if label == "DML-IVQR":
        has_cr = (
            df["cr_lower"].notna()
            & df["cr_upper"].notna()
            & df["cr_length"].notna()
        )
        df["cr_status_standardized"] = np.where(
            ~has_cr,
            "missing_or_failed",
            np.where(
                np.isclose(df["cr_length"], 0.0, atol=GRID_TOL),
                "empty_or_zero_length",
                np.where(
                    np.isclose(df["cr_lower"], GRID_LOWER, atol=GRID_TOL)
                    & np.isclose(df["cr_upper"], GRID_UPPER, atol=GRID_TOL)
                    & np.isclose(
                        df["cr_length"],
                        GRID_UPPER - GRID_LOWER,
                        atol=GRID_TOL,
                    ),
                    "full_grid_valid",
                    "valid_observed",
                ),
            ),
        )
        df["resolved"] = has_cr & df["converged"].fillna(False)
    else:
        df["cr_status_standardized"] = df["cr_status"].fillna("missing")
        df["resolved"] = (
            df["cr_is_numerically_resolved"].fillna(False).astype(bool)
            & df["converged"].fillna(False).astype(bool)
        )

    # Derived estimation metrics.
    df["estimation_error"] = df["alpha_hat"] - df["alpha_true"]
    df["absolute_error"] = df["estimation_error"].abs()
    df["squared_error"] = df["estimation_error"] ** 2

    # Diagnostics.
    df["cr_observed"] = (
        df["cr_lower"].notna()
        & df["cr_upper"].notna()
        & df["cr_length"].notna()
    )
    df["empty_region"] = (
        df["cr_status_standardized"].eq("empty_valid")
        | df["cr_status_standardized"].eq("empty_or_zero_length")
    )
    df["full_grid_region"] = (
        df["cr_status_standardized"].eq("full_grid_valid")
        | (
            df["cr_observed"]
            & np.isclose(df["cr_lower"], GRID_LOWER, atol=GRID_TOL)
            & np.isclose(df["cr_upper"], GRID_UPPER, atol=GRID_TOL)
            & np.isclose(
                df["cr_length"],
                GRID_UPPER - GRID_LOWER,
                atol=GRID_TOL,
            )
        )
    )
    df["unresolved"] = ~df["resolved"]
    df["boundary_estimate"] = (
        np.isclose(df["alpha_hat"], GRID_LOWER, atol=GRID_TOL)
        | np.isclose(df["alpha_hat"], GRID_UPPER, atol=GRID_TOL)
    )
    df["iteration_warning"] = (
        pd.to_numeric(df["iteration_warning_evaluations"], errors="coerce")
        .fillna(0)
        .gt(0)
    )
    df["rank_failure"] = (
        pd.to_numeric(df["rank_deficient_covariance_failures"], errors="coerce")
        .fillna(0)
        .gt(0)
    )
    df["refinement_limit"] = (
        df["refinement_limit_hit"].fillna(False).astype(bool)
        | df["max_alpha_evaluations_hit"].fillna(False).astype(bool)
    )

    # Coverage used for primary reporting:
    # only resolved replications enter the denominator.
    df["covered_resolved"] = np.where(
        df["resolved"],
        df["covered"].astype(float),
        np.nan,
    )

    return df


def validate_panel(frames: dict[str, pd.DataFrame]) -> dict:
    diagnostics: dict[str, object] = {}

    key_sets = {}
    for label, df in frames.items():
        duplicate_count = int(df.duplicated(DESIGN_KEYS).sum())
        if duplicate_count:
            raise ValueError(
                f"{label}: {duplicate_count} duplicate design-replication rows"
            )

        cell_sizes = df.groupby(GROUP_KEYS, observed=True).size()
        if not cell_sizes.eq(500).all():
            bad = cell_sizes.loc[~cell_sizes.eq(500)]
            raise ValueError(
                f"{label}: expected 500 replications per design; "
                f"bad cells: {bad.head().to_dict()}"
            )

        key_sets[label] = set(map(tuple, df[DESIGN_KEYS].to_numpy()))
        diagnostics[label] = {
            "rows": int(len(df)),
            "design_cells": int(df[GROUP_KEYS].drop_duplicates().shape[0]),
            "replications_per_cell_min": int(cell_sizes.min()),
            "replications_per_cell_max": int(cell_sizes.max()),
            "duplicates": duplicate_count,
        }

    reference_label = next(iter(key_sets))
    reference = key_sets[reference_label]
    for label, keys in key_sets.items():
        if keys != reference:
            raise ValueError(
                f"Design-replication keys differ between "
                f"{reference_label} and {label}"
            )

    diagnostics["cross_estimator_keys_identical"] = True
    return diagnostics


def monte_carlo_se(coverage: pd.Series, n_resolved: pd.Series) -> pd.Series:
    return np.sqrt(coverage * (1.0 - coverage) / n_resolved)


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    grouped = df.groupby(by, observed=True, dropna=False)

    out = grouped.agg(
        replications=("rep", "size"),
        resolved_replications=("resolved", "sum"),
        converged_rate=("converged", "mean"),
        bias=("estimation_error", "mean"),
        median_bias=("estimation_error", "median"),
        mae=("absolute_error", "mean"),
        rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))),
        estimate_sd=("alpha_hat", "std"),
        coverage=("covered_resolved", "mean"),
        mean_cr_length=("cr_length", "mean"),
        median_cr_length=("cr_length", "median"),
        full_grid_rate=("full_grid_region", "mean"),
        empty_region_rate=("empty_region", "mean"),
        unresolved_rate=("unresolved", "mean"),
        disconnected_rate=(
            "cr_disconnected",
            lambda x: pd.to_numeric(x, errors="coerce").mean(),
        ),
        boundary_estimate_rate=("boundary_estimate", "mean"),
        iteration_warning_rate=("iteration_warning", "mean"),
        rank_failure_rate=("rank_failure", "mean"),
        refinement_limit_rate=("refinement_limit", "mean"),
        mean_selected_controls=(
            "n_selected_controls",
            lambda x: pd.to_numeric(x, errors="coerce").mean(),
        ),
    ).reset_index()

    out["coverage_mcse"] = monte_carlo_se(
        out["coverage"], out["resolved_replications"]
    )
    out["coverage_ci_lower"] = np.maximum(
        0.0, out["coverage"] - 1.96 * out["coverage_mcse"]
    )
    out["coverage_ci_upper"] = np.minimum(
        1.0, out["coverage"] + 1.96 * out["coverage_mcse"]
    )
    return out


def worst_cells(cell_summary: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    columns = [
        "estimator_label",
        "dgp",
        "n",
        "p",
        "pi",
        "tau",
        "resolved_replications",
        "coverage",
        "coverage_mcse",
        "bias",
        "rmse",
        "mean_cr_length",
        "full_grid_rate",
        "unresolved_rate",
    ]
    return (
        cell_summary.sort_values(
            ["coverage", "estimator_label", "dgp", "n", "p", "pi", "tau"],
            ascending=[True, True, True, True, True, True, True],
        )
        .head(n)
        .loc[:, columns]
        .reset_index(drop=True)
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.10g")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        "DML-IVQR": read_estimator(args.dml, "DML-IVQR"),
        "Oracle IVQR": read_estimator(args.oracle, "Oracle IVQR"),
        "Post-selection IVQR": read_estimator(
            args.post_selection, "Post-selection IVQR"
        ),
    }

    validation = validate_panel(frames)
    combined = pd.concat(frames.values(), ignore_index=True, sort=False)

    estimator_order = [
        "DML-IVQR",
        "Oracle IVQR",
        "Post-selection IVQR",
    ]
    combined["estimator_label"] = pd.Categorical(
        combined["estimator_label"],
        categories=estimator_order,
        ordered=True,
    )

    overall = summarize(combined, ["estimator_label"])
    by_quantile = summarize(combined, ["estimator_label", "tau"])
    by_strength = summarize(combined, ["estimator_label", "pi"])
    by_np = summarize(combined, ["estimator_label", "n", "p"])
    by_cell = summarize(
        combined,
        ["estimator_label", "dgp", "n", "p", "pi", "tau"],
    )
    diagnostics = summarize(combined, ["estimator_label"])
    worst = worst_cells(by_cell, n=10)

    write_csv(
        combined,
        args.output_dir / "combined_standardized_results.csv",
    )
    write_csv(overall, args.output_dir / "table_01_overall.csv")
    write_csv(by_quantile, args.output_dir / "table_02_by_quantile.csv")
    write_csv(by_strength, args.output_dir / "table_03_by_strength.csv")
    write_csv(by_np, args.output_dir / "table_04_by_n_p.csv")
    write_csv(by_cell, args.output_dir / "table_05_by_design_cell.csv")
    write_csv(worst, args.output_dir / "table_06_worst_cells.csv")
    write_csv(diagnostics, args.output_dir / "table_07_diagnostics.csv")

    with (args.output_dir / "validation.json").open("w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    print("Analysis completed.")
    print(f"Output directory: {args.output_dir}")
    print("\nOverall summary:")
    print(
        overall[
            [
                "estimator_label",
                "coverage",
                "bias",
                "rmse",
                "mean_cr_length",
                "full_grid_rate",
                "unresolved_rate",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
