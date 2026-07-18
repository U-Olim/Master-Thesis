"""Diagnose Oracle inverse-IVQR confidence-region grid resolution.

The script compares fixed and adaptively refined alpha grids using the same
adaptive helper as production. Direct true-alpha acceptance is read
from the preceding Oracle calibration diagnostic and joined by deterministic
design seed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from dgp.designs import Design
from dgp.generators import generate_data
from dgp.true_parameters import get_oracle_control_indices, true_alpha
from ivqr.ch_inverse import (
    HARD_FAILURE_POLICIES,
    ITERATION_WARNING_POLICIES,
    AlphaEvaluation,
    IterationWarningPolicy,
    HardFailurePolicy,
    evaluate_alpha_ch_ivqr,
    evaluate_ch_alpha_grid,
    validate_hard_failure_policy,
)
from ivqr.confidence_regions import (
    argmin_grid_usable,
    classify_alpha_grid,
    critical_value_chi_square,
    invert_score_test,
    serialize_cr_components,
    sanitize_grid_statistics,
)
from simulation.config import DEFAULT_BASE_SEED, DEFAULT_QUANTREG_MAX_ITER
from simulation.runner import make_simulation_grid


POLICY_OUTPUT_DIRECTORY = Path("results/diagnostics/iteration_warning_policy")
PROTECTED_CALIBRATION_OUTPUTS = (
    Path("results/diagnostics/oracle_calibration_summary.csv"),
    Path("results/diagnostics/oracle_calibration_replications.csv"),
)
CRITICAL_VALUE = critical_value_chi_square(0.95, df=1)
ADAPTIVE_MAX_SPACING = 0.025
FIXED_GRIDS = {
    "grid_21": np.linspace(-1.0, 3.0, 21),
    "grid_41": np.linspace(-1.0, 3.0, 41),
    "grid_81": np.linspace(-1.0, 3.0, 81),
}


def adaptive_refinement_grid(
    initial_grid: np.ndarray,
    evaluate_alpha: Callable[[float], AlphaEvaluation],
    *,
    critical_value: float = CRITICAL_VALUE,
    max_spacing: float = ADAPTIVE_MAX_SPACING,
) -> np.ndarray:
    """Return the canonical production-adaptive grid for a diagnostic."""
    return evaluate_ch_alpha_grid(
        initial_grid,
        evaluate_alpha,
        critical_value=critical_value,
        grid_strategy="adaptive",
        refinement_tolerance=max_spacing,
    ).alphas


def interpolated_acceptance_at_alpha(
    alphas: np.ndarray,
    statistics: np.ndarray,
    alpha: float,
    *,
    critical_value: float = CRITICAL_VALUE,
) -> bool:
    """Return acceptance under piecewise-linear interpolation of grid statistics."""
    alphas = np.asarray(alphas, dtype=float)
    statistics = np.asarray(statistics, dtype=float)
    if alphas.ndim != 1 or statistics.shape != alphas.shape or alphas.size == 0:
        raise ValueError("alphas and statistics must be nonempty matching vectors")
    if not np.all(np.diff(alphas) > 0.0):
        raise ValueError("alphas must be strictly increasing")
    if not np.all(np.isfinite(alphas)) or not np.all(np.isfinite(statistics)):
        raise ValueError("alphas and statistics must be finite")
    if alpha < alphas[0] or alpha > alphas[-1]:
        return False
    interpolated = float(np.interp(alpha, alphas, statistics))
    return bool(interpolated <= critical_value)


def exact_grid_consistency(
    alphas: np.ndarray,
    accepted: np.ndarray,
    alpha_true: float,
    covered: bool,
    direct_accepted: bool | None = None,
) -> tuple[bool, bool]:
    """Return exact-grid membership and any direct-acceptance/coverage mismatch."""
    alphas = np.asarray(alphas, dtype=float)
    accepted = np.asarray(accepted)
    matches = np.flatnonzero(np.isclose(alphas, alpha_true, rtol=0.0, atol=1e-12))
    if matches.size == 0:
        return False, False
    accepted_at_true = bool(accepted[int(matches[0])])
    reference_acceptance = (
        accepted_at_true if direct_accepted is None else direct_accepted
    )
    return True, bool(reference_acceptance != covered)


def _evaluate_grid(
    *,
    grid_variant: str,
    alphas: np.ndarray,
    cached_evaluations: dict[float, tuple[AlphaEvaluation, float]],
    alpha_true: float,
    direct_accepted: bool | None,
    hard_failure_policy: HardFailurePolicy = "unresolved",
) -> dict[str, Any]:
    """Construct one production-style confidence region from cached evaluations."""
    raw_statistics = np.array(
        [cached_evaluations[float(alpha)][0].statistic for alpha in alphas],
        dtype=float,
    )
    usable_flags = np.array(
        [cached_evaluations[float(alpha)][0].usable for alpha in alphas],
        dtype=bool,
    )
    evaluations = [cached_evaluations[float(alpha)][0] for alpha in alphas]
    usable_flags &= np.isfinite(raw_statistics)
    failed_count = int(np.sum(~usable_flags))
    statistics = raw_statistics.copy()
    inference_usable = usable_flags.copy()
    if hard_failure_policy == "legacy_reject":
        statistics, _ = sanitize_grid_statistics(raw_statistics, usable_flags)
        inference_usable = np.ones(len(alphas), dtype=bool)
    warning_evaluations = [
        item for item in evaluations if item.warning_type is not None
    ]
    usable_warning_count = sum(item.usable for item in warning_evaluations)
    converged_count = sum(item.converged for item in evaluations)
    iteration_warning = any(
        item.warning_type == "iteration_limit" for item in evaluations
    )
    runtime = float(sum(cached_evaluations[float(alpha)][1] for alpha in alphas))
    all_failed = failed_count == len(alphas)
    if all_failed:
        return {
            "grid_variant": grid_variant,
            "alpha_hat": np.nan,
            "bias": np.nan,
            "squared_error": np.nan,
            "covered": np.nan,
            "cr_lower": np.nan,
            "cr_upper": np.nan,
            "cr_length": np.nan,
            "cr_components": "[]",
            "full_grid_accepted": False,
            "empty_cr": True,
            "number_of_connected_components": 0,
            "number_of_alpha_evaluations": len(alphas),
            "failed_alpha_evaluations": failed_count,
            "fully_converged_evaluations": converged_count,
            "iteration_limit_warning_evaluations": len(warning_evaluations),
            "usable_warning_evaluations": usable_warning_count,
            "unusable_warning_evaluations": len(warning_evaluations)
            - usable_warning_count,
            "runtime_seconds": runtime,
            "converged": False,
            "usable": False,
            "warning_type": "iteration_limit" if iteration_warning else "",
            "iteration_limit_warning": iteration_warning,
            "finite_statistic_available": False,
            "statistic_used_for_inference": False,
            "failure_reason": "All alpha-grid evaluations failed",
            "hard_failure_policy": hard_failure_policy,
            "cr_status": "fully_unresolved",
            "cr_is_numerically_resolved": False,
            "coverage_status": "coverage_unresolved",
            "point_estimate_status": "fully_unresolved",
            "usable_alpha_evaluations": 0,
            "unresolved_alpha_evaluations": failed_count,
            "accepted_alpha_evaluations": 0,
            "cr_unresolved_alphas": json.dumps([float(value) for value in alphas]),
            "interpolated_true_alpha_accepted": np.nan,
            "direct_vs_interpolated_mismatch": np.nan,
            "interpolation_vs_region_mismatch": np.nan,
            "coverage_loss_vs_direct": np.nan,
            "interpolation_loss_vs_direct": np.nan,
            "reconstruction_loss": np.nan,
            "alpha_true_exact_grid_point": False,
            "exact_grid_point_consistency_mismatch": False,
        }

    point_estimate = argmin_grid_usable(alphas, statistics, inference_usable)
    if point_estimate.alpha_hat is None:
        raise RuntimeError("at least one usable alpha was expected")
    alpha_hat = point_estimate.alpha_hat
    region = invert_score_test(
        alphas=alphas,
        statistics=statistics,
        critical_value=CRITICAL_VALUE,
        alpha_true=alpha_true,
        statistic_reference=0.0,
        inversion_type="absolute",
        usable=inference_usable,
    )
    masks = classify_alpha_grid(statistics, inference_usable, CRITICAL_VALUE)
    accepted = masks.accepted
    interpolated_accepted: bool | float = (
        np.nan
        if region.coverage_status == "coverage_unresolved"
        else bool(region.covers_true)
    )
    covered: bool | float = (
        np.nan if region.covers_true is None else bool(region.covers_true)
    )
    exact_point, exact_mismatch = exact_grid_consistency(
        alphas,
        accepted,
        alpha_true,
        False if region.covers_true is None else bool(region.covers_true),
        direct_accepted if region.covers_true is not None else None,
    )
    direct_available = direct_accepted is not None
    return {
        "grid_variant": grid_variant,
        "alpha_hat": alpha_hat,
        "bias": alpha_hat - alpha_true,
        "squared_error": (alpha_hat - alpha_true) ** 2,
        "covered": covered,
        "cr_lower": np.nan if region.lower is None else region.lower,
        "cr_upper": np.nan if region.upper is None else region.upper,
        "cr_length": region.length,
        "cr_components": serialize_cr_components(region.blocks),
        "full_grid_accepted": region.full_grid_accepted,
        "empty_cr": region.empty,
        "number_of_connected_components": region.n_blocks,
        "number_of_alpha_evaluations": len(alphas),
        "failed_alpha_evaluations": failed_count,
        "fully_converged_evaluations": converged_count,
        "iteration_limit_warning_evaluations": len(warning_evaluations),
        "usable_warning_evaluations": usable_warning_count,
        "unusable_warning_evaluations": len(warning_evaluations) - usable_warning_count,
        "runtime_seconds": runtime,
        "converged": True,
        "usable": True,
        "warning_type": "iteration_limit" if iteration_warning else "",
        "iteration_limit_warning": iteration_warning,
        "finite_statistic_available": any(
            item.warning_type == "iteration_limit" and np.isfinite(item.statistic)
            for item in evaluations
        ),
        "statistic_used_for_inference": usable_warning_count > 0,
        "failure_reason": (
            ""
            if failed_count == 0
            else f"ok; failed_alpha_points={failed_count}/{len(alphas)}"
        ),
        "hard_failure_policy": hard_failure_policy,
        "cr_status": region.status,
        "cr_is_numerically_resolved": region.is_numerically_resolved,
        "coverage_status": region.coverage_status,
        "point_estimate_status": point_estimate.status,
        "usable_alpha_evaluations": int(np.sum(usable_flags)),
        "unresolved_alpha_evaluations": (
            failed_count if hard_failure_policy == "unresolved" else 0
        ),
        "accepted_alpha_evaluations": int(np.sum(masks.accepted)),
        "cr_unresolved_alphas": json.dumps(region.unresolved_alphas),
        "interpolated_true_alpha_accepted": interpolated_accepted,
        "direct_vs_interpolated_mismatch": (
            bool(direct_accepted != interpolated_accepted)
            if direct_available and not pd.isna(interpolated_accepted)
            else np.nan
        ),
        "interpolation_vs_region_mismatch": (
            bool(interpolated_accepted != covered)
            if not pd.isna(interpolated_accepted) and not pd.isna(covered)
            else np.nan
        ),
        "coverage_loss_vs_direct": (
            bool(direct_accepted and not covered)
            if direct_available and not pd.isna(covered)
            else np.nan
        ),
        "interpolation_loss_vs_direct": (
            bool(direct_accepted and not interpolated_accepted)
            if direct_available and not pd.isna(interpolated_accepted)
            else np.nan
        ),
        "reconstruction_loss": (
            bool(interpolated_accepted and not covered)
            if not pd.isna(interpolated_accepted) and not pd.isna(covered)
            else np.nan
        ),
        "alpha_true_exact_grid_point": exact_point,
        "exact_grid_point_consistency_mismatch": exact_mismatch,
    }


def load_direct_calibration(path: Path) -> dict[tuple[object, ...], bool | None]:
    """Load robust Hall-Sheather true-alpha acceptance by deterministic key."""
    frame = pd.read_csv(path)
    required = {
        "dgp",
        "n",
        "p",
        "pi",
        "tau",
        "rep",
        "seed",
        "covariance_variant",
        "converged",
        "rejected",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Calibration input is missing columns: {', '.join(missing)}")
    frame = frame.loc[frame["covariance_variant"] == "robust_epa_hsheather"].copy()
    if frame.empty:
        raise ValueError("Calibration input has no robust_epa_hsheather rows")
    key_columns = ["dgp", "n", "p", "pi", "tau", "rep", "seed"]
    if frame.duplicated(key_columns).any():
        raise ValueError("Calibration input contains duplicate robust design keys")

    lookup: dict[tuple[object, ...], bool | None] = {}
    for row in frame.itertuples(index=False):
        key = (
            str(row.dgp),
            int(row.n),
            int(row.p),
            float(row.pi),
            float(row.tau),
            int(row.rep),
            int(row.seed),
        )
        usable = bool(getattr(row, "usable", row.converged))
        legacy_reject = getattr(row, "hard_failure_policy", "") == "legacy_reject"
        lookup[key] = (
            bool(not row.rejected) if usable else (False if legacy_reject else None)
        )
    return lookup


def run_diagnostic(
    designs: list[Design],
    direct_lookup: dict[tuple[object, ...], bool | None],
    *,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    iteration_warning_policy: IterationWarningPolicy = "use_if_valid",
    hard_failure_policy: HardFailurePolicy = "unresolved",
) -> pd.DataFrame:
    """Evaluate fixed and adaptive grids for all deterministic designs."""
    hard_failure_policy = validate_hard_failure_policy(hard_failure_policy)
    rows: list[dict[str, Any]] = []
    for design_index, design in enumerate(designs, start=1):
        data = generate_data(design)
        alpha_true = true_alpha(design.tau, design.dgp)
        oracle_indices = get_oracle_control_indices(design.dgp, design.p)
        x_oracle = data.x[:, oracle_indices]
        cache: dict[float, tuple[AlphaEvaluation, float]] = {}

        def evaluate(alpha: float) -> AlphaEvaluation:
            alpha = float(alpha)
            if alpha not in cache:
                start = perf_counter()
                evaluation = evaluate_alpha_ch_ivqr(
                    y=data.y,
                    d=data.d,
                    x_controls=x_oracle,
                    z=data.z,
                    alpha=alpha,
                    tau=design.tau,
                    max_iter=max_iter,
                    iteration_warning_policy=iteration_warning_policy,
                )
                cache[alpha] = (evaluation, perf_counter() - start)
            return cache[alpha][0]

        key = (
            design.dgp,
            design.n,
            design.p,
            design.pi,
            design.tau,
            design.rep,
            design.seed,
        )
        if key not in direct_lookup:
            raise ValueError(f"Calibration input is missing design key: {key}")
        direct_accepted = direct_lookup[key]

        grids: dict[str, np.ndarray] = {}
        for name, fixed_grid in FIXED_GRIDS.items():
            grid = np.asarray(fixed_grid, dtype=float)
            for alpha in grid:
                evaluate(float(alpha))
            grids[name] = grid
        grids["adaptive_21_to_0.025"] = adaptive_refinement_grid(
            FIXED_GRIDS["grid_21"], evaluate
        )

        for grid_variant, grid in grids.items():
            grid_result = _evaluate_grid(
                grid_variant=grid_variant,
                alphas=grid,
                cached_evaluations=cache,
                alpha_true=alpha_true,
                direct_accepted=direct_accepted,
                hard_failure_policy=hard_failure_policy,
            )
            rows.append(
                {
                    "dgp": design.dgp,
                    "n": design.n,
                    "p": design.p,
                    "pi": design.pi,
                    "tau": design.tau,
                    "replication": design.rep,
                    "seed": design.seed,
                    "alpha_true": alpha_true,
                    "iteration_warning_policy": iteration_warning_policy,
                    "hard_failure_policy": hard_failure_policy,
                    "direct_true_alpha_accepted": (
                        np.nan if direct_accepted is None else direct_accepted
                    ),
                    "direct_true_alpha_rejected": (
                        np.nan if direct_accepted is None else not direct_accepted
                    ),
                    "direct_true_alpha_available": direct_accepted is not None,
                    **grid_result,
                }
            )
        if design_index % 25 == 0 or design_index == len(designs):
            print(
                f"Completed {design_index:,}/{len(designs):,} datasets",
                flush=True,
            )
    return pd.DataFrame(rows)


def _summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    successful = group.loc[group["converged"].astype(bool)]
    requested = len(group)
    successes = len(successful)
    coverage_resolved = successful.loc[
        successful["coverage_status"].isin(["covered", "not_covered"])
    ]
    coverage_unresolved = successful.loc[
        successful["coverage_status"] == "coverage_unresolved"
    ]

    def mean_boolean(column: str) -> float:
        values = successful[column].dropna()
        return float(values.astype(bool).mean()) if len(values) else np.nan

    return {
        "replications_requested": requested,
        "replications_successful": successes,
        "failures": requested - successes,
        "direct_true_alpha_available_rate": mean_boolean("direct_true_alpha_available"),
        "direct_true_alpha_acceptance": mean_boolean("direct_true_alpha_accepted"),
        "direct_true_alpha_rejection_rate": mean_boolean("direct_true_alpha_rejected"),
        "coverage": mean_boolean("covered"),
        "valid_coverage_denominator": len(coverage_resolved),
        "covered_count": int((coverage_resolved["coverage_status"] == "covered").sum()),
        "not_covered_count": int(
            (coverage_resolved["coverage_status"] == "not_covered").sum()
        ),
        "coverage_unresolved_count": len(coverage_unresolved),
        "conditional_coverage_resolved": (
            float((coverage_resolved["coverage_status"] == "covered").mean())
            if len(coverage_resolved)
            else np.nan
        ),
        "unresolved_replication_rate": float(
            successful["cr_status"]
            .isin(["partially_unresolved", "fully_unresolved"])
            .mean()
        )
        if successes
        else np.nan,
        "resolved_replications": int(
            successful["cr_is_numerically_resolved"].astype(bool).sum()
        )
        if "cr_is_numerically_resolved" in successful
        else int(
            (
                ~successful["cr_status"].isin(
                    ["partially_unresolved", "fully_unresolved"]
                )
            ).sum()
        ),
        "partially_unresolved_replications": int(
            (successful["cr_status"] == "partially_unresolved").sum()
        ),
        "fully_unresolved_replications": int(
            (group["cr_status"] == "fully_unresolved").sum()
        ),
        "valid_empty_region_rate": float(
            (successful["cr_status"] == "empty_valid").mean()
        )
        if successes
        else np.nan,
        "unresolved_apparent_empty_count": int(
            (
                successful["empty_cr"].astype(bool)
                & (successful["cr_status"] == "partially_unresolved")
            ).sum()
        ),
        "valid_full_grid_rate": float(
            (successful["cr_status"] == "full_grid_valid").mean()
        )
        if successes
        else np.nan,
        "unresolved_apparent_full_grid_count": int(
            (
                (successful["unresolved_alpha_evaluations"] > 0)
                & (
                    successful["accepted_alpha_evaluations"]
                    == successful["usable_alpha_evaluations"]
                )
            ).sum()
        ),
        "bias": float(successful["bias"].mean()) if successes else np.nan,
        "rmse": (
            float(np.sqrt(successful["squared_error"].mean())) if successes else np.nan
        ),
        "mean_cr_length": (
            float(successful["cr_length"].mean()) if successes else np.nan
        ),
        "median_cr_length": (
            float(successful["cr_length"].median()) if successes else np.nan
        ),
        "full_grid_acceptance_rate": mean_boolean("full_grid_accepted"),
        "empty_cr_rate": mean_boolean("empty_cr"),
        "mean_number_of_connected_components": (
            float(successful["number_of_connected_components"].mean())
            if successes
            else np.nan
        ),
        "multiple_components_rate": (
            float((successful["number_of_connected_components"] > 1).mean())
            if successes
            else np.nan
        ),
        "mean_alpha_evaluations": (
            float(successful["number_of_alpha_evaluations"].mean())
            if successes
            else np.nan
        ),
        "mean_runtime_seconds": (
            float(successful["runtime_seconds"].mean()) if successes else np.nan
        ),
        "direct_vs_interpolated_mismatch_rate": mean_boolean(
            "direct_vs_interpolated_mismatch"
        ),
        "interpolation_vs_region_mismatch_rate": mean_boolean(
            "interpolation_vs_region_mismatch"
        ),
        "coverage_loss_vs_direct_rate": mean_boolean("coverage_loss_vs_direct"),
        "interpolation_loss_vs_direct_rate": mean_boolean(
            "interpolation_loss_vs_direct"
        ),
        "reconstruction_loss_rate": mean_boolean("reconstruction_loss"),
        "exact_grid_point_cases": int(successful["alpha_true_exact_grid_point"].sum()),
        "exact_grid_point_consistency_mismatches": int(
            successful["exact_grid_point_consistency_mismatch"].sum()
        ),
        "total_failed_alpha_evaluations": int(
            successful["failed_alpha_evaluations"].sum()
        ),
        "total_alpha_evaluations": int(group["number_of_alpha_evaluations"].sum()),
        "fully_converged_evaluations": int(group["fully_converged_evaluations"].sum()),
        "iteration_limit_warning_evaluations": int(
            group["iteration_limit_warning_evaluations"].sum()
        ),
        "warning_evaluations_usable": int(group["usable_warning_evaluations"].sum()),
        "hard_failures": int(
            group["failed_alpha_evaluations"].sum()
            - group["unusable_warning_evaluations"].sum()
        ),
        "unusable_warning_fits": int(group["unusable_warning_evaluations"].sum()),
        "percentage_warnings_with_finite_usable_statistics": (
            100.0
            * float(group["usable_warning_evaluations"].sum())
            / float(group["iteration_limit_warning_evaluations"].sum())
            if group["iteration_limit_warning_evaluations"].sum()
            else np.nan
        ),
    }


def summarize_replications(replications: pd.DataFrame) -> pd.DataFrame:
    """Return design-cell rows plus overall rows for every grid variant."""
    group_columns = [
        "dgp",
        "n",
        "p",
        "pi",
        "tau",
        "iteration_warning_policy",
        "hard_failure_policy",
        "grid_variant",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in replications.groupby(group_columns, sort=False):
        rows.append(
            {
                "aggregation": "design_cell",
                **dict(zip(group_columns, keys, strict=True)),
                **_summarize_group(group),
            }
        )
    for grid_variant, group in replications.groupby("grid_variant", sort=False):
        rows.append(
            {
                "aggregation": "overall",
                "dgp": "all",
                "n": np.nan,
                "p": np.nan,
                "pi": np.nan,
                "tau": np.nan,
                "iteration_warning_policy": str(
                    group["iteration_warning_policy"].iloc[0]
                ),
                "hard_failure_policy": str(group["hard_failure_policy"].iloc[0]),
                "grid_variant": grid_variant,
                **_summarize_group(group),
            }
        )
    return pd.DataFrame(rows)


def _safe_output_path(path: Path) -> Path:
    """Protect production raw results and preceding calibration diagnostics."""
    resolved = path.resolve()
    raw_results = (Path.cwd() / "results" / "raw").resolve()
    protected = {item.resolve() for item in PROTECTED_CALIBRATION_OUTPUTS}
    if resolved == raw_results or raw_results in resolved.parents:
        raise ValueError("Grid diagnostic output must not be written under results/raw")
    if resolved in protected:
        raise ValueError("Grid diagnostic must not overwrite prior calibration output")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgps", nargs="+", default=["dgp1", "dgp2"])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--p", type=int, default=200)
    parser.add_argument("--pi", nargs="+", type=float, default=[0.1, 1.0])
    parser.add_argument("--tau", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_QUANTREG_MAX_ITER)
    parser.add_argument(
        "--iteration-warning-policy",
        choices=ITERATION_WARNING_POLICIES,
        default="use_if_valid",
        help=(
            "QuantReg warning policy: use_if_valid is the production default; "
            "reject reproduces legacy results"
        ),
    )
    parser.add_argument(
        "--hard-failure-policy",
        choices=HARD_FAILURE_POLICIES,
        default="unresolved",
        help=(
            "unresolved is the production default; legacy_reject reproduces "
            "sentinel rejection"
        ),
    )
    parser.add_argument(
        "--calibration-replications",
        type=Path,
        default=None,
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--replication-output", type=Path, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    policy = args.iteration_warning_policy
    hard_failure_policy = args.hard_failure_policy
    policy_stem = f"{policy}_{hard_failure_policy}"
    summary_output = _safe_output_path(
        args.summary_output
        or POLICY_OUTPUT_DIRECTORY / f"oracle_grid_resolution_{policy_stem}_summary.csv"
    )
    replication_output = _safe_output_path(
        args.replication_output
        or POLICY_OUTPUT_DIRECTORY
        / f"oracle_grid_resolution_{policy_stem}_replications.csv"
    )
    calibration_input = args.calibration_replications or (
        POLICY_OUTPUT_DIRECTORY / f"oracle_calibration_{policy_stem}_replications.csv"
    )
    direct_lookup = load_direct_calibration(calibration_input)
    designs = make_simulation_grid(
        dgps=tuple(args.dgps),
        n_values=(args.n,),
        p_values=(args.p,),
        pi_values=tuple(args.pi),
        taus=tuple(args.tau),
        reps=args.reps,
        base_seed=args.base_seed,
    )
    print(
        f"Oracle grid-resolution diagnostic: {len(designs):,} datasets, "
        f"4 grid variants, base seed {args.base_seed}"
    )
    print("QuantReg covariance: robust / epa / hsheather (production default)")
    print(f"Iteration-warning policy: {policy}")
    print(f"Hard-failure policy: {hard_failure_policy}")
    print(f"Direct true-alpha calibration input: {calibration_input}")

    replications = run_diagnostic(
        designs,
        direct_lookup,
        max_iter=args.max_iter,
        iteration_warning_policy=policy,
        hard_failure_policy=hard_failure_policy,
    )
    summary = summarize_replications(replications)
    replication_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    replications.to_csv(replication_output, index=False)
    summary.to_csv(summary_output, index=False)

    overall = summary.loc[summary["aggregation"] == "overall"]
    display_columns = [
        "grid_variant",
        "replications_successful",
        "direct_true_alpha_acceptance",
        "coverage",
        "coverage_loss_vs_direct_rate",
        "interpolation_vs_region_mismatch_rate",
        "exact_grid_point_consistency_mismatches",
        "mean_alpha_evaluations",
        "mean_runtime_seconds",
    ]
    print("\nOverall comparison across all requested design cells:")
    print(overall[display_columns].to_string(index=False, float_format="%.4f"))
    print(f"\nWrote {summary_output}")
    print(f"Wrote {replication_output}")


if __name__ == "__main__":
    main()
