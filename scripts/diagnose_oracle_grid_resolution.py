"""Diagnose Oracle inverse-IVQR confidence-region grid resolution.

The script compares fixed and adaptively refined alpha grids while leaving the
production Oracle estimator unchanged.  Direct true-alpha acceptance is read
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
from ivqr.ch_inverse import AlphaEvaluation, evaluate_alpha_ch_ivqr
from ivqr.confidence_regions import (
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    sanitize_grid_statistics,
)
from simulation.config import DEFAULT_BASE_SEED, DEFAULT_QUANTREG_MAX_ITER
from simulation.runner import make_simulation_grid


DEFAULT_SUMMARY_OUTPUT = Path("results/diagnostics/oracle_grid_resolution_summary.csv")
DEFAULT_REPLICATION_OUTPUT = Path(
    "results/diagnostics/oracle_grid_resolution_replications.csv"
)
DEFAULT_CALIBRATION_INPUT = Path(
    "results/diagnostics/oracle_calibration_replications.csv"
)
PROTECTED_CALIBRATION_OUTPUTS = (
    Path("results/diagnostics/oracle_calibration_summary.csv"),
    DEFAULT_CALIBRATION_INPUT,
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
    evaluate_statistic: Callable[[float], float],
    *,
    critical_value: float = CRITICAL_VALUE,
    max_spacing: float = ADAPTIVE_MAX_SPACING,
) -> np.ndarray:
    """Refine acceptance-transition intervals until their spacing is small."""
    if max_spacing <= 0.0 or not np.isfinite(max_spacing):
        raise ValueError("max_spacing must be positive and finite")
    points = np.unique(np.asarray(initial_grid, dtype=float))
    if points.ndim != 1 or points.size < 2 or not np.all(np.isfinite(points)):
        raise ValueError("initial_grid must contain at least two finite points")
    if not np.all(np.diff(points) > 0.0):
        raise ValueError("initial_grid must be strictly increasing")

    statistics = {
        float(alpha): float(evaluate_statistic(float(alpha))) for alpha in points
    }
    while True:
        transitions: list[tuple[float, float]] = []
        for left, right in zip(points[:-1], points[1:], strict=True):
            left_value = statistics[float(left)]
            right_value = statistics[float(right)]
            changes = (left_value <= critical_value) != (right_value <= critical_value)
            if changes and right - left > max_spacing + 1e-12:
                transitions.append((float(left), float(right)))
        if not transitions:
            return points
        for left, right in transitions:
            midpoint = float((left + right) / 2.0)
            statistics[midpoint] = float(evaluate_statistic(midpoint))
        points = np.array(sorted(statistics), dtype=float)


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


def _components_json(blocks: tuple[tuple[float, float], ...]) -> str:
    return json.dumps(
        [[lower, upper] for lower, upper in blocks], separators=(",", ":")
    )


def _evaluate_grid(
    *,
    grid_variant: str,
    alphas: np.ndarray,
    cached_evaluations: dict[float, tuple[AlphaEvaluation, float]],
    alpha_true: float,
    direct_accepted: bool | None,
) -> dict[str, Any]:
    """Construct one production-style confidence region from cached evaluations."""
    raw_statistics = np.array(
        [cached_evaluations[float(alpha)][0].statistic for alpha in alphas],
        dtype=float,
    )
    converged_flags = np.array(
        [cached_evaluations[float(alpha)][0].converged for alpha in alphas],
        dtype=bool,
    )
    statistics, failed_count = sanitize_grid_statistics(raw_statistics, converged_flags)
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
            "runtime_seconds": runtime,
            "converged": False,
            "failure_reason": "All alpha-grid evaluations failed",
            "interpolated_true_alpha_accepted": np.nan,
            "direct_vs_interpolated_mismatch": np.nan,
            "interpolation_vs_region_mismatch": np.nan,
            "coverage_loss_vs_direct": np.nan,
            "interpolation_loss_vs_direct": np.nan,
            "reconstruction_loss": np.nan,
            "alpha_true_exact_grid_point": False,
            "exact_grid_point_consistency_mismatch": False,
        }

    alpha_hat, _minimum, _boundary = argmin_grid(alphas, statistics)
    region = invert_score_test(
        alphas=alphas,
        statistics=statistics,
        critical_value=CRITICAL_VALUE,
        alpha_true=alpha_true,
        statistic_reference=0.0,
        inversion_type="absolute",
    )
    accepted = statistics <= CRITICAL_VALUE
    interpolated_accepted = interpolated_acceptance_at_alpha(
        alphas, statistics, alpha_true
    )
    covered = bool(region.covers_true)
    exact_point, exact_mismatch = exact_grid_consistency(
        alphas, accepted, alpha_true, covered, direct_accepted
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
        "cr_components": _components_json(region.blocks),
        "full_grid_accepted": bool(np.all(accepted)),
        "empty_cr": region.empty,
        "number_of_connected_components": region.n_blocks,
        "number_of_alpha_evaluations": len(alphas),
        "failed_alpha_evaluations": failed_count,
        "runtime_seconds": runtime,
        # Match production behavior: partial alpha failures are sanitized, while
        # a replication fails only when every alpha evaluation fails.
        "converged": True,
        "failure_reason": (
            ""
            if failed_count == 0
            else f"ok; failed_alpha_points={failed_count}/{len(alphas)}"
        ),
        "interpolated_true_alpha_accepted": interpolated_accepted,
        "direct_vs_interpolated_mismatch": (
            bool(direct_accepted != interpolated_accepted)
            if direct_available
            else np.nan
        ),
        "interpolation_vs_region_mismatch": bool(interpolated_accepted != covered),
        "coverage_loss_vs_direct": (
            bool(direct_accepted and not covered) if direct_available else np.nan
        ),
        "interpolation_loss_vs_direct": (
            bool(direct_accepted and not interpolated_accepted)
            if direct_available
            else np.nan
        ),
        "reconstruction_loss": bool(interpolated_accepted and not covered),
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
        lookup[key] = bool(not row.rejected) if bool(row.converged) else None
    return lookup


def run_diagnostic(
    designs: list[Design],
    direct_lookup: dict[tuple[object, ...], bool | None],
    *,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
) -> pd.DataFrame:
    """Evaluate fixed and adaptive grids for all deterministic designs."""
    rows: list[dict[str, Any]] = []
    for design_index, design in enumerate(designs, start=1):
        data = generate_data(design)
        alpha_true = true_alpha(design.tau, design.dgp)
        oracle_indices = get_oracle_control_indices(design.dgp, design.p)
        x_oracle = data.x[:, oracle_indices]
        cache: dict[float, tuple[AlphaEvaluation, float]] = {}

        def evaluate(alpha: float) -> float:
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
                )
                cache[alpha] = (evaluation, perf_counter() - start)
            return cache[alpha][0].statistic

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
    }


def summarize_replications(replications: pd.DataFrame) -> pd.DataFrame:
    """Return design-cell rows plus overall rows for every grid variant."""
    group_columns = ["dgp", "n", "p", "pi", "tau", "grid_variant"]
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
        "--calibration-replications",
        type=Path,
        default=DEFAULT_CALIBRATION_INPUT,
    )
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument(
        "--replication-output", type=Path, default=DEFAULT_REPLICATION_OUTPUT
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary_output = _safe_output_path(args.summary_output)
    replication_output = _safe_output_path(args.replication_output)
    direct_lookup = load_direct_calibration(args.calibration_replications)
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
    print(f"Direct true-alpha calibration input: {args.calibration_replications}")

    replications = run_diagnostic(designs, direct_lookup, max_iter=args.max_iter)
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
