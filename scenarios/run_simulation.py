"""Run the IVQR Monte Carlo simulation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import subprocess
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from simulation.config import (  # noqa: E402
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_BASE_SEED,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_DML_QUANTILE_PENALTY,
    DEFAULT_DML_QUANTILE_SOLVER,
    DEFAULT_DML_RIDGE_ALPHA,
    DEFAULT_N_JOBS,
    DEFAULT_QUANTREG_MAX_ITER,
    DEFAULT_SELECTION_LASSO_MULTIPLIER,
    DEFAULT_GRID_STRATEGY,
    DEFAULT_REFINEMENT_TOLERANCE,
    DEFAULT_MAX_REFINEMENT_DEPTH,
    DEFAULT_MAX_ALPHA_EVALUATIONS,
    DEFAULT_ITERATION_WARNING_POLICY,
    DEFAULT_HARD_FAILURE_POLICY,
    DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    DEFAULT_ALPHA_HAT_GRID,
    DGPS,
    FAST_OUTPUT,
    N_VALUES,
    PI_VALUES,
    P_VALUES,
    R_FAST,
    TAUS,
    build_estimator_run_config,
    runner_kwargs,
)
from simulation.runner import (  # noqa: E402
    MULTI_ESTIMATOR_REMOVAL_MESSAGE,
    SEED_RULE_TEXT,
    filter_completed_designs,
    make_simulation_grid,
    normalize_estimator_names,
    run_simulation_batch,
)
from simulation.results import RESULT_SCHEMA_VERSION  # noqa: E402


VALID_MODES = ("fast",)


def _git_metadata() -> dict[str, object | None]:
    """Return reproducibility metadata, tolerating non-Git environments."""
    def run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    commit = run_git("rev-parse", "HEAD")
    branch = run_git("branch", "--show-current")
    status = run_git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": None if status is None else bool(status),
    }


def _default_output_for_mode(mode: str) -> Path:
    if mode == "fast":
        return Path(FAST_OUTPUT)
    raise ValueError(f"Unknown mode: {mode}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the IVQR Monte Carlo simulation."
    )
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
    parser.add_argument("--estimators", nargs="+", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument(
        "--rep-start",
        type=int,
        default=0,
        help="First replication index to run, inclusive. Default: 0.",
    )
    parser.add_argument(
        "--rep-end",
        type=int,
        default=None,
        help="Last replication index to run, inclusive. Default: reps - 1.",
    )
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_MIN)
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_MAX)
    parser.add_argument("--alpha-grid-size", type=int, default=DEFAULT_ALPHA_GRID_SIZE)
    parser.add_argument(
        "--grid-strategy", choices=("fixed", "adaptive"), default=DEFAULT_GRID_STRATEGY
    )
    parser.add_argument(
        "--refinement-tolerance", type=float, default=DEFAULT_REFINEMENT_TOLERANCE
    )
    parser.add_argument(
        "--max-refinement-depth", type=int, default=DEFAULT_MAX_REFINEMENT_DEPTH
    )
    parser.add_argument(
        "--max-alpha-evaluations", type=int, default=DEFAULT_MAX_ALPHA_EVALUATIONS
    )
    parser.add_argument(
        "--iteration-warning-policy",
        choices=("use_if_valid", "reject"),
        default=DEFAULT_ITERATION_WARNING_POLICY,
    )
    parser.add_argument(
        "--hard-failure-policy",
        choices=("unresolved", "legacy_reject"),
        default=DEFAULT_HARD_FAILURE_POLICY,
    )
    parser.add_argument(
        "--adaptive-midpoint-probe",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    )
    parser.add_argument(
        "--alpha-hat-grid",
        choices=("initial", "all_evaluated"),
        default=DEFAULT_ALPHA_HAT_GRID,
    )
    parser.add_argument(
        "--critical-value-multiplier",
        type=float,
        default=DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    )
    parser.add_argument(
        "--selection-lasso-multiplier",
        type=float,
        default=DEFAULT_SELECTION_LASSO_MULTIPLIER,
        help=(
            "Multiplies the LassoCV-selected penalty used in post-selection "
            "control selection. Affects only the post_selection estimator. "
            "Default 1.0 preserves baseline behavior."
        ),
    )
    parser.add_argument("--dml-k-folds", type=int, default=DEFAULT_DML_K_FOLDS)
    parser.add_argument(
        "--dml-quantile-penalty",
        type=float,
        default=DEFAULT_DML_QUANTILE_PENALTY,
        help="Penalty alpha for DML quantile nuisance fits. Default: 0.01.",
    )
    parser.add_argument(
        "--dml-ridge-alpha",
        type=float,
        default=DEFAULT_DML_RIDGE_ALPHA,
        help="Ridge alpha for DML instrument residualization. Default: 1.0.",
    )
    parser.add_argument(
        "--dml-quantile-solver",
        default=DEFAULT_DML_QUANTILE_SOLVER,
        choices=("highs-ds", "highs-ipm", "highs", "interior-point", "revised simplex"),
        help='Solver for DML QuantileRegressor nuisance fits. Default: "highs".',
    )
    parser.add_argument("--quantreg-max-iter", type=int, default=DEFAULT_QUANTREG_MAX_ITER)
    parser.add_argument("--dgps", nargs="+", default=None)
    parser.add_argument("--n-values", nargs="+", type=int, default=None)
    parser.add_argument("--p-values", nargs="+", type=int, default=None)
    parser.add_argument("--pi-values", nargs="+", type=float, default=None)
    parser.add_argument("--taus", nargs="+", type=float, default=None)
    parser.add_argument("--max-designs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-quantreg-warnings", action="store_true")
    arguments = list(sys.argv[1:] if argv is None else argv)
    full_mode_requested = "--mode=full" in arguments or any(
        token == "--mode" and index + 1 < len(arguments) and arguments[index + 1] == "full"
        for index, token in enumerate(arguments)
    )
    if full_mode_requested:
        parser.error(MULTI_ESTIMATOR_REMOVAL_MESSAGE)
    return parser.parse_args(arguments)


def _apply_defaults(args: argparse.Namespace) -> None:
    args.estimators = normalize_estimator_names(args.estimators)
    args.dgps = list(DGPS) if args.dgps is None else args.dgps
    args.n_values = list(N_VALUES) if args.n_values is None else args.n_values
    args.p_values = list(P_VALUES) if args.p_values is None else args.p_values
    args.pi_values = list(PI_VALUES) if args.pi_values is None else args.pi_values
    args.taus = list(TAUS) if args.taus is None else args.taus
    args.reps = R_FAST if args.reps is None else args.reps
    args.rep_end = args.reps - 1 if args.rep_end is None else args.rep_end
    args.output = _default_output_for_mode(args.mode) if args.output is None else Path(args.output)
    args.manifest = None if args.manifest is None else Path(args.manifest)


def _validate_args(args: argparse.Namespace) -> None:
    if len(args.estimators) != 1:
        raise ValueError(MULTI_ESTIMATOR_REMOVAL_MESSAGE)
    if args.reps < 1:
        raise ValueError("--reps must be at least 1")
    if args.rep_start < 0:
        raise ValueError("--rep-start must be at least 0")
    if args.rep_end < args.rep_start:
        raise ValueError("--rep-end must be greater than or equal to --rep-start")
    if args.rep_end >= args.reps:
        raise ValueError("--rep-end must be less than --reps")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_designs is not None and args.max_designs < 1:
        raise ValueError("--max-designs must be at least 1")
    if args.alpha_grid_size < 3:
        raise ValueError("--alpha-grid-size must be at least 3")
    if args.alpha_max <= args.alpha_min:
        raise ValueError("--alpha-max must exceed --alpha-min")
    if not np.isfinite(args.refinement_tolerance) or args.refinement_tolerance <= 0:
        raise ValueError("--refinement-tolerance must be positive and finite")
    if args.max_refinement_depth < 0:
        raise ValueError("--max-refinement-depth must be nonnegative")
    if args.max_alpha_evaluations < args.alpha_grid_size:
        raise ValueError("--max-alpha-evaluations must cover the initial grid")
    if args.dml_k_folds < 2:
        raise ValueError("--dml-k-folds must be at least 2")
    if not np.isfinite(args.dml_quantile_penalty) or args.dml_quantile_penalty < 0:
        raise ValueError("--dml-quantile-penalty must be nonnegative")
    if not np.isfinite(args.dml_ridge_alpha) or args.dml_ridge_alpha < 0:
        raise ValueError("--dml-ridge-alpha must be nonnegative")
    if args.quantreg_max_iter < 1:
        raise ValueError("--quantreg-max-iter must be at least 1")
    if args.critical_value_multiplier <= 0:
        raise ValueError("--critical-value-multiplier must be positive")
    if (
        not np.isfinite(args.selection_lasso_multiplier)
        or args.selection_lasso_multiplier <= 0
    ):
        raise ValueError("--selection-lasso-multiplier must be positive")
    if args.resume and args.manifest is None:
        raise ValueError("--resume requires --manifest")


def _resume_signature(args: argparse.Namespace) -> dict[str, object]:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "mode": args.mode,
        "dgps": list(args.dgps),
        "n_values": list(args.n_values),
        "p_values": list(args.p_values),
        "pi_values": list(args.pi_values),
        "taus": list(args.taus),
        "reps": args.reps,
        "rep_start": args.rep_start,
        "rep_end": args.rep_end,
        "base_seed": args.base_seed,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_grid_size": args.alpha_grid_size,
        "grid_strategy": args.grid_strategy,
        "refinement_tolerance": args.refinement_tolerance,
        "max_refinement_depth": args.max_refinement_depth,
        "max_alpha_evaluations": args.max_alpha_evaluations,
        "iteration_warning_policy": args.iteration_warning_policy,
        "hard_failure_policy": args.hard_failure_policy,
        "adaptive_midpoint_probe": args.adaptive_midpoint_probe,
        "alpha_hat_grid": args.alpha_hat_grid,
        "critical_value_multiplier": args.critical_value_multiplier,
        "selection_lasso_multiplier": args.selection_lasso_multiplier,
        "estimators": list(args.estimators),
        "dml_k_folds": args.dml_k_folds,
        "dml_quantile_penalty": args.dml_quantile_penalty,
        "dml_ridge_alpha": args.dml_ridge_alpha,
        "dml_quantile_solver": args.dml_quantile_solver,
        "quantreg_max_iter": args.quantreg_max_iter,
    }


def _validate_resume_manifest(args: argparse.Namespace) -> None:
    if not args.manifest.exists():
        raise FileNotFoundError("--resume requires an existing --manifest file")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    previous = payload.get("resume_signature")
    parameters = payload.get("parameters")
    previous_mode = previous.get("mode") if isinstance(previous, dict) else None
    if previous_mode is None and isinstance(parameters, dict):
        previous_mode = parameters.get("mode")
    if previous_mode == "full":
        raise ValueError(
            f"Full-mode resume is no longer supported.\n{MULTI_ESTIMATOR_REMOVAL_MESSAGE}"
        )
    if previous is not None and previous != _resume_signature(args):
        raise ValueError("Manifest resume signature does not match current run settings")


def _write_manifest(
    args: argparse.Namespace,
    *,
    total_designs: int,
    pending_designs: int,
    designs_in_run: int,
    alphas: np.ndarray,
) -> None:
    if args.manifest is None:
        return
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "parameters": vars(args),
        "result_schema_version": RESULT_SCHEMA_VERSION,
        **_git_metadata(),
        "resume_signature": _resume_signature(args),
        "base_seed": args.base_seed,
        "seed_rule": SEED_RULE_TEXT,
        "total_designs": total_designs,
        "pending_designs": pending_designs,
        "designs_in_run": designs_in_run,
        "estimators": list(args.estimators),
        "iteration_warning_policy": args.iteration_warning_policy,
        "hard_failure_policy": args.hard_failure_policy,
        "grid_strategy": args.grid_strategy,
        "adaptive_midpoint_probe": args.adaptive_midpoint_probe,
        "refinement_tolerance": args.refinement_tolerance,
        "max_refinement_depth": args.max_refinement_depth,
        "max_alpha_evaluations": args.max_alpha_evaluations,
        "alpha_hat_grid": args.alpha_hat_grid,
        "confidence_level": 0.95,
        "lock_file": {
            "path": "pixi.lock",
            "exists": (PROJECT_ROOT / "pixi.lock").is_file(),
        },
        "alpha_grid": {
            "size": int(alphas.size),
            "min": float(alphas.min()),
            "max": float(alphas.max()),
            "values": [float(value) for value in alphas],
        },
        "output_path": str(args.output),
    }
    args.manifest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _print_dry_run(
    args: argparse.Namespace,
    designs_in_run: int,
    alphas: np.ndarray,
    first_design_seed: int | None,
) -> None:
    print(f"Mode: {args.mode}")
    print(f"Replications per design: {args.reps}")
    print(f"Replication block: {args.rep_start} to {args.rep_end}")
    print(f"Base seed: {args.base_seed}")
    print("Seed rule: deterministic by design cell, independent of estimator/order")
    print(f"First design seed: {first_design_seed}")
    print(f"DGPs: {', '.join(args.dgps)}")
    print(f"n values: {', '.join(map(str, args.n_values))}")
    print(f"p values: {', '.join(map(str, args.p_values))}")
    print(f"pi values: {', '.join(map(str, args.pi_values))}")
    print(f"taus: {', '.join(map(str, args.taus))}")
    print(f"Estimators: {', '.join(args.estimators)}")
    print(f"Post-selection Lasso multiplier: {args.selection_lasso_multiplier}")
    print(f"CH grid strategy: {args.grid_strategy}")
    print(f"CH refinement tolerance: {args.refinement_tolerance}")
    print(f"CH midpoint probe: {args.adaptive_midpoint_probe}")
    print(f"CH point-estimate grid: {args.alpha_hat_grid}")
    print(f"DML quantile penalty: {args.dml_quantile_penalty}")
    print(f"DML ridge alpha: {args.dml_ridge_alpha}")
    print(f"DML quantile solver: {args.dml_quantile_solver}")
    print(
        "Alpha grid: "
        f"min={args.alpha_min}, max={args.alpha_max}, size={args.alpha_grid_size}, "
        f"step={(args.alpha_max - args.alpha_min) / (args.alpha_grid_size - 1):g}"
    )
    print(f"Expected design rows: {designs_in_run}")
    print(f"Output: {args.output}")
    print(f"Manifest: {args.manifest}")


def _count_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return len(pd.read_csv(path, usecols=["estimator"]))
    except (ValueError, pd.errors.EmptyDataError):
        return None


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _apply_defaults(args)
    _validate_args(args)
    run_config = build_estimator_run_config(args)
    if args.rerun_failed and not args.resume:
        print("--rerun-failed has no effect without --resume.")

    alpha_grid = run_config.alpha_grid
    execution = run_config.execution
    design = run_config.design
    alphas = np.linspace(
        alpha_grid.alpha_min, alpha_grid.alpha_max, alpha_grid.alpha_grid_size
    )
    designs = make_simulation_grid(
        dgps=design.dgps,
        n_values=design.sample_sizes,
        p_values=design.dimensions,
        pi_values=design.instrument_strengths,
        taus=design.quantiles,
        reps=execution.reps,
        base_seed=execution.base_seed,
        rep_start=execution.rep_start,
        rep_end=execution.rep_end,
    )
    if args.resume:
        _validate_resume_manifest(args)
        pending_designs = filter_completed_designs(
            designs,
            args.output,
            estimators=tuple(args.estimators),
            rerun_failed=args.rerun_failed,
        )
    else:
        if args.output.exists() and not args.dry_run:
            raise FileExistsError(
                f"Output file already exists: {args.output}. Use --resume or choose a new --output."
            )
        pending_designs = designs
    designs_to_run = (
        pending_designs[: execution.max_designs]
        if execution.max_designs
        else pending_designs
    )

    if args.dry_run:
        first_design_seed = designs_to_run[0].seed if designs_to_run else None
        _print_dry_run(args, len(designs_to_run), alphas, first_design_seed)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        args,
        total_designs=len(designs),
        pending_designs=len(pending_designs),
        designs_in_run=len(designs_to_run),
        alphas=alphas,
    )

    start = time.perf_counter()
    completed = 0
    print(f"Running estimators: {', '.join(args.estimators)}")
    effective_runner_kwargs = runner_kwargs(run_config)
    # The generic CLI historically forwards and validates irrelevant options too.
    # Retain that compatibility without storing them on estimator-owned configs.
    effective_runner_kwargs.setdefault("quantreg_max_iter", args.quantreg_max_iter)
    effective_runner_kwargs.setdefault("dml_k_folds", args.dml_k_folds)
    effective_runner_kwargs.setdefault(
        "dml_quantile_penalty", args.dml_quantile_penalty
    )
    effective_runner_kwargs.setdefault("dml_ridge_alpha", args.dml_ridge_alpha)
    effective_runner_kwargs.setdefault(
        "dml_quantile_solver", args.dml_quantile_solver
    )
    effective_runner_kwargs.setdefault(
        "selection_lasso_multiplier", args.selection_lasso_multiplier
    )
    effective_runner_kwargs.setdefault(
        "show_quantreg_warnings", args.show_quantreg_warnings
    )
    effective_runner_kwargs.setdefault("grid_strategy", args.grid_strategy)
    effective_runner_kwargs.setdefault(
        "refinement_tolerance", args.refinement_tolerance
    )
    effective_runner_kwargs.setdefault(
        "max_refinement_depth", args.max_refinement_depth
    )
    effective_runner_kwargs.setdefault(
        "max_alpha_evaluations", args.max_alpha_evaluations
    )
    effective_runner_kwargs.setdefault(
        "iteration_warning_policy", args.iteration_warning_policy
    )
    effective_runner_kwargs.setdefault(
        "hard_failure_policy", args.hard_failure_policy
    )
    effective_runner_kwargs.setdefault(
        "adaptive_midpoint_probe", args.adaptive_midpoint_probe
    )
    effective_runner_kwargs.setdefault("alpha_hat_grid", args.alpha_hat_grid)
    for batch_start in range(0, len(designs_to_run), execution.batch_size):
        batch = designs_to_run[
            batch_start : batch_start + execution.batch_size
        ]
        run_simulation_batch(
            batch,
            alphas,
            estimators=tuple(args.estimators),
            output_path=args.output,
            append=args.resume or completed > 0,
            n_jobs=execution.n_jobs,
            **effective_runner_kwargs,
        )
        completed += len(batch)
        elapsed = time.perf_counter() - start
        print(f"Completed {completed}/{len(designs_to_run)} designs in {elapsed:.2f} seconds")

    final_rows = _count_rows(args.output)
    print(f"Mode: {args.mode}")
    print(f"Completed designs: {completed}")
    print(f"Pending before max-designs: {len(pending_designs)}")
    print(f"Output: {args.output}")
    print(f"Final row count: {final_rows}" if final_rows is not None else "Final row count unavailable")


if __name__ == "__main__":
    main()
