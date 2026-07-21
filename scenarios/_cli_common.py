"""Shared orchestration for estimator-locked simulation CLIs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

from simulation.config import (
    DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    DEFAULT_ALPHA_HAT_GRID,
    DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_DML_QUANTILE_PENALTY,
    DEFAULT_DML_QUANTILE_SOLVER,
    DEFAULT_DML_RIDGE_ALPHA,
    DEFAULT_GRID_STRATEGY,
    DEFAULT_HARD_FAILURE_POLICY,
    DEFAULT_ITERATION_WARNING_POLICY,
    DEFAULT_MAX_ALPHA_EVALUATIONS,
    DEFAULT_MAX_REFINEMENT_DEPTH,
    DEFAULT_QUANTREG_MAX_ITER,
    DEFAULT_REFINEMENT_TOLERANCE,
    DEFAULT_SELECTION_LASSO_MULTIPLIER,
    DGPS,
    EstimatorRunConfig,
    FAST_OUTPUT,
    N_VALUES,
    PI_VALUES,
    P_VALUES,
    R_FAST,
    TAUS,
    build_dml_run_config,
    build_oracle_run_config,
    build_post_selection_run_config,
    runner_kwargs,
)
from simulation.runner import (
    MULTI_ESTIMATOR_REMOVAL_MESSAGE,
    SEED_RULE_TEXT,
    filter_completed_designs,
    make_simulation_grid,
    run_simulation_batch,
)
from simulation.results import RESULT_SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_MANIFEST_MODE = "fast"


def build_run_config(
    estimator: str, namespace: argparse.Namespace
) -> EstimatorRunConfig:
    builders = {
        "oracle": build_oracle_run_config,
        "post_selection": build_post_selection_run_config,
        "dml": build_dml_run_config,
    }
    try:
        builder = builders[estimator]
    except KeyError as exc:
        raise ValueError(f"Unknown dedicated estimator: {estimator}") from exc
    return builder(namespace)


def prepare_namespace(
    estimator: str, parsed: argparse.Namespace
) -> argparse.Namespace:
    """Restore historical internal fields without exposing generic CLI flags."""
    compatibility = {
        "mode": HISTORICAL_MANIFEST_MODE,
        "estimators": [estimator],
        "grid_strategy": DEFAULT_GRID_STRATEGY,
        "refinement_tolerance": DEFAULT_REFINEMENT_TOLERANCE,
        "max_refinement_depth": DEFAULT_MAX_REFINEMENT_DEPTH,
        "max_alpha_evaluations": DEFAULT_MAX_ALPHA_EVALUATIONS,
        "iteration_warning_policy": DEFAULT_ITERATION_WARNING_POLICY,
        "hard_failure_policy": DEFAULT_HARD_FAILURE_POLICY,
        "adaptive_midpoint_probe": DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
        "alpha_hat_grid": DEFAULT_ALPHA_HAT_GRID,
        "critical_value_multiplier": DEFAULT_CRITICAL_VALUE_MULTIPLIER,
        "selection_lasso_multiplier": DEFAULT_SELECTION_LASSO_MULTIPLIER,
        "dml_k_folds": DEFAULT_DML_K_FOLDS,
        "dml_quantile_penalty": DEFAULT_DML_QUANTILE_PENALTY,
        "dml_ridge_alpha": DEFAULT_DML_RIDGE_ALPHA,
        "dml_quantile_solver": DEFAULT_DML_QUANTILE_SOLVER,
        "quantreg_max_iter": DEFAULT_QUANTREG_MAX_ITER,
        "show_quantreg_warnings": False,
    }
    compatibility.update(vars(parsed))
    args = argparse.Namespace(**compatibility)
    args.dgps = list(DGPS) if args.dgps is None else args.dgps
    args.n_values = list(N_VALUES) if args.n_values is None else args.n_values
    args.p_values = list(P_VALUES) if args.p_values is None else args.p_values
    args.pi_values = list(PI_VALUES) if args.pi_values is None else args.pi_values
    args.taus = list(TAUS) if args.taus is None else args.taus
    args.reps = R_FAST if args.reps is None else args.reps
    args.rep_end = args.reps - 1 if args.rep_end is None else args.rep_end
    args.output = Path(FAST_OUTPUT) if args.output is None else Path(args.output)
    args.manifest = None if args.manifest is None else Path(args.manifest)
    return args


def validate_namespace(args: argparse.Namespace) -> None:
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


def resume_signature(args: argparse.Namespace) -> dict[str, object]:
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


def validate_resume_manifest(args: argparse.Namespace) -> None:
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
            f"Full-mode resume is no longer supported.\n"
            f"{MULTI_ESTIMATOR_REMOVAL_MESSAGE}"
        )
    if previous is not None and previous != resume_signature(args):
        raise ValueError("Manifest resume signature does not match current run settings")


def git_metadata() -> dict[str, object | None]:
    def run_git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments], cwd=PROJECT_ROOT, capture_output=True,
                text=True, check=False,
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


def write_manifest(
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
        **git_metadata(),
        "resume_signature": resume_signature(args),
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
    args.manifest.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


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
        f"min={args.alpha_min}, max={args.alpha_max}, "
        f"size={args.alpha_grid_size}, "
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


def execute(estimator: str, parsed: argparse.Namespace) -> None:
    args = prepare_namespace(estimator, parsed)
    validate_namespace(args)
    config = build_run_config(estimator, args)
    if args.rerun_failed and not args.resume:
        print("--rerun-failed has no effect without --resume.")
    alphas = np.linspace(
        config.alpha_grid.alpha_min,
        config.alpha_grid.alpha_max,
        config.alpha_grid.alpha_grid_size,
    )
    designs = make_simulation_grid(
        dgps=config.design.dgps,
        n_values=config.design.sample_sizes,
        p_values=config.design.dimensions,
        pi_values=config.design.instrument_strengths,
        taus=config.design.quantiles,
        reps=config.execution.reps,
        base_seed=config.execution.base_seed,
        rep_start=config.execution.rep_start,
        rep_end=config.execution.rep_end,
    )
    if args.resume:
        validate_resume_manifest(args)
        pending = filter_completed_designs(
            designs, args.output, estimators=(estimator,),
            rerun_failed=args.rerun_failed,
        )
    else:
        if args.output.exists() and not args.dry_run:
            raise FileExistsError(
                f"Output file already exists: {args.output}. "
                "Use --resume or choose a new --output."
            )
        pending = designs
    to_run = (
        pending[: config.execution.max_designs]
        if config.execution.max_designs
        else pending
    )
    if args.dry_run:
        first_seed = to_run[0].seed if to_run else None
        _print_dry_run(args, len(to_run), alphas, first_seed)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(
        args, total_designs=len(designs), pending_designs=len(pending),
        designs_in_run=len(to_run), alphas=alphas,
    )
    started = time.perf_counter()
    completed = 0
    print(f"Running estimators: {estimator}")
    effective = runner_kwargs(config)
    for name in (
        "quantreg_max_iter", "dml_k_folds", "dml_quantile_penalty",
        "dml_ridge_alpha", "dml_quantile_solver",
        "selection_lasso_multiplier", "show_quantreg_warnings",
        "grid_strategy", "refinement_tolerance", "max_refinement_depth",
        "max_alpha_evaluations", "iteration_warning_policy",
        "hard_failure_policy", "adaptive_midpoint_probe", "alpha_hat_grid",
    ):
        effective.setdefault(name, getattr(args, name))
    for start in range(0, len(to_run), config.execution.batch_size):
        batch = to_run[start : start + config.execution.batch_size]
        run_simulation_batch(
            batch, alphas, estimators=(estimator,), output_path=args.output,
            append=args.resume or completed > 0, n_jobs=config.execution.n_jobs,
            **effective,
        )
        completed += len(batch)
        elapsed = time.perf_counter() - started
        print(f"Completed {completed}/{len(to_run)} designs in {elapsed:.2f} seconds")
    final_rows = _count_rows(args.output)
    print(f"Mode: {args.mode}")
    print(f"Completed designs: {completed}")
    print(f"Pending before max-designs: {len(pending)}")
    print(f"Output: {args.output}")
    print(
        f"Final row count: {final_rows}"
        if final_rows is not None
        else "Final row count unavailable"
    )


__all__ = [
    "build_run_config", "execute", "git_metadata", "prepare_namespace",
    "resume_signature", "validate_namespace", "validate_resume_manifest",
]
