"""Run the IVQR Monte Carlo simulation."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from reporting.figures import write_figures  # noqa: E402
from reporting.summaries import aggregate_results_file  # noqa: E402
from reporting.tables import write_tables  # noqa: E402
from simulation.config import (  # noqa: E402
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_BASE_SEED,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_N_JOBS,
    DEFAULT_QUANTREG_MAX_ITER,
    DGPS,
    FAST_FIGURES_DIR,
    FAST_OUTPUT,
    FAST_SUMMARY_OUTPUT,
    FAST_TABLES_DIR,
    FULL_FIGURES_DIR,
    FULL_OUTPUT,
    FULL_SUMMARY_OUTPUT,
    FULL_TABLES_DIR,
    N_VALUES,
    PI_VALUES,
    P_VALUES,
    R_FAST,
    R_FULL,
    TAUS,
)
from simulation.runner import (  # noqa: E402
    SEED_RULE_TEXT,
    filter_completed_designs,
    make_simulation_grid,
    normalize_estimator_names,
    run_simulation_batch,
)


VALID_MODES = ("fast", "full")


def _default_output_for_mode(mode: str) -> Path:
    if mode == "fast":
        return Path(FAST_OUTPUT)
    if mode == "full":
        return Path(FULL_OUTPUT)
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
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_MIN)
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_MAX)
    parser.add_argument("--alpha-grid-size", type=int, default=DEFAULT_ALPHA_GRID_SIZE)
    parser.add_argument(
        "--critical-value-multiplier",
        type=float,
        default=DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    )
    parser.add_argument("--dml-k-folds", type=int, default=DEFAULT_DML_K_FOLDS)
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
    parser.add_argument("--no-reports", action="store_true")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--tables-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--show-quantreg-warnings", action="store_true")
    return parser.parse_args(argv)


def _apply_defaults(args: argparse.Namespace) -> None:
    args.estimators = normalize_estimator_names(args.estimators)
    args.dgps = list(DGPS) if args.dgps is None else args.dgps
    args.n_values = list(N_VALUES) if args.n_values is None else args.n_values
    args.p_values = list(P_VALUES) if args.p_values is None else args.p_values
    args.pi_values = list(PI_VALUES) if args.pi_values is None else args.pi_values
    args.taus = list(TAUS) if args.taus is None else args.taus
    args.reps = (R_FAST if args.mode == "fast" else R_FULL) if args.reps is None else args.reps
    args.output = _default_output_for_mode(args.mode) if args.output is None else Path(args.output)
    args.manifest = None if args.manifest is None else Path(args.manifest)
    args.summary_output = (
        Path(FAST_SUMMARY_OUTPUT if args.mode == "fast" else FULL_SUMMARY_OUTPUT)
        if args.summary_output is None
        else Path(args.summary_output)
    )
    args.tables_dir = (
        Path(FAST_TABLES_DIR if args.mode == "fast" else FULL_TABLES_DIR)
        if args.tables_dir is None
        else Path(args.tables_dir)
    )
    args.figures_dir = (
        Path(FAST_FIGURES_DIR if args.mode == "fast" else FULL_FIGURES_DIR)
        if args.figures_dir is None
        else Path(args.figures_dir)
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.reps < 1:
        raise ValueError("--reps must be at least 1")
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
    if args.dml_k_folds < 2:
        raise ValueError("--dml-k-folds must be at least 2")
    if args.quantreg_max_iter < 1:
        raise ValueError("--quantreg-max-iter must be at least 1")
    if args.critical_value_multiplier <= 0:
        raise ValueError("--critical-value-multiplier must be positive")
    if args.resume and args.manifest is None:
        raise ValueError("--resume requires --manifest")


def _resume_signature(args: argparse.Namespace) -> dict[str, object]:
    return {
        "mode": args.mode,
        "dgps": list(args.dgps),
        "n_values": list(args.n_values),
        "p_values": list(args.p_values),
        "pi_values": list(args.pi_values),
        "taus": list(args.taus),
        "reps": args.reps,
        "base_seed": args.base_seed,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_grid_size": args.alpha_grid_size,
        "critical_value_multiplier": args.critical_value_multiplier,
        "estimators": list(args.estimators),
        "dml_k_folds": args.dml_k_folds,
        "quantreg_max_iter": args.quantreg_max_iter,
    }


def _validate_resume_manifest(args: argparse.Namespace) -> None:
    if args.manifest is None:
        raise ValueError("--resume requires --manifest")
    if not args.manifest.exists():
        raise FileNotFoundError("--resume requires an existing --manifest file")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    previous = payload.get("resume_signature")
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
        "resume_signature": _resume_signature(args),
        "base_seed": args.base_seed,
        "seed_rule": SEED_RULE_TEXT,
        "total_designs": total_designs,
        "pending_designs": pending_designs,
        "designs_in_run": designs_in_run,
        "estimators": list(args.estimators),
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
    print(f"Base seed: {args.base_seed}")
    print("Seed rule: deterministic by design cell, independent of estimator/order")
    print(f"First design seed: {first_design_seed}")
    print(f"DGPs: {', '.join(args.dgps)}")
    print(f"n values: {', '.join(map(str, args.n_values))}")
    print(f"p values: {', '.join(map(str, args.p_values))}")
    print(f"pi values: {', '.join(map(str, args.pi_values))}")
    print(f"taus: {', '.join(map(str, args.taus))}")
    print(f"Estimators: {', '.join(args.estimators)}")
    print(
        "Alpha grid: "
        f"min={args.alpha_min}, max={args.alpha_max}, size={args.alpha_grid_size}, "
        f"step={(args.alpha_max - args.alpha_min) / (args.alpha_grid_size - 1):g}"
    )
    print(f"Expected design rows: {designs_in_run}")
    print(f"Output: {args.output}")
    print(f"Manifest: {args.manifest}")
    print("Reports: skipped by --no-reports" if args.no_reports else "Reports: generated after successful run")


def _make_reports(args: argparse.Namespace) -> None:
    if args.no_reports:
        print("Reports: skipped by --no-reports")
        return
    summary = aggregate_results_file(
        args.output,
        args.summary_output,
        expected_replications=args.reps,
    )
    tables = write_tables(summary, args.tables_dir)
    figures = write_figures(summary, args.figures_dir)
    print(f"Summary: {args.summary_output}")
    for name, path in tables.items():
        print(f"Table ({name}): {path}")
    for name, path in figures.items():
        print(f"Figure ({name}): {path}")


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
    if args.rerun_failed and not args.resume:
        print("--rerun-failed has no effect without --resume.")

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_grid_size)
    designs = make_simulation_grid(
        dgps=tuple(args.dgps),
        n_values=tuple(args.n_values),
        p_values=tuple(args.p_values),
        pi_values=tuple(args.pi_values),
        taus=tuple(args.taus),
        reps=args.reps,
        base_seed=args.base_seed,
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
    designs_to_run = pending_designs[: args.max_designs] if args.max_designs else pending_designs

    if args.dry_run:
        first_design_seed = designs_to_run[0].seed if designs_to_run else None
        _print_dry_run(args, len(designs_to_run), alphas, first_design_seed)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
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
    for batch_start in range(0, len(designs_to_run), args.batch_size):
        batch = designs_to_run[batch_start : batch_start + args.batch_size]
        run_simulation_batch(
            batch,
            alphas,
            estimators=tuple(args.estimators),
            output_path=args.output,
            append=args.resume or completed > 0,
            quantreg_max_iter=args.quantreg_max_iter,
            dml_k_folds=args.dml_k_folds,
            critical_value_multiplier=args.critical_value_multiplier,
            n_jobs=args.n_jobs,
            show_quantreg_warnings=args.show_quantreg_warnings,
        )
        completed += len(batch)
        elapsed = time.perf_counter() - start
        print(f"Completed {completed}/{len(designs_to_run)} designs in {elapsed:.2f} seconds")

    _make_reports(args)
    final_rows = _count_rows(args.output)
    print(f"Mode: {args.mode}")
    print(f"Completed designs: {completed}")
    print(f"Pending before max-designs: {len(pending_designs)}")
    print(f"Output: {args.output}")
    print(f"Final row count: {final_rows}" if final_rows is not None else "Final row count unavailable")


if __name__ == "__main__":
    main()
