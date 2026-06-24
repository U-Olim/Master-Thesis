"""Run main IVQR simulation scenarios.

Modes:
- fast: main design with R=10 for diagnostics.
- full: main design with R=500 for thesis results.

Full-control IVQR is deliberately excluded. Run scenarios/full_control_ivqr.py
for the separate naive benchmark.
"""

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
from simulation.batching import filter_completed_designs, run_simulation_batch  # noqa: E402
from simulation.chunking import select_design_chunk, validate_chunk_args  # noqa: E402
from simulation.config import (  # noqa: E402
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_N_JOBS,
    DEFAULT_QUANTREG_MAX_ITER,
    DGPS,
    N_VALUES,
    P_VALUES,
    PI_VALUES,
    R_FAST,
    R_MAIN,
    TAUS,
)
from simulation.runner import VALID_ESTIMATORS, make_simulation_grid  # noqa: E402


DEFAULT_RESULTS_DIR = Path("results/raw")
MAIN_ESTIMATORS = ("oracle", "post_selection", "dml")
VALID_MODES = ("fast", "full")


def _default_output_for_mode(mode: str) -> Path:
    if mode == "fast":
        return DEFAULT_RESULTS_DIR / "fast_mode_results.csv"
    if mode == "full":
        return DEFAULT_RESULTS_DIR / "full_mode_results.csv"
    raise ValueError(f"Unknown mode: {mode}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the main IVQR Monte Carlo simulation in batches."
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="fast",
        help="fast uses R=10; full uses R=500. Full-control IVQR is separate.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--alpha-min", type=float, default=-1.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-grid-size", type=int, default=None)
    parser.add_argument("--dml-k-folds", type=int, default=DEFAULT_DML_K_FOLDS)
    parser.add_argument("--quantreg-max-iter", type=int, default=DEFAULT_QUANTREG_MAX_ITER)
    parser.add_argument("--show-quantreg-warnings", action="store_true")
    parser.add_argument("--estimators", nargs="+", choices=VALID_ESTIMATORS, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--max-designs", type=int, default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dgps", nargs="+", default=None)
    parser.add_argument("--n-values", nargs="+", type=int, default=None)
    parser.add_argument("--p-values", nargs="+", type=int, default=None)
    parser.add_argument("--pi-values", nargs="+", type=float, default=None)
    parser.add_argument("--taus", nargs="+", type=float, default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--tables-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    return parser.parse_args()


def _apply_mode_defaults(args: argparse.Namespace) -> None:
    args.estimators = list(MAIN_ESTIMATORS) if args.estimators is None else args.estimators
    args.dgps = list(DGPS) if args.dgps is None else args.dgps
    args.n_values = list(N_VALUES) if args.n_values is None else args.n_values
    args.p_values = list(P_VALUES) if args.p_values is None else args.p_values
    args.pi_values = list(PI_VALUES) if args.pi_values is None else args.pi_values
    args.taus = list(TAUS) if args.taus is None else args.taus
    args.reps = (R_FAST if args.mode == "fast" else R_MAIN) if args.reps is None else args.reps
    args.alpha_grid_size = (
        DEFAULT_ALPHA_GRID_SIZE
        if args.alpha_grid_size is None
        else args.alpha_grid_size
    )
    args.output = (
        _default_output_for_mode(args.mode)
        if args.output is None
        else Path(args.output)
    )
    if args.summary_output is None:
        args.summary_output = Path("results/summary/main_simulation_summary.csv")
    if args.tables_dir is None:
        args.tables_dir = Path("results/tables/main")
    if args.figures_dir is None:
        args.figures_dir = Path("results/figures/main")


def _validate_args(args: argparse.Namespace) -> None:
    if args.reps < 1:
        raise ValueError("--reps must be at least 1")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_designs is not None and args.max_designs < 1:
        raise ValueError("--max-designs must be at least 1")
    validate_chunk_args(args.chunk_index, args.num_chunks)
    if args.alpha_grid_size < 3:
        raise ValueError("--alpha-grid-size must be at least 3")
    if args.dml_k_folds < 2:
        raise ValueError("--dml-k-folds must be at least 2")
    if args.quantreg_max_iter < 1:
        raise ValueError("--quantreg-max-iter must be at least 1")
    if args.alpha_max <= args.alpha_min:
        raise ValueError("--alpha-max must exceed --alpha-min")
    if set(args.estimators) - set(MAIN_ESTIMATORS):
        raise ValueError("Main runner only allows oracle, post_selection, and dml.")


def _validate_output_path(output_path: Path, *, resume: bool) -> None:
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --resume to continue, pass --output to choose a new file, "
            "or delete the existing file manually."
        )


def _count_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return len(pd.read_csv(path, usecols=["estimator"]))
    except (ValueError, pd.errors.EmptyDataError):
        return None


def _make_reports(args: argparse.Namespace) -> None:
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


def _print_dry_run(
    args: argparse.Namespace,
    *,
    number_of_designs: int,
    alpha_grid_size: int,
) -> None:
    print(f"Mode: {args.mode}")
    print(f"Designs: {number_of_designs}")
    print(f"Replications per design: {args.reps}")
    print(f"Alpha grid size: {alpha_grid_size}")
    print(f"Output: {args.output}")
    print(f"Resume: {str(args.resume).lower()}")
    print("Reports: automatic after successful run")


def _write_manifest(
    manifest_path: str | Path | None,
    args: argparse.Namespace,
    *,
    total_designs: int,
    pending_designs: int,
    designs_in_run: int,
    estimators: tuple[str, ...],
    alphas: np.ndarray,
) -> None:
    if manifest_path is None:
        return
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "parameters": vars(args),
        "total_designs": total_designs,
        "pending_designs": pending_designs,
        "designs_in_run": designs_in_run,
        "estimators": list(estimators),
        "alpha_grid": {
            "size": int(alphas.size),
            "min": float(alphas.min()),
            "max": float(alphas.max()),
            "values": [float(value) for value in alphas],
        },
        "output_path": str(args.output),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _apply_mode_defaults(args)
    _validate_args(args)
    if args.rerun_failed and not args.resume:
        print("--rerun-failed has no effect without --resume.")

    estimators = tuple(args.estimators)
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
    designs_to_run = select_design_chunk(designs, args.chunk_index, args.num_chunks)
    if args.max_designs is not None:
        designs_to_run = designs_to_run[: args.max_designs]

    if args.dry_run:
        scenario_count = len(
            {
                (design.dgp, design.n, design.p, design.pi, design.tau)
                for design in designs_to_run
            }
        )
        _print_dry_run(
            args,
            number_of_designs=scenario_count,
            alpha_grid_size=alphas.size,
        )
        return

    output_path = Path(args.output)
    _validate_output_path(output_path, resume=args.resume)
    pending_designs = (
        filter_completed_designs(
            designs,
            output_path,
            estimators=estimators,
            rerun_failed=args.rerun_failed,
        )
        if args.resume
        else designs
    )
    designs_to_run = select_design_chunk(
        pending_designs, args.chunk_index, args.num_chunks
    )
    if args.max_designs is not None:
        designs_to_run = designs_to_run[: args.max_designs]

    _write_manifest(
        args.manifest,
        args,
        total_designs=len(designs),
        pending_designs=len(pending_designs),
        designs_in_run=len(designs_to_run),
        estimators=estimators,
        alphas=alphas,
    )

    start = time.perf_counter()
    completed = 0
    for batch_start in range(0, len(designs_to_run), args.batch_size):
        batch = designs_to_run[batch_start : batch_start + args.batch_size]
        append = args.resume or completed > 0
        run_simulation_batch(
            batch,
            alphas,
            estimators=estimators,
            output_path=output_path,
            append=append,
            quantreg_max_iter=args.quantreg_max_iter,
            dml_k_folds=args.dml_k_folds,
            n_jobs=args.n_jobs,
            show_quantreg_warnings=args.show_quantreg_warnings,
        )
        completed += len(batch)
        elapsed = time.perf_counter() - start
        print(
            f"Completed {completed}/{len(designs_to_run)} designs "
            f"in {elapsed:.2f} seconds"
        )

    final_rows = _count_rows(output_path)
    _make_reports(args)
    print(f"Mode: {args.mode}")
    print(f"Completed designs: {completed}")
    print(f"Output: {output_path}")
    print(
        f"Final row count: {final_rows}"
        if final_rows is not None
        else "Final row count unavailable"
    )


if __name__ == "__main__":
    main()
