"""Run the main IVQR Monte Carlo simulation.

Modes:
- fast: same main design with R=10 for diagnostics.
- full: same main design with R=500 for thesis results.

Full-control IVQR is deliberately excluded. Run scripts/04_run_full_control_ivqr.py
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
    DEFAULT_OUTPUT,
    DEFAULT_QUANTREG_MAX_ITER,
    DGPS,
    FAST_OUTPUT,
    N_VALUES,
    P_VALUES,
    PI_VALUES,
    R_FAST,
    R_MAIN,
    TAUS,
)
from simulation.runner import VALID_ESTIMATORS, make_simulation_grid  # noqa: E402


MAIN_ESTIMATORS = ("oracle", "post_selection", "dml")
VALID_MODES = ("fast", "full")


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
    # Backward-compatible alias for older commands/tests.
    parser.add_argument("--preset", choices=("main",), default=None, help=argparse.SUPPRESS)
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
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--dgps", nargs="+", default=None)
    parser.add_argument("--n-values", nargs="+", type=int, default=None)
    parser.add_argument("--p-values", nargs="+", type=int, default=None)
    parser.add_argument("--pi-values", nargs="+", type=float, default=None)
    parser.add_argument("--taus", nargs="+", type=float, default=None)
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Skip automatic aggregation, tables, and figures after the run.",
    )
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--tables-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    return parser.parse_args()


def _apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.estimators is None:
        args.estimators = list(MAIN_ESTIMATORS)
    if args.dgps is None:
        args.dgps = list(DGPS)
    if args.n_values is None:
        args.n_values = list(N_VALUES)
    if args.p_values is None:
        args.p_values = list(P_VALUES)
    if args.pi_values is None:
        args.pi_values = list(PI_VALUES)
    if args.taus is None:
        args.taus = list(TAUS)
    if args.reps is None:
        args.reps = R_FAST if args.mode == "fast" else R_MAIN
    if args.alpha_grid_size is None:
        args.alpha_grid_size = DEFAULT_ALPHA_GRID_SIZE
    if args.output is None:
        args.output = FAST_OUTPUT if args.mode == "fast" else DEFAULT_OUTPUT
    if args.summary_output is None:
        args.summary_output = f"results/summary/{args.mode}_summary.csv"
    if args.tables_dir is None:
        args.tables_dir = "results/tables"
    if args.figures_dir is None:
        args.figures_dir = "results/figures"


def _apply_quick_test(args: argparse.Namespace) -> None:
    if not args.quick_test:
        return
    original_n_jobs = args.n_jobs
    args.dgps = ["dgp1"]
    args.n_values = [80]
    args.p_values = [10]
    args.pi_values = [1.0]
    args.taus = [0.5]
    args.reps = 2
    args.alpha_grid_size = 5
    args.estimators = ["oracle", "post_selection", "dml"]
    args.batch_size = 2
    if original_n_jobs == DEFAULT_N_JOBS:
        args.n_jobs = 1


def _print_plan(
    *,
    args: argparse.Namespace,
    output_path: Path,
    total_designs: int,
    pending_designs: int,
    designs_in_run: int,
    estimators: tuple[str, ...],
    alphas: np.ndarray,
) -> None:
    print("Main IVQR simulation plan")
    print(f"mode: {args.mode}")
    print(f"total designs: {total_designs}")
    print(f"pending designs after resume: {pending_designs}")
    print(f"chunk: {args.chunk_index}/{args.num_chunks}" if args.chunk_index is not None else "chunk: none")
    print(f"designs in this run: {designs_in_run}")
    print(f"expected rows: {designs_in_run * len(estimators)}")
    print(f"dry_run: {args.dry_run}")
    print(f"output path: {output_path}")
    print(f"estimators: {','.join(estimators)}")
    print(f"alpha grid: size={alphas.size}, min={float(alphas.min())}, max={float(alphas.max())}")
    print(f"replications per scenario: {args.reps}")
    print(f"batch size: {args.batch_size}")
    print(f"Parallel workers: {args.n_jobs}")
    print(f"DML folds: {args.dml_k_folds}")
    print(f"QuantReg max iterations: {args.quantreg_max_iter}")
    print(f"Show QuantReg warnings: {args.show_quantreg_warnings}")
    print(f"resume: {args.resume}")
    print(f"rerun_failed: {args.rerun_failed}")
    print("Full-control IVQR is excluded. Use scripts/04_run_full_control_ivqr.py separately.")


def _write_manifest(
    manifest_path: str | Path | None,
    args: argparse.Namespace,
    output_path: Path,
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
        "output_path": str(output_path),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    tables = write_tables(summary, Path(args.tables_dir))
    figures = write_figures(summary, Path(args.figures_dir))
    print(f"summary: {args.summary_output}")
    print("tables:")
    for name, path in tables.items():
        print(f"  {name}: {path}")
    print("figures:")
    for name, path in figures.items():
        print(f"  {name}: {path}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1")
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
    validate_chunk_args(args.chunk_index, args.num_chunks)
    if args.max_designs is not None and args.max_designs < 1:
        raise ValueError("--max-designs must be at least 1")


def main() -> None:
    args = _parse_args()
    _apply_mode_defaults(args)
    _apply_quick_test(args)
    _validate_args(args)
    if args.rerun_failed and not args.resume:
        print("--rerun-failed has no effect without --resume.")

    output_path = Path(args.output)
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
    total_designs = len(designs)
    pending_designs = (
        filter_completed_designs(designs, output_path, estimators=estimators, rerun_failed=args.rerun_failed)
        if args.resume
        else designs
    )
    designs_to_run = select_design_chunk(pending_designs, args.chunk_index, args.num_chunks)
    if args.max_designs is not None:
        designs_to_run = designs_to_run[: args.max_designs]

    _print_plan(
        args=args,
        output_path=output_path,
        total_designs=total_designs,
        pending_designs=len(pending_designs),
        designs_in_run=len(designs_to_run),
        estimators=estimators,
        alphas=alphas,
    )
    _write_manifest(args.manifest, args, output_path, total_designs, len(pending_designs), len(designs_to_run), estimators, alphas)
    if args.dry_run:
        print("Dry run requested; no result rows written.")
        return

    start = time.perf_counter()
    completed = 0
    for batch_start in range(0, len(designs_to_run), args.batch_size):
        batch = designs_to_run[batch_start : batch_start + args.batch_size]
        append = output_path.exists() if args.resume else completed > 0
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
        print(f"completed {completed}/{len(designs_to_run)} designs, elapsed {elapsed:.2f} seconds")

    final_rows = _count_rows(output_path)
    print(f"final row count: {final_rows}" if final_rows is not None else "final row count unavailable")
    if not args.skip_reports:
        _make_reports(args)


if __name__ == "__main__":
    main()
