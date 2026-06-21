"""Run the full Monte Carlo simulation in resumable batches."""

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

from simulation.batching import filter_completed_designs, run_simulation_batch  # noqa: E402
from simulation.chunking import select_design_chunk, validate_chunk_args  # noqa: E402
from simulation.config import (  # noqa: E402
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_N_JOBS,
    DEFAULT_OUTPUT,
    DEFAULT_QUANTREG_MAX_ITER,
    DGPS,
    FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE,
    FULL_CONTROL_BENCHMARK_DGPS,
    FULL_CONTROL_BENCHMARK_N_VALUES,
    FULL_CONTROL_BENCHMARK_OUTPUT,
    FULL_CONTROL_BENCHMARK_PI_VALUES,
    FULL_CONTROL_BENCHMARK_P_VALUES,
    FULL_CONTROL_BENCHMARK_TAUS,
    N_VALUES,
    P_VALUES,
    PI_VALUES,
    R_FULL_CONTROL_BENCHMARK,
    R_MAIN,
    TAUS,
)
from simulation.runner import VALID_ESTIMATORS, make_simulation_grid  # noqa: E402


PRESET_MAIN = "main"
PRESET_FULL_CONTROL = "full-control-benchmark"
DEFAULT_DGPS = tuple(DGPS)
DEFAULT_N_VALUES = tuple(N_VALUES)
DEFAULT_P_VALUES = tuple(P_VALUES)
DEFAULT_PI_VALUES = tuple(PI_VALUES)
DEFAULT_TAUS = tuple(TAUS)
MAIN_ESTIMATORS = ("oracle", "post_selection", "dml")
FULL_CONTROL_BENCHMARK_ESTIMATORS = ("full",)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full IVQR Monte Carlo simulation in batches."
    )
    parser.add_argument(
        "--preset",
        choices=(PRESET_MAIN, PRESET_FULL_CONTROL),
        default=PRESET_MAIN,
        help=(
            "Simulation preset. 'main' excludes full-control IVQR; "
            "'full-control-benchmark' runs the full-control benchmark grid."
        ),
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=(
            "Number of parallel worker processes for independent simulation "
            "designs. Default is 6. Use --n-jobs 1 for serial execution."
        ),
    )
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--alpha-min", type=float, default=-1.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-grid-size", type=int, default=None)
    parser.add_argument(
        "--dml-k-folds",
        type=int,
        default=DEFAULT_DML_K_FOLDS,
        help=(
            "Number of cross-fitting folds for DML-IVQR. Default is 3 for "
            "faster diagnostics/main simulations; use 5 for robustness checks."
        ),
    )
    parser.add_argument(
        "--quantreg-max-iter",
        type=int,
        default=DEFAULT_QUANTREG_MAX_ITER,
        help=(
            "Maximum iterations for statsmodels QuantReg used by "
            "full-control/oracle/post-selection IVQR. Default is 1000."
        ),
    )
    parser.add_argument(
        "--show-quantreg-warnings",
        action="store_true",
        help="Show statsmodels QuantReg IterationLimitWarning messages.",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=VALID_ESTIMATORS,
        default=None,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--max-designs", type=int, default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--dgps", nargs="+", default=None)
    parser.add_argument(
        "--n-values", nargs="+", type=int, default=None
    )
    parser.add_argument(
        "--p-values", nargs="+", type=int, default=None
    )
    parser.add_argument(
        "--pi-values", nargs="+", type=float, default=None
    )
    parser.add_argument("--taus", nargs="+", type=float, default=None)
    return parser.parse_args()


def _apply_preset_defaults(args: argparse.Namespace) -> None:
    if args.preset == PRESET_FULL_CONTROL:
        preset_estimators = FULL_CONTROL_BENCHMARK_ESTIMATORS
        preset_dgps = tuple(FULL_CONTROL_BENCHMARK_DGPS)
        preset_n_values = tuple(FULL_CONTROL_BENCHMARK_N_VALUES)
        preset_p_values = tuple(FULL_CONTROL_BENCHMARK_P_VALUES)
        preset_pi_values = tuple(FULL_CONTROL_BENCHMARK_PI_VALUES)
        preset_taus = tuple(FULL_CONTROL_BENCHMARK_TAUS)
        preset_reps = R_FULL_CONTROL_BENCHMARK
        preset_alpha_grid_size = FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE
        preset_output = FULL_CONTROL_BENCHMARK_OUTPUT
    else:
        preset_estimators = MAIN_ESTIMATORS
        manual_single_benchmark = tuple(args.estimators or ()) in {
            FULL_CONTROL_BENCHMARK_ESTIMATORS,
            ("oracle",),
        }
        if manual_single_benchmark:
            preset_dgps = ("dgp1",)
            preset_n_values = tuple(FULL_CONTROL_BENCHMARK_N_VALUES)
            preset_p_values = tuple(FULL_CONTROL_BENCHMARK_P_VALUES)
            preset_pi_values = (1.0,)
            preset_taus = (0.5,)
        else:
            preset_dgps = DEFAULT_DGPS
            preset_n_values = DEFAULT_N_VALUES
            preset_p_values = DEFAULT_P_VALUES
            preset_pi_values = DEFAULT_PI_VALUES
            preset_taus = DEFAULT_TAUS
        preset_reps = R_MAIN
        preset_alpha_grid_size = DEFAULT_ALPHA_GRID_SIZE
        preset_output = DEFAULT_OUTPUT

    if args.estimators is None:
        args.estimators = list(preset_estimators)
    if args.dgps is None:
        args.dgps = list(preset_dgps)
    if args.n_values is None:
        args.n_values = list(preset_n_values)
    if args.p_values is None:
        args.p_values = list(preset_p_values)
    if args.pi_values is None:
        args.pi_values = list(preset_pi_values)
    if args.taus is None:
        args.taus = list(preset_taus)
    if args.reps is None:
        args.reps = preset_reps
    if args.alpha_grid_size is None:
        args.alpha_grid_size = preset_alpha_grid_size
    if args.output is None:
        args.output = preset_output


def _apply_quick_test(args: argparse.Namespace) -> None:
    if not args.quick_test:
        return

    args.dgps = ["dgp1"]
    args.n_values = [80]
    args.p_values = [5]
    args.pi_values = [1.0]
    args.taus = [0.5]
    args.reps = 2
    args.alpha_grid_size = 5
    args.estimators = ["post_selection", "dml"]
    args.batch_size = 2


def _print_plan(
    output_path: Path,
    total_designs: int,
    pending_designs: int,
    designs_in_run: int,
    estimators: tuple[str, ...],
    alphas: np.ndarray,
    batch_size: int,
    n_jobs: int,
    resume: bool,
    rerun_failed: bool,
    dry_run: bool,
    chunk_index: int | None,
    num_chunks: int | None,
    preset: str,
    dml_k_folds: int,
    quantreg_max_iter: int,
    show_quantreg_warnings: bool,
) -> None:
    expected_rows = designs_in_run * len(estimators)
    print("Full simulation plan")
    print(f"total designs: {total_designs}")
    print(f"pending designs after resume: {pending_designs}")
    if chunk_index is not None and num_chunks is not None:
        print(f"chunk: {chunk_index}/{num_chunks}")
    else:
        print("chunk: none")
    print(f"designs in this run: {designs_in_run}")
    print(f"expected rows: {expected_rows}")
    print(f"dry_run: {dry_run}")
    print(f"output path: {output_path}")
    print(f"estimators: {','.join(estimators)}")
    print(
        "alpha grid: "
        f"size={alphas.size}, min={float(alphas.min())}, max={float(alphas.max())}"
    )
    print(f"batch size: {batch_size}")
    print(f"Parallel workers: {n_jobs}")
    print(f"DML folds: {dml_k_folds}")
    print(f"QuantReg max iterations: {quantreg_max_iter}")
    print(f"Show QuantReg warnings: {show_quantreg_warnings}")
    print(f"resume: {resume}")
    print(f"rerun_failed: {rerun_failed}")
    print(f"preset: {preset}")
    if "full" in estimators:
        print(
            "Full-control IVQR is running as requested; infeasible cases are "
            "recorded as failed rows."
        )
    else:
        print(
            "Full-control IVQR is not part of the main default run. Use "
            "--preset full-control-benchmark or --estimators full to run it."
        )


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
        "selected_chunk": {
            "chunk_index": args.chunk_index,
            "num_chunks": args.num_chunks,
        },
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


def main() -> None:
    args = _parse_args()
    _apply_preset_defaults(args)
    _apply_quick_test(args)

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
    validate_chunk_args(args.chunk_index, args.num_chunks)
    if args.max_designs is not None and args.max_designs < 1:
        raise ValueError("--max-designs must be at least 1")
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

    _print_plan(
        output_path=output_path,
        total_designs=total_designs,
        pending_designs=len(pending_designs),
        designs_in_run=len(designs_to_run),
        estimators=estimators,
        alphas=alphas,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        resume=args.resume,
        rerun_failed=args.rerun_failed,
        dry_run=args.dry_run,
        chunk_index=args.chunk_index,
        num_chunks=args.num_chunks,
        preset=args.preset,
        dml_k_folds=args.dml_k_folds,
        quantreg_max_iter=args.quantreg_max_iter,
        show_quantreg_warnings=args.show_quantreg_warnings,
    )
    _write_manifest(
        args.manifest,
        args,
        output_path,
        total_designs,
        len(pending_designs),
        len(designs_to_run),
        estimators,
        alphas,
    )
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
        print(
            f"completed {completed}/{len(designs_to_run)} designs, "
            f"elapsed {elapsed:.2f} seconds"
        )

    final_rows = _count_rows(output_path)
    if final_rows is not None:
        print(f"final row count: {final_rows}")
    else:
        print("final row count unavailable")


if __name__ == "__main__":
    main()
