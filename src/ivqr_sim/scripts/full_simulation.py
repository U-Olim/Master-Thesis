"""Run the full thesis Monte Carlo simulation in resumable batches."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from ivqr_sim.simulation.runner import (
    VALID_ESTIMATORS,
    filter_completed_designs,
    make_simulation_grid,
    run_simulation_batch,
)


DEFAULT_DGPS = ("dgp1", "dgp2", "dgp3")
DEFAULT_N_VALUES = (250, 500, 1000)
DEFAULT_P_VALUES = (200, 300, 500)
DEFAULT_PI_VALUES = (1.0, 0.5, 0.25, 0.10)
DEFAULT_TAUS = (0.25, 0.50, 0.75)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full IVQR Monte Carlo simulation in batches."
    )
    parser.add_argument("--output", default="results/raw/full_simulation_results.csv")
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=12345)
    parser.add_argument("--alpha-min", type=float, default=-1.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-grid-size", type=int, default=17)
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=VALID_ESTIMATORS,
        default=list(VALID_ESTIMATORS),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--max-designs", type=int, default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--dgps", nargs="+", default=list(DEFAULT_DGPS))
    parser.add_argument("--n-values", nargs="+", type=int, default=list(DEFAULT_N_VALUES))
    parser.add_argument("--p-values", nargs="+", type=int, default=list(DEFAULT_P_VALUES))
    parser.add_argument("--pi-values", nargs="+", type=float, default=list(DEFAULT_PI_VALUES))
    parser.add_argument("--taus", nargs="+", type=float, default=list(DEFAULT_TAUS))
    return parser.parse_args()


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
    args.estimators = list(VALID_ESTIMATORS)
    args.batch_size = 2


def _validate_chunk_args(chunk_index: int | None, num_chunks: int | None) -> None:
    if (chunk_index is None) != (num_chunks is None):
        raise ValueError("--chunk-index and --num-chunks must be provided together")
    if chunk_index is None or num_chunks is None:
        return
    if num_chunks < 1:
        raise ValueError("--num-chunks must be at least 1")
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError("--chunk-index must satisfy 0 <= chunk_index < num_chunks")


def select_design_chunk(designs: list, chunk_index: int | None, num_chunks: int | None) -> list:
    """Select one deterministic strided chunk of designs."""
    _validate_chunk_args(chunk_index, num_chunks)
    if chunk_index is None or num_chunks is None:
        return list(designs)
    return list(designs[chunk_index::num_chunks])


def _print_plan(
    output_path: Path,
    total_designs: int,
    pending_designs: int,
    designs_in_run: int,
    estimators: tuple[str, ...],
    alphas: np.ndarray,
    batch_size: int,
    resume: bool,
    rerun_failed: bool,
    dry_run: bool,
    chunk_index: int | None,
    num_chunks: int | None,
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
    print(f"resume: {resume}")
    print(f"rerun_failed: {rerun_failed}")
    print(
        "Full-control IVQR is included by default; infeasible high-dimensional "
        "cases are recorded as failures."
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
    _apply_quick_test(args)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.alpha_grid_size < 3:
        raise ValueError("--alpha-grid-size must be at least 3")
    if args.alpha_max <= args.alpha_min:
        raise ValueError("--alpha-max must exceed --alpha-min")
    _validate_chunk_args(args.chunk_index, args.num_chunks)
    if args.max_designs is not None and args.max_designs < 1:
        raise ValueError("--max-designs must be at least 1")
    if args.rerun_failed and not args.resume:
        print("--rerun-failed has no effect without --resume.")

    warnings.filterwarnings("ignore", category=IterationLimitWarning)
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
    designs_to_run = select_design_chunk(pending_designs, args.chunk_index, args.num_chunks)
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
        resume=args.resume,
        rerun_failed=args.rerun_failed,
        dry_run=args.dry_run,
        chunk_index=args.chunk_index,
        num_chunks=args.num_chunks,
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
            dml_k_folds=5,
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
