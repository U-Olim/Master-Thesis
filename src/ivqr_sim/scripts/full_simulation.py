"""Run the full thesis Monte Carlo simulation in resumable batches."""

from __future__ import annotations

import argparse
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


def _print_plan(
    output_path: Path,
    total_designs: int,
    pending_designs: int,
    estimators: tuple[str, ...],
    alphas: np.ndarray,
    batch_size: int,
    resume: bool,
    rerun_failed: bool,
) -> None:
    expected_rows = pending_designs * len(estimators)
    print("Full simulation plan")
    print(f"total designs: {total_designs}")
    print(f"pending designs: {pending_designs}")
    print(f"expected new rows: {expected_rows}")
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

    _print_plan(
        output_path=output_path,
        total_designs=total_designs,
        pending_designs=len(pending_designs),
        estimators=estimators,
        alphas=alphas,
        batch_size=args.batch_size,
        resume=args.resume,
        rerun_failed=args.rerun_failed,
    )

    start = time.perf_counter()
    completed = 0
    for batch_start in range(0, len(pending_designs), args.batch_size):
        batch = pending_designs[batch_start : batch_start + args.batch_size]
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
            f"completed {completed}/{len(pending_designs)} designs, "
            f"elapsed {elapsed:.2f} seconds"
        )

    final_rows = _count_rows(output_path)
    if final_rows is not None:
        print(f"final row count: {final_rows}")
    else:
        print("final row count unavailable")
