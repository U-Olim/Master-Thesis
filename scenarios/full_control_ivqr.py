"""Run full-control IVQR benchmark scenarios.

This separate, deliberately limited benchmark evaluates the naive full-control
estimator. It is not part of the main high-dimensional estimator comparison.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = PROJECT_ROOT / "scenarios"
SRC_PATH = PROJECT_ROOT / "src"
if str(SCENARIOS_PATH) not in sys.path:
    sys.path.insert(0, str(SCENARIOS_PATH))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dgp.designs import Design  # noqa: E402
from dgp.generators import generate_data  # noqa: E402
from dgp.true_parameters import true_alpha  # noqa: E402
from estimators.base import EstimationResult  # noqa: E402
from estimators.full_control_ivqr import estimate_full_control_ivqr  # noqa: E402
from _common import (  # noqa: E402
    make_reports as _make_reports,
    print_dry_run_common,
    validate_output_path as _validate_output_path,
    validate_resume_manifest,
)
from simulation.chunking import select_design_chunk, validate_chunk_args  # noqa: E402
from simulation._validation import (  # noqa: E402
    design_key,
    parse_explicit_bool,
    row_design_key,
)
from simulation.config import (  # noqa: E402
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_N_JOBS,
    DEFAULT_QUANTREG_MAX_ITER,
    FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE,
    FULL_CONTROL_BENCHMARK_DGPS,
    FULL_CONTROL_BENCHMARK_N_VALUES,
    FULL_CONTROL_BENCHMARK_OUTPUT,
    FULL_CONTROL_BENCHMARK_PI_VALUES,
    FULL_CONTROL_BENCHMARK_P_VALUES,
    FULL_CONTROL_BENCHMARK_TAUS,
    R_FULL_CONTROL_BENCHMARK,
)
from simulation.runner import (  # noqa: E402
    DESIGN_KEY_COLUMNS,
    RESULT_COLUMNS,
    make_simulation_grid,
    quantreg_iteration_warning_filter,
)

try:
    from statsmodels.tools.sm_exceptions import IterationLimitWarning
except ImportError:  # pragma: no cover - statsmodels is a project dependency.
    IterationLimitWarning = Warning


DEFAULT_FULL_CONTROL_OUTPUT = Path(FULL_CONTROL_BENCHMARK_OUTPUT)
ESTIMATOR_NAME = "full_control_ivqr"
MAX_ERROR_MESSAGE_LENGTH = 500


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the separate full-control IVQR benchmark."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FULL_CONTROL_OUTPUT,
        help="Path to save raw full-control benchmark results.",
    )
    parser.add_argument("--reps", type=int, default=R_FULL_CONTROL_BENCHMARK)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--base-seed", type=int, default=54321)
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_MIN)
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_MAX)
    parser.add_argument(
        "--alpha-grid-size",
        type=int,
        default=FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE,
    )
    parser.add_argument(
        "--quantreg-max-iter",
        type=int,
        default=DEFAULT_QUANTREG_MAX_ITER,
    )
    parser.add_argument(
        "--show-quantreg-warnings",
        action="store_true",
        help="Show quantile-regression iteration warnings instead of suppressing them.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="When resuming, rerun designs whose previous rows ended in failure.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--max-designs", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dgps", nargs="+", default=list(FULL_CONTROL_BENCHMARK_DGPS))
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=list(FULL_CONTROL_BENCHMARK_N_VALUES),
    )
    parser.add_argument(
        "--p-values",
        nargs="+",
        type=int,
        default=list(FULL_CONTROL_BENCHMARK_P_VALUES),
    )
    parser.add_argument(
        "--pi-values",
        nargs="+",
        type=float,
        default=list(FULL_CONTROL_BENCHMARK_PI_VALUES),
    )
    parser.add_argument(
        "--taus",
        nargs="+",
        type=float,
        default=list(FULL_CONTROL_BENCHMARK_TAUS),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("results/summary/full_control_ivqr_summary.csv"),
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables/full_control"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("results/figures/full_control"),
    )
    return parser.parse_args()


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
    if args.quantreg_max_iter < 1:
        raise ValueError("--quantreg-max-iter must be at least 1")
    if args.alpha_max <= args.alpha_min:
        raise ValueError("--alpha-max must exceed --alpha-min")


def _configure_warnings(show_quantreg_warnings: bool) -> None:
    if show_quantreg_warnings:
        return
    warnings.filterwarnings("ignore", category=IterationLimitWarning)
    warnings.filterwarnings("ignore", message=r"Maximum number of iterations reached.*")


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_csv(path))
    except pd.errors.EmptyDataError:
        return 0


def _resume_signature(args: argparse.Namespace) -> dict[str, object]:
    return {
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
        "quantreg_max_iter": args.quantreg_max_iter,
    }


def _validate_resume_manifest(
    manifest_path: Path | None,
    args: argparse.Namespace,
) -> None:
    validate_resume_manifest(manifest_path, _resume_signature(args))


def _design_key(design: Design) -> tuple[object, ...]:
    return design_key(design)


def _row_design_key(row: pd.Series) -> tuple[object, ...]:
    return row_design_key(row)


def _as_bool(value: object) -> bool:
    return parse_explicit_bool(value)


def _successful_rows(existing: pd.DataFrame) -> pd.Series:
    if "failed" in existing.columns:
        return ~existing["failed"].map(_as_bool)
    if "failure" in existing.columns:
        return ~existing["failure"].map(_as_bool)
    if "status" in existing.columns:
        return existing["status"].astype(str).str.lower().isin({"ok", "success"})
    return pd.Series(True, index=existing.index)


def _completed_design_keys(
    output_path: Path, *, rerun_failed: bool
) -> set[tuple[object, ...]]:
    if not output_path.exists():
        return set()
    try:
        existing = pd.read_csv(output_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("results CSV is empty or malformed") from exc

    required_columns = set(DESIGN_KEY_COLUMNS + ("estimator",))
    missing = sorted(required_columns - set(existing.columns))
    if missing:
        raise ValueError(f"results CSV is missing required resume columns: {missing}")

    existing = existing.loc[existing["estimator"].astype(str) == ESTIMATOR_NAME]
    if rerun_failed:
        existing = existing.loc[_successful_rows(existing)]
    return {_row_design_key(row) for _, row in existing.iterrows()}


def _print_dry_run(
    args: argparse.Namespace,
    *,
    number_of_designs: int,
) -> None:
    print_dry_run_common(
        mode="full-control IVQR benchmark",
        number_of_designs=number_of_designs,
        reps=args.reps,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_grid_size=args.alpha_grid_size,
        output=args.output,
        resume=args.resume,
        extra_lines=(f"Rerun failed: {str(args.rerun_failed).lower()}",),
    )


def _result_to_row(design: Design, result: EstimationResult) -> dict[str, object]:
    bias = None
    absolute_error = None
    squared_error = None
    if result.alpha_hat is not None and result.alpha_true is not None:
        bias = result.alpha_hat - result.alpha_true
        absolute_error = abs(bias)
        squared_error = bias**2
    return {
        "dgp": design.dgp,
        "n": design.n,
        "p": design.p,
        "pi": design.pi,
        "tau": design.tau,
        "rep": design.rep,
        "seed": design.seed,
        "estimator": result.estimator,
        "alpha_hat": result.alpha_hat,
        "alpha_true": result.alpha_true,
        "bias": bias,
        "absolute_error": absolute_error,
        "squared_error": squared_error,
        "status": "failed" if result.failed else "ok",
        "error_type": "EstimatorFailure" if result.failed else None,
        "error_message": (
            result.message[:MAX_ERROR_MESSAGE_LENGTH] if result.failed else None
        ),
        "failed": result.failed,
        "converged": result.converged,
        "cr_lower": result.cr_lower,
        "cr_upper": result.cr_upper,
        "cr_length": result.cr_length,
        "cr_empty": result.cr_empty,
        "cr_disconnected": result.cr_disconnected,
        "cr_covers_true": result.cr_covers_true,
        "selected_controls": result.selected_controls,
        "runtime_seconds": result.runtime_seconds,
        "failed_alpha_count": result.failed_alpha_count,
        "alpha_grid_size": result.alpha_grid_size,
        "message": result.message,
    }


def _failure_row(
    design: Design, alphas: np.ndarray, exc: Exception
) -> dict[str, object]:
    try:
        alpha_true = true_alpha(design.tau, design.dgp)
    except Exception:
        alpha_true = None
    message = f"{type(exc).__name__}: {str(exc)[:MAX_ERROR_MESSAGE_LENGTH]}"
    return {
        "dgp": design.dgp,
        "n": design.n,
        "p": design.p,
        "pi": design.pi,
        "tau": design.tau,
        "rep": design.rep,
        "seed": design.seed,
        "estimator": ESTIMATOR_NAME,
        "alpha_hat": None,
        "alpha_true": alpha_true,
        "bias": None,
        "absolute_error": None,
        "squared_error": None,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:MAX_ERROR_MESSAGE_LENGTH],
        "failed": True,
        "converged": False,
        "cr_lower": None,
        "cr_upper": None,
        "cr_length": None,
        "cr_empty": True,
        "cr_disconnected": None,
        "cr_covers_true": None,
        "selected_controls": None,
        "runtime_seconds": None,
        "failed_alpha_count": None,
        "alpha_grid_size": len(alphas),
        "message": message,
    }


def _run_one(
    design: Design,
    alphas: np.ndarray,
    quantreg_max_iter: int,
    show_quantreg_warnings: bool,
) -> dict[str, object]:
    try:
        data = generate_data(design)
        with quantreg_iteration_warning_filter(show_quantreg_warnings):
            result = estimate_full_control_ivqr(
                data,
                tau=design.tau,
                alphas=alphas,
                max_iter=quantreg_max_iter,
            )
        return _result_to_row(design, result)
    except Exception as exc:  # noqa: BLE001 - record failed replications.
        return _failure_row(design, alphas, exc)


def _run_batch(
    designs: list[Design],
    alphas: np.ndarray,
    quantreg_max_iter: int,
    n_jobs: int,
    show_quantreg_warnings: bool,
) -> pd.DataFrame:
    if n_jobs == 1 or len(designs) <= 1:
        rows = [
            _run_one(
                design,
                alphas,
                quantreg_max_iter,
                show_quantreg_warnings,
            )
            for design in designs
        ]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(designs))) as executor:
            futures = {
                executor.submit(
                    _run_one,
                    design,
                    alphas,
                    quantreg_max_iter,
                    show_quantreg_warnings,
                ): design
                for design in designs
            }
            for future in as_completed(futures):
                try:
                    rows.append(future.result())
                except Exception as exc:  # noqa: BLE001 - record worker failure.
                    rows.append(_failure_row(futures[future], alphas, exc))
        rows.sort(
            key=lambda row: (
                row["dgp"],
                row["n"],
                row["p"],
                row["pi"],
                row["tau"],
                row["rep"],
                row["seed"],
            )
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _write_manifest(
    path: Path | None,
    args: argparse.Namespace,
    *,
    total_designs: int,
    chunk_designs: int,
    pending_designs: int,
    designs_in_run: int,
    alphas: np.ndarray,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "parameters": vars(args),
        "resume_signature": _resume_signature(args),
        "total_designs": total_designs,
        "chunk_designs": chunk_designs,
        "pending_designs": pending_designs,
        "designs_in_run": designs_in_run,
        "estimator": ESTIMATOR_NAME,
        "alpha_grid": [float(value) for value in alphas],
    }
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    _configure_warnings(args.show_quantreg_warnings)

    alphas = np.linspace(
        args.alpha_min,
        args.alpha_max,
        args.alpha_grid_size,
    )
    all_designs = make_simulation_grid(
        dgps=tuple(args.dgps),
        n_values=tuple(args.n_values),
        p_values=tuple(args.p_values),
        pi_values=tuple(args.pi_values),
        taus=tuple(args.taus),
        reps=args.reps,
        base_seed=args.base_seed,
    )
    chunk_designs = select_design_chunk(
        all_designs,
        args.chunk_index,
        args.num_chunks,
    )
    designs = chunk_designs
    if args.max_designs is not None:
        designs = designs[: args.max_designs]

    if args.dry_run:
        scenario_count = len(
            {
                (design.dgp, design.n, design.p, design.pi, design.tau)
                for design in designs
            }
        )
        _print_dry_run(args, number_of_designs=scenario_count)
        return

    output_path = args.output
    _validate_output_path(output_path, resume=args.resume)
    if args.resume:
        _validate_resume_manifest(args.manifest, args)
    if args.rerun_failed and not args.resume:
        print("--rerun-failed has no effect without --resume.")

    if args.resume:
        completed_keys = _completed_design_keys(
            output_path,
            rerun_failed=args.rerun_failed,
        )
        pending_designs = [
            design
            for design in chunk_designs
            if _design_key(design) not in completed_keys
        ]
    else:
        pending_designs = chunk_designs
    designs = pending_designs
    if args.max_designs is not None:
        designs = designs[: args.max_designs]

    _write_manifest(
        args.manifest,
        args,
        total_designs=len(all_designs),
        chunk_designs=len(chunk_designs),
        pending_designs=len(pending_designs),
        designs_in_run=len(designs),
        alphas=alphas,
    )

    start = time.perf_counter()
    completed = 0
    for batch_start in range(0, len(designs), args.batch_size):
        batch = designs[batch_start : batch_start + args.batch_size]
        batch_df = _run_batch(
            batch,
            alphas,
            args.quantreg_max_iter,
            args.n_jobs,
            args.show_quantreg_warnings,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        batch_df.to_csv(
            output_path,
            mode="a",
            header=not output_path.exists(),
            index=False,
        )
        completed += len(batch)
        elapsed = time.perf_counter() - start
        print(f"Completed {completed}/{len(designs)} designs in {elapsed:.2f} seconds")

    _make_reports(args)
    print("Mode: full-control IVQR benchmark")
    print(f"Completed designs: {completed}")
    print(f"Pending before max-designs: {len(pending_designs)}")
    print(f"Output: {output_path}")
    print(f"Final row count: {_count_rows(output_path)}")


if __name__ == "__main__":
    main()
