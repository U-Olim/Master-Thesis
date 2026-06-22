"""Run the separate naive Full-Control IVQR benchmark.

This script is intentionally independent from the main simulation script. It uses a
limited benchmark design because full-control IVQR is slow and not appropriate as a
main high-dimensional estimator.
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

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dgp.designs import Design  # noqa: E402
from dgp.generators import generate_data  # noqa: E402
from dgp.true_parameters import true_alpha  # noqa: E402
from estimators.base import EstimationResult  # noqa: E402
from estimators.full_control_ivqr import estimate_full_control_ivqr  # noqa: E402
from reporting.figures import write_figures  # noqa: E402
from reporting.summaries import aggregate_results_file  # noqa: E402
from reporting.tables import write_tables  # noqa: E402
from simulation.chunking import select_design_chunk, validate_chunk_args  # noqa: E402
from simulation.config import (  # noqa: E402
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


ESTIMATOR_NAME = "full_control_ivqr"
MAX_ERROR_MESSAGE_LENGTH = 500


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run separate Full-Control IVQR benchmark.")
    parser.add_argument("--output", default=FULL_CONTROL_BENCHMARK_OUTPUT)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--base-seed", type=int, default=54321)
    parser.add_argument("--alpha-min", type=float, default=-1.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--alpha-grid-size", type=int, default=None)
    parser.add_argument("--quantreg-max-iter", type=int, default=DEFAULT_QUANTREG_MAX_ITER)
    parser.add_argument("--show-quantreg-warnings", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunk-index", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--max-designs", type=int, default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dgps", nargs="+", default=list(FULL_CONTROL_BENCHMARK_DGPS))
    parser.add_argument("--n-values", nargs="+", type=int, default=list(FULL_CONTROL_BENCHMARK_N_VALUES))
    parser.add_argument("--p-values", nargs="+", type=int, default=list(FULL_CONTROL_BENCHMARK_P_VALUES))
    parser.add_argument("--pi-values", nargs="+", type=float, default=list(FULL_CONTROL_BENCHMARK_PI_VALUES))
    parser.add_argument("--taus", nargs="+", type=float, default=list(FULL_CONTROL_BENCHMARK_TAUS))
    parser.add_argument("--summary-output", default="results/summary/full_control_ivqr_summary.csv")
    parser.add_argument("--tables-dir", default="results/tables/full_control")
    parser.add_argument("--figures-dir", default="results/figures/full_control")
    return parser.parse_args()


def _apply_defaults(args: argparse.Namespace) -> None:
    if args.reps is None:
        args.reps = R_FULL_CONTROL_BENCHMARK
    if args.alpha_grid_size is None:
        args.alpha_grid_size = FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE


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
        "error_message": result.message[:MAX_ERROR_MESSAGE_LENGTH] if result.failed else None,
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


def _failure_row(design: Design, alphas: np.ndarray, exc: Exception) -> dict[str, object]:
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
            _run_one(design, alphas, quantreg_max_iter, show_quantreg_warnings)
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
                except Exception as exc:
                    rows.append(_failure_row(futures[future], alphas, exc))
        rows.sort(key=lambda r: (r["dgp"], r["n"], r["p"], r["pi"], r["tau"], r["rep"], r["seed"]))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _write_manifest(path: str | Path | None, args: argparse.Namespace, designs: list[Design], alphas: np.ndarray) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "parameters": vars(args),
        "designs_in_run": len(designs),
        "estimator": ESTIMATOR_NAME,
        "alpha_grid": [float(v) for v in alphas],
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_reports(args: argparse.Namespace) -> None:
    summary = aggregate_results_file(args.output, args.summary_output, expected_replications=args.reps)
    tables = write_tables(summary, Path(args.tables_dir))
    figures = write_figures(summary, Path(args.figures_dir))
    print(f"summary: {args.summary_output}")
    print("tables:")
    for name, path in tables.items():
        print(f"  {name}: {path}")
    print("figures:")
    for name, path in figures.items():
        print(f"  {name}: {path}")


def _design_key(design: Design) -> tuple[object, ...]:
    return (
        design.dgp,
        design.n,
        design.p,
        design.pi,
        design.tau,
        design.rep,
        design.seed,
    )


def _row_design_key(row: pd.Series) -> tuple[object, ...]:
    return (
        row["dgp"],
        int(row["n"]),
        int(row["p"]),
        float(row["pi"]),
        float(row["tau"]),
        int(row["rep"]),
        int(row["seed"]),
    )


def _filter_completed_designs(designs: list[Design], results_path: Path) -> list[Design]:
    if not results_path.exists():
        return designs

    required_columns = DESIGN_KEY_COLUMNS + ["estimator"]
    try:
        existing = pd.read_csv(results_path, usecols=required_columns)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("results CSV is empty or malformed") from exc
    except ValueError as exc:
        raise ValueError("results CSV is missing required resume columns") from exc

    completed = {
        _row_design_key(row)
        for _, row in existing.iterrows()
        if str(row["estimator"]) == ESTIMATOR_NAME
    }
    return [design for design in designs if _design_key(design) not in completed]


def _validate_args(args: argparse.Namespace) -> None:
    if args.reps < 1:
        raise ValueError("--reps must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1")
    if args.alpha_grid_size < 3:
        raise ValueError("--alpha-grid-size must be at least 3")
    if args.alpha_max <= args.alpha_min:
        raise ValueError("--alpha-max must exceed --alpha-min")
    validate_chunk_args(args.chunk_index, args.num_chunks)
    if args.max_designs is not None and args.max_designs < 1:
        raise ValueError("--max-designs must be at least 1")


def main() -> None:
    args = _parse_args()
    _apply_defaults(args)
    _validate_args(args)
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_grid_size)
    output_path = Path(args.output)

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
    pending_designs = _filter_completed_designs(designs, output_path) if args.resume else designs
    designs = select_design_chunk(pending_designs, args.chunk_index, args.num_chunks)
    if args.max_designs is not None:
        designs = designs[: args.max_designs]

    print("Full-Control IVQR benchmark plan")
    print(f"total designs: {total_designs}")
    print(f"pending designs after resume: {len(pending_designs)}")
    print(f"chunk: {args.chunk_index}/{args.num_chunks}" if args.chunk_index is not None else "chunk: none")
    print(f"designs in this run: {len(designs)}")
    print(f"expected rows: {len(designs)}")
    print(f"dry_run: {args.dry_run}")
    print(f"replications per scenario: {args.reps}")
    print(f"output path: {output_path}")
    print(f"alpha grid: size={len(alphas)}, min={float(alphas.min())}, max={float(alphas.max())}")
    print(f"Parallel workers: {args.n_jobs}")
    print(f"QuantReg max iterations: {args.quantreg_max_iter}")
    print(f"Show QuantReg warnings: {args.show_quantreg_warnings}")
    print(f"resume: {args.resume}")
    print("This is a separate naive benchmark, not part of the main estimator comparison.")
    _write_manifest(args.manifest, args, designs, alphas)
    if args.dry_run:
        print("Dry run requested; no result rows written.")
        return

    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"{output_path} already exists. Use --resume to continue from existing "
            "results or delete the file manually before starting a fresh run."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        batch_df.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)
        completed += len(batch)
        print(f"completed {completed}/{len(designs)} designs, elapsed {time.perf_counter() - start:.2f} seconds")

    print(f"final row count: {len(pd.read_csv(output_path))}")
    _make_reports(args)


if __name__ == "__main__":
    main()
