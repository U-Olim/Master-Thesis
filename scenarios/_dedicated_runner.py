"""Shared CLI adapter for estimator-locked simulation entry points."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from scenarios import run_simulation
from simulation.config import (
    DEFAULT_ADAPTIVE_MIDPOINT_PROBE,
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_ALPHA_HAT_GRID,
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_BASE_SEED,
    DEFAULT_BATCH_SIZE,
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
    DEFAULT_N_JOBS,
    DEFAULT_QUANTREG_MAX_ITER,
    DEFAULT_REFINEMENT_TOLERANCE,
)


EstimatorName = str


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-end", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_MIN)
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_MAX)
    parser.add_argument(
        "--alpha-grid-size", type=int, default=DEFAULT_ALPHA_GRID_SIZE
    )
    parser.add_argument("--dgps", nargs="+", default=None)
    parser.add_argument("--n-values", nargs="+", type=int, default=None)
    parser.add_argument("--p-values", nargs="+", type=int, default=None)
    parser.add_argument("--pi-values", nargs="+", type=float, default=None)
    parser.add_argument("--taus", nargs="+", type=float, default=None)
    parser.add_argument("--max-designs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _add_critical_value_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--critical-value-multiplier",
        type=float,
        default=DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    )


def _add_ch_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--grid-strategy",
        choices=("fixed", "adaptive"),
        default=DEFAULT_GRID_STRATEGY,
    )
    parser.add_argument(
        "--refinement-tolerance",
        type=float,
        default=DEFAULT_REFINEMENT_TOLERANCE,
    )
    parser.add_argument(
        "--max-refinement-depth",
        type=int,
        default=DEFAULT_MAX_REFINEMENT_DEPTH,
    )
    parser.add_argument(
        "--max-alpha-evaluations",
        type=int,
        default=DEFAULT_MAX_ALPHA_EVALUATIONS,
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
    _add_critical_value_argument(parser)
    parser.add_argument(
        "--quantreg-max-iter", type=int, default=DEFAULT_QUANTREG_MAX_ITER
    )
    parser.add_argument("--show-quantreg-warnings", action="store_true")


def build_parser(estimator: EstimatorName, *, prog: str) -> argparse.ArgumentParser:
    """Build an estimator-specific parser without generic mode selection."""
    labels = {
        "oracle": "Oracle IVQR",
        "post_selection": "Post-selection IVQR",
        "dml": "DML-IVQR",
    }
    if estimator not in labels:
        raise ValueError(f"Unknown dedicated estimator: {estimator}")
    parser = argparse.ArgumentParser(
        prog=prog,
        description=f"Run the {labels[estimator]} Monte Carlo simulation.",
    )
    _add_common_arguments(parser)
    if estimator in {"oracle", "post_selection"}:
        _add_ch_arguments(parser)
    if estimator == "post_selection":
        parser.add_argument(
            "--selection-lasso-multiplier",
            type=float,
            default=1.0,
        )
    if estimator == "dml":
        _add_critical_value_argument(parser)
        parser.add_argument(
            "--dml-k-folds", type=int, default=DEFAULT_DML_K_FOLDS
        )
        parser.add_argument(
            "--dml-quantile-penalty",
            type=float,
            default=DEFAULT_DML_QUANTILE_PENALTY,
        )
        parser.add_argument(
            "--dml-ridge-alpha",
            type=float,
            default=DEFAULT_DML_RIDGE_ALPHA,
        )
        parser.add_argument(
            "--dml-quantile-solver",
            choices=(
                "highs-ds",
                "highs-ipm",
                "highs",
                "interior-point",
                "revised simplex",
            ),
            default=DEFAULT_DML_QUANTILE_SOLVER,
        )
    return parser


def run_dedicated(
    estimator: EstimatorName,
    *,
    prog: str,
    argv: Sequence[str] | None = None,
) -> None:
    """Validate the dedicated CLI and delegate to generic single-estimator mode."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    build_parser(estimator, prog=prog).parse_args(arguments)
    run_simulation.main(
        ["--mode", "fast", "--estimators", estimator, *arguments]
    )


__all__ = ["build_parser", "run_dedicated"]
