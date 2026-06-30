"""Compatibility wrapper for the historical full-control IVQR command."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = PROJECT_ROOT / "scenarios"
SRC_PATH = PROJECT_ROOT / "src"
if str(SCENARIOS_PATH) not in sys.path:
    sys.path.insert(0, str(SCENARIOS_PATH))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import run_simulation as _unified  # noqa: E402
from simulation.config import (  # noqa: E402
    FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE,
    FULL_CONTROL_BENCHMARK_DGPS,
    FULL_CONTROL_BENCHMARK_N_VALUES,
    FULL_CONTROL_BENCHMARK_OUTPUT,
    FULL_CONTROL_BENCHMARK_PI_VALUES,
    FULL_CONTROL_BENCHMARK_P_VALUES,
    FULL_CONTROL_BENCHMARK_TAUS,
    R_FULL_CONTROL_BENCHMARK,
)
from simulation._validation import parse_explicit_bool, row_design_key  # noqa: E402
from simulation.runner import _result_to_row  # noqa: E402


def _has_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in args)


def _legacy_defaults(argv: list[str]) -> list[str]:
    defaults: list[str] = []
    if not _has_option(argv, "--mode"):
        defaults.extend(["--mode", "fast"])
    if not _has_option(argv, "--estimators"):
        defaults.extend(["--estimators", "full_control"])
    if not _has_option(argv, "--output"):
        defaults.extend(["--output", FULL_CONTROL_BENCHMARK_OUTPUT])
    if not _has_option(argv, "--reps"):
        defaults.extend(["--reps", str(R_FULL_CONTROL_BENCHMARK)])
    if not _has_option(argv, "--base-seed"):
        defaults.extend(["--base-seed", "54321"])
    if not _has_option(argv, "--alpha-grid-size"):
        defaults.extend(
            ["--alpha-grid-size", str(FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE)]
        )
    if not _has_option(argv, "--dgps"):
        defaults.extend(["--dgps", *FULL_CONTROL_BENCHMARK_DGPS])
    if not _has_option(argv, "--n-values"):
        defaults.extend(["--n-values", *map(str, FULL_CONTROL_BENCHMARK_N_VALUES)])
    if not _has_option(argv, "--p-values"):
        defaults.extend(["--p-values", *map(str, FULL_CONTROL_BENCHMARK_P_VALUES)])
    if not _has_option(argv, "--pi-values"):
        defaults.extend(["--pi-values", *map(str, FULL_CONTROL_BENCHMARK_PI_VALUES)])
    if not _has_option(argv, "--taus"):
        defaults.extend(["--taus", *map(str, FULL_CONTROL_BENCHMARK_TAUS)])
    if not _has_option(argv, "--summary-output"):
        defaults.extend(
            ["--summary-output", "results/summary/full_control_ivqr_summary.csv"]
        )
    if not _has_option(argv, "--tables-dir"):
        defaults.extend(["--tables-dir", "results/tables/full_control"])
    if not _has_option(argv, "--figures-dir"):
        defaults.extend(["--figures-dir", "results/figures/full_control"])
    return [*defaults, *argv]


def _ensure_full_control_only(args) -> None:
    if tuple(args.estimators) != ("full_control",):
        raise ValueError(
            "Estimator(s) are not supported for scenario 'full_control'. "
            "Valid choices: full_control."
        )


def _parse_args(argv: list[str] | None = None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = _unified._parse_args(_legacy_defaults(raw_args))
    _unified._apply_mode_defaults(args)
    if args.manifest is not None:
        args.manifest = Path(args.manifest)
    _ensure_full_control_only(args)
    return args


def _validate_args(args) -> None:
    _unified._validate_args(args)


def _resume_signature(args) -> dict[str, object]:
    return _unified._resume_signature(args)


def _validate_resume_manifest(manifest_path, args) -> None:
    _unified._validate_resume_manifest(manifest_path, args)


def _validate_output_path(output_path, *, resume: bool) -> None:
    _unified._validate_output_path(output_path, resume=resume)


def _row_design_key(row):
    return row_design_key(row)


def _as_bool(value: object) -> bool:
    return parse_explicit_bool(value)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_args(args)
    raw_args = list(sys.argv[1:] if argv is None else argv)
    _unified.main(_legacy_defaults(raw_args))


if __name__ == "__main__":
    main()
