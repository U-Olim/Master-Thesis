"""Diagnose Oracle IVQR grid-inversion coverage loss.

This diagnostic uses the same simulated dataset for two checks:

1. Direct CH-IVQR Wald acceptance at the exact true structural alpha.
2. Coverage from the existing Oracle grid-inverted confidence region.

The key row-level discrepancy is ``direct_accept_cr_miss``: the exact null test
accepts ``alpha_true``, but the grid-based confidence region excludes it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dgp.designs import Design, SimData  # noqa: E402
from dgp.generators import generate_data  # noqa: E402
from dgp.true_parameters import get_oracle_control_indices  # noqa: E402
from estimators.oracle import estimate_oracle_ivqr  # noqa: E402
from ivqr.ch_inverse import evaluate_alpha_ch_ivqr  # noqa: E402
from simulation.runner import make_design_seed  # noqa: E402


DEFAULT_BASE_SEED = 12345
DEFAULT_REPS = 100
DEFAULT_FINE_REPS = 20
DEFAULT_DGPS = ("dgp1", "dgp2", "dgp3")
DEFAULT_N_VALUES = (500, 1000)
DEFAULT_P_VALUES = (200, 500)
DEFAULT_PI_VALUES = (1.0, 0.5, 0.25, 0.1)
DEFAULT_TAUS = (0.25, 0.50, 0.75)
DEFAULT_ALPHA_MIN = -1.0
DEFAULT_ALPHA_MAX = 3.0
DEFAULT_GRID_SIZE = 21
DEFAULT_FINE_GRID_SIZE = 81
DEFAULT_QUANTREG_MAX_ITER = 10000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_CRITICAL_VALUE_MULTIPLIER = 1.0
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "diagnostics" / "oracle_grid_inversion_R100.csv"
DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT / "results" / "diagnostics" / "oracle_grid_inversion_R100_summary.csv"
)
DEFAULT_FINE_OUTPUT = (
    PROJECT_ROOT / "results" / "diagnostics" / "oracle_grid_inversion_grid81_R20.csv"
)
DEFAULT_FINE_SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "oracle_grid_inversion_grid81_R20_summary.csv"
)

DETAIL_COLUMNS = (
    "grid_label",
    "dgp",
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "seed",
    "alpha_true",
    "direct_test_statistic",
    "critical_value",
    "direct_converged",
    "direct_accepts_true",
    "cr_covers_true",
    "cr_lower",
    "cr_upper",
    "cr_length",
    "cr_empty",
    "cr_disconnected",
    "cr_boundary_hit",
    "failed_alpha_count",
    "failed_alpha_rate",
    "accepted_alpha_blocks",
    "direct_message",
    "oracle_message",
    "direct_accept_cr_cover",
    "direct_accept_cr_miss",
    "direct_reject_cr_miss",
    "direct_reject_cr_cover",
)


@dataclass(frozen=True)
class DiagnosticConfig:
    base_seed: int = DEFAULT_BASE_SEED
    reps: int = DEFAULT_REPS
    dgps: tuple[str, ...] = DEFAULT_DGPS
    n_values: tuple[int, ...] = DEFAULT_N_VALUES
    p_values: tuple[int, ...] = DEFAULT_P_VALUES
    pi_values: tuple[float, ...] = DEFAULT_PI_VALUES
    taus: tuple[float, ...] = DEFAULT_TAUS
    alpha_min: float = DEFAULT_ALPHA_MIN
    alpha_max: float = DEFAULT_ALPHA_MAX
    grid_size: int = DEFAULT_GRID_SIZE
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    critical_value_multiplier: float = DEFAULT_CRITICAL_VALUE_MULTIPLIER
    output: Path = DEFAULT_OUTPUT
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT
    grid_label: str = "grid21"


def alpha_grid(config: DiagnosticConfig) -> np.ndarray:
    """Return the diagnostic alpha grid."""
    if config.grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if config.alpha_max <= config.alpha_min:
        raise ValueError("alpha_max must exceed alpha_min")
    return np.linspace(config.alpha_min, config.alpha_max, config.grid_size)


def make_designs(config: DiagnosticConfig) -> list[Design]:
    """Return deterministic designs using the existing seed rule."""
    if config.reps < 1:
        raise ValueError("reps must be at least 1")
    designs: list[Design] = []
    for dgp in config.dgps:
        for n in config.n_values:
            for p in config.p_values:
                for pi in config.pi_values:
                    for tau in config.taus:
                        for rep in range(config.reps):
                            seed = make_design_seed(
                                base_seed=config.base_seed,
                                dgp=dgp,
                                n=n,
                                p=p,
                                pi=pi,
                                tau=tau,
                                rep=rep,
                            )
                            designs.append(Design(dgp, n, p, pi, tau, rep, seed))
    return designs


def discrepancy_flags(
    *,
    direct_accepts_true: bool,
    cr_covers_true: bool,
) -> dict[str, bool]:
    """Classify agreement and disagreement between direct and grid checks."""
    return {
        "direct_accept_cr_cover": bool(direct_accepts_true and cr_covers_true),
        "direct_accept_cr_miss": bool(direct_accepts_true and not cr_covers_true),
        "direct_reject_cr_miss": bool((not direct_accepts_true) and not cr_covers_true),
        "direct_reject_cr_cover": bool((not direct_accepts_true) and cr_covers_true),
    }


def _bool_or_false(value: object) -> bool:
    return bool(value) if value is not None and not pd.isna(value) else False


def evaluate_design(
    design: Design,
    *,
    alphas: np.ndarray,
    critical_value: float,
    confidence_level: float,
    critical_value_multiplier: float,
    quantreg_max_iter: int,
    data_generator: Callable[[Design], SimData] = generate_data,
    oracle_estimator: Callable[..., object] = estimate_oracle_ivqr,
) -> dict[str, object]:
    """Evaluate direct true-alpha acceptance and grid CR coverage on one dataset."""
    data = data_generator(design)
    if data.alpha_true is None:
        raise ValueError("generated data must include alpha_true")
    oracle_indices = get_oracle_control_indices(design.dgp, design.p)

    direct = evaluate_alpha_ch_ivqr(
        y=data.y,
        d=data.d,
        x_controls=data.x[:, oracle_indices],
        z=data.z,
        alpha=data.alpha_true,
        tau=design.tau,
        max_iter=quantreg_max_iter,
    )
    direct_accepts_true = bool(
        direct.converged and direct.statistic <= critical_value
    )

    grid_result = oracle_estimator(
        data,
        tau=design.tau,
        alphas=alphas,
        oracle_indices=oracle_indices,
        max_iter=quantreg_max_iter,
        confidence_level=confidence_level,
        critical_value_multiplier=critical_value_multiplier,
    )
    cr_covers_true = _bool_or_false(getattr(grid_result, "cr_covers_true", False))
    flags = discrepancy_flags(
        direct_accepts_true=direct_accepts_true,
        cr_covers_true=cr_covers_true,
    )

    return {
        "dgp": design.dgp,
        "n": design.n,
        "p": design.p,
        "pi": design.pi,
        "tau": design.tau,
        "rep": design.rep,
        "seed": design.seed,
        "alpha_true": data.alpha_true,
        "direct_test_statistic": direct.statistic,
        "critical_value": critical_value,
        "direct_converged": direct.converged,
        "direct_accepts_true": direct_accepts_true,
        "cr_covers_true": cr_covers_true,
        "cr_lower": getattr(grid_result, "cr_lower", None),
        "cr_upper": getattr(grid_result, "cr_upper", None),
        "cr_length": getattr(grid_result, "cr_length", None),
        "cr_empty": _bool_or_false(getattr(grid_result, "cr_empty", False)),
        "cr_disconnected": _bool_or_false(
            getattr(grid_result, "cr_disconnected", False)
        ),
        "cr_boundary_hit": _bool_or_false(
            getattr(grid_result, "cr_hits_any_boundary", False)
        ),
        "failed_alpha_count": getattr(grid_result, "failed_alpha_count", None),
        "failed_alpha_rate": getattr(grid_result, "failed_alpha_rate", None),
        # The current Oracle estimator result exposes block counts but not the
        # accepted block intervals themselves. Keep this column explicit without
        # reimplementing confidence-region inversion here.
        "accepted_alpha_blocks": "",
        "direct_message": direct.message,
        "oracle_message": getattr(grid_result, "message", ""),
        **flags,
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize direct/grid discrepancies by design cell."""
    grouped = results.groupby(["dgp", "n", "p", "pi", "tau"], sort=True)
    rows: list[dict[str, object]] = []
    for key, group in grouped:
        rows.append(
            {
                "dgp": key[0],
                "n": key[1],
                "p": key[2],
                "pi": key[3],
                "tau": key[4],
                "replications": int(len(group)),
                "direct_acceptance_rate": float(group["direct_accepts_true"].mean()),
                "grid_cr_coverage_rate": float(group["cr_covers_true"].mean()),
                "direct_accept_grid_miss_rate": float(
                    group["direct_accept_cr_miss"].mean()
                ),
                "direct_reject_grid_cover_rate": float(
                    group["direct_reject_cr_cover"].mean()
                ),
                "empty_region_rate": float(group["cr_empty"].mean()),
                "disconnected_region_rate": float(group["cr_disconnected"].mean()),
                "boundary_hit_rate": float(group["cr_boundary_hit"].mean()),
                "failed_alpha_rate": float(
                    pd.to_numeric(group["failed_alpha_rate"], errors="coerce")
                    .fillna(1.0)
                    .mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_diagnostic(config: DiagnosticConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the grid-inversion diagnostic and write detailed/summary CSVs."""
    critical_value = float(chi2.ppf(config.confidence_level, df=1))
    alphas = alpha_grid(config)
    rows = [
        {
            "grid_label": config.grid_label,
            **evaluate_design(
                design,
                alphas=alphas,
                critical_value=critical_value,
                confidence_level=config.confidence_level,
                critical_value_multiplier=config.critical_value_multiplier,
                quantreg_max_iter=config.quantreg_max_iter,
            ),
        }
        for design in make_designs(config)
    ]
    results = pd.DataFrame(rows, columns=DETAIL_COLUMNS)
    summary = summarize_results(results)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.summary_output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output, index=False)
    summary.to_csv(config.summary_output, index=False)
    return results, summary


def _print_overall_results(results: pd.DataFrame, *, output: Path, summary_output: Path) -> None:
    direct_coverage = float(results["direct_accepts_true"].mean())
    grid_coverage = float(results["cr_covers_true"].mean())
    difference = direct_coverage - grid_coverage
    print(f"Rows: {len(results)}")
    print(f"Direct-test implied coverage: {direct_coverage:.6f}")
    print(f"Grid-based CR coverage: {grid_coverage:.6f}")
    print(f"Percentage-point difference: {100.0 * difference:.3f}")
    print(
        "Direct-accept/grid-miss rate: "
        f"{float(results['direct_accept_cr_miss'].mean()):.6f}"
    )
    print(
        "Direct-reject/grid-cover rate: "
        f"{float(results['direct_reject_cr_cover'].mean()):.6f}"
    )
    print(f"Detailed output: {output}")
    print(f"Summary output: {summary_output}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct true-alpha Oracle IVQR acceptance with grid CR coverage."
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--dgps", nargs="+", default=list(DEFAULT_DGPS))
    parser.add_argument("--n-values", nargs="+", type=int, default=list(DEFAULT_N_VALUES))
    parser.add_argument("--p-values", nargs="+", type=int, default=list(DEFAULT_P_VALUES))
    parser.add_argument("--pi-values", nargs="+", type=float, default=list(DEFAULT_PI_VALUES))
    parser.add_argument("--taus", nargs="+", type=float, default=list(DEFAULT_TAUS))
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_MIN)
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_MAX)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--quantreg-max-iter", type=int, default=DEFAULT_QUANTREG_MAX_ITER)
    parser.add_argument("--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL)
    parser.add_argument(
        "--critical-value-multiplier",
        type=float,
        default=DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--run-fine-grid", action="store_true")
    parser.add_argument("--fine-reps", type=int, default=DEFAULT_FINE_REPS)
    parser.add_argument("--fine-grid-size", type=int, default=DEFAULT_FINE_GRID_SIZE)
    parser.add_argument("--fine-output", type=Path, default=DEFAULT_FINE_OUTPUT)
    parser.add_argument(
        "--fine-summary-output",
        type=Path,
        default=DEFAULT_FINE_SUMMARY_OUTPUT,
    )
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> DiagnosticConfig:
    return DiagnosticConfig(
        base_seed=args.base_seed,
        reps=args.reps,
        dgps=tuple(args.dgps),
        n_values=tuple(args.n_values),
        p_values=tuple(args.p_values),
        pi_values=tuple(args.pi_values),
        taus=tuple(args.taus),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        grid_size=args.grid_size,
        quantreg_max_iter=args.quantreg_max_iter,
        confidence_level=args.confidence_level,
        critical_value_multiplier=args.critical_value_multiplier,
        output=args.output,
        summary_output=args.summary_output,
        grid_label=f"grid{args.grid_size}",
    )


def _fine_config_from_args(args: argparse.Namespace) -> DiagnosticConfig:
    return DiagnosticConfig(
        base_seed=args.base_seed,
        reps=args.fine_reps,
        dgps=tuple(args.dgps),
        n_values=tuple(args.n_values),
        p_values=tuple(args.p_values),
        pi_values=tuple(args.pi_values),
        taus=tuple(args.taus),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        grid_size=args.fine_grid_size,
        quantreg_max_iter=args.quantreg_max_iter,
        confidence_level=args.confidence_level,
        critical_value_multiplier=args.critical_value_multiplier,
        output=args.fine_output,
        summary_output=args.fine_summary_output,
        grid_label=f"grid{args.fine_grid_size}",
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _config_from_args(args)
    results, _summary = run_diagnostic(config)
    _print_overall_results(
        results,
        output=config.output,
        summary_output=config.summary_output,
    )

    if args.run_fine_grid:
        fine_config = _fine_config_from_args(args)
        fine_results, _fine_summary = run_diagnostic(fine_config)
        print("")
        print("Fine-grid diagnostic:")
        _print_overall_results(
            fine_results,
            output=fine_config.output,
            summary_output=fine_config.summary_output,
        )


if __name__ == "__main__":
    main()
