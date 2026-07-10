"""Diagnose Oracle IVQR Wald-test size at the true structural alpha.

This standalone diagnostic evaluates the existing Oracle CH-IVQR Wald test
directly at ``alpha_true``. It deliberately bypasses the alpha grid and
confidence-region inversion.

Interpretation:
If rejection at the true alpha is around 10%, then the Wald test itself is
oversized. If rejection is near 5% while grid coverage is near 90%, then the
grid/inversion procedure is responsible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dgp.designs import Design  # noqa: E402
from dgp.generators import generate_data  # noqa: E402
from dgp.true_parameters import get_oracle_control_indices  # noqa: E402
from ivqr.ch_inverse import evaluate_alpha_ch_ivqr  # noqa: E402
from ivqr.confidence_regions import critical_value_chi_square  # noqa: E402
from simulation.runner import make_design_seed  # noqa: E402


DEFAULT_BASE_SEED = 12345
DEFAULT_REPS = 100
DEFAULT_DGPS = ("dgp1", "dgp2", "dgp3")
DEFAULT_N_VALUES = (500, 1000)
DEFAULT_P_VALUES = (200, 500)
DEFAULT_PI_VALUES = (1.0, 0.5, 0.25, 0.1)
DEFAULT_TAUS = (0.25, 0.50, 0.75)
DEFAULT_QUANTREG_MAX_ITER = 10000
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "diagnostics" / "oracle_true_alpha_test_R100.csv"
DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT / "results" / "diagnostics" / "oracle_true_alpha_test_R100_summary.csv"
)

DETAIL_COLUMNS = (
    "dgp",
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "seed",
    "alpha_true",
    "test_statistic",
    "critical_value",
    "converged",
    "rejected_if_converged",
    "rejected_failure_as_reject",
    "message",
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
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER
    output: Path = DEFAULT_OUTPUT
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT


def make_designs(config: DiagnosticConfig) -> list[Design]:
    """Return the deterministic diagnostic design grid."""
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


def rejection_flags(
    *,
    statistic: float,
    converged: bool,
    critical_value: float,
) -> tuple[bool, bool]:
    """Return both practical rejection definitions for one alpha evaluation."""
    rejected_if_converged = bool(converged and statistic > critical_value)
    rejected_failure_as_reject = bool((not converged) or statistic > critical_value)
    return rejected_if_converged, rejected_failure_as_reject


def evaluate_design(
    design: Design,
    *,
    critical_value: float,
    quantreg_max_iter: int,
) -> dict[str, object]:
    """Evaluate the existing Oracle IVQR Wald statistic at ``alpha_true``."""
    data = generate_data(design)
    if data.alpha_true is None:
        raise ValueError("generated data must include alpha_true")

    oracle_indices = get_oracle_control_indices(design.dgp, design.p)
    evaluation = evaluate_alpha_ch_ivqr(
        y=data.y,
        d=data.d,
        x_controls=data.x[:, oracle_indices],
        z=data.z,
        alpha=data.alpha_true,
        tau=design.tau,
        max_iter=quantreg_max_iter,
    )
    rejected_if_converged, rejected_failure_as_reject = rejection_flags(
        statistic=evaluation.statistic,
        converged=evaluation.converged,
        critical_value=critical_value,
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
        "test_statistic": evaluation.statistic,
        "critical_value": critical_value,
        "converged": evaluation.converged,
        "rejected_if_converged": rejected_if_converged,
        "rejected_failure_as_reject": rejected_failure_as_reject,
        "message": evaluation.message,
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize null rejection rates by design cell."""
    grouped = results.groupby(["dgp", "n", "p", "pi", "tau"], sort=True)
    rows: list[dict[str, object]] = []
    for key, group in grouped:
        converged = group["converged"].astype(bool)
        converged_stats = pd.to_numeric(
            group.loc[converged, "test_statistic"],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        rejection_among_converged = (
            float(group.loc[converged, "rejected_if_converged"].mean())
            if converged.any()
            else float("nan")
        )
        unconditional_rejection = float(group["rejected_failure_as_reject"].mean())
        rows.append(
            {
                "dgp": key[0],
                "n": key[1],
                "p": key[2],
                "pi": key[3],
                "tau": key[4],
                "replications": int(len(group)),
                "convergence_rate": float(converged.mean()),
                "unconditional_rejection_rate": unconditional_rejection,
                "rejection_rate_among_converged": rejection_among_converged,
                "mean_test_statistic": float(converged_stats.mean()),
                "median_test_statistic": float(converged_stats.median()),
                "implied_coverage": 1.0 - unconditional_rejection,
            }
        )
    return pd.DataFrame(rows)


def run_diagnostic(config: DiagnosticConfig) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Run the true-alpha Oracle IVQR null-size diagnostic and write CSVs."""
    critical_value = critical_value_chi_square(level=0.95, df=1)
    designs = make_designs(config)
    rows = [
        evaluate_design(
            design,
            critical_value=critical_value,
            quantreg_max_iter=config.quantreg_max_iter,
        )
        for design in designs
    ]
    results = pd.DataFrame(rows, columns=DETAIL_COLUMNS)
    summary = summarize_results(results)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.summary_output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output, index=False)
    summary.to_csv(config.summary_output, index=False)
    return results, summary, critical_value


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Oracle IVQR Wald-test rejection at the true alpha."
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--dgps", nargs="+", default=list(DEFAULT_DGPS))
    parser.add_argument("--n-values", nargs="+", type=int, default=list(DEFAULT_N_VALUES))
    parser.add_argument("--p-values", nargs="+", type=int, default=list(DEFAULT_P_VALUES))
    parser.add_argument("--pi-values", nargs="+", type=float, default=list(DEFAULT_PI_VALUES))
    parser.add_argument("--taus", nargs="+", type=float, default=list(DEFAULT_TAUS))
    parser.add_argument("--quantreg-max-iter", type=int, default=DEFAULT_QUANTREG_MAX_ITER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
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
        quantreg_max_iter=args.quantreg_max_iter,
        output=args.output,
        summary_output=args.summary_output,
    )


def _print_overall_summary(
    results: pd.DataFrame,
    *,
    critical_value: float,
    output: Path,
    summary_output: Path,
) -> None:
    converged = results["converged"].astype(bool)
    rejection_among_converged = (
        float(results.loc[converged, "rejected_if_converged"].mean())
        if converged.any()
        else float("nan")
    )
    unconditional_rejection = float(results["rejected_failure_as_reject"].mean())
    print(f"Total rows: {len(results)}")
    print(f"Critical value: {critical_value:.12g}")
    print(f"Convergence rate: {float(converged.mean()):.6f}")
    print(f"Unconditional rejection rate: {unconditional_rejection:.6f}")
    print(f"Rejection rate among converged fits: {rejection_among_converged:.6f}")
    print(f"Implied coverage: {1.0 - unconditional_rejection:.6f}")
    print(f"Detailed output: {output}")
    print(f"Summary output: {summary_output}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _config_from_args(args)
    results, _summary, critical_value = run_diagnostic(config)
    _print_overall_summary(
        results,
        critical_value=critical_value,
        output=config.output,
        summary_output=config.summary_output,
    )


if __name__ == "__main__":
    main()
