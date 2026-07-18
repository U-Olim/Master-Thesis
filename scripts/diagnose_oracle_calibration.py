"""Diagnose finite-sample calibration of the Oracle inverse-IVQR Wald test.

This standalone diagnostic evaluates the excluded-instrument restriction only
at the known structural parameter.  It deliberately does not construct or
invert an alpha grid and does not call or modify any production estimator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from dgp.designs import Design, SimData
from dgp.generators import generate_data
from dgp.true_parameters import get_oracle_control_indices, true_alpha
from ivqr.ch_inverse import (
    ITERATION_WARNING_POLICIES,
    IterationWarningPolicy,
    ch_ivqr_design,
    validate_iteration_warning_policy,
    wald_statistic,
)
from simulation.config import DEFAULT_BASE_SEED, DEFAULT_QUANTREG_MAX_ITER
from simulation.runner import make_simulation_grid


POLICY_OUTPUT_DIRECTORY = Path("results/diagnostics/iteration_warning_policy")
THEORETICAL_CHI2_Q95 = float(chi2.ppf(0.95, df=1))


@dataclass(frozen=True)
class CovarianceVariant:
    """One statsmodels QuantReg covariance configuration."""

    vcov: str
    kernel: str
    bandwidth: str

    @property
    def name(self) -> str:
        return f"{self.vcov}_{self.kernel}_{self.bandwidth}"


COVARIANCE_VARIANTS = (
    CovarianceVariant("robust", "epa", "hsheather"),
    CovarianceVariant("robust", "epa", "bofinger"),
    CovarianceVariant("robust", "epa", "chamberlain"),
    CovarianceVariant("iid", "epa", "hsheather"),
)


def evaluate_true_alpha_wald(
    data: SimData,
    *,
    dgp: str,
    tau: float,
    p: int,
    variant: CovarianceVariant,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    iteration_warning_policy: IterationWarningPolicy = "use_if_valid",
) -> dict[str, object]:
    """Fit the Oracle QR at true alpha and return instrument-Wald diagnostics."""
    alpha = true_alpha(tau, dgp)
    iteration_warning_policy = validate_iteration_warning_policy(
        iteration_warning_policy
    )
    oracle_indices = get_oracle_control_indices(dgp, p)
    x_oracle = np.asarray(data.x, dtype=float)[:, oracle_indices]
    design, z_block = ch_ivqr_design(x_oracle, data.z)
    y_alpha = np.asarray(data.y, dtype=float) - np.asarray(data.d, dtype=float) * alpha

    base: dict[str, object] = {
        "alpha_true": alpha,
        "oracle_controls": int(oracle_indices.size),
        "covariance_variant": variant.name,
        "vcov": variant.vcov,
        "kernel": variant.kernel,
        "bandwidth": variant.bandwidth,
        "converged": False,
        "usable": False,
        "iteration_warning_policy": iteration_warning_policy,
        "warning_type": "",
        "failure_reason": "",
        "iteration_limit_warning": False,
        "finite_statistic_available": False,
        "statistic_used_for_inference": False,
        "failure_message": "",
        "iterations": np.nan,
        "gamma_hat": np.nan,
        "cov_gamma": np.nan,
        "wald": np.nan,
        "rejected": np.nan,
    }

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IterationLimitWarning)
            result = QuantReg(y_alpha, design).fit(
                q=tau,
                vcov=variant.vcov,
                kernel=variant.kernel,
                bandwidth=variant.bandwidth,
                max_iter=max_iter,
            )
        base["iterations"] = int(getattr(result, "iterations", 0))
        iteration_limit_warning = any(
            issubclass(item.category, IterationLimitWarning) for item in caught
        )
        base["iteration_limit_warning"] = iteration_limit_warning
        if iteration_limit_warning:
            base["warning_type"] = "iteration_limit"
        if iteration_limit_warning and iteration_warning_policy == "reject":
            reason = "QuantReg reached iteration limit"
            base["failure_reason"] = reason
            base["failure_message"] = reason
            return base

        params = np.asarray(result.params, dtype=float)
        covariance = np.asarray(result.cov_params(), dtype=float)
        if params.shape != (design.shape[1],):
            raise ValueError("invalid_parameter_dimension")
        if not np.all(np.isfinite(params)):
            raise ValueError("nonfinite_parameters")
        if covariance.shape != (design.shape[1], design.shape[1]):
            raise ValueError("invalid_covariance_dimension")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("nonfinite_covariance")

        gamma_hat = params[z_block]
        cov_gamma = covariance[z_block, z_block]
        if gamma_hat.shape != (1,):
            raise ValueError("invalid_instrument_coefficient_dimension")
        if cov_gamma.shape != (1, 1):
            raise ValueError("invalid_instrument_covariance_dimension")
        variance = float(cov_gamma[0, 0])
        if variance < -1e-10:
            raise ValueError("negative_instrument_variance")
        if variance <= 1e-10:
            raise ValueError("zero_instrument_variance")
        statistic = wald_statistic(gamma_hat, cov_gamma)
    except Exception as exc:  # noqa: BLE001 - statsmodels failures vary by version.
        reason = f"{type(exc).__name__}: {exc}"
        base["failure_reason"] = reason
        base["failure_message"] = reason
        return base

    converged = not bool(base["iteration_limit_warning"])
    base.update(
        {
            "converged": converged,
            "usable": True,
            "failure_reason": "",
            "gamma_hat": float(gamma_hat[0]),
            "cov_gamma": float(cov_gamma[0, 0]),
            "wald": statistic,
            "rejected": bool(statistic > THEORETICAL_CHI2_Q95),
            "finite_statistic_available": True,
            "statistic_used_for_inference": True,
        }
    )
    return base


def summarize_replications(replications: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replication-level Wald diagnostics by design and covariance."""
    replications = replications.copy()
    defaults: dict[str, object] = {
        "iteration_warning_policy": "reject",
        "usable": replications["converged"],
        "iteration_limit_warning": False,
        "finite_statistic_available": np.isfinite(replications["wald"]),
    }
    for column, default in defaults.items():
        if column not in replications:
            replications[column] = default
    group_columns = [
        "dgp",
        "n",
        "p",
        "pi",
        "tau",
        "iteration_warning_policy",
        "covariance_variant",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in replications.groupby(group_columns, sort=False):
        successful = group.loc[group["usable"].astype(bool)]
        warnings_group = group.loc[group["iteration_limit_warning"].astype(bool)]
        wald = successful["wald"].astype(float)
        requested = len(group)
        successes = len(successful)
        rejection_rate = (
            float(successful["rejected"].astype(bool).mean()) if successes else np.nan
        )
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "replications_requested": requested,
                "replications_successful": successes,
                "failures": requested - successes,
                "rejection_rate": rejection_rate,
                "implied_coverage": 1.0 - rejection_rate,
                "mean_wald": float(wald.mean()) if successes else np.nan,
                "median_wald": float(wald.median()) if successes else np.nan,
                "empirical_wald_q90": (
                    float(wald.quantile(0.90)) if successes else np.nan
                ),
                "empirical_wald_q95": (
                    float(wald.quantile(0.95)) if successes else np.nan
                ),
                "empirical_wald_q99": (
                    float(wald.quantile(0.99)) if successes else np.nan
                ),
                "theoretical_chi2_q95": THEORETICAL_CHI2_Q95,
                "total_alpha_evaluations": requested,
                "fully_converged_evaluations": int(group["converged"].sum()),
                "iteration_limit_warning_evaluations": len(warnings_group),
                "warning_evaluations_usable": int(warnings_group["usable"].sum()),
                "hard_failures": int(
                    (
                        (~group["usable"].astype(bool))
                        & (~group["iteration_limit_warning"].astype(bool))
                    ).sum()
                ),
                "unusable_warning_fits": int(
                    (
                        (~group["usable"].astype(bool))
                        & group["iteration_limit_warning"].astype(bool)
                    ).sum()
                ),
                "percentage_warnings_with_finite_usable_statistics": (
                    100.0
                    * float(
                        (
                            warnings_group["usable"].astype(bool)
                            & warnings_group["finite_statistic_available"].astype(bool)
                        ).sum()
                    )
                    / len(warnings_group)
                    if len(warnings_group)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def run_diagnostic(
    designs: list[Design],
    *,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    iteration_warning_policy: IterationWarningPolicy = "use_if_valid",
) -> pd.DataFrame:
    """Generate each design once and evaluate every covariance variant."""
    rows: list[dict[str, object]] = []
    for index, design in enumerate(designs, start=1):
        data = generate_data(design)
        for variant in COVARIANCE_VARIANTS:
            diagnostics = evaluate_true_alpha_wald(
                data,
                dgp=design.dgp,
                tau=design.tau,
                p=design.p,
                variant=variant,
                max_iter=max_iter,
                iteration_warning_policy=iteration_warning_policy,
            )
            rows.append(
                {
                    "dgp": design.dgp,
                    "n": design.n,
                    "p": design.p,
                    "pi": design.pi,
                    "tau": design.tau,
                    "rep": design.rep,
                    "seed": design.seed,
                    **diagnostics,
                }
            )
        if index % 100 == 0 or index == len(designs):
            print(f"Completed {index:,}/{len(designs):,} datasets", flush=True)
    return pd.DataFrame(rows)


def _safe_output_path(path: Path) -> Path:
    """Reject output paths inside the immutable production raw-results tree."""
    resolved = path.resolve()
    raw_results = (Path.cwd() / "results" / "raw").resolve()
    if resolved == raw_results or raw_results in resolved.parents:
        raise ValueError("Diagnostic output must not be written under results/raw")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgps", nargs="+", default=["dgp1", "dgp2"])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--p", type=int, default=200)
    parser.add_argument("--pi", nargs="+", type=float, default=[0.1, 1.0])
    parser.add_argument("--tau", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_QUANTREG_MAX_ITER)
    parser.add_argument(
        "--iteration-warning-policy",
        choices=ITERATION_WARNING_POLICIES,
        default="use_if_valid",
        help=(
            "QuantReg warning policy: use_if_valid is the production default; "
            "reject reproduces legacy results"
        ),
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--replication-output", type=Path, default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    policy = args.iteration_warning_policy
    summary_output = _safe_output_path(
        args.summary_output
        or POLICY_OUTPUT_DIRECTORY / f"oracle_calibration_{policy}_summary.csv"
    )
    replication_output = _safe_output_path(
        args.replication_output
        or POLICY_OUTPUT_DIRECTORY / f"oracle_calibration_{policy}_replications.csv"
    )
    designs = make_simulation_grid(
        dgps=tuple(args.dgps),
        n_values=(args.n,),
        p_values=(args.p,),
        pi_values=tuple(args.pi),
        taus=tuple(args.tau),
        reps=args.reps,
        base_seed=args.base_seed,
    )

    print(
        f"Oracle calibration diagnostic: {len(designs):,} datasets, "
        f"{len(COVARIANCE_VARIANTS)} covariance variants, base seed "
        f"{args.base_seed}"
    )
    print("Requested statsmodels covariance variants:")
    print(f"Iteration-warning policy: {policy}")
    for variant in COVARIANCE_VARIANTS:
        print(
            f"  {variant.name}: vcov={variant.vcov}, kernel={variant.kernel}, "
            f"bandwidth={variant.bandwidth}"
        )

    replications = run_diagnostic(
        designs,
        max_iter=args.max_iter,
        iteration_warning_policy=policy,
    )
    summary = summarize_replications(replications)
    replication_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    replications.to_csv(replication_output, index=False)
    summary.to_csv(summary_output, index=False)

    display_columns = [
        "dgp",
        "pi",
        "tau",
        "covariance_variant",
        "replications_successful",
        "failures",
        "rejection_rate",
        "implied_coverage",
        "empirical_wald_q95",
    ]
    print("\nCalibration summary:")
    print(summary[display_columns].to_string(index=False, float_format="%.4f"))
    failed = replications.loc[~replications["usable"].astype(bool)]
    if not failed.empty:
        print("\nFailure messages:")
        print(
            failed.groupby(["covariance_variant", "failure_message"])
            .size()
            .rename("count")
            .to_string()
        )
    print(f"\nWrote {summary_output}")
    print(f"Wrote {replication_output}")


if __name__ == "__main__":
    main()
