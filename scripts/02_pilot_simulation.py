"""Run a small Phase 5A pilot simulation."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from ivqr_sim.simulation.runner import run_pilot_simulation


def _summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for estimator, group in results.groupby("estimator", dropna=False):
        bias = group["bias"].dropna()
        coverage = group["cr_covers_true"].dropna()
        disconnected = group["cr_disconnected"].fillna(False)
        rows.append(
            {
                "estimator": estimator,
                "mean_alpha_hat": group["alpha_hat"].mean(),
                "median_alpha_hat": group["alpha_hat"].median(),
                "mean_bias": bias.mean(),
                "median_bias": bias.median(),
                "rmse": float((bias.pow(2).mean()) ** 0.5) if not bias.empty else float("nan"),
                "mae": bias.abs().mean(),
                "failure_rate": group["failed"].mean(),
                "convergence_rate": group["converged"].mean(),
                "coverage_rate": coverage.mean() if not coverage.empty else float("nan"),
                "cr_empty_rate": group["cr_empty"].mean(),
                "cr_disconnected_rate": disconnected.mean(),
                "mean_cr_length": group["cr_length"].mean(),
                "mean_runtime_seconds": group["runtime_seconds"].mean(),
                "mean_failed_alpha_count": group["failed_alpha_count"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _print_warnings(summary: pd.DataFrame) -> None:
    for row in summary.to_dict("records"):
        estimator = row["estimator"]
        if row["failure_rate"] > 0.2:
            print(
                f"WARNING: {estimator} has failure rate {row['failure_rate']:.2f}. "
                "Check estimator stability."
            )
        if row["cr_empty_rate"] > 0.5:
            print(
                f"WARNING: {estimator} has CR empty rate {row['cr_empty_rate']:.2f}. "
                "Check grid, statistic, or estimator stability."
            )
        if pd.notna(row["coverage_rate"]) and row["coverage_rate"] < 0.5:
            print(
                f"WARNING: {estimator} has coverage rate {row['coverage_rate']:.2f}. "
                "Check confidence-region behavior."
            )
        if row["mean_runtime_seconds"] > 10:
            print(
                f"WARNING: {estimator} has mean runtime {row['mean_runtime_seconds']:.2f}s. "
                "Check pilot feasibility before full simulation."
            )
        if row["mean_failed_alpha_count"] > 0:
            print(
                f"WARNING: {estimator} has mean failed alpha count "
                f"{row['mean_failed_alpha_count']:.2f}. Check grid evaluation stability."
            )


def main() -> None:
    warnings.filterwarnings("ignore", category=IterationLimitWarning)
    # Pilot grid. Final Monte Carlo may use a denser grid depending on runtime.
    alphas = np.linspace(-1.0, 3.0, 17)
    print(f"Pilot alpha grid: size={alphas.size}, min={alphas.min()}, max={alphas.max()}")

    results = run_pilot_simulation(
        dgp="dgp1",
        n=250,
        p=200,
        pi=1.0,
        tau=0.5,
        reps=10,
        alphas=alphas,
    )

    output_path = Path("results/raw/pilot_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Saved raw pilot results to {output_path}")
    summary = _summarize(results)
    print(summary.to_string(index=False))
    _print_warnings(summary)


if __name__ == "__main__":
    main()
