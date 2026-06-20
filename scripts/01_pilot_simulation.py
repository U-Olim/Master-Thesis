"""Run a small pilot simulation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import IterationLimitWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from simulation.config import DEFAULT_ALPHA_GRID_SIZE  # noqa: E402
from simulation.runner import (  # noqa: E402
    DEFAULT_PILOT_ESTIMATORS,
    VALID_ESTIMATORS,
    run_pilot_simulation,
)


MODE_CONFIGS = {
    "quick": {
        "dgp": "dgp1",
        "n": 100,
        "p": 20,
        "pi": 1.0,
        "tau": 0.5,
        "reps": 3,
        "alpha_grid_size": DEFAULT_ALPHA_GRID_SIZE,
        "quantreg_max_iter": 500,
        "selection_cv": 3,
        "dml_k_folds": 3,
        "estimators": DEFAULT_PILOT_ESTIMATORS,
        "output_path": Path("results/raw/pilot_quick_results.csv"),
    },
    "stress": {
        "dgp": "dgp1",
        "n": 250,
        "p": 200,
        "pi": 1.0,
        "tau": 0.5,
        "reps": 2,
        "alpha_grid_size": DEFAULT_ALPHA_GRID_SIZE,
        "quantreg_max_iter": 500,
        "selection_cv": 3,
        "dml_k_folds": 3,
        "estimators": DEFAULT_PILOT_ESTIMATORS,
        "output_path": Path("results/raw/pilot_stress_results.csv"),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small pilot simulation.")
    parser.add_argument(
        "--mode",
        choices=tuple(MODE_CONFIGS),
        default="quick",
        help="Pilot configuration to run.",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=VALID_ESTIMATORS,
        default=None,
        help="Optional estimator subset.",
    )
    return parser.parse_args()


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
                "rmse": float((bias.pow(2).mean()) ** 0.5)
                if not bias.empty
                else float("nan"),
                "mae": bias.abs().mean(),
                "failure_rate": group["failed"].mean(),
                "convergence_rate": group["converged"].mean(),
                "coverage_rate": coverage.mean()
                if not coverage.empty
                else float("nan"),
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


def _print_config(
    mode: str, config: dict[str, object], estimators: tuple[str, ...]
) -> None:
    print(f"Pilot mode: {mode}")
    print(
        "Configuration: "
        f"dgp={config['dgp']}, n={config['n']}, p={config['p']}, "
        f"pi={config['pi']}, tau={config['tau']}, reps={config['reps']}"
    )
    print(
        "Grid and controls: "
        f"alpha_grid_size={config['alpha_grid_size']}, "
        f"estimators={','.join(estimators)}, "
        f"quantreg_max_iter={config['quantreg_max_iter']}"
    )


def main() -> None:
    args = _parse_args()
    config = MODE_CONFIGS[args.mode]
    estimators = (
        tuple(args.estimators) if args.estimators is not None else config["estimators"]
    )

    warnings.filterwarnings("ignore", category=IterationLimitWarning)
    _print_config(args.mode, config, estimators)

    results = run_pilot_simulation(
        dgp=config["dgp"],
        n=config["n"],
        p=config["p"],
        pi=config["pi"],
        tau=config["tau"],
        reps=config["reps"],
        estimators=estimators,
        alpha_grid_size=config["alpha_grid_size"],
        quantreg_max_iter=config["quantreg_max_iter"],
        selection_cv=config["selection_cv"],
        dml_k_folds=config["dml_k_folds"],
    )

    output_path = config["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"Saved raw pilot results to {output_path}")
    summary = _summarize(results)
    print(summary.to_string(index=False))
    _print_warnings(summary)


if __name__ == "__main__":
    main()
