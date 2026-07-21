"""Run a small heterogeneous production CH preflight and summarize diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ivqr.confidence_regions import parse_cr_components, validate_cr_geometry  # noqa: E402
from simulation.oracle_output import clean_oracle_results_frame  # noqa: E402
from simulation.post_selection_output import (  # noqa: E402
    clean_post_selection_results_frame,
)
from simulation.results import RESULT_COLUMNS  # noqa: E402
from simulation.runner import make_simulation_grid, run_simulation_design  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, choices=range(1, 11), default=5)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/diagnostics/final_ch_preflight"),
    )
    return parser.parse_args()


def _validate_components(frame: pd.DataFrame) -> None:
    for row in frame.itertuples(index=False):
        components = parse_cr_components(row.cr_components)
        if components is None:
            raise ValueError("CH preflight row is missing cr_components")
        validate_cr_geometry(
            components,
            lower=None if pd.isna(row.cr_lower) else float(row.cr_lower),
            upper=None if pd.isna(row.cr_upper) else float(row.cr_upper),
            length=None if pd.isna(row.cr_length) else float(row.cr_length),
            n_blocks=int(row.cr_n_blocks),
            disconnected=bool(row.cr_disconnected),
        )


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for estimator, group in frame.groupby("estimator", sort=True):
        records.append(
            {
                "estimator": estimator,
                "rows": len(group),
                "estimator_failures": int(group["failed"].sum()),
                "fully_unresolved_crs": int(group["cr_status"].eq("fully_unresolved").sum()),
                "partially_unresolved_crs": int(group["cr_status"].eq("partially_unresolved").sum()),
                "unresolved_coverage": int(group["coverage_status"].eq("coverage_unresolved").sum()),
                "iteration_warning_evaluations": int(group["iteration_warning_evaluations"].fillna(0).sum()),
                "unusable_alpha_evaluations": int(group["unresolved_alpha_evaluations"].fillna(0).sum()),
                "rank_deficient_covariance_failures": int(group["rank_deficient_covariance_failures"].fillna(0).sum()),
                "mean_initial_alpha_evaluations": float(group["initial_alpha_grid_size"].mean()),
                "mean_final_alpha_evaluations": float(group["final_alpha_evaluations"].mean()),
                "midpoint_evaluations": int(group["midpoint_evaluations_added"].fillna(0).sum()),
                "maximum_refinement_depth": int(group["refinement_depth_reached"].fillna(0).max()),
                "refinement_limit_hits": int(group["refinement_limit_hit"].fillna(False).sum()),
                "midpoint_probe_limit_hits": int(group["midpoint_probe_limit_hit"].fillna(False).sum()),
                "disconnected_cr_rate": float(group["cr_disconnected"].mean()),
                "boundary_touching_rate": float(group["cr_hits_any_boundary"].mean()),
                "full_grid_rate": float(group["cr_status"].eq("full_grid_valid").mean()),
                "component_json_valid": True,
                "total_runtime_seconds": float(group["runtime_seconds"].sum()),
                "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Preflight output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    designs = make_simulation_grid(
        dgps=("dgp1", "dgp2"),
        n_values=(args.n,),
        p_values=(args.p,),
        pi_values=(1.0, 0.1),
        taus=(0.25, 0.5, 0.75),
        reps=args.reps,
        base_seed=12345,
    )
    alphas = np.linspace(-1.0, 3.0, 21)
    internal_frames: list[pd.DataFrame] = []
    output_paths: dict[str, str] = {}
    for estimator, cleaner in (
        ("oracle", clean_oracle_results_frame),
        ("post_selection", clean_post_selection_results_frame),
    ):
        rows = [
            row
            for design in designs
            for row in run_simulation_design(
                design,
                alphas,
                estimators=(estimator,),
                iteration_warning_policy="use_if_valid",
                hard_failure_policy="unresolved",
                grid_strategy="adaptive",
                adaptive_midpoint_probe=True,
                refinement_tolerance=0.025,
                max_refinement_depth=10,
                max_alpha_evaluations=201,
                alpha_hat_grid="initial",
            )
        ]
        internal = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        _validate_components(internal)
        internal_frames.append(internal)
        output_path = args.output_dir / f"{estimator}_rows.csv"
        cleaner(internal).to_csv(output_path, index=False)
        output_paths[estimator] = output_path.name
    summary = _summarize(pd.concat(internal_frames, ignore_index=True))
    summary.to_csv(args.output_dir / "preflight_summary.csv", index=False)
    (args.output_dir / "preflight_config.json").write_text(
        json.dumps(
            {
                "dgps": ["dgp1", "dgp2"],
                "n": args.n,
                "p": args.p,
                "pi": [1.0, 0.1],
                "tau": [0.25, 0.5, 0.75],
                "reps": args.reps,
                "estimators": ["oracle", "post_selection"],
                "estimator_outputs": output_paths,
                "n_jobs": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
