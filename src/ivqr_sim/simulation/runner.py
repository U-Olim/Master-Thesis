"""Pilot Monte Carlo runner utilities."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ivqr_sim.dgp import generate_data
from ivqr_sim.estimators.base import EstimationResult
from ivqr_sim.estimators.dml_ivqr import estimate_dml_ivqr
from ivqr_sim.estimators.full_ivqr import estimate_full_ivqr
from ivqr_sim.estimators.post_selection_ivqr import estimate_post_selection_ivqr
from ivqr_sim.simulation.design import Design
from ivqr_sim.true_effects import true_alpha


EstimatorFn = Callable[..., EstimationResult]
VALID_ESTIMATORS = ("full", "post_selection", "dml")
VALID_DGPS = ("dgp1", "dgp2", "dgp3")
ESTIMATOR_OUTPUT_NAMES = {
    "full": "full_ivqr",
    "post_selection": "post_selection_ivqr",
    "dml": "dml_ivqr",
}
RESULT_COLUMNS = [
    "dgp",
    "n",
    "p",
    "pi",
    "tau",
    "rep",
    "seed",
    "estimator",
    "alpha_hat",
    "alpha_true",
    "bias",
    "failed",
    "converged",
    "cr_lower",
    "cr_upper",
    "cr_length",
    "cr_empty",
    "cr_disconnected",
    "cr_covers_true",
    "selected_controls",
    "runtime_seconds",
    "failed_alpha_count",
    "alpha_grid_size",
    "message",
]
DESIGN_KEY_COLUMNS = ["dgp", "n", "p", "pi", "tau", "rep", "seed"]


def _validate_estimators(estimators: tuple[str, ...]) -> None:
    if len(estimators) == 0:
        raise ValueError("estimators must contain at least one estimator name")

    invalid = sorted(set(estimators) - set(VALID_ESTIMATORS))
    if invalid:
        valid = ", ".join(VALID_ESTIMATORS)
        raise ValueError(f"Unknown estimator(s): {invalid}. Valid estimators: {valid}")


def _validate_dgps(dgps: tuple[str, ...]) -> None:
    invalid_dgps = sorted(set(dgps) - set(VALID_DGPS))
    if invalid_dgps:
        raise ValueError(f"Unknown DGP(s): {invalid_dgps}. Valid DGPs: {VALID_DGPS}")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return False


def _design_key(design: Design) -> tuple[object, ...]:
    return (design.dgp, design.n, design.p, design.pi, design.tau, design.rep, design.seed)


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


def _result_to_row(design: Design, result: EstimationResult) -> dict[str, object]:
    bias = None
    if result.alpha_hat is not None:
        bias = result.alpha_hat - result.alpha_true

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


def _failure_rows_for_design(
    design: Design,
    estimators: tuple[str, ...],
    alphas: np.ndarray,
    exc: Exception,
) -> list[dict[str, object]]:
    try:
        alpha_true = true_alpha(design.tau, design.dgp)
    except Exception:
        alpha_true = None

    message = f"{type(exc).__name__}: {exc}"
    return [
        _base_failure_row(design, estimator, alphas, alpha_true, message)
        for estimator in estimators
    ]


def _base_failure_row(
    design: Design,
    estimator: str,
    alphas: np.ndarray,
    alpha_true: float | None,
    message: str,
) -> dict[str, object]:
    return {
        "dgp": design.dgp,
        "n": design.n,
        "p": design.p,
        "pi": design.pi,
        "tau": design.tau,
        "rep": design.rep,
        "seed": design.seed,
        "estimator": ESTIMATOR_OUTPUT_NAMES[estimator],
        "alpha_hat": None,
        "alpha_true": alpha_true,
        "bias": None,
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


def _failure_row_for_estimator(
    design: Design,
    estimator: str,
    alphas: np.ndarray,
    exc: Exception,
) -> dict[str, object]:
    """Build one failed result row for an unexpected estimator exception."""
    try:
        alpha_true = true_alpha(design.tau, design.dgp)
    except Exception:
        alpha_true = None

    message = f"Unexpected estimator error: {type(exc).__name__}: {exc}"
    return _base_failure_row(design, estimator, alphas, alpha_true, message)


def make_simulation_grid(
    dgps: tuple[str, ...],
    n_values: tuple[int, ...],
    p_values: tuple[int, ...],
    pi_values: tuple[float, ...],
    taus: tuple[float, ...],
    reps: int,
    base_seed: int = 12345,
) -> list[Design]:
    """Create the deterministic full Monte Carlo design grid."""
    inputs = {
        "dgps": dgps,
        "n_values": n_values,
        "p_values": p_values,
        "pi_values": pi_values,
        "taus": taus,
    }
    empty = [name for name, values in inputs.items() if len(values) == 0]
    if empty:
        raise ValueError(f"Simulation grid inputs cannot be empty: {empty}")
    _validate_dgps(dgps)
    if reps < 1:
        raise ValueError("reps must be at least 1")
    if any(n <= 0 for n in n_values):
        raise ValueError("all n values must be positive")
    if any(p <= 0 for p in p_values):
        raise ValueError("all p values must be positive")
    if any(pi < 0 for pi in pi_values):
        raise ValueError("all pi values must be nonnegative")
    if any(tau <= 0 or tau >= 1 for tau in taus):
        raise ValueError("all tau values must be strictly between 0 and 1")

    designs: list[Design] = []
    seeds: set[int] = set()
    for dgp_idx, dgp in enumerate(dgps):
        for n_idx, n in enumerate(n_values):
            for p_idx, p in enumerate(p_values):
                for pi_idx, pi in enumerate(pi_values):
                    for tau_idx, tau in enumerate(taus):
                        for rep in range(reps):
                            seed = (
                                base_seed
                                + dgp_idx * 10_000_000
                                + n_idx * 1_000_000
                                + p_idx * 100_000
                                + pi_idx * 10_000
                                + tau_idx * 1_000
                                + rep
                            )
                            designs.append(
                                Design(
                                    dgp=dgp,
                                    n=n,
                                    p=p,
                                    pi=pi,
                                    tau=tau,
                                    rep=rep,
                                    seed=seed,
                                )
                            )
                            seeds.add(seed)

    if len(seeds) != len(designs):
        raise ValueError("generated seeds are not unique")
    return designs


def run_single_replication(
    design: Design,
    alphas: np.ndarray,
    estimators: tuple[str, ...] = ("full", "post_selection", "dml"),
    quantreg_max_iter: int = 500,
    selection_cv: int = 3,
    selection_max_iter: int = 10000,
    dml_k_folds: int = 3,
    dml_quantile_penalty: float = 0.01,
    dml_ridge_alpha: float = 1.0,
    dml_fold_random_state: int | None = 123,
    gmm_ridge: float = 1e-8,
) -> list[dict[str, object]]:
    """Generate one dataset and run requested estimators on it."""
    _validate_estimators(estimators)
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("alphas must be a nonempty one-dimensional array")

    data = generate_data(design)
    estimator_map: dict[str, EstimatorFn] = {
        "full": estimate_full_ivqr,
        "post_selection": estimate_post_selection_ivqr,
        "dml": estimate_dml_ivqr,
    }

    rows: list[dict[str, object]] = []
    for estimator_name in estimators:
        if estimator_name not in estimator_map:
            raise ValueError(f"Unknown estimator: {estimator_name}")

        try:
            estimator = estimator_map[estimator_name]
            if estimator_name == "post_selection":
                result = estimator(
                    data,
                    tau=design.tau,
                    alphas=alphas,
                    selection_cv=selection_cv,
                    selection_max_iter=selection_max_iter,
                    quantreg_max_iter=quantreg_max_iter,
                    gmm_ridge=gmm_ridge,
                )
            elif estimator_name == "dml":
                result = estimator(
                    data,
                    tau=design.tau,
                    alphas=alphas,
                    k_folds=dml_k_folds,
                    fold_random_state=dml_fold_random_state,
                    quantile_penalty=dml_quantile_penalty,
                    ridge_alpha=dml_ridge_alpha,
                    gmm_ridge=gmm_ridge,
                )
            else:
                result = estimator(
                    data,
                    tau=design.tau,
                    alphas=alphas,
                    max_iter=quantreg_max_iter,
                    gmm_ridge=gmm_ridge,
                )
            rows.append(_result_to_row(design, result))
        except Exception as exc:
            rows.append(_failure_row_for_estimator(design, estimator_name, alphas, exc))

    return rows


def run_simulation_batch(
    designs: list[Design],
    alphas: np.ndarray,
    estimators: tuple[str, ...] = ("full", "post_selection", "dml"),
    output_path: str | Path | None = None,
    append: bool = False,
    quantreg_max_iter: int = 500,
    selection_cv: int = 3,
    selection_max_iter: int = 10000,
    dml_k_folds: int = 5,
    dml_quantile_penalty: float = 0.01,
    dml_ridge_alpha: float = 1.0,
    dml_fold_random_state: int | None = 123,
    gmm_ridge: float = 1e-8,
) -> pd.DataFrame:
    """Run a batch of simulation designs and optionally persist it to CSV."""
    _validate_estimators(estimators)
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("alphas must be a nonempty one-dimensional array")

    rows: list[dict[str, object]] = []
    for design in designs:
        try:
            rows.extend(
                run_single_replication(
                    design,
                    alphas,
                    estimators=estimators,
                    quantreg_max_iter=quantreg_max_iter,
                    selection_cv=selection_cv,
                    selection_max_iter=selection_max_iter,
                    dml_k_folds=dml_k_folds,
                    dml_quantile_penalty=dml_quantile_penalty,
                    dml_ridge_alpha=dml_ridge_alpha,
                    dml_fold_random_state=dml_fold_random_state,
                    gmm_ridge=gmm_ridge,
                )
            )
        except Exception as exc:
            rows.extend(_failure_rows_for_design(design, estimators, alphas, exc))

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not (append and path.exists())
        results.to_csv(path, mode="a" if append else "w", header=write_header, index=False)
    return results


def observed_design_keys(results_path: str | Path) -> set[tuple[object, ...]]:
    """Return design keys with at least one persisted result row."""
    path = Path(results_path)
    if not path.exists():
        return set()

    try:
        existing = pd.read_csv(path, usecols=DESIGN_KEY_COLUMNS)
    except ValueError as exc:
        raise ValueError("results CSV is missing required design-key columns") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError("results CSV is empty or malformed") from exc

    return {_row_design_key(row) for _, row in existing.drop_duplicates().iterrows()}


def completed_design_keys(results_path: str | Path) -> set[tuple[object, ...]]:
    """Deprecated alias for observed_design_keys."""
    return observed_design_keys(results_path)


def filter_completed_designs(
    designs: list[Design],
    results_path: str | Path,
    estimators: tuple[str, ...],
    rerun_failed: bool = False,
) -> list[Design]:
    """Return designs that do not yet have all requested estimator rows."""
    _validate_estimators(estimators)
    path = Path(results_path)
    if not path.exists():
        return designs

    required_columns = DESIGN_KEY_COLUMNS + ["estimator"]
    if rerun_failed:
        required_columns += ["failed"]
    try:
        existing = pd.read_csv(path, usecols=required_columns)
    except ValueError as exc:
        raise ValueError("results CSV is missing required resume columns") from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError("results CSV is empty or malformed") from exc

    expected_estimators = {ESTIMATOR_OUTPUT_NAMES[estimator] for estimator in estimators}
    completed: dict[tuple[object, ...], set[str]] = {}
    for _, row in existing.iterrows():
        if rerun_failed and _as_bool(row["failed"]):
            continue
        key = _row_design_key(row)
        completed.setdefault(key, set()).add(str(row["estimator"]))

    return [
        design
        for design in designs
        if not expected_estimators.issubset(completed.get(_design_key(design), set()))
    ]


def run_pilot_simulation(
    dgp: str = "dgp1",
    n: int = 250,
    p: int = 200,
    pi: float = 1.0,
    tau: float = 0.5,
    reps: int = 10,
    base_seed: int = 12345,
    alphas: np.ndarray | None = None,
    estimators: tuple[str, ...] = ("full", "post_selection", "dml"),
    alpha_grid_size: int = 9,
    alpha_min: float = -1.0,
    alpha_max: float = 3.0,
    quantreg_max_iter: int = 500,
    selection_cv: int = 3,
    selection_max_iter: int = 10000,
    dml_k_folds: int = 3,
    dml_quantile_penalty: float = 0.01,
    dml_ridge_alpha: float = 1.0,
    gmm_ridge: float = 1e-8,
) -> pd.DataFrame:
    """Run a small pilot simulation and return raw estimator-level rows."""
    if reps < 1:
        raise ValueError("reps must be positive")
    if alpha_grid_size < 3:
        raise ValueError("alpha_grid_size must be at least 3")
    _validate_estimators(estimators)

    if alphas is None:
        alphas = np.linspace(alpha_min, alpha_max, alpha_grid_size)
    else:
        alphas = np.asarray(alphas, dtype=float)

    rows: list[dict[str, object]] = []
    for rep in range(reps):
        design = Design(
            dgp=dgp,
            n=n,
            p=p,
            pi=pi,
            tau=tau,
            rep=rep,
            seed=base_seed + rep,
        )
        rows.extend(
            run_single_replication(
                design,
                alphas,
                estimators=estimators,
                quantreg_max_iter=quantreg_max_iter,
                selection_cv=selection_cv,
                selection_max_iter=selection_max_iter,
                dml_k_folds=dml_k_folds,
                dml_quantile_penalty=dml_quantile_penalty,
                dml_ridge_alpha=dml_ridge_alpha,
                gmm_ridge=gmm_ridge,
            )
        )

    return pd.DataFrame(rows)
