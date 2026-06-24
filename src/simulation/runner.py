"""Monte Carlo runner utilities."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Callable
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from dgp.generators import generate_data
from dgp.true_parameters import get_oracle_control_indices, true_alpha
from estimators.base import EstimationResult
from estimators.dml_ivqr import estimate_dml_ivqr
from estimators.oracle_ivqr import estimate_oracle_ivqr
from estimators.post_selection_ivqr import estimate_post_selection_ivqr
from dgp.designs import Design
from simulation.config import (
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_QUANTREG_MAX_ITER,
    DGPS,
    MAIN_ESTIMATORS,
)


EstimatorFn = Callable[..., EstimationResult]
VALID_ESTIMATORS: tuple[str, ...] = MAIN_ESTIMATORS
DEFAULT_SIMULATION_ESTIMATORS: tuple[str, ...] = MAIN_ESTIMATORS
VALID_DGPS: tuple[str, ...] = DGPS
ESTIMATOR_OUTPUT_NAMES = {
    "oracle": "oracle",
    "post_selection": "post_selection_ivqr",
    "dml": "dml_ivqr",
}
RESULT_COLUMNS: tuple[str, ...] = (
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
    "absolute_error",
    "squared_error",
    "status",
    "error_type",
    "error_message",
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
)
DESIGN_KEY_COLUMNS: tuple[str, ...] = ("dgp", "n", "p", "pi", "tau", "rep", "seed")
MAX_ERROR_MESSAGE_LENGTH = 500


__all__ = [
    "DEFAULT_SIMULATION_ESTIMATORS",
    "DESIGN_KEY_COLUMNS",
    "ESTIMATOR_OUTPUT_NAMES",
    "MAX_ERROR_MESSAGE_LENGTH",
    "RESULT_COLUMNS",
    "VALID_DGPS",
    "VALID_ESTIMATORS",
    "make_simulation_grid",
    "quantreg_iteration_warning_filter",
    "run_single_replication",
    "run_small_simulation",
]


@contextmanager
def quantreg_iteration_warning_filter(show_warnings: bool = False):
    """Suppress repeated statsmodels QuantReg iteration-limit warnings by default."""
    if show_warnings:
        yield
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=IterationLimitWarning)
        yield


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_nonnegative_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _validate_probability_quantile(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must satisfy 0 < {name} < 1")
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must satisfy 0 < {name} < 1")
    return value


def _validate_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _validate_finite_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_alpha_candidates(alphas: np.ndarray) -> np.ndarray:
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or alphas.size == 0:
        raise ValueError("alphas must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(alphas)):
        raise ValueError("alphas must be finite")
    if not np.all(np.diff(alphas) > 0):
        raise ValueError("alphas must be strictly increasing")
    return alphas


def _validate_estimators(
    estimators: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    if isinstance(estimators, str):
        raise ValueError("estimators must be a sequence of estimator names")
    try:
        estimators = tuple(estimators)
    except TypeError as exc:
        raise ValueError("estimators must be a sequence of estimator names") from exc
    if len(estimators) == 0:
        raise ValueError("estimators must contain at least one estimator name")
    if any(not isinstance(estimator, str) for estimator in estimators):
        raise ValueError("estimators must contain only strings")
    if len(set(estimators)) != len(estimators):
        raise ValueError("estimators must not contain duplicates")
    invalid = sorted(set(estimators) - set(VALID_ESTIMATORS))
    if invalid:
        valid = ", ".join(VALID_ESTIMATORS)
        raise ValueError(f"Unknown estimator(s): {invalid}. Valid estimators: {valid}")
    return estimators


def _validate_dgps(dgps: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(dgps, str):
        raise ValueError("dgps must be a sequence of DGP names")
    try:
        dgps = tuple(dgps)
    except TypeError as exc:
        raise ValueError("dgps must be a sequence of DGP names") from exc
    if len(dgps) == 0:
        raise ValueError("dgps must be nonempty")
    if any(not isinstance(dgp, str) for dgp in dgps):
        raise ValueError("dgps must contain only strings")
    if len(set(dgps)) != len(dgps):
        raise ValueError("dgps must not contain duplicates")
    invalid_dgps = sorted(set(dgps) - set(VALID_DGPS))
    if invalid_dgps:
        raise ValueError(f"Unknown DGP(s): {invalid_dgps}. Valid DGPs: {VALID_DGPS}")
    return dgps


def _validate_unique_sequence(name: str, values: tuple | list) -> tuple:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty sequence")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty sequence") from exc
    if not values:
        raise ValueError(f"{name} must be nonempty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _result_to_row(design: Design, result: EstimationResult) -> dict[str, object]:
    bias = None
    absolute_error = None
    squared_error = None
    if result.alpha_hat is not None and result.alpha_true is not None:
        bias = result.alpha_hat - result.alpha_true
        absolute_error = abs(bias)
        squared_error = bias**2
    status = "failed" if result.failed else "ok"

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
        "status": status,
        "error_type": "EstimatorFailure" if result.failed else None,
        "error_message": result.message[:MAX_ERROR_MESSAGE_LENGTH]
        if result.failed
        else None,
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


def _short_error_message(exc: Exception) -> str:
    return str(exc)[:MAX_ERROR_MESSAGE_LENGTH]


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

    message = f"{type(exc).__name__}: {_short_error_message(exc)}"
    return [
        _base_failure_row(design, estimator, alphas, alpha_true, exc, message)
        for estimator in estimators
    ]


def _base_failure_row(
    design: Design,
    estimator: str,
    alphas: np.ndarray,
    alpha_true: float | None,
    exc: Exception,
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
        "absolute_error": None,
        "squared_error": None,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": _short_error_message(exc),
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

    message = f"Unexpected estimator error: {type(exc).__name__}: {_short_error_message(exc)}"
    return _base_failure_row(design, estimator, alphas, alpha_true, exc, message)


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
    dgps = _validate_dgps(dgps)
    n_values = _validate_unique_sequence("n_values", n_values)
    p_values = _validate_unique_sequence("p_values", p_values)
    pi_values = _validate_unique_sequence("pi_values", pi_values)
    taus = _validate_unique_sequence("taus", taus)
    reps = _validate_positive_int("reps", reps)
    if reps >= 1000:
        raise ValueError("reps must be less than 1000 under the deterministic seed schedule")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError("base_seed must be an integer")
    n_values = tuple(_validate_positive_int("n", n) for n in n_values)
    p_values = tuple(_validate_positive_int("p", p) for p in p_values)
    pi_values = tuple(_validate_nonnegative_float("pi", pi) for pi in pi_values)
    taus = tuple(_validate_probability_quantile("tau", tau) for tau in taus)

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
    estimators: tuple[str, ...] = DEFAULT_SIMULATION_ESTIMATORS,
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    selection_cv: int = 3,
    selection_max_iter: int = 10000,
    dml_k_folds: int = DEFAULT_DML_K_FOLDS,
    dml_quantile_penalty: float = 0.01,
    dml_ridge_alpha: float = 1.0,
    dml_fold_random_state: int | None = None,
    gmm_ridge: float = 1e-8,
    show_quantreg_warnings: bool = False,
) -> list[dict[str, object]]:
    """Generate one dataset and run requested estimators on it."""
    estimators = _validate_estimators(estimators)
    quantreg_max_iter = _validate_positive_int("quantreg_max_iter", quantreg_max_iter)
    selection_cv = _validate_positive_int("selection_cv", selection_cv)
    selection_max_iter = _validate_positive_int("selection_max_iter", selection_max_iter)
    dml_k_folds = _validate_positive_int("dml_k_folds", dml_k_folds)
    dml_quantile_penalty = _validate_nonnegative_float(
        "dml_quantile_penalty", dml_quantile_penalty
    )
    dml_ridge_alpha = _validate_nonnegative_float("dml_ridge_alpha", dml_ridge_alpha)
    gmm_ridge = _validate_nonnegative_float("gmm_ridge", gmm_ridge)
    show_quantreg_warnings = _validate_bool(
        "show_quantreg_warnings", show_quantreg_warnings
    )
    if dml_fold_random_state is not None and (
        not isinstance(dml_fold_random_state, int)
        or isinstance(dml_fold_random_state, bool)
    ):
        raise ValueError("dml_fold_random_state must be an integer or None")
    alphas = _validate_alpha_candidates(alphas)

    data = generate_data(design)
    estimator_map: dict[str, EstimatorFn] = {
        "oracle": estimate_oracle_ivqr,
        "post_selection": estimate_post_selection_ivqr,
        "dml": estimate_dml_ivqr,
    }

    rows: list[dict[str, object]] = []
    for estimator_name in estimators:
        if estimator_name not in estimator_map:
            raise ValueError(f"Unknown estimator: {estimator_name}")

        try:
            estimator = estimator_map[estimator_name]
            with quantreg_iteration_warning_filter(show_quantreg_warnings):
                if estimator_name == "post_selection":
                    result = estimator(
                        data,
                        tau=design.tau,
                        alphas=alphas,
                        selection_cv=selection_cv,
                        selection_max_iter=selection_max_iter,
                        quantreg_max_iter=quantreg_max_iter,
                    )
                elif estimator_name == "dml":
                    fold_random_state = (
                        design.seed
                        if dml_fold_random_state is None
                        else dml_fold_random_state
                    )
                    result = estimator(
                        data,
                        tau=design.tau,
                        alphas=alphas,
                        k_folds=dml_k_folds,
                        fold_random_state=fold_random_state,
                        quantile_penalty=dml_quantile_penalty,
                        ridge_alpha=dml_ridge_alpha,
                        gmm_ridge=gmm_ridge,
                    )
                elif estimator_name == "oracle":
                    oracle_indices = get_oracle_control_indices(design.dgp, design.p)
                    result = estimator(
                        data,
                        tau=design.tau,
                        alphas=alphas,
                        oracle_indices=oracle_indices,
                        max_iter=quantreg_max_iter,
                        gmm_ridge=gmm_ridge,
                    )
                else:
                    raise ValueError(f"Estimator is not available in main runner: {estimator_name}")
            rows.append(_result_to_row(design, result))
        except Exception as exc:
            rows.append(_failure_row_for_estimator(design, estimator_name, alphas, exc))

    return rows


def run_small_simulation(
    dgp: str = "dgp1",
    n: int = 250,
    p: int = 200,
    pi: float = 1.0,
    tau: float = 0.5,
    reps: int = 10,
    base_seed: int = 12345,
    alphas: np.ndarray | None = None,
    estimators: tuple[str, ...] = DEFAULT_SIMULATION_ESTIMATORS,
    alpha_grid_size: int = DEFAULT_ALPHA_GRID_SIZE,
    alpha_min: float = -1.0,
    alpha_max: float = 3.0,
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    selection_cv: int = 3,
    selection_max_iter: int = 10000,
    dml_k_folds: int = DEFAULT_DML_K_FOLDS,
    dml_quantile_penalty: float = 0.01,
    dml_ridge_alpha: float = 1.0,
    gmm_ridge: float = 1e-8,
    show_quantreg_warnings: bool = False,
) -> pd.DataFrame:
    """Run a small simulation and return raw estimator-level rows."""
    dgp = _validate_dgps((dgp,))[0]
    n = _validate_positive_int("n", n)
    p = _validate_positive_int("p", p)
    pi = _validate_nonnegative_float("pi", pi)
    tau = _validate_probability_quantile("tau", tau)
    reps = _validate_positive_int("reps", reps)
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise ValueError("base_seed must be an integer")
    alpha_grid_size = _validate_positive_int("alpha_grid_size", alpha_grid_size)
    if alpha_grid_size < 3:
        raise ValueError("alpha_grid_size must be at least 3")
    alpha_min = _validate_finite_float("alpha_min", alpha_min)
    alpha_max = _validate_finite_float("alpha_max", alpha_max)
    if alpha_max <= alpha_min:
        raise ValueError("alpha_max must be greater than alpha_min")
    quantreg_max_iter = _validate_positive_int("quantreg_max_iter", quantreg_max_iter)
    selection_cv = _validate_positive_int("selection_cv", selection_cv)
    selection_max_iter = _validate_positive_int("selection_max_iter", selection_max_iter)
    dml_k_folds = _validate_positive_int("dml_k_folds", dml_k_folds)
    dml_quantile_penalty = _validate_nonnegative_float(
        "dml_quantile_penalty", dml_quantile_penalty
    )
    dml_ridge_alpha = _validate_nonnegative_float("dml_ridge_alpha", dml_ridge_alpha)
    gmm_ridge = _validate_nonnegative_float("gmm_ridge", gmm_ridge)
    show_quantreg_warnings = _validate_bool(
        "show_quantreg_warnings", show_quantreg_warnings
    )
    estimators = _validate_estimators(estimators)

    if alphas is None:
        alphas = np.linspace(alpha_min, alpha_max, alpha_grid_size)
    alphas = _validate_alpha_candidates(alphas)

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
                show_quantreg_warnings=show_quantreg_warnings,
            )
        )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
