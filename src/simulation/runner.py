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
from estimators.post_selection_ivqr_aligned import estimate_post_selection_ivqr_aligned
from estimators.post_selection_quantile_ivqr import (
    estimate_post_selection_quantile_ivqr,
)
from dgp.designs import Design
from simulation._validation import (
    validate_alpha_candidates,
    validate_bool,
    validate_design,
    validate_dgps,
    validate_estimators,
    validate_finite_float,
    validate_k_folds_for_designs,
    validate_nonnegative_float,
    validate_nonnegative_int,
    validate_optional_nonnegative_int,
    validate_positive_float,
    validate_positive_int,
    validate_probability_quantile,
    validate_unique_sequence,
)
from simulation.config import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_MAIN_ESTIMATORS,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    DEFAULT_QUANTREG_MAX_ITER,
    DGPS,
    MAIN_ESTIMATORS,
)
from simulation.results import (
    MAX_ERROR_MESSAGE_LENGTH,
    RESULT_COLUMNS,
    build_failure_result_row,
    build_simulation_result_row,
)


EstimatorFn = Callable[..., EstimationResult]
VALID_ESTIMATORS: tuple[str, ...] = MAIN_ESTIMATORS
DEFAULT_SIMULATION_ESTIMATORS: tuple[str, ...] = DEFAULT_MAIN_ESTIMATORS
VALID_DGPS: tuple[str, ...] = DGPS
ESTIMATOR_OUTPUT_NAMES = {
    "oracle": "oracle",
    "post_selection": "post_selection_ivqr",
    "post_selection_quantile": "post_selection_quantile",
    "post_selection_ivqr_aligned": "post_selection_ivqr_aligned",
    "dml": "dml_ivqr",
}
DESIGN_KEY_COLUMNS: tuple[str, ...] = ("dgp", "n", "p", "pi", "tau", "rep", "seed")


__all__ = [
    "DEFAULT_SIMULATION_ESTIMATORS",
    "DESIGN_KEY_COLUMNS",
    "ESTIMATOR_OUTPUT_NAMES",
    "MAX_ERROR_MESSAGE_LENGTH",
    "RESULT_COLUMNS",
    "VALID_DGPS",
    "VALID_ESTIMATORS",
    "failure_rows_for_design",
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


def _result_to_row(
    design: Design,
    result: EstimationResult,
    alphas: np.ndarray,
    critical_value_multiplier: float | None = None,
) -> dict[str, object]:
    row = build_simulation_result_row(design, result, alphas)
    if (
        critical_value_multiplier is not None
        and pd.isna(row["critical_value_multiplier"])
    ):
        row["critical_value_multiplier"] = critical_value_multiplier
    return row


def _short_error_message(exc: Exception) -> str:
    return str(exc)[:MAX_ERROR_MESSAGE_LENGTH]


def _failure_rows_for_design(
    design: Design,
    estimators: tuple[str, ...],
    alphas: np.ndarray,
    exc: Exception,
    critical_value_multiplier: float | None = None,
) -> list[dict[str, object]]:
    try:
        alpha_true = true_alpha(design.tau, design.dgp)
    except Exception:
        alpha_true = None

    message = f"{type(exc).__name__}: {_short_error_message(exc)}"
    return [
        _base_failure_row(
            design,
            estimator,
            alphas,
            alpha_true,
            exc,
            message,
            critical_value_multiplier=critical_value_multiplier,
        )
        for estimator in estimators
    ]


def failure_rows_for_design(
    design: Design,
    estimators: tuple[str, ...],
    alphas: np.ndarray,
    exc: Exception,
    critical_value_multiplier: float | None = None,
) -> list[dict[str, object]]:
    """Build failed result rows for every requested estimator."""
    return _failure_rows_for_design(
        design,
        estimators,
        alphas,
        exc,
        critical_value_multiplier=critical_value_multiplier,
    )


def _base_failure_row(
    design: Design,
    estimator: str,
    alphas: np.ndarray,
    alpha_true: float | None,
    exc: Exception,
    message: str,
    critical_value_multiplier: float | None = None,
) -> dict[str, object]:
    return build_failure_result_row(
        design=design,
        estimator=ESTIMATOR_OUTPUT_NAMES[estimator],
        alphas=alphas,
        alpha_true=alpha_true,
        exc=exc,
        message=message,
        max_error_message_length=MAX_ERROR_MESSAGE_LENGTH,
        critical_value_multiplier=critical_value_multiplier,
    )


def _failure_row_for_estimator(
    design: Design,
    estimator: str,
    alphas: np.ndarray,
    exc: Exception,
    critical_value_multiplier: float | None = None,
) -> dict[str, object]:
    """Build one failed result row for an unexpected estimator exception."""
    try:
        alpha_true = true_alpha(design.tau, design.dgp)
    except Exception:
        alpha_true = None

    message = f"Unexpected estimator error: {type(exc).__name__}: {_short_error_message(exc)}"
    return _base_failure_row(
        design,
        estimator,
        alphas,
        alpha_true,
        exc,
        message,
        critical_value_multiplier=critical_value_multiplier,
    )


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
    dgps = validate_dgps(dgps)
    n_values = validate_unique_sequence("n_values", n_values)
    p_values = validate_unique_sequence("p_values", p_values)
    pi_values = validate_unique_sequence("pi_values", pi_values)
    taus = validate_unique_sequence("taus", taus)
    reps = validate_positive_int("reps", reps)
    if reps >= 1000:
        raise ValueError("reps must be less than 1000 under the deterministic seed schedule")
    base_seed = validate_nonnegative_int("base_seed", base_seed)
    n_values = tuple(validate_positive_int("n", n) for n in n_values)
    p_values = tuple(validate_positive_int("p", p) for p in p_values)
    pi_values = tuple(validate_nonnegative_float("pi", pi) for pi in pi_values)
    taus = tuple(validate_probability_quantile("tau", tau) for tau in taus)

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
    critical_value_multiplier: float = DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    show_quantreg_warnings: bool = False,
) -> list[dict[str, object]]:
    """Generate one dataset and run requested estimators on it."""
    design = validate_design(design)
    estimators = validate_estimators(estimators)
    quantreg_max_iter = validate_positive_int("quantreg_max_iter", quantreg_max_iter)
    selection_cv = validate_positive_int("selection_cv", selection_cv)
    selection_max_iter = validate_positive_int("selection_max_iter", selection_max_iter)
    dml_k_folds = validate_k_folds_for_designs(dml_k_folds, [design])
    dml_quantile_penalty = validate_nonnegative_float(
        "dml_quantile_penalty", dml_quantile_penalty
    )
    dml_ridge_alpha = validate_nonnegative_float("dml_ridge_alpha", dml_ridge_alpha)
    gmm_ridge = validate_nonnegative_float("gmm_ridge", gmm_ridge)
    critical_value_multiplier = validate_positive_float(
        "critical_value_multiplier",
        critical_value_multiplier,
    )
    show_quantreg_warnings = validate_bool(
        "show_quantreg_warnings", show_quantreg_warnings
    )
    dml_fold_random_state = validate_optional_nonnegative_int(
        "dml_fold_random_state", dml_fold_random_state
    )
    alphas = validate_alpha_candidates(alphas)

    data = generate_data(design)
    estimator_map: dict[str, EstimatorFn] = {
        "oracle": estimate_oracle_ivqr,
        "post_selection": estimate_post_selection_ivqr,
        "post_selection_quantile": estimate_post_selection_quantile_ivqr,
        "post_selection_ivqr_aligned": estimate_post_selection_ivqr_aligned,
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
                        selection_random_state=design.seed,
                        critical_value_multiplier=critical_value_multiplier,
                    )
                elif estimator_name == "post_selection_quantile":
                    result = estimator(
                        data,
                        tau=design.tau,
                        alphas=alphas,
                        selection_cv=selection_cv,
                        selection_max_iter=selection_max_iter,
                        quantreg_max_iter=quantreg_max_iter,
                        selection_random_state=design.seed,
                        critical_value_multiplier=critical_value_multiplier,
                    )
                elif estimator_name == "post_selection_ivqr_aligned":
                    result = estimator(
                        data,
                        tau=design.tau,
                        alphas=alphas,
                        selection_cv=selection_cv,
                        selection_max_iter=selection_max_iter,
                        quantreg_max_iter=quantreg_max_iter,
                        selection_random_state=design.seed,
                        critical_value_multiplier=critical_value_multiplier,
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
                        critical_value_multiplier=critical_value_multiplier,
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
                        critical_value_multiplier=critical_value_multiplier,
                    )
                else:
                    raise ValueError(f"Estimator is not available in main runner: {estimator_name}")
            rows.append(
                _result_to_row(
                    design,
                    result,
                    alphas,
                    critical_value_multiplier=critical_value_multiplier,
                )
            )
        except Exception as exc:
            rows.append(
                _failure_row_for_estimator(
                    design,
                    estimator_name,
                    alphas,
                    exc,
                    critical_value_multiplier=critical_value_multiplier,
                )
            )

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
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    selection_cv: int = 3,
    selection_max_iter: int = 10000,
    dml_k_folds: int = DEFAULT_DML_K_FOLDS,
    dml_quantile_penalty: float = 0.01,
    dml_ridge_alpha: float = 1.0,
    gmm_ridge: float = 1e-8,
    critical_value_multiplier: float = DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    show_quantreg_warnings: bool = False,
) -> pd.DataFrame:
    """Run a small simulation and return raw estimator-level rows."""
    dgp = validate_dgps((dgp,))[0]
    n = validate_positive_int("n", n)
    p = validate_positive_int("p", p)
    pi = validate_nonnegative_float("pi", pi)
    tau = validate_probability_quantile("tau", tau)
    reps = validate_positive_int("reps", reps)
    base_seed = validate_nonnegative_int("base_seed", base_seed)
    alpha_grid_size = validate_positive_int("alpha_grid_size", alpha_grid_size)
    if alpha_grid_size < 3:
        raise ValueError("alpha_grid_size must be at least 3")
    alpha_min = validate_finite_float("alpha_min", alpha_min)
    alpha_max = validate_finite_float("alpha_max", alpha_max)
    if alpha_max <= alpha_min:
        raise ValueError("alpha_max must be greater than alpha_min")
    quantreg_max_iter = validate_positive_int("quantreg_max_iter", quantreg_max_iter)
    selection_cv = validate_positive_int("selection_cv", selection_cv)
    selection_max_iter = validate_positive_int("selection_max_iter", selection_max_iter)
    dml_k_folds = validate_k_folds_for_designs(
        dml_k_folds,
        [Design(dgp, n, p, pi, tau, rep=0, seed=base_seed)],
    )
    dml_quantile_penalty = validate_nonnegative_float(
        "dml_quantile_penalty", dml_quantile_penalty
    )
    dml_ridge_alpha = validate_nonnegative_float("dml_ridge_alpha", dml_ridge_alpha)
    gmm_ridge = validate_nonnegative_float("gmm_ridge", gmm_ridge)
    critical_value_multiplier = validate_positive_float(
        "critical_value_multiplier",
        critical_value_multiplier,
    )
    show_quantreg_warnings = validate_bool(
        "show_quantreg_warnings", show_quantreg_warnings
    )
    estimators = validate_estimators(estimators)

    if alphas is None:
        alphas = np.linspace(alpha_min, alpha_max, alpha_grid_size)
    alphas = validate_alpha_candidates(alphas)

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
                critical_value_multiplier=critical_value_multiplier,
                show_quantreg_warnings=show_quantreg_warnings,
            )
        )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
