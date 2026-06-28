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
from estimators.base import EstimationResult, POST_SELECTION_DIAGNOSTIC_FIELDS
from estimators.dml_ivqr import estimate_dml_ivqr
from estimators.oracle_ivqr import estimate_oracle_ivqr
from estimators.post_selection_ivqr import (
    empty_post_selection_diagnostics,
    estimate_post_selection_ivqr,
)
from dgp.designs import Design
from inference.confidence_regions import summarize_alpha_grid_diagnostics
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
    validate_positive_int,
    validate_probability_quantile,
    validate_unique_sequence,
)
from simulation.config import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
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
    "alpha_grid_min",
    "alpha_grid_max",
    "alpha_grid_size",
    "alpha_grid_step",
    "alpha_hat_at_lower_boundary",
    "alpha_hat_at_upper_boundary",
    "alpha_hat_at_any_boundary",
    "cr_lower",
    "cr_upper",
    "cr_length",
    "cr_hits_lower_boundary",
    "cr_hits_upper_boundary",
    "cr_hits_any_boundary",
    "cr_empty",
    "cr_accepted_alpha_count",
    "cr_acceptance_rate",
    "cr_n_blocks",
    "cr_disconnected",
    "cr_hull_length",
    "cr_covers_true",
    "selected_controls",
    "runtime_seconds",
    "failed_alpha_count",
    "failed_alpha_rate",
    "min_test_stat",
    "max_test_stat",
    "test_stat_at_alpha_hat",
    "critical_value",
    "ps_n_selected_controls",
    "ps_n_selected_instruments",
    "ps_n_selected_total",
    "ps_share_selected_controls",
    "ps_share_selected_instruments",
    "ps_selected_no_controls",
    "ps_selected_no_instruments",
    "ps_selected_empty_total",
    "ps_first_stage_r2",
    "ps_first_stage_adj_r2",
    "ps_first_stage_partial_r2",
    "ps_first_stage_f_stat",
    "ps_first_stage_condition_number",
    "ps_selection_method",
    "ps_lasso_alpha_controls",
    "ps_lasso_alpha_instruments",
    "ps_lasso_alpha_first_stage",
    "ps_lasso_cv_folds",
    "ps_selection_failed",
    "ps_first_stage_failed",
    "ps_rank_deficient",
    "ps_warning_code",
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


def _diagnostic_value(
    result: EstimationResult,
    name: str,
    fallback: object,
) -> object:
    value = getattr(result, name)
    return fallback if value is None else value


def _result_diagnostics(
    result: EstimationResult,
    alphas: np.ndarray,
) -> dict[str, object]:
    failed_alpha_count = result.failed_alpha_count
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=alphas,
        accepted_mask=None,
        alpha_hat=result.alpha_hat,
        failed_alpha_count=0 if failed_alpha_count is None else failed_alpha_count,
    )
    if failed_alpha_count is None:
        diagnostics["failed_alpha_count"] = None
        diagnostics["failed_alpha_rate"] = np.nan

    for name in (
        "alpha_grid_min",
        "alpha_grid_max",
        "alpha_grid_step",
        "alpha_hat_at_lower_boundary",
        "alpha_hat_at_upper_boundary",
        "alpha_hat_at_any_boundary",
        "cr_hits_lower_boundary",
        "cr_hits_upper_boundary",
        "cr_hits_any_boundary",
        "cr_accepted_alpha_count",
        "cr_acceptance_rate",
        "cr_n_blocks",
        "cr_hull_length",
        "failed_alpha_rate",
        "min_test_stat",
        "max_test_stat",
        "test_stat_at_alpha_hat",
        "critical_value",
    ):
        diagnostics[name] = _diagnostic_value(result, name, diagnostics[name])

    diagnostics["alpha_grid_size"] = _diagnostic_value(
        result,
        "alpha_grid_size",
        diagnostics["alpha_grid_size"],
    )
    diagnostics["failed_alpha_count"] = _diagnostic_value(
        result,
        "failed_alpha_count",
        diagnostics["failed_alpha_count"],
    )
    diagnostics["cr_lower"] = result.cr_lower if result.cr_lower is not None else diagnostics["cr_lower"]
    diagnostics["cr_upper"] = result.cr_upper if result.cr_upper is not None else diagnostics["cr_upper"]
    diagnostics["cr_length"] = result.cr_length if result.cr_length is not None else diagnostics["cr_length"]
    diagnostics["cr_empty"] = result.cr_empty
    diagnostics["cr_disconnected"] = (
        result.cr_disconnected
        if result.cr_disconnected is not None
        else diagnostics["cr_disconnected"]
    )
    return diagnostics


def _post_selection_diagnostics(result: EstimationResult) -> dict[str, object]:
    diagnostics = empty_post_selection_diagnostics()
    for name in POST_SELECTION_DIAGNOSTIC_FIELDS:
        value = getattr(result, name)
        if value is not None:
            diagnostics[name] = value
    return diagnostics


def _result_to_row(
    design: Design,
    result: EstimationResult,
    alphas: np.ndarray,
) -> dict[str, object]:
    bias = None
    absolute_error = None
    squared_error = None
    if result.alpha_hat is not None and result.alpha_true is not None:
        bias = result.alpha_hat - result.alpha_true
        absolute_error = abs(bias)
        squared_error = bias**2
    status = "failed" if result.failed else "ok"
    diagnostics = _result_diagnostics(result, alphas)
    ps_diagnostics = _post_selection_diagnostics(result)

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
        "alpha_grid_min": diagnostics["alpha_grid_min"],
        "alpha_grid_max": diagnostics["alpha_grid_max"],
        "alpha_grid_size": diagnostics["alpha_grid_size"],
        "alpha_grid_step": diagnostics["alpha_grid_step"],
        "alpha_hat_at_lower_boundary": diagnostics["alpha_hat_at_lower_boundary"],
        "alpha_hat_at_upper_boundary": diagnostics["alpha_hat_at_upper_boundary"],
        "alpha_hat_at_any_boundary": diagnostics["alpha_hat_at_any_boundary"],
        "cr_lower": diagnostics["cr_lower"],
        "cr_upper": diagnostics["cr_upper"],
        "cr_length": diagnostics["cr_length"],
        "cr_hits_lower_boundary": diagnostics["cr_hits_lower_boundary"],
        "cr_hits_upper_boundary": diagnostics["cr_hits_upper_boundary"],
        "cr_hits_any_boundary": diagnostics["cr_hits_any_boundary"],
        "cr_empty": diagnostics["cr_empty"],
        "cr_accepted_alpha_count": diagnostics["cr_accepted_alpha_count"],
        "cr_acceptance_rate": diagnostics["cr_acceptance_rate"],
        "cr_n_blocks": diagnostics["cr_n_blocks"],
        "cr_disconnected": diagnostics["cr_disconnected"],
        "cr_hull_length": diagnostics["cr_hull_length"],
        "cr_covers_true": result.cr_covers_true,
        "selected_controls": result.selected_controls,
        "runtime_seconds": result.runtime_seconds,
        "failed_alpha_count": diagnostics["failed_alpha_count"],
        "failed_alpha_rate": diagnostics["failed_alpha_rate"],
        "min_test_stat": diagnostics["min_test_stat"],
        "max_test_stat": diagnostics["max_test_stat"],
        "test_stat_at_alpha_hat": diagnostics["test_stat_at_alpha_hat"],
        "critical_value": diagnostics["critical_value"],
        **ps_diagnostics,
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


def failure_rows_for_design(
    design: Design,
    estimators: tuple[str, ...],
    alphas: np.ndarray,
    exc: Exception,
) -> list[dict[str, object]]:
    """Build failed result rows for every requested estimator."""
    return _failure_rows_for_design(design, estimators, alphas, exc)


def _base_failure_row(
    design: Design,
    estimator: str,
    alphas: np.ndarray,
    alpha_true: float | None,
    exc: Exception,
    message: str,
) -> dict[str, object]:
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=alphas,
        accepted_mask=None,
        alpha_hat=None,
        failed_alpha_count=0,
    )
    diagnostics["failed_alpha_count"] = None
    diagnostics["failed_alpha_rate"] = np.nan
    ps_diagnostics = empty_post_selection_diagnostics()
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
        "alpha_grid_min": diagnostics["alpha_grid_min"],
        "alpha_grid_max": diagnostics["alpha_grid_max"],
        "alpha_grid_size": diagnostics["alpha_grid_size"],
        "alpha_grid_step": diagnostics["alpha_grid_step"],
        "alpha_hat_at_lower_boundary": diagnostics["alpha_hat_at_lower_boundary"],
        "alpha_hat_at_upper_boundary": diagnostics["alpha_hat_at_upper_boundary"],
        "alpha_hat_at_any_boundary": diagnostics["alpha_hat_at_any_boundary"],
        "cr_lower": diagnostics["cr_lower"],
        "cr_upper": diagnostics["cr_upper"],
        "cr_length": diagnostics["cr_length"],
        "cr_hits_lower_boundary": diagnostics["cr_hits_lower_boundary"],
        "cr_hits_upper_boundary": diagnostics["cr_hits_upper_boundary"],
        "cr_hits_any_boundary": diagnostics["cr_hits_any_boundary"],
        "cr_empty": diagnostics["cr_empty"],
        "cr_accepted_alpha_count": diagnostics["cr_accepted_alpha_count"],
        "cr_acceptance_rate": diagnostics["cr_acceptance_rate"],
        "cr_n_blocks": diagnostics["cr_n_blocks"],
        "cr_disconnected": diagnostics["cr_disconnected"],
        "cr_hull_length": diagnostics["cr_hull_length"],
        "cr_covers_true": None,
        "selected_controls": None,
        "runtime_seconds": None,
        "failed_alpha_count": diagnostics["failed_alpha_count"],
        "failed_alpha_rate": diagnostics["failed_alpha_rate"],
        "min_test_stat": diagnostics["min_test_stat"],
        "max_test_stat": diagnostics["max_test_stat"],
        "test_stat_at_alpha_hat": diagnostics["test_stat_at_alpha_hat"],
        "critical_value": diagnostics["critical_value"],
        **ps_diagnostics,
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
            rows.append(_result_to_row(design, result, alphas))
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
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
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
                show_quantreg_warnings=show_quantreg_warnings,
            )
        )

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
