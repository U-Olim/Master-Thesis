"""Single Monte Carlo simulation engine."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import TypeVar, cast, get_args
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from dgp.designs import Design
from dgp.generators import generate_data
from dgp.true_parameters import get_oracle_control_indices, true_alpha
from estimators.base import EstimationResult
from estimators.dml import QuantileSolver, estimate_dml_ivqr
from estimators.oracle import estimate_oracle_ivqr
from estimators.post_selection import (
    estimate_post_selection_ivqr,
    validate_selection_lasso_multiplier,
)
from ivqr.ch_inverse import (
    validate_grid_strategy,
    validate_hard_failure_policy,
    validate_iteration_warning_policy,
)
from simulation.config import (
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_BASE_SEED,
    DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_DML_QUANTILE_PENALTY,
    DEFAULT_DML_QUANTILE_SOLVER,
    DEFAULT_DML_RIDGE_ALPHA,
    DEFAULT_ESTIMATORS,
    DEFAULT_N_JOBS,
    DEFAULT_GRID_STRATEGY,
    DEFAULT_REFINEMENT_TOLERANCE,
    DEFAULT_MAX_REFINEMENT_DEPTH,
    DEFAULT_MAX_ALPHA_EVALUATIONS,
    DEFAULT_ITERATION_WARNING_POLICY,
    DEFAULT_HARD_FAILURE_POLICY,
    DEFAULT_QUANTREG_MAX_ITER,
    DGPS,
    ESTIMATORS,
)
from simulation.dml_output import clean_core_results_frame, clean_dml_results_frame
from simulation.post_selection_output import clean_post_selection_results_frame
from simulation.results import (
    MAX_ERROR_MESSAGE_LENGTH,
    RESULT_COLUMNS,
    build_failure_result_row,
    build_simulation_result_row,
)
from utils.validation import validate_alpha_grid, validate_positive_int, validate_tau


_T = TypeVar("_T")
VALID_ESTIMATORS: tuple[str, ...] = ESTIMATORS
DEFAULT_SIMULATION_ESTIMATORS: tuple[str, ...] = DEFAULT_ESTIMATORS
VALID_DGPS: tuple[str, ...] = DGPS
ESTIMATOR_OUTPUT_NAMES: dict[str, str] = {
    "oracle": "oracle",
    "post_selection": "post_selection_ivqr",
    "dml": "dml_ivqr",
}
ESTIMATOR_ALIASES: dict[str, str] = {
    "oracle": "oracle",
    "oracle_ivqr": "oracle",
    "post_selection": "post_selection",
    "post_selection_ivqr": "post_selection",
    "post-selection": "post_selection",
    "post-selection-ivqr": "post_selection",
    "post_selection-ivqr": "post_selection",
    "dml": "dml",
    "dml_ivqr": "dml",
    "dml-ivqr": "dml",
}
DESIGN_KEY_COLUMNS: tuple[str, ...] = ("dgp", "n", "p", "pi", "tau", "rep", "seed")
SEED_RULE_TEXT = (
    "seed = sha256(base_seed, dgp, n, p, pi, tau, rep), "
    "independent of estimator and execution order"
)


__all__ = [
    "DEFAULT_SIMULATION_ESTIMATORS",
    "DESIGN_KEY_COLUMNS",
    "ESTIMATOR_ALIASES",
    "ESTIMATOR_OUTPUT_NAMES",
    "RESULT_COLUMNS",
    "VALID_DGPS",
    "VALID_ESTIMATORS",
    "filter_completed_designs",
    "make_design_seed",
    "make_simulation_grid",
    "normalize_estimator_names",
    "quantreg_iteration_warning_filter",
    "run_simulation_batch",
    "run_simulation_design",
    "run_single_replication",
    "run_small_simulation",
    "SEED_RULE_TEXT",
]


@dataclass(frozen=True)
class WorkerArgs:
    design: Design
    alphas: np.ndarray
    estimators: tuple[str, ...]
    quantreg_max_iter: int
    dml_k_folds: int
    dml_quantile_penalty: float
    dml_ridge_alpha: float
    dml_quantile_solver: str
    gmm_ridge: float
    critical_value_multiplier: float
    selection_lasso_multiplier: float
    show_quantreg_warnings: bool
    grid_strategy: str
    refinement_tolerance: float
    max_refinement_depth: int
    max_alpha_evaluations: int
    iteration_warning_policy: str
    hard_failure_policy: str


@contextmanager
def quantreg_iteration_warning_filter(show_warnings: bool = False):
    """Suppress repeated statsmodels QuantReg iteration-limit warnings by default."""
    if show_warnings:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=IterationLimitWarning)
        warnings.filterwarnings("ignore", message=r"Maximum number of iterations reached.*")
        yield


def normalize_estimator_names(raw_estimators: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize estimator names and aliases to canonical four-estimator names."""
    if raw_estimators is None:
        return DEFAULT_SIMULATION_ESTIMATORS
    if isinstance(raw_estimators, str):
        raise ValueError("estimators must be a sequence of estimator names")

    normalized: list[str] = []
    invalid: list[str] = []
    for raw in raw_estimators:
        if not isinstance(raw, str) or not raw.strip():
            invalid.append(str(raw))
            continue
        token = raw.strip().lower().replace(" ", "_")
        canonical = ESTIMATOR_ALIASES.get(token)
        if canonical is None:
            invalid.append(raw)
            continue
        if canonical not in normalized:
            normalized.append(canonical)
    if invalid:
        valid = ", ".join(VALID_ESTIMATORS)
        aliases = ", ".join(sorted(ESTIMATOR_ALIASES))
        raise ValueError(
            f"Unknown estimator name(s): {invalid}. "
            f"Valid estimators: {valid}. Supported aliases: {aliases}."
        )
    if not normalized:
        raise ValueError("estimators must contain at least one estimator name")
    return tuple(normalized)


def _validate_estimators(estimators: Sequence[str]) -> tuple[str, ...]:
    estimators = normalize_estimator_names(estimators)
    invalid = sorted(set(estimators) - set(VALID_ESTIMATORS))
    if invalid:
        valid = ", ".join(VALID_ESTIMATORS)
        raise ValueError(f"Unknown estimator(s): {invalid}. Valid estimators: {valid}")
    return estimators


def _validate_unique_sequence(name: str, values: Sequence[_T]) -> tuple[_T, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    try:
        values_tuple = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc
    if not values_tuple:
        raise ValueError(f"{name} must not be empty")
    seen: list[_T] = []
    for value in values_tuple:
        if value in seen:
            raise ValueError(f"{name} must not contain duplicates")
        seen.append(value)
    return values_tuple


def _validate_dgp_sequence(name: str, values: Sequence[str]) -> tuple[str, ...]:
    dgps = tuple(str(dgp) for dgp in _validate_unique_sequence(name, values))
    invalid_dgps = sorted(set(dgps) - set(VALID_DGPS))
    if invalid_dgps:
        raise ValueError(f"Unknown DGP(s): {invalid_dgps}")
    return dgps


def _validate_positive_int_sequence(
    name: str,
    values: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        validate_positive_int(name, int(value))
        for value in _validate_unique_sequence(name, values)
    )


def _validate_nonnegative_float_sequence(
    name: str,
    values: Sequence[float],
) -> tuple[float, ...]:
    return tuple(
        _validate_nonnegative_float(name, float(value))
        for value in _validate_unique_sequence(name, values)
    )


def _validate_tau_sequence(name: str, values: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        validate_tau(float(value))
        for value in _validate_unique_sequence(name, values)
    )


def _validate_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_positive_float(name: str, value: float) -> float:
    value = _validate_float(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_nonnegative_float(name: str, value: float) -> float:
    value = _validate_float(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _validate_nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def make_design_seed(
    *,
    base_seed: int,
    dgp: str,
    n: int,
    p: int,
    pi: float,
    tau: float,
    rep: int,
) -> int:
    """Return a stable design-cell seed independent of estimator execution."""
    base_seed = _validate_nonnegative_int("base_seed", base_seed)
    dgp = str(dgp).lower()
    if dgp not in VALID_DGPS:
        raise ValueError(f"Unknown DGP: {dgp}")
    n = validate_positive_int("n", int(n))
    p = validate_positive_int("p", int(p))
    pi = _validate_nonnegative_float("pi", float(pi))
    tau = validate_tau(float(tau))
    rep = _validate_nonnegative_int("rep", rep)

    key = f"{base_seed}|{dgp}|{n}|{p}|{pi:.12g}|{tau:.12g}|{rep}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # Use a 63-bit positive seed to make collisions negligible on the full
    # Monte Carlo grid while remaining portable in CSV/JSON integer fields.
    seed = int(digest[:16], 16) % (2**63 - 1)
    return seed if seed != 0 else 1


def _validate_design(design: Design) -> Design:
    if not isinstance(design, Design):
        raise ValueError("design must be a Design object")
    if design.dgp not in VALID_DGPS:
        raise ValueError(f"Unknown DGP: {design.dgp}")
    validate_positive_int("n", design.n)
    validate_positive_int("p", design.p)
    _validate_nonnegative_float("pi", design.pi)
    validate_tau(design.tau)
    _validate_nonnegative_int("rep", design.rep)
    _validate_nonnegative_int("seed", design.seed)
    return design


def _design_key(design: Design) -> tuple[object, ...]:
    return (design.dgp, design.n, design.p, design.pi, design.tau, design.rep, design.seed)


def _row_design_key(row: pd.Series) -> tuple[object, ...]:
    return (
        str(row["dgp"]),
        int(row["n"]),
        int(row["p"]),
        float(row["pi"]),
        float(row["tau"]),
        int(row["rep"]),
        int(row["seed"]),
    )


def make_simulation_grid(
    dgps: tuple[str, ...],
    n_values: tuple[int, ...],
    p_values: tuple[int, ...],
    pi_values: tuple[float, ...],
    taus: tuple[float, ...],
    reps: int,
    base_seed: int = DEFAULT_BASE_SEED,
    rep_start: int = 0,
    rep_end: int | None = None,
) -> list[Design]:
    """Create the deterministic full Monte Carlo design grid.

    Each design seed is a stable hash of the design cell, so data generation is
    independent of estimator list, estimator order, workers, batch size, and
    resume status. ``reps`` is the full planned replication count; ``rep_start``
    and ``rep_end`` select the global replication indices included in this grid.
    """
    dgps = _validate_dgp_sequence("dgps", dgps)
    n_values = _validate_positive_int_sequence("n_values", n_values)
    p_values = _validate_positive_int_sequence("p_values", p_values)
    pi_values = _validate_nonnegative_float_sequence("pi_values", pi_values)
    taus = _validate_tau_sequence("taus", taus)
    reps = validate_positive_int("reps", reps)
    base_seed = _validate_nonnegative_int("base_seed", base_seed)
    rep_start = _validate_nonnegative_int("rep_start", rep_start)
    rep_end = reps - 1 if rep_end is None else _validate_nonnegative_int("rep_end", rep_end)
    if rep_end < rep_start:
        raise ValueError("rep_end must be greater than or equal to rep_start")
    if rep_end >= reps:
        raise ValueError("rep_end must be less than reps")

    designs: list[Design] = []
    seeds: set[int] = set()
    for dgp in dgps:
        for n in n_values:
            for p in p_values:
                for pi in pi_values:
                    for tau in taus:
                        for rep in range(rep_start, rep_end + 1):
                            seed = make_design_seed(
                                base_seed=base_seed,
                                dgp=dgp,
                                n=n,
                                p=p,
                                pi=pi,
                                tau=tau,
                                rep=rep,
                            )
                            designs.append(Design(dgp, n, p, pi, tau, rep, seed))
                            seeds.add(seed)
    if len(seeds) != len(designs):
        raise ValueError("generated seeds are not unique")
    return designs


def _result_to_row(
    design: Design,
    result: EstimationResult,
    alphas: np.ndarray,
    critical_value_multiplier: float,
) -> dict[str, object]:
    row = build_simulation_result_row(design, result, alphas)
    if pd.isna(row["critical_value_multiplier"]):
        row["critical_value_multiplier"] = critical_value_multiplier
    return row


def _failure_row_for_estimator(
    design: Design,
    estimator: str,
    alphas: np.ndarray,
    exc: Exception,
    critical_value_multiplier: float,
) -> dict[str, object]:
    try:
        alpha_true = true_alpha(design.tau, design.dgp)
    except Exception:
        alpha_true = None
    message = f"Unexpected estimator error: {type(exc).__name__}: {str(exc)[:MAX_ERROR_MESSAGE_LENGTH]}"
    return build_failure_result_row(
        design=design,
        estimator=ESTIMATOR_OUTPUT_NAMES[estimator],
        alphas=alphas,
        alpha_true=alpha_true,
        exc=exc,
        message=message,
        critical_value_multiplier=critical_value_multiplier,
    )


def _estimator_random_state(design_seed: int) -> int:
    """Return a deterministic sklearn-compatible random state for estimators."""
    return int(design_seed % (2**32 - 1))


def _validate_dml_quantile_solver(solver: str) -> QuantileSolver:
    """Validate and narrow a runner-level solver string."""
    if solver not in get_args(QuantileSolver):
        raise ValueError(f"Unknown quantile solver: {solver}")
    return cast(QuantileSolver, solver)


def run_simulation_design(
    design: Design,
    alphas: np.ndarray,
    estimators: tuple[str, ...] = DEFAULT_SIMULATION_ESTIMATORS,
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    dml_k_folds: int = DEFAULT_DML_K_FOLDS,
    dml_quantile_penalty: float = DEFAULT_DML_QUANTILE_PENALTY,
    dml_ridge_alpha: float = DEFAULT_DML_RIDGE_ALPHA,
    dml_quantile_solver: str = DEFAULT_DML_QUANTILE_SOLVER,
    gmm_ridge: float = 1e-8,
    critical_value_multiplier: float = DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    selection_lasso_multiplier: float = 1.0,
    show_quantreg_warnings: bool = False,
    grid_strategy: str = DEFAULT_GRID_STRATEGY,
    refinement_tolerance: float = DEFAULT_REFINEMENT_TOLERANCE,
    max_refinement_depth: int = DEFAULT_MAX_REFINEMENT_DEPTH,
    max_alpha_evaluations: int = DEFAULT_MAX_ALPHA_EVALUATIONS,
    iteration_warning_policy: str = DEFAULT_ITERATION_WARNING_POLICY,
    hard_failure_policy: str = DEFAULT_HARD_FAILURE_POLICY,
) -> list[dict[str, object]]:
    """Generate one dataset and run requested estimators on it."""
    design = _validate_design(design)
    estimators = _validate_estimators(estimators)
    quantreg_max_iter = validate_positive_int("quantreg_max_iter", quantreg_max_iter)
    dml_k_folds = validate_positive_int("dml_k_folds", dml_k_folds)
    if dml_k_folds < 2 or dml_k_folds > design.n:
        raise ValueError("dml_k_folds must satisfy 2 <= dml_k_folds <= n")
    dml_quantile_penalty = _validate_nonnegative_float(
        "dml_quantile_penalty", dml_quantile_penalty
    )
    dml_ridge_alpha = _validate_nonnegative_float("dml_ridge_alpha", dml_ridge_alpha)
    dml_quantile_solver = str(dml_quantile_solver)
    gmm_ridge = _validate_nonnegative_float("gmm_ridge", gmm_ridge)
    critical_value_multiplier = _validate_positive_float(
        "critical_value_multiplier", critical_value_multiplier
    )
    selection_lasso_multiplier = validate_selection_lasso_multiplier(
        selection_lasso_multiplier
    )
    alphas = validate_alpha_grid(alphas)
    grid_strategy = validate_grid_strategy(grid_strategy)
    iteration_warning_policy = validate_iteration_warning_policy(
        iteration_warning_policy
    )
    hard_failure_policy = validate_hard_failure_policy(hard_failure_policy)

    data = generate_data(design)
    estimator_random_state = _estimator_random_state(design.seed)
    rows: list[dict[str, object]] = []
    for estimator_name in estimators:
        try:
            with quantreg_iteration_warning_filter(show_quantreg_warnings):
                if estimator_name == "oracle":
                    result = estimate_oracle_ivqr(
                        data,
                        tau=design.tau,
                        alphas=alphas,
                        oracle_indices=get_oracle_control_indices(design.dgp, design.p),
                        max_iter=quantreg_max_iter,
                        gmm_ridge=gmm_ridge,
                        critical_value_multiplier=critical_value_multiplier,
                        grid_strategy=grid_strategy,
                        refinement_tolerance=refinement_tolerance,
                        max_refinement_depth=max_refinement_depth,
                        max_alpha_evaluations=max_alpha_evaluations,
                        iteration_warning_policy=iteration_warning_policy,
                        hard_failure_policy=hard_failure_policy,
                    )
                elif estimator_name == "post_selection":
                    result = estimate_post_selection_ivqr(
                        data,
                        tau=design.tau,
                        alphas=alphas,
                        selection_cv=3,
                        selection_max_iter=10000,
                        quantreg_max_iter=quantreg_max_iter,
                        selection_random_state=estimator_random_state,
                        selection_lasso_multiplier=selection_lasso_multiplier,
                        critical_value_multiplier=critical_value_multiplier,
                        grid_strategy=grid_strategy,
                        refinement_tolerance=refinement_tolerance,
                        max_refinement_depth=max_refinement_depth,
                        max_alpha_evaluations=max_alpha_evaluations,
                        iteration_warning_policy=iteration_warning_policy,
                        hard_failure_policy=hard_failure_policy,
                    )
                elif estimator_name == "dml":
                    quantile_solver = _validate_dml_quantile_solver(
                        dml_quantile_solver
                    )
                    result = estimate_dml_ivqr(
                        data,
                        tau=design.tau,
                        alphas=alphas,
                        k_folds=dml_k_folds,
                        fold_random_state=estimator_random_state,
                        quantile_penalty=dml_quantile_penalty,
                        ridge_alpha=dml_ridge_alpha,
                        quantile_solver=quantile_solver,
                        gmm_ridge=gmm_ridge,
                        critical_value_multiplier=critical_value_multiplier,
                    )
                else:
                    raise ValueError(f"Unknown estimator: {estimator_name}")
            rows.append(
                _result_to_row(
                    design, result, alphas, critical_value_multiplier
                )
            )
        except Exception as exc:  # noqa: BLE001 - record failed replications.
            rows.append(
                _failure_row_for_estimator(
                    design, estimator_name, alphas, exc, critical_value_multiplier
                )
            )
    return rows


def run_single_replication(*args, **kwargs) -> list[dict[str, object]]:
    """Backward-compatible alias for run_simulation_design."""
    return run_simulation_design(*args, **kwargs)


def _run_worker(args: WorkerArgs) -> list[dict[str, object]]:
    return run_simulation_design(
        args.design,
        args.alphas,
        estimators=args.estimators,
        quantreg_max_iter=args.quantreg_max_iter,
        dml_k_folds=args.dml_k_folds,
        dml_quantile_penalty=args.dml_quantile_penalty,
        dml_ridge_alpha=args.dml_ridge_alpha,
        dml_quantile_solver=args.dml_quantile_solver,
        gmm_ridge=args.gmm_ridge,
        critical_value_multiplier=args.critical_value_multiplier,
        selection_lasso_multiplier=args.selection_lasso_multiplier,
        show_quantreg_warnings=args.show_quantreg_warnings,
        grid_strategy=args.grid_strategy,
        refinement_tolerance=args.refinement_tolerance,
        max_refinement_depth=args.max_refinement_depth,
        max_alpha_evaluations=args.max_alpha_evaluations,
        iteration_warning_policy=args.iteration_warning_policy,
        hard_failure_policy=args.hard_failure_policy,
    )


def _row_sort_key(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row[column] for column in (*DESIGN_KEY_COLUMNS, "estimator"))


def run_simulation_batch(
    designs: list[Design],
    alphas: np.ndarray,
    estimators: tuple[str, ...] = DEFAULT_SIMULATION_ESTIMATORS,
    output_path: str | Path | None = None,
    append: bool = False,
    quantreg_max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    dml_k_folds: int = DEFAULT_DML_K_FOLDS,
    dml_quantile_penalty: float = DEFAULT_DML_QUANTILE_PENALTY,
    dml_ridge_alpha: float = DEFAULT_DML_RIDGE_ALPHA,
    dml_quantile_solver: str = DEFAULT_DML_QUANTILE_SOLVER,
    gmm_ridge: float = 1e-8,
    critical_value_multiplier: float = DEFAULT_CRITICAL_VALUE_MULTIPLIER,
    selection_lasso_multiplier: float = 1.0,
    n_jobs: int = DEFAULT_N_JOBS,
    show_quantreg_warnings: bool = False,
    grid_strategy: str = DEFAULT_GRID_STRATEGY,
    refinement_tolerance: float = DEFAULT_REFINEMENT_TOLERANCE,
    max_refinement_depth: int = DEFAULT_MAX_REFINEMENT_DEPTH,
    max_alpha_evaluations: int = DEFAULT_MAX_ALPHA_EVALUATIONS,
    iteration_warning_policy: str = DEFAULT_ITERATION_WARNING_POLICY,
    hard_failure_policy: str = DEFAULT_HARD_FAILURE_POLICY,
) -> pd.DataFrame:
    """Run a batch of designs and optionally persist the raw rows to CSV."""
    designs = [_validate_design(design) for design in designs]
    estimators = _validate_estimators(estimators)
    alphas = validate_alpha_grid(alphas)
    n_jobs = validate_positive_int("n_jobs", n_jobs)
    selection_lasso_multiplier = validate_selection_lasso_multiplier(
        selection_lasso_multiplier
    )
    dml_quantile_penalty = _validate_nonnegative_float(
        "dml_quantile_penalty", dml_quantile_penalty
    )
    dml_ridge_alpha = _validate_nonnegative_float("dml_ridge_alpha", dml_ridge_alpha)
    dml_quantile_solver = str(dml_quantile_solver)

    worker_args = [
        WorkerArgs(
            design=design,
            alphas=alphas,
            estimators=estimators,
            quantreg_max_iter=quantreg_max_iter,
            dml_k_folds=dml_k_folds,
            dml_quantile_penalty=dml_quantile_penalty,
            dml_ridge_alpha=dml_ridge_alpha,
            dml_quantile_solver=dml_quantile_solver,
            gmm_ridge=gmm_ridge,
            critical_value_multiplier=critical_value_multiplier,
            selection_lasso_multiplier=selection_lasso_multiplier,
            show_quantreg_warnings=show_quantreg_warnings,
            grid_strategy=grid_strategy,
            refinement_tolerance=refinement_tolerance,
            max_refinement_depth=max_refinement_depth,
            max_alpha_evaluations=max_alpha_evaluations,
            iteration_warning_policy=iteration_warning_policy,
            hard_failure_policy=hard_failure_policy,
        )
        for design in designs
    ]
    rows: list[dict[str, object]] = []
    if n_jobs == 1 or len(worker_args) <= 1:
        for args in worker_args:
            rows.extend(_run_worker(args))
    else:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(worker_args))) as executor:
            futures = {executor.submit(_run_worker, args): args for args in worker_args}
            for future in as_completed(futures):
                rows.extend(future.result())
        rows.sort(key=_row_sort_key)

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if estimators == ("dml",):
        results = clean_dml_results_frame(results)
    elif estimators == ("oracle",):
        results = clean_core_results_frame(results, estimator="oracle")
    elif estimators == ("post_selection",):
        results = clean_post_selection_results_frame(results)
    if output_path is not None:
        path = Path(output_path)
        if path.exists() and path.is_dir():
            raise ValueError("output_path must be a file path")
        if append and path.exists() and path.stat().st_size > 0:
            existing_columns = tuple(pd.read_csv(path, nrows=0).columns)
            if existing_columns != tuple(results.columns):
                raise ValueError(
                    "cannot append results with a different schema: "
                    f"existing file has {len(existing_columns)} columns, "
                    f"new rows have {len(results.columns)}"
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(
            path,
            mode="a" if append else "w",
            header=not (append and path.exists()),
            index=False,
        )
    return results


def _completed_successes(existing: pd.DataFrame, rerun_failed: bool) -> pd.DataFrame:
    if not rerun_failed or "failed" not in existing.columns:
        return existing
    failed = existing["failed"].astype(str).str.lower().isin({"true", "1", "yes"})
    return existing.loc[~failed]


def filter_completed_designs(
    designs: list[Design],
    results_path: str | Path,
    estimators: tuple[str, ...],
    rerun_failed: bool = False,
) -> list[Design]:
    """Return designs that do not yet have all requested estimator rows."""
    path = Path(results_path)
    if not path.exists():
        return designs
    required = list(DESIGN_KEY_COLUMNS) + ["estimator"]
    if rerun_failed:
        required.append("failed")
    existing = _completed_successes(pd.read_csv(path, usecols=required), rerun_failed)
    expected = {ESTIMATOR_OUTPUT_NAMES[name] for name in _validate_estimators(estimators)}
    completed: dict[tuple[object, ...], set[str]] = {}
    for _, row in existing.iterrows():
        completed.setdefault(_row_design_key(row), set()).add(str(row["estimator"]))
    return [
        design
        for design in designs
        if not expected.issubset(completed.get(_design_key(design), set()))
    ]


def run_small_simulation(
    dgp: str = "dgp1",
    n: int = 80,
    p: int = 5,
    pi: float = 1.0,
    tau: float = 0.5,
    reps: int = 1,
    base_seed: int = DEFAULT_BASE_SEED,
    alphas: np.ndarray | None = None,
    estimators: tuple[str, ...] = DEFAULT_SIMULATION_ESTIMATORS,
    alpha_grid_size: int = DEFAULT_ALPHA_GRID_SIZE,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    **kwargs,
) -> pd.DataFrame:
    """Run a small simulation and return raw estimator-level rows."""
    if alphas is None:
        alphas = np.linspace(alpha_min, alpha_max, alpha_grid_size)
    designs = make_simulation_grid(
        dgps=(dgp,),
        n_values=(n,),
        p_values=(p,),
        pi_values=(pi,),
        taus=(tau,),
        reps=reps,
        base_seed=base_seed,
    )
    return run_simulation_batch(
        designs,
        alphas,
        estimators=estimators,
        n_jobs=1,
        **kwargs,
    )
