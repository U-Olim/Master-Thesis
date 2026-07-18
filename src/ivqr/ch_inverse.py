"""Common Chernozhukov-Hansen inverse-IVQR grid routines.

This module contains low-dimensional CH-IVQR machinery. The same inversion
logic is shared by the Oracle and post-selection IVQR estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, cast
import warnings

import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from dgp.designs import SimData
from estimators.base import EstimationResult, estimation_result_diagnostic_kwargs
from ivqr.alpha_grid import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_ALPHA_STEP,
    alpha_grid,
)
from ivqr.confidence_regions import (
    adjust_critical_value,
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    merge_region_and_grid_diagnostics,
    sanitize_grid_statistics,
    summarize_alpha_grid_diagnostics,
    validate_critical_value_multiplier,
)
from simulation.config import DEFAULT_QUANTREG_MAX_ITER
from utils.timing import RuntimeDiagnosticColumns, estimator_runtime_columns
from utils.validation import (
    validate_1d_array,
    validate_2d_array,
    validate_alpha_grid,
    validate_data_arrays,
    validate_tau,
)


IterationWarningPolicy = Literal["reject", "use_if_valid"]
ITERATION_WARNING_POLICIES: tuple[IterationWarningPolicy, ...] = (
    "use_if_valid",
    "reject",
)
_COVARIANCE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class AlphaEvaluation:
    """Evaluation of one structural alpha candidate in inverse IVQR.

    ``converged`` records optimizer convergence, while ``usable`` records the
    separate statistical judgment that the returned Wald statistic is valid
    for inference.  An iteration-limit fit can therefore be usable without
    being classified as converged.
    """

    statistic: float
    gamma_hat: np.ndarray
    cov_gamma: np.ndarray
    dim_z: int
    converged: bool
    usable: bool
    warning_type: str | None
    failure_reason: str | None
    message: str
    covariance_rank: int | None = None
    covariance_condition_number: float | None = None


def validate_iteration_warning_policy(
    policy: IterationWarningPolicy | str,
) -> IterationWarningPolicy:
    """Validate and return the QuantReg iteration-warning policy."""
    if policy not in ITERATION_WARNING_POLICIES:
        choices = ", ".join(ITERATION_WARNING_POLICIES)
        raise ValueError(f"iteration_warning_policy must be one of: {choices}")
    return cast(IterationWarningPolicy, policy)


def _failed_alpha_evaluation(
    *,
    dim_z: int,
    converged: bool,
    warning_type: str | None,
    failure_reason: str,
    gamma_hat: np.ndarray | None = None,
    cov_gamma: np.ndarray | None = None,
    covariance_rank: int | None = None,
    covariance_condition_number: float | None = None,
) -> AlphaEvaluation:
    return AlphaEvaluation(
        statistic=np.inf,
        gamma_hat=(
            np.full(dim_z, np.nan)
            if gamma_hat is None
            else np.asarray(gamma_hat, dtype=float)
        ),
        cov_gamma=(
            np.full((dim_z, dim_z), np.nan)
            if cov_gamma is None
            else np.atleast_2d(np.asarray(cov_gamma, dtype=float))
        ),
        dim_z=dim_z,
        converged=converged,
        usable=False,
        warning_type=warning_type,
        failure_reason=failure_reason,
        message=failure_reason,
        covariance_rank=covariance_rank,
        covariance_condition_number=covariance_condition_number,
    )


def add_intercept(x: np.ndarray) -> np.ndarray:
    """Return a design matrix with a leading intercept column."""
    x = validate_2d_array("x", x)
    if x.shape[0] == 0:
        raise ValueError("x must contain at least one row")
    return np.column_stack([np.ones(x.shape[0]), x])


def as_2d_instruments(z: np.ndarray) -> np.ndarray:
    """Validate excluded instruments and return a two-dimensional array."""
    z_array = np.asarray(z, dtype=float)
    if z_array.ndim == 1:
        z_array = z_array.reshape(-1, 1)
    if z_array.ndim != 2:
        raise ValueError("z must be one- or two-dimensional")
    if z_array.shape[0] == 0:
        raise ValueError("z must contain at least one row")
    if z_array.shape[1] == 0:
        raise ValueError("z must contain at least one excluded instrument")
    if not np.all(np.isfinite(z_array)):
        raise ValueError("z must contain only finite values")
    return z_array


def ch_ivqr_design(x_controls: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, slice]:
    """Build the profiled QR design [1, selected controls, excluded instruments]."""
    x_controls = validate_2d_array("x_controls", x_controls)
    z_2d = as_2d_instruments(z)
    if x_controls.shape[0] != z_2d.shape[0]:
        raise ValueError("x_controls and z must have the same number of rows")

    design = np.column_stack([np.ones(x_controls.shape[0]), x_controls, z_2d])
    z_start = 1 + x_controls.shape[1]
    z_stop = z_start + z_2d.shape[1]
    return design, slice(z_start, z_stop)


def wald_statistic(gamma_hat: np.ndarray, cov_gamma: np.ndarray) -> float:
    """Return the Wald statistic for the excluded-instrument coefficients.

    CH writes the statistic as ``W_n(a) = n * gamma_hat' A_hat gamma_hat``.
    Statsmodels ``cov_params()`` estimates ``Var(gamma_hat)``, which already
    contains the inverse sample-size scaling. Therefore
    ``gamma_hat' Var(gamma_hat)^(-1) gamma_hat`` is the CH Wald statistic;
    multiplying by ``n`` again would double-count the sample-size scaling.
    """
    gamma_hat = np.asarray(gamma_hat, dtype=float).reshape(-1)
    cov_gamma = np.atleast_2d(np.asarray(cov_gamma, dtype=float))
    if cov_gamma.shape != (gamma_hat.size, gamma_hat.size):
        raise ValueError("cov_gamma shape must match gamma_hat dimension")
    if not np.all(np.isfinite(gamma_hat)) or not np.all(np.isfinite(cov_gamma)):
        raise ValueError("gamma_hat and cov_gamma must be finite")

    statistic = float(gamma_hat @ np.linalg.pinv(cov_gamma) @ gamma_hat)
    if statistic < 0.0 and statistic > -1e-10:
        statistic = 0.0
    if not np.isfinite(statistic) or statistic < 0.0:
        raise ValueError("Wald statistic must be finite and nonnegative")
    return statistic


def evaluate_alpha_ch_ivqr(
    *,
    y: np.ndarray,
    d: np.ndarray,
    x_controls: np.ndarray,
    z: np.ndarray,
    alpha: float,
    tau: float,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    iteration_warning_policy: IterationWarningPolicy = "use_if_valid",
) -> AlphaEvaluation:
    """Evaluate a CH inverse-IVQR statistic at one structural alpha.

    For a candidate alpha, run the tau-quantile regression of Y - D*alpha on
    included controls and excluded instruments. At the true alpha, the IVQR
    restriction implies that the excluded-instrument coefficient should be zero.

    The production policy ``"use_if_valid"`` retains an iteration-limit fit
    only when all inferential validity checks pass.  Pass ``"reject"`` to
    reproduce the legacy behavior that rejects every iteration-limit fit.
    """
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    iteration_warning_policy = validate_iteration_warning_policy(
        iteration_warning_policy
    )
    validate_tau(tau)
    alpha = float(alpha)
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")
    y = validate_1d_array("y", y)
    d = validate_1d_array("d", d)
    x_controls = validate_2d_array("x_controls", x_controls)
    z_2d = as_2d_instruments(z)
    if not (len(y) == len(d) == x_controls.shape[0] == z_2d.shape[0]):
        raise ValueError("y, d, x_controls, and z must have consistent row counts")

    design, z_block = ch_ivqr_design(x_controls, z_2d)
    y_alpha = y - d * alpha
    dim_z = z_2d.shape[1]

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IterationLimitWarning)
            result = QuantReg(y_alpha, design).fit(q=tau, max_iter=max_iter)
        iteration_limit_warning = any(
            issubclass(warning.category, IterationLimitWarning) for warning in caught
        )
        if iteration_limit_warning and iteration_warning_policy == "reject":
            return _failed_alpha_evaluation(
                dim_z=dim_z,
                converged=False,
                warning_type="iteration_limit",
                failure_reason="QuantReg reached iteration limit",
            )
    except Exception as exc:  # noqa: BLE001 - QuantReg can fail in several ways.
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=False,
            warning_type=None,
            failure_reason=f"quantreg_fit_failed: {type(exc).__name__}: {exc}",
        )

    warning_type = "iteration_limit" if iteration_limit_warning else None
    converged = not iteration_limit_warning
    try:
        params = np.asarray(result.params, dtype=float)
    except Exception as exc:  # noqa: BLE001 - malformed result objects vary.
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason=f"params_unavailable: {type(exc).__name__}: {exc}",
        )
    if params.shape != (design.shape[1],):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="invalid_parameter_dimension",
        )
    if not np.all(np.isfinite(params)):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="nonfinite_parameters",
        )
    try:
        cov_params = np.asarray(result.cov_params(), dtype=float)
    except Exception as exc:  # noqa: BLE001 - covariance failures vary.
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason=f"covariance_unavailable: {type(exc).__name__}: {exc}",
        )
    if cov_params.shape != (design.shape[1], design.shape[1]):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="invalid_covariance_dimension",
        )
    if not np.all(np.isfinite(cov_params)):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="nonfinite_covariance",
        )

    gamma_hat = params[z_block]
    cov_gamma = cov_params[z_block, z_block]
    if gamma_hat.shape != (dim_z,):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="invalid_instrument_coefficient_dimension",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
        )
    if cov_gamma.shape != (dim_z, dim_z):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="invalid_instrument_covariance_dimension",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
        )
    try:
        covariance_rank = int(np.linalg.matrix_rank(cov_gamma))
        covariance_condition_number = float(np.linalg.cond(cov_gamma))
    except np.linalg.LinAlgError as exc:
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason=f"invalid_instrument_covariance: {exc}",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
        )
    if dim_z > 1 and covariance_rank == 0:
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="zero_instrument_covariance",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
            covariance_rank=covariance_rank,
            covariance_condition_number=covariance_condition_number,
        )
    if not np.allclose(cov_gamma, cov_gamma.T, rtol=1e-8, atol=_COVARIANCE_TOLERANCE):
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="nonsymmetric_instrument_covariance",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
            covariance_rank=covariance_rank,
            covariance_condition_number=covariance_condition_number,
        )
    try:
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(cov_gamma)))
    except np.linalg.LinAlgError as exc:
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason=f"invalid_instrument_covariance: {exc}",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
            covariance_rank=covariance_rank,
            covariance_condition_number=covariance_condition_number,
        )
    if minimum_eigenvalue < -_COVARIANCE_TOLERANCE:
        reason = (
            "negative_instrument_variance"
            if dim_z == 1
            else "indefinite_instrument_covariance"
        )
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason=reason,
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
            covariance_rank=covariance_rank,
            covariance_condition_number=covariance_condition_number,
        )
    if dim_z == 1 and float(cov_gamma[0, 0]) <= _COVARIANCE_TOLERANCE:
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason="zero_instrument_variance",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
            covariance_rank=covariance_rank,
            covariance_condition_number=covariance_condition_number,
        )
    try:
        statistic = wald_statistic(gamma_hat, cov_gamma)
    except ValueError as exc:
        return _failed_alpha_evaluation(
            dim_z=dim_z,
            converged=converged,
            warning_type=warning_type,
            failure_reason=f"invalid_wald_statistic: {exc}",
            gamma_hat=gamma_hat,
            cov_gamma=cov_gamma,
            covariance_rank=covariance_rank,
            covariance_condition_number=covariance_condition_number,
        )

    return AlphaEvaluation(
        statistic=statistic,
        gamma_hat=np.asarray(gamma_hat, dtype=float),
        cov_gamma=np.atleast_2d(np.asarray(cov_gamma, dtype=float)),
        dim_z=dim_z,
        converged=converged,
        usable=True,
        warning_type=warning_type,
        failure_reason=None,
        message="ok" if converged else "usable despite QuantReg iteration limit",
        covariance_rank=covariance_rank,
        covariance_condition_number=covariance_condition_number,
    )


def failed_ch_ivqr_result(
    *,
    data: SimData,
    tau: float,
    estimator: str,
    message: str,
    runtime_seconds: float,
    alpha_grid_size: int | None,
    failed_alpha_count: int | None,
    selected_controls: int | None = None,
    runtime_diagnostics: RuntimeDiagnosticColumns | None = None,
) -> EstimationResult:
    """Create a standard failed result for any CH-IVQR-style estimator."""
    if not np.isfinite(runtime_seconds) or runtime_seconds < 0:
        raise ValueError("runtime_seconds must be finite and nonnegative")
    if alpha_grid_size is not None and alpha_grid_size < 1:
        raise ValueError("alpha_grid_size must be at least 1 when provided")
    if failed_alpha_count is not None and failed_alpha_count < 0:
        raise ValueError("failed_alpha_count must be nonnegative when provided")
    if (
        alpha_grid_size is not None
        and failed_alpha_count is not None
        and failed_alpha_count > alpha_grid_size
    ):
        raise ValueError("failed_alpha_count must not exceed alpha_grid_size")
    return EstimationResult(
        estimator=estimator,
        alpha_hat=None,
        alpha_true=data.alpha_true,
        tau=tau,
        converged=False,
        failed=True,
        message=message,
        objective_value=None,
        at_grid_boundary=False,
        alpha_grid_size=alpha_grid_size,
        failed_alpha_count=failed_alpha_count,
        cr_lower=None,
        cr_upper=None,
        cr_length=None,
        cr_covers_true=None,
        cr_empty=True,
        cr_disconnected=None,
        selected_controls=selected_controls,
        runtime_seconds=runtime_seconds,
        **(
            estimator_runtime_columns(estimator=estimator, total_sec=runtime_seconds)
            if runtime_diagnostics is None
            else runtime_diagnostics
        ),
    )


def estimate_ch_ivqr_controls(
    data: SimData,
    tau: float,
    x_controls: np.ndarray,
    estimator_name: str,
    alphas: np.ndarray | None = None,
    alpha_min: float = DEFAULT_ALPHA_MIN,
    alpha_max: float = DEFAULT_ALPHA_MAX,
    alpha_step: float = DEFAULT_ALPHA_STEP,
    confidence_level: float = 0.95,
    critical_value_multiplier: float = 1.0,
    max_iter: int = DEFAULT_QUANTREG_MAX_ITER,
    iteration_warning_policy: IterationWarningPolicy = "use_if_valid",
    selected_controls: int | None = None,
) -> EstimationResult:
    """Estimate alpha by CH inverse-IVQR using a supplied control matrix.

    Valid iteration-limit fits are used in normal production operation.  Set
    ``iteration_warning_policy="reject"`` for legacy-result reproducibility.
    """
    start = perf_counter()
    alpha_loop_sec = float("nan")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    iteration_warning_policy = validate_iteration_warning_policy(
        iteration_warning_policy
    )
    critical_value_multiplier = validate_critical_value_multiplier(
        critical_value_multiplier
    )
    validate_tau(tau)
    y, d, z, original_x = validate_data_arrays(data.y, data.d, data.x, data.z)
    # original_x is validated only to ensure the SimData object is well-formed.
    # CH inverse-IVQR estimation uses the supplied x_controls matrix.
    _ = original_x
    x_controls = validate_2d_array("x_controls", x_controls)
    if x_controls.shape[0] != len(y):
        raise ValueError("x_controls must have the same number of rows as y")

    z_2d = as_2d_instruments(z)
    n_regressors = 1 + x_controls.shape[1] + z_2d.shape[1]
    if n_regressors >= len(y):
        return failed_ch_ivqr_result(
            data=data,
            tau=tau,
            estimator=estimator_name,
            message=(
                "CH-IVQR infeasible: intercept + controls + instruments must be "
                f"less than n. n={len(y)}, regressors={n_regressors}."
            ),
            runtime_seconds=perf_counter() - start,
            alpha_grid_size=None,
            failed_alpha_count=None,
            selected_controls=selected_controls,
        )

    if alphas is None:
        alphas = alpha_grid(alpha_min, alpha_max, alpha_step)
    else:
        alphas = validate_alpha_grid(alphas)

    statistics = np.empty(len(alphas), dtype=float)
    usable_flags: list[bool] = []
    alpha_loop_start = perf_counter()
    for j, alpha in enumerate(alphas):
        evaluation = evaluate_alpha_ch_ivqr(
            y=y,
            d=d,
            x_controls=x_controls,
            z=z_2d,
            alpha=float(alpha),
            tau=tau,
            max_iter=max_iter,
            iteration_warning_policy=iteration_warning_policy,
        )
        statistics[j] = evaluation.statistic
        usable_flags.append(evaluation.usable)
    alpha_loop_sec = perf_counter() - alpha_loop_start

    statistics, num_failed = sanitize_grid_statistics(statistics, usable_flags)
    if num_failed == len(alphas):
        runtime_seconds = perf_counter() - start
        return failed_ch_ivqr_result(
            data=data,
            tau=tau,
            estimator=estimator_name,
            message=(
                "All alpha-grid evaluations failed; "
                f"failed_alpha_points={num_failed}/{len(alphas)}"
            ),
            runtime_seconds=runtime_seconds,
            alpha_grid_size=len(alphas),
            failed_alpha_count=num_failed,
            selected_controls=selected_controls,
            runtime_diagnostics=estimator_runtime_columns(
                estimator=estimator_name,
                total_sec=runtime_seconds,
                alpha_loop_sec=alpha_loop_sec,
            ),
        )

    confidence_region_start = perf_counter()
    alpha_hat, min_statistic, at_boundary = argmin_grid(alphas, statistics)
    critical = critical_value_chi_square(confidence_level, df=z_2d.shape[1])
    adjusted_critical = adjust_critical_value(critical, critical_value_multiplier)
    accepted_mask = statistics <= adjusted_critical
    diagnostics = summarize_alpha_grid_diagnostics(
        alpha_grid=alphas,
        accepted_mask=accepted_mask,
        alpha_hat=alpha_hat,
        failed_alpha_count=num_failed,
        test_stats=statistics,
        critical_value=adjusted_critical,
        critical_value_nominal=critical,
        critical_value_multiplier=critical_value_multiplier,
        critical_value_adjusted=adjusted_critical,
    )
    region = invert_score_test(
        alphas=alphas,
        statistics=statistics,
        critical_value=critical,
        critical_value_multiplier=critical_value_multiplier,
        alpha_true=data.alpha_true,
        statistic_reference=0.0,
        inversion_type="absolute",
    )
    diagnostics = merge_region_and_grid_diagnostics(region, diagnostics)
    confidence_region_sec = perf_counter() - confidence_region_start
    runtime_seconds = perf_counter() - start

    return EstimationResult(
        estimator=estimator_name,
        alpha_hat=alpha_hat,
        alpha_true=data.alpha_true,
        tau=tau,
        converged=True,
        failed=False,
        message=f"ok; failed_alpha_points={num_failed}/{len(alphas)}",
        objective_value=min_statistic,
        at_grid_boundary=at_boundary,
        alpha_grid_size=len(alphas),
        failed_alpha_count=num_failed,
        cr_lower=diagnostics["cr_lower"],
        cr_upper=diagnostics["cr_upper"],
        cr_length=diagnostics["cr_length"],
        cr_covers_true=region.covers_true,
        cr_empty=diagnostics["cr_empty"],
        cr_disconnected=diagnostics["cr_disconnected"],
        selected_controls=selected_controls,
        runtime_seconds=runtime_seconds,
        **estimator_runtime_columns(
            estimator=estimator_name,
            total_sec=runtime_seconds,
            alpha_loop_sec=alpha_loop_sec,
            confidence_region_sec=confidence_region_sec,
        ),
        **estimation_result_diagnostic_kwargs(diagnostics),
    )


__all__ = [
    "AlphaEvaluation",
    "IterationWarningPolicy",
    "ITERATION_WARNING_POLICIES",
    "add_intercept",
    "as_2d_instruments",
    "ch_ivqr_design",
    "wald_statistic",
    "evaluate_alpha_ch_ivqr",
    "failed_ch_ivqr_result",
    "estimate_ch_ivqr_controls",
    "validate_iteration_warning_policy",
]
