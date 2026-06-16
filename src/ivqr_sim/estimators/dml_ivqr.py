"""DML-IVQR estimator."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ivqr_sim.data import SimData
from ivqr_sim.estimators.base import EstimationResult
from ivqr_sim.inference import (
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    sanitize_grid_statistics,
)
from ivqr_sim.moments import alpha_grid, quantile_score, weighted_gmm_statistic


def _validate_tau(tau: float) -> None:
    if not 0 < tau < 1:
        raise ValueError("tau must satisfy 0 < tau < 1")


def _validate_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_feature_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_data_arrays(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = _validate_vector(y, "y")
    d = _validate_vector(d, "d")
    z = _validate_vector(z, "z")
    x = _validate_feature_matrix(x, "x")

    if not (len(y) == len(d) == len(z) == x.shape[0]):
        raise ValueError("y, d, z, and x must have consistent row counts")

    return y, d, z, x


def _validate_outcome_control_arrays(
    y: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _validate_vector(y, "y")
    d = _validate_vector(d, "d")
    x = _validate_feature_matrix(x, "x")

    if not (len(y) == len(d) == x.shape[0]):
        raise ValueError("y, d, and x must have consistent row counts")

    return y, d, x


def _validate_alpha_candidates(alphas: np.ndarray) -> np.ndarray:
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1:
        raise ValueError("alphas must be one-dimensional")
    if alphas.size == 0:
        raise ValueError("alphas must be nonempty")
    if not np.all(np.isfinite(alphas)):
        raise ValueError("alphas must contain only finite values")
    if not np.all(np.diff(alphas) > 0):
        raise ValueError("alphas must be sorted strictly increasing")
    return alphas


def _failed_result(
    data: SimData,
    tau: float,
    message: str,
    runtime_seconds: float,
) -> EstimationResult:
    return EstimationResult(
        estimator="dml_ivqr",
        alpha_hat=None,
        alpha_true=data.alpha_true,
        tau=tau,
        converged=False,
        failed=True,
        message=message,
        objective_value=None,
        at_grid_boundary=False,
        cr_lower=None,
        cr_upper=None,
        cr_length=None,
        cr_covers_true=None,
        cr_empty=True,
        cr_disconnected=None,
        selected_controls=None,
        runtime_seconds=runtime_seconds,
    )


def make_folds(
    n: int,
    k_folds: int = 5,
    random_state: int | None = 123,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create shuffled K-fold train/test indices."""
    if n <= 1:
        raise ValueError("n must be greater than 1")
    if k_folds < 2 or k_folds > n:
        raise ValueError("k_folds must satisfy 2 <= k_folds <= n")

    splitter = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    folds = [(train_idx, test_idx) for train_idx, test_idx in splitter.split(np.arange(n))]

    test_counts = np.zeros(n, dtype=int)
    for train_idx, test_idx in folds:
        if np.intersect1d(train_idx, test_idx).size > 0:
            raise RuntimeError("train and test fold indices overlap")
        test_counts[test_idx] += 1
    if not np.all(test_counts == 1):
        raise RuntimeError("each observation must appear exactly once in test folds")

    return folds


def standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize features using training-fold moments only."""
    x_train = _validate_feature_matrix(x_train, "x_train")
    x_test = _validate_feature_matrix(x_test, "x_test")

    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("x_train and x_test must have the same number of columns")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler


def fit_quantile_nuisance(
    y_train: np.ndarray,
    d_train: np.ndarray,
    x_train: np.ndarray,
    alpha_value: float,
    tau: float,
    penalty: float = 0.01,
    solver: str = "highs",
) -> tuple[Any | None, bool, str]:
    """Fit penalized quantile nuisance for Y - D alpha on X."""
    _validate_tau(tau)
    y_train, d_train, x_train = _validate_outcome_control_arrays(y_train, d_train, x_train)

    if penalty < 0:
        raise ValueError("penalty must be nonnegative")

    y_tilde_train = y_train - d_train * alpha_value
    try:
        model = QuantileRegressor(
            quantile=tau,
            alpha=penalty,
            fit_intercept=True,
            solver=solver,
        )
        model.fit(x_train, y_tilde_train)
    except Exception as exc:  # noqa: BLE001 - solver failures should be reported.
        return None, False, str(exc)

    return model, True, "ok"


def fit_instrument_residualizer(
    z_train: np.ndarray,
    x_train: np.ndarray,
    ridge_alpha: float = 1.0,
) -> tuple[Any | None, bool, str]:
    """Fit Ridge residualizer for Z on X."""
    z_train = _validate_vector(z_train, "z_train")
    x_train = _validate_feature_matrix(x_train, "x_train")

    if len(z_train) != x_train.shape[0]:
        raise ValueError("z_train and x_train must have consistent row counts")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be nonnegative")

    try:
        model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        model.fit(x_train, z_train)
    except Exception as exc:  # noqa: BLE001 - nuisance failures should be reported.
        return None, False, str(exc)

    return model, True, "ok"


def evaluate_dml_ivqr_alpha(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
    alpha_value: float,
    tau: float,
    k_folds: int = 5,
    fold_random_state: int | None = 123,
    quantile_penalty: float = 0.01,
    ridge_alpha: float = 1.0,
    quantile_solver: str = "highs",
    gmm_ridge: float = 1e-8,
) -> tuple[float, bool, str]:
    """Evaluate the weighted cross-fitted scalar DML-IVQR statistic."""
    y, d, z, x = _validate_data_arrays(y, d, z, x)
    _validate_tau(tau)

    n = len(y)
    folds = make_folds(n=n, k_folds=k_folds, random_state=fold_random_state)
    moment_contributions = np.empty(n, dtype=float)

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        x_train_scaled, x_test_scaled, _ = standardize_train_test(
            x[train_idx],
            x[test_idx],
        )

        model_beta, beta_converged, beta_message = fit_quantile_nuisance(
            y_train=y[train_idx],
            d_train=d[train_idx],
            x_train=x_train_scaled,
            alpha_value=alpha_value,
            tau=tau,
            penalty=quantile_penalty,
            solver=quantile_solver,
        )
        if not beta_converged or model_beta is None:
            return np.inf, False, f"Quantile nuisance failed in fold {fold_id}: {beta_message}"

        model_delta, delta_converged, delta_message = fit_instrument_residualizer(
            z_train=z[train_idx],
            x_train=x_train_scaled,
            ridge_alpha=ridge_alpha,
        )
        if not delta_converged or model_delta is None:
            return np.inf, False, f"Instrument residualizer failed in fold {fold_id}: {delta_message}"

        q_hat_test = model_beta.predict(x_test_scaled)
        z_resid_test = z[test_idx] - model_delta.predict(x_test_scaled)
        residual_test = y[test_idx] - d[test_idx] * alpha_value - q_hat_test
        score_test = quantile_score(residual_test, tau)
        moment_contributions[test_idx] = score_test * z_resid_test

    contributions = moment_contributions.reshape(-1, 1)
    statistic = weighted_gmm_statistic(contributions, ridge=gmm_ridge)
    return float(statistic), True, "ok"


def estimate_dml_ivqr(
    data: SimData,
    tau: float,
    alphas: np.ndarray | None = None,
    alpha_min: float = -2.0,
    alpha_max: float = 4.0,
    alpha_step: float = 0.05,
    confidence_level: float = 0.95,
    k_folds: int = 5,
    fold_random_state: int | None = 123,
    quantile_penalty: float = 0.01,
    ridge_alpha: float = 1.0,
    quantile_solver: str = "highs",
    gmm_ridge: float = 1e-8,
) -> EstimationResult:
    """Estimate DML-IVQR by cross-fitted weighted score inversion."""
    start = perf_counter()
    _validate_tau(tau)
    y, d, z, x = _validate_data_arrays(data.y, data.d, data.z, data.x)
    make_folds(n=len(y), k_folds=k_folds, random_state=fold_random_state)

    if alphas is None:
        alphas = alpha_grid(alpha_min, alpha_max, alpha_step)
    else:
        alphas = _validate_alpha_candidates(alphas)

    statistics = np.empty(len(alphas), dtype=float)
    converged_flags: list[bool] = []

    for j, alpha_value in enumerate(alphas):
        try:
            statistic, converged, message = evaluate_dml_ivqr_alpha(
                y=y,
                d=d,
                z=z,
                x=x,
                alpha_value=float(alpha_value),
                tau=tau,
                k_folds=k_folds,
                fold_random_state=fold_random_state,
                quantile_penalty=quantile_penalty,
                ridge_alpha=ridge_alpha,
                quantile_solver=quantile_solver,
                gmm_ridge=gmm_ridge,
            )
        except Exception as exc:  # noqa: BLE001 - failed grid points are recorded.
            statistic, converged, message = np.inf, False, str(exc)
        statistics[j] = statistic
        converged_flags.append(converged)

    statistics, num_failed = sanitize_grid_statistics(statistics, converged_flags)
    if num_failed == len(alphas):
        return _failed_result(
            data=data,
            tau=tau,
            message="All alpha-grid evaluations failed.",
            runtime_seconds=perf_counter() - start,
        )

    alpha_hat, min_statistic, at_boundary = argmin_grid(alphas, statistics)
    critical = critical_value_chi_square(confidence_level, df=1)
    region = invert_score_test(
        alphas=alphas,
        statistics=statistics,
        critical_value=critical,
        alpha_true=data.alpha_true,
    )

    all_converged = num_failed == 0
    return EstimationResult(
        estimator="dml_ivqr",
        alpha_hat=alpha_hat,
        alpha_true=data.alpha_true,
        tau=tau,
        converged=all_converged,
        failed=False,
        message=f"ok; failed_alpha_points={num_failed}/{len(alphas)}",
        objective_value=min_statistic,
        at_grid_boundary=at_boundary,
        cr_lower=region.lower,
        cr_upper=region.upper,
        cr_length=region.length,
        cr_covers_true=region.covers_true,
        cr_empty=region.empty,
        cr_disconnected=region.disconnected,
        selected_controls=None,
        runtime_seconds=perf_counter() - start,
    )
