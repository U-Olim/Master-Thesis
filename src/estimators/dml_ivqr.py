"""DML-IVQR estimator."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from dgp.designs import SimData
from estimators.base import EstimationResult
from inference.confidence_regions import (
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    sanitize_grid_statistics,
)
from inference.alpha_grid import alpha_grid
from inference.moments import quantile_score, weighted_gmm_statistic
from utils.validation import (
    validate_1d_array,
    validate_2d_array,
    validate_alpha_grid,
    validate_data_arrays,
    validate_k_folds,
    validate_tau,
)


QuantileSolver = Literal[
    "highs-ds",
    "highs-ipm",
    "highs",
    "interior-point",
    "revised simplex",
    "warn",
]


@dataclass(frozen=True)
class DMLFoldCache:
    """Alpha-independent fold state reused across DML-IVQR grid evaluations."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    x_train_scaled: np.ndarray
    x_test_scaled: np.ndarray
    z_resid_test: np.ndarray


def _failed_result(
    data: SimData,
    tau: float,
    message: str,
    runtime_seconds: float,
    alpha_grid_size: int | None = None,
    failed_alpha_count: int | None = None,
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
        alpha_grid_size=alpha_grid_size,
        failed_alpha_count=failed_alpha_count,
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
    validate_k_folds(k_folds, n)

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
    x_train = validate_2d_array("x_train", x_train)
    x_test = validate_2d_array("x_test", x_test)

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
    solver: QuantileSolver = "highs",
) -> tuple[Any | None, bool, str]:
    """Fit penalized quantile nuisance for Y - D alpha on X."""
    validate_tau(tau)
    y_train, d_train, x_train = validate_data_arrays(y_train, d_train, x_train)

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
    z_train = validate_1d_array("z_train", z_train)
    x_train = validate_2d_array("x_train", x_train)

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


def _build_dml_fold_cache(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    x: np.ndarray,
    *,
    k_folds: int,
    random_state: int | None,
    ridge_alpha: float = 1.0,
) -> list[DMLFoldCache]:
    """Precompute fold objects that do not depend on the alpha candidate.

    Fold splits, scaling, and Z residualization are independent of the
    structural alpha candidate. They are cached once and reused across the
    alpha grid. Outcome nuisance fits are still recomputed for each alpha
    because y - d*alpha depends on alpha.
    """
    y, d, z, x = validate_data_arrays(y, d, x, z)
    folds = make_folds(n=len(y), k_folds=k_folds, random_state=random_state)

    cache: list[DMLFoldCache] = []
    for fold_id, (train_idx, test_idx) in enumerate(folds):
        x_train_scaled, x_test_scaled, _ = standardize_train_test(
            x[train_idx],
            x[test_idx],
        )

        model_delta, delta_converged, delta_message = fit_instrument_residualizer(
            z_train=z[train_idx],
            x_train=x_train_scaled,
            ridge_alpha=ridge_alpha,
        )
        if not delta_converged or model_delta is None:
            raise RuntimeError(
                f"Instrument residualizer failed in fold {fold_id}: {delta_message}"
            )

        cache.append(
            DMLFoldCache(
                train_idx=train_idx,
                test_idx=test_idx,
                x_train_scaled=x_train_scaled,
                x_test_scaled=x_test_scaled,
                z_resid_test=z[test_idx] - model_delta.predict(x_test_scaled),
            )
        )

    return cache


def _evaluate_dml_ivqr_alpha_with_cache(
    y: np.ndarray,
    d: np.ndarray,
    fold_cache: list[DMLFoldCache],
    alpha_value: float,
    tau: float,
    quantile_penalty: float = 0.01,
    quantile_solver: QuantileSolver = "highs",
    gmm_ridge: float = 1e-8,
) -> tuple[float, bool, str]:
    """Evaluate the weighted cross-fitted scalar statistic using cached folds."""
    y = validate_1d_array("y", y)
    d = validate_1d_array("d", d)
    if len(y) != len(d):
        raise ValueError("y and d must have consistent lengths")
    validate_tau(tau)

    n = len(y)
    moment_contributions = np.empty(n, dtype=float)

    for fold_id, fold in enumerate(fold_cache):
        model_beta, beta_converged, beta_message = fit_quantile_nuisance(
            y_train=y[fold.train_idx],
            d_train=d[fold.train_idx],
            x_train=fold.x_train_scaled,
            alpha_value=alpha_value,
            tau=tau,
            penalty=quantile_penalty,
            solver=quantile_solver,
        )
        if not beta_converged or model_beta is None:
            return np.inf, False, f"Quantile nuisance failed in fold {fold_id}: {beta_message}"

        q_hat_test = model_beta.predict(fold.x_test_scaled)
        residual_test = y[fold.test_idx] - d[fold.test_idx] * alpha_value - q_hat_test
        score_test = quantile_score(residual_test, tau)
        moment_contributions[fold.test_idx] = score_test * fold.z_resid_test

    contributions = moment_contributions.reshape(-1, 1)
    statistic = weighted_gmm_statistic(contributions, ridge=gmm_ridge)
    return float(statistic), True, "ok"


def _evaluate_dml_ivqr_alpha_uncached(
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
    quantile_solver: QuantileSolver = "highs",
    gmm_ridge: float = 1e-8,
) -> tuple[float, bool, str]:
    """Evaluate DML-IVQR alpha with the original uncached fold setup."""
    y, d, z, x = validate_data_arrays(y, d, x, z)
    validate_tau(tau)

    try:
        fold_cache = _build_dml_fold_cache(
            y,
            d,
            z,
            x,
            k_folds=k_folds,
            random_state=fold_random_state,
            ridge_alpha=ridge_alpha,
        )
    except RuntimeError as exc:
        return np.inf, False, str(exc)

    return _evaluate_dml_ivqr_alpha_with_cache(
        y=y,
        d=d,
        fold_cache=fold_cache,
        alpha_value=alpha_value,
        tau=tau,
        quantile_penalty=quantile_penalty,
        quantile_solver=quantile_solver,
        gmm_ridge=gmm_ridge,
    )


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
    quantile_solver: QuantileSolver = "highs",
    gmm_ridge: float = 1e-8,
    use_cache: bool = True,
) -> tuple[float, bool, str]:
    """Evaluate the weighted cross-fitted scalar DML-IVQR statistic."""
    if not use_cache:
        return _evaluate_dml_ivqr_alpha_uncached(
            y=y,
            d=d,
            z=z,
            x=x,
            alpha_value=alpha_value,
            tau=tau,
            k_folds=k_folds,
            fold_random_state=fold_random_state,
            quantile_penalty=quantile_penalty,
            ridge_alpha=ridge_alpha,
            quantile_solver=quantile_solver,
            gmm_ridge=gmm_ridge,
        )

    try:
        fold_cache = _build_dml_fold_cache(
            y,
            d,
            z,
            x,
            k_folds=k_folds,
            random_state=fold_random_state,
            ridge_alpha=ridge_alpha,
        )
    except RuntimeError as exc:
        return np.inf, False, str(exc)

    return _evaluate_dml_ivqr_alpha_with_cache(
        y=y,
        d=d,
        fold_cache=fold_cache,
        alpha_value=alpha_value,
        tau=tau,
        quantile_penalty=quantile_penalty,
        quantile_solver=quantile_solver,
        gmm_ridge=gmm_ridge,
    )


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
    quantile_solver: QuantileSolver = "highs",
    gmm_ridge: float = 1e-8,
    use_cache: bool = True,
) -> EstimationResult:
    """Estimate DML-IVQR by cross-fitted weighted score inversion.

    use_cache controls whether alpha-independent fold computations are reused
    across alpha values. Setting it to False is mainly for regression tests.
    """
    start = perf_counter()
    validate_tau(tau)
    y, d, z, x = validate_data_arrays(data.y, data.d, data.x, data.z)

    if alphas is None:
        alphas = alpha_grid(alpha_min, alpha_max, alpha_step)
    else:
        alphas = validate_alpha_grid(alphas)

    fold_cache: list[DMLFoldCache] | None = None
    if use_cache:
        try:
            fold_cache = _build_dml_fold_cache(
                y,
                d,
                z,
                x,
                k_folds=k_folds,
                random_state=fold_random_state,
                ridge_alpha=ridge_alpha,
            )
        except RuntimeError:
            fold_cache = None
    else:
        make_folds(n=len(y), k_folds=k_folds, random_state=fold_random_state)

    statistics = np.empty(len(alphas), dtype=float)
    converged_flags: list[bool] = []

    for j, alpha_value in enumerate(alphas):
        try:
            if fold_cache is not None:
                statistic, converged, message = _evaluate_dml_ivqr_alpha_with_cache(
                    y=y,
                    d=d,
                    fold_cache=fold_cache,
                    alpha_value=float(alpha_value),
                    tau=tau,
                    quantile_penalty=quantile_penalty,
                    quantile_solver=quantile_solver,
                    gmm_ridge=gmm_ridge,
                )
            else:
                statistic, converged, message = _evaluate_dml_ivqr_alpha_uncached(
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
        except Exception:  # noqa: BLE001 - failed grid points are recorded.
            statistic, converged = np.inf, False
        statistics[j] = statistic
        converged_flags.append(converged)

    statistics, num_failed = sanitize_grid_statistics(statistics, converged_flags)
    if num_failed == len(alphas):
        return _failed_result(
            data=data,
            tau=tau,
            message=(
                "All alpha-grid evaluations failed; "
                f"failed_alpha_points={num_failed}/{len(alphas)}"
            ),
            runtime_seconds=perf_counter() - start,
            alpha_grid_size=len(alphas),
            failed_alpha_count=num_failed,
        )

    alpha_hat, min_statistic, at_boundary = argmin_grid(alphas, statistics)
    critical = critical_value_chi_square(confidence_level, df=1)
    region = invert_score_test(
        alphas=alphas,
        statistics=statistics,
        critical_value=critical,
        alpha_true=data.alpha_true,
    )

    return EstimationResult(
        estimator="dml_ivqr",
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
        cr_lower=region.lower,
        cr_upper=region.upper,
        cr_length=region.length,
        cr_covers_true=region.covers_true,
        cr_empty=region.empty,
        cr_disconnected=region.disconnected,
        selected_controls=None,
        runtime_seconds=perf_counter() - start,
    )
