# Consolidated tests for the thematic project structure.

import numpy as np
import pytest
import warnings
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from dgp import generate_data
from dgp.designs import Design
from estimators import ch_inverse_ivqr
from estimators.base import EstimationResult
from estimators.dml_ivqr import (
    _build_dml_fold_cache,
    _evaluate_dml_ivqr_alpha_uncached,
    estimate_dml_ivqr,
    evaluate_dml_ivqr_alpha,
    fit_instrument_residualizer,
    fit_quantile_nuisance,
    make_folds,
    standardize_train_test,
)
from estimators.ch_inverse_ivqr import add_intercept
from estimators.full_control_ivqr import estimate_full_control_ivqr
from estimators.oracle_ivqr import estimate_oracle_ivqr
from estimators.post_selection_ivqr import (
    estimate_post_selection_ivqr,
    evaluate_post_selection_alpha,
    select_controls_lasso,
)


def require_float(value: float | None, name: str = "value") -> float:
    assert value is not None, f"{name} should not be None"
    return value


def require_array(value: np.ndarray | None, name: str = "array") -> np.ndarray:
    assert value is not None, f"{name} should not be None"
    return value


def test_make_folds_covers_each_observation_once() -> None:
    folds = make_folds(n=20, k_folds=5, random_state=123)

    assert len(folds) == 5

    test_indices = []
    for train_idx, test_idx in folds:
        assert np.intersect1d(train_idx, test_idx).size == 0
        test_indices.extend(test_idx.tolist())

    assert np.array_equal(np.sort(test_indices), np.arange(20))
    assert np.all(np.bincount(test_indices, minlength=20) == 1)


def test_standardize_train_test_uses_training_moments() -> None:
    x_train = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    x_test = np.array([[70.0, 140.0], [90.0, 180.0]])

    x_train_scaled, x_test_scaled, scaler = standardize_train_test(x_train, x_test)

    assert np.allclose(x_train_scaled.mean(axis=0), 0.0)
    assert np.allclose(x_train_scaled.std(axis=0), 1.0)
    assert x_test_scaled.shape == x_test.shape
    scaler_mean = require_array(scaler.mean_, "scaler.mean_")
    np.testing.assert_allclose(scaler_mean, x_train.mean(axis=0))
    assert not np.allclose(scaler_mean, np.vstack([x_train, x_test]).mean(axis=0))


def test_fit_quantile_nuisance_returns_fitted_model() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alpha_true = require_float(data.alpha_true, "alpha_true")

    model, converged, message = fit_quantile_nuisance(
        data.y,
        data.d,
        data.x,
        alpha_value=alpha_true,
        tau=0.5,
        penalty=0.01,
    )

    assert message == "ok"
    assert converged is True
    assert model is not None
    predictions = model.predict(data.x)
    assert predictions.shape == (data.x.shape[0],)
    assert np.all(np.isfinite(predictions))


def test_fit_instrument_residualizer_returns_fitted_model() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))

    model, converged, message = fit_instrument_residualizer(
        data.z,
        data.x,
        ridge_alpha=1.0,
    )

    assert message == "ok"
    assert converged is True
    assert model is not None
    predictions = model.predict(data.x)
    residuals = data.z - predictions
    assert predictions.shape == (data.x.shape[0],)
    assert residuals.shape == (data.x.shape[0],)
    assert np.all(np.isfinite(predictions))
    assert np.all(np.isfinite(residuals))


def test_evaluate_dml_ivqr_alpha_returns_finite_statistic() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alpha_true = require_float(data.alpha_true, "alpha_true")

    statistic, converged, message = evaluate_dml_ivqr_alpha(
        data.y,
        data.d,
        data.z,
        data.x,
        alpha_value=alpha_true,
        tau=0.5,
        k_folds=3,
        fold_random_state=123,
        gmm_ridge=1e-6,
    )

    assert message == "ok"
    assert converged is True
    assert np.isfinite(statistic)
    assert statistic >= 0.0


@pytest.mark.parametrize("alpha_value", [np.nan, np.inf, -np.inf, True])
def test_evaluate_dml_ivqr_alpha_rejects_nonfinite_alpha(
    alpha_value: float,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match="alpha_value must be finite"):
        evaluate_dml_ivqr_alpha(
            data.y,
            data.d,
            data.z,
            data.x,
            alpha_value=alpha_value,
            tau=0.5,
            k_folds=3,
        )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("quantile_penalty", np.nan),
        ("quantile_penalty", -1.0),
        ("quantile_penalty", True),
        ("ridge_alpha", np.nan),
        ("ridge_alpha", -1.0),
        ("ridge_alpha", True),
        ("gmm_ridge", np.nan),
        ("gmm_ridge", -1.0),
        ("gmm_ridge", True),
    ],
)
def test_evaluate_dml_ivqr_alpha_rejects_invalid_penalties(
    argument: str,
    value: float,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match=f"{argument} must"):
        evaluate_dml_ivqr_alpha(
            data.y,
            data.d,
            data.z,
            data.x,
            alpha_value=1.0,
            tau=0.5,
            k_folds=3,
            **{argument: value},
        )


@pytest.mark.parametrize("solver", ["warn", "bad_solver"])
def test_evaluate_dml_ivqr_alpha_rejects_invalid_solver(solver: str) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match="Unknown quantile solver"):
        evaluate_dml_ivqr_alpha(
            data.y,
            data.d,
            data.z,
            data.x,
            alpha_value=1.0,
            tau=0.5,
            k_folds=3,
            quantile_solver=solver,
        )


def test_evaluate_dml_ivqr_accepts_single_column_instrument() -> None:
    data = generate_data(
        Design("dgp1", n=60, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    statistic, converged, message = evaluate_dml_ivqr_alpha(
        data.y,
        data.d,
        data.z[:, None],
        data.x,
        alpha_value=1.0,
        tau=0.5,
        k_folds=3,
    )

    assert converged is True
    assert message == "ok"
    assert np.isfinite(statistic)


def test_evaluate_dml_ivqr_rejects_multiple_instruments() -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    z_multiple = np.column_stack([data.z, data.z])

    with pytest.raises(
        ValueError,
        match="DML-IVQR currently supports exactly one excluded instrument",
    ):
        evaluate_dml_ivqr_alpha(
            data.y,
            data.d,
            z_multiple,
            data.x,
            alpha_value=1.0,
            tau=0.5,
            k_folds=3,
        )


def test_evaluate_dml_ivqr_rejects_zero_control_matrix() -> None:
    n = 30

    with pytest.raises(
        ValueError,
        match="DML-IVQR requires at least one control column",
    ):
        evaluate_dml_ivqr_alpha(
            np.ones(n),
            np.zeros(n),
            np.arange(n, dtype=float),
            np.empty((n, 0)),
            alpha_value=1.0,
            tau=0.5,
            k_folds=3,
        )


@pytest.mark.parametrize("use_cache", ["False", 1])
def test_evaluate_dml_ivqr_rejects_nonboolean_use_cache(
    use_cache: object,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match="use_cache must be a boolean"):
        evaluate_dml_ivqr_alpha(
            data.y,
            data.d,
            data.z,
            data.x,
            alpha_value=1.0,
            tau=0.5,
            k_folds=3,
            use_cache=use_cache,
        )


def test_dml_fold_cache_has_disjoint_train_test_splits() -> None:
    data = generate_data(Design("dgp1", n=90, p=8, pi=1.0, tau=0.5, rep=0, seed=123))

    cache = _build_dml_fold_cache(
        data.y,
        data.d,
        data.z,
        data.x,
        k_folds=3,
        random_state=123,
        ridge_alpha=1.0,
    )

    test_counts = np.zeros(data.x.shape[0], dtype=int)
    for fold in cache:
        assert np.intersect1d(fold.train_idx, fold.test_idx).size == 0
        assert fold.x_train_scaled.shape[0] == len(fold.train_idx)
        assert fold.x_test_scaled.shape[0] == len(fold.test_idx)
        assert fold.z_resid_test.shape == (len(fold.test_idx),)
        assert np.all(np.isfinite(fold.z_resid_test))
        test_counts[fold.test_idx] += 1

    assert np.all(test_counts == 1)


def test_dml_cached_and_uncached_alpha_statistics_match() -> None:
    data = generate_data(Design("dgp1", n=90, p=8, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.array([-0.5, 0.0, 0.5])

    cached_stats = []
    uncached_stats = []
    for alpha_value in alphas:
        cached_stat, cached_converged, cached_message = evaluate_dml_ivqr_alpha(
            data.y,
            data.d,
            data.z,
            data.x,
            alpha_value=float(alpha_value),
            tau=0.5,
            k_folds=3,
            fold_random_state=123,
            quantile_penalty=0.01,
            gmm_ridge=1e-6,
            use_cache=True,
        )
        uncached_stat, uncached_converged, uncached_message = (
            _evaluate_dml_ivqr_alpha_uncached(
                data.y,
                data.d,
                data.z,
                data.x,
                alpha_value=float(alpha_value),
                tau=0.5,
                k_folds=3,
                fold_random_state=123,
                quantile_penalty=0.01,
                gmm_ridge=1e-6,
            )
        )

        assert cached_message == uncached_message == "ok"
        assert cached_converged is uncached_converged is True
        cached_stats.append(cached_stat)
        uncached_stats.append(uncached_stat)

    np.testing.assert_allclose(cached_stats, uncached_stats, rtol=1e-12, atol=1e-12)


def test_estimate_dml_ivqr_returns_estimation_result() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 9)

    result = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
        fold_random_state=123,
        quantile_penalty=0.01,
        gmm_ridge=1e-6,
    )

    assert result.estimator == "dml_ivqr"
    assert result.alpha_true == pytest.approx(data.alpha_true)
    assert result.tau == 0.5
    assert result.failed is False
    assert result.alpha_hat is not None
    assert result.objective_value is not None
    assert np.isfinite(result.objective_value)
    assert result.cr_disconnected is not None
    assert result.alpha_grid_size == len(alphas)
    assert result.failed_alpha_count == 0
    assert result.runtime_seconds >= 0.0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"alpha_min": np.nan}, "alpha_min must be finite"),
        ({"alpha_max": np.inf}, "alpha_max must be finite"),
        ({"alpha_step": np.nan}, "alpha_step must be finite and positive"),
        ({"alpha_step": 0.0}, "alpha_step must be finite and positive"),
        ({"alpha_step": True}, "alpha_step must be finite and positive"),
        (
            {"alpha_min": 1.0, "alpha_max": 1.0},
            "alpha_max must exceed alpha_min",
        ),
        (
            {"alpha_min": 2.0, "alpha_max": 1.0},
            "alpha_max must exceed alpha_min",
        ),
    ],
)
def test_estimate_dml_ivqr_rejects_invalid_alpha_grid_bounds(
    arguments: dict[str, object],
    message: str,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match=message):
        estimate_dml_ivqr(
            data,
            tau=0.5,
            k_folds=3,
            **arguments,
        )


@pytest.mark.parametrize("use_cache", ["False", 1])
def test_estimate_dml_ivqr_rejects_nonboolean_use_cache(
    use_cache: object,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(ValueError, match="use_cache must be a boolean"):
        estimate_dml_ivqr(
            data,
            tau=0.5,
            alphas=np.linspace(0.0, 2.0, 3),
            k_folds=3,
            use_cache=use_cache,
        )


@pytest.mark.parametrize("fold_random_state", [True, 1.5, "123"])
def test_estimate_dml_ivqr_rejects_invalid_fold_random_state(
    fold_random_state: object,
) -> None:
    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )

    with pytest.raises(
        ValueError,
        match="fold_random_state must be an integer or None",
    ):
        estimate_dml_ivqr(
            data,
            tau=0.5,
            alphas=np.linspace(0.0, 2.0, 3),
            k_folds=3,
            fold_random_state=fold_random_state,
        )


def test_estimate_dml_ivqr_uses_absolute_score_inversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import estimators.dml_ivqr as dml_module

    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    alphas = np.array([0.0, 1.0, 2.0])
    statistics = {0.0: 5.0, 1.0: 6.0, 2.0: 10.0}
    captured: dict[str, object] = {}
    original_invert = dml_module.invert_score_test

    monkeypatch.setattr(dml_module, "_build_dml_fold_cache", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        dml_module,
        "_evaluate_dml_ivqr_alpha_with_cache",
        lambda **kwargs: (statistics[kwargs["alpha_value"]], True, "ok"),
    )
    monkeypatch.setattr(
        dml_module,
        "critical_value_chi_square",
        lambda confidence_level, df: (
            captured.update({"critical_df": df}) or 3.84
        ),
    )

    def capture_inversion(**kwargs):
        captured.update(kwargs)
        return original_invert(**kwargs)

    monkeypatch.setattr(dml_module, "invert_score_test", capture_inversion)

    result = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
        use_cache=True,
    )

    assert captured["statistic_reference"] is None
    assert captured["inversion_type"] == "absolute"
    assert captured["critical_df"] == 1
    assert result.cr_empty is True


def test_estimate_dml_ivqr_cache_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import estimators.dml_ivqr as dml_module

    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    calls = {"cache": 0, "uncached": 0}

    def fail_cache(*args, **kwargs):
        calls["cache"] += 1
        raise RuntimeError("forced cache failure")

    def record_uncached(*args, **kwargs):
        calls["uncached"] += 1
        return 0.0, True, "ok"

    monkeypatch.setattr(dml_module, "_build_dml_fold_cache", fail_cache)
    monkeypatch.setattr(
        dml_module,
        "_evaluate_dml_ivqr_alpha_uncached",
        record_uncached,
    )

    result = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=np.linspace(0.0, 2.0, 3),
        k_folds=3,
        use_cache=True,
    )

    assert result.failed is True
    assert "Fold cache construction failed: forced cache failure" in result.message
    assert calls == {"cache": 1, "uncached": 0}


@pytest.mark.parametrize(
    ("diagnostics", "message"),
    [
        ({"runtime_seconds": -1.0}, "runtime_seconds must be nonnegative"),
        ({"alpha_grid_size": 0}, "alpha_grid_size must be at least 1"),
        ({"failed_alpha_count": -1}, "failed_alpha_count must be nonnegative"),
        (
            {"alpha_grid_size": 2, "failed_alpha_count": 3},
            "failed_alpha_count cannot exceed alpha_grid_size",
        ),
    ],
)
def test_dml_failed_result_rejects_invalid_diagnostics(
    diagnostics: dict[str, int | float],
    message: str,
) -> None:
    import estimators.dml_ivqr as dml_module

    data = generate_data(
        Design("dgp1", n=20, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    arguments: dict[str, object] = {
        "data": data,
        "tau": 0.5,
        "message": "failed",
        "runtime_seconds": 0.0,
        "alpha_grid_size": 3,
        "failed_alpha_count": 1,
    }
    arguments.update(diagnostics)

    with pytest.raises(ValueError, match=message):
        dml_module._failed_result(**arguments)


def test_estimate_dml_ivqr_all_alpha_points_fail_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import estimators.dml_ivqr as dml_module

    data = generate_data(
        Design("dgp1", n=30, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    )
    alphas = np.linspace(0.0, 2.0, 3)
    monkeypatch.setattr(
        dml_module,
        "_build_dml_fold_cache",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        dml_module,
        "_evaluate_dml_ivqr_alpha_with_cache",
        lambda **kwargs: (np.inf, False, "forced alpha failure"),
    )

    result = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
    )

    assert result.failed is True
    assert result.failed_alpha_count == len(alphas)
    assert result.alpha_grid_size == len(alphas)
    assert "All alpha grid points failed" in result.message
    assert "first_failure=forced alpha failure" in result.message


def test_estimate_dml_ivqr_cached_and_uncached_results_match() -> None:
    data = generate_data(Design("dgp1", n=90, p=8, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.array([-0.5, 0.0, 0.5])

    cached = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
        fold_random_state=123,
        quantile_penalty=0.01,
        gmm_ridge=1e-6,
        use_cache=True,
    )
    uncached = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
        fold_random_state=123,
        quantile_penalty=0.01,
        gmm_ridge=1e-6,
        use_cache=False,
    )

    assert cached.failed is False
    assert uncached.failed is False
    assert cached.alpha_hat is not None
    assert uncached.alpha_hat is not None
    assert cached.objective_value is not None
    assert uncached.objective_value is not None
    assert cached.alpha_hat == pytest.approx(uncached.alpha_hat)
    assert cached.objective_value == pytest.approx(uncached.objective_value)
    if cached.cr_lower is None:
        assert uncached.cr_lower is None
    else:
        assert uncached.cr_lower is not None
        assert cached.cr_lower == pytest.approx(uncached.cr_lower)
    if cached.cr_upper is None:
        assert uncached.cr_upper is None
    else:
        assert uncached.cr_upper is not None
        assert cached.cr_upper == pytest.approx(uncached.cr_upper)
    if cached.cr_length is None:
        assert uncached.cr_length is None
    else:
        assert uncached.cr_length is not None
        assert cached.cr_length == pytest.approx(uncached.cr_length)
    assert cached.cr_covers_true is uncached.cr_covers_true
    assert cached.cr_empty is uncached.cr_empty
    assert cached.cr_disconnected is uncached.cr_disconnected


def test_estimate_dml_ivqr_invalid_k_folds_raises_value_error() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 5)

    with pytest.raises(ValueError):
        estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=0)
    with pytest.raises(ValueError):
        estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=1)
    with pytest.raises(ValueError, match="k_folds must be an integer"):
        estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=True)
    with pytest.raises(ValueError):
        estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=data.x.shape[0] + 1)


def test_estimate_dml_ivqr_accepts_five_folds_for_robustness() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    result = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=np.linspace(0.0, 2.0, 3),
        k_folds=5,
        fold_random_state=123,
        quantile_penalty=0.01,
    )

    assert result.alpha_grid_size == 3
    assert result.estimator == "dml_ivqr"


def test_estimate_dml_ivqr_invalid_tau_raises_value_error() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))

    with pytest.raises(ValueError):
        estimate_dml_ivqr(data, tau=0.0, alphas=np.linspace(0.0, 2.0, 5))


def test_estimate_dml_ivqr_output_is_deterministic() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 9)

    result_1 = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
        fold_random_state=123,
        quantile_penalty=0.01,
        gmm_ridge=1e-6,
    )
    result_2 = estimate_dml_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        k_folds=3,
        fold_random_state=123,
        quantile_penalty=0.01,
        gmm_ridge=1e-6,
    )

    assert result_1.alpha_hat is not None
    assert result_2.alpha_hat is not None
    assert result_1.objective_value is not None
    assert result_2.objective_value is not None
    assert result_1.alpha_hat == pytest.approx(result_2.alpha_hat)
    assert result_1.objective_value == pytest.approx(result_2.objective_value)
    if result_1.cr_lower is None:
        assert result_2.cr_lower is None
    else:
        assert result_2.cr_lower is not None
        assert result_1.cr_lower == pytest.approx(result_2.cr_lower)
    if result_1.cr_upper is None:
        assert result_2.cr_upper is None
    else:
        assert result_2.cr_upper is not None
        assert result_1.cr_upper == pytest.approx(result_2.cr_upper)
