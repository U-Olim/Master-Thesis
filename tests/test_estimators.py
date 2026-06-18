# Consolidated tests for the thematic project structure.

import _path  # noqa: F401
import numpy as np
import pytest

from dgp import generate_data
from dgp.designs import Design
from estimators import full_ivqr as full_ivqr_module
from estimators.base import EstimationResult
from estimators.dml_ivqr import (
    estimate_dml_ivqr,
    evaluate_dml_ivqr_alpha,
    fit_instrument_residualizer,
    fit_quantile_nuisance,
    make_folds,
    standardize_train_test,
)
from estimators.full_ivqr import (
    add_intercept,
    estimate_full_ivqr,
    evaluate_full_ivqr_alpha,
    fit_profile_beta,
)
from estimators.post_selection_ivqr import (
    estimate_post_selection_ivqr,
    evaluate_post_selection_alpha,
    fit_post_selection_beta,
    select_controls_lasso,
)


def test_estimation_result_can_be_instantiated() -> None:
    result = EstimationResult(
        estimator="full_ivqr",
        alpha_hat=None,
        alpha_true=1.0,
        tau=0.5,
        converged=False,
        failed=True,
        message="not implemented",
        objective_value=None,
        at_grid_boundary=False,
        alpha_grid_size=None,
        failed_alpha_count=None,
        cr_lower=None,
        cr_upper=None,
        cr_length=None,
        cr_covers_true=None,
        cr_empty=True,
        cr_disconnected=None,
        selected_controls=None,
        runtime_seconds=0.0,
    )

    assert result.estimator == "full_ivqr"
    assert result.failed is True
    assert result.cr_empty is True
    assert result.cr_disconnected is None


def test_empty_confidence_region_is_separate_from_estimator_failure() -> None:
    result = EstimationResult(
        estimator="dml_ivqr",
        alpha_hat=1.0,
        alpha_true=1.0,
        tau=0.5,
        converged=True,
        failed=False,
        message="ok",
        objective_value=10.0,
        at_grid_boundary=False,
        alpha_grid_size=5,
        failed_alpha_count=0,
        cr_lower=None,
        cr_upper=None,
        cr_length=None,
        cr_covers_true=False,
        cr_empty=True,
        cr_disconnected=False,
        selected_controls=None,
        runtime_seconds=0.0,
    )

    assert result.failed is False
    assert result.converged is True
    assert result.cr_empty is True

def test_add_intercept_prepends_ones() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    x_design = add_intercept(x)

    assert x_design.shape == (2, 3)
    assert np.allclose(x_design[:, 0], 1.0)
    assert np.allclose(x_design[:, 1:], x)


def test_fit_profile_beta_works_on_small_data() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)

    beta_hat, converged, message = fit_profile_beta(
        data.y,
        data.d,
        data.x,
        alpha=data.alpha_true,
        tau=0.5,
    )

    assert message == "ok"
    assert converged is True
    assert beta_hat.shape == (data.x.shape[1] + 1,)
    assert np.all(np.isfinite(beta_hat))


def test_evaluate_full_ivqr_alpha_returns_finite_statistic() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)

    statistic, converged, message = evaluate_full_ivqr_alpha(
        data.y,
        data.d,
        data.z,
        data.x,
        alpha=data.alpha_true,
        tau=0.5,
        gmm_ridge=1e-6,
    )

    assert message == "ok"
    assert converged is True
    assert np.isfinite(statistic)
    assert statistic >= 0.0


def test_estimate_full_ivqr_returns_estimation_result() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alphas = np.linspace(0.0, 2.0, 11)

    result = estimate_full_ivqr(data, tau=0.5, alphas=alphas, gmm_ridge=1e-6)

    assert result.estimator == "full_ivqr"
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


def test_estimate_full_ivqr_infeasible_high_dimensional_case_fails_cleanly() -> None:
    design = Design("dgp1", n=20, p=25, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)

    result = estimate_full_ivqr(data, tau=0.5, alphas=np.linspace(0.0, 2.0, 5))

    assert result.failed is True
    assert result.converged is False
    assert result.alpha_hat is None
    assert result.alpha_grid_size is None
    assert result.failed_alpha_count is None
    assert "infeasible" in result.message


def test_estimate_full_ivqr_some_failed_alphas_still_converges(monkeypatch: pytest.MonkeyPatch) -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alphas = np.array([0.0, 1.0, 2.0])
    original = full_ivqr_module.evaluate_full_ivqr_alpha

    def fake_evaluate(*args: object, **kwargs: object) -> tuple[float, bool, str]:
        if kwargs["alpha"] == 1.0:
            return np.inf, False, "forced failure"
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(full_ivqr_module, "evaluate_full_ivqr_alpha", fake_evaluate)

    result = estimate_full_ivqr(data, tau=0.5, alphas=alphas)

    assert result.failed is False
    assert result.converged is True
    assert result.alpha_hat is not None
    assert result.alpha_grid_size == len(alphas)
    assert result.failed_alpha_count == 1
    assert "failed_alpha_points=1/3" in result.message


def test_estimate_full_ivqr_all_failed_alphas_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alphas = np.array([0.0, 1.0, 2.0])

    def fake_evaluate(*args: object, **kwargs: object) -> tuple[float, bool, str]:
        return np.inf, False, "forced failure"

    monkeypatch.setattr(full_ivqr_module, "evaluate_full_ivqr_alpha", fake_evaluate)

    result = estimate_full_ivqr(data, tau=0.5, alphas=alphas)

    assert result.failed is True
    assert result.converged is False
    assert result.alpha_hat is None
    assert result.alpha_grid_size == len(alphas)
    assert result.failed_alpha_count == len(alphas)
    assert "All alpha-grid evaluations failed" in result.message


def test_estimate_full_ivqr_invalid_tau_raises_value_error() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)

    with pytest.raises(ValueError):
        estimate_full_ivqr(data, tau=0.0, alphas=np.linspace(0.0, 2.0, 5))


def test_estimate_full_ivqr_confidence_region_fields_are_coherent() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)

    result = estimate_full_ivqr(data, tau=0.5, alphas=np.linspace(0.0, 2.0, 11))

    if result.cr_empty is False:
        assert result.cr_lower is not None
        assert result.cr_upper is not None
        assert result.cr_length is not None
        assert result.cr_upper >= result.cr_lower


def test_estimate_full_ivqr_output_is_deterministic() -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    alphas = np.linspace(0.0, 2.0, 11)

    result_1 = estimate_full_ivqr(data, tau=0.5, alphas=alphas, gmm_ridge=1e-6)
    result_2 = estimate_full_ivqr(data, tau=0.5, alphas=alphas, gmm_ridge=1e-6)

    assert result_1.alpha_hat == pytest.approx(result_2.alpha_hat)
    assert result_1.objective_value == pytest.approx(result_2.objective_value)
    assert result_1.cr_lower == pytest.approx(result_2.cr_lower)
    assert result_1.cr_upper == pytest.approx(result_2.cr_upper)

def test_select_controls_lasso_returns_valid_indices() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))

    selected, message = select_controls_lasso(
        data.y,
        data.d,
        data.x,
        tau=0.5,
        random_state=123,
        cv=3,
    )

    assert selected.ndim == 1
    assert np.issubdtype(selected.dtype, np.integer)
    assert np.all(selected >= 0)
    assert np.all(selected < data.x.shape[1])
    assert np.all(np.diff(selected) > 0) or selected.size <= 1
    assert isinstance(message, str)


def test_select_controls_lasso_handles_no_signal_artificial_data() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(40, 5))
    y = np.zeros(40)
    d = np.zeros(40)

    selected, message = select_controls_lasso(
        y,
        d,
        x,
        tau=0.5,
        random_state=123,
        cv=3,
    )

    assert selected.ndim == 1
    assert np.issubdtype(selected.dtype, np.integer)
    assert selected.size <= x.shape[1]
    assert isinstance(message, str)


def test_fit_post_selection_beta_works_with_selected_controls() -> None:
    data = generate_data(Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123))
    x_selected = data.x[:, :3]

    beta_hat, converged, message = fit_post_selection_beta(
        data.y,
        data.d,
        x_selected,
        alpha=data.alpha_true,
        tau=0.5,
    )

    assert message == "ok"
    assert converged is True
    assert beta_hat.shape == (4,)
    assert np.all(np.isfinite(beta_hat))


def test_fit_post_selection_beta_works_with_zero_selected_controls() -> None:
    data = generate_data(Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123))
    x_selected = np.empty((data.x.shape[0], 0))

    beta_hat, converged, message = fit_post_selection_beta(
        data.y,
        data.d,
        x_selected,
        alpha=data.alpha_true,
        tau=0.5,
    )

    assert message == "ok"
    assert converged is True
    assert beta_hat.shape == (1,)
    assert np.all(np.isfinite(beta_hat))


def test_evaluate_post_selection_alpha_returns_finite_statistic() -> None:
    data = generate_data(Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123))
    x_selected = data.x[:, :3]

    statistic, converged, message = evaluate_post_selection_alpha(
        data.y,
        data.d,
        data.z,
        x_selected,
        alpha=data.alpha_true,
        tau=0.5,
        gmm_ridge=1e-6,
    )

    assert message == "ok"
    assert converged is True
    assert np.isfinite(statistic)
    assert statistic >= 0.0


def test_estimate_post_selection_ivqr_returns_estimation_result() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 11)

    result = estimate_post_selection_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        selection_cv=3,
        gmm_ridge=1e-6,
    )

    assert result.estimator == "post_selection_ivqr"
    assert result.alpha_true == pytest.approx(data.alpha_true)
    assert result.tau == 0.5
    assert result.failed is False
    assert result.alpha_hat is not None
    assert result.selected_controls is not None
    assert result.objective_value is not None
    assert np.isfinite(result.objective_value)
    assert result.cr_disconnected is not None
    assert result.alpha_grid_size == len(alphas)
    assert result.failed_alpha_count == 0
    assert result.runtime_seconds >= 0.0


def test_fit_post_selection_beta_infeasible_selected_dimension_fails_cleanly() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([0.0, 1.0, 0.0, 1.0])
    x_selected = np.ones((4, 3))

    beta_hat, converged, message = fit_post_selection_beta(
        y,
        d,
        x_selected,
        alpha=1.0,
        tau=0.5,
    )

    assert converged is False
    assert np.all(np.isnan(beta_hat))
    assert "infeasible" in message


def test_estimate_post_selection_ivqr_invalid_tau_raises_value_error() -> None:
    data = generate_data(Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123))

    with pytest.raises(ValueError):
        estimate_post_selection_ivqr(data, tau=0.0, alphas=np.linspace(0.0, 2.0, 5))


def test_estimate_post_selection_ivqr_output_is_deterministic() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 11)

    result_1 = estimate_post_selection_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        selection_random_state=123,
        selection_cv=3,
        gmm_ridge=1e-6,
    )
    result_2 = estimate_post_selection_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        selection_random_state=123,
        selection_cv=3,
        gmm_ridge=1e-6,
    )

    assert result_1.alpha_hat == pytest.approx(result_2.alpha_hat)
    assert result_1.selected_controls == result_2.selected_controls
    assert result_1.objective_value == pytest.approx(result_2.objective_value)
    if result_1.cr_lower is None:
        assert result_2.cr_lower is None
    else:
        assert result_1.cr_lower == pytest.approx(result_2.cr_lower)
    if result_1.cr_upper is None:
        assert result_2.cr_upper is None
    else:
        assert result_1.cr_upper == pytest.approx(result_2.cr_upper)

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
    x_test = np.array([[7.0, 14.0], [9.0, 18.0]])

    x_train_scaled, x_test_scaled, scaler = standardize_train_test(x_train, x_test)

    assert np.allclose(x_train_scaled.mean(axis=0), 0.0)
    assert np.allclose(x_train_scaled.std(axis=0), 1.0)
    assert x_test_scaled.shape == x_test.shape
    assert np.allclose(scaler.mean_, x_train.mean(axis=0))


def test_fit_quantile_nuisance_returns_fitted_model() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))

    model, converged, message = fit_quantile_nuisance(
        data.y,
        data.d,
        data.x,
        alpha_value=data.alpha_true,
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

    statistic, converged, message = evaluate_dml_ivqr_alpha(
        data.y,
        data.d,
        data.z,
        data.x,
        alpha_value=data.alpha_true,
        tau=0.5,
        k_folds=3,
        fold_random_state=123,
        gmm_ridge=1e-6,
    )

    assert message == "ok"
    assert converged is True
    assert np.isfinite(statistic)
    assert statistic >= 0.0


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


def test_estimate_dml_ivqr_invalid_k_folds_raises_value_error() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 5)

    with pytest.raises(ValueError):
        estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=1)
    with pytest.raises(ValueError):
        estimate_dml_ivqr(data, tau=0.5, alphas=alphas, k_folds=data.x.shape[0] + 1)


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

    assert result_1.alpha_hat == pytest.approx(result_2.alpha_hat)
    assert result_1.objective_value == pytest.approx(result_2.objective_value)
    if result_1.cr_lower is None:
        assert result_2.cr_lower is None
    else:
        assert result_1.cr_lower == pytest.approx(result_2.cr_lower)
    if result_1.cr_upper is None:
        assert result_2.cr_upper is None
    else:
        assert result_1.cr_upper == pytest.approx(result_2.cr_upper)
