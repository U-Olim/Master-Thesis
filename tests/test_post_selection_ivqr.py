import numpy as np
import pytest

from ivqr_sim.dgp import generate_data
from ivqr_sim.estimators.post_selection_ivqr import (
    estimate_post_selection_ivqr,
    evaluate_post_selection_alpha,
    fit_post_selection_beta,
    select_controls_lasso,
)
from ivqr_sim.simulation.design import Design


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
    )

    assert message == "ok"
    assert converged is True
    assert np.isfinite(statistic)
    assert statistic >= 0.0


def test_estimate_post_selection_ivqr_returns_estimation_result() -> None:
    data = generate_data(Design("dgp1", n=100, p=10, pi=1.0, tau=0.5, rep=0, seed=123))
    alphas = np.linspace(0.0, 2.0, 11)

    result = estimate_post_selection_ivqr(data, tau=0.5, alphas=alphas, selection_cv=3)

    assert result.estimator == "post_selection_ivqr"
    assert result.alpha_true == pytest.approx(data.alpha_true)
    assert result.tau == 0.5
    assert result.failed is False
    assert result.alpha_hat is not None
    assert result.selected_controls is not None
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
    )
    result_2 = estimate_post_selection_ivqr(
        data,
        tau=0.5,
        alphas=alphas,
        selection_random_state=123,
        selection_cv=3,
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
