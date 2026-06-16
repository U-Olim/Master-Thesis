import numpy as np
import pytest

from ivqr_sim.dgp import generate_data
from ivqr_sim.estimators.full_ivqr import (
    add_intercept,
    estimate_full_ivqr,
    evaluate_full_ivqr_alpha,
    fit_profile_beta,
)
from ivqr_sim.simulation.design import Design


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
    assert result.runtime_seconds >= 0.0


def test_estimate_full_ivqr_infeasible_high_dimensional_case_fails_cleanly() -> None:
    design = Design("dgp1", n=20, p=25, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)

    result = estimate_full_ivqr(data, tau=0.5, alphas=np.linspace(0.0, 2.0, 5))

    assert result.failed is True
    assert result.converged is False
    assert result.alpha_hat is None
    assert "infeasible" in result.message


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
