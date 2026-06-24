"""Tests for reusable IVQR and DML-IVQR moment helpers."""

import numpy as np
import pytest

import inference.moments as moments
from inference.alpha_grid import alpha_grid
from inference.moments import (
    evaluate_grid,
    make_instruments,
    moment_contributions,
    moment_covariance,
    quantile_score,
    residuals_alpha,
    sample_moment,
    score_statistic,
    weighted_gmm_statistic,
)


def test_quantile_score_uses_weak_inequality_at_zero() -> None:
    residuals = np.array([-1.0, 0.0, 2.0])

    scores = quantile_score(residuals, tau=0.5)

    assert np.allclose(scores, np.array([-0.5, -0.5, 0.5]))
    assert scores.shape == residuals.shape


def test_quantile_score_uses_validated_numpy_tau() -> None:
    scores = quantile_score(np.array([-1.0, 1.0]), tau=np.float64(0.5))

    assert scores.dtype == float
    assert np.allclose(scores, np.array([-0.5, 0.5]))


def test_quantile_score_rejects_empty_residuals() -> None:
    with pytest.raises(ValueError, match="residuals must be nonempty"):
        quantile_score(np.array([]), tau=0.5)


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf])
def test_quantile_score_validates_tau(tau: float) -> None:
    with pytest.raises(ValueError):
        quantile_score(np.array([1.0]), tau=tau)


@pytest.mark.parametrize(
    "residuals",
    [
        np.array([[1.0]]),
        np.array([np.nan]),
        np.array([np.inf]),
    ],
)
def test_quantile_score_validates_residuals(residuals: np.ndarray) -> None:
    with pytest.raises(ValueError):
        quantile_score(residuals, tau=0.5)


def test_residuals_alpha_implements_formula() -> None:
    y = np.array([3.0, 4.0, 8.0])
    d = np.array([1.0, 0.0, 2.0])
    x_beta = np.array([0.5, 1.5, -1.0])

    residuals = residuals_alpha(y, d, x_beta, alpha=2.0)

    assert np.allclose(residuals, np.array([0.5, 2.5, 5.0]))


def test_residuals_alpha_validates_equal_lengths() -> None:
    with pytest.raises(ValueError):
        residuals_alpha(
            np.array([1.0, 2.0]),
            np.array([1.0]),
            np.array([0.0, 0.0]),
            alpha=1.0,
        )


def test_residuals_alpha_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="y must be nonempty"):
        residuals_alpha(np.array([]), np.array([]), np.array([]), alpha=1.0)


@pytest.mark.parametrize(
    ("y", "d", "x_beta", "alpha"),
    [
        (np.array([[1.0]]), np.array([1.0]), np.array([0.0]), 1.0),
        (np.array([1.0]), np.array([[1.0]]), np.array([0.0]), 1.0),
        (np.array([1.0]), np.array([1.0]), np.array([[0.0]]), 1.0),
        (np.array([np.nan]), np.array([1.0]), np.array([0.0]), 1.0),
        (np.array([1.0]), np.array([np.inf]), np.array([0.0]), 1.0),
        (np.array([1.0]), np.array([1.0]), np.array([np.nan]), 1.0),
        (np.array([1.0]), np.array([1.0]), np.array([0.0]), np.nan),
        (np.array([1.0]), np.array([1.0]), np.array([0.0]), np.inf),
        (np.array([1.0]), np.array([1.0]), np.array([0.0]), True),
    ],
)
def test_residuals_alpha_validates_inputs(
    y: np.ndarray,
    d: np.ndarray,
    x_beta: np.ndarray,
    alpha: float,
) -> None:
    with pytest.raises(ValueError):
        residuals_alpha(y, d, x_beta, alpha)


def test_make_instruments_returns_z_column_without_controls() -> None:
    z = np.array([1.0, 0.0, 1.0])

    instruments = make_instruments(z)

    assert instruments.shape == (3, 1)
    assert np.allclose(instruments[:, 0], z)


def test_make_instruments_returns_fresh_array() -> None:
    z = np.array([1.0, 0.0, 1.0])

    instruments = make_instruments(z)
    instruments[0, 0] = 999.0

    assert z[0] == pytest.approx(1.0)


def test_make_instruments_stacks_selected_controls() -> None:
    z = np.array([1.0, 0.0, 1.0])
    x_selected = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])

    instruments = make_instruments(z, x_selected)

    assert instruments.shape == (3, 3)
    assert np.allclose(instruments[:, 0], z)
    assert np.allclose(instruments[:, 1:], x_selected)


def test_make_instruments_accepts_vector_valued_z() -> None:
    z = np.array([[1.0, 2.0], [0.0, 3.0], [1.0, 4.0]])

    instruments = make_instruments(z)

    assert instruments.shape == (3, 2)
    assert np.allclose(instruments, z)


def test_make_instruments_accepts_one_dimensional_controls() -> None:
    instruments = make_instruments(
        np.array([1.0, 0.0, 1.0]),
        np.array([2.0, 3.0, 4.0]),
    )

    assert instruments.shape == (3, 2)


def test_make_instruments_accepts_zero_selected_control_columns() -> None:
    z = np.array([1.0, 0.0, 1.0])

    instruments = make_instruments(z, np.empty((3, 0)))

    assert instruments.shape == (3, 1)
    assert np.allclose(instruments[:, 0], z)


def test_make_instruments_rejects_empty_one_dimensional_controls() -> None:
    with pytest.raises(ValueError):
        make_instruments(np.ones(3), np.array([]))


def test_make_instruments_validates_row_counts() -> None:
    with pytest.raises(ValueError):
        make_instruments(np.array([1.0, 0.0]), np.ones((3, 2)))


@pytest.mark.parametrize(
    "z",
    [
        np.array(1.0),
        np.ones((2, 1, 1)),
        np.empty((2, 0)),
        np.array([1.0, np.nan]),
        np.array([[1.0], [np.inf]]),
    ],
)
def test_make_instruments_validates_z(z: np.ndarray) -> None:
    with pytest.raises(ValueError):
        make_instruments(z)


@pytest.mark.parametrize(
    "x_selected",
    [
        np.array(1.0),
        np.ones((2, 1, 1)),
        np.array([1.0, np.nan]),
        np.array([[1.0], [np.inf]]),
        np.empty((2, 0)),
    ],
)
def test_make_instruments_validates_selected_controls(
    x_selected: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        make_instruments(np.array([1.0, 0.0, 1.0]), x_selected)


def test_sample_moment_returns_instrument_dimension() -> None:
    residuals = np.array([-1.0, 2.0, 3.0])
    instruments = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])

    moment = sample_moment(residuals, tau=0.5, instruments=instruments)

    assert moment.shape == (2,)


def test_moment_contributions_shape() -> None:
    residuals = np.array([-1.0, 2.0, 3.0])
    instruments = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])

    contributions = moment_contributions(residuals, tau=0.5, instruments=instruments)

    assert contributions.shape == (3, 2)
    assert np.all(np.isfinite(contributions))


@pytest.mark.parametrize(
    ("residuals", "instruments"),
    [
        (np.array([]), np.empty((0, 1))),
        (np.array([1.0, 2.0]), np.empty((2, 0))),
        (np.array([1.0]), np.ones((2, 1))),
    ],
)
def test_moment_contributions_validates_instrument_dimensions(
    residuals: np.ndarray,
    instruments: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        moment_contributions(residuals, tau=0.5, instruments=instruments)


def test_sample_moment_equals_mean_of_contributions() -> None:
    residuals = np.array([-1.0, 2.0, 3.0])
    instruments = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])

    contributions = moment_contributions(residuals, tau=0.5, instruments=instruments)
    moment = sample_moment(residuals, tau=0.5, instruments=instruments)

    assert np.allclose(moment, contributions.mean(axis=0))


def test_moment_covariance_shape_and_symmetry() -> None:
    contributions = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0]])

    sigma = moment_covariance(contributions, ridge=1e-8)

    assert sigma.shape == (2, 2)
    assert np.allclose(sigma, sigma.T)
    assert np.all(np.isfinite(sigma))


def test_moment_covariance_ridge_adds_positive_diagonal() -> None:
    contributions = np.ones((4, 2))

    sigma = moment_covariance(contributions, ridge=1e-4)

    assert np.all(np.diag(sigma) > 0.0)
    assert np.allclose(np.diag(sigma), np.array([1e-4, 1e-4]))


def test_moment_covariance_constant_contributions_without_ridge_is_zero() -> None:
    sigma = moment_covariance(np.ones((4, 2)), ridge=0.0)

    assert sigma.shape == (2, 2)
    assert np.all(np.isfinite(sigma))
    assert np.allclose(sigma, np.zeros((2, 2)))


def test_weighted_gmm_statistic_is_finite_and_nonnegative() -> None:
    contributions = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0]])

    statistic = weighted_gmm_statistic(contributions, ridge=1e-8)

    assert np.isfinite(statistic)
    assert statistic >= 0.0


def test_weighted_gmm_statistic_matches_manual_scalar_case() -> None:
    contributions = np.array([[1.0], [2.0], [3.0], [4.0]])
    ridge = 1e-8
    n = contributions.shape[0]
    g_hat = contributions.mean(axis=0)
    centered = contributions - g_hat
    sigma = centered.T @ centered / n + ridge * np.eye(1)
    expected = n * g_hat @ np.linalg.inv(sigma) @ g_hat

    statistic = weighted_gmm_statistic(contributions, ridge=ridge, use_pinv=False)

    assert statistic == pytest.approx(float(expected))


@pytest.mark.parametrize(
    ("contributions", "ridge"),
    [
        (np.array([1.0, 2.0, 3.0]), 1e-8),
        (np.array([[1.0], [np.inf]]), 1e-8),
        (np.array([[1.0]]), 1e-8),
        (np.empty((2, 0)), 1e-8),
        (np.array([[1.0], [2.0]]), True),
        (np.array([[1.0], [2.0]]), np.nan),
        (np.array([[1.0], [2.0]]), np.inf),
        (np.array([[1.0], [2.0]]), -1e-8),
    ],
)
def test_moment_covariance_validates_inputs(
    contributions: np.ndarray,
    ridge: float,
) -> None:
    with pytest.raises(ValueError):
        moment_covariance(contributions, ridge=ridge)


@pytest.mark.parametrize(
    ("contributions", "ridge", "use_pinv"),
    [
        (np.array([[1.0]]), 1e-8, True),
        (np.empty((2, 0)), 1e-8, True),
        (np.array([[1.0], [2.0]]), True, True),
        (np.array([[1.0], [2.0]]), 1e-8, 1),
        (np.array([[1.0], [2.0]]), 1e-8, "yes"),
    ],
)
def test_weighted_gmm_statistic_validates_inputs(
    contributions: np.ndarray,
    ridge: float,
    use_pinv: bool,
) -> None:
    with pytest.raises(ValueError):
        weighted_gmm_statistic(
            contributions,
            ridge=ridge,
            use_pinv=use_pinv,
        )


def test_weighted_gmm_statistic_handles_singular_covariance_with_pinv() -> None:
    contributions = np.ones((3, 1))

    statistic = weighted_gmm_statistic(contributions, ridge=0.0, use_pinv=True)

    assert statistic == pytest.approx(0.0)


def test_weighted_gmm_statistic_singular_covariance_raises_without_pinv() -> None:
    with pytest.raises(np.linalg.LinAlgError):
        weighted_gmm_statistic(np.ones((3, 1)), ridge=0.0, use_pinv=False)


def test_score_statistic_is_nonnegative() -> None:
    moment_vector = np.array([-0.2, 0.4, 0.1])

    statistic = score_statistic(moment_vector)

    assert statistic >= 0.0
    assert statistic == pytest.approx(0.21)


@pytest.mark.parametrize(
    "moment_vector",
    [
        np.array([[1.0, 2.0]]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_score_statistic_validates_moment_vector(
    moment_vector: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        score_statistic(moment_vector)


def test_evaluate_grid_returns_finite_values() -> None:
    alphas = np.array([0.0, 0.5, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([0.0, 1.0, 1.0, 0.0])
    x_beta = np.array([0.2, 0.3, 0.4, 0.5])
    instruments = make_instruments(
        np.array([1.0, 0.0, 1.0, 0.0]),
        np.array([[0.1], [0.2], [0.3], [0.4]]),
    )

    scores = evaluate_grid(alphas, y, d, x_beta, tau=0.5, instruments=instruments)

    assert scores.shape == alphas.shape
    assert np.all(np.isfinite(scores))


def test_moments_public_api_excludes_alpha_grid() -> None:
    assert "alpha_grid" not in moments.__all__
    assert "weighted_gmm_statistic" in moments.__all__
    assert "quantile_score" in moments.__all__


@pytest.mark.parametrize(
    "alphas",
    [
        np.array([]),
        np.array([0.0, -1.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, np.nan]),
        np.array([[0.0, 1.0]]),
    ],
)
def test_evaluate_grid_validates_alpha_grid(alphas: np.ndarray) -> None:
    with pytest.raises(ValueError):
        evaluate_grid(
            alphas,
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            tau=0.5,
            instruments=np.ones((2, 1)),
        )


@pytest.mark.parametrize(
    ("y", "d", "x_beta", "instruments"),
    [
        (
            np.array([[1.0, 2.0]]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.ones((2, 1)),
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.0]),
            np.array([0.0, 0.0]),
            np.ones((2, 1)),
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, np.nan]),
            np.ones((2, 1)),
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.ones(2),
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.empty((2, 0)),
        ),
        (
            np.array([1.0, 2.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.ones((3, 1)),
        ),
    ],
)
def test_evaluate_grid_validates_data(
    y: np.ndarray,
    d: np.ndarray,
    x_beta: np.ndarray,
    instruments: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        evaluate_grid(
            np.array([0.0, 1.0]),
            y,
            d,
            x_beta,
            tau=0.5,
            instruments=instruments,
        )


def test_moment_outputs_are_deterministic() -> None:
    alphas = alpha_grid(-0.5, 0.5, 0.25)
    y = np.array([1.0, -1.0, 2.0, -2.0])
    d = np.array([1.0, 0.0, 1.0, 0.0])
    x_beta = np.array([0.1, -0.2, 0.3, -0.4])
    instruments = make_instruments(np.array([1.0, 0.0, 1.0, 0.0]))

    first = evaluate_grid(alphas, y, d, x_beta, tau=0.25, instruments=instruments)
    second = evaluate_grid(alphas, y, d, x_beta, tau=0.25, instruments=instruments)

    assert np.allclose(first, second)

