import numpy as np
import pytest

from ivqr_sim.moments import (
    alpha_grid,
    evaluate_grid,
    make_instruments,
    quantile_score,
    residuals_alpha,
    sample_moment,
    score_statistic,
)


def test_quantile_score_uses_weak_inequality_at_zero() -> None:
    residuals = np.array([-1.0, 0.0, 2.0])

    scores = quantile_score(residuals, tau=0.5)

    assert np.allclose(scores, np.array([-0.5, -0.5, 0.5]))
    assert scores.shape == residuals.shape


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.1])
def test_quantile_score_validates_tau(tau: float) -> None:
    with pytest.raises(ValueError):
        quantile_score(np.array([1.0]), tau=tau)


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


def test_make_instruments_returns_z_column_without_controls() -> None:
    z = np.array([1.0, 0.0, 1.0])

    instruments = make_instruments(z)

    assert instruments.shape == (3, 1)
    assert np.allclose(instruments[:, 0], z)


def test_make_instruments_stacks_selected_controls() -> None:
    z = np.array([1.0, 0.0, 1.0])
    x_selected = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])

    instruments = make_instruments(z, x_selected)

    assert instruments.shape == (3, 3)
    assert np.allclose(instruments[:, 0], z)
    assert np.allclose(instruments[:, 1:], x_selected)


def test_make_instruments_validates_row_counts() -> None:
    with pytest.raises(ValueError):
        make_instruments(np.array([1.0, 0.0]), np.ones((3, 2)))


def test_sample_moment_returns_instrument_dimension() -> None:
    residuals = np.array([-1.0, 2.0, 3.0])
    instruments = np.array([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])

    moment = sample_moment(residuals, tau=0.5, instruments=instruments)

    assert moment.shape == (2,)


def test_score_statistic_is_nonnegative() -> None:
    moment_vector = np.array([-0.2, 0.4, 0.1])

    statistic = score_statistic(moment_vector)

    assert statistic >= 0.0
    assert statistic == pytest.approx(0.21)


def test_alpha_grid_has_expected_length_and_endpoint() -> None:
    grid = alpha_grid(-2.0, 4.0, 0.01)

    assert len(grid) == 601
    assert grid[0] == pytest.approx(-2.0)
    assert grid[-1] == pytest.approx(4.0)


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


def test_moment_outputs_are_deterministic() -> None:
    alphas = alpha_grid(-0.5, 0.5, 0.25)
    y = np.array([1.0, -1.0, 2.0, -2.0])
    d = np.array([1.0, 0.0, 1.0, 0.0])
    x_beta = np.array([0.1, -0.2, 0.3, -0.4])
    instruments = make_instruments(np.array([1.0, 0.0, 1.0, 0.0]))

    first = evaluate_grid(alphas, y, d, x_beta, tau=0.25, instruments=instruments)
    second = evaluate_grid(alphas, y, d, x_beta, tau=0.25, instruments=instruments)

    assert np.allclose(first, second)
