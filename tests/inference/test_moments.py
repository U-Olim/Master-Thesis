"""Tests for production moment helpers."""

import numpy as np
import pytest

import inference.moments as moments
from inference.moments import quantile_score, weighted_gmm_statistic


def test_quantile_score_uses_weak_inequality_at_zero() -> None:
    residuals = np.array([-1.0, 0.0, 2.0])

    scores = quantile_score(residuals, tau=0.5)

    np.testing.assert_array_equal(scores, np.array([-0.5, -0.5, 0.5]))
    assert scores.shape == residuals.shape


def test_quantile_score_uses_validated_numpy_tau() -> None:
    scores = quantile_score(np.array([-1.0, 1.0]), tau=np.float64(0.5))

    assert scores.dtype == float
    np.testing.assert_array_equal(scores, np.array([-0.5, 0.5]))


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


def test_weighted_gmm_statistic_is_deterministic_for_fixed_inputs() -> None:
    contributions = np.array(
        [
            [0.25, -0.50],
            [0.75, 0.25],
            [-0.25, 0.50],
            [1.00, -0.75],
        ]
    )

    first = weighted_gmm_statistic(contributions, ridge=1e-8)
    second = weighted_gmm_statistic(contributions, ridge=1e-8)

    assert first == pytest.approx(second)


@pytest.mark.parametrize(
    ("contributions", "ridge", "use_pinv"),
    [
        (np.array([[1.0]]), 1e-8, True),
        (np.empty((2, 0)), 1e-8, True),
        (np.array([1.0, 2.0, 3.0]), 1e-8, True),
        (np.array([[1.0], [np.inf]]), 1e-8, True),
        (np.array([[1.0], [2.0]]), True, True),
        (np.array([[1.0], [2.0]]), np.nan, True),
        (np.array([[1.0], [2.0]]), np.inf, True),
        (np.array([[1.0], [2.0]]), -1e-8, True),
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


def test_moments_public_api_exposes_only_production_helpers() -> None:
    assert moments.__all__ == [
        "quantile_score",
        "weighted_gmm_statistic",
    ]
