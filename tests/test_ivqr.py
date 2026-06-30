import numpy as np

from dgp import Design, generate_data
from ivqr.alpha_grid import alpha_grid
from ivqr.ch_inverse import evaluate_alpha_ch_ivqr
from ivqr.confidence_regions import invert_score_test
from ivqr.moments import quantile_score


def test_alpha_grid_size_and_bounds() -> None:
    grid = alpha_grid(-1.0, 3.0, 1.0)
    np.testing.assert_allclose(grid, np.array([-1.0, 0.0, 1.0, 2.0, 3.0]))


def test_confidence_region_inversion_on_known_vector() -> None:
    region = invert_score_test(
        np.array([-1.0, 0.0, 1.0, 2.0]),
        np.array([5.0, 1.0, 2.0, 6.0]),
        critical_value=3.0,
        alpha_true=1.0,
    )
    assert region.empty is False
    assert region.covers_true is True
    assert region.lower is not None
    assert region.upper is not None
    assert region.lower <= 0.0
    assert region.upper >= 1.0


def test_quantile_score_basic_behavior() -> None:
    scores = quantile_score(np.array([-1.0, 0.0, 1.0]), tau=0.5)
    np.testing.assert_allclose(scores, np.array([-0.5, -0.5, 0.5]))


def test_ch_inverse_can_evaluate_tiny_grid() -> None:
    data = generate_data(Design("dgp1", 50, 4, 1.0, 0.5, rep=0, seed=123))
    evaluation = evaluate_alpha_ch_ivqr(
        y=data.y,
        d=data.d,
        x_controls=data.x[:, :2],
        z=data.z,
        alpha=1.0,
        tau=0.5,
        max_iter=100,
    )
    assert np.isfinite(evaluation.statistic)
