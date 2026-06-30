import numpy as np

from dgp import Design, generate_data, get_oracle_control_indices, true_alpha
from simulation.config import DGPS, TAUS


def test_design_validation_works_through_generator() -> None:
    data = generate_data(Design("dgp1", n=40, p=5, pi=1.0, tau=0.5, rep=0, seed=123))
    assert data.y.shape == (40,)
    assert data.d.shape == (40,)
    assert data.z.shape == (40,)
    assert data.x.shape == (40, 5)


def test_generate_data_is_reproducible_for_same_design_and_seed() -> None:
    design = Design("dgp1", n=40, p=10, pi=0.5, tau=0.5, rep=0, seed=123)
    first = generate_data(design)
    second = generate_data(design)
    np.testing.assert_allclose(first.y, second.y)
    np.testing.assert_allclose(first.d, second.d)
    np.testing.assert_allclose(first.z, second.z)
    np.testing.assert_allclose(first.x, second.x)


def test_true_alpha_is_finite_for_all_design_quantiles() -> None:
    for dgp in DGPS:
        for tau in TAUS:
            assert np.isfinite(true_alpha(tau, dgp))


def test_oracle_control_indices_are_valid() -> None:
    p = 25
    for dgp in DGPS:
        indices = get_oracle_control_indices(dgp, p)
        assert indices.ndim == 1
        assert len(indices) > 0
        assert np.all(indices >= 0)
        assert np.all(indices < p)


def test_oracle_control_indices_reject_too_small_p_helpfully() -> None:
    try:
        get_oracle_control_indices("dgp1", 5)
    except ValueError as exc:
        assert "requires at least 10 controls" in str(exc)
    else:
        raise AssertionError("Expected dgp1 oracle support to reject p=5")
