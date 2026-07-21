import numpy as np

from dgp import (
    Design,
    generate_data,
    get_oracle_control_indices,
    true_active_control_indices,
    true_alpha,
)
from dgp.true_parameters import true_sparse_coefficients
from simulation.config import DGPS, TAUS


EXPECTED_ACTIVE_CONTROLS = {"dgp1": 5, "dgp2": 10, "dgp3": 5}


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
    p = 20
    for dgp in DGPS:
        indices = get_oracle_control_indices(dgp, p)
        assert indices.ndim == 1
        assert len(indices) == EXPECTED_ACTIVE_CONTROLS[dgp]
        assert np.all(indices >= 0)
        assert np.all(indices < p)


def test_declared_and_coefficient_implied_support_are_exactly_equal() -> None:
    for dgp, minimum_p in EXPECTED_ACTIVE_CONTROLS.items():
        for p in (minimum_p, 20, 200):
            np.testing.assert_array_equal(
                get_oracle_control_indices(dgp, p),
                true_active_control_indices(dgp, p),
            )


def test_sparse_coefficient_vectors_have_expected_active_counts() -> None:
    p = 20
    for dgp, expected in EXPECTED_ACTIVE_CONTROLS.items():
        beta, gamma = true_sparse_coefficients(dgp, p)
        active = np.flatnonzero((np.abs(beta) > 1e-12) | (np.abs(gamma) > 1e-12))
        assert len(active) == expected


def test_oracle_control_indices_reject_too_small_p_helpfully() -> None:
    try:
        get_oracle_control_indices("dgp2", 5)
    except ValueError as exc:
        assert "requires p >= 10 because it uses 10 active controls" in str(exc)
    else:
        raise AssertionError("Expected dgp2 oracle support to reject p=5")
