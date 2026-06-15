import numpy as np
import pytest

from ivqr_sim.config import DGPS, K_FOLDS, N_VALUES, P_VALUES, PI_VALUES, R_MAIN, TAUS
from ivqr_sim.dgp import (
    generate_coefficients,
    generate_data,
    generate_errors,
    generate_x,
    make_covariance_matrix,
)
from ivqr_sim.simulation.design import Design


def test_project_design_constants_exist() -> None:
    assert N_VALUES == [250, 500, 1000]
    assert P_VALUES == [200, 300, 500]
    assert PI_VALUES == [1.0, 0.5, 0.25, 0.10]
    assert TAUS == [0.25, 0.50, 0.75]
    assert DGPS == ["dgp1", "dgp2", "dgp3"]
    assert R_MAIN == 1000
    assert K_FOLDS == 5


def test_covariance_matrix_shape_and_values() -> None:
    sigma = make_covariance_matrix(p=4, rho_x=0.5)

    assert sigma.shape == (4, 4)
    assert sigma[0, 0] == pytest.approx(1.0)
    assert sigma[0, 1] == pytest.approx(0.5)
    assert sigma[0, 2] == pytest.approx(0.25)
    assert sigma[1, 3] == pytest.approx(0.25)


def test_generate_x_shape() -> None:
    rng = np.random.default_rng(123)

    x = generate_x(n=50, p=10, rho_x=0.5, rng=rng)

    assert x.shape == (50, 10)


def test_dgp1_coefficient_sparsity() -> None:
    coefs = generate_coefficients("dgp1", p=30)

    assert coefs["beta"].shape == (30,)
    assert coefs["gamma"].shape == (30,)
    assert np.count_nonzero(coefs["beta"]) == 10
    assert np.count_nonzero(coefs["gamma"]) == 10


def test_dgp2_coefficient_sparsity() -> None:
    coefs = generate_coefficients("dgp2", p=30)

    assert np.count_nonzero(coefs["beta"]) == 20
    assert np.count_nonzero(coefs["gamma"]) == 20


def test_dgp3_coefficient_sparsity() -> None:
    coefs = generate_coefficients("dgp3", p=30)

    assert np.count_nonzero(coefs["beta"]) == 10
    assert np.count_nonzero(coefs["gamma"]) == 10


def test_error_generation_shapes() -> None:
    u, v = generate_errors(
        "dgp1",
        n=100,
        rho_uv=0.5,
        df=5,
        rng=np.random.default_rng(123),
    )

    assert u.shape == (100,)
    assert v.shape == (100,)


def test_dgp3_errors_are_finite() -> None:
    u, v = generate_errors(
        "dgp3",
        n=100,
        rho_uv=0.5,
        df=5,
        rng=np.random.default_rng(123),
    )

    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(v))


def test_full_data_generation_shapes() -> None:
    design = Design(dgp="dgp1", n=100, p=20, pi=0.5, tau=0.5, rep=0, seed=123)

    data = generate_data(design)

    assert data.x.shape == (100, 20)
    assert data.y.shape == (100,)
    assert data.d.shape == (100,)
    assert data.z.shape == (100,)
    assert data.u is not None
    assert data.v is not None
    assert data.u.shape == (100,)
    assert data.v.shape == (100,)
    assert isinstance(data.alpha_true, float)


def test_binary_treatment() -> None:
    design = Design(dgp="dgp1", n=100, p=20, pi=0.5, tau=0.5, rep=0, seed=123)
    data = generate_data(design)

    assert set(np.unique(data.d)).issubset({0, 1})


def test_generate_data_is_reproducible() -> None:
    design = Design(dgp="dgp1", n=100, p=20, pi=0.5, tau=0.5, rep=0, seed=123)

    data1 = generate_data(design)
    data2 = generate_data(design)

    assert np.allclose(data1.y, data2.y)
    assert np.allclose(data1.d, data2.d)
    assert np.allclose(data1.z, data2.z)
    assert np.allclose(data1.x, data2.x)


def test_different_seed_gives_different_data() -> None:
    design1 = Design(dgp="dgp1", n=100, p=20, pi=0.5, tau=0.5, rep=0, seed=123)
    design2 = Design(dgp="dgp1", n=100, p=20, pi=0.5, tau=0.5, rep=0, seed=456)

    data1 = generate_data(design1)
    data2 = generate_data(design2)

    assert not np.allclose(data1.y, data2.y)


def test_median_true_alpha_is_one() -> None:
    design = Design(dgp="dgp1", n=100, p=20, pi=0.5, tau=0.5, rep=0, seed=123)

    data = generate_data(design)

    assert data.alpha_true == pytest.approx(1.0)


def test_data_arrays_are_invariant_to_tau_for_same_seed() -> None:
    design_25 = Design(dgp="dgp1", n=500, p=50, pi=0.5, tau=0.25, rep=0, seed=123)
    design_50 = Design(dgp="dgp1", n=500, p=50, pi=0.5, tau=0.50, rep=0, seed=123)
    design_75 = Design(dgp="dgp1", n=500, p=50, pi=0.5, tau=0.75, rep=0, seed=123)

    data_25 = generate_data(design_25)
    data_50 = generate_data(design_50)
    data_75 = generate_data(design_75)

    assert np.allclose(data_25.x, data_50.x)
    assert np.allclose(data_50.x, data_75.x)
    assert np.allclose(data_25.z, data_50.z)
    assert np.allclose(data_50.z, data_75.z)
    assert np.allclose(data_25.d, data_50.d)
    assert np.allclose(data_50.d, data_75.d)
    assert data_25.u is not None
    assert data_50.u is not None
    assert data_75.u is not None
    assert data_25.v is not None
    assert data_50.v is not None
    assert data_75.v is not None
    assert np.allclose(data_25.u, data_50.u)
    assert np.allclose(data_50.u, data_75.u)
    assert np.allclose(data_25.v, data_50.v)
    assert np.allclose(data_50.v, data_75.v)
    assert np.allclose(data_25.y, data_50.y)
    assert np.allclose(data_50.y, data_75.y)

    assert data_25.alpha_true != data_50.alpha_true
    assert data_50.alpha_true != data_75.alpha_true
    assert data_50.alpha_true == pytest.approx(1.0)


def test_outcome_equation_is_nonseparable_structural_design() -> None:
    design = Design(dgp="dgp1", n=500, p=50, pi=0.5, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    beta = generate_coefficients("dgp1", 50)["beta"]

    assert data.u is not None
    expected_y = data.d + data.x @ beta + (1.0 + data.d) * data.u

    assert np.allclose(data.y, expected_y)


def test_dgp3_outcome_is_invariant_to_tau_for_same_seed() -> None:
    design_25 = Design(dgp="dgp3", n=500, p=50, pi=0.5, tau=0.25, rep=0, seed=123)
    design_50 = Design(dgp="dgp3", n=500, p=50, pi=0.5, tau=0.50, rep=0, seed=123)
    design_75 = Design(dgp="dgp3", n=500, p=50, pi=0.5, tau=0.75, rep=0, seed=123)

    data_25 = generate_data(design_25)
    data_50 = generate_data(design_50)
    data_75 = generate_data(design_75)

    assert np.allclose(data_25.y, data_50.y)
    assert np.allclose(data_50.y, data_75.y)
    assert data_25.alpha_true != data_50.alpha_true
    assert data_50.alpha_true != data_75.alpha_true


def test_invalid_dgp_raises_value_error() -> None:
    with pytest.raises(ValueError):
        generate_coefficients("wrong_dgp", p=10)


@pytest.mark.parametrize("n", [0, -1])
def test_nonpositive_n_raises_value_error(n: int) -> None:
    with pytest.raises(ValueError):
        generate_x(n=n, p=10, rho_x=0.5, rng=np.random.default_rng(123))


@pytest.mark.parametrize("p", [0, -1])
def test_nonpositive_p_raises_value_error(p: int) -> None:
    with pytest.raises(ValueError):
        make_covariance_matrix(p=p, rho_x=0.5)


@pytest.mark.parametrize("rho_x", [1.0, -1.0])
def test_invalid_rho_x_raises_value_error(rho_x: float) -> None:
    with pytest.raises(ValueError):
        make_covariance_matrix(p=10, rho_x=rho_x)


@pytest.mark.parametrize("rho_uv", [1.0, -1.0])
def test_invalid_rho_uv_raises_value_error(rho_uv: float) -> None:
    with pytest.raises(ValueError):
        generate_errors(
            "dgp1",
            n=100,
            rho_uv=rho_uv,
            df=5,
            rng=np.random.default_rng(123),
        )


@pytest.mark.parametrize("df", [2, 1])
def test_invalid_df_for_dgp3_raises_value_error(df: int) -> None:
    with pytest.raises(ValueError):
        generate_errors(
            "dgp3",
            n=100,
            rho_uv=0.5,
            df=df,
            rng=np.random.default_rng(123),
        )


def test_generate_data_invalid_design_values_raise_value_error() -> None:
    with pytest.raises(ValueError):
        generate_data(Design(dgp="dgp1", n=0, p=20, pi=0.5, tau=0.5, rep=0, seed=123))
    with pytest.raises(ValueError):
        generate_data(Design(dgp="dgp1", n=100, p=0, pi=0.5, tau=0.5, rep=0, seed=123))
    with pytest.raises(ValueError):
        generate_data(Design(dgp="dgp1", n=100, p=20, pi=-0.1, tau=0.5, rep=0, seed=123))
