# Consolidated tests for the thematic project structure.

from math import sqrt

import numpy as np
import pytest
from scipy.stats import norm, t

from dgp import (
    generate_coefficients,
    generate_data,
    generate_errors,
    generate_x,
    make_covariance_matrix,
)
from dgp.designs import Design
from dgp.generators import _structural_outcome, _treatment_from_latent_index
from dgp.true_parameters import true_alpha
from dgp.true_parameters import get_oracle_control_count, get_oracle_control_indices
from simulation.config import (
    DEFAULT_ALPHA_GRID_SIZE,
    DEFAULT_DML_K_FOLDS,
    DEFAULT_N_JOBS,
    DEFAULT_OUTPUT,
    DEFAULT_QUANTREG_MAX_ITER,
    DGPS,
    FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE,
    FULL_CONTROL_BENCHMARK_DGPS,
    FULL_CONTROL_BENCHMARK_N_VALUES,
    FULL_CONTROL_BENCHMARK_OUTPUT,
    FULL_CONTROL_BENCHMARK_PI_VALUES,
    FULL_CONTROL_BENCHMARK_P_VALUES,
    FULL_CONTROL_BENCHMARK_TAUS,
    K_FOLDS,
    N_VALUES,
    P_VALUES,
    PI_VALUES,
    R_FAST,
    R_FULL_CONTROL_BENCHMARK,
    R_MAIN,
    TAUS,
)


def test_project_design_constants_exist() -> None:
    assert N_VALUES == [500, 1000]
    assert P_VALUES == [200, 500]
    assert FULL_CONTROL_BENCHMARK_DGPS == ["dgp1"]
    assert FULL_CONTROL_BENCHMARK_N_VALUES == [500, 1000]
    assert FULL_CONTROL_BENCHMARK_P_VALUES == [20, 50, 100]
    assert FULL_CONTROL_BENCHMARK_PI_VALUES == [1.0]
    assert FULL_CONTROL_BENCHMARK_TAUS == [0.25, 0.5, 0.75]
    assert FULL_CONTROL_BENCHMARK_OUTPUT == "results/raw/full_control_ivqr_results.csv"
    assert FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE == 9
    assert PI_VALUES == [1.0, 0.5, 0.25, 0.10]
    assert TAUS == [0.25, 0.50, 0.75]
    assert DGPS == ["dgp1", "dgp2", "dgp3"]
    assert R_FAST == 10
    assert R_MAIN == 500
    assert R_FULL_CONTROL_BENCHMARK == 500
    assert DEFAULT_OUTPUT == "results/raw/main_simulation_results.csv"
    assert DEFAULT_ALPHA_GRID_SIZE == 9
    assert DEFAULT_DML_K_FOLDS == 3
    assert DEFAULT_N_JOBS == 6
    assert DEFAULT_QUANTREG_MAX_ITER == 1000
    assert K_FOLDS == 3


def test_get_oracle_control_indices_returns_true_active_support() -> None:
    np.testing.assert_array_equal(
        get_oracle_control_indices("dgp1", p=100),
        np.arange(10),
    )
    np.testing.assert_array_equal(
        get_oracle_control_indices("dgp2", p=100),
        np.arange(20),
    )
    np.testing.assert_array_equal(
        get_oracle_control_indices("dgp3", p=100),
        np.arange(10),
    )
    assert get_oracle_control_count("dgp2", p=100) == 20


def test_get_oracle_control_indices_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="Unknown DGP"):
        get_oracle_control_indices("bad_dgp", p=100)
    with pytest.raises(ValueError, match="requires at least 10 controls"):
        get_oracle_control_indices("dgp1", p=9)
    with pytest.raises(ValueError, match="requires at least 20 controls"):
        get_oracle_control_indices("dgp2", p=19)


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


def test_dgp2_is_denser_than_dgp1() -> None:
    dgp1_coefs = generate_coefficients("dgp1", p=30)
    dgp2_coefs = generate_coefficients("dgp2", p=30)

    assert np.count_nonzero(dgp2_coefs["beta"]) == 20
    assert np.count_nonzero(dgp2_coefs["gamma"]) == 20
    assert np.count_nonzero(dgp2_coefs["beta"]) > np.count_nonzero(
        dgp1_coefs["beta"]
    )
    assert np.count_nonzero(dgp2_coefs["gamma"]) > np.count_nonzero(
        dgp1_coefs["gamma"]
    )


def test_dgp2_coefficients_match_denser_sparse_design() -> None:
    coefs = generate_coefficients("dgp2", p=30)
    beta = coefs["beta"]
    gamma = coefs["gamma"]

    assert np.count_nonzero(beta) == 20
    assert np.count_nonzero(gamma) == 20

    for j in range(1, 21):
        assert beta[j - 1] == pytest.approx(0.5 / np.sqrt(j))
        assert gamma[j - 1] == pytest.approx(0.4 / np.sqrt(j))

    assert np.all(beta[20:] == 0.0)
    assert np.all(gamma[20:] == 0.0)


def test_dgp2_effective_sparsity_is_capped_by_p() -> None:
    coefs = generate_coefficients("dgp2", p=5)

    assert np.count_nonzero(coefs["beta"]) == 5
    assert np.count_nonzero(coefs["gamma"]) == 5


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


def test_dgp3_uses_heavy_tailed_marginals() -> None:
    design_dgp1 = Design(dgp="dgp1", n=20_000, p=20, pi=0.5, tau=0.5, rep=0, seed=123)
    design_dgp3 = Design(dgp="dgp3", n=20_000, p=20, pi=0.5, tau=0.5, rep=0, seed=123)

    data_dgp1 = generate_data(design_dgp1)
    data_dgp3 = generate_data(design_dgp3)

    assert data_dgp1.u is not None
    assert data_dgp3.u is not None
    assert data_dgp1.v is not None
    assert data_dgp3.v is not None

    u_tail_dgp1 = np.quantile(np.abs(data_dgp1.u), 0.95)
    u_tail_dgp3 = np.quantile(np.abs(data_dgp3.u), 0.95)
    v_tail_dgp1 = np.quantile(np.abs(data_dgp1.v), 0.95)
    v_tail_dgp3 = np.quantile(np.abs(data_dgp3.v), 0.95)

    assert u_tail_dgp3 > u_tail_dgp1
    assert v_tail_dgp3 > v_tail_dgp1


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


@pytest.mark.parametrize("pi", [1.0, 0.5, 0.25, 0.10])
def test_treatment_share_is_nondegenerate_for_dgp1(pi: float) -> None:
    design = Design(dgp="dgp1", n=2000, p=50, pi=pi, tau=0.5, rep=0, seed=123)

    data = generate_data(design)
    share = data.d.mean()

    assert 0.10 < share < 0.90


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


def test_outcome_formula_matches_structural_equation() -> None:
    x = np.array([[1.0, 2.0], [-1.0, 0.5], [0.0, -2.0]])
    beta = np.array([0.5, -0.25])
    u = np.array([-0.5, 0.0, 1.0])
    d = np.array([0, 1, 1])

    expected_y = 1.0 + x @ beta + u + d * (1.0 + u)

    np.testing.assert_allclose(_structural_outcome(x, beta, u, d), expected_y)


def test_treatment_formula_matches_latent_index() -> None:
    x = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
    z = np.array([1.0, -1.0, 0.5])
    gamma = np.array([0.5, -0.25])
    v = np.array([-0.75, 1.0, 0.0])
    pi = 0.5

    expected_d = (pi * z + x @ gamma + v > 0).astype(int)

    np.testing.assert_array_equal(
        _treatment_from_latent_index(x, z, gamma, v, pi),
        expected_d,
    )


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


def test_non_string_dgp_raises_value_error() -> None:
    rng = np.random.default_rng(123)

    with pytest.raises(ValueError):
        generate_coefficients(None, p=20)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        generate_errors(123, n=100, rho_uv=0.5, df=5, rng=rng)  # type: ignore[arg-type]


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
        generate_data(
            Design(dgp="dgp1", n=100, p=20, pi=-0.1, tau=0.5, rep=0, seed=123)
        )


def test_design_can_be_instantiated() -> None:
    design = Design(dgp="dgp1", n=250, p=200, pi=1.0, tau=0.5, rep=0, seed=123)

    assert design.dgp == "dgp1"
    assert design.n == 250
    assert design.p == 200
    assert design.pi == 1.0
    assert design.tau == 0.5
    assert design.rep == 0
    assert design.seed == 123


def test_median_effects_equal_one() -> None:
    assert true_alpha(0.5, "dgp1") == pytest.approx(1.0)
    assert true_alpha(0.5, "dgp2") == pytest.approx(1.0)
    assert true_alpha(0.5, "dgp3") == pytest.approx(1.0)


def test_dgp2_true_alpha_equals_dgp1() -> None:
    for tau in [0.25, 0.5, 0.75]:
        assert true_alpha(tau, "dgp1") == pytest.approx(true_alpha(tau, "dgp2"))


def test_dgp1_true_alpha_formula() -> None:
    assert true_alpha(0.25, "dgp1") == pytest.approx(1.0 + norm.ppf(0.25))
    assert true_alpha(0.75, "dgp1") == pytest.approx(1.0 + norm.ppf(0.75))


def test_dgp3_true_alpha_uses_scaled_student_t() -> None:
    scale = sqrt((5 - 2) / 5)

    assert true_alpha(0.25, "dgp3", df=5) == pytest.approx(
        1.0 + scale * t.ppf(0.25, df=5)
    )
    assert true_alpha(0.75, "dgp3", df=5) == pytest.approx(
        1.0 + scale * t.ppf(0.75, df=5)
    )


def test_dgp_names_are_case_insensitive() -> None:
    assert true_alpha(0.5, "DGP1") == pytest.approx(1.0)
    assert true_alpha(0.5, "Dgp3") == pytest.approx(1.0)


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.1])
def test_invalid_tau_raises_value_error(tau: float) -> None:
    with pytest.raises(ValueError):
        true_alpha(tau, "dgp1")


def test_true_alpha_invalid_dgp_raises_value_error() -> None:
    with pytest.raises(ValueError):
        true_alpha(0.5, "wrong_dgp")


@pytest.mark.parametrize("dgp", [None, 123])
def test_true_alpha_non_string_dgp_raises_value_error(dgp: object) -> None:
    with pytest.raises(ValueError):
        true_alpha(0.5, dgp)  # type: ignore[arg-type]


@pytest.mark.parametrize("df", [2, 1])
def test_invalid_student_t_degrees_of_freedom_raise_value_error(df: int) -> None:
    with pytest.raises(ValueError):
        true_alpha(0.5, "dgp3", df=df)


def test_true_alpha_returns_python_float() -> None:
    assert isinstance(true_alpha(0.25, "dgp1"), float)
    assert isinstance(true_alpha(0.25, "dgp3"), float)
