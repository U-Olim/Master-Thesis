from math import sqrt

import pytest
from scipy.stats import norm, t

from ivqr_sim.true_effects import true_alpha


def test_median_effects_equal_one() -> None:
    assert true_alpha(0.5, "dgp1") == pytest.approx(1.0)
    assert true_alpha(0.5, "dgp2") == pytest.approx(1.0)
    assert true_alpha(0.5, "dgp3") == pytest.approx(1.0)


def test_dgp1_and_dgp2_are_equal() -> None:
    for tau in [0.25, 0.5, 0.75]:
        assert true_alpha(tau, "dgp1") == pytest.approx(true_alpha(tau, "dgp2"))


def test_dgp1_values_use_normal_quantiles() -> None:
    assert true_alpha(0.25, "dgp1") == pytest.approx(1.0 + norm.ppf(0.25))
    assert true_alpha(0.75, "dgp1") == pytest.approx(1.0 + norm.ppf(0.75))


def test_dgp3_values_use_scaled_student_t_quantiles() -> None:
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


def test_invalid_dgp_raises_value_error() -> None:
    with pytest.raises(ValueError):
        true_alpha(0.5, "wrong_dgp")


@pytest.mark.parametrize("df", [2, 1])
def test_invalid_student_t_degrees_of_freedom_raise_value_error(df: int) -> None:
    with pytest.raises(ValueError):
        true_alpha(0.5, "dgp3", df=df)


def test_true_alpha_returns_python_float() -> None:
    assert isinstance(true_alpha(0.25, "dgp1"), float)
    assert isinstance(true_alpha(0.25, "dgp3"), float)
