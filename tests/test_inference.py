import numpy as np
import pytest

from ivqr_sim.inference import (
    argmin_grid,
    critical_value_chi_square,
    invert_score_test,
    is_disconnected_region,
    summarize_region,
)


def test_critical_value_chi_square_default_scalar_score() -> None:
    cv = critical_value_chi_square(level=0.95, df=1)

    assert cv == pytest.approx(3.841458820694124, rel=1e-6)


@pytest.mark.parametrize("level", [0.0, -0.1, 1.0, 1.1])
def test_critical_value_chi_square_validates_level(level: float) -> None:
    with pytest.raises(ValueError):
        critical_value_chi_square(level=level, df=1)


@pytest.mark.parametrize("df", [0, 1.5])
def test_critical_value_chi_square_validates_df(df: int) -> None:
    with pytest.raises(ValueError):
        critical_value_chi_square(level=0.95, df=df)  # type: ignore[arg-type]


def test_argmin_grid_returns_interior_minimum() -> None:
    alphas = np.array([-1.0, 0.0, 1.0])
    stats = np.array([4.0, 1.0, 3.0])

    alpha_hat, min_stat, at_boundary = argmin_grid(alphas, stats)

    assert alpha_hat == pytest.approx(0.0)
    assert min_stat == pytest.approx(1.0)
    assert at_boundary is False


def test_argmin_grid_reports_boundary_minimum() -> None:
    alphas = np.array([-1.0, 0.0, 1.0])
    stats = np.array([1.0, 2.0, 3.0])

    alpha_hat, min_stat, at_boundary = argmin_grid(alphas, stats)

    assert alpha_hat == pytest.approx(-1.0)
    assert min_stat == pytest.approx(1.0)
    assert at_boundary is True


def test_invert_score_test_connected_region_includes_critical_boundary() -> None:
    alphas = np.array([-2, -1, 0, 1, 2], dtype=float)
    stats = np.array([10, 4, 1, 4, 10], dtype=float)

    region = invert_score_test(alphas, stats, critical_value=4.0, alpha_true=0.0)

    assert np.allclose(region.selected_grid, np.array([-1.0, 0.0, 1.0]))
    assert region.lower == pytest.approx(-1.0)
    assert region.upper == pytest.approx(1.0)
    assert region.length == pytest.approx(2.0)
    assert region.empty is False
    assert region.disconnected is False
    assert region.covers_true is True


def test_invert_score_test_empty_region() -> None:
    alphas = np.array([-1.0, 0.0, 1.0])
    stats = np.array([10.0, 10.0, 10.0])

    region = invert_score_test(alphas, stats, critical_value=1.0, alpha_true=0.0)

    assert region.empty is True
    assert region.lower is None
    assert region.upper is None
    assert region.length is None
    assert len(region.selected_grid) == 0
    assert region.disconnected is False
    assert region.covers_true is False


def test_invert_score_test_disconnected_region_uses_convex_hull_length() -> None:
    alphas = np.array([0, 1, 2, 3, 4], dtype=float)
    stats = np.array([1, 1, 10, 1, 1], dtype=float)

    region = invert_score_test(alphas, stats, critical_value=2.0)

    assert np.allclose(region.selected_grid, np.array([0.0, 1.0, 3.0, 4.0]))
    assert region.lower == pytest.approx(0.0)
    assert region.upper == pytest.approx(4.0)
    assert region.length == pytest.approx(4.0)
    assert region.disconnected is True


def test_invert_score_test_coverage_false() -> None:
    alphas = np.array([0, 1, 2, 3, 4], dtype=float)
    stats = np.array([1, 1, 10, 1, 1], dtype=float)

    region = invert_score_test(alphas, stats, critical_value=2.0, alpha_true=10.0)

    assert region.covers_true is False


def test_is_disconnected_region_examples() -> None:
    full_grid = np.array([0, 1, 2, 3, 4], dtype=float)

    assert is_disconnected_region(np.array([0, 1, 4], dtype=float), full_grid) is True
    assert is_disconnected_region(np.array([1, 2, 3], dtype=float), full_grid) is False
    assert is_disconnected_region(np.array([], dtype=float), full_grid) is False


@pytest.mark.parametrize(
    ("alphas", "stats", "critical_value"),
    [
        (np.array([0.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0]), 1.0),
        (np.array([0.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]), 1.0),
        (np.array([0.0, 1.0]), np.array([1.0, 2.0, 3.0]), 1.0),
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, np.inf, 3.0]), 1.0),
        (np.array([0.0, np.nan, 2.0]), np.array([1.0, 2.0, 3.0]), 1.0),
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]), 0.0),
        (np.array([[0.0, 1.0, 2.0]]), np.array([1.0, 2.0, 3.0]), 1.0),
        (np.array([0.0, 1.0, 2.0]), np.array([[1.0, 2.0, 3.0]]), 1.0),
    ],
)
def test_invert_score_test_validates_inputs(
    alphas: np.ndarray,
    stats: np.ndarray,
    critical_value: float,
) -> None:
    with pytest.raises(ValueError):
        invert_score_test(alphas, stats, critical_value=critical_value)


@pytest.mark.parametrize(
    ("alphas", "stats"),
    [
        (np.array([0.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([0.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([0.0, 1.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([0.0, 1.0, 2.0]), np.array([1.0, np.nan, 3.0])),
        (np.array([0.0, np.inf, 2.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([[0.0, 1.0, 2.0]]), np.array([1.0, 2.0, 3.0])),
        (np.array([0.0, 1.0, 2.0]), np.array([[1.0, 2.0, 3.0]])),
    ],
)
def test_argmin_grid_validates_inputs(alphas: np.ndarray, stats: np.ndarray) -> None:
    with pytest.raises(ValueError):
        argmin_grid(alphas, stats)


def test_summarize_region_returns_estimation_result_fields() -> None:
    alphas = np.array([-2, -1, 0, 1, 2], dtype=float)
    stats = np.array([10, 4, 1, 4, 10], dtype=float)
    region = invert_score_test(alphas, stats, critical_value=4.0, alpha_true=0.0)

    summary = summarize_region(region)

    assert summary == {
        "cr_lower": pytest.approx(-1.0),
        "cr_upper": pytest.approx(1.0),
        "cr_length": pytest.approx(2.0),
        "cr_empty": False,
        "cr_disconnected": False,
        "cr_covers_true": True,
    }
