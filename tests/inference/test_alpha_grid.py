"""Tests for alpha-grid construction."""

import numpy as np
import pytest

from inference.alpha_grid import (
    DEFAULT_ALPHA_MAX,
    DEFAULT_ALPHA_MIN,
    DEFAULT_ALPHA_STEP,
    alpha_grid,
    default_alpha_grid,
)


def test_default_alpha_grid_matches_project_simulation_grid() -> None:
    grid = default_alpha_grid()

    assert len(grid) == 81
    assert grid[0] == pytest.approx(DEFAULT_ALPHA_MIN)
    assert grid[-1] == pytest.approx(DEFAULT_ALPHA_MAX)
    assert DEFAULT_ALPHA_MIN == -1.0
    assert DEFAULT_ALPHA_MAX == 3.0
    assert DEFAULT_ALPHA_STEP == pytest.approx(0.05)
    assert np.allclose(np.diff(grid), DEFAULT_ALPHA_STEP)


def test_alpha_grid_has_expected_length_and_endpoint() -> None:
    grid = alpha_grid(-2.0, 4.0, 0.01)

    assert len(grid) == 601
    assert grid[0] == pytest.approx(-2.0)
    assert grid[-1] == pytest.approx(4.0)


@pytest.mark.parametrize(("size", "step"), [(9, 0.5), (13, 1.0 / 3.0)])
def test_alpha_grid_supports_default_and_robustness_grid_sizes(
    size: int,
    step: float,
) -> None:
    grid = alpha_grid(-1.0, 3.0, step)

    assert len(grid) == size
    assert grid[0] == pytest.approx(-1.0)
    assert grid[-1] == pytest.approx(3.0)


def test_alpha_grid_appends_endpoint_for_non_dividing_step() -> None:
    grid = alpha_grid(0.0, 1.0, 0.3)

    assert np.all(np.diff(grid) > 0)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_min": np.nan, "alpha_max": 1.0, "step": 0.1},
        {"alpha_min": 0.0, "alpha_max": np.inf, "step": 0.1},
        {"alpha_min": 0.0, "alpha_max": 1.0, "step": np.nan},
        {"alpha_min": True, "alpha_max": 1.0, "step": 0.1},
        {"alpha_min": 0.0, "alpha_max": True, "step": 0.1},
        {"alpha_min": 0.0, "alpha_max": 1.0, "step": True},
        {"alpha_min": 1.0, "alpha_max": 1.0, "step": 0.1},
        {"alpha_min": 2.0, "alpha_max": 1.0, "step": 0.1},
        {"alpha_min": 0.0, "alpha_max": 1.0, "step": 0.0},
        {"alpha_min": 0.0, "alpha_max": 1.0, "step": -0.1},
    ],
)
def test_alpha_grid_rejects_invalid_inputs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        alpha_grid(**kwargs)


