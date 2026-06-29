"""Tests for shared quantile-selection experiment helpers."""

import numpy as np
import pytest

from estimators.quantile_selection import build_alpha_anchors, union_selected_indices


def test_build_alpha_anchors_for_default_grid() -> None:
    np.testing.assert_allclose(build_alpha_anchors(-1.0, 3.0), [0.0, 1.0, 2.0])


def test_build_alpha_anchors_for_wide_grid() -> None:
    np.testing.assert_allclose(build_alpha_anchors(-2.0, 4.0), [-0.5, 1.0, 2.5])


@pytest.mark.parametrize(("alpha_min", "alpha_max"), [(1.0, 1.0), (2.0, 1.0)])
def test_build_alpha_anchors_rejects_invalid_bounds(
    alpha_min: float,
    alpha_max: float,
) -> None:
    with pytest.raises(ValueError, match="alpha_max must exceed"):
        build_alpha_anchors(alpha_min, alpha_max)


def test_union_selected_indices_sorts_and_removes_duplicates() -> None:
    selected = union_selected_indices(
        [np.array([3, 1]), np.array([1, 2]), np.array([], dtype=int)],
        total=5,
    )

    np.testing.assert_array_equal(selected, np.array([1, 2, 3]))


def test_union_selected_indices_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="out-of-range"):
        union_selected_indices([np.array([0, 4])], total=4)
