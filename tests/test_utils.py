# Consolidated tests for validation utilities.

import numpy as np
import pytest

from utils.validation import (
    check_finite,
    validate_1d_array,
    validate_2d_array,
    validate_alpha_grid,
    validate_data_arrays,
    validate_k_folds,
    validate_nonempty_sequence,
    validate_positive_int,
    validate_tau,
)


def test_validate_tau_accepts_open_unit_interval() -> None:
    assert validate_tau(0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("tau", [0.0, 1.0, -0.1, 1.1])
def test_validate_tau_rejects_invalid_values(tau: float) -> None:
    with pytest.raises(ValueError, match="tau must satisfy"):
        validate_tau(tau)


def test_validate_alpha_grid_accepts_strictly_increasing_grid() -> None:
    grid = validate_alpha_grid([0.0, 0.5, 1.0])

    assert isinstance(grid, np.ndarray)
    assert grid.shape == (3,)
    assert np.allclose(grid, [0.0, 0.5, 1.0])


@pytest.mark.parametrize(
    "alphas",
    [
        [],
        [[0.0, 1.0]],
        [0.0, np.nan],
        [0.0, 0.0],
        [1.0, 0.0],
    ],
)
def test_validate_alpha_grid_rejects_invalid_grids(alphas: object) -> None:
    with pytest.raises(ValueError):
        validate_alpha_grid(alphas)


def test_validate_1d_array_returns_finite_float_array() -> None:
    array = validate_1d_array("x", [1, 2, 3], length=3)

    assert array.dtype == float
    assert array.shape == (3,)


def test_validate_1d_array_rejects_wrong_shape_length_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_1d_array("x", [[1.0]])
    with pytest.raises(ValueError, match="length 2"):
        validate_1d_array("x", [1.0], length=2)
    with pytest.raises(ValueError, match="finite"):
        validate_1d_array("x", [1.0, np.inf])


def test_validate_2d_array_returns_finite_float_matrix() -> None:
    matrix = validate_2d_array("x", [[1, 2], [3, 4]], n_rows=2)

    assert matrix.dtype == float
    assert matrix.shape == (2, 2)


def test_validate_2d_array_rejects_wrong_shape_rows_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_2d_array("x", [1.0])
    with pytest.raises(ValueError, match="2 rows"):
        validate_2d_array("x", [[1.0]], n_rows=2)
    with pytest.raises(ValueError, match="finite"):
        validate_2d_array("x", [[1.0, np.nan]])


def test_validate_data_arrays_without_instrument_checks_lengths() -> None:
    y, d, x = validate_data_arrays([1, 2], [0, 1], [[1], [2]])

    assert y.shape == (2,)
    assert d.shape == (2,)
    assert x.shape == (2, 1)


def test_validate_data_arrays_with_instrument_checks_lengths() -> None:
    y, d, z, x = validate_data_arrays([1, 2], [0, 1], [[1], [2]], [0.5, -0.5])

    assert y.shape == (2,)
    assert d.shape == (2,)
    assert z.shape == (2,)
    assert x.shape == (2, 1)


def test_validate_data_arrays_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="consistent row counts"):
        validate_data_arrays([1, 2], [0, 1], [[1]])
    with pytest.raises(ValueError, match="consistent row counts"):
        validate_data_arrays([1, 2], [0, 1], [[1], [2]], [0.5])


def test_check_finite_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        check_finite("x", np.array([1.0, np.nan]))


def test_validate_positive_int() -> None:
    assert validate_positive_int("n", 1) == 1
    with pytest.raises(ValueError, match="integer"):
        validate_positive_int("n", True)
    with pytest.raises(ValueError, match="positive"):
        validate_positive_int("n", 0)


def test_validate_k_folds() -> None:
    assert validate_k_folds(3, 10) == 3
    with pytest.raises(ValueError, match="greater than 1"):
        validate_k_folds(2, 1)
    with pytest.raises(ValueError, match="2 <= k_folds <= n"):
        validate_k_folds(1, 10)
    with pytest.raises(ValueError, match="2 <= k_folds <= n"):
        validate_k_folds(11, 10)


def test_validate_nonempty_sequence() -> None:
    value = [1]
    assert validate_nonempty_sequence("items", value) is value
    with pytest.raises(ValueError, match="nonempty"):
        validate_nonempty_sequence("items", [])
