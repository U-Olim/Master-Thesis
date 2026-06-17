"""Data-generating process helpers."""

from dgp.designs import Design, SimData
from dgp.generators import (
    generate_coefficients,
    generate_data,
    generate_errors,
    generate_x,
    make_covariance_matrix,
)
from dgp.true_parameters import true_alpha

__all__ = [
    "Design",
    "SimData",
    "generate_coefficients",
    "generate_data",
    "generate_errors",
    "generate_x",
    "make_covariance_matrix",
    "true_alpha",
]
