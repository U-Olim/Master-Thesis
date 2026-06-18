"""Data-generating process helpers."""

if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    src_path = Path(__file__).resolve().parents[1]
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from dgp.designs import Design, SimData
    from dgp.generators import (
        generate_coefficients,
        generate_data,
        generate_errors,
        generate_x,
        make_covariance_matrix,
    )
    from dgp.true_parameters import true_alpha
else:
    from .designs import Design, SimData
    from .generators import (
        generate_coefficients,
        generate_data,
        generate_errors,
        generate_x,
        make_covariance_matrix,
    )
    from .true_parameters import true_alpha

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
