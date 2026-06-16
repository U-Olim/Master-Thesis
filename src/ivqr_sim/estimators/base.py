"""Common estimator result objects."""

from dataclasses import dataclass


@dataclass
class EstimationResult:
    """Standard result object returned by every estimator.

    All estimator implementations must return this object so simulation,
    aggregation, and reporting code can consume results uniformly.
    """

    estimator: str
    alpha_hat: float | None
    alpha_true: float
    tau: float

    converged: bool
    failed: bool
    message: str

    objective_value: float | None
    at_grid_boundary: bool

    cr_lower: float | None
    cr_upper: float | None
    cr_length: float | None
    cr_covers_true: bool | None
    cr_empty: bool
    cr_disconnected: bool | None

    selected_controls: int | None
    runtime_seconds: float
