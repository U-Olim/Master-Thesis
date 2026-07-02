"""IVQR grid, moment, and confidence-region helpers."""

from typing import Any

from ivqr.alpha_grid import alpha_grid
from ivqr.confidence_regions import (
    ConfidenceRegion,
    critical_value_chi_square,
    invert_score_test,
)

__all__ = [
    "ConfidenceRegion",
    "alpha_grid",
    "critical_value_chi_square",
    "estimate_ch_ivqr_controls",
    "evaluate_alpha_ch_ivqr",
    "invert_score_test",
]


def estimate_ch_ivqr_controls(*args: Any, **kwargs: Any) -> Any:
    """Lazily dispatch to the CH-IVQR controls estimator."""
    from ivqr.ch_inverse import estimate_ch_ivqr_controls as _estimate

    return _estimate(*args, **kwargs)


def evaluate_alpha_ch_ivqr(*args: Any, **kwargs: Any) -> Any:
    """Lazily dispatch to the CH-IVQR alpha evaluator."""
    from ivqr.ch_inverse import evaluate_alpha_ch_ivqr as _evaluate

    return _evaluate(*args, **kwargs)
