"""Inference, moment, and metric helpers."""

if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    src_path = Path(__file__).resolve().parents[1]
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from inference.alpha_grid import alpha_grid
    from inference.confidence_regions import (
        ConfidenceRegion,
        FAILED_ALPHA_STATISTIC,
        argmin_grid,
        critical_value_chi_square,
        invert_score_test,
        is_disconnected_region,
        sanitize_grid_statistics,
        summarize_region,
    )
else:
    from .alpha_grid import alpha_grid
    from .confidence_regions import (
        ConfidenceRegion,
        FAILED_ALPHA_STATISTIC,
        argmin_grid,
        critical_value_chi_square,
        invert_score_test,
        is_disconnected_region,
        sanitize_grid_statistics,
        summarize_region,
    )

__all__ = [
    "ConfidenceRegion",
    "FAILED_ALPHA_STATISTIC",
    "alpha_grid",
    "argmin_grid",
    "critical_value_chi_square",
    "invert_score_test",
    "is_disconnected_region",
    "sanitize_grid_statistics",
    "summarize_region",
]
