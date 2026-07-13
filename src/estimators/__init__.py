"""Estimator interfaces for the IVQR simulation package.

Estimator functions are imported lazily to avoid circular imports between
the estimator package and shared IVQR routines.
"""

from __future__ import annotations

from estimators.base import EstimationResult

__all__ = [
    "EstimationResult",
    "estimate_dml_ivqr",
    "estimate_oracle_ivqr",
    "estimate_post_selection_ivqr",
]


def __getattr__(name: str):
    if name == "estimate_dml_ivqr":
        from estimators.dml import estimate_dml_ivqr

        return estimate_dml_ivqr

    if name == "estimate_oracle_ivqr":
        from estimators.oracle import estimate_oracle_ivqr

        return estimate_oracle_ivqr

    if name == "estimate_post_selection_ivqr":
        from estimators.post_selection import estimate_post_selection_ivqr

        return estimate_post_selection_ivqr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

