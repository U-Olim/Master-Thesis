"""Oracle/full-control IVQR estimator alias.

The current simulation design treats the full-control IVQR estimator as the
oracle benchmark because it uses all controls rather than a selected subset.
"""

from estimators.full_ivqr import add_intercept
from estimators.full_ivqr import estimate_full_ivqr as estimate_oracle_ivqr

__all__ = ["add_intercept", "estimate_oracle_ivqr"]
