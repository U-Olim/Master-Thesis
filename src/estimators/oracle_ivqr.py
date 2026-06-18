"""Oracle/full-control IVQR estimator alias.

The current simulation design treats the full-control IVQR estimator as the
oracle benchmark because it uses all controls rather than a selected subset.
"""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    src_path = Path(__file__).resolve().parents[1]
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from estimators.full_ivqr import add_intercept
from estimators.full_ivqr import estimate_full_ivqr as estimate_oracle_ivqr

__all__ = ["add_intercept", "estimate_oracle_ivqr"]
