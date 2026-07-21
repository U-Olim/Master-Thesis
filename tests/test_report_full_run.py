from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "report_full_run.py"
SPEC = importlib.util.spec_from_file_location("report_full_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report_module
SPEC.loader.exec_module(report_module)
standardize_estimator = report_module.standardize_estimator
summarize = report_module.summarize


def _common_frame(size: int, estimator: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dgp": ["dgp1"] * size,
            "n": [500] * size,
            "p": [200] * size,
            "pi": [0.5] * size,
            "tau": [0.5] * size,
            "rep": range(size),
            "seed": range(100, 100 + size),
            "estimator": [estimator] * size,
            "alpha_hat": [0.0] * size,
            "alpha_true": [0.0] * size,
            "converged": [True] * size,
        }
    )


def test_explicit_empty_is_resolved_uncovered_but_unresolved_is_excluded() -> None:
    source = _common_frame(4, "oracle").assign(
        cr_lower=[-1.0, 1.0, np.nan, -1.0],
        cr_upper=[1.0, 2.0, np.nan, 1.0],
        cr_length=[2.0, 1.0, np.nan, 2.0],
        covered=[True, False, False, True],
        cr_components=["[[-1.0,1.0]]", "[[1.0,2.0]]", "[]", "[[-1.0,1.0]]"],
        cr_n_blocks=[1, 1, 0, 1],
        cr_disconnected=[False, False, False, False],
        cr_status=["valid", "valid", "empty_valid", "partially_unresolved"],
        cr_is_numerically_resolved=[True, True, True, False],
    )

    standardized = standardize_estimator(source, "Oracle IVQR")
    overall = summarize(standardized, ["estimator_label"]).iloc[0]

    assert standardized["coverage_status"].tolist() == [
        "covered",
        "not_covered",
        "not_covered",
        "unresolved",
    ]
    assert standardized.loc[2, "cr_length_analysis"] == pytest.approx(0.0)
    assert overall["resolved_replications"] == 3
    assert overall["unresolved_replications"] == 1
    assert overall["empirical_coverage"] == pytest.approx(1 / 3)
    assert overall["mean_cr_length"] == pytest.approx(1.0)


def test_dml_missing_cr_is_unresolved_unknown_and_zero_length_is_not_empty() -> None:
    source = _common_frame(3, "dml_ivqr").assign(
        cr_lower=[-1.0, 1.0, np.nan],
        cr_upper=[1.0, 1.0, np.nan],
        cr_length=[2.0, 0.0, np.nan],
        covered=[True, False, False],
    )

    standardized = standardize_estimator(source, "DML-IVQR")
    overall = summarize(standardized, ["estimator_label"]).iloc[0]

    assert standardized["resolved"].tolist() == [True, True, False]
    assert standardized["cr_status_standardized"].tolist() == [
        "observed_status_unavailable",
        "observed_status_unavailable",
        "missing_status_unavailable",
    ]
    assert standardized["empty_region"].isna().all()
    assert standardized["disconnected_region"].isna().all()
    assert standardized["coverage_status"].tolist() == [
        "covered",
        "not_covered",
        "unresolved",
    ]
    assert overall["resolved_replications"] == 2
    assert overall["empirical_coverage"] == pytest.approx(0.5)
    assert np.isnan(overall["empty_region_rate"])


def test_empty_valid_cannot_claim_coverage_without_a_component() -> None:
    source = _common_frame(1, "oracle").assign(
        cr_lower=[np.nan],
        cr_upper=[np.nan],
        cr_length=[np.nan],
        covered=[True],
        cr_components=["[]"],
        cr_n_blocks=[0],
        cr_disconnected=[False],
        cr_status=["empty_valid"],
        cr_is_numerically_resolved=[True],
    )

    with pytest.raises(ValueError, match="covered flag conflicts"):
        standardize_estimator(source, "Oracle IVQR")
