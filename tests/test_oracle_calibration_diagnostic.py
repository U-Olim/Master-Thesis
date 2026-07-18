from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_oracle_calibration.py"
SPEC = importlib.util.spec_from_file_location("diagnose_oracle_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


def test_requested_covariance_variants_are_explicit() -> None:
    configurations = {
        (variant.vcov, variant.kernel, variant.bandwidth)
        for variant in diagnostic.COVARIANCE_VARIANTS
    }
    assert configurations == {
        ("robust", "epa", "hsheather"),
        ("robust", "epa", "bofinger"),
        ("robust", "epa", "chamberlain"),
        ("iid", "epa", "hsheather"),
    }


def test_summary_uses_successful_replications_and_scipy_critical_value() -> None:
    frame = pd.DataFrame(
        {
            "dgp": ["dgp1"] * 3,
            "n": [500] * 3,
            "p": [200] * 3,
            "pi": [1.0] * 3,
            "tau": [0.5] * 3,
            "covariance_variant": ["robust_epa_hsheather"] * 3,
            "converged": [True, True, False],
            "wald": [1.0, 5.0, np.nan],
            "rejected": [False, True, np.nan],
        }
    )

    summary = diagnostic.summarize_replications(frame).iloc[0]

    assert summary["replications_requested"] == 3
    assert summary["replications_successful"] == 2
    assert summary["failures"] == 1
    assert summary["rejection_rate"] == pytest.approx(0.5)
    assert summary["implied_coverage"] == pytest.approx(0.5)
    assert summary["mean_wald"] == pytest.approx(3.0)
    assert summary["theoretical_chi2_q95"] == pytest.approx(
        diagnostic.chi2.ppf(0.95, df=1)
    )


def test_raw_results_output_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be written under results/raw"):
        diagnostic._safe_output_path(Path("results/raw/diagnostic.csv"))


def test_fixed_seed_diagnostic_summary_is_deterministic() -> None:
    design = diagnostic.Design("dgp1", 50, 5, 1.0, 0.5, rep=0, seed=2468)
    first = diagnostic.summarize_replications(
        diagnostic.run_diagnostic([design], iteration_warning_policy="use_if_valid")
    )
    second = diagnostic.summarize_replications(
        diagnostic.run_diagnostic([design], iteration_warning_policy="use_if_valid")
    )
    pd.testing.assert_frame_equal(first, second)
