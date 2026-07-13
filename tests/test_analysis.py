from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.data import COMMON_COLUMNS, IDENTIFIER_COLUMNS, load_all_results, validate_results
from analysis.figures import plot_coverage_vs_strength
from analysis.tables import make_performance_by_strength_table, summarize_performance


@pytest.fixture(scope="module")
def final_results() -> pd.DataFrame:
    return load_all_results()


def test_completed_files_load_and_validate(final_results: pd.DataFrame) -> None:
    assert len(final_results) == 216_000
    assert set(final_results["estimator"]) == {"oracle", "post_selection", "dml"}
    assert set(COMMON_COLUMNS).issubset(final_results.columns)
    assert not final_results.duplicated(IDENTIFIER_COLUMNS).any()
    counts = final_results.groupby(
        ["estimator", "dgp", "n", "p", "pi", "tau"]
    )["rep"].nunique()
    assert counts.eq(500).all()
    validate_results(final_results)


def _synthetic_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "estimator": ["oracle", "oracle"],
            "pi": [0.5, 0.5],
            "alpha_hat": [1.0, 3.0],
            "alpha_true": [2.0, 2.0],
            "covered": [True, False],
            "cr_length": [2.0, 4.0],
            "converged": [True, False],
        }
    )


def test_metric_calculations_are_exact() -> None:
    metrics = summarize_performance(_synthetic_results(), ["estimator"]).iloc[0]
    assert metrics["mean_estimate"] == pytest.approx(2.0)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["absolute_bias"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["estimate_sd"] == pytest.approx(np.sqrt(2.0))
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["average_cr_length"] == pytest.approx(3.0)
    assert metrics["median_cr_length"] == pytest.approx(3.0)
    assert metrics["valid_rate"] == pytest.approx(0.5)


def test_strength_table_has_expected_grouping() -> None:
    table = make_performance_by_strength_table(_synthetic_results())
    assert list(table[["pi", "estimator"]].itertuples(index=False, name=None)) == [
        (0.5, "Oracle IVQR")
    ]
    assert {"bias", "rmse", "coverage", "average_cr_length"}.issubset(table.columns)


def test_coverage_figure_smoke(tmp_path: Path) -> None:
    paths = plot_coverage_vs_strength(_synthetic_results(), tmp_path)
    assert {path.suffix for path in paths} == {".pdf", ".png"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
