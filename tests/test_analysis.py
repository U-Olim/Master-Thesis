import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.data import (
    COMMON_COLUMNS,
    IDENTIFIER_COLUMNS,
    PROJECT_ROOT,
    load_all_results,
    validate_results,
    verify_raw_manifest,
)
from analysis.figures import plot_coverage_vs_strength
from analysis.tables import (
    make_performance_by_design_cell_table,
    make_performance_by_strength_table,
    summarize_performance,
)


@pytest.fixture(scope="module")
def final_results() -> pd.DataFrame:
    return load_all_results()


def _write_test_manifest(manifest: Path, source: Path, sha256: str) -> None:
    payload = {
        "schema_version": 1,
        "number_of_files": 1,
        "total_rows": 1,
        "files": [
            {
                "estimator": "oracle",
                "path": source.resolve().relative_to(PROJECT_ROOT).as_posix(),
                "filename": source.name,
                "rows": 1,
                "columns": 2,
                "size_bytes": source.stat().st_size if source.exists() else 0,
                "sha256": sha256,
            }
        ],
        "final_run_metadata": {},
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")


def test_raw_manifest_verification_accepts_valid_file(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source, digest)

    verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_verification_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source, "0" * 64)

    with pytest.raises(ValueError, match=r"SHA-256 mismatch.*expected 0000.*observed"):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_verification_rejects_missing_file(tmp_path: Path) -> None:
    source = tmp_path / "missing.csv"
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source, "0" * 64)

    with pytest.raises(FileNotFoundError, match=r"Canonical raw result.*missing\.csv"):
        verify_raw_manifest(manifest, {"oracle": source})


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
            "dgp": ["dgp1", "dgp1"],
            "n": [500, 500],
            "p": [200, 200],
            "pi": [0.5, 0.5],
            "tau": [0.5, 0.5],
            "alpha_hat": [1.0, 3.0],
            "alpha_true": [2.0, 2.0],
            "cr_lower": [1.0, 0.0],
            "cr_upper": [3.0, 4.0],
            "covered": [True, False],
            "cr_length": [2.0, 4.0],
            "converged": [True, True],
        }
    )


def _post_selection_validation_results(multipliers: list[float]) -> pd.DataFrame:
    size = len(multipliers)
    return pd.DataFrame(
        {
            "estimator": ["post_selection"] * size,
            "dgp": ["dgp1"] * size,
            "n": [500] * size,
            "p": [200] * size,
            "pi": [0.5] * size,
            "tau": [0.5] * size,
            "rep": range(size),
            "seed": range(100, 100 + size),
            "alpha_hat": [1.0] * size,
            "alpha_true": [1.0] * size,
            "cr_lower": [0.5] * size,
            "cr_upper": [1.5] * size,
            "cr_length": [1.0] * size,
            "covered": [True] * size,
            "converged": [True] * size,
            "n_selected_controls": [5] * size,
            "selection_lasso_multiplier": multipliers,
        }
    )


def test_single_selection_lasso_multiplier_is_valid() -> None:
    results = _post_selection_validation_results([1.8, 1.8, 1.8])
    validate_results(results, expected_replications=3)


def test_conflicting_selection_lasso_multipliers_are_rejected() -> None:
    results = _post_selection_validation_results([1.8, 1.8, 1.2])
    with pytest.raises(ValueError, match=r"found 2: \[1\.2, 1\.8\]"):
        validate_results(results, expected_replications=3)


def test_missing_selection_lasso_multipliers_are_rejected() -> None:
    results = _post_selection_validation_results([np.nan, np.nan])
    with pytest.raises(ValueError, match=r"found 0: \[\]"):
        validate_results(results, expected_replications=2)


def test_metric_calculations_are_exact() -> None:
    metrics = summarize_performance(_synthetic_results(), ["estimator"]).iloc[0]
    assert metrics["mean_estimate"] == pytest.approx(2.0)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["abs_bias"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["estimate_sd"] == pytest.approx(np.sqrt(2.0))
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["average_cr_length"] == pytest.approx(3.0)
    assert metrics["median_cr_length"] == pytest.approx(3.0)
    assert metrics["valid_rate"] == pytest.approx(1.0)


def test_failed_estimate_does_not_poison_point_metrics() -> None:
    results = pd.DataFrame(
        {
            "estimator": ["oracle"] * 3,
            "alpha_hat": [1.0, -1.0, np.nan],
            "alpha_true": [0.0, 0.0, 0.0],
            "cr_lower": [-1.0, -1.0, np.nan],
            "cr_upper": [1.0, 1.0, np.nan],
            "cr_length": [2.0, 2.0, np.nan],
            "covered": [True, True, False],
            "converged": [True, True, False],
        }
    )

    metrics = summarize_performance(results, ["estimator"]).iloc[0]
    assert metrics["n_results"] == 3
    assert metrics["valid_rate"] == pytest.approx(2 / 3)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["abs_bias"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)


def test_all_invalid_point_estimates_produce_nan_metrics() -> None:
    results = pd.DataFrame(
        {
            "estimator": ["oracle", "oracle"],
            "alpha_hat": [0.0, np.inf],
            "alpha_true": [np.nan, 0.0],
            "cr_lower": [np.nan, np.nan],
            "cr_upper": [np.nan, np.nan],
            "cr_length": [np.nan, np.nan],
            "covered": [False, False],
            "converged": [True, True],
        }
    )

    metrics = summarize_performance(results, ["estimator"]).iloc[0]
    assert metrics["n_results"] == 2
    assert metrics["valid_rate"] == pytest.approx(0.0)
    for column in (
        "mean_estimate",
        "bias",
        "abs_bias",
        "mae",
        "rmse",
        "estimate_sd",
    ):
        assert np.isnan(metrics[column])


def test_invalid_confidence_region_does_not_count_as_noncoverage() -> None:
    results = pd.DataFrame(
        {
            "estimator": ["oracle"] * 3,
            "alpha_hat": [0.0, 0.0, 0.0],
            "alpha_true": [0.0, 0.0, 0.0],
            "cr_lower": [-1.0, np.nan, -1.0],
            "cr_upper": [1.0, np.nan, 1.0],
            "cr_length": [2.0, np.nan, 2.0],
            "covered": [True, False, False],
            "converged": [True, True, False],
        }
    )

    metrics = summarize_performance(results, ["estimator"]).iloc[0]
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["average_cr_length"] == pytest.approx(2.0)
    assert metrics["median_cr_length"] == pytest.approx(2.0)


def test_nonzero_bias_and_mae_are_distinct_metrics() -> None:
    results = _synthetic_results().assign(
        alpha_true=[0.0, 0.0], alpha_hat=[1.0, 3.0]
    )
    metrics = summarize_performance(results, ["estimator"]).iloc[0]
    assert metrics["bias"] == pytest.approx(2.0)
    assert metrics["abs_bias"] == pytest.approx(2.0)
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5.0))


def test_design_cell_table_preserves_all_grouping_dimensions() -> None:
    results = _synthetic_results().iloc[[0]].copy()
    results = pd.concat(
        [results, results.assign(pi=1.0, tau=0.75)], ignore_index=True
    )
    table = make_performance_by_design_cell_table(results)
    keys = ["dgp", "n", "p", "pi", "tau", "estimator"]
    assert len(table) == 2
    assert not table.duplicated(keys).any()
    assert set(table["pi"]) == {0.5, 1.0}
    assert set(table["tau"]) == {0.5, 0.75}


def test_strength_table_has_expected_grouping() -> None:
    table = make_performance_by_strength_table(_synthetic_results())
    assert list(table[["pi", "estimator"]].itertuples(index=False, name=None)) == [
        (0.5, "Oracle IVQR")
    ]
    assert {"bias", "abs_bias", "mae", "rmse", "coverage", "average_cr_length"}.issubset(
        table.columns
    )


def test_coverage_figure_smoke(tmp_path: Path) -> None:
    paths = plot_coverage_vs_strength(_synthetic_results(), tmp_path)
    assert {path.suffix for path in paths} == {".pdf", ".png"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
