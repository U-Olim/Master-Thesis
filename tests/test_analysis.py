import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.data import (
    CANONICAL_DML_RESULTS,
    CANONICAL_ORACLE_RESULTS,
    CANONICAL_POST_SELECTION_RESULTS,
    COMMON_COLUMNS,
    CURRENT_OUTPUT_COLUMN_COUNTS,
    HISTORICAL_ARTIFACT_COLUMN_COUNTS,
    IDENTIFIER_COLUMNS,
    PROJECT_ROOT,
    RAW_MANIFEST_PATH,
    RAW_MANIFEST_SCHEMA_VERSION,
    RAW_RESULT_FILES,
    load_all_results,
    load_dml_results,
    load_oracle_results,
    load_post_selection_results,
    sha256_file,
    validate_results,
    verify_raw_manifest,
)
from analysis.figures import plot_coverage_vs_strength
from analysis.tables import (
    format_confidence_region_components,
    make_performance_by_design_cell_table,
    make_performance_by_strength_table,
    summarize_performance,
)


def test_reporting_formats_disconnected_components_without_using_the_hull() -> None:
    assert format_confidence_region_components(
        "[[-1.0,-0.42],[0.18,1.36]]"
    ) == "[-1.000, -0.420] ∪ [0.180, 1.360]"
    assert format_confidence_region_components(None) is None


@pytest.fixture(scope="module")
def final_results() -> pd.DataFrame:
    return load_all_results()


def _artifact_frame(estimator: str = "oracle") -> pd.DataFrame:
    label = {
        "oracle": "oracle",
        "post_selection": "post_selection_ivqr",
        "dml": "dml_ivqr",
    }[estimator]
    return pd.DataFrame(
        {
            "dgp": ["dgp1"],
            "n": [40],
            "p": [5],
            "pi": [0.5],
            "tau": [0.5],
            "rep": [0],
            "estimator": [label],
        }
    )


def _write_test_manifest(
    manifest: Path, source: Path, *, estimator: str = "oracle"
) -> dict[str, object]:
    frame = pd.read_csv(source)
    natural_key = ["dgp", "n", "p", "pi", "tau", "rep", "estimator"]
    entry = {
        "estimator": estimator,
        "canonical_path": source.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "artifact_role": "validated_r500_thesis_result",
        "artifact_schema_name": f"test_{estimator}",
        "artifact_schema_version": 1,
        "current_code_column_count": CURRENT_OUTPUT_COLUMN_COUNTS[estimator],
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "column_names": list(frame.columns),
        "sha256": sha256_file(source),
        "file_size_bytes": source.stat().st_size,
        "replication_min": int(frame["rep"].min()),
        "replication_max": int(frame["rep"].max()),
        "unique_replications": int(frame["rep"].nunique()),
        "rows_per_replication": sorted(
            int(value) for value in frame.groupby("rep").size().unique()
        ),
        "natural_key": natural_key,
        "duplicate_natural_key_count": int(frame.duplicated(natural_key).sum()),
        "created_or_recorded_timestamp": "2026-07-21T00:00:00+00:00",
        "source_git_reference": "pre-refactor-r500",
    }
    payload: dict[str, object] = {
        "manifest_schema_version": RAW_MANIFEST_SCHEMA_VERSION,
        "artifact_set": "validated_r500_thesis_results",
        "artifact_policy": {"immutable": True},
        "number_of_files": 1,
        "total_rows": len(frame),
        "files": {estimator: entry},
        "recorded_run_settings": {
            "estimators": {
                "post_selection": {"selection_lasso_multiplier": 1.8}
            }
        },
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_canonical_path_constants_and_files() -> None:
    assert CANONICAL_ORACLE_RESULTS == Path("results/raw/oracle_ivqr.csv")
    assert CANONICAL_POST_SELECTION_RESULTS == Path(
        "results/raw/post_selection_ivqr.csv"
    )
    assert CANONICAL_DML_RESULTS == Path("results/raw/dml_ivqr.csv")
    assert all(path.is_file() for path in RAW_RESULT_FILES.values())


def test_canonical_manifest_matches_actual_artifacts() -> None:
    payload = json.loads(RAW_MANIFEST_PATH.read_text(encoding="utf-8"))
    verify_raw_manifest()
    for estimator, source in RAW_RESULT_FILES.items():
        entry = payload["files"][estimator]
        header = pd.read_csv(source, nrows=0)
        assert entry["canonical_path"] == source.relative_to(PROJECT_ROOT).as_posix()
        assert entry["sha256"] == sha256_file(source)
        assert entry["row_count"] == 72_000
        assert entry["column_count"] == len(header.columns)
        assert entry["column_names"] == list(header.columns)
        assert entry["replication_min"] == 0
        assert entry["replication_max"] == 499
        assert entry["unique_replications"] == 500
        assert entry["rows_per_replication"] == [144]
        assert entry["duplicate_natural_key_count"] == 0


def test_historical_and_current_schema_counts_are_distinct() -> None:
    assert HISTORICAL_ARTIFACT_COLUMN_COUNTS == {
        "oracle": 43,
        "post_selection": 52,
        "dml": 15,
    }
    assert CURRENT_OUTPUT_COLUMN_COUNTS == {
        "oracle": 26,
        "post_selection": 52,
        "dml": 43,
    }


@pytest.mark.parametrize(
    ("estimator", "loader"),
    [
        ("oracle", load_oracle_results),
        ("post_selection", load_post_selection_results),
        ("dml", load_dml_results),
    ],
)
def test_historical_schema_loads_without_rewriting(
    estimator: str, loader: Callable[[], pd.DataFrame]
) -> None:
    source = RAW_RESULT_FILES[estimator]
    before = (source.stat().st_size, source.stat().st_mtime_ns, sha256_file(source))
    loaded = loader()
    after = (source.stat().st_size, source.stat().st_mtime_ns, sha256_file(source))
    assert len(loaded) == 72_000
    assert before == after


def test_raw_manifest_verification_accepts_valid_file(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    _artifact_frame().to_csv(source, index=False)
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    _artifact_frame().to_csv(source, index=False)
    manifest = tmp_path / "manifest.json"
    payload = _write_test_manifest(manifest, source)
    payload["manifest_schema_version"] = 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"observed 1, supported version is 4"):
        verify_raw_manifest(manifest, {"oracle": source})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("row_count", 2, "row-count mismatch"),
        ("column_count", 8, "column-count mismatch"),
        ("column_names", ["wrong"], "column order mismatch"),
        ("sha256", "0" * 64, "SHA-256 mismatch"),
        ("replication_max", 1, "replication_max mismatch"),
        ("duplicate_natural_key_count", 1, "duplicate-key mismatch"),
    ],
)
def test_raw_manifest_rejects_stale_artifact_metadata(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    source = tmp_path / "oracle.csv"
    _artifact_frame().to_csv(source, index=False)
    manifest = tmp_path / "manifest.json"
    payload = _write_test_manifest(manifest, source)
    payload["files"]["oracle"][field] = value
    if field == "row_count":
        payload["total_rows"] = value
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_verification_rejects_missing_file(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    _artifact_frame().to_csv(source, index=False)
    manifest = tmp_path / "manifest.json"
    payload = _write_test_manifest(manifest, source)
    source.unlink()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"Canonical raw result.*oracle\.csv"):
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
            "coverage_status": ["covered", "not_covered"],
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


def _validation_results(
    *,
    alpha_hat: list[float],
    converged: list[bool],
    cr_lower: list[float] | None = None,
    cr_upper: list[float] | None = None,
    cr_length: list[float] | None = None,
    covered: list[bool] | None = None,
) -> pd.DataFrame:
    size = len(alpha_hat)
    covered_values = covered if covered is not None else [False] * size
    return pd.DataFrame(
        {
            "estimator": ["oracle"] * size,
            "dgp": ["dgp1"] * size,
            "n": [500] * size,
            "p": [200] * size,
            "pi": [0.5] * size,
            "tau": [0.5] * size,
            "rep": range(size),
            "seed": range(100, 100 + size),
            "alpha_hat": alpha_hat,
            "alpha_true": [0.0] * size,
            "cr_lower": cr_lower if cr_lower is not None else [np.nan] * size,
            "cr_upper": cr_upper if cr_upper is not None else [np.nan] * size,
            "cr_length": cr_length if cr_length is not None else [np.nan] * size,
            "covered": covered_values,
            "coverage_status": [
                "covered"
                if is_covered
                else ("not_covered" if is_converged else "unknown")
                for is_covered, is_converged in zip(
                    covered_values, converged, strict=True
                )
            ],
            "converged": converged,
        }
    )


def _validate_and_summarize(results: pd.DataFrame) -> pd.Series:
    validate_results(results, expected_replications=len(results))
    return summarize_performance(results, ["estimator"]).iloc[0]


def test_failed_row_with_missing_outputs_validates() -> None:
    results = _validation_results(alpha_hat=[np.nan], converged=[False])

    metrics = _validate_and_summarize(results)

    assert metrics["n_results"] == 1
    assert metrics["valid_rate"] == pytest.approx(0.0)


def test_successful_row_with_missing_estimate_is_rejected() -> None:
    results = _validation_results(alpha_hat=[np.nan], converged=[True])

    with pytest.raises(ValueError, match="Successful rows must have a finite alpha_hat"):
        validate_results(results, expected_replications=1)


def test_failed_row_is_in_denominator_but_not_point_metrics() -> None:
    results = _validation_results(
        alpha_hat=[1.0, -1.0, np.nan],
        converged=[True, True, False],
        cr_lower=[-1.0, -1.0, np.nan],
        cr_upper=[1.0, 1.0, np.nan],
        cr_length=[2.0, 2.0, np.nan],
        covered=[True, True, False],
    )

    metrics = _validate_and_summarize(results)

    assert metrics["n_results"] == 3
    assert metrics["valid_rate"] == pytest.approx(2 / 3)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["abs_bias"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)


def test_all_failed_rows_produce_nan_point_metrics() -> None:
    results = _validation_results(
        alpha_hat=[np.nan, np.nan], converged=[False, False]
    )

    metrics = _validate_and_summarize(results)

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


def test_failed_confidence_region_is_excluded_from_coverage() -> None:
    results = _validation_results(
        alpha_hat=[0.0, np.nan],
        converged=[True, False],
        cr_lower=[-1.0, np.nan],
        cr_upper=[1.0, np.nan],
        cr_length=[2.0, np.nan],
        covered=[True, False],
    )

    metrics = _validate_and_summarize(results)

    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["average_cr_length"] == pytest.approx(2.0)
    assert metrics["median_cr_length"] == pytest.approx(2.0)


def test_successful_row_with_partially_missing_confidence_region_is_rejected() -> None:
    results = _validation_results(
        alpha_hat=[0.0],
        converged=[True],
        cr_lower=[-1.0],
        cr_upper=[np.nan],
        cr_length=[np.nan],
    )

    with pytest.raises(ValueError, match="partially missing confidence regions"):
        validate_results(results, expected_replications=1)


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
            "coverage_status": ["covered", "coverage_unresolved", "unknown"],
            "converged": [True, True, False],
        }
    )

    metrics = summarize_performance(results, ["estimator"]).iloc[0]
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["average_cr_length"] == pytest.approx(2.0)
    assert metrics["median_cr_length"] == pytest.approx(2.0)


def test_coverage_summary_separates_resolved_unresolved_and_failures() -> None:
    results = pd.DataFrame(
        {
            "estimator": ["oracle"] * 4,
            "alpha_hat": [0.0, 0.0, 0.0, np.nan],
            "alpha_true": [0.0] * 4,
            "cr_lower": [-1.0, 1.0, np.nan, np.nan],
            "cr_upper": [1.0, 2.0, np.nan, np.nan],
            "cr_length": [2.0, 1.0, np.nan, np.nan],
            "covered": [True, False, False, False],
            "coverage_status": [
                "covered",
                "not_covered",
                "coverage_unresolved",
                "unknown",
            ],
            "converged": [True, True, True, False],
            "failed": [False, False, False, True],
        }
    )
    metrics = summarize_performance(results, ["estimator"]).iloc[0]
    assert metrics["coverage_conditional_on_resolved"] == pytest.approx(0.5)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["n_coverage_resolved"] == 2
    assert metrics["n_coverage_unresolved"] == 1
    assert metrics["coverage_unresolved_rate"] == pytest.approx(0.25)
    assert metrics["estimator_failure_rate"] == pytest.approx(0.25)


def test_coverage_summary_with_no_resolved_rows_is_nan() -> None:
    results = pd.DataFrame(
        {
            "estimator": ["oracle"],
            "alpha_hat": [0.0],
            "alpha_true": [0.0],
            "cr_lower": [np.nan],
            "cr_upper": [np.nan],
            "cr_length": [np.nan],
            "covered": [False],
            "coverage_status": ["coverage_unresolved"],
            "converged": [True],
        }
    )
    metrics = summarize_performance(results, ["estimator"]).iloc[0]
    assert np.isnan(metrics["coverage_conditional_on_resolved"])
    assert metrics["n_coverage_resolved"] == 0


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
