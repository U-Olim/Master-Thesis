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
    sha256_canonical_lf_file,
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


def _test_final_run_metadata() -> dict[str, object]:
    return {
        "common": {
            "alpha_grid_min": -1.0,
            "alpha_grid_max": 3.0,
            "alpha_grid_size": 21,
            "base_seed": 12345,
            "critical_value_multiplier": 1.0,
        },
        "estimators": {
            "oracle": {"pixi_task": "final_oracle"},
            "post_selection": {
                "pixi_task": "final_post_selection",
                "selection_lasso_multiplier": 1.8,
            },
            "dml": {
                "pixi_task": "final_dml",
                "k_folds": 3,
                "quantile_penalty": 0.07,
                "quantile_solver": "highs-ipm",
                "ridge_alpha": 1.0,
            },
        },
    }


def _write_test_manifest(
    manifest: Path,
    source: Path,
    *,
    estimator: str = "oracle",
    hashed_source: Path | None = None,
    size_bytes: int | None = None,
) -> None:
    provenance_source = source if hashed_source is None else hashed_source
    payload = {
        "schema_version": 3,
        "number_of_files": 1,
        "total_rows": 1,
        "files": [
            {
                "estimator": estimator,
                "path": source.resolve().relative_to(PROJECT_ROOT).as_posix(),
                "filename": source.name,
                "rows": 1,
                "columns": 2,
                "size_bytes": (
                    provenance_source.stat().st_size
                    if size_bytes is None
                    else size_bytes
                ),
                "sha256_bytes": sha256_file(provenance_source),
                "sha256_canonical_lf": sha256_canonical_lf_file(
                    provenance_source
                ),
            }
        ],
        "final_run_metadata": _test_final_run_metadata(),
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")


def test_raw_manifest_verification_accepts_valid_file(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)

    verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"observed 1, supported version is 3"):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_wrong_number_of_files(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["number_of_files"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="number_of_files mismatch with files list"):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_wrong_total_rows(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["total_rows"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="total_rows mismatch with file entries"):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_nonobject_final_run_metadata(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["final_run_metadata"] = []
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="final_run_metadata must be an object"):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_missing_common_metadata(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["final_run_metadata"]["common"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"final_run_metadata\.common must be an object"):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_missing_estimators_metadata(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["final_run_metadata"]["estimators"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"final_run_metadata\.estimators must be an object"
    ):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_missing_canonical_metadata_key(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["final_run_metadata"]["estimators"]["dml"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"missing keys: \['dml'\]"):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_unexpected_metadata_key(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["final_run_metadata"]["estimators"]["extra"] = {}
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unexpected keys: \['extra'\]"):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_post_selection_multiplier_conflict(
    tmp_path: Path,
) -> None:
    source = tmp_path / "post_selection.csv"
    source.write_bytes(b"selection_lasso_multiplier,x\n1.2,1\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source, estimator="post_selection")

    with pytest.raises(
        ValueError, match=r"raw data has 1\.2, metadata has 1\.8"
    ):
        verify_raw_manifest(manifest, {"post_selection": source})


def test_raw_manifest_rejects_wrong_file_row_count(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"][0]["rows"] = 2
    payload["total_rows"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"row-count mismatch.*expected 2, observed 1"
    ):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_wrong_file_column_count(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"][0]["columns"] = 3
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"column-count mismatch.*expected 3, observed 2"
    ):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_duplicate_estimator_entry(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"].append(payload["files"][0].copy())
    payload["number_of_files"] = 2
    payload["total_rows"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="repeats estimator: oracle"):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_rejects_missing_estimator_entry(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle.csv"
    oracle.write_bytes(b"a,b\n1,2\n")
    dml = tmp_path / "dml.csv"
    dml.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, oracle)

    with pytest.raises(ValueError, match=r"missing estimator entries: \['dml'\]"):
        verify_raw_manifest(manifest, {"oracle": oracle, "dml": dml})


def test_raw_manifest_rejects_unexpected_estimator_entry(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    unexpected = payload["files"][0].copy()
    unexpected["estimator"] = "dml"
    payload["files"].append(unexpected)
    payload["number_of_files"] = 2
    payload["total_rows"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"unexpected estimator entries: \['dml'\]"):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_accepts_lf_and_crlf_equivalent_content(tmp_path: Path) -> None:
    lf_source = tmp_path / "lf.csv"
    lf_source.write_bytes(b"a,b\n1,2\n")
    crlf_source = tmp_path / "oracle.csv"
    crlf_source.write_bytes(b"a,b\r\n1,2\r\n")
    cr_source = tmp_path / "cr.csv"
    cr_source.write_bytes(b"a,b\r1,2\r")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, crlf_source, hashed_source=lf_source)

    assert sha256_file(lf_source) != sha256_file(crlf_source)
    canonical_hash = sha256_canonical_lf_file(lf_source, chunk_size=4)
    assert canonical_hash == sha256_canonical_lf_file(crlf_source, chunk_size=4)
    assert canonical_hash == sha256_canonical_lf_file(cr_source, chunk_size=4)
    with pytest.warns(UserWarning, match="canonical LF content matches"):
        verify_raw_manifest(manifest, {"oracle": crlf_source})


def test_raw_manifest_rejects_canonical_content_mutation(tmp_path: Path) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source)
    source.write_bytes(b"a,b\n1,9\n")

    with pytest.raises(ValueError, match="canonical LF SHA-256 mismatch"):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_accepts_size_mismatch_when_canonical_hash_matches(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oracle.csv"
    source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source, size_bytes=source.stat().st_size + 1)

    with pytest.warns(UserWarning, match="canonical LF content matches"):
        verify_raw_manifest(manifest, {"oracle": source})


def test_raw_manifest_verification_rejects_missing_file(tmp_path: Path) -> None:
    source = tmp_path / "missing.csv"
    provenance_source = tmp_path / "reference.csv"
    provenance_source.write_bytes(b"a,b\n1,2\n")
    manifest = tmp_path / "manifest.json"
    _write_test_manifest(manifest, source, hashed_source=provenance_source)

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
            "covered": covered if covered is not None else [False] * size,
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
