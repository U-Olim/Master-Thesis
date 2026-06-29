from pathlib import Path

import pandas as pd
import pytest

import reporting.summaries as summaries_module
from reporting.summaries import (
    GROUP_COLUMNS,
    RAW_UNIQUE_COLUMNS,
    SUMMARY_METRIC_COLUMNS,
    aggregate_results,
    aggregate_results_file,
    compare_result_files,
    incomplete_groups,
    load_raw_results,
    save_summary,
    validate_no_duplicate_raw_rows,
)
from tests.reporting.helpers import raw_results, row as result_row

def test_aggregate_results_returns_one_row_per_group() -> None:
    summary = aggregate_results(raw_results(), expected_replications=2)

    assert len(summary) == 2
    assert set(summary["estimator"]) == {"dml_ivqr", "post_selection_ivqr"}

def test_aggregate_results_preserves_group_columns() -> None:
    summary = aggregate_results(raw_results(), expected_replications=2)

    assert GROUP_COLUMNS == ("dgp", "n", "p", "pi", "tau", "estimator")
    assert all(column in summary.columns for column in GROUP_COLUMNS)
    assert all(column in summary.columns for column in SUMMARY_METRIC_COLUMNS)

def test_aggregate_results_includes_strict_and_valid_only_length_metrics() -> None:
    summary = aggregate_results(raw_results(), expected_replications=2)
    row = result_row(summary, "post_selection_ivqr")

    assert row["avg_cr_length"] == pytest.approx(1.5)
    assert row["avg_cr_length_valid_only"] == pytest.approx(1.5)
    assert row["avg_cr_hull_length"] == pytest.approx(1.8)

def test_aggregate_results_dml_metrics() -> None:
    summary = aggregate_results(raw_results(), expected_replications=2)
    row = result_row(summary, "dml_ivqr")

    assert row["replications"] == 2
    assert row["valid_estimates"] == 2
    assert row["bias"] == pytest.approx(0.0)
    assert row["rmse"] == pytest.approx(0.1)
    assert row["mae"] == pytest.approx(0.1)
    assert row["coverage"] == pytest.approx(1.0)
    assert row["avg_cr_length"] == pytest.approx(1.5)
    assert row["failure_rate"] == pytest.approx(0.0)
    assert row["non_convergence_rate"] == pytest.approx(0.0)
    assert row["boundary_rate"] == pytest.approx(0.5)
    assert row["alpha_hat_boundary_rate"] == pytest.approx(0.5)
    assert row["cr_boundary_hit_rate"] == pytest.approx(0.5)
    assert row["mean_failed_alpha_rate"] == pytest.approx(0.0)
    assert row["observed_replications"] == 2
    assert row["expected_replications"] == 2
    assert row["completion_rate"] == pytest.approx(1.0)

def test_aggregate_results_post_selection_metrics() -> None:
    summary = aggregate_results(raw_results(), expected_replications=2)
    row = result_row(summary, "post_selection_ivqr")

    assert row["replications"] == 2
    assert row["valid_estimates"] == 1
    assert row["failure_rate"] == pytest.approx(0.5)
    assert row["non_convergence_rate"] == pytest.approx(0.5)
    assert row["cr_empty_rate"] == pytest.approx(0.5)
    assert row["coverage"] == pytest.approx(0.0)
    assert row["coverage_valid_only"] == pytest.approx(0.0)
    assert row["alpha_hat_boundary_rate"] == pytest.approx(0.0)
    assert row["cr_boundary_hit_rate"] == pytest.approx(1.0)
    assert row["mean_failed_alpha_rate"] == pytest.approx(0.1)
    assert row["mean_selected_controls"] == pytest.approx(3.5)

def test_aggregate_results_ignores_status_failed_rows_for_performance_metrics() -> None:
    raw = pd.DataFrame(
        {
            "dgp": ["dgp1", "dgp1"],
            "n": [20, 20],
            "p": [25, 25],
            "pi": [1.0, 1.0],
            "tau": [0.5, 0.5],
            "rep": [0, 1],
            "estimator": ["full_control_ivqr", "full_control_ivqr"],
            "alpha_hat": [1.2, None],
            "alpha_true": [1.0, 1.0],
            "failed": [False, True],
            "status": ["ok", "failed"],
            "converged": [True, False],
            "cr_length": [2.0, None],
            "cr_covers_true": [True, None],
            "cr_empty": [False, True],
            "runtime_seconds": [0.1, None],
        }
    )

    summary = aggregate_results(raw, expected_replications=2)
    row = result_row(summary, "full_control_ivqr")

    assert row["valid_estimates"] == 1
    assert row["bias"] == pytest.approx(0.2)
    assert row["rmse"] == pytest.approx(0.2)
    assert row["coverage"] == pytest.approx(1.0)
    assert row["avg_cr_length"] == pytest.approx(2.0)
    assert row["failure_rate"] == pytest.approx(0.5)

def test_load_raw_results_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_results(tmp_path / "missing.csv")

def test_load_raw_results_missing_group_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    raw_results().drop(columns=["dgp"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required group columns"):
        load_raw_results(path)

def test_aggregate_results_file_writes_summary_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "summary.csv"
    raw_results().to_csv(input_path, index=False)

    summary = aggregate_results_file(
        input_path,
        output_path,
        expected_replications=2,
    )

    written = pd.read_csv(output_path)
    assert len(summary) == 2
    assert len(written) == 2
    assert set(written["estimator"]) == {"dml_ivqr", "post_selection_ivqr"}

def test_compare_result_files_labels_separate_and_combined_raw_files(
    tmp_path: Path,
) -> None:
    baseline = raw_results().loc[
        lambda df: df["estimator"] == "post_selection_ivqr"
    ].copy()
    wide = raw_results().copy()
    baseline_path = tmp_path / "baseline_post.csv"
    wide_path = tmp_path / "wide_combined.csv"
    baseline.to_csv(baseline_path, index=False)
    wide.to_csv(wide_path, index=False)

    comparison = compare_result_files(
        [baseline_path, wide_path],
        ["grid21_post", "grid31_wide"],
        expected_replications=2,
    )

    assert comparison["run_label"].tolist() == [
        "grid21_post",
        "grid31_wide",
        "grid31_wide",
    ]
    assert set(comparison["estimator"]) == {"dml_ivqr", "post_selection_ivqr"}
    assert "avg_cr_hull_length" in comparison.columns
    assert "mean_failed_alpha_rate" in comparison.columns

def test_incomplete_groups_returns_completion_rate_below_one() -> None:
    summary = aggregate_results(raw_results(), expected_replications=3)

    incomplete = incomplete_groups(summary)

    assert len(incomplete) == 2
    assert incomplete["completion_rate"].tolist() == pytest.approx([2 / 3, 2 / 3])

def test_incomplete_groups_empty_when_expected_replications_missing() -> None:
    summary = aggregate_results(raw_results(), expected_replications=None)

    incomplete = incomplete_groups(summary)

    assert incomplete.empty
    assert summary["completion_rate"].isna().all()

def test_aggregate_results_sorting_is_deterministic() -> None:
    raw = raw_results().iloc[[2, 3, 0, 1]].copy()

    summary = aggregate_results(raw, expected_replications=2)

    assert summary["estimator"].tolist() == ["dml_ivqr", "post_selection_ivqr"]

def test_aggregate_results_observed_replications_uses_unique_rep_values() -> None:
    raw = pd.concat([raw_results(), raw_results().iloc[[0]]], ignore_index=True)

    summary = aggregate_results(raw, expected_replications=2)
    dml = result_row(summary, "dml_ivqr")

    assert dml["replications"] == 3
    assert dml["observed_replications"] == 2
    assert dml["completion_rate"] == pytest.approx(1.0)

def test_aggregate_results_without_rep_column_uses_row_count() -> None:
    raw = raw_results().drop(columns=["rep"])

    summary = aggregate_results(raw, expected_replications=2)

    assert summary["observed_replications"].tolist() == [2, 2]

def test_aggregate_results_duplicate_raw_rows_raise_value_error() -> None:
    raw = raw_results().copy()
    raw["seed"] = [123, 124, 123, 124]
    duplicate = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        aggregate_results(duplicate)

def test_aggregate_results_different_rep_rows_are_not_duplicates() -> None:
    raw = raw_results().copy()
    raw["seed"] = [123, 124, 123, 124]
    extra = raw.iloc[[0]].copy()
    extra["rep"] = 2
    extra["seed"] = 125

    summary = aggregate_results(pd.concat([raw, extra], ignore_index=True))
    dml = result_row(summary, "dml_ivqr")

    assert dml["observed_replications"] == 3

def test_aggregate_results_without_seed_still_aggregates_artificial_data() -> None:
    summary = aggregate_results(raw_results(), expected_replications=2)

    assert len(summary) == 2

def test_incomplete_groups_returns_empty_if_column_missing() -> None:
    summary = aggregate_results(raw_results()).drop(columns=["completion_rate"])

    assert incomplete_groups(summary).empty

def test_summary_constants_and_public_api_are_immutable_and_complete() -> None:
    assert isinstance(GROUP_COLUMNS, tuple)
    assert isinstance(RAW_UNIQUE_COLUMNS, tuple)
    assert isinstance(SUMMARY_METRIC_COLUMNS, tuple)
    for name in summaries_module.__all__:
        assert hasattr(summaries_module, name)
    assert all(not name.startswith("_") for name in summaries_module.__all__)

def test_load_raw_results_rejects_directory_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a file"):
        load_raw_results(tmp_path)

def test_load_raw_results_missing_metric_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    raw_results().drop(columns=["alpha_hat"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_raw_results(path)

def test_validate_no_duplicate_raw_rows_validates_direct_input() -> None:
    with pytest.raises(TypeError):
        validate_no_duplicate_raw_rows([])  # type: ignore[arg-type]

    duplicate_columns = pd.concat(
        [raw_results(), raw_results()[["alpha_hat"]]],
        axis=1,
    )
    with pytest.raises(ValueError, match="duplicate columns"):
        validate_no_duplicate_raw_rows(duplicate_columns)

@pytest.mark.parametrize("expected_replications", [0, True, 1.5])
def test_aggregate_results_rejects_invalid_expected_replications(
    expected_replications,
) -> None:
    with pytest.raises(ValueError):
        aggregate_results(
            raw_results(),
            expected_replications=expected_replications,
        )

def test_aggregate_results_allows_completion_rate_above_one() -> None:
    summary = aggregate_results(raw_results(), expected_replications=1)

    assert summary["completion_rate"].tolist() == [2.0, 2.0]

def test_aggregate_results_empty_input_preserves_summary_schema() -> None:
    summary = aggregate_results(raw_results().iloc[0:0])
    expected_columns = list(GROUP_COLUMNS + SUMMARY_METRIC_COLUMNS) + [
        "expected_replications",
        "observed_replications",
        "completion_rate",
    ]

    assert summary.empty
    assert summary.columns.tolist() == expected_columns

@pytest.mark.parametrize(
    ("rep", "message"),
    [
        ("bad", "numeric"),
        (-1, "nonnegative"),
        (0.5, "integer-valued"),
    ],
)
def test_aggregate_results_rejects_malformed_rep_values(
    rep,
    message: str,
) -> None:
    raw = raw_results().copy()
    raw["rep"] = raw["rep"].astype(object)
    raw.loc[raw.index[0], "rep"] = rep

    with pytest.raises(ValueError, match=message):
        aggregate_results(raw)

def test_aggregate_results_ignores_missing_rep_values() -> None:
    raw = raw_results().copy()
    raw.loc[raw.index[0], "rep"] = None

    summary = aggregate_results(raw)

    assert result_row(summary, "dml_ivqr")["observed_replications"] == 1

def test_aggregate_results_duplicate_rep_ids_count_once() -> None:
    raw = raw_results().copy()
    raw.loc[raw["estimator"] == "dml_ivqr", "rep"] = 0

    summary = aggregate_results(raw)

    assert result_row(summary, "dml_ivqr")["observed_replications"] == 1

def test_save_summary_validates_input_and_paths(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        save_summary([], tmp_path / "summary.csv")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be a file path"):
        save_summary(pd.DataFrame(), tmp_path)

    parent_file = tmp_path / "parent"
    parent_file.write_text("file", encoding="utf-8")
    with pytest.raises(ValueError, match="parent must be a directory"):
        save_summary(pd.DataFrame(), parent_file / "summary.csv")

def test_save_summary_creates_nested_directory_and_round_trips(tmp_path: Path) -> None:
    summary = aggregate_results(raw_results(), expected_replications=2)
    path = tmp_path / "nested" / "summary.csv"

    save_summary(summary, path)

    pd.testing.assert_frame_equal(
        pd.read_csv(path),
        summary,
        check_dtype=False,
    )

def test_aggregate_results_file_propagates_validation_errors(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    raw_results().to_csv(input_path, index=False)

    with pytest.raises(ValueError):
        aggregate_results_file(input_path, expected_replications=0)

    output_directory = tmp_path / "output"
    output_directory.mkdir()
    with pytest.raises(ValueError, match="must be a file path"):
        aggregate_results_file(input_path, output_directory)

def test_incomplete_groups_validates_input_and_ignores_nonnumeric() -> None:
    with pytest.raises(TypeError):
        incomplete_groups([])  # type: ignore[arg-type]

    duplicate_columns = pd.DataFrame([[0.5, 0.6]], columns=["completion_rate"] * 2)
    with pytest.raises(ValueError, match="duplicate columns"):
        incomplete_groups(duplicate_columns)

    summary = pd.DataFrame({"completion_rate": ["bad", None, 0.5, 1.5]})
    incomplete = incomplete_groups(summary)

    assert incomplete.index.tolist() == [2]

def test_aggregate_results_validates_metric_columns() -> None:
    raw = raw_results().drop(columns=["alpha_hat"])

    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_results(raw)
