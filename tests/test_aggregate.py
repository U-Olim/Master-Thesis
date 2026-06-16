from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ivqr_sim.simulation.aggregate import (
    GROUP_COLUMNS,
    SUMMARY_METRIC_COLUMNS,
    aggregate_results,
    aggregate_results_file,
    incomplete_groups,
    load_raw_results,
)


def _raw_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dgp": ["dgp1", "dgp1", "dgp1", "dgp1"],
            "n": [80, 80, 80, 80],
            "p": [5, 5, 5, 5],
            "pi": [1.0, 1.0, 1.0, 1.0],
            "tau": [0.5, 0.5, 0.5, 0.5],
            "rep": [0, 1, 0, 1],
            "estimator": [
                "dml_ivqr",
                "dml_ivqr",
                "post_selection_ivqr",
                "post_selection_ivqr",
            ],
            "alpha_hat": [1.1, 0.9, 1.2, None],
            "alpha_true": [1.0, 1.0, 1.0, 1.0],
            "failed": [False, False, False, True],
            "converged": [True, True, True, False],
            "cr_length": [1.0, 2.0, 1.5, None],
            "cr_covers_true": [True, True, False, None],
            "cr_empty": [False, False, False, True],
            "cr_disconnected": [False, False, True, None],
            "runtime_seconds": [0.1, 0.2, 0.3, 0.4],
            "failed_alpha_count": [0, 0, 0, 1],
            "selected_controls": [None, None, 3, 4],
        }
    )


def _row(summary: pd.DataFrame, estimator: str) -> pd.Series:
    return summary.loc[summary["estimator"] == estimator].iloc[0]


def test_aggregate_results_returns_one_row_per_group() -> None:
    summary = aggregate_results(_raw_results(), expected_replications=2)

    assert len(summary) == 2
    assert set(summary["estimator"]) == {"dml_ivqr", "post_selection_ivqr"}


def test_aggregate_results_preserves_group_columns() -> None:
    summary = aggregate_results(_raw_results(), expected_replications=2)

    assert GROUP_COLUMNS == ["dgp", "n", "p", "pi", "tau", "estimator"]
    assert all(column in summary.columns for column in GROUP_COLUMNS)
    assert all(column in summary.columns for column in SUMMARY_METRIC_COLUMNS)


def test_aggregate_results_includes_strict_and_valid_only_length_metrics() -> None:
    summary = aggregate_results(_raw_results(), expected_replications=2)
    row = _row(summary, "post_selection_ivqr")

    assert row["avg_cr_length"] == pytest.approx(1.5 / 2)
    assert row["avg_cr_length_valid_only"] == pytest.approx(1.5)


def test_aggregate_results_dml_metrics() -> None:
    summary = aggregate_results(_raw_results(), expected_replications=2)
    row = _row(summary, "dml_ivqr")

    assert row["replications"] == 2
    assert row["valid_estimates"] == 2
    assert row["bias"] == pytest.approx(0.0)
    assert row["rmse"] == pytest.approx(0.1)
    assert row["mae"] == pytest.approx(0.1)
    assert row["coverage"] == pytest.approx(1.0)
    assert row["avg_cr_length"] == pytest.approx(1.5)
    assert row["failure_rate"] == pytest.approx(0.0)
    assert row["non_convergence_rate"] == pytest.approx(0.0)
    assert row["observed_replications"] == 2
    assert row["expected_replications"] == 2
    assert row["completion_rate"] == pytest.approx(1.0)


def test_aggregate_results_post_selection_metrics() -> None:
    summary = aggregate_results(_raw_results(), expected_replications=2)
    row = _row(summary, "post_selection_ivqr")

    assert row["replications"] == 2
    assert row["valid_estimates"] == 1
    assert row["failure_rate"] == pytest.approx(0.5)
    assert row["non_convergence_rate"] == pytest.approx(0.5)
    assert row["cr_empty_rate"] == pytest.approx(0.5)
    assert row["coverage"] == pytest.approx(0.0)
    assert row["coverage_valid_only"] == pytest.approx(0.0)
    assert row["mean_selected_controls"] == pytest.approx(3.5)


def test_load_raw_results_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_results(tmp_path / "missing.csv")


def test_load_raw_results_missing_group_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    _raw_results().drop(columns=["dgp"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required group columns"):
        load_raw_results(path)


def test_aggregate_results_file_writes_summary_csv(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "summary.csv"
    _raw_results().to_csv(input_path, index=False)

    summary = aggregate_results_file(
        input_path,
        output_path,
        expected_replications=2,
    )

    written = pd.read_csv(output_path)
    assert len(summary) == 2
    assert len(written) == 2
    assert set(written["estimator"]) == {"dml_ivqr", "post_selection_ivqr"}


def test_incomplete_groups_returns_completion_rate_below_one() -> None:
    summary = aggregate_results(_raw_results(), expected_replications=3)

    incomplete = incomplete_groups(summary)

    assert len(incomplete) == 2
    assert incomplete["completion_rate"].tolist() == pytest.approx([2 / 3, 2 / 3])


def test_incomplete_groups_empty_when_expected_replications_missing() -> None:
    summary = aggregate_results(_raw_results(), expected_replications=None)

    incomplete = incomplete_groups(summary)

    assert incomplete.empty
    assert summary["completion_rate"].isna().all()


def test_aggregate_results_sorting_is_deterministic() -> None:
    raw = _raw_results().iloc[[2, 3, 0, 1]].copy()

    summary = aggregate_results(raw, expected_replications=2)

    assert summary["estimator"].tolist() == ["dml_ivqr", "post_selection_ivqr"]


def test_aggregate_results_observed_replications_uses_unique_rep_values() -> None:
    raw = pd.concat([_raw_results(), _raw_results().iloc[[0]]], ignore_index=True)

    summary = aggregate_results(raw, expected_replications=2)
    dml = _row(summary, "dml_ivqr")

    assert dml["replications"] == 3
    assert dml["observed_replications"] == 2
    assert dml["completion_rate"] == pytest.approx(1.0)


def test_aggregate_results_without_rep_column_uses_row_count() -> None:
    raw = _raw_results().drop(columns=["rep"])

    summary = aggregate_results(raw, expected_replications=2)

    assert summary["observed_replications"].tolist() == [2, 2]


def test_aggregate_results_duplicate_raw_rows_raise_value_error() -> None:
    raw = _raw_results().copy()
    raw["seed"] = [123, 124, 123, 124]
    duplicate = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        aggregate_results(duplicate)


def test_aggregate_results_different_rep_rows_are_not_duplicates() -> None:
    raw = _raw_results().copy()
    raw["seed"] = [123, 124, 123, 124]
    extra = raw.iloc[[0]].copy()
    extra["rep"] = 2
    extra["seed"] = 125

    summary = aggregate_results(pd.concat([raw, extra], ignore_index=True))
    dml = _row(summary, "dml_ivqr")

    assert dml["observed_replications"] == 3


def test_aggregate_results_without_seed_still_aggregates_artificial_data() -> None:
    summary = aggregate_results(_raw_results(), expected_replications=2)

    assert len(summary) == 2


def test_incomplete_groups_returns_empty_if_column_missing() -> None:
    summary = aggregate_results(_raw_results()).drop(columns=["completion_rate"])

    assert incomplete_groups(summary).empty


def test_aggregate_results_validates_metric_columns() -> None:
    raw = _raw_results().drop(columns=["alpha_hat"])

    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_results(raw)
