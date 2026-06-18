# Consolidated tests for the thematic project structure.

from pathlib import Path

import _path  # noqa: F401
import pandas as pd
import pytest

from reporting.summaries import (
    GROUP_COLUMNS,
    SUMMARY_METRIC_COLUMNS,
    aggregate_results,
    aggregate_results_file,
    incomplete_groups,
    load_raw_results,
)
from reporting.tables import (
    ESTIMATOR_LABELS,
    add_estimator_labels,
    filter_summary,
    load_summary,
    make_comparison_table,
    make_diagnostic_table,
    make_wide_metric_table,
    write_tables,
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

def _summary() -> pd.DataFrame:
    rows = []
    for dgp, n, p, pi, tau in [
        ("dgp1", 80, 5, 1.0, 0.5),
        ("dgp1", 100, 10, 0.5, 0.75),
    ]:
        rows.extend(
            [
                {
                    "dgp": dgp,
                    "n": n,
                    "p": p,
                    "pi": pi,
                    "tau": tau,
                    "estimator": "dml_ivqr",
                    "bias": 0.123456,
                    "median_bias": 0.1,
                    "rmse": 0.23456,
                    "mae": 0.2,
                    "coverage": 0.95,
                    "avg_cr_length": 1.23456,
                    "avg_cr_length_valid_only": 1.5,
                    "failure_rate": 0.0,
                    "non_convergence_rate": 0.0,
                    "cr_empty_rate": 0.0,
                    "cr_disconnected_rate": 0.0,
                    "mean_runtime_seconds": 0.45678,
                    "replications": 2,
                    "valid_estimates": 2,
                    "expected_replications": 2,
                    "observed_replications": 2,
                    "completion_rate": 1.0,
                    "boundary_rate": 0.0,
                    "mean_failed_alpha_count": 0.0,
                    "mean_selected_controls": 3.0,
                },
                {
                    "dgp": dgp,
                    "n": n,
                    "p": p,
                    "pi": pi,
                    "tau": tau,
                    "estimator": "post_selection_ivqr",
                    "bias": -0.2,
                    "median_bias": -0.2,
                    "rmse": 0.3,
                    "mae": 0.25,
                    "coverage": 0.9,
                    "avg_cr_length": 1.5,
                    "avg_cr_length_valid_only": 2.0,
                    "failure_rate": 0.1,
                    "non_convergence_rate": 0.1,
                    "cr_empty_rate": 0.05,
                    "cr_disconnected_rate": 0.0,
                    "mean_runtime_seconds": 0.8,
                    "replications": 2,
                    "valid_estimates": 1,
                    "expected_replications": 2,
                    "observed_replications": 2,
                    "completion_rate": 1.0,
                    "boundary_rate": 0.0,
                    "mean_failed_alpha_count": 1.0,
                    "mean_selected_controls": 4.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_add_estimator_labels_maps_known_and_preserves_unknown() -> None:
    summary = pd.concat(
        [
            _summary().iloc[[0]],
            pd.DataFrame(
                [
                    {
                        **_summary().iloc[0].to_dict(),
                        "estimator": "custom_ivqr",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    labeled = add_estimator_labels(summary)

    assert "estimator_label" in labeled.columns
    assert ESTIMATOR_LABELS["dml_ivqr"] in labeled["estimator_label"].tolist()
    assert "custom_ivqr" in labeled["estimator_label"].tolist()
    assert "estimator_label" not in summary.columns


def test_filter_summary_filters_values_and_empty_matches() -> None:
    summary = _summary()

    filtered = filter_summary(
        summary,
        dgp="dgp1",
        n=80,
        p=5,
        pi=1.0,
        tau=0.5,
        estimators=("dml_ivqr",),
    )
    empty = filter_summary(summary, dgp="dgp3")

    assert len(filtered) == 1
    assert filtered.iloc[0]["estimator"] == "dml_ivqr"
    assert empty.empty


def test_make_wide_metric_table_uses_display_labels_and_values() -> None:
    wide = make_wide_metric_table(_summary(), "rmse", round_digits=2)

    assert "Post-selection IVQR" in wide.columns
    assert "DML-IVQR" in wide.columns
    assert wide.iloc[0]["DML-IVQR"] == pytest.approx(0.23)
    assert wide.iloc[0]["Post-selection IVQR"] == pytest.approx(0.3)


def test_make_wide_metric_table_missing_metric_raises() -> None:
    with pytest.raises(ValueError, match="metric column not found"):
        make_wide_metric_table(_summary(), "not_a_metric")


def test_make_wide_metric_table_duplicate_rows_raise() -> None:
    duplicate = pd.concat([_summary(), _summary().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate scenario-estimator"):
        make_wide_metric_table(duplicate, "rmse")


def test_make_comparison_table_contains_metrics_labels_and_rounded_values() -> None:
    table = make_comparison_table(_summary(), metrics=["bias", "rmse"], round_digits=2)

    assert table.columns.tolist() == [
        "dgp",
        "n",
        "p",
        "pi",
        "tau",
        "estimator",
        "estimator_label",
        "bias",
        "rmse",
    ]
    assert table.iloc[0]["estimator_label"] == "Post-selection IVQR"
    assert table.loc[table["estimator_label"] == "DML-IVQR", "bias"].iloc[0] == pytest.approx(
        0.12
    )


def test_make_diagnostic_table_includes_available_columns_only() -> None:
    summary = _summary().drop(columns=["boundary_rate", "mean_selected_controls"])

    table = make_diagnostic_table(summary, round_digits=3)

    assert "replications" in table.columns
    assert "completion_rate" in table.columns
    assert "avg_cr_length_valid_only" in table.columns
    assert "boundary_rate" not in table.columns
    assert "mean_selected_controls" not in table.columns


def test_cr_length_wide_uses_strict_avg_cr_length(tmp_path: Path) -> None:
    written = write_tables(_summary(), tmp_path, round_digits=3)

    cr_length = pd.read_csv(written["cr_length"])

    assert cr_length.iloc[0]["DML-IVQR"] == pytest.approx(1.235)


def test_write_tables_writes_expected_csv_files(tmp_path: Path) -> None:
    written = write_tables(_summary(), tmp_path, round_digits=3)

    expected_keys = {
        "comparison",
        "diagnostic",
        "bias",
        "rmse",
        "mae",
        "coverage",
        "cr_length",
        "runtime",
        "failure_rate",
    }
    assert expected_keys.issubset(written)
    assert all(path.exists() for path in written.values())


def test_load_summary_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_summary(tmp_path / "missing.csv")


def test_load_summary_missing_group_columns_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    _summary().drop(columns=["dgp"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_summary(path)


def test_load_summary_requires_at_least_one_metric_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    _summary()[["dgp", "n", "p", "pi", "tau", "estimator"]].to_csv(path, index=False)

    with pytest.raises(ValueError, match="at least one metric"):
        load_summary(path)
