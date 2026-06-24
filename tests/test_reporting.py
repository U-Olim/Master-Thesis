# Consolidated tests for the thematic project structure.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import reporting.summaries as summaries_module
from reporting.figures import (
    DEFAULT_FIGURE_METRICS,
    make_metric_figure,
    write_figures,
)
from reporting.summaries import (
    GROUP_COLUMNS,
    RAW_UNIQUE_COLUMNS,
    SUMMARY_METRIC_COLUMNS,
    aggregate_results,
    aggregate_results_file,
    incomplete_groups,
    load_raw_results,
    save_summary,
    validate_no_duplicate_raw_rows,
)
from reporting.tables import (
    ESTIMATOR_LABELS,
    TABLE_GROUP_COLUMNS,
    WIDE_TABLE_METRICS,
    _round_numeric,
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


def _r10_style_raw_results() -> pd.DataFrame:
    rows = []
    for rep in range(2):
        rows.extend(
            [
                {
                    "dgp": "dgp1",
                    "n": 500,
                    "p": 200,
                    "pi": 0.1,
                    "tau": 0.25,
                    "rep": rep,
                    "seed": 100 + rep,
                    "estimator": "oracle",
                    "alpha_hat": 1.0 + 0.1 * rep,
                    "alpha_true": 1.0,
                    "failed": False,
                    "converged": True,
                    "cr_length": 1.0,
                    "cr_covers_true": rep == 0,
                    "cr_empty": False,
                    "cr_disconnected": False,
                    "runtime_seconds": 0.1,
                    "failed_alpha_count": 0,
                    "selected_controls": 10,
                },
                {
                    "dgp": "dgp1",
                    "n": 500,
                    "p": 200,
                    "pi": 0.1,
                    "tau": 0.25,
                    "rep": rep,
                    "seed": 100 + rep,
                    "estimator": "post_selection_ivqr",
                    "alpha_hat": 0.9 + 0.1 * rep,
                    "alpha_true": 1.0,
                    "failed": False,
                    "converged": True,
                    "cr_length": 1.5,
                    "cr_covers_true": True,
                    "cr_empty": False,
                    "cr_disconnected": False,
                    "runtime_seconds": 0.2,
                    "failed_alpha_count": 0,
                    "selected_controls": 20,
                },
                {
                    "dgp": "dgp1",
                    "n": 500,
                    "p": 200,
                    "pi": 0.1,
                    "tau": 0.25,
                    "rep": rep,
                    "seed": 100 + rep,
                    "estimator": "dml_ivqr",
                    "alpha_hat": 1.1 + 0.1 * rep,
                    "alpha_true": 1.0,
                    "failed": False,
                    "converged": True,
                    "cr_length": 2.0,
                    "cr_covers_true": True,
                    "cr_empty": False,
                    "cr_disconnected": False,
                    "runtime_seconds": 0.3,
                    "failed_alpha_count": 0,
                    "selected_controls": None,
                },
            ]
        )
    return pd.DataFrame(rows)


def _row(summary: pd.DataFrame, estimator: str) -> pd.Series:
    return summary.loc[summary["estimator"] == estimator].iloc[0]


def test_aggregate_results_returns_one_row_per_group() -> None:
    summary = aggregate_results(_raw_results(), expected_replications=2)

    assert len(summary) == 2
    assert set(summary["estimator"]) == {"dml_ivqr", "post_selection_ivqr"}


def test_aggregate_results_preserves_group_columns() -> None:
    summary = aggregate_results(_raw_results(), expected_replications=2)

    assert GROUP_COLUMNS == ("dgp", "n", "p", "pi", "tau", "estimator")
    assert all(column in summary.columns for column in GROUP_COLUMNS)
    assert all(column in summary.columns for column in SUMMARY_METRIC_COLUMNS)


def test_aggregate_results_includes_strict_and_valid_only_length_metrics() -> None:
    summary = aggregate_results(_raw_results(), expected_replications=2)
    row = _row(summary, "post_selection_ivqr")

    assert row["avg_cr_length"] == pytest.approx(1.5)
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
    row = _row(summary, "full_control_ivqr")

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
    _raw_results().drop(columns=["alpha_hat"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_raw_results(path)


def test_validate_no_duplicate_raw_rows_validates_direct_input() -> None:
    with pytest.raises(TypeError):
        validate_no_duplicate_raw_rows([])  # type: ignore[arg-type]

    duplicate_columns = pd.concat(
        [_raw_results(), _raw_results()[["alpha_hat"]]],
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
            _raw_results(),
            expected_replications=expected_replications,
        )


def test_aggregate_results_allows_completion_rate_above_one() -> None:
    summary = aggregate_results(_raw_results(), expected_replications=1)

    assert summary["completion_rate"].tolist() == [2.0, 2.0]


def test_aggregate_results_empty_input_preserves_summary_schema() -> None:
    summary = aggregate_results(_raw_results().iloc[0:0])
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
    raw = _raw_results().copy()
    raw["rep"] = raw["rep"].astype(object)
    raw.loc[raw.index[0], "rep"] = rep

    with pytest.raises(ValueError, match=message):
        aggregate_results(raw)


def test_aggregate_results_ignores_missing_rep_values() -> None:
    raw = _raw_results().copy()
    raw.loc[raw.index[0], "rep"] = None

    summary = aggregate_results(raw)

    assert _row(summary, "dml_ivqr")["observed_replications"] == 1


def test_aggregate_results_duplicate_rep_ids_count_once() -> None:
    raw = _raw_results().copy()
    raw.loc[raw["estimator"] == "dml_ivqr", "rep"] = 0

    summary = aggregate_results(raw)

    assert _row(summary, "dml_ivqr")["observed_replications"] == 1


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
    summary = aggregate_results(_raw_results(), expected_replications=2)
    path = tmp_path / "nested" / "summary.csv"

    save_summary(summary, path)

    pd.testing.assert_frame_equal(
        pd.read_csv(path),
        summary,
        check_dtype=False,
    )


def test_aggregate_results_file_propagates_validation_errors(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    _raw_results().to_csv(input_path, index=False)

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


def test_add_estimator_labels_maps_oracle() -> None:
    summary = pd.DataFrame(
        [
            {
                **_summary().iloc[0].to_dict(),
                "estimator": "oracle",
            }
        ]
    )

    labeled = add_estimator_labels(summary)

    assert labeled.iloc[0]["estimator_label"] == "Oracle IVQR"


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

    assert "post_selection_rmse" in wide.columns
    assert "dml_rmse" in wide.columns
    assert not wide.columns.duplicated().any()
    assert wide.iloc[0]["dml_rmse"] == pytest.approx(0.23)
    assert wide.iloc[0]["post_selection_rmse"] == pytest.approx(0.3)


def test_make_wide_metric_table_keeps_oracle_column_unique() -> None:
    summary = aggregate_results(_r10_style_raw_results(), expected_replications=2)

    wide = make_wide_metric_table(summary, "rmse", round_digits=3)

    assert wide.columns.tolist() == [
        "dgp",
        "n",
        "p",
        "pi",
        "tau",
        "oracle_rmse",
        "post_selection_rmse",
        "dml_rmse",
    ]
    assert not wide.columns.duplicated().any()


def test_make_wide_metric_table_missing_metric_raises() -> None:
    with pytest.raises(ValueError, match="metric column not found"):
        make_wide_metric_table(_summary(), "not_a_metric")


def test_make_wide_metric_table_duplicate_rows_raise() -> None:
    duplicate = pd.concat([_summary(), _summary().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate scenario-estimator"):
        make_wide_metric_table(duplicate, "rmse")


def test_round_numeric_duplicate_column_raises_clear_error() -> None:
    table = pd.DataFrame([[1.234, 5.678]], columns=["rmse", "rmse"])

    with pytest.raises(ValueError, match="Column 'rmse' is duplicated"):
        _round_numeric(table, ["rmse"], round_digits=2)


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

    assert cr_length.iloc[0]["dml_avg_cr_length"] == pytest.approx(1.235)


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


def test_write_tables_from_r10_style_raw_results_does_not_crash(tmp_path: Path) -> None:
    summary = aggregate_results(_r10_style_raw_results(), expected_replications=2)

    written = write_tables(summary, tmp_path, round_digits=3)

    assert written["coverage"].exists()
    assert written["rmse"].exists()


def test_wide_csv_files_do_not_have_duplicate_columns(tmp_path: Path) -> None:
    summary = aggregate_results(_r10_style_raw_results(), expected_replications=2)
    written = write_tables(summary, tmp_path, round_digits=3)

    coverage = pd.read_csv(written["coverage"])
    rmse = pd.read_csv(written["rmse"])

    assert not coverage.columns.duplicated().any()
    assert not rmse.columns.duplicated().any()


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


def test_table_constants_are_immutable() -> None:
    assert isinstance(TABLE_GROUP_COLUMNS, tuple)
    with pytest.raises(TypeError):
        ESTIMATOR_LABELS["new"] = "New"  # type: ignore[index]
    with pytest.raises(TypeError):
        WIDE_TABLE_METRICS["new"] = "new.csv"  # type: ignore[index]


@pytest.mark.parametrize(
    "summary",
    [
        [],
        pd.DataFrame(columns=["dgp", "n", "p", "pi", "tau", "estimator"]),
    ],
)
def test_table_builders_reject_invalid_summary(summary) -> None:
    expected = TypeError if not isinstance(summary, pd.DataFrame) else ValueError
    with pytest.raises(expected):
        make_comparison_table(summary, metrics=["rmse"])


def test_table_builders_reject_duplicate_summary_columns() -> None:
    summary = pd.concat([_summary(), _summary()[["rmse"]]], axis=1)

    with pytest.raises(ValueError, match="duplicate columns"):
        make_comparison_table(summary, metrics=["rmse"])


@pytest.mark.parametrize(
    "metrics",
    ["rmse", [], ["rmse", "rmse"], [""], ["not_a_metric"]],
)
def test_make_comparison_table_validates_metrics(metrics) -> None:
    with pytest.raises(ValueError):
        make_comparison_table(_summary(), metrics=metrics)


@pytest.mark.parametrize("round_digits", [True, -1, 1.5])
def test_table_builders_validate_round_digits(round_digits) -> None:
    with pytest.raises(ValueError):
        make_comparison_table(
            _summary(),
            metrics=["rmse"],
            round_digits=round_digits,
        )


def test_round_numeric_none_returns_unrounded_copy() -> None:
    table = pd.DataFrame({"rmse": [0.123456]})

    result = _round_numeric(table, ["rmse"], round_digits=None)

    assert result is not table
    assert result.iloc[0]["rmse"] == pytest.approx(0.123456)


@pytest.mark.parametrize(
    "estimators",
    ["dml_ivqr", (), ("dml_ivqr", "dml_ivqr"), (1,)],
)
def test_filter_summary_rejects_invalid_estimator_filters(estimators) -> None:
    with pytest.raises(ValueError):
        filter_summary(_summary(), estimators=estimators)


@pytest.mark.parametrize(
    "index_columns",
    ["dgp", [], ["dgp", "dgp"], ["missing"]],
)
def test_make_wide_metric_table_validates_index_columns(index_columns) -> None:
    with pytest.raises(ValueError):
        make_wide_metric_table(
            _summary(),
            "rmse",
            index_columns=index_columns,
        )


def test_make_wide_metric_table_accepts_custom_index_columns() -> None:
    wide = make_wide_metric_table(
        _summary(),
        "rmse",
        index_columns=["dgp", "n", "p", "pi", "tau"],
    )

    assert "dml_rmse" in wide.columns


def test_make_wide_metric_table_rejects_all_nonnumeric_metric() -> None:
    summary = _summary()
    summary["rmse"] = "invalid"

    with pytest.raises(ValueError, match="has no numeric values"):
        make_wide_metric_table(summary, "rmse")


def test_make_wide_metric_table_keeps_partial_nonnumeric_as_nan() -> None:
    summary = _summary()
    summary["rmse"] = summary["rmse"].astype(object)
    summary.loc[summary.index[0], "rmse"] = "invalid"

    wide = make_wide_metric_table(summary, "rmse", round_digits=None)

    assert wide["dml_rmse"].isna().any()


def test_make_comparison_table_rejects_all_nonnumeric_metrics() -> None:
    summary = _summary()
    summary["rmse"] = "invalid"

    with pytest.raises(ValueError, match="no numeric values"):
        make_comparison_table(summary, metrics=["rmse"])


def test_make_diagnostic_table_without_diagnostics_returns_identifiers() -> None:
    summary = _summary()[
        ["dgp", "n", "p", "pi", "tau", "estimator", "rmse"]
    ]

    table = make_diagnostic_table(summary)

    assert table.columns.tolist() == [
        "dgp",
        "n",
        "p",
        "pi",
        "tau",
        "estimator",
        "estimator_label",
    ]


def test_load_summary_rejects_nonnumeric_metrics(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    summary = _summary()[
        ["dgp", "n", "p", "pi", "tau", "estimator", "rmse"]
    ].copy()
    summary["rmse"] = "invalid"
    summary.to_csv(path, index=False)

    with pytest.raises(ValueError, match="must contain at least one numeric value"):
        load_summary(path)


def test_write_tables_rejects_existing_file_output_dir(tmp_path: Path) -> None:
    output = tmp_path / "tables"
    output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a directory path"):
        write_tables(_summary(), output)


def test_write_tables_rejects_summary_without_core_metrics(tmp_path: Path) -> None:
    summary = _summary()[
        ["dgp", "n", "p", "pi", "tau", "estimator", "replications"]
    ]

    with pytest.raises(ValueError, match="does not contain any core metrics"):
        write_tables(summary, tmp_path)


def test_unknown_estimators_sort_alphabetically_after_known() -> None:
    base = _summary().iloc[[0]]
    summary = pd.concat(
        [
            base,
            base.assign(estimator="z_estimator"),
            base.assign(estimator="a_estimator"),
        ],
        ignore_index=True,
    )

    labeled = add_estimator_labels(summary)

    assert labeled["estimator"].astype(str).tolist() == [
        "dml_ivqr",
        "a_estimator",
        "z_estimator",
    ]


def test_make_metric_figure_writes_png(tmp_path: Path) -> None:
    path = make_metric_figure(_summary(), "rmse", tmp_path / "rmse.png", title="RMSE")

    assert path.exists()
    assert path.stat().st_size > 0


def test_make_metric_figure_creates_parent_directory(tmp_path: Path) -> None:
    path = make_metric_figure(
        _summary(),
        "coverage",
        tmp_path / "nested" / "coverage.png",
    )

    assert path.exists()


def test_make_metric_figure_does_not_mutate_summary(tmp_path: Path) -> None:
    summary = _summary()
    original = summary.copy(deep=True)

    make_metric_figure(summary, "rmse", tmp_path / "rmse.png")

    pd.testing.assert_frame_equal(summary, original)


@pytest.mark.parametrize(
    "summary",
    [
        [],
        pd.DataFrame(),
        pd.DataFrame({"dgp": ["dgp1"]}),
    ],
)
def test_make_metric_figure_rejects_invalid_summary(
    summary,
    tmp_path: Path,
) -> None:
    expected = TypeError if not isinstance(summary, pd.DataFrame) else ValueError
    with pytest.raises(expected):
        make_metric_figure(summary, "rmse", tmp_path / "rmse.png")


def test_make_metric_figure_rejects_duplicate_columns(tmp_path: Path) -> None:
    summary = pd.concat([_summary(), _summary()[["rmse"]]], axis=1)

    with pytest.raises(ValueError, match="duplicate columns"):
        make_metric_figure(summary, "rmse", tmp_path / "rmse.png")


def test_make_metric_figure_rejects_nonnumeric_metric(tmp_path: Path) -> None:
    summary = _summary()
    summary["rmse"] = "invalid"

    with pytest.raises(ValueError, match="has no numeric values"):
        make_metric_figure(summary, "rmse", tmp_path / "rmse.png")


@pytest.mark.parametrize("metric", ["", "not_a_metric"])
def test_make_metric_figure_rejects_invalid_metric(
    metric: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        make_metric_figure(_summary(), metric, tmp_path / "metric.png")


def test_write_figures_writes_default_available_metrics(tmp_path: Path) -> None:
    written = write_figures(_summary(), tmp_path)

    assert {
        "bias",
        "rmse",
        "coverage",
        "avg_cr_length",
        "failure_rate",
    }.issubset(written)
    assert all(path.exists() for path in written.values())


def test_write_figures_skips_missing_metrics(tmp_path: Path) -> None:
    summary = _summary().drop(columns=["failure_rate"])

    written = write_figures(summary, tmp_path)

    assert "failure_rate" not in written


def test_write_figures_accepts_custom_specs(tmp_path: Path) -> None:
    written = write_figures(
        _summary(),
        tmp_path,
        metrics={
            "rmse": "Root mean squared error",
            "coverage": ("custom_coverage.png", "Coverage"),
        },
    )

    assert written["rmse"].name == "fig_rmse.png"
    assert written["coverage"].name == "custom_coverage.png"
    assert all(path.exists() for path in written.values())


@pytest.mark.parametrize(
    "metrics",
    [
        [],
        {},
        {"rmse": ("only_one",)},
        {"rmse": ""},
        {"": "RMSE"},
    ],
)
def test_write_figures_rejects_invalid_metric_specs(metrics, tmp_path: Path) -> None:
    expected = TypeError if isinstance(metrics, list) else ValueError
    with pytest.raises(expected):
        write_figures(_summary(), tmp_path, metrics=metrics)


def test_make_metric_figure_closes_figure(tmp_path: Path) -> None:
    before = set(plt.get_fignums())

    make_metric_figure(_summary(), "rmse", tmp_path / "rmse.png")

    assert set(plt.get_fignums()) == before


def test_make_metric_figure_preserves_unknown_estimator_label(tmp_path: Path) -> None:
    summary = pd.concat(
        [
            _summary(),
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

    path = make_metric_figure(summary, "rmse", tmp_path / "custom.png")

    assert path.exists()


def test_default_figure_metrics_is_immutable() -> None:
    with pytest.raises(TypeError):
        DEFAULT_FIGURE_METRICS["new"] = "New"  # type: ignore[index]
