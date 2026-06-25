from pathlib import Path

import pandas as pd
import pytest

from reporting.summaries import aggregate_results
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
from tests.reporting.helpers import r10_style_raw_results, summary as make_summary

def test_add_estimator_labels_maps_known_and_preserves_unknown() -> None:
    summary = pd.concat(
        [
            make_summary().iloc[[0]],
            pd.DataFrame(
                [
                    {
                        **make_summary().iloc[0].to_dict(),
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
                **make_summary().iloc[0].to_dict(),
                "estimator": "oracle",
            }
        ]
    )

    labeled = add_estimator_labels(summary)

    assert labeled.iloc[0]["estimator_label"] == "Oracle IVQR"

def test_filter_summary_filters_values_and_empty_matches() -> None:
    summary = make_summary()

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
    wide = make_wide_metric_table(make_summary(), "rmse", round_digits=2)

    assert "post_selection_rmse" in wide.columns
    assert "dml_rmse" in wide.columns
    assert not wide.columns.duplicated().any()
    assert wide.iloc[0]["dml_rmse"] == pytest.approx(0.23)
    assert wide.iloc[0]["post_selection_rmse"] == pytest.approx(0.3)

def test_make_wide_metric_table_keeps_oracle_column_unique() -> None:
    summary = aggregate_results(r10_style_raw_results(), expected_replications=2)

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
        make_wide_metric_table(make_summary(), "not_a_metric")

def test_make_wide_metric_table_duplicate_rows_raise() -> None:
    duplicate = pd.concat([make_summary(), make_summary().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate scenario-estimator"):
        make_wide_metric_table(duplicate, "rmse")

def test_round_numeric_duplicate_column_raises_clear_error() -> None:
    table = pd.DataFrame([[1.234, 5.678]], columns=["rmse", "rmse"])

    with pytest.raises(ValueError, match="Column 'rmse' is duplicated"):
        _round_numeric(table, ["rmse"], round_digits=2)

def test_make_comparison_table_contains_metrics_labels_and_rounded_values() -> None:
    table = make_comparison_table(
        make_summary(), metrics=["bias", "rmse"], round_digits=2
    )

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
    summary = make_summary().drop(
        columns=["boundary_rate", "mean_selected_controls"]
    )

    table = make_diagnostic_table(summary, round_digits=3)

    assert "replications" in table.columns
    assert "completion_rate" in table.columns
    assert "avg_cr_length_valid_only" in table.columns
    assert "boundary_rate" not in table.columns
    assert "mean_selected_controls" not in table.columns

def test_cr_length_wide_uses_strict_avg_cr_length(tmp_path: Path) -> None:
    written = write_tables(make_summary(), tmp_path, round_digits=3)

    cr_length = pd.read_csv(written["cr_length"])

    assert cr_length.iloc[0]["dml_avg_cr_length"] == pytest.approx(1.235)

def test_write_tables_writes_expected_csv_files(tmp_path: Path) -> None:
    written = write_tables(make_summary(), tmp_path, round_digits=3)

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
    summary = aggregate_results(r10_style_raw_results(), expected_replications=2)

    written = write_tables(summary, tmp_path, round_digits=3)

    assert written["coverage"].exists()
    assert written["rmse"].exists()

def test_wide_csv_files_do_not_have_duplicate_columns(tmp_path: Path) -> None:
    summary = aggregate_results(r10_style_raw_results(), expected_replications=2)
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
    make_summary().drop(columns=["dgp"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_summary(path)

def test_load_summary_requires_at_least_one_metric_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    make_summary()[["dgp", "n", "p", "pi", "tau", "estimator"]].to_csv(
        path, index=False
    )

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
    summary = pd.concat([make_summary(), make_summary()[["rmse"]]], axis=1)

    with pytest.raises(ValueError, match="duplicate columns"):
        make_comparison_table(summary, metrics=["rmse"])

@pytest.mark.parametrize(
    "metrics",
    ["rmse", [], ["rmse", "rmse"], [""], ["not_a_metric"]],
)
def test_make_comparison_table_validates_metrics(metrics) -> None:
    with pytest.raises(ValueError):
        make_comparison_table(make_summary(), metrics=metrics)

@pytest.mark.parametrize("round_digits", [True, -1, 1.5])
def test_table_builders_validate_round_digits(round_digits) -> None:
    with pytest.raises(ValueError):
        make_comparison_table(
            make_summary(),
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
        filter_summary(make_summary(), estimators=estimators)

@pytest.mark.parametrize(
    "index_columns",
    ["dgp", [], ["dgp", "dgp"], ["missing"]],
)
def test_make_wide_metric_table_validates_index_columns(index_columns) -> None:
    with pytest.raises(ValueError):
        make_wide_metric_table(
            make_summary(),
            "rmse",
            index_columns=index_columns,
        )

def test_make_wide_metric_table_accepts_custom_index_columns() -> None:
    wide = make_wide_metric_table(
        make_summary(),
        "rmse",
        index_columns=["dgp", "n", "p", "pi", "tau"],
    )

    assert "dml_rmse" in wide.columns

def test_make_wide_metric_table_rejects_all_nonnumeric_metric() -> None:
    summary = make_summary()
    summary["rmse"] = "invalid"

    with pytest.raises(ValueError, match="has no numeric values"):
        make_wide_metric_table(summary, "rmse")

def test_make_wide_metric_table_keeps_partial_nonnumeric_as_nan() -> None:
    summary = make_summary()
    summary["rmse"] = summary["rmse"].astype(object)
    summary.loc[summary.index[0], "rmse"] = "invalid"

    wide = make_wide_metric_table(summary, "rmse", round_digits=None)

    assert wide["dml_rmse"].isna().any()

def test_make_comparison_table_rejects_all_nonnumeric_metrics() -> None:
    summary = make_summary()
    summary["rmse"] = "invalid"

    with pytest.raises(ValueError, match="no numeric values"):
        make_comparison_table(summary, metrics=["rmse"])

def test_make_diagnostic_table_without_diagnostics_returns_identifiers() -> None:
    summary = make_summary()[
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
    summary = make_summary()[
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
        write_tables(make_summary(), output)

def test_write_tables_rejects_summary_without_core_metrics(tmp_path: Path) -> None:
    summary = make_summary()[
        ["dgp", "n", "p", "pi", "tau", "estimator", "replications"]
    ]

    with pytest.raises(ValueError, match="does not contain any core metrics"):
        write_tables(summary, tmp_path)

def test_unknown_estimators_sort_alphabetically_after_known() -> None:
    base = make_summary().iloc[[0]]
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
