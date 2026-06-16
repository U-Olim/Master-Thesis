from pathlib import Path

import pandas as pd
import pytest

from ivqr_sim.reporting.tables import (
    ESTIMATOR_LABELS,
    add_estimator_labels,
    filter_summary,
    load_summary,
    make_comparison_table,
    make_diagnostic_table,
    make_wide_metric_table,
    write_tables,
)


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
