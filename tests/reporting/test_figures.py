from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from reporting.figures import (
    DEFAULT_FIGURE_METRICS,
    make_metric_figure,
    write_figures,
)
from tests.reporting.helpers import summary as make_summary

def test_make_metric_figure_writes_png(tmp_path: Path) -> None:
    path = make_metric_figure(
        make_summary(), "rmse", tmp_path / "rmse.png", title="RMSE"
    )

    assert path.exists()
    assert path.stat().st_size > 0

def test_make_metric_figure_creates_parent_directory(tmp_path: Path) -> None:
    path = make_metric_figure(
        make_summary(),
        "coverage",
        tmp_path / "nested" / "coverage.png",
    )

    assert path.exists()

def test_make_metric_figure_does_not_mutate_summary(tmp_path: Path) -> None:
    summary = make_summary()
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
    summary = pd.concat([make_summary(), make_summary()[["rmse"]]], axis=1)

    with pytest.raises(ValueError, match="duplicate columns"):
        make_metric_figure(summary, "rmse", tmp_path / "rmse.png")

def test_make_metric_figure_rejects_nonnumeric_metric(tmp_path: Path) -> None:
    summary = make_summary()
    summary["rmse"] = "invalid"

    with pytest.raises(ValueError, match="has no numeric values"):
        make_metric_figure(summary, "rmse", tmp_path / "rmse.png")

@pytest.mark.parametrize("metric", ["", "not_a_metric"])
def test_make_metric_figure_rejects_invalid_metric(
    metric: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        make_metric_figure(make_summary(), metric, tmp_path / "metric.png")

def test_write_figures_writes_default_available_metrics(tmp_path: Path) -> None:
    written = write_figures(make_summary(), tmp_path)

    assert {
        "bias",
        "rmse",
        "coverage",
        "avg_cr_length",
        "failure_rate",
    }.issubset(written)
    assert all(path.exists() for path in written.values())

def test_write_figures_skips_missing_metrics(tmp_path: Path) -> None:
    summary = make_summary().drop(columns=["failure_rate"])

    written = write_figures(summary, tmp_path)

    assert "failure_rate" not in written

def test_write_figures_accepts_custom_specs(tmp_path: Path) -> None:
    written = write_figures(
        make_summary(),
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
        write_figures(make_summary(), tmp_path, metrics=metrics)

def test_make_metric_figure_closes_figure(tmp_path: Path) -> None:
    before = set(plt.get_fignums())

    make_metric_figure(make_summary(), "rmse", tmp_path / "rmse.png")

    assert set(plt.get_fignums()) == before

def test_make_metric_figure_preserves_unknown_estimator_label(tmp_path: Path) -> None:
    summary = pd.concat(
        [
            make_summary(),
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

    path = make_metric_figure(summary, "rmse", tmp_path / "custom.png")

    assert path.exists()

def test_default_figure_metrics_is_immutable() -> None:
    with pytest.raises(TypeError):
        DEFAULT_FIGURE_METRICS["new"] = "New"  # type: ignore[index]
