from pathlib import Path

import pandas as pd

from reporting.figures import write_figures
from reporting.summaries import aggregate_results
from reporting.tables import ESTIMATOR_ORDER, write_tables


def _raw() -> pd.DataFrame:
    rows = []
    for estimator in ("oracle", "post_selection_ivqr", "full_control_ivqr", "dml_ivqr"):
        rows.append(
            {
                "dgp": "dgp1",
                "n": 50,
                "p": 4,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 0,
                "seed": 123,
                "estimator": estimator,
                "alpha_hat": 1.0,
                "alpha_true": 1.0,
                "failed": False,
                "converged": True,
                "cr_length": 1.0,
                "cr_hull_length": 1.0,
                "cr_covers_true": True,
                "cr_empty": False,
                "cr_disconnected": False,
                "cr_hits_any_boundary": False,
                "alpha_hat_at_any_boundary": False,
                "failed_alpha_count": 0,
                "failed_alpha_rate": 0.0,
                "selected_controls": 2,
                "critical_value_multiplier": 1.0,
                "critical_value_adjusted": 3.84,
                "runtime_seconds": 0.01,
            }
        )
    return pd.DataFrame(rows)


def _synthetic_report_frame(estimators: tuple[str, ...] = ("oracle", "dml_ivqr")) -> pd.DataFrame:
    rows = []
    for estimator_index, estimator in enumerate(estimators):
        for pi in (0.25, 1.0):
            for tau in (0.25, 0.75):
                for rep in range(3):
                    rows.append(
                        {
                            "estimator": estimator,
                            "pi": pi,
                            "tau": tau,
                            "covers": rep != 0 or estimator_index == 0,
                            "runtime_sec": (estimator_index + 1) * (rep + 1) * 0.1,
                            "cr_length": 1.0 + estimator_index + pi,
                            "boundary_hit": pi == 0.25 and rep == 0,
                            "status": "ok" if rep != 2 else "failed",
                        }
                    )
    return pd.DataFrame(rows)


def test_summary_and_compact_tables_work_on_tiny_frame(tmp_path: Path) -> None:
    assert ESTIMATOR_ORDER == (
        "oracle",
        "post_selection_ivqr",
        "full_control_ivqr",
        "dml_ivqr",
    )
    summary = aggregate_results(_raw(), expected_replications=1)
    written = write_tables(summary, tmp_path)
    assert (tmp_path / "main_summary.csv").exists()
    assert (tmp_path / "coverage_by_pi.csv").exists()
    assert (tmp_path / "runtime_summary.csv").exists()
    assert (tmp_path / "coverage_by_tau.csv").exists()
    assert set(written) == {
        "main_summary",
        "coverage_by_pi",
        "runtime_summary",
        "coverage_by_tau",
    }
    assert set(summary["estimator"]) == set(ESTIMATOR_ORDER)


def test_compact_reporting_outputs_for_synthetic_frame(tmp_path: Path) -> None:
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"

    table_paths = write_tables(_synthetic_report_frame(), tables_dir)
    figure_paths = write_figures(_synthetic_report_frame(), figures_dir)

    assert set(table_paths) == {
        "main_summary",
        "coverage_by_pi",
        "runtime_summary",
        "coverage_by_tau",
    }
    assert (tables_dir / "main_summary.csv").exists()
    assert (tables_dir / "coverage_by_pi.csv").exists()
    assert (tables_dir / "runtime_summary.csv").exists()
    assert (tables_dir / "coverage_by_tau.csv").exists()
    assert set(figure_paths) == {
        "coverage_by_pi",
        "runtime_by_estimator",
        "coverage_overall",
        "weak_iv_diagnostic",
    }
    assert (figures_dir / "coverage_by_pi.png").exists()
    assert (figures_dir / "runtime_by_estimator.png").exists()
    assert (figures_dir / "coverage_overall.png").exists()
    assert (figures_dir / "weak_iv_diagnostic.png").exists()

    main_summary = pd.read_csv(tables_dir / "main_summary.csv")
    assert list(main_summary.columns) == [
        "estimator",
        "rows",
        "coverage",
        "mean_ci_length",
        "mean_runtime_sec",
        "median_runtime_sec",
        "p95_runtime_sec",
        "failed_share",
        "empty_ci_share",
        "boundary_share",
    ]


def test_single_estimator_skips_overall_coverage_figure(tmp_path: Path) -> None:
    frame = _synthetic_report_frame(("oracle",))

    table_paths = write_tables(frame, tmp_path / "tables")
    figure_paths = write_figures(frame, tmp_path / "figures")

    assert "main_summary" in table_paths
    assert "coverage_by_pi" in table_paths
    assert "runtime_summary" in table_paths
    assert "coverage_by_tau" in table_paths
    assert "coverage_overall" not in figure_paths
    assert "coverage_by_pi" in figure_paths
    assert "runtime_by_estimator" in figure_paths
    assert "weak_iv_diagnostic" in figure_paths


def test_reporting_tolerates_missing_optional_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "estimator": ["oracle", "dml_ivqr"],
            "covers": [True, False],
            "runtime_sec": [0.1, 1.2],
        }
    )

    table_paths = write_tables(frame, tmp_path / "tables")
    figure_paths = write_figures(frame, tmp_path / "figures")

    assert set(table_paths) == {"main_summary", "runtime_summary"}
    assert set(figure_paths) == {"runtime_by_estimator", "coverage_overall"}
    main_summary = pd.read_csv(tmp_path / "tables" / "main_summary.csv")
    assert "mean_ci_length" in main_summary.columns
    assert main_summary["mean_ci_length"].isna().all()


def test_reporting_without_pi_skips_pi_outputs(tmp_path: Path) -> None:
    frame = _synthetic_report_frame().drop(columns=["pi"])

    table_paths = write_tables(frame, tmp_path / "tables")
    figure_paths = write_figures(frame, tmp_path / "figures")

    assert "coverage_by_pi" not in table_paths
    assert "coverage_by_pi" not in figure_paths
    assert "weak_iv_diagnostic" not in figure_paths
    assert "main_summary" in table_paths
    assert "runtime_summary" in table_paths
