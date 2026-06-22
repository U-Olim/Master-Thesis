from pathlib import Path
import subprocess
import sys

import inference
import pandas as pd
from dgp.designs import Design
from estimators.base import EstimationResult
from inference import metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_SIMULATION_SCRIPT = PROJECT_ROOT / "scripts" / "02_run_full_simulation.py"
FULL_CONTROL_SCRIPT = PROJECT_ROOT / "scripts" / "04_run_full_control_ivqr.py"


def _run_script(script: Path, *args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_core_phase1_imports_work() -> None:
    assert inference is not None
    assert metrics is not None
    assert Design is not None
    assert EstimationResult is not None


def test_main_simulation_help_uses_modes_not_full_control_preset() -> None:
    result = _run_script(FULL_SIMULATION_SCRIPT, "--help", timeout=60)

    assert result.returncode == 0
    assert "Run the main IVQR Monte Carlo simulation" in result.stdout
    assert "--mode {fast,full}" in result.stdout
    assert "full-control-benchmark" not in result.stdout


def test_main_simulation_fast_dry_run_excludes_full_control() -> None:
    result = _run_script(
        FULL_SIMULATION_SCRIPT,
        "--mode",
        "fast",
        "--dry-run",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-values",
        "500",
        "--p-values",
        "200",
    )

    assert result.returncode == 0
    assert "mode: fast" in result.stdout
    assert "replications per scenario: 1" in result.stdout
    assert "estimators: oracle,post_selection,dml" in result.stdout
    assert "Full-control IVQR is excluded" in result.stdout


def test_main_simulation_full_dry_run_uses_500_reps() -> None:
    result = _run_script(
        FULL_SIMULATION_SCRIPT,
        "--mode",
        "full",
        "--dry-run",
        "--dgps",
        "dgp1",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-values",
        "500",
        "--p-values",
        "200",
    )

    assert result.returncode == 0
    assert "mode: full" in result.stdout
    assert "replications per scenario: 500" in result.stdout
    assert "estimators: oracle,post_selection,dml" in result.stdout


def test_main_simulation_rejects_full_control_estimator() -> None:
    result = _run_script(FULL_SIMULATION_SCRIPT, "--estimators", "full", "--dry-run")

    assert result.returncode != 0
    assert "invalid choice: 'full'" in result.stderr


def test_full_control_script_dry_run_uses_limited_design() -> None:
    result = _run_script(FULL_CONTROL_SCRIPT, "--dry-run")

    assert result.returncode == 0
    assert "Full-Control IVQR benchmark plan" in result.stdout
    assert "replications per scenario: 100" in result.stdout
    assert "separate naive benchmark" in result.stdout


def test_main_fast_smoke_creates_reports(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "fast.csv"
    summary = tmp_path / "summary" / "fast_summary.csv"
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"

    result = _run_script(
        FULL_SIMULATION_SCRIPT,
        "--mode",
        "fast",
        "--quick-test",
        "--reps",
        "1",
        "--output",
        str(raw),
        "--summary-output",
        str(summary),
        "--tables-dir",
        str(tables),
        "--figures-dir",
        str(figures),
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert raw.exists()
    assert summary.exists()
    assert (tables / "comparison_table.csv").exists()
    assert (tables / "bias_wide.csv").exists()
    assert (tables / "rmse_wide.csv").exists()
    assert (tables / "coverage_wide.csv").exists()
    assert (tables / "cr_length_wide.csv").exists()
    assert (tables / "failure_rate_wide.csv").exists()
    assert (figures / "fig_bias.png").exists()
    assert (figures / "fig_rmse.png").exists()
    assert (figures / "fig_coverage.png").exists()
    assert (figures / "fig_cr_length.png").exists()
    assert (figures / "fig_failure_rate.png").exists()

    written = pd.read_csv(raw)
    assert set(written["estimator"]) == {"oracle", "post_selection_ivqr", "dml_ivqr"}


def test_full_control_smoke_creates_reports(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "full_control.csv"
    summary = tmp_path / "summary" / "full_control_summary.csv"
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"

    result = _run_script(
        FULL_CONTROL_SCRIPT,
        "--quick-test",
        "--reps",
        "1",
        "--output",
        str(raw),
        "--summary-output",
        str(summary),
        "--tables-dir",
        str(tables),
        "--figures-dir",
        str(figures),
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert raw.exists()
    assert summary.exists()
    assert (tables / "comparison_table.csv").exists()
    assert (tables / "bias_wide.csv").exists()
    assert (tables / "rmse_wide.csv").exists()
    assert (tables / "coverage_wide.csv").exists()
    assert (tables / "cr_length_wide.csv").exists()
    assert (tables / "failure_rate_wide.csv").exists()
    assert (figures / "fig_bias.png").exists()
    assert (figures / "fig_rmse.png").exists()
    assert (figures / "fig_coverage.png").exists()
    assert (figures / "fig_cr_length.png").exists()
    assert (figures / "fig_failure_rate.png").exists()

    written = pd.read_csv(raw)
    assert set(written["estimator"]) == {"full_control_ivqr"}


def test_make_tables_script_help_runs() -> None:
    script = PROJECT_ROOT / "scripts" / "03_make_tables.py"
    result = _run_script(script, "--help", timeout=60)

    assert result.returncode == 0
    assert "Create tables from IVQR simulation results" in result.stdout


def test_pilot_script_runs_end_to_end(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "scripts" / "01_pilot_simulation.py"
    result = subprocess.run(
        [sys.executable, str(script), "--estimators", "dml"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = tmp_path / "results" / "raw" / "pilot_quick_results.csv"
    assert result.returncode == 0
    assert output.exists()
    assert "dml_ivqr" in result.stdout
    assert "dml_k_folds=3" in result.stdout
