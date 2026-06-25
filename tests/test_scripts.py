from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

import inference
import pandas as pd
import pytest
from dgp.designs import Design
from estimators.base import EstimationResult
from inference import metrics
from tests.helpers import load_full_control_cli, load_main_simulation_cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_SIMULATION_SCRIPT = PROJECT_ROOT / "scenarios" / "main_simulation.py"
FULL_CONTROL_SCRIPT = PROJECT_ROOT / "scenarios" / "full_control_ivqr.py"


@dataclass(frozen=True)
class CliRun:
    returncode: int
    stdout: str
    stderr: str


@pytest.fixture(scope="module")
def main_cli():
    return load_main_simulation_cli()


@pytest.fixture(scope="module")
def full_control_cli():
    return load_full_control_cli()


def _run_cli_in_process(
    module,
    args: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    *,
    expected_exit: int | None = None,
) -> CliRun:
    monkeypatch.chdir(tmp_path)
    script_path = Path(module.__file__)
    monkeypatch.setattr(sys, "argv", [str(script_path), *args])

    if expected_exit is None:
        module.main()
        captured = capsys.readouterr()
        return CliRun(returncode=0, stdout=captured.out, stderr=captured.err)

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == expected_exit
    captured = capsys.readouterr()
    return CliRun(
        returncode=int(exc_info.value.code),
        stdout=captured.out,
        stderr=captured.err,
    )


def _run_script(
    script: Path,
    *args: str,
    tmp_path: Path,
    timeout: int = 240,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path
        if not existing_pythonpath
        else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib-cache")

    return subprocess.run(
        [sys.executable, str(script), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=tmp_path,
        env=env,
    )


def _assert_standard_reports_exist(
    *,
    raw: Path,
    summary: Path,
    tables: Path,
    figures: Path,
) -> None:
    assert raw.exists()
    assert summary.exists()

    for filename in [
        "comparison_table.csv",
        "bias_wide.csv",
        "rmse_wide.csv",
        "coverage_wide.csv",
        "cr_length_wide.csv",
        "failure_rate_wide.csv",
    ]:
        assert (tables / filename).exists()

    for filename in [
        "fig_bias.png",
        "fig_rmse.png",
        "fig_coverage.png",
        "fig_cr_length.png",
        "fig_failure_rate.png",
    ]:
        assert (figures / filename).exists()


def test_core_phase1_imports_work() -> None:
    assert inference is not None
    assert metrics is not None
    assert Design is not None
    assert EstimationResult is not None


def test_main_simulation_help_uses_modes(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = _run_cli_in_process(
        main_cli,
        ["--help"],
        monkeypatch,
        capsys,
        tmp_path,
        expected_exit=0,
    )

    assert result.returncode == 0
    assert "Run the main IVQR Monte Carlo simulation" in result.stdout
    assert "--mode {fast,full}" in result.stdout
    assert "full-control-benchmark" not in result.stdout


def test_main_simulation_fast_dry_run_excludes_full_control(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "raw" / "fast_mode_results.csv"
    result = _run_cli_in_process(
        main_cli,
        [
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
            "--output",
            str(output),
        ],
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert result.returncode == 0
    assert "Mode: fast" in result.stdout
    assert "Replications per design: 1" in result.stdout
    assert "Alpha grid size: 9" in result.stdout
    assert str(output) in result.stdout
    assert "Reports: automatic after successful run" in result.stdout
    assert "full-control IVQR benchmark" not in result.stdout
    assert not output.exists()


def test_main_simulation_full_dry_run_uses_500_reps(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "raw" / "full_mode_results.csv"
    result = _run_cli_in_process(
        main_cli,
        [
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
            "--output",
            str(output),
        ],
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert result.returncode == 0
    assert "Mode: full" in result.stdout
    assert "Replications per design: 500" in result.stdout
    assert "Alpha grid size: 9" in result.stdout
    assert str(output) in result.stdout
    assert "Reports: automatic after successful run" in result.stdout
    assert not output.exists()


def test_main_simulation_rejects_full_control_estimator(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = _run_cli_in_process(
        main_cli,
        ["--estimators", "full", "--dry-run"],
        monkeypatch,
        capsys,
        tmp_path,
        expected_exit=2,
    )

    assert result.returncode == 2
    assert "invalid choice: 'full'" in result.stderr


def test_full_control_script_dry_run_uses_limited_design(
    full_control_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "raw" / "full_control_ivqr_results.csv"
    result = _run_cli_in_process(
        full_control_cli,
        ["--dry-run", "--output", str(output)],
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert result.returncode == 0
    assert "Mode: full-control IVQR benchmark" in result.stdout
    assert "Replications per design: 500" in result.stdout
    assert "Alpha grid size: 9" in result.stdout
    assert str(output) in result.stdout
    assert "Reports: automatic after successful run" in result.stdout
    assert not output.exists()


@pytest.mark.slow
def test_main_fast_smoke_creates_reports(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "fast.csv"
    summary = tmp_path / "summary" / "fast_summary.csv"
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"

    result = _run_script(
        FULL_SIMULATION_SCRIPT,
        "--mode",
        "fast",
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-values",
        "80",
        "--p-values",
        "10",
        "--alpha-grid-size",
        "3",
        "--n-jobs",
        "1",
        "--output",
        str(raw),
        "--summary-output",
        str(summary),
        "--tables-dir",
        str(tables),
        "--figures-dir",
        str(figures),
        tmp_path=tmp_path,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr
    _assert_standard_reports_exist(
        raw=raw,
        summary=summary,
        tables=tables,
        figures=figures,
    )

    written = pd.read_csv(raw)
    assert set(written["estimator"]) == {"oracle", "post_selection_ivqr", "dml_ivqr"}


@pytest.mark.slow
def test_full_control_smoke_creates_reports(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "full_control.csv"
    summary = tmp_path / "summary" / "full_control_summary.csv"
    tables = tmp_path / "tables"
    figures = tmp_path / "figures"

    result = _run_script(
        FULL_CONTROL_SCRIPT,
        "--reps",
        "1",
        "--dgps",
        "dgp1",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--n-values",
        "80",
        "--p-values",
        "10",
        "--alpha-grid-size",
        "3",
        "--n-jobs",
        "1",
        "--output",
        str(raw),
        "--summary-output",
        str(summary),
        "--tables-dir",
        str(tables),
        "--figures-dir",
        str(figures),
        tmp_path=tmp_path,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr
    _assert_standard_reports_exist(
        raw=raw,
        summary=summary,
        tables=tables,
        figures=figures,
    )

    written = pd.read_csv(raw)
    assert set(written["estimator"]) == {"full_control_ivqr"}
