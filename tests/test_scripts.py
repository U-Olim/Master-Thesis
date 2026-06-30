from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

import inference
import numpy as np
import pandas as pd
import pytest
from dgp.designs import Design
from estimators.base import EstimationResult
from inference import metrics
from tests.helpers import (
    load_full_control_cli,
    load_main_simulation_cli,
    load_main_simulation_wrapper_cli,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_SIMULATION_SCRIPT = PROJECT_ROOT / "scenarios" / "run_simulation.py"
MAIN_SIMULATION_WRAPPER_SCRIPT = PROJECT_ROOT / "scenarios" / "main_simulation.py"
FULL_CONTROL_SCRIPT = PROJECT_ROOT / "scenarios" / "full_control_ivqr.py"
RESUME_REQUIRES_MANIFEST_MESSAGE = (
    "--resume requires --manifest so run configuration compatibility can be validated."
)
RESUME_REQUIRES_EXISTING_MANIFEST_MESSAGE = (
    "--resume requires an existing --manifest file so run configuration "
    "compatibility can be validated."
)


@dataclass(frozen=True)
class CliRun:
    returncode: int
    stdout: str
    stderr: str


def _system_exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    raise AssertionError(f"Expected integer SystemExit code, got {code!r}")


@pytest.fixture(scope="module")
def main_cli():
    return load_main_simulation_cli()


@pytest.fixture(scope="module")
def full_control_cli():
    return load_full_control_cli()


@pytest.fixture(scope="module")
def main_wrapper_cli():
    return load_main_simulation_wrapper_cli()


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

    exit_code = _system_exit_code(exc_info.value)
    assert exit_code == expected_exit
    captured = capsys.readouterr()
    return CliRun(
        returncode=exit_code,
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


def test_run_simulation_help_uses_modes(
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
    assert "Run the unified IVQR Monte Carlo simulation" in result.stdout
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
    assert "alpha_min = -1.0" in result.stdout
    assert "alpha_max = 3.0" in result.stdout
    assert "alpha_grid_size = 21" in result.stdout
    assert "alpha_grid_step = 0.2" in result.stdout
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
    assert "alpha_min = -1.0" in result.stdout
    assert "alpha_max = 3.0" in result.stdout
    assert "alpha_grid_size = 21" in result.stdout
    assert "alpha_grid_step = 0.2" in result.stdout
    assert str(output) in result.stdout
    assert "Reports: automatic after successful run" in result.stdout
    assert not output.exists()


@pytest.mark.parametrize(
    (
        "mode",
        "expected_output",
        "expected_summary",
        "expected_tables",
        "expected_figures",
    ),
    [
        (
            "fast",
            Path("results/raw/fast_mode_results.csv"),
            Path("results/summary/fast_mode_summary.csv"),
            Path("results/tables/fast"),
            Path("results/figures/fast"),
        ),
        (
            "full",
            Path("results/raw/full_mode_results.csv"),
            Path("results/summary/full_mode_summary.csv"),
            Path("results/tables/full"),
            Path("results/figures/full"),
        ),
    ],
)
def test_main_mode_defaults_use_separate_report_paths(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_output: Path,
    expected_summary: Path,
    expected_tables: Path,
    expected_figures: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [str(main_cli.__file__), "--mode", mode, "--dry-run"],
    )
    args = main_cli._parse_args()
    main_cli._apply_mode_defaults(args)

    assert args.output == expected_output
    assert args.summary_output == expected_summary
    assert args.tables_dir == expected_tables
    assert args.figures_dir == expected_figures


def test_main_simulation_cli_runs_only_requested_estimators(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "raw" / "selected.csv"
    manifest = tmp_path / "selected_manifest.json"
    captured_estimators: list[tuple[str, ...]] = []
    captured_multiplier: list[float] = []

    def fake_run_simulation_batch(designs, alphas, **kwargs):
        estimators = tuple(kwargs["estimators"])
        captured_estimators.append(estimators)
        captured_multiplier.append(kwargs["critical_value_multiplier"])
        output_names = {
            "oracle": "oracle",
            "dml": "dml_ivqr",
            "post_selection": "post_selection_ivqr",
            "post_selection_quantile": "post_selection_quantile",
            "post_selection_ivqr_aligned": "post_selection_ivqr_aligned",
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "estimator": [output_names[estimator] for estimator in estimators]
            }
        ).to_csv(output, index=False)

    monkeypatch.setattr(main_cli, "run_simulation_batch", fake_run_simulation_batch)
    monkeypatch.setattr(main_cli, "_make_reports", lambda args: None)

    result = _run_cli_in_process(
        main_cli,
        [
            "--estimators",
            "oracle",
            "post-selection",
            "--alpha-min",
            "-2",
            "--alpha-max",
            "4",
            "--alpha-grid-size",
            "31",
            "--reps",
            "1",
            "--dgps",
            "dgp1",
            "--n-values",
            "80",
            "--p-values",
            "5",
            "--pi-values",
            "1.0",
            "--taus",
            "0.5",
            "--max-designs",
            "1",
            "--n-jobs",
            "1",
            "--critical-value-multiplier",
            "1.1",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert result.returncode == 0
    assert captured_estimators == [("oracle", "post_selection")]
    assert captured_multiplier == [pytest.approx(1.1)]
    written = pd.read_csv(output)
    assert set(written["estimator"]) == {"oracle", "post_selection_ivqr"}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["estimators"] == ["oracle", "post_selection"]
    assert payload["resume_signature"]["estimators"] == ["oracle", "post_selection"]
    assert payload["resume_signature"]["alpha_min"] == -2.0
    assert payload["resume_signature"]["alpha_max"] == 4.0
    assert payload["resume_signature"]["alpha_grid_size"] == 31
    assert payload["resume_signature"]["critical_value_multiplier"] == pytest.approx(1.1)
    assert payload["alpha_grid"]["min"] == pytest.approx(-2.0)
    assert payload["alpha_grid"]["max"] == pytest.approx(4.0)
    assert payload["alpha_grid"]["size"] == 31
    alpha_values = payload["alpha_grid"]["values"]
    assert alpha_values[1] - alpha_values[0] == pytest.approx(0.2)


def test_main_simulation_cli_runs_quantile_post_selection_only(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "raw" / "psq.csv"
    manifest = tmp_path / "psq_manifest.json"
    captured_estimators: list[tuple[str, ...]] = []

    def fake_run_simulation_batch(designs, alphas, **kwargs):
        captured_estimators.append(tuple(kwargs["estimators"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"estimator": ["post_selection_quantile"]}).to_csv(
            output,
            index=False,
        )

    monkeypatch.setattr(main_cli, "run_simulation_batch", fake_run_simulation_batch)
    monkeypatch.setattr(main_cli, "_make_reports", lambda args: None)

    result = _run_cli_in_process(
        main_cli,
        [
            "--estimators",
            "post_selection_quantile",
            "--reps",
            "1",
            "--dgps",
            "dgp1",
            "--n-values",
            "80",
            "--p-values",
            "5",
            "--pi-values",
            "1.0",
            "--taus",
            "0.5",
            "--max-designs",
            "1",
            "--n-jobs",
            "1",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert result.returncode == 0
    assert captured_estimators == [("post_selection_quantile",)]
    written = pd.read_csv(output)
    assert set(written["estimator"]) == {"post_selection_quantile"}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["estimators"] == ["post_selection_quantile"]
    assert payload["resume_signature"]["estimators"] == ["post_selection_quantile"]


def test_main_simulation_cli_runs_aligned_post_selection_only(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "raw" / "psa.csv"
    manifest = tmp_path / "psa_manifest.json"
    captured_estimators: list[tuple[str, ...]] = []

    def fake_run_simulation_batch(designs, alphas, **kwargs):
        captured_estimators.append(tuple(kwargs["estimators"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"estimator": ["post_selection_ivqr_aligned"]}).to_csv(
            output,
            index=False,
        )

    monkeypatch.setattr(main_cli, "run_simulation_batch", fake_run_simulation_batch)
    monkeypatch.setattr(main_cli, "_make_reports", lambda args: None)

    result = _run_cli_in_process(
        main_cli,
        [
            "--estimators",
            "post_selection_ivqr_aligned",
            "--reps",
            "1",
            "--dgps",
            "dgp1",
            "--n-values",
            "80",
            "--p-values",
            "5",
            "--pi-values",
            "1.0",
            "--taus",
            "0.5",
            "--max-designs",
            "1",
            "--n-jobs",
            "1",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert result.returncode == 0
    assert captured_estimators == [("post_selection_ivqr_aligned",)]
    written = pd.read_csv(output)
    assert set(written["estimator"]) == {"post_selection_ivqr_aligned"}
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["estimators"] == ["post_selection_ivqr_aligned"]
    assert payload["resume_signature"]["estimators"] == ["post_selection_ivqr_aligned"]


def test_run_simulation_accepts_full_control_estimator(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = _run_cli_in_process(
        main_cli,
        ["--estimators", "full_control", "--dry-run"],
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert result.returncode == 0
    assert "Running estimators: full_control" in result.stdout


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
    assert "Mode: fast" in result.stdout
    assert "Replications per design: 500" in result.stdout
    assert "alpha_min = -1.0" in result.stdout
    assert "alpha_max = 3.0" in result.stdout
    assert "alpha_grid_size = 21" in result.stdout
    assert "alpha_grid_step = 0.2" in result.stdout
    assert str(output) in result.stdout
    assert "Reports: automatic after successful run" in result.stdout
    assert not output.exists()


def test_full_control_result_row_uses_authoritative_region_geometry(
    full_control_cli,
) -> None:
    design = Design("dgp1", n=80, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    result = EstimationResult(
        estimator="full_control_ivqr",
        alpha_hat=0.0,
        alpha_true=0.0,
        tau=0.5,
        converged=True,
        failed=False,
        message="ok",
        objective_value=0.0,
        at_grid_boundary=False,
        alpha_grid_size=4,
        failed_alpha_count=0,
        cr_lower=0.0,
        cr_upper=1.0,
        cr_length=0.4,
        cr_covers_true=True,
        cr_empty=False,
        cr_disconnected=True,
        selected_controls=5,
        runtime_seconds=0.1,
        cr_n_blocks=2,
        cr_hull_length=1.0,
        cr_accepted_alpha_count=4,
        cr_hits_any_boundary=True,
    )

    row = full_control_cli._result_to_row(
        design,
        result,
        np.array([0.0, 0.2, 0.8, 1.0]),
    )

    assert row["cr_length"] == pytest.approx(0.4)
    assert row["cr_hull_length"] == pytest.approx(1.0)
    assert row["cr_n_blocks"] == 2
    assert row["cr_disconnected"] is True
    assert "cr_accepted_alpha_count" in row
    assert "cr_hits_any_boundary" in row


def test_main_resume_without_manifest_exits_before_run(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "main.csv"
    result = _run_cli_in_process(
        main_cli,
        ["--resume", "--dry-run", "--output", str(output)],
        monkeypatch,
        capsys,
        tmp_path,
        expected_exit=2,
    )

    assert RESUME_REQUIRES_MANIFEST_MESSAGE in result.stderr
    assert not output.exists()


def test_full_control_resume_without_manifest_exits_before_run(
    full_control_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "full_control.csv"
    result = _run_cli_in_process(
        full_control_cli,
        ["--resume", "--dry-run", "--output", str(output)],
        monkeypatch,
        capsys,
        tmp_path,
        expected_exit=2,
    )

    assert RESUME_REQUIRES_MANIFEST_MESSAGE in result.stderr
    assert not output.exists()


def test_main_resume_with_missing_manifest_is_rejected(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [str(main_cli.__file__), "--mode", "fast"],
    )
    args = main_cli._parse_args()
    main_cli._apply_mode_defaults(args)
    missing_manifest = tmp_path / "missing_manifest.json"

    with pytest.raises(FileNotFoundError, match=RESUME_REQUIRES_EXISTING_MANIFEST_MESSAGE):
        main_cli._validate_resume_manifest(missing_manifest, args)


def test_full_control_resume_with_missing_manifest_is_rejected(
    full_control_cli,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(full_control_cli.__file__)])
    args = full_control_cli._parse_args()
    missing_manifest = tmp_path / "missing_manifest.json"

    with pytest.raises(FileNotFoundError, match=RESUME_REQUIRES_EXISTING_MANIFEST_MESSAGE):
        full_control_cli._validate_resume_manifest(missing_manifest, args)


def test_main_non_resume_run_allows_new_manifest_path(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "raw" / "selected.csv"
    manifest = tmp_path / "selected_manifest.json"

    def fake_run_simulation_batch(designs, alphas, **kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        return pd.DataFrame(
            [
                {
                    "dgp": "dgp1",
                    "n": 80,
                    "p": 5,
                    "pi": 1.0,
                    "tau": 0.5,
                    "rep": 0,
                    "seed": 12345,
                    "estimator": "oracle",
                }
            ]
        )

    monkeypatch.setattr(main_cli, "run_simulation_batch", fake_run_simulation_batch)
    monkeypatch.setattr(main_cli, "_make_reports", lambda args: None)
    monkeypatch.setattr(main_cli, "_count_rows", lambda path: 1)

    result = _run_cli_in_process(
        main_cli,
        [
            "--estimators",
            "oracle",
            "--reps",
            "1",
            "--dgps",
            "dgp1",
            "--n-values",
            "80",
            "--p-values",
            "5",
            "--pi-values",
            "1.0",
            "--taus",
            "0.5",
            "--max-designs",
            "1",
            "--n-jobs",
            "1",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert result.returncode == 0
    assert manifest.exists()


def test_main_resume_applies_chunking_before_filtering(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "main.csv"
    manifest = tmp_path / "main_manifest.json"
    pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": 80,
                "p": 20,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 1,
                "seed": 12346,
                "estimator": estimator,
            }
            for estimator in ("oracle", "post_selection_ivqr", "dml_ivqr")
        ]
    ).to_csv(output, index=False)
    captured_designs: list[Design] = []
    cli_args = [
        "--resume",
        "--reps",
        "4",
        "--dgps",
        "dgp1",
        "--n-values",
        "80",
        "--p-values",
        "20",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--chunk-index",
        "0",
        "--num-chunks",
        "2",
        "--batch-size",
        "10",
        "--n-jobs",
        "1",
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    ]
    monkeypatch.setattr(sys, "argv", [str(main_cli.__file__), *cli_args])
    args = main_cli._parse_args()
    main_cli._apply_mode_defaults(args)
    manifest.write_text(
        json.dumps({"resume_signature": main_cli._resume_signature(args)}),
        encoding="utf-8",
    )

    def fake_run_simulation_batch(designs, alphas, **kwargs):
        captured_designs.extend(designs)
        return pd.DataFrame()

    monkeypatch.setattr(main_cli, "run_simulation_batch", fake_run_simulation_batch)
    monkeypatch.setattr(main_cli, "_make_reports", lambda args: None)
    monkeypatch.setattr(main_cli, "_count_rows", lambda path: 3)

    _run_cli_in_process(
        main_cli,
        cli_args,
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert [design.rep for design in captured_designs] == [0, 2]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["total_designs"] == 4
    assert payload["chunk_designs"] == 2
    assert payload["pending_designs"] == 2
    assert payload["designs_in_run"] == 2
    assert payload["resume_signature"]["mode"] == "fast"
    assert payload["resume_signature"]["batch_size"] == 10
    assert payload["resume_signature"]["n_jobs"] == 1
    assert payload["resume_signature"]["estimators"] == [
        "oracle",
        "dml",
        "post_selection",
    ]
    assert payload["estimators"] == ["oracle", "dml", "post_selection"]


def test_full_control_resume_applies_chunking_before_filtering(
    full_control_cli,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "full_control.csv"
    manifest = tmp_path / "full_control_manifest.json"
    pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": 80,
                "p": 20,
                "pi": 1.0,
                "tau": 0.5,
                "rep": 1,
                "seed": 54322,
                "estimator": "full_control_ivqr",
            }
        ]
    ).to_csv(output, index=False)
    captured_designs: list[Design] = []
    cli_args = [
        "--resume",
        "--reps",
        "4",
        "--dgps",
        "dgp1",
        "--n-values",
        "80",
        "--p-values",
        "20",
        "--pi-values",
        "1.0",
        "--taus",
        "0.5",
        "--chunk-index",
        "0",
        "--num-chunks",
        "2",
        "--batch-size",
        "10",
        "--n-jobs",
        "1",
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    ]
    monkeypatch.setattr(sys, "argv", [str(full_control_cli.__file__), *cli_args])
    args = full_control_cli._parse_args()
    manifest.write_text(
        json.dumps(
            {"resume_signature": full_control_cli._resume_signature(args)}
        ),
        encoding="utf-8",
    )

    def fake_run_simulation_batch(designs, alphas, **kwargs):
        captured_designs.extend(designs)
        return pd.DataFrame()

    monkeypatch.setattr(
        full_control_cli._unified,
        "run_simulation_batch",
        fake_run_simulation_batch,
    )
    monkeypatch.setattr(full_control_cli._unified, "_make_reports", lambda args: None)
    monkeypatch.setattr(full_control_cli._unified, "_count_rows", lambda path: 1)

    _run_cli_in_process(
        full_control_cli,
        cli_args,
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert [design.rep for design in captured_designs] == [0, 2]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["total_designs"] == 4
    assert payload["chunk_designs"] == 2
    assert payload["pending_designs"] == 2
    assert payload["designs_in_run"] == 2
    assert "resume_signature" in payload
    assert payload["resume_signature"]["batch_size"] == 10
    assert payload["resume_signature"]["n_jobs"] == 1
    assert payload["resume_signature"]["estimators"] == ["full_control"]
    assert payload["estimators"] == ["full_control"]


def test_full_control_resume_key_rejects_decimal_integer_fields(
    full_control_cli,
) -> None:
    row = pd.Series(
        {
            "dgp": "dgp1",
            "n": 80.5,
            "p": 5,
            "pi": 1.0,
            "tau": 0.5,
            "rep": 0,
            "seed": 123,
        }
    )

    with pytest.raises(ValueError, match="invalid design-key values"):
        full_control_cli._row_design_key(row)


def test_full_control_as_bool_uses_explicit_parsing(full_control_cli) -> None:
    assert full_control_cli._as_bool(1) is True
    assert full_control_cli._as_bool(0) is False
    assert full_control_cli._as_bool(2) is False
    assert full_control_cli._as_bool("yes") is True
    assert full_control_cli._as_bool("no") is False


def test_scenario_output_validation_rejects_directories(
    main_cli,
    full_control_cli,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="file path"):
        main_cli._validate_output_path(tmp_path, resume=False)
    with pytest.raises(ValueError, match="file path"):
        full_control_cli._validate_output_path(tmp_path, resume=False)

    parent_file = tmp_path / "parent"
    parent_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="parent must be a directory"):
        main_cli._validate_output_path(parent_file / "main.csv", resume=False)
    with pytest.raises(ValueError, match="parent must be a directory"):
        full_control_cli._validate_output_path(
            parent_file / "full_control.csv",
            resume=False,
        )


@pytest.mark.parametrize(
    "changed_field",
    [
        "alpha_min",
        "alpha_max",
        "alpha_grid_size",
        "estimators",
        "mode",
        "batch_size",
        "n_jobs",
        "critical_value_multiplier",
    ],
)
def test_main_resume_manifest_rejects_incompatible_settings(
    main_cli,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed_field: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [str(main_cli.__file__), "--mode", "fast"],
    )
    args = main_cli._parse_args()
    main_cli._apply_mode_defaults(args)
    manifest = tmp_path / "main_manifest.json"
    manifest.write_text(
        json.dumps({"resume_signature": main_cli._resume_signature(args)}),
        encoding="utf-8",
    )

    main_cli._validate_resume_manifest(manifest, args)
    if changed_field == "alpha_min":
        args.alpha_min -= 1
    elif changed_field == "alpha_max":
        args.alpha_max += 1
    elif changed_field == "alpha_grid_size":
        args.alpha_grid_size += 1
    elif changed_field == "estimators":
        args.estimators = ("oracle", "post_selection")
    elif changed_field == "batch_size":
        args.batch_size += 1
    elif changed_field == "n_jobs":
        args.n_jobs += 1
    elif changed_field == "critical_value_multiplier":
        args.critical_value_multiplier = 1.1
    else:
        args.mode = "full"

    with pytest.raises(ValueError, match="resume signature"):
        main_cli._validate_resume_manifest(manifest, args)


@pytest.mark.parametrize(
    "changed_field",
    ["alpha_grid_size", "batch_size", "n_jobs", "critical_value_multiplier"],
)
def test_full_control_resume_manifest_rejects_incompatible_settings(
    full_control_cli,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed_field: str,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(full_control_cli.__file__)])
    args = full_control_cli._parse_args()
    manifest = tmp_path / "full_control_manifest.json"
    manifest.write_text(
        json.dumps({"resume_signature": full_control_cli._resume_signature(args)}),
        encoding="utf-8",
    )

    full_control_cli._validate_resume_manifest(manifest, args)
    if changed_field == "alpha_grid_size":
        args.alpha_grid_size += 1
    elif changed_field == "batch_size":
        args.batch_size += 1
    elif changed_field == "n_jobs":
        args.n_jobs += 1
    else:
        args.critical_value_multiplier = 1.1

    with pytest.raises(ValueError, match="resume signature"):
        full_control_cli._validate_resume_manifest(manifest, args)


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
