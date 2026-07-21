import ast
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ESTIMATOR_OUTPUT_MODULES = {
    "simulation.oracle_output",
    "simulation.post_selection_output",
    "simulation.dml_output",
}


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env=env,
        timeout=120,
    )


def _imported_modules(relative_path: str) -> set[str]:
    source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def test_direct_ch_inverse_import_works_in_clean_process() -> None:
    result = _run_python(
        "from ivqr.ch_inverse import evaluate_alpha_ch_ivqr; "
        "print(evaluate_alpha_ch_ivqr.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "evaluate_alpha_ch_ivqr"


def test_public_estimator_imports_work_in_clean_process() -> None:
    result = _run_python(
        "from estimators import EstimationResult, estimate_oracle_ivqr, "
        "estimate_post_selection_ivqr, estimate_dml_ivqr; "
        "print(EstimationResult.__name__, estimate_oracle_ivqr.__name__, "
        "estimate_post_selection_ivqr.__name__, estimate_dml_ivqr.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "EstimationResult estimate_oracle_ivqr estimate_post_selection_ivqr "
        "estimate_dml_ivqr"
    )


def test_ch_inverse_then_public_estimator_import_has_no_cycle() -> None:
    result = _run_python(
        "from ivqr.ch_inverse import evaluate_alpha_ch_ivqr; "
        "from estimators import estimate_oracle_ivqr; "
        "print(evaluate_alpha_ch_ivqr.__name__, estimate_oracle_ivqr.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "evaluate_alpha_ch_ivqr estimate_oracle_ivqr"


def test_shared_output_modules_do_not_import_estimator_serializers() -> None:
    for module in (
        "src/simulation/output_schemas.py",
        "src/simulation/output_validation.py",
    ):
        assert _imported_modules(module).isdisjoint(ESTIMATOR_OUTPUT_MODULES)


def test_oracle_and_post_selection_do_not_import_dml_output() -> None:
    for module in (
        "src/simulation/oracle_output.py",
        "src/simulation/post_selection_output.py",
    ):
        assert "simulation.dml_output" not in _imported_modules(module)


def test_shared_output_modules_import_cleanly_in_fresh_process() -> None:
    result = _run_python(
        "from simulation.output_schemas import OUTPUT_COLUMNS_BY_ESTIMATOR; "
        "from simulation.output_validation import validate_component_columns; "
        "print(tuple(OUTPUT_COLUMNS_BY_ESTIMATOR), "
        "validate_component_columns.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "('oracle', 'post_selection', 'dml') validate_component_columns"
    )


def test_public_simulation_config_imports_work_in_clean_process() -> None:
    result = _run_python(
        "from simulation.config import ExecutionConfig, DesignConfig, "
        "AlphaGridConfig, CHInferenceConfig, PostSelectionConfig, DMLConfig, "
        "OracleRunConfig, PostSelectionRunConfig, DMLRunConfig, "
        "build_estimator_run_config; "
        "print(ExecutionConfig.__name__, OracleRunConfig.__name__, "
        "DMLRunConfig.__name__, build_estimator_run_config.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "ExecutionConfig OracleRunConfig DMLRunConfig "
        "build_estimator_run_config"
    )


def test_execution_modules_and_runner_facade_import_cleanly() -> None:
    result = _run_python(
        "from simulation.seeds import make_design_seed; "
        "from simulation.designs import make_simulation_grid; "
        "from simulation.dispatch import normalize_estimator_names; "
        "from simulation.execution import run_simulation_design, run_simulation_batch; "
        "from simulation.resume import filter_completed_designs; "
        "from simulation.persistence import persist_results_frame; "
        "from simulation.runner import run_single_replication; "
        "print(make_design_seed.__name__, make_simulation_grid.__name__, "
        "normalize_estimator_names.__name__, run_simulation_design.__name__, "
        "run_simulation_batch.__name__, filter_completed_designs.__name__, "
        "persist_results_frame.__name__, run_single_replication.__name__)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (
        "make_design_seed make_simulation_grid normalize_estimator_names "
        "run_simulation_design run_simulation_batch filter_completed_designs "
        "persist_results_frame run_single_replication"
    )


def test_low_level_execution_modules_do_not_import_cli_scripts() -> None:
    for module in (
        "src/simulation/seeds.py",
        "src/simulation/designs.py",
        "src/simulation/dispatch.py",
        "src/simulation/execution.py",
        "src/simulation/resume.py",
        "src/simulation/persistence.py",
    ):
        imported = _imported_modules(module)
        assert "scenarios.run_simulation" not in imported
        assert all(not name.startswith("scenarios.") for name in imported)
    assert all(
        "output" not in name
        for name in _imported_modules("src/simulation/seeds.py")
    )
    assert all(
        "output" not in name
        for name in _imported_modules("src/simulation/designs.py")
    )
    assert all(
        not name.startswith("estimators")
        for name in _imported_modules("src/simulation/persistence.py")
    )


def test_dedicated_cli_import_direction_and_retired_stub_isolation() -> None:
    for module in (
        "scenarios/run_oracle_ivqr.py",
        "scenarios/run_post_selection_ivqr.py",
        "scenarios/run_dml_ivqr.py",
        "scenarios/_dedicated_runner.py",
        "scenarios/_cli_common.py",
    ):
        assert "scenarios.run_simulation" not in _imported_modules(module)
    stub_imports = _imported_modules("scenarios/run_simulation.py")
    assert all(not name.startswith("simulation") for name in stub_imports)
    assert all(not name.startswith("dgp") for name in stub_imports)
    assert all(not name.startswith("estimators") for name in stub_imports)
    assert all(not name.startswith("ivqr") for name in stub_imports)
