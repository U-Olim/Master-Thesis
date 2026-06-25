import os
from pathlib import Path

import pytest

from tests.helpers import (
    PROJECT_ROOT,
    SCENARIOS_DIR,
    SIMULATION_RESULT_REQUIRED_KEYS,
    load_full_control_cli,
    load_main_simulation_cli,
    load_module_from_path,
)


def test_project_root_and_scenarios_dir_exist() -> None:
    assert PROJECT_ROOT.exists()
    assert SCENARIOS_DIR.exists()
    assert (SCENARIOS_DIR / "main_simulation.py").is_file()
    assert (SCENARIOS_DIR / "full_control_ivqr.py").is_file()


def test_simulation_result_required_keys_is_immutable() -> None:
    assert isinstance(SIMULATION_RESULT_REQUIRED_KEYS, frozenset)
    assert "alpha_hat" in SIMULATION_RESULT_REQUIRED_KEYS
    with pytest.raises(AttributeError):
        SIMULATION_RESULT_REQUIRED_KEYS.add("new_key")  # type: ignore[attr-defined]


def test_load_module_from_path_validates_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="module_name"):
        load_module_from_path("", tmp_path / "x.py")

    with pytest.raises(FileNotFoundError):
        load_module_from_path("missing_module", tmp_path / "missing.py")

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="module path must be a file"):
        load_module_from_path("directory_module", directory)


def test_cli_loader_helpers_return_modules_with_main() -> None:
    main_cli = load_main_simulation_cli()
    full_control_cli = load_full_control_cli()

    assert callable(main_cli.main)
    assert callable(full_control_cli.main)


def test_matplotlib_test_cache_is_isolated() -> None:
    assert os.environ["MPLBACKEND"] == "Agg"
    assert ".pytest_tmp" in os.environ["MPLCONFIGDIR"]
    assert Path(os.environ["MPLCONFIGDIR"]).exists()
