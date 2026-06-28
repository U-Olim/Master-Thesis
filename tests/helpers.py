"""Shared helpers for tests that need scenario loading or result schemas."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SCENARIOS_DIR: Path = PROJECT_ROOT / "scenarios"


def load_module_from_path(module_name: str, path: str | Path) -> ModuleType:
    if not isinstance(module_name, str) or not module_name:
        raise ValueError("module_name must be a nonempty string")

    module_path = Path(path)
    if not module_path.exists():
        raise FileNotFoundError(module_path)
    if not module_path.is_file():
        raise ValueError("module path must be a file")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_main_simulation_cli() -> ModuleType:
    return load_module_from_path(
        "main_simulation_cli",
        SCENARIOS_DIR / "main_simulation.py",
    )


def load_full_control_cli() -> ModuleType:
    return load_module_from_path(
        "full_control_cli",
        SCENARIOS_DIR / "full_control_ivqr.py",
    )


SIMULATION_RESULT_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "dgp",
        "n",
        "p",
        "pi",
        "tau",
        "rep",
        "seed",
        "estimator",
        "alpha_hat",
        "alpha_true",
        "bias",
        "absolute_error",
        "squared_error",
        "status",
        "error_type",
        "error_message",
        "failed",
        "converged",
        "alpha_grid_min",
        "alpha_grid_max",
        "alpha_grid_size",
        "alpha_grid_step",
        "alpha_hat_at_lower_boundary",
        "alpha_hat_at_upper_boundary",
        "alpha_hat_at_any_boundary",
        "cr_lower",
        "cr_upper",
        "cr_length",
        "cr_hits_lower_boundary",
        "cr_hits_upper_boundary",
        "cr_hits_any_boundary",
        "cr_empty",
        "cr_accepted_alpha_count",
        "cr_acceptance_rate",
        "cr_n_blocks",
        "cr_disconnected",
        "cr_hull_length",
        "cr_covers_true",
        "selected_controls",
        "runtime_seconds",
        "failed_alpha_count",
        "failed_alpha_rate",
        "min_test_stat",
        "max_test_stat",
        "test_stat_at_alpha_hat",
        "critical_value",
        "ps_n_selected_controls",
        "ps_n_selected_instruments",
        "ps_n_selected_total",
        "ps_share_selected_controls",
        "ps_share_selected_instruments",
        "ps_selected_no_controls",
        "ps_selected_no_instruments",
        "ps_selected_empty_total",
        "ps_first_stage_r2",
        "ps_first_stage_adj_r2",
        "ps_first_stage_partial_r2",
        "ps_first_stage_f_stat",
        "ps_first_stage_condition_number",
        "ps_selection_method",
        "ps_lasso_alpha_controls",
        "ps_lasso_alpha_instruments",
        "ps_lasso_alpha_first_stage",
        "ps_lasso_cv_folds",
        "ps_selection_failed",
        "ps_first_stage_failed",
        "ps_rank_deficient",
        "ps_warning_code",
        "message",
    }
)


__all__ = [
    "PROJECT_ROOT",
    "SCENARIOS_DIR",
    "SIMULATION_RESULT_REQUIRED_KEYS",
    "load_full_control_cli",
    "load_main_simulation_cli",
    "load_module_from_path",
]
