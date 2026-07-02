"""Simulation grid, execution, and result-row utilities."""

from typing import Any

__all__ = [
    "make_design_seed",
    "make_simulation_grid",
    "run_simulation_batch",
    "run_simulation_design",
]


def make_design_seed(*args: Any, **kwargs: Any) -> int:
    """Lazily dispatch to the deterministic design-seed helper."""
    from simulation.runner import make_design_seed as _make_design_seed

    return _make_design_seed(*args, **kwargs)


def make_simulation_grid(*args: Any, **kwargs: Any) -> Any:
    """Lazily dispatch to the simulation-grid builder."""
    from simulation.runner import make_simulation_grid as _make_simulation_grid

    return _make_simulation_grid(*args, **kwargs)


def run_simulation_batch(*args: Any, **kwargs: Any) -> Any:
    """Lazily dispatch to the batch simulation runner."""
    from simulation.runner import run_simulation_batch as _run_simulation_batch

    return _run_simulation_batch(*args, **kwargs)


def run_simulation_design(*args: Any, **kwargs: Any) -> Any:
    """Lazily dispatch to the single-design simulation runner."""
    from simulation.runner import run_simulation_design as _run_simulation_design

    return _run_simulation_design(*args, **kwargs)
