"""Simulation grid, execution, and result-row utilities."""

__all__ = [
    "make_design_seed",
    "make_simulation_grid",
    "run_simulation_batch",
    "run_simulation_design",
]


def __getattr__(name: str):
    if name in __all__:
        from simulation import runner

        return getattr(runner, name)
    raise AttributeError(name)
