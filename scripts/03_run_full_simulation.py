"""Wrapper for the full simulation command."""

from __future__ import annotations


try:
    from ivqr_sim.scripts.full_simulation import main
except ModuleNotFoundError as exc:
    if exc.name == "ivqr_sim":
        raise SystemExit(
            "Could not import the 'ivqr_sim' package.\n"
            "Install the project in editable mode from the repository root:\n\n"
            "    pip install -e .\n\n"
            "Then rerun this script.\n"
            "Alternative for temporary local execution:\n\n"
            "    PYTHONPATH=src python scripts/03_run_full_simulation.py\n"
        ) from exc
    raise


if __name__ == "__main__":
    main()
