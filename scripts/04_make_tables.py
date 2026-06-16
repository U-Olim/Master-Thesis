"""Wrapper for thesis table generation."""

from __future__ import annotations


try:
    from ivqr_sim.scripts.make_tables import main
except ModuleNotFoundError as exc:
    if exc.name == "ivqr_sim":
        raise SystemExit(
            "Could not import the 'ivqr_sim' package.\n"
            "Install the project in editable mode from the repository root:\n\n"
            "    pip install -e .\n\n"
            "Then rerun this script.\n"
            "Alternative for temporary local execution:\n\n"
            "    PYTHONPATH=src python scripts/04_make_tables.py\n"
        ) from exc
    raise


if __name__ == "__main__":
    main()
