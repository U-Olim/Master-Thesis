"""Command-line entry point for deterministic Phase 5 release checks."""

from __future__ import annotations

from pathlib import Path

from analysis.r500_release_hygiene import PHASE5_FILENAMES, finalize_release_hygiene


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    status = finalize_release_hygiene(PROJECT_ROOT)
    print(f"Release classification: {status['classification']}")
    print(f"Wrote {len(PHASE5_FILENAMES)} deterministic Phase 5 files")
    if status["classification"] == "not_ready":
        raise RuntimeError("Phase 5 found a blocking release-hygiene defect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
