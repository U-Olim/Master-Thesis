"""Regenerate provenance metadata for the three canonical raw result files."""

import json
from pathlib import Path
import sys
from typing import TypedDict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from analysis.data import RAW_MANIFEST_PATH, RAW_RESULT_FILES, sha256_file  # noqa: E402


class RawFileManifestEntry(TypedDict):
    estimator: str
    path: str
    filename: str
    rows: int
    columns: int
    size_bytes: int
    sha256: str


class RawManifest(TypedDict):
    schema_version: int
    number_of_files: int
    total_rows: int
    files: list[RawFileManifestEntry]
    final_run_metadata: dict[str, float | int | str]


FINAL_RUN_METADATA = {
    "alpha_grid_min": -1.0,
    "alpha_grid_max": 3.0,
    "alpha_grid_size": 21,
    "base_seed": 12345,
    "critical_value_multiplier": 1.0,
    "post_selection_lasso_multiplier": 1.8,
    "dml_k_folds": 3,
    "dml_quantile_penalty": 0.07,
    "dml_quantile_solver": "highs-ipm",
    "dml_ridge_alpha": 1.0,
}


def main() -> None:
    files: list[RawFileManifestEntry] = []
    for estimator in ("oracle", "post_selection", "dml"):
        path = RAW_RESULT_FILES[estimator]
        if not path.is_file():
            raise FileNotFoundError(f"Canonical raw result does not exist: {path}")
        frame = pd.read_csv(path)
        files.append(
            {
                "estimator": estimator,
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "filename": path.name,
                "rows": len(frame),
                "columns": len(frame.columns),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest: RawManifest = {
        "schema_version": 1,
        "number_of_files": len(files),
        "total_rows": sum(item["rows"] for item in files),
        "files": files,
        "final_run_metadata": FINAL_RUN_METADATA,
    }
    RAW_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {RAW_MANIFEST_PATH.relative_to(PROJECT_ROOT)} "
        f"for {len(files)} files and {manifest['total_rows']:,} rows"
    )


if __name__ == "__main__":
    main()
