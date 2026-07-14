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

from analysis.data import (  # noqa: E402
    RAW_MANIFEST_PATH,
    RAW_RESULT_FILES,
    sha256_canonical_lf_file,
    sha256_file,
)


class RawFileManifestEntry(TypedDict):
    estimator: str
    path: str
    filename: str
    rows: int
    columns: int
    size_bytes: int
    sha256_bytes: str
    sha256_canonical_lf: str


class CommonFinalRunMetadata(TypedDict):
    alpha_grid_min: float
    alpha_grid_max: float
    alpha_grid_size: int
    base_seed: int
    critical_value_multiplier: float


class OracleFinalRunMetadata(TypedDict):
    pixi_task: str


class PostSelectionFinalRunMetadata(TypedDict):
    pixi_task: str
    selection_lasso_multiplier: float


class DmlFinalRunMetadata(TypedDict):
    pixi_task: str
    k_folds: int
    quantile_penalty: float
    quantile_solver: str
    ridge_alpha: float


class EstimatorFinalRunMetadata(TypedDict):
    oracle: OracleFinalRunMetadata
    post_selection: PostSelectionFinalRunMetadata
    dml: DmlFinalRunMetadata


class FinalRunMetadata(TypedDict):
    common: CommonFinalRunMetadata
    estimators: EstimatorFinalRunMetadata


class RawManifest(TypedDict):
    schema_version: int
    number_of_files: int
    total_rows: int
    files: list[RawFileManifestEntry]
    final_run_metadata: FinalRunMetadata


FINAL_RUN_METADATA: FinalRunMetadata = {
    "common": {
        "alpha_grid_min": -1.0,
        "alpha_grid_max": 3.0,
        "alpha_grid_size": 21,
        "base_seed": 12345,
        "critical_value_multiplier": 1.0,
    },
    "estimators": {
        "oracle": {"pixi_task": "final_oracle"},
        "post_selection": {
            "pixi_task": "final_post_selection",
            "selection_lasso_multiplier": 1.8,
        },
        "dml": {
            "pixi_task": "final_dml",
            "k_folds": 3,
            "quantile_penalty": 0.07,
            "quantile_solver": "highs-ipm",
            "ridge_alpha": 1.0,
        },
    },
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
                "sha256_bytes": sha256_file(path),
                "sha256_canonical_lf": sha256_canonical_lf_file(path),
            }
        )

    manifest: RawManifest = {
        "schema_version": 3,
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
