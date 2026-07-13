"""Generate final thesis tables and figures from completed R=500 results."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from analysis.data import RAW_RESULT_FILES, load_all_results  # noqa: E402
from analysis.figures import write_all_figures  # noqa: E402
from analysis.tables import write_all_tables  # noqa: E402


def main() -> None:
    results = load_all_results()
    table_files = write_all_tables(results, PROJECT_ROOT / "results" / "tables")
    figure_files = write_all_figures(results, PROJECT_ROOT / "results" / "figures")

    print(f"Validated {len(results):,} rows from {len(RAW_RESULT_FILES)} R=500 datasets:")
    for estimator, path in RAW_RESULT_FILES.items():
        rows = int(results["estimator"].eq(estimator).sum())
        print(f"  {estimator}: {path.relative_to(PROJECT_ROOT)} ({rows:,} rows)")
    print(f"Wrote {sum(len(paths) for paths in table_files.values())} table files to results/tables")
    print(f"Wrote {sum(len(paths) for paths in figure_files.values())} figure files to results/figures")


if __name__ == "__main__":
    main()
