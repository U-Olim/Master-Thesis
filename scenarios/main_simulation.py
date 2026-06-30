import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = PROJECT_ROOT / "scenarios"
SRC_PATH = PROJECT_ROOT / "src"
if str(SCENARIOS_PATH) not in sys.path:
    sys.path.insert(0, str(SCENARIOS_PATH))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from run_simulation import main  # noqa: E402


if __name__ == "__main__":
    main()
