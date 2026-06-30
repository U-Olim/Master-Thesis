"""Simulation constants for the IVQR Monte Carlo study."""

N_VALUES: tuple[int, ...] = (500, 1000)
P_VALUES: tuple[int, ...] = (200, 500)
PI_VALUES: tuple[float, ...] = (1.0, 0.5, 0.25, 0.10)
TAUS: tuple[float, ...] = (0.25, 0.50, 0.75)
DGPS: tuple[str, ...] = ("dgp1", "dgp2", "dgp3")

ESTIMATORS: tuple[str, ...] = ("oracle", "post_selection", "full_control", "dml")
DEFAULT_ESTIMATORS: tuple[str, ...] = ("oracle", "post_selection", "dml")

R_FAST: int = 10
R_FULL: int = 500

DEFAULT_ALPHA_MIN: float = -1.0
DEFAULT_ALPHA_MAX: float = 3.0
DEFAULT_ALPHA_GRID_SIZE: int = 21

DEFAULT_N_JOBS: int = 4
DEFAULT_BATCH_SIZE: int = 10
DEFAULT_BASE_SEED: int = 12345
DEFAULT_QUANTREG_MAX_ITER: int = 1000
DEFAULT_DML_K_FOLDS: int = 3
DEFAULT_CRITICAL_VALUE_MULTIPLIER: float = 1.0

RHO_X: float = 0.5
RHO_UV: float = 0.5
DF_T: int = 5

FAST_OUTPUT: str = "results/raw/fast_results.csv"
FULL_OUTPUT: str = "results/raw/full_results.csv"

FAST_SUMMARY_OUTPUT: str = "results/summary/fast_summary.csv"
FULL_SUMMARY_OUTPUT: str = "results/summary/full_summary.csv"

FAST_TABLES_DIR: str = "results/tables/fast"
FULL_TABLES_DIR: str = "results/tables/full"

FAST_FIGURES_DIR: str = "results/figures/fast"
FULL_FIGURES_DIR: str = "results/figures/full"


__all__ = [
    "DEFAULT_ALPHA_GRID_SIZE",
    "DEFAULT_ALPHA_MAX",
    "DEFAULT_ALPHA_MIN",
    "DEFAULT_BASE_SEED",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CRITICAL_VALUE_MULTIPLIER",
    "DEFAULT_DML_K_FOLDS",
    "DEFAULT_ESTIMATORS",
    "DEFAULT_N_JOBS",
    "DEFAULT_QUANTREG_MAX_ITER",
    "DF_T",
    "DGPS",
    "ESTIMATORS",
    "FAST_FIGURES_DIR",
    "FAST_OUTPUT",
    "FAST_SUMMARY_OUTPUT",
    "FAST_TABLES_DIR",
    "FULL_FIGURES_DIR",
    "FULL_OUTPUT",
    "FULL_SUMMARY_OUTPUT",
    "FULL_TABLES_DIR",
    "N_VALUES",
    "PI_VALUES",
    "P_VALUES",
    "R_FAST",
    "R_FULL",
    "RHO_UV",
    "RHO_X",
    "TAUS",
]
