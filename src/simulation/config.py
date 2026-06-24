"""Project-wide simulation configuration constants.

The main simulation compares oracle IVQR, post-selection IVQR, and DML-IVQR
over the thesis DGPs, sample sizes, instrument strengths, and quantiles.

The full-control IVQR benchmark is intentionally separated because it uses all
controls without selection or regularization and is computationally expensive
in high-dimensional designs.
"""

N_VALUES: tuple[int, ...] = (500, 1000)
P_VALUES: tuple[int, ...] = (200, 500)
PI_VALUES: tuple[float, ...] = (1.0, 0.5, 0.25, 0.10)
TAUS: tuple[float, ...] = (0.25, 0.50, 0.75)
DGPS: tuple[str, ...] = ("dgp1", "dgp2", "dgp3")
MAIN_ESTIMATORS: tuple[str, ...] = ("oracle", "post_selection", "dml")

R_FAST: int = 10
R_MAIN: int = 500
R_FULL_CONTROL_BENCHMARK: int = 500

FAST_OUTPUT: str = "results/raw/fast_mode_results.csv"
FULL_OUTPUT: str = "results/raw/full_mode_results.csv"
DEFAULT_OUTPUT: str = FULL_OUTPUT

# A coarse nine-point alpha grid keeps the Monte Carlo runtime manageable.
DEFAULT_ALPHA_GRID_SIZE: int = 9
DEFAULT_DML_K_FOLDS: int = 3
DEFAULT_N_JOBS: int = 6
DEFAULT_BATCH_SIZE: int = 10
DEFAULT_CHUNK_COUNT: int = 1
DEFAULT_QUANTREG_MAX_ITER: int = 1000
K_FOLDS: int = DEFAULT_DML_K_FOLDS

RHO_X: float = 0.5
RHO_UV: float = 0.5
DF_T: int = 5

FULL_CONTROL_BENCHMARK_ESTIMATOR: str = "full_control_ivqr"
FULL_CONTROL_BENCHMARK_DGPS: tuple[str, ...] = ("dgp1",)
FULL_CONTROL_BENCHMARK_N_VALUES: tuple[int, ...] = (500, 1000)
# Full-control IVQR is run separately on smaller p because it is slow and can be unstable.
FULL_CONTROL_BENCHMARK_P_VALUES: tuple[int, ...] = (20, 50, 100)
FULL_CONTROL_BENCHMARK_PI_VALUES: tuple[float, ...] = (1.0,)
FULL_CONTROL_BENCHMARK_TAUS: tuple[float, ...] = (0.25, 0.5, 0.75)
FULL_CONTROL_BENCHMARK_OUTPUT: str = "results/raw/full_control_ivqr_results.csv"
FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE: int = 9


__all__ = [
    "N_VALUES",
    "P_VALUES",
    "PI_VALUES",
    "TAUS",
    "DGPS",
    "MAIN_ESTIMATORS",
    "R_FAST",
    "R_MAIN",
    "R_FULL_CONTROL_BENCHMARK",
    "FAST_OUTPUT",
    "FULL_OUTPUT",
    "DEFAULT_OUTPUT",
    "DEFAULT_ALPHA_GRID_SIZE",
    "DEFAULT_DML_K_FOLDS",
    "DEFAULT_N_JOBS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHUNK_COUNT",
    "DEFAULT_QUANTREG_MAX_ITER",
    "K_FOLDS",
    "RHO_X",
    "RHO_UV",
    "DF_T",
    "FULL_CONTROL_BENCHMARK_ESTIMATOR",
    "FULL_CONTROL_BENCHMARK_DGPS",
    "FULL_CONTROL_BENCHMARK_N_VALUES",
    "FULL_CONTROL_BENCHMARK_P_VALUES",
    "FULL_CONTROL_BENCHMARK_PI_VALUES",
    "FULL_CONTROL_BENCHMARK_TAUS",
    "FULL_CONTROL_BENCHMARK_OUTPUT",
    "FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE",
]
