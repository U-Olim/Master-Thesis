"""Project-wide simulation configuration constants.

The main simulation compares oracle IVQR, post-selection IVQR, and DML-style
residualized IVQR over the thesis DGPs, sample sizes, instrument strengths,
and quantiles.

The full-control IVQR benchmark is intentionally separated because it uses all
controls without selection or regularization and is computationally expensive
in high-dimensional designs.
"""

N_VALUES: tuple[int, ...] = (500, 1000)
P_VALUES: tuple[int, ...] = (200, 500)
PI_VALUES: tuple[float, ...] = (1.0, 0.5, 0.25, 0.10)
TAUS: tuple[float, ...] = (0.25, 0.50, 0.75)
DGPS: tuple[str, ...] = ("dgp1", "dgp2", "dgp3")
MAIN_ESTIMATORS: tuple[str, ...] = (
    "oracle",
    "dml",
    "post_selection",
    "post_selection_quantile",
    "post_selection_ivqr_aligned",
)
DEFAULT_MAIN_ESTIMATORS: tuple[str, ...] = ("oracle", "dml", "post_selection")

R_FAST: int = 10
R_MAIN: int = 500
R_FULL_CONTROL_BENCHMARK: int = 500

FAST_OUTPUT: str = "results/raw/fast_mode_results.csv"
FULL_OUTPUT: str = "results/raw/full_mode_results.csv"
DEFAULT_OUTPUT: str = FULL_OUTPUT
FAST_SUMMARY_OUTPUT: str = "results/summary/fast_mode_summary.csv"
FULL_SUMMARY_OUTPUT: str = "results/summary/full_mode_summary.csv"
FAST_TABLES_DIR: str = "results/tables/fast"
FULL_TABLES_DIR: str = "results/tables/full"
FAST_FIGURES_DIR: str = "results/figures/fast"
FULL_FIGURES_DIR: str = "results/figures/full"

DEFAULT_ALPHA_MIN: float = -1.0
DEFAULT_ALPHA_MAX: float = 3.0
DEFAULT_ALPHA_GRID_SIZE: int = 21
DEFAULT_DML_K_FOLDS: int = 3
DEFAULT_N_JOBS: int = 4
DEFAULT_BATCH_SIZE: int = 10
DEFAULT_QUANTREG_MAX_ITER: int = 1000
DEFAULT_QUANTILE_SELECTION_ALPHAS: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1)
DEFAULT_QUANTILE_SELECTION_CV_FOLDS: int = 3
DEFAULT_SELECTION_COEF_TOL: float = 1e-10
DEFAULT_CRITICAL_VALUE_MULTIPLIER: float = 1.0

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
FULL_CONTROL_BENCHMARK_ALPHA_GRID_SIZE: int = DEFAULT_ALPHA_GRID_SIZE


__all__ = [
    "N_VALUES",
    "P_VALUES",
    "PI_VALUES",
    "TAUS",
    "DGPS",
    "MAIN_ESTIMATORS",
    "DEFAULT_MAIN_ESTIMATORS",
    "R_FAST",
    "R_MAIN",
    "R_FULL_CONTROL_BENCHMARK",
    "FAST_OUTPUT",
    "FULL_OUTPUT",
    "DEFAULT_OUTPUT",
    "FAST_SUMMARY_OUTPUT",
    "FULL_SUMMARY_OUTPUT",
    "FAST_TABLES_DIR",
    "FULL_TABLES_DIR",
    "FAST_FIGURES_DIR",
    "FULL_FIGURES_DIR",
    "DEFAULT_ALPHA_MIN",
    "DEFAULT_ALPHA_MAX",
    "DEFAULT_ALPHA_GRID_SIZE",
    "DEFAULT_DML_K_FOLDS",
    "DEFAULT_N_JOBS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_QUANTREG_MAX_ITER",
    "DEFAULT_QUANTILE_SELECTION_ALPHAS",
    "DEFAULT_QUANTILE_SELECTION_CV_FOLDS",
    "DEFAULT_SELECTION_COEF_TOL",
    "DEFAULT_CRITICAL_VALUE_MULTIPLIER",
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
