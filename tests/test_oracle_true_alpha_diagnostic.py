import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType

import pandas as pd

from simulation.runner import make_design_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scenarios" / "diagnose_oracle_true_alpha.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("diagnose_oracle_true_alpha", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load diagnose_oracle_true_alpha.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIAGNOSTIC = _load_module()


def test_diagnostic_design_seeds_are_deterministic() -> None:
    config = DIAGNOSTIC.DiagnosticConfig(
        base_seed=12345,
        reps=2,
        dgps=("dgp1",),
        n_values=(500,),
        p_values=(200,),
        pi_values=(1.0,),
        taus=(0.5,),
    )
    first = DIAGNOSTIC.make_designs(config)
    second = DIAGNOSTIC.make_designs(config)
    assert first == second
    assert first[0].seed == make_design_seed(
        base_seed=12345,
        dgp="dgp1",
        n=500,
        p=200,
        pi=1.0,
        tau=0.5,
        rep=0,
    )
    assert first[1].seed == make_design_seed(
        base_seed=12345,
        dgp="dgp1",
        n=500,
        p=200,
        pi=1.0,
        tau=0.5,
        rep=1,
    )


def test_diagnostic_design_row_count_for_small_configuration() -> None:
    config = DIAGNOSTIC.DiagnosticConfig(
        reps=2,
        dgps=("dgp1", "dgp3"),
        n_values=(40, 50),
        p_values=(10,),
        pi_values=(1.0, 0.5),
        taus=(0.25, 0.5),
    )
    designs = DIAGNOSTIC.make_designs(config)
    assert len(designs) == 2 * 2 * 2 * 1 * 2 * 2


def test_rejection_flags_are_computed_correctly() -> None:
    assert DIAGNOSTIC.rejection_flags(
        statistic=4.0,
        converged=True,
        critical_value=3.84,
    ) == (True, True)
    assert DIAGNOSTIC.rejection_flags(
        statistic=1.0,
        converged=True,
        critical_value=3.84,
    ) == (False, False)
    assert DIAGNOSTIC.rejection_flags(
        statistic=float("inf"),
        converged=False,
        critical_value=3.84,
    ) == (False, True)


def test_summary_rates_use_failure_as_unconditional_rejection() -> None:
    results = pd.DataFrame(
        [
            {
                "dgp": "dgp1",
                "n": 40,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "converged": True,
                "test_statistic": 4.0,
                "rejected_if_converged": True,
                "rejected_failure_as_reject": True,
            },
            {
                "dgp": "dgp1",
                "n": 40,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "converged": True,
                "test_statistic": 1.0,
                "rejected_if_converged": False,
                "rejected_failure_as_reject": False,
            },
            {
                "dgp": "dgp1",
                "n": 40,
                "p": 5,
                "pi": 1.0,
                "tau": 0.5,
                "converged": False,
                "test_statistic": float("inf"),
                "rejected_if_converged": False,
                "rejected_failure_as_reject": True,
            },
        ]
    )
    summary = DIAGNOSTIC.summarize_results(results)
    row = summary.iloc[0]
    assert row["replications"] == 3
    assert row["convergence_rate"] == 2 / 3
    assert row["unconditional_rejection_rate"] == 2 / 3
    assert row["rejection_rate_among_converged"] == 1 / 2
    assert math.isclose(row["implied_coverage"], 1 / 3)
    assert row["mean_test_statistic"] == 2.5
    assert row["median_test_statistic"] == 2.5


def test_diagnostic_creates_output_directories_and_runs_tiny_config(
    tmp_path: Path,
) -> None:
    output = tmp_path / "nested" / "details" / "oracle_true_alpha.csv"
    summary_output = tmp_path / "nested" / "summary" / "oracle_true_alpha_summary.csv"
    config = DIAGNOSTIC.DiagnosticConfig(
        base_seed=12345,
        reps=1,
        dgps=("dgp1",),
        n_values=(40,),
        p_values=(5,),
        pi_values=(1.0,),
        taus=(0.5,),
        quantreg_max_iter=1000,
        output=output,
        summary_output=summary_output,
    )
    results, summary, critical_value = DIAGNOSTIC.run_diagnostic(config)

    assert output.exists()
    assert summary_output.exists()
    assert len(results) == 1
    assert len(summary) == 1
    assert critical_value > 0

    written = pd.read_csv(output)
    written_summary = pd.read_csv(summary_output)
    assert len(written) == 1
    assert len(written_summary) == 1
    for column in DIAGNOSTIC.DETAIL_COLUMNS:
        assert column in written.columns
