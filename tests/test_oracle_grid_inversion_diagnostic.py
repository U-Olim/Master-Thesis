import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pandas as pd

from dgp.designs import Design
from dgp.generators import generate_data
from simulation.runner import make_design_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scenarios" / "diagnose_oracle_grid_inversion.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("diagnose_oracle_grid_inversion", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load diagnose_oracle_grid_inversion.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DIAGNOSTIC = _load_module()


class _FakeOracleResult:
    cr_covers_true = True
    cr_lower = 0.0
    cr_upper = 2.0
    cr_length = 2.0
    cr_empty = False
    cr_disconnected = False
    cr_hits_any_boundary = False
    failed_alpha_count = 0
    failed_alpha_rate = 0.0
    message = "ok"


def test_discrepancy_classification() -> None:
    assert DIAGNOSTIC.discrepancy_flags(
        direct_accepts_true=True,
        cr_covers_true=True,
    ) == {
        "direct_accept_cr_cover": True,
        "direct_accept_cr_miss": False,
        "direct_reject_cr_miss": False,
        "direct_reject_cr_cover": False,
    }
    assert DIAGNOSTIC.discrepancy_flags(
        direct_accepts_true=True,
        cr_covers_true=False,
    )["direct_accept_cr_miss"]
    assert DIAGNOSTIC.discrepancy_flags(
        direct_accepts_true=False,
        cr_covers_true=False,
    )["direct_reject_cr_miss"]
    assert DIAGNOSTIC.discrepancy_flags(
        direct_accepts_true=False,
        cr_covers_true=True,
    )["direct_reject_cr_cover"]


def test_diagnostic_seed_reuse_is_deterministic() -> None:
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


def test_same_dataset_object_is_used_for_direct_and_grid_tests() -> None:
    design = Design("dgp1", n=40, p=5, pi=1.0, tau=0.5, rep=0, seed=123)
    data = generate_data(design)
    seen = {}

    def data_generator(received_design: Design):
        assert received_design == design
        seen["generated_id"] = id(data)
        return data

    def oracle_estimator(received_data, **_kwargs):
        seen["oracle_id"] = id(received_data)
        return _FakeOracleResult()

    row = DIAGNOSTIC.evaluate_design(
        design,
        alphas=DIAGNOSTIC.alpha_grid(
            DIAGNOSTIC.DiagnosticConfig(grid_size=5)
        ),
        critical_value=3.841458820694124,
        confidence_level=0.95,
        critical_value_multiplier=1.0,
        quantreg_max_iter=1000,
        data_generator=data_generator,
        oracle_estimator=oracle_estimator,
    )

    assert seen["generated_id"] == seen["oracle_id"]
    assert row["cr_covers_true"] is True
    assert "direct_accept_cr_cover" in row


def test_expected_row_count_for_tiny_configuration() -> None:
    config = DIAGNOSTIC.DiagnosticConfig(
        reps=2,
        dgps=("dgp1", "dgp3"),
        n_values=(40,),
        p_values=(10,),
        pi_values=(1.0, 0.5),
        taus=(0.25, 0.5),
        grid_size=5,
    )
    assert len(DIAGNOSTIC.make_designs(config)) == 2 * 2 * 1 * 1 * 2 * 2


def test_grid_diagnostic_creates_csvs_for_tiny_configuration(
    tmp_path: Path,
) -> None:
    output = tmp_path / "details" / "grid.csv"
    summary_output = tmp_path / "summary" / "grid_summary.csv"
    config = DIAGNOSTIC.DiagnosticConfig(
        base_seed=12345,
        reps=1,
        dgps=("dgp1",),
        n_values=(40,),
        p_values=(5,),
        pi_values=(1.0,),
        taus=(0.5,),
        grid_size=5,
        quantreg_max_iter=1000,
        output=output,
        summary_output=summary_output,
    )

    results, summary = DIAGNOSTIC.run_diagnostic(config)

    assert output.exists()
    assert summary_output.exists()
    assert len(results) == 1
    assert len(summary) == 1
    written = pd.read_csv(output)
    written_summary = pd.read_csv(summary_output)
    assert len(written) == 1
    assert len(written_summary) == 1
    for column in DIAGNOSTIC.DETAIL_COLUMNS:
        assert column in written.columns
    for column in (
        "direct_acceptance_rate",
        "grid_cr_coverage_rate",
        "direct_accept_grid_miss_rate",
        "failed_alpha_rate",
    ):
        assert column in written_summary.columns
