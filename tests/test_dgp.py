from ivqr_sim.config import DGPS, K_FOLDS, N_VALUES, P_VALUES, PI_VALUES, R_MAIN, TAUS


def test_project_design_constants_exist() -> None:
    assert N_VALUES == [250, 500, 1000]
    assert P_VALUES == [200, 300, 500]
    assert PI_VALUES == [1.0, 0.5, 0.25, 0.10]
    assert TAUS == [0.25, 0.50, 0.75]
    assert DGPS == ["dgp1", "dgp2", "dgp3"]
    assert R_MAIN == 1000
    assert K_FOLDS == 5
