from ivqr_sim.simulation.design import Design


def test_design_can_be_instantiated() -> None:
    design = Design(dgp="dgp1", n=250, p=200, pi=1.0, tau=0.5, rep=0, seed=123)

    assert design.dgp == "dgp1"
    assert design.n == 250
    assert design.p == 200
    assert design.pi == 1.0
    assert design.tau == 0.5
    assert design.rep == 0
    assert design.seed == 123
