"""Automated tests for generalization across unseen traffic scenarios."""

import pytest
from baselines import run_exact_enumeration
from ising import qubo_to_ising, verify_qubo_ising_equivalence
from metrics import validate_qubo_surrogate
from qubo_builder import build_traffic_qubo
from scenario import parse_scenario


@pytest.mark.parametrize("scenario_file", ["scenarios/heavy_ns.json", "scenarios/balanced.json"])
def test_unseen_scenarios_pipeline(scenario_file):
    """Ensure the pipeline runs seamlessly without scenario-specific exceptions on unseen conditions."""
    sc = parse_scenario(scenario_file)
    qubo = build_traffic_qubo(sc)
    ising = qubo_to_ising(qubo)

    # 1. Equivalence check
    valid, max_err = verify_qubo_ising_equivalence(qubo, ising)
    assert valid
    assert max_err < 1e-10

    # 2. Enumeration and surrogate validation
    all_costs, ranked = run_exact_enumeration(sc)
    surr = validate_qubo_surrogate(sc, qubo, all_costs, ranked)

    assert surr.correlation > 0.75
    # Heavy NS and balanced scenarios have low regret
    assert surr.surrogate_regret_percent < 25.0
