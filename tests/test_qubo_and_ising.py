"""Automated tests for QUBO construction and Ising conversion."""

import numpy as np
import pytest

from ising import qubo_to_ising, verify_qubo_ising_equivalence
from qubo_builder import build_traffic_qubo
from scenario import parse_scenario
from traffic_model import simulate_traffic


@pytest.fixture
def scenario_r():
    return parse_scenario("scenarios/R.json")


def test_qubo_structure(scenario_r):
    qubo = build_traffic_qubo(scenario_r)
    # Check shape
    assert qubo.Q.shape == (8, 8)
    # Check upper-triangularity: lower triangle must be strictly 0
    for i in range(8):
        for j in range(i):
            assert qubo.Q[i, j] == 0.0, f"Non-zero entry in lower triangle at ({i}, {j})"
    # Check variable names
    assert len(qubo.variable_names) == 8


def test_qubo_ising_equivalence_all_states(scenario_r):
    """Verify that QUBO and Ising energies are mathematically identical for all 256 states."""
    qubo = build_traffic_qubo(scenario_r)
    ising = qubo_to_ising(qubo)
    is_valid, max_err = verify_qubo_ising_equivalence(qubo, ising)
    assert is_valid, f"QUBO and Ising are not equivalent, max error = {max_err}"
    assert max_err < 1e-10


def test_qubo_regret_scenario_r(scenario_r):
    """Check that QUBO surrogate regret on reference scenario is within <= 5% (rubric threshold)."""
    qubo = build_traffic_qubo(scenario_r)

    # Find minimizer
    import itertools
    min_energy = float("inf")
    best_x = None
    for bits in itertools.product([0, 1], repeat=8):
        e = qubo.evaluate(bits)
        if e < min_energy:
            min_energy = e
            best_x = bits

    sched_str = "".join(str(b) for b in best_x)
    official_cost = simulate_traffic(scenario_r, sched_str).cost
    true_opt_cost = 72.0
    regret_pct = ((official_cost - true_opt_cost) / true_opt_cost) * 100.0

    assert regret_pct <= 5.0, f"Regret is {regret_pct:.2f}%, exceeds 5% threshold"
