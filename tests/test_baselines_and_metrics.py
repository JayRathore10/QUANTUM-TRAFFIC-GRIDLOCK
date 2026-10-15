"""Automated tests for baselines, Lift, and Elite Enrichment calculations."""

import pytest
from baselines import (
    find_qubo_minimizer,
    run_alternating_baseline,
    run_exact_enumeration,
    run_greedy_baseline,
    run_hybrid_refinement,
)
from metrics import compute_sampling_quality, validate_qubo_surrogate
from qubo_builder import build_traffic_qubo
from scenario import parse_scenario


@pytest.fixture
def scenario_r():
    return parse_scenario("scenarios/R.json")


def test_baselines(scenario_r):
    # Alternating baseline
    alt = run_alternating_baseline(scenario_r)
    assert alt.official_cost == 116.0
    assert alt.schedule == "10101010"

    # Greedy baseline
    greedy = run_greedy_baseline(scenario_r)
    assert greedy.official_cost == 92.5
    assert greedy.schedule == "01111001"

    # QUBO minimizer
    qubo = build_traffic_qubo(scenario_r)
    q_min = find_qubo_minimizer(scenario_r, qubo)
    assert q_min.official_cost == 72.0
    assert q_min.schedule == "11101010"

    # Hybrid refinement
    hybrid = run_hybrid_refinement(scenario_r, "00000000")
    assert hybrid.official_cost < 237.0  # improves upon worst-case


def test_metrics_computation(scenario_r):
    all_costs, ranked = run_exact_enumeration(scenario_r)
    c_star = ranked[0][0]
    c_rand = sum(all_costs.values()) / len(all_costs)

    # Synthetic uniform counts
    uniform_counts = {sched: 16 for sched in all_costs.keys()}
    metrics_uniform = compute_sampling_quality(all_costs, ranked, uniform_counts, 4096)
    assert metrics_uniform.lift == pytest.approx(0.0, abs=1e-5)
    assert metrics_uniform.elite_enrichment == pytest.approx(1.0, abs=1e-5)

    # Perfect elite counts
    perfect_counts = {ranked[0][1]: 4096}
    metrics_perfect = compute_sampling_quality(all_costs, ranked, perfect_counts, 4096)
    assert metrics_perfect.lift == pytest.approx(1.0, abs=1e-5)
    assert metrics_perfect.elite_enrichment > 1.0
