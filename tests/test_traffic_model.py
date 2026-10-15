"""Automated tests for traffic dynamics and official scorer."""

import pytest
from scenario import parse_scenario
from traffic_model import parse_schedule_bits, score_schedule, simulate_traffic


@pytest.fixture
def scenario_r():
    return parse_scenario("scenarios/R.json")


def test_four_official_reference_vectors(scenario_r):
    """Verify that the official evaluator exactly reproduces the 4 published vectors."""
    vectors = [
        ("10101010", 116.0),
        ("11111111", 146.0),
        ("00000000", 237.0),
        ("01111001", 92.5),
    ]
    for sched, expected in vectors:
        cost = score_schedule(scenario_r, sched)
        assert cost == pytest.approx(expected, abs=1e-6), f"Mismatch for {sched}: got {cost}, expected {expected}"


def test_schedule_bit_parsing():
    """Verify bitstring and list schedule parsing."""
    assert parse_schedule_bits("10101010") == (1, 0, 1, 0, 1, 0, 1, 0)
    assert parse_schedule_bits("1010 | 1010") == (1, 0, 1, 0, 1, 0, 1, 0)
    assert parse_schedule_bits([1, 1, 0, 0, 0, 0, 1, 1]) == (1, 1, 0, 0, 0, 0, 1, 1)

    with pytest.raises(ValueError):
        parse_schedule_bits("101")  # too short

    with pytest.raises(ValueError):
        parse_schedule_bits("10101012")  # invalid digit


def test_switching_cost_exactness(scenario_r):
    """Verify switching penalty calculation: changes between adjacent slots."""
    # 0000 | 0000 has 0 switches
    res_0 = simulate_traffic(scenario_r, "00000000")
    assert res_0.switches == 0
    assert res_0.switches_A == 0
    assert res_0.switches_B == 0

    # 1010 | 1010 has 3 switches at A (1->0, 0->1, 1->0) and 3 at B = 6 switches
    res_alt = simulate_traffic(scenario_r, "10101010")
    assert res_alt.switches == 6
    assert res_alt.switches_A == 3
    assert res_alt.switches_B == 3


def test_overflow_and_transfer_dynamics(scenario_r):
    """Test transfer rounding and boundary conditions."""
    res = simulate_traffic(scenario_r, "11101010")
    assert res.cost == 72.0
    assert res.delay == 66
    assert res.overflow == 1
    assert res.switches == 4

    # Check slot snapshots
    assert len(res.snapshots) == 4
    # Slot 4 (t=3) outgoing transfer should leave model
    snap_4 = res.snapshots[3]
    assert snap_4.outgoing_transfer == 3
