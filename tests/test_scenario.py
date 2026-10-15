"""Automated tests for scenario parsing and validation."""

import pytest
from scenario import parse_scenario


def test_valid_scenario_parsing():
    sc = parse_scenario("scenarios/R.json")
    assert sc.name == "Reference_Scenario_R"
    assert sc.capacity == 5
    assert sc.storage == 7
    assert sc.w_overflow == 4.0
    assert sc.w_switch == 0.5
    assert sc.link_from_approach == "EW"
    assert sc.link_to_approach == "EW"
    assert sc.link_fraction == 0.6
    assert len(sc.arrivals["A"]["NS"]) == 4


def test_scenario_validation_errors():
    # Negative capacity
    with pytest.raises(ValueError):
        parse_scenario({"capacity": -1, "storage": 7, "arrivals": {"A": {"NS": [1, 1, 1, 1]}}})

    # Negative arrivals
    with pytest.raises(ValueError):
        parse_scenario(
            {
                "capacity": 5,
                "storage": 7,
                "arrivals": {
                    "A": {"NS": [-1, 2, 3, 4], "EW": [1, 1, 1, 1]},
                    "B": {"NS": [1, 1, 1, 1], "EW": [1, 1, 1, 1]},
                },
            }
        )

    # Invalid transfer fraction
    with pytest.raises(ValueError):
        parse_scenario(
            {
                "capacity": 5,
                "storage": 7,
                "link": {"fraction": 1.5},
                "arrivals": {
                    "A": {"NS": [1, 1, 1, 1], "EW": [1, 1, 1, 1]},
                    "B": {"NS": [1, 1, 1, 1], "EW": [1, 1, 1, 1]},
                },
            }
        )
