"""Classical baselines, enumeration benchmark, and hybrid refinement."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from qubo_builder import QUBOResult
from scenario import Scenario
from traffic_model import simulate_traffic


@dataclass
class BaselineResult:
    """Standardized result format for any optimization baseline."""

    name: str
    schedule: str
    official_cost: float
    delay: int
    overflow: int
    switches: int
    additional_info: Dict[str, Any]


def run_alternating_baseline(scenario: Scenario) -> BaselineResult:
    """Evaluate the 4 canonical alternating schedules and select the best one.

    The 4 alternating schedules are formed by {1010, 0101} at A and {1010, 0101} at B:
      - 10101010
      - 10100101
      - 01011010
      - 01010101
    """
    patterns = ["1010", "0101"]
    candidates = [f"{pA}{pB}" for pA in patterns for pB in patterns]

    best_result = None
    all_alternating = {}

    for cand in candidates:
        res = simulate_traffic(scenario, cand)
        all_alternating[cand] = res.cost
        if best_result is None or res.cost < best_result.cost:
            best_result = res

    assert best_result is not None
    return BaselineResult(
        name="Best Alternating Schedule",
        schedule=best_result.schedule_str,
        official_cost=best_result.cost,
        delay=best_result.delay,
        overflow=best_result.overflow,
        switches=best_result.switches,
        additional_info={"all_evaluated": all_alternating},
    )


def run_greedy_baseline(scenario: Scenario) -> BaselineResult:
    """Per-slot greedy control serving the larger queue after arrivals (ties broken to NS).

    Simulates traffic online slot-by-slot:
      At slot t:
        q_NS += arr_NS; q_EW += arr_EW (+ incoming transfer)
        Choose NS if q_NS >= q_EW else EW
        Serve up to capacity
    """
    cap = scenario.capacity
    storage = scenario.storage
    frac = scenario.link_fraction
    src_app = scenario.link_from_approach
    dst_app = scenario.link_to_approach

    q_A_NS = 0
    q_A_EW = 0
    q_B_NS = 0
    q_B_EW = 0
    pending_trans = 0

    sched_bits_A = []
    sched_bits_B = []

    for t in range(scenario.num_slots):
        # 1. Add arrivals
        q_A_NS += scenario.arrivals["A"]["NS"][t]
        q_A_EW += scenario.arrivals["A"]["EW"][t]
        q_B_NS += scenario.arrivals["B"]["NS"][t]
        q_B_EW += scenario.arrivals["B"]["EW"][t]

        if dst_app == "EW":
            q_B_EW += pending_trans
        else:
            q_B_NS += pending_trans
        pending_trans = 0

        # 2. Greedy decision at A: serve larger queue after arrivals (ties to NS)
        bit_A = 1 if q_A_NS >= q_A_EW else 0
        sched_bits_A.append(bit_A)

        dep_A_NS = 0
        dep_A_EW = 0
        if bit_A == 1:
            dep_A_NS = min(q_A_NS, cap)
            q_A_NS -= dep_A_NS
        else:
            dep_A_EW = min(q_A_EW, cap)
            q_A_EW -= dep_A_EW

        # Greedy decision at B: serve larger queue after arrivals (ties to NS)
        bit_B = 1 if q_B_NS >= q_B_EW else 0
        sched_bits_B.append(bit_B)

        dep_B_NS = 0
        dep_B_EW = 0
        if bit_B == 1:
            dep_B_NS = min(q_B_NS, cap)
            q_B_NS -= dep_B_NS
        else:
            dep_B_EW = min(q_B_EW, cap)
            q_B_EW -= dep_B_EW

        # Compute transfer
        link_dep = dep_A_EW if src_app == "EW" else dep_A_NS
        pending_trans = int(math.floor(frac * link_dep + 0.5))

    greedy_schedule = "".join(str(b) for b in sched_bits_A + sched_bits_B)
    official_eval = simulate_traffic(scenario, greedy_schedule)

    return BaselineResult(
        name="Per-Slot Greedy Control",
        schedule=greedy_schedule,
        official_cost=official_eval.cost,
        delay=official_eval.delay,
        overflow=official_eval.overflow,
        switches=official_eval.switches,
        additional_info={"decision_rule": "Serve max(q_NS, q_EW) after arrivals, ties broken to NS"},
    )


def run_exact_enumeration(scenario: Scenario) -> Tuple[Dict[str, float], List[Tuple[float, str]]]:
    """Exhaustively enumerate and evaluate all 256 eight-bit traffic schedules."""
    all_costs: Dict[str, float] = {}
    ranked_list: List[Tuple[float, str]] = []

    for bits in itertools.product([0, 1], repeat=8):
        bit_str = "".join(str(b) for b in bits)
        res = simulate_traffic(scenario, bit_str)
        all_costs[bit_str] = res.cost
        ranked_list.append((res.cost, bit_str))

    # Canonical ascending sort: primary key is cost, secondary is bitstring
    ranked_list.sort(key=lambda item: (item[0], item[1]))
    return all_costs, ranked_list


def find_qubo_minimizer(scenario: Scenario, qubo: QUBOResult) -> BaselineResult:
    """Find the exact minimizer of the constructed QUBO and evaluate with official scorer."""
    min_energy = float("inf")
    best_x = None

    for bits in itertools.product([0, 1], repeat=8):
        energy = qubo.evaluate(bits)
        if energy < min_energy:
            min_energy = energy
            best_x = bits

    assert best_x is not None
    sched_str = "".join(str(b) for b in best_x)
    official_eval = simulate_traffic(scenario, sched_str)

    return BaselineResult(
        name="Exact QUBO Minimizer",
        schedule=sched_str,
        official_cost=official_eval.cost,
        delay=official_eval.delay,
        overflow=official_eval.overflow,
        switches=official_eval.switches,
        additional_info={
            "qubo_energy": min_energy,
            "minimizer_bits": list(best_x),
        },
    )


def run_hybrid_refinement(scenario: Scenario, initial_schedule: str) -> BaselineResult:
    """Perform classical 1-bit flip hill-climbing local search starting from initial schedule.

    This provides a strictly separated hybrid classical refinement.
    """
    current_bits = [int(c) for c in initial_schedule]
    current_cost = simulate_traffic(scenario, current_bits).cost

    improved = True
    iterations = 0
    while improved and iterations < 16:
        improved = False
        iterations += 1
        for i in range(8):
            neighbor_bits = list(current_bits)
            neighbor_bits[i] = 1 - neighbor_bits[i]
            neighbor_cost = simulate_traffic(scenario, neighbor_bits).cost
            if neighbor_cost < current_cost:
                current_bits = neighbor_bits
                current_cost = neighbor_cost
                improved = True
                break

    best_sched = "".join(str(b) for b in current_bits)
    eval_res = simulate_traffic(scenario, best_sched)

    return BaselineResult(
        name="Hybrid QAOA + Local Search",
        schedule=best_sched,
        official_cost=eval_res.cost,
        delay=eval_res.delay,
        overflow=eval_res.overflow,
        switches=eval_res.switches,
        additional_info={
            "initial_schedule": initial_schedule,
            "refinement_iterations": iterations,
        },
    )
