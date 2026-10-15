"""Official Traffic Evaluator and Simulation Dynamics for Quantum Traffic Gridlock."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Union

from scenario import Scenario


@dataclass(frozen=True)
class SlotSnapshot:
    """Detailed state of both intersections at a single 30-second time slot."""

    slot: int
    bit_A: int
    bit_B: int
    incoming_transfer: int
    departures: Dict[str, int]
    queues_after_service: Dict[str, int]
    slot_delay: int
    slot_overflow: int
    outgoing_transfer: int


@dataclass(frozen=True)
class TrafficCostBreakdown:
    """Complete cost breakdown of an 8-bit signal schedule under a traffic scenario."""

    cost: float
    delay: int
    overflow: int
    switches: int
    switches_A: int
    switches_B: int
    schedule_str: str
    schedule_bits: Tuple[int, ...]
    snapshots: List[SlotSnapshot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Union[float, int, str]]:
        """Return summary dictionary of costs."""
        return {
            "schedule": self.schedule_str,
            "cost": self.cost,
            "delay": self.delay,
            "overflow": self.overflow,
            "switches": self.switches,
            "switches_A": self.switches_A,
            "switches_B": self.switches_B,
        }


def parse_schedule_bits(schedule: Union[str, Sequence[int]]) -> Tuple[int, ...]:
    """Parse and validate an 8-bit schedule into a tuple of 8 integers (0 or 1).

    Accepts bitstrings like "10101010" or "1010|1010" or sequence of ints.
    Bit order is: A_slot1, A_slot2, A_slot3, A_slot4, B_slot1, B_slot2, B_slot3, B_slot4.
    """
    if isinstance(schedule, str):
        cleaned = schedule.replace(" ", "").replace("|", "").replace("-", "")
        if len(cleaned) != 8 or not all(c in "01" for c in cleaned):
            raise ValueError(f"Invalid schedule bitstring '{schedule}'. Must contain exactly 8 binary digits.")
        return tuple(int(c) for c in cleaned)

    bits = tuple(int(b) for b in schedule)
    if len(bits) != 8 or not all(b in (0, 1) for b in bits):
        raise ValueError(f"Schedule sequence must have length 8 with elements in {{0, 1}}, got {schedule}")
    return bits


def simulate_traffic(scenario: Scenario, schedule: Union[str, Sequence[int]]) -> TrafficCostBreakdown:
    """Simulate the deterministic traffic dynamics and calculate the official cost.

    Rules from the challenge specification:
      1. Two connected intersections, A and B, over 4 slots.
      2. Signal phase: 1 = North-South (NS), 0 = East-West (EW).
      3. For each slot t:
         - (1) add external arrivals and incoming transfers from preceding slot.
         - (2) green approach serves up to its capacity: d = min(q, capacity).
         - (3) record delay (remaining queue) and overflow (max(0, q - storage)) after service.
         - (4) transfer floor(fraction * departures + 0.5) from configured approach of A to B next slot.
      4. Switches: changes between adjacent slots at each intersection. Pre-slot-1 has no cost.
      5. Final cost = delay + overflow_weight * overflow + switch_weight * switches.
    """
    bits = parse_schedule_bits(schedule)
    bits_A = bits[:4]
    bits_B = bits[4:]

    # Initial queues start empty
    q_A_NS = 0
    q_A_EW = 0
    q_B_NS = 0
    q_B_EW = 0

    pending_transfer = 0
    total_delay = 0
    total_overflow = 0
    snapshots: List[SlotSnapshot] = []

    cap = scenario.capacity
    storage = scenario.storage
    frac = scenario.link_fraction
    src_app = scenario.link_from_approach
    dst_app = scenario.link_to_approach

    for t in range(scenario.num_slots):
        # 1. Add external arrivals
        q_A_NS += scenario.arrivals["A"]["NS"][t]
        q_A_EW += scenario.arrivals["A"]["EW"][t]
        q_B_NS += scenario.arrivals["B"]["NS"][t]
        q_B_EW += scenario.arrivals["B"]["EW"][t]

        # Add incoming transfer from previous slot
        inc_trans = pending_transfer
        if dst_app == "EW":
            q_B_EW += inc_trans
        else:
            q_B_NS += inc_trans
        pending_transfer = 0

        # 2. Service at intersection A
        dep_A_NS = 0
        dep_A_EW = 0
        if bits_A[t] == 1:
            dep_A_NS = min(q_A_NS, cap)
            q_A_NS -= dep_A_NS
        else:
            dep_A_EW = min(q_A_EW, cap)
            q_A_EW -= dep_A_EW

        # Service at intersection B
        dep_B_NS = 0
        dep_B_EW = 0
        if bits_B[t] == 1:
            dep_B_NS = min(q_B_NS, cap)
            q_B_NS -= dep_B_NS
        else:
            dep_B_EW = min(q_B_EW, cap)
            q_B_EW -= dep_B_EW

        # 3. Record delay and overflow from queues remaining after service
        slot_delay = q_A_NS + q_A_EW + q_B_NS + q_B_EW
        slot_overflow = (
            max(0, q_A_NS - storage)
            + max(0, q_A_EW - storage)
            + max(0, q_B_NS - storage)
            + max(0, q_B_EW - storage)
        )
        total_delay += slot_delay
        total_overflow += slot_overflow

        # 4. Compute transfer for next slot: floor(fraction * departures + 0.5)
        link_dep = dep_A_EW if src_app == "EW" else dep_A_NS
        out_trans = int(math.floor(frac * link_dep + 0.5))
        pending_transfer = out_trans

        snapshots.append(
            SlotSnapshot(
                slot=t + 1,
                bit_A=bits_A[t],
                bit_B=bits_B[t],
                incoming_transfer=inc_trans,
                departures={
                    "A_NS": dep_A_NS,
                    "A_EW": dep_A_EW,
                    "B_NS": dep_B_NS,
                    "B_EW": dep_B_EW,
                },
                queues_after_service={
                    "A_NS": q_A_NS,
                    "A_EW": q_A_EW,
                    "B_NS": q_B_NS,
                    "B_EW": q_B_EW,
                },
                slot_delay=slot_delay,
                slot_overflow=slot_overflow,
                outgoing_transfer=out_trans,
            )
        )

    # 5. Switching penalty
    switches_A = sum(1 for t in range(scenario.num_slots - 1) if bits_A[t] != bits_A[t + 1])
    switches_B = sum(1 for t in range(scenario.num_slots - 1) if bits_B[t] != bits_B[t + 1])
    total_switches = switches_A + switches_B

    cost = total_delay + scenario.w_overflow * total_overflow + scenario.w_switch * total_switches
    sched_str = "".join(str(b) for b in bits)

    return TrafficCostBreakdown(
        cost=cost,
        delay=total_delay,
        overflow=total_overflow,
        switches=total_switches,
        switches_A=switches_A,
        switches_B=switches_B,
        schedule_str=sched_str,
        schedule_bits=bits,
        snapshots=snapshots,
    )


def score_schedule(scenario: Scenario, schedule: Union[str, Sequence[int]]) -> float:
    """Convenience function returning only the scalar official traffic cost."""
    return simulate_traffic(scenario, schedule).cost
