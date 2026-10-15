"""Scenario parsing, data structures, and validation for Quantum Traffic Gridlock."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union


@dataclass(frozen=True)
class Scenario:
    """Strongly typed, validated representation of a traffic gridlock scenario."""

    name: str
    capacity: int
    storage: int
    w_overflow: float
    w_switch: float
    link_from_intersection: str
    link_to_intersection: str
    link_from_approach: str
    link_to_approach: str
    link_fraction: float
    arrivals: Dict[str, Dict[str, List[int]]]
    num_slots: int = 4
    slot_duration_sec: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to a standard dictionary representation."""
        return {
            "name": self.name,
            "num_slots": self.num_slots,
            "slot_duration_sec": self.slot_duration_sec,
            "capacity": self.capacity,
            "storage": self.storage,
            "weights": {
                "overflow": self.w_overflow,
                "switch": self.w_switch,
            },
            "link": {
                "from_intersection": self.link_from_intersection,
                "to_intersection": self.link_to_intersection,
                "from_approach": self.link_from_approach,
                "to_approach": self.link_to_approach,
                "fraction": self.link_fraction,
            },
            "arrivals": {
                "A": {
                    "NS": list(self.arrivals["A"]["NS"]),
                    "EW": list(self.arrivals["A"]["EW"]),
                },
                "B": {
                    "NS": list(self.arrivals["B"]["NS"]),
                    "EW": list(self.arrivals["B"]["EW"]),
                },
            },
        }


def parse_scenario(data: Union[Dict[str, Any], str, Path]) -> Scenario:
    """Parse scenario data from a dictionary, JSON string, or file path.

    Supports diverse JSON schemas (nested dicts, flat keys, varied naming conventions)
    to ensure seamless compatibility with unseen organizer scenarios.
    """
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = json.loads(str(data))
    elif isinstance(data, dict):
        raw = data
    else:
        raise TypeError(f"Expected dict, str, or Path, got {type(data).__name__}")

    name = str(raw.get("name", raw.get("scenario_name", "unnamed_scenario")))
    num_slots = int(raw.get("num_slots", 4))
    slot_duration_sec = int(raw.get("slot_duration_sec", 30))

    # Capacity
    cap_val = raw.get("capacity", raw.get("service_capacity", raw.get("cap", 5)))
    capacity = int(cap_val)
    if capacity <= 0:
        raise ValueError(f"Capacity must be positive, got {capacity}")

    # Storage
    storage_val = raw.get("storage", raw.get("nominal_storage", raw.get("queue_storage", 7)))
    storage = int(storage_val)
    if storage <= 0:
        raise ValueError(f"Storage must be positive, got {storage}")

    # Weights
    weights_dict = raw.get("weights", {})
    w_overflow = float(weights_dict.get("overflow", raw.get("overflow_weight", raw.get("w_overflow", 4.0))))
    w_switch = float(weights_dict.get("switch", raw.get("switch_weight", raw.get("switching_weight", raw.get("w_switch", 0.5)))))
    if w_overflow < 0:
        raise ValueError(f"Overflow weight must be non-negative, got {w_overflow}")
    if w_switch < 0:
        raise ValueError(f"Switch weight must be non-negative, got {w_switch}")

    # Link configuration
    link_info = raw.get("link", {})
    from_inter = str(link_info.get("from_intersection", raw.get("link_from", "A"))).upper()
    to_inter = str(link_info.get("to_intersection", raw.get("link_to", "B"))).upper()

    from_app = str(link_info.get("from_approach", raw.get("link_approach", raw.get("link_from_approach", "EW")))).upper()
    to_app = str(link_info.get("to_approach", raw.get("link_target_approach", raw.get("link_to_approach", from_app)))).upper()
    fraction = float(link_info.get("fraction", raw.get("transfer_fraction", raw.get("fraction", 0.6))))

    if from_app not in {"NS", "EW"}:
        raise ValueError(f"Source link approach must be 'NS' or 'EW', got '{from_app}'")
    if to_app not in {"NS", "EW"}:
        raise ValueError(f"Target link approach must be 'NS' or 'EW', got '{to_app}'")
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"Transfer fraction must be in [0.0, 1.0], got {fraction}")

    # Arrivals parsing
    arrivals: Dict[str, Dict[str, List[int]]] = {"A": {}, "B": {}}
    arr_raw = raw.get("arrivals", {})

    if "A" in arr_raw and "B" in arr_raw:
        for inter in ["A", "B"]:
            for app in ["NS", "EW"]:
                seq = arr_raw[inter].get(app, arr_raw[inter].get(app.lower()))
                if seq is None:
                    raise KeyError(f"Missing arrival data for {inter} {app}")
                arrivals[inter][app] = [int(v) for v in seq]
    else:
        # Check flat keys like arrivals_A_NS or A_NS_arrivals
        for inter in ["A", "B"]:
            for app in ["NS", "EW"]:
                candidates = [
                    f"arrivals_{inter}_{app}",
                    f"{inter}_{app}_arrivals",
                    f"{inter}_{app}",
                    f"arr_{inter}_{app}",
                ]
                found = None
                for c in candidates:
                    if c in raw:
                        found = raw[c]
                        break
                    if c.lower() in raw:
                        found = raw[c.lower()]
                        break
                if found is not None:
                    arrivals[inter][app] = [int(v) for v in found]
                else:
                    raise KeyError(f"Could not locate arrivals for intersection {inter}, approach {app}")

    # Validate arrivals lengths and values
    for inter in ["A", "B"]:
        for app in ["NS", "EW"]:
            seq = arrivals[inter][app]
            if len(seq) != num_slots:
                raise ValueError(
                    f"Expected {num_slots} arrivals for {inter} {app}, got {len(seq)}: {seq}"
                )
            for v in seq:
                if v < 0:
                    raise ValueError(f"Arrival count cannot be negative: {v} in {inter} {app}")

    return Scenario(
        name=name,
        capacity=capacity,
        storage=storage,
        w_overflow=w_overflow,
        w_switch=w_switch,
        link_from_intersection=from_inter,
        link_to_intersection=to_inter,
        link_from_approach=from_app,
        link_to_approach=to_app,
        link_fraction=fraction,
        arrivals=arrivals,
        num_slots=num_slots,
        slot_duration_sec=slot_duration_sec,
    )
