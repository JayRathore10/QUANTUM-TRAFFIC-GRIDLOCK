"""Scenario-independent QUBO construction for Quantum Traffic Gridlock."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from scenario import Scenario


@dataclass
class QUBOResult:
    """Complete mathematical specification and metadata of the constructed QUBO."""

    Q: np.ndarray
    offset: float
    variable_names: List[str]
    variable_to_bit: Dict[str, str]
    convention: str = "upper_triangular: E(x) = offset + sum_i Q[i,i]*x[i] + sum_{i<j} Q[i,j]*x[i]*x[j]"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, x: Sequence[int]) -> float:
        """Evaluate QUBO objective for a binary vector x of length 8."""
        vec = np.array(x, dtype=float)
        return float(self.offset + vec @ self.Q @ vec)

    def to_dict(self) -> Dict[str, Any]:
        """Convert QUBO to JSON-serializable dictionary representation."""
        return {
            "offset": float(self.offset),
            "matrix": self.Q.tolist(),
            "variable_names": self.variable_names,
            "variable_to_bit_mapping": self.variable_to_bit,
            "coefficient_convention": self.convention,
            "auxiliary_variables": "None (Compact direct 8-variable physical formulation within 16-qubit limit)",
            "modeling_rationale": (
                "Constructed via scenario-independent analytical traffic dynamics: "
                "(1) Exact quadratic switching penalties: w_switch * (x_t + x_{t+1} - 2*x_t*x_{t+1}). "
                "(2) Linear vehicle delay: w_lin * q_t with w_lin = 1.0. "
                "(3) Soft quadratic queue barrier: w_quad * (q_t)^2 with w_quad = w_overflow / (2 * storage * capacity), "
                "calibrated so marginal cost at storage threshold equals w_overflow. "
                "(4) Dynamic link coupling: A-to-B transfers couple A green decisions at t-1 to B queue accumulation at t."
            ),
            "parameters": self.metadata,
        }


def build_traffic_qubo(scenario: Scenario) -> QUBOResult:
    """Build a scenario-independent QUBO model directly from traffic network dynamics.

    Variables:
      x[0] = A_slot1 (1 = NS, 0 = EW)
      x[1] = A_slot2
      x[2] = A_slot3
      x[3] = A_slot4
      x[4] = B_slot1 (1 = NS, 0 = EW)
      x[5] = B_slot2
      x[6] = B_slot3
      x[7] = B_slot4
    """
    n_vars = 8
    Q = np.zeros((n_vars, n_vars), dtype=float)
    offset = 0.0

    cap = scenario.capacity
    storage = scenario.storage
    w_sw = scenario.w_switch
    w_ov = scenario.w_overflow
    frac = scenario.link_fraction
    src_app = scenario.link_from_approach
    dst_app = scenario.link_to_approach

    # 1. Exact Switching Cost:
    # At intersection A (indices 0..3) and B (indices 4..7):
    # Cost = w_sw * sum_{t=0}^2 (x_t + x_{t+1} - 2 * x_t * x_{t+1})
    for base in [0, 4]:
        for t in range(3):
            u = base + t
            v = base + t + 1
            Q[u, u] += w_sw
            Q[v, v] += w_sw
            Q[u, v] += -2.0 * w_sw

    # Helper function to accumulate polynomial terms into upper-triangular Q and offset
    # expr: array of length 9 representing linear affine form: c0 + sum_{i=0}^7 c_{i+1} * x[i]
    def add_affine_polynomial(expr: np.ndarray, weight_lin: float, weight_quad: float) -> None:
        nonlocal offset, Q
        c0 = expr[0]
        c = expr[1:]

        # Linear contribution: weight_lin * (c0 + sum c_i x_i)
        offset += weight_lin * c0
        for i in range(n_vars):
            Q[i, i] += weight_lin * c[i]

        # Quadratic contribution: weight_quad * (c0 + sum c_i x_i)^2
        # = c0^2 + 2*c0*sum c_i x_i + sum c_i^2 x_i + 2 * sum_{i<j} c_i c_j x_i x_j
        offset += weight_quad * (c0 ** 2)
        for i in range(n_vars):
            Q[i, i] += weight_quad * (2.0 * c0 * c[i] + (c[i] ** 2))
            for j in range(i + 1, n_vars):
                Q[i, j] += weight_quad * (2.0 * c[i] * c[j])

    # Weights for linear delay and quadratic queue barrier
    w_lin = 1.0
    w_quad = w_ov / (2.0 * storage * cap)

    # Physical effective capacity:
    # In slot 0, initial queue is 0, so service is capped by initial arrivals min(arr[0], cap).
    # For subsequent slots s >= 1, full nominal capacity can be served.
    cap_A_NS = [min(scenario.arrivals["A"]["NS"][0], cap)] + [cap] * (scenario.num_slots - 1)
    cap_A_EW = [min(scenario.arrivals["A"]["EW"][0], cap)] + [cap] * (scenario.num_slots - 1)
    cap_B_NS = [min(scenario.arrivals["B"]["NS"][0], cap)] + [cap] * (scenario.num_slots - 1)
    cap_B_EW = [min(scenario.arrivals["B"]["EW"][0], cap)] + [cap] * (scenario.num_slots - 1)

    # Transfer rate per green slot from A to B:
    tau_cap = int(math.floor(frac * cap + 0.5))

    # Formulate queue equations for each slot t in 0..3:
    for t in range(scenario.num_slots):
        # A_NS queue at slot t:
        # q = sum_{s<=t} arr_s - sum_{s<=t} cap_s * x_s
        expr_A_NS = np.zeros(9, dtype=float)
        expr_A_NS[0] = sum(scenario.arrivals["A"]["NS"][: t + 1])
        for s in range(t + 1):
            expr_A_NS[1 + s] -= cap_A_NS[s]

        # A_EW queue at slot t:
        # q = sum_{s<=t} arr_s - sum_{s<=t} cap_s * (1 - x_s)
        expr_A_EW = np.zeros(9, dtype=float)
        expr_A_EW[0] = sum(scenario.arrivals["A"]["EW"][: t + 1]) - sum(cap_A_EW[: t + 1])
        for s in range(t + 1):
            expr_A_EW[1 + s] += cap_A_EW[s]

        # B_NS queue at slot t:
        # q = sum_{s<=t} arr_s - sum_{s<=t} cap_s * x_{4+s} + (transfers if dst_app == 'NS')
        expr_B_NS = np.zeros(9, dtype=float)
        expr_B_NS[0] = sum(scenario.arrivals["B"]["NS"][: t + 1])
        for s in range(t + 1):
            expr_B_NS[1 + 4 + s] -= cap_B_NS[s]

        # B_EW queue at slot t:
        # q = sum_{s<=t} arr_s - sum_{s<=t} cap_s * (1 - x_{4+s}) + (transfers if dst_app == 'EW')
        expr_B_EW = np.zeros(9, dtype=float)
        expr_B_EW[0] = sum(scenario.arrivals["B"]["EW"][: t + 1]) - sum(cap_B_EW[: t + 1])
        for s in range(t + 1):
            expr_B_EW[1 + 4 + s] += cap_B_EW[s]

        # Link coupling:
        # Transfers originating from A arrive at B starting from slot t >= 1
        # If src_app is EW: departures are proportional to (1 - x_s)
        # If src_app is NS: departures are proportional to x_s
        if t >= 1:
            target_expr = expr_B_EW if dst_app == "EW" else expr_B_NS
            if src_app == "EW":
                target_expr[0] += tau_cap * t
                for s in range(t):
                    target_expr[1 + s] -= tau_cap
            else:
                for s in range(t):
                    target_expr[1 + s] += tau_cap

        # Add queue objectives
        for expr in [expr_A_NS, expr_A_EW, expr_B_NS, expr_B_EW]:
            add_affine_polynomial(expr, w_lin, w_quad)

    var_names = [
        "A_slot1",
        "A_slot2",
        "A_slot3",
        "A_slot4",
        "B_slot1",
        "B_slot2",
        "B_slot3",
        "B_slot4",
    ]
    mapping = {name: f"x[{i}]" for i, name in enumerate(var_names)}

    metadata = {
        "scenario_name": scenario.name,
        "capacity": cap,
        "storage": storage,
        "w_lin": w_lin,
        "w_quad": w_quad,
        "w_switch": w_sw,
        "w_overflow": w_ov,
        "link_fraction": frac,
        "link_source": f"A_{src_app}",
        "link_target": f"B_{dst_app}",
        "tau_cap": tau_cap,
    }

    return QUBOResult(
        Q=Q,
        offset=offset,
        variable_names=var_names,
        variable_to_bit=mapping,
        metadata=metadata,
    )
