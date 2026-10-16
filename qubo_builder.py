"""Scenario-independent QUBO construction for Quantum Traffic Gridlock."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Sequence
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
    ising_dict: Optional[Dict[str, Any]] = None

    def evaluate(self, x: Sequence[int]) -> float:
        """Evaluate QUBO objective for a binary vector x."""
        vec = np.array(x, dtype=float)
        return float(self.offset + vec @ self.Q @ vec)

    def to_dict(self) -> Dict[str, Any]:
        """Convert QUBO and equivalent Ising Hamiltonian to JSON-serializable dictionary."""
        data = {
            "offset": float(self.offset),
            "matrix": self.Q.tolist(),
            "variable_names": self.variable_names,
            "variable_to_bit_mapping": self.variable_to_bit,
            "coefficient_convention": self.convention,
            "auxiliary_variables": [],
            "num_decision_variables": 8,
            "num_auxiliary_variables": 0,
            "total_qubits": int(self.Q.shape[0]),
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
        if self.ising_dict is not None:
            data["ising_hamiltonian"] = self.ising_dict
        return data


def build_traffic_qubo(
    scenario: Scenario,
    quad_scale: float = 1.0,
) -> QUBOResult:
    """Build a scenario-independent QUBO model directly from traffic network dynamics."""
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
    for base in [0, 4]:
        for t in range(3):
            u = base + t
            v = base + t + 1
            Q[u, u] += w_sw
            Q[v, v] += w_sw
            Q[u, v] += -2.0 * w_sw

    def add_affine_polynomial(expr: np.ndarray, weight_lin: float, weight_quad: float) -> None:
        nonlocal offset, Q
        c0 = expr[0]
        c = expr[1:]
        offset += weight_lin * c0
        for i in range(n_vars):
            Q[i, i] += weight_lin * c[i]

        offset += weight_quad * (c0 ** 2)
        for i in range(n_vars):
            Q[i, i] += weight_quad * (2.0 * c0 * c[i] + (c[i] ** 2))
            for j in range(i + 1, n_vars):
                Q[i, j] += weight_quad * (2.0 * c[i] * c[j])

    w_lin = 1.0
    w_quad = quad_scale * (w_ov / (2.0 * storage * cap))

    # Physical effective capacity: initial slot capped by initial arrivals
    cap_A_NS = [min(scenario.arrivals["A"]["NS"][0], cap)] + [cap] * (scenario.num_slots - 1)
    cap_A_EW = [min(scenario.arrivals["A"]["EW"][0], cap)] + [cap] * (scenario.num_slots - 1)
    cap_B_NS = [min(scenario.arrivals["B"]["NS"][0], cap)] + [cap] * (scenario.num_slots - 1)
    cap_B_EW = [min(scenario.arrivals["B"]["EW"][0], cap)] + [cap] * (scenario.num_slots - 1)

    # Transfer rate per green slot from A to B
    tau_cap = int(
        (Decimal(str(frac)) * Decimal(str(cap))).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )

    # Formulate queue equations for each slot t in 0..3:
    for t in range(scenario.num_slots):
        expr_A_NS = np.zeros(9, dtype=float)
        expr_A_NS[0] = sum(scenario.arrivals["A"]["NS"][: t + 1])
        for s in range(t + 1):
            expr_A_NS[1 + s] -= cap_A_NS[s]

        expr_A_EW = np.zeros(9, dtype=float)
        expr_A_EW[0] = sum(scenario.arrivals["A"]["EW"][: t + 1]) - sum(cap_A_EW[: t + 1])
        for s in range(t + 1):
            expr_A_EW[1 + s] += cap_A_EW[s]

        expr_B_NS = np.zeros(9, dtype=float)
        expr_B_NS[0] = sum(scenario.arrivals["B"]["NS"][: t + 1])
        for s in range(t + 1):
            expr_B_NS[1 + 4 + s] -= cap_B_NS[s]

        expr_B_EW = np.zeros(9, dtype=float)
        expr_B_EW[0] = sum(scenario.arrivals["B"]["EW"][: t + 1]) - sum(cap_B_EW[: t + 1])
        for s in range(t + 1):
            expr_B_EW[1 + 4 + s] += cap_B_EW[s]

        if t >= 1:
            target_expr = expr_B_EW if dst_app == "EW" else expr_B_NS
            if src_app == "EW":
                target_expr[0] += tau_cap * t
                for s in range(t):
                    target_expr[1 + s] -= tau_cap
            else:
                for s in range(t):
                    target_expr[1 + s] += tau_cap

        for expr in [expr_A_NS, expr_A_EW, expr_B_NS, expr_B_EW]:
            add_affine_polynomial(expr, w_lin, w_quad)

    var_names = [
        "A_slot1", "A_slot2", "A_slot3", "A_slot4",
        "B_slot1", "B_slot2", "B_slot3", "B_slot4",
    ]
    mapping = {name: f"x[{i}]" for i, name in enumerate(var_names)}
    
    metadata = {
        "scenario_name": scenario.name,
        "capacity": cap,
        "storage": storage,
        "w_lin": w_lin,
        "w_quad": w_quad,
        "quad_scale": quad_scale,
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