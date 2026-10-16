"""QAOA Circuit construction, parameter optimization, and quantum sampling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from ising import IsingHamiltonian
from qubo_builder import QUBOResult


@dataclass
class QAOAResult:
    """Complete results from QAOA execution."""

    optimal_gammas: List[float]
    optimal_betas: List[float]
    best_energy: float
    num_evaluations: int
    training_time_sec: float
    sampling_time_sec: float
    raw_counts: Dict[str, int]
    decoded_counts: Dict[str, int]
    qasm3_code: str
    p_depth: int
    num_qubits: int = 8
    total_shots: int = 4096
    training_history: List[Dict[str, float]] = field(default_factory=list)


def build_parameterized_qaoa_circuit(
    ising: IsingHamiltonian, p: int
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """Construct a parameterized QAOA circuit for the given Ising Hamiltonian.

    Uses standard X mixer for unconstrained binary schedules:
      U_C(gamma) = exp(-i * gamma * H_C)
      U_M(beta)  = exp(-i * beta * sum_i X_i)
    """
    n = ising.num_qubits
    gamma = ParameterVector("gamma", p)
    beta = ParameterVector("beta", p)

    qc = QuantumCircuit(n)
    # Uniform superposition initialization
    qc.h(range(n))

    for layer in range(p):
        # 1. Cost Hamiltonian Phase Separator
        # Linear terms: exp(-i * gamma * h_i * Z_i) = Rz(2 * gamma * h_i)
        for i in range(n):
            coeff = ising.h[i]
            if abs(coeff) > 1e-12:
                qc.rz(2.0 * gamma[layer] * coeff, i)

        # Quadratic terms: exp(-i * gamma * J_ij * Z_i * Z_j) = Rzz(2 * gamma * J_ij)
        for i in range(n):
            for j in range(i + 1, n):
                coeff = ising.J[i, j]
                if abs(coeff) > 1e-12:
                    qc.rzz(2.0 * gamma[layer] * coeff, i, j)

        # 2. Standard Transverse-Field Mixer
        # exp(-i * beta * X_i) = Rx(2 * beta)
        for i in range(n):
            qc.rx(2.0 * beta[layer], i)

    return qc, gamma, beta


def decode_qiskit_bitstring(raw_bitstring: str) -> str:
    """Decode a raw Qiskit measurement bitstring into canonical schedule order.

    In Qiskit little-endian convention:
      raw_bitstring[0] = qubit 7 (B_slot4)
      raw_bitstring[1] = qubit 6 (B_slot3)
      raw_bitstring[2] = qubit 5 (B_slot2)
      raw_bitstring[3] = qubit 4 (B_slot1)
      raw_bitstring[4] = qubit 3 (A_slot4)
      raw_bitstring[5] = qubit 2 (A_slot3)
      raw_bitstring[6] = qubit 1 (A_slot2)
      raw_bitstring[7] = qubit 0 (A_slot1)

    Canonical schedule order is:
      A_slot1, A_slot2, A_slot3, A_slot4, B_slot1, B_slot2, B_slot3, B_slot4

    Therefore, reversing the string raw_bitstring[::-1] exactly matches the required schedule order.
    """
    return raw_bitstring[::-1]

def solve_qaoa(
    qubo: QUBOResult,
    ising: IsingHamiltonian,
    p: int = 3,
    max_evaluations: int = 270,
    seed: int = 42,
    final_shots: int = 4096,
    init_strategy: str = "multistart",
) -> QAOAResult:
    """Train QAOA circuit parameters and sample final measurement distribution.

    Enforces challenge constraints:
      - p <= 4
      - <= 300 evaluations total across all starts
      - <= 16 qubits (here 8 qubits)
      - Exactly 4096 final shots
      - Deterministic seed
    """
    if p < 1 or p > 4:
        raise ValueError(f"QAOA depth p must be between 1 and 4, got {p}")
    if max_evaluations > 300:
        raise ValueError(f"Training evaluations cannot exceed 300, got {max_evaluations}")

    n = ising.num_qubits

    energy_diag = np.zeros(1 << n, dtype=float)
    for state_int in range(1 << n):
        x = [(state_int >> k) & 1 for k in range(n)]
        energy_diag[state_int] = qubo.evaluate(x)

    qc_param, gamma_vec, beta_vec = build_parameterized_qaoa_circuit(ising, p)

    training_history: List[Dict[str, Any]] = []
    eval_count = 0
    best_eval_energy = float("inf")
    best_eval_params: Optional[np.ndarray] = None

    def objective(params: np.ndarray) -> float:
        nonlocal eval_count, best_eval_energy, best_eval_params
        if eval_count >= max_evaluations:
            return best_eval_energy
        eval_count += 1
        gammas = params[:p]
        betas = params[p:]

        param_dict = {}
        for layer in range(p):
            param_dict[gamma_vec[layer]] = float(gammas[layer])
            param_dict[beta_vec[layer]] = float(betas[layer])

        bound_circuit = qc_param.assign_parameters(param_dict)
        sv = Statevector.from_instruction(bound_circuit)
        probs = sv.probabilities()
        expected_energy = float(np.dot(probs, energy_diag))

        if expected_energy < best_eval_energy:
            best_eval_energy = expected_energy
            best_eval_params = params.copy()

        training_history.append(
            {
                "eval": eval_count,
                "energy": expected_energy,
                "params": params.tolist(),
            }
        )
        return expected_energy

    rng = np.random.default_rng(seed)
    max_h = float(np.max(np.abs(ising.h))) if len(ising.h) > 0 else 1.0
    max_j = float(np.max(np.abs(ising.J))) if ising.J.size > 0 else 1.0
    h_scale = max(1.0, 0.5 * (max_h + max_j))

    # Construct deterministic starting points
    starts: List[Tuple[np.ndarray, float]] = []
    # Start 1: Standard linear ramp
    g1 = [0.05 * (i + 1) for i in range(p)]
    b1 = [0.40 * (1.0 - i / (p + 1)) for i in range(p)]
    p1 = np.array(g1 + b1, dtype=float) + 0.01 * (rng.random(2 * p) - 0.5)
    starts.append((p1, 0.20))

    if init_strategy == "multistart" and max_evaluations >= 120:
        # Start 2: Spectral-normalized adiabatic ramp
        g2 = [(0.6 / h_scale) * ((i + 0.5) / p) for i in range(p)]
        b2 = [0.55 * (1.0 - (i + 0.5) / p) for i in range(p)]
        p2 = np.array(g2 + b2, dtype=float) + 0.02 * (rng.random(2 * p) - 0.5)
        starts.append((p2, 0.15))

        # Start 3: Seeded harmonic ramp for local-minimum escape
        g3 = [(0.9 / h_scale) * ((i + 1) / (p + 1)) for i in range(p)]
        b3 = [0.35 * (1.0 - 0.5 * (i / max(1, p - 1))) for i in range(p)]
        p3 = np.array(g3 + b3, dtype=float) + 0.08 * (rng.random(2 * p) - 0.5)
        starts.append((p3, 0.25))
    elif init_strategy == "spectral":
        g2 = [(0.6 / h_scale) * ((i + 0.5) / p) for i in range(p)]
        b2 = [0.55 * (1.0 - (i + 0.5) / p) for i in range(p)]
        starts = [(np.array(g2 + b2, dtype=float), 0.15)]
    elif init_strategy == "random":
        starts = [(rng.uniform(0.01, 0.5, size=2 * p), 0.20)]

    start_train = time.perf_counter()
    per_start_budget = max_evaluations // len(starts)
    for idx, (init_p, rhobeg) in enumerate(starts):
        rem = max_evaluations - eval_count
        if rem <= 0:
            break
        budget = rem if idx == len(starts) - 1 else min(per_start_budget, rem)
        minimize(
            objective,
            init_p,
            method="COBYLA",
            options={"maxiter": budget, "rhobeg": rhobeg},
        )
    train_time = time.perf_counter() - start_train

    assert best_eval_params is not None
    opt_gammas = best_eval_params[:p].tolist()
    opt_betas = best_eval_params[p:].tolist()

    final_dict = {}
    for layer in range(p):
        final_dict[gamma_vec[layer]] = opt_gammas[layer]
        final_dict[beta_vec[layer]] = opt_betas[layer]

    bound_final_qc = qc_param.assign_parameters(final_dict)
    bound_final_qc_measured = bound_final_qc.copy()
    bound_final_qc_measured.measure_all()
    qasm3_str = qasm3.dumps(bound_final_qc_measured)

    start_sampling = time.perf_counter()
    simulator = AerSimulator(seed_simulator=seed)
    job = simulator.run(bound_final_qc_measured, shots=final_shots)
    sim_result = job.result()
    raw_counts = sim_result.get_counts()
    sample_time = time.perf_counter() - start_sampling

    total_shots = sum(raw_counts.values())
    if total_shots != final_shots:
        raise RuntimeError(f"Expected {final_shots} shots, obtained {total_shots}")

    decoded_counts: Dict[str, int] = {}
    for raw_bitstr, count in raw_counts.items():
        sched_bitstr = decode_qiskit_bitstring(raw_bitstr)
        decoded_counts[sched_bitstr] = count

    return QAOAResult(
        optimal_gammas=opt_gammas,
        optimal_betas=opt_betas,
        best_energy=float(best_eval_energy),
        num_evaluations=eval_count,
        training_time_sec=train_time,
        sampling_time_sec=sample_time,
        raw_counts=raw_counts,
        decoded_counts=decoded_counts,
        qasm3_code=qasm3_str,
        p_depth=p,
        num_qubits=n,
        total_shots=total_shots,
        training_history=training_history,
    )