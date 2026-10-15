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
    max_evaluations: int = 150,
    seed: int = 42,
    final_shots: int = 4096,
) -> QAOAResult:
    """Train QAOA circuit parameters and sample final measurement distribution.

    Enforces challenge constraints:
      - p <= 4
      - <= 300 evaluations
      - <= 16 qubits (here exactly 8 qubits)
      - Exactly 4096 final shots
      - Deterministic seed
    """
    if p < 1 or p > 4:
        raise ValueError(f"QAOA depth p must be between 1 and 4, got {p}")
    if max_evaluations > 300:
        raise ValueError(f"Training evaluations cannot exceed 300, got {max_evaluations}")

    n = ising.num_qubits

    # Precalculate diagonal Ising energies for all 256 computational basis states
    # Statevector index state_int: qubit k is bit (state_int >> k) & 1.
    energy_diag = np.zeros(1 << n, dtype=float)
    for state_int in range(1 << n):
        # Schedule bit k corresponds to qubit k
        x = [(state_int >> k) & 1 for k in range(n)]
        energy_diag[state_int] = qubo.evaluate(x)

    # Parametric circuit for fast statevector simulation
    qc_param, gamma_vec, beta_vec = build_parameterized_qaoa_circuit(ising, p)

    training_history: List[Dict[str, float]] = []
    eval_count = 0

    def objective(params: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        gammas = params[:p]
        betas = params[p:]

        # Assign parameters
        param_dict = {}
        for layer in range(p):
            param_dict[gamma_vec[layer]] = gammas[layer]
            param_dict[beta_vec[layer]] = betas[layer]

        bound_circuit = qc_param.assign_parameters(param_dict)
        sv = Statevector.from_instruction(bound_circuit)
        probs = sv.probabilities()

        expected_energy = float(np.dot(probs, energy_diag))
        training_history.append(
            {
                "eval": eval_count,
                "energy": expected_energy,
                "params": params.tolist(),
            }
        )
        return expected_energy

    # Deterministic parameter initialization
    rng = np.random.default_rng(seed)
    # Standard linear ramp schedule for initial angles
    init_gammas = [0.05 * (i + 1) for i in range(p)]
    init_betas = [0.4 * (1.0 - i / (p + 1)) for i in range(p)]
    init_params = np.array(init_gammas + init_betas, dtype=float)
    # Add minor deterministic perturbation from seed
    init_params += 0.01 * (rng.random(2 * p) - 0.5)

    start_train = time.perf_counter()
    res = minimize(
        objective,
        init_params,
        method="COBYLA",
        options={"maxiter": max_evaluations, "rhobeg": 0.2},
    )
    train_time = time.perf_counter() - start_train

    optimal_params = res.x
    opt_gammas = optimal_params[:p].tolist()
    opt_betas = optimal_params[p:].tolist()

    # Construct and bind final circuit
    final_dict = {}
    for layer in range(p):
        final_dict[gamma_vec[layer]] = opt_gammas[layer]
        final_dict[beta_vec[layer]] = opt_betas[layer]

    bound_final_qc = qc_param.assign_parameters(final_dict)

    # Export to OpenQASM 3 before adding measurements
    # (Or add measurements and export - both are valid, standard includes measurements)
    bound_final_qc_measured = bound_final_qc.copy()
    bound_final_qc_measured.measure_all()
    qasm3_str = qasm3.dumps(bound_final_qc_measured)

    # Execute final 4096 shots on AerSimulator
    start_sampling = time.perf_counter()
    simulator = AerSimulator(seed_simulator=seed)
    job = simulator.run(bound_final_qc_measured, shots=final_shots)
    sim_result = job.result()
    raw_counts = sim_result.get_counts()
    sample_time = time.perf_counter() - start_sampling

    # Verify shot count
    total_shots = sum(raw_counts.values())
    if total_shots != final_shots:
        raise RuntimeError(f"Expected {final_shots} shots, obtained {total_shots}")

    # Decode counts into canonical schedule ordering
    decoded_counts: Dict[str, int] = {}
    for raw_bitstr, count in raw_counts.items():
        sched_bitstr = decode_qiskit_bitstring(raw_bitstr)
        decoded_counts[sched_bitstr] = count

    return QAOAResult(
        optimal_gammas=opt_gammas,
        optimal_betas=opt_betas,
        best_energy=float(res.fun),
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
