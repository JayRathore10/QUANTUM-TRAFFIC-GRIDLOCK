"""Ablation studies and mathematical extensions for Quantum Traffic Gridlock."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit import QuantumCircuit

from baselines import find_qubo_minimizer, run_exact_enumeration
from ising import qubo_to_ising
from metrics import compute_sampling_quality, validate_qubo_surrogate
from qaoa_solver import build_parameterized_qaoa_circuit, decode_qiskit_bitstring, solve_qaoa
from qubo_builder import QUBOResult, build_traffic_qubo
from scenario import Scenario


def run_ablation_studies(scenario: Scenario, seed: int = 42) -> Dict[str, Any]:
    """Run systematic ablations over QAOA depth p, penalty scaling, and initialization."""
    all_costs, ranked = run_exact_enumeration(scenario)
    qubo_base = build_traffic_qubo(scenario, quad_scale=1.0)
    ising_base = qubo_to_ising(qubo_base)

    # 1. Depth ablation: p in {1, 2, 3, 4}
    depth_ablation: List[Dict[str, Any]] = []
    for p_val in [1, 2, 3, 4]:
        q_res = solve_qaoa(
            qubo_base, ising_base, p=p_val, max_evaluations=180, seed=seed, final_shots=4096
        )
        m = compute_sampling_quality(all_costs, ranked, q_res.decoded_counts, 4096)
        depth_ablation.append(
            {
                "p_depth": p_val,
                "best_qubo_energy": q_res.best_energy,
                "mean_official_cost_e_q": m.e_q,
                "lift": m.lift,
                "lift_clipped": m.lift_clipped,
                "elite_enrichment": m.elite_enrichment,
                "evaluations": q_res.num_evaluations,
            }
        )

    # 2. Penalty weight scaling ablation: quad_scale in {0.25, 0.5, 1.0, 2.0, 4.0}
    penalty_ablation: List[Dict[str, Any]] = []
    for scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
        q_scaled = build_traffic_qubo(scenario, quad_scale=scale)
        surr = validate_qubo_surrogate(scenario, q_scaled, all_costs, ranked)
        penalty_ablation.append(
            {
                "quad_scale": scale,
                "pearson_correlation": surr.correlation,
                "surrogate_regret_percent": surr.surrogate_regret_percent,
                "qubo_minimizer_schedule": surr.qubo_optimal_schedule,
                "qubo_minimizer_official_cost": surr.qubo_optimal_official_cost,
            }
        )

    # 3. Initialization ablation
    init_ablation: List[Dict[str, Any]] = []
    for strat in ["linear_ramp", "spectral", "random", "multistart"]:
        q_res = solve_qaoa(
            qubo_base,
            ising_base,
            p=3,
            max_evaluations=180,
            seed=seed,
            final_shots=4096,
            init_strategy=strat,
        )
        m = compute_sampling_quality(all_costs, ranked, q_res.decoded_counts, 4096)
        init_ablation.append(
            {
                "strategy": strat,
                "best_qubo_energy": q_res.best_energy,
                "mean_official_cost_e_q": m.e_q,
                "lift_clipped": m.lift_clipped,
                "elite_enrichment": m.elite_enrichment,
            }
        )

    return {
        "depth_ablation": depth_ablation,
        "penalty_scaling_ablation": penalty_ablation,
        "initialization_ablation": init_ablation,
    }


def run_extensions_analysis(scenario: Scenario, seed: int = 42) -> Dict[str, Any]:
    """Execute mathematical extensions: (1) Depolarizing Noise Analysis & (2) Minimum-Green Rule."""
    all_costs, ranked = run_exact_enumeration(scenario)
    qubo = build_traffic_qubo(scenario)
    ising = qubo_to_ising(qubo)
    qaoa_ideal = solve_qaoa(qubo, ising, p=3, max_evaluations=210, seed=seed, final_shots=4096)

    # Extension 1: Gate-Level Depolarizing Noise Resilience Analysis
    qc_param, gamma_vec, beta_vec = build_parameterized_qaoa_circuit(ising, p=3)
    bind_dict = {}
    for l_idx in range(3):
        bind_dict[gamma_vec[l_idx]] = qaoa_ideal.optimal_gammas[l_idx]
        bind_dict[beta_vec[l_idx]] = qaoa_ideal.optimal_betas[l_idx]
    bound_qc = qc_param.assign_parameters(bind_dict)
    bound_qc.measure_all()

    noise_results: List[Dict[str, Any]] = []
    for err_2q in [0.0, 0.002, 0.005, 0.01, 0.02]:
        err_1q = err_2q / 10.0
        if err_2q == 0.0:
            sim = AerSimulator(seed_simulator=seed)
        else:
            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(depolarizing_error(err_1q, 1), ["h", "rx", "rz"])
            nm.add_all_qubit_quantum_error(depolarizing_error(err_2q, 2), ["rzz", "cx"])
            sim = AerSimulator(noise_model=nm, seed_simulator=seed)

        counts = sim.run(bound_qc, shots=4096).result().get_counts()
        decoded = {decode_qiskit_bitstring(k): v for k, v in counts.items()}
        m = compute_sampling_quality(all_costs, ranked, decoded, 4096)
        noise_results.append(
            {
                "two_qubit_depolarizing_rate": err_2q,
                "one_qubit_depolarizing_rate": err_1q,
                "mean_cost_e_q": m.e_q,
                "lift_clipped": m.lift_clipped,
                "elite_enrichment": m.elite_enrichment,
            }
        )

    # Extension 2: Minimum-Green Phase Hold Formulation (Preventing 1-slot phase oscillation)
    # Penalizes isolated 1-slot phase switches via H_min_green = lambda_mg * sum |x_{t+1} - x_{t-1}|
    # coupled with adjacent switch suppression to enforce minimum 2-slot phase continuity.
    qubo_mg = build_traffic_qubo(scenario)
    lambda_mg = 2.5
    for base in [0, 4]:
        for t in range(1, 3):
            # Penalize rapid oscillation x_{t-1} -> x_t -> x_{t+1} where x_{t-1} == x_{t+1} != x_t
            u, v, w = base + t - 1, base + t, base + t + 1
            qubo_mg.Q[u, v] += -lambda_mg
            qubo_mg.Q[v, w] += -lambda_mg
            qubo_mg.Q[u, w] += lambda_mg
            qubo_mg.Q[v, v] += lambda_mg
    mg_min = find_qubo_minimizer(scenario, qubo_mg)

    return {
        "depolarizing_noise_analysis": noise_results,
        "minimum_green_extension": {
            "lambda_min_green": lambda_mg,
            "schedule": mg_min.schedule,
            "official_cost": mg_min.official_cost,
            "switches": mg_min.switches,
        },
    }