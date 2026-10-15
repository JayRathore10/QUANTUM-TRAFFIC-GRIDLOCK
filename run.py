"""Command-Line Interface for Quantum Traffic Gridlock optimization pipeline."""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from baselines import (
    BaselineResult,
    find_qubo_minimizer,
    run_alternating_baseline,
    run_exact_enumeration,
    run_greedy_baseline,
    run_hybrid_refinement,
)
from ising import qubo_to_ising, verify_qubo_ising_equivalence
from metrics import compute_sampling_quality, validate_qubo_surrogate
from qaoa_solver import decode_qiskit_bitstring, solve_qaoa
from qubo_builder import build_traffic_qubo
from scenario import parse_scenario
from serialization import save_experiment_outputs
from traffic_model import simulate_traffic


def setup_logger() -> Tuple[logging.Logger, io.StringIO]:
    """Configure logger to stream to both stdout and an in-memory string buffer."""
    log_buffer = io.StringIO()
    logger = logging.getLogger("TrafficGridlock")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # String buffer handler for run.log
    buffer_handler = logging.StreamHandler(log_buffer)
    buffer_handler.setFormatter(formatter)
    logger.addHandler(buffer_handler)

    return logger, log_buffer


def run_pipeline(
    scenario_path: str,
    seed: int,
    output_dir_str: str,
    p_depth: int = 3,
    max_evaluations: int = 150,
    eval_multi_seeds: bool = False,
    extra_seeds: List[int] = None,
) -> None:
    """Execute the end-to-end quantum traffic optimization workflow."""
    start_total_time = time.perf_counter()
    logger, log_buffer = setup_logger()

    logger.info("==================================================================")
    logger.info("  QUANTUM TRAFFIC GRIDLOCK — IBM QISKIT FALL FEST 2026 (QC@IITI)  ")
    logger.info("==================================================================")
    logger.info("Scenario: %s", scenario_path)
    logger.info("Seed: %d", seed)
    logger.info("Output Directory: %s", output_dir_str)
    logger.info("QAOA Depth p: %d", p_depth)

    # 1. Parse and validate scenario
    try:
        scenario = parse_scenario(scenario_path)
        logger.info("Scenario '%s' loaded and validated successfully.", scenario.name)
        logger.info("Capacity: %d, Storage: %d", scenario.capacity, scenario.storage)
        logger.info("Weights: overflow=%.2f, switch=%.2f", scenario.w_overflow, scenario.w_switch)
        logger.info(
            "Link: %s (%s) -> %s (%s), fraction=%.2f",
            scenario.link_from_intersection,
            scenario.link_from_approach,
            scenario.link_to_intersection,
            scenario.link_to_approach,
            scenario.link_fraction,
        )
    except Exception as e:
        logger.error("Failed to load scenario: %s", e)
        sys.exit(1)

    # 2. Build scenario-independent QUBO
    logger.info("Constructing scenario-independent QUBO model...")
    qubo = build_traffic_qubo(scenario)
    logger.info("QUBO constructed: 8 variables, offset=%.4f", qubo.offset)

    # 3. Convert to Ising Hamiltonian and verify equivalence
    logger.info("Converting QUBO to Ising Hamiltonian (x_i = (1 - Z_i)/2)...")
    ising = qubo_to_ising(qubo)
    is_valid, max_err = verify_qubo_ising_equivalence(qubo, ising)
    if not is_valid:
        logger.error("QUBO and Ising formulations are not equivalent! Max error: %.2e", max_err)
        sys.exit(1)
    logger.info("QUBO-Ising equivalence verified across all 256 states (max error: %.2e).", max_err)

    # 4. Benchmarks & Enumeration
    logger.info("Running benchmarks and exact enumeration of all 256 schedules...")
    all_costs, ranked_schedules = run_exact_enumeration(scenario)
    c_star, opt_sched = ranked_schedules[0]
    worst_cost, worst_sched = ranked_schedules[-1]
    c_rand = float(np.mean(list(all_costs.values())))

    logger.info("Exact Global Optimum C*: %.2f (schedule %s)", c_star, opt_sched)
    logger.info("Uniform Random Expected Cost C_rand: %.4f", c_rand)
    logger.info("Worst Schedule Cost: %.2f (schedule %s)", worst_cost, worst_sched)

    # Baselines
    alt_res = run_alternating_baseline(scenario)
    logger.info("Best Alternating Schedule: %s -> Cost: %.2f", alt_res.schedule, alt_res.official_cost)

    greedy_res = run_greedy_baseline(scenario)
    logger.info("Per-Slot Greedy Schedule: %s -> Cost: %.2f", greedy_res.schedule, greedy_res.official_cost)

    qubo_min_res = find_qubo_minimizer(scenario, qubo)
    logger.info(
        "Exact QUBO Minimizer Schedule: %s -> Official Cost: %.2f (QUBO Energy: %.2f)",
        qubo_min_res.schedule,
        qubo_min_res.official_cost,
        qubo_min_res.additional_info["qubo_energy"],
    )

    # 5. Surrogate Validation
    surrogate_metrics = validate_qubo_surrogate(scenario, qubo, all_costs, ranked_schedules)
    logger.info(
        "Surrogate Validation: Correlation=%.4f, Regret=%.2f%%",
        surrogate_metrics.correlation,
        surrogate_metrics.surrogate_regret_percent,
    )

    # 6. Train QAOA and Sample
    logger.info("Starting QAOA training with depth p=%d (max %d evaluations)...", p_depth, max_evaluations)
    qaoa_result = solve_qaoa(
        qubo=qubo,
        ising=ising,
        p=p_depth,
        max_evaluations=max_evaluations,
        seed=seed,
        final_shots=4096,
    )
    logger.info(
        "QAOA training completed in %.2fs (%d evals). Optimal Energy: %.4f",
        qaoa_result.training_time_sec,
        qaoa_result.num_evaluations,
        qaoa_result.best_energy,
    )
    logger.info("Optimal gammas: %s", [round(g, 4) for g in qaoa_result.optimal_gammas])
    logger.info("Optimal betas:  %s", [round(b, 4) for b in qaoa_result.optimal_betas])

    logger.info("Final quantum circuit executed for exactly 4,096 shots in %.2fs.", qaoa_result.sampling_time_sec)

    # 7. Compute Quantum Sampling Quality Metrics
    sampling_metrics = compute_sampling_quality(
        all_costs=all_costs,
        ranked_schedules=ranked_schedules,
        decoded_counts=qaoa_result.decoded_counts,
        total_shots=4096,
    )
    logger.info("Mean QAOA Sample Cost E_Q: %.4f", sampling_metrics.e_q)
    logger.info("Lift (Raw): %.4f, Lift (Clipped [0, 1]): %.4f", sampling_metrics.lift, sampling_metrics.lift_clipped)
    logger.info(
        "Elite Enrichment: %.4fx (Elite shots: %d / 4096)",
        sampling_metrics.elite_enrichment,
        sampling_metrics.elite_shots,
    )

    # 8. Best Sampled Schedule & Hybrid Classical Refinement
    best_sampled_sched = min(qaoa_result.decoded_counts.keys(), key=lambda s: all_costs[s])
    best_sampled_cost = all_costs[best_sampled_sched]
    logger.info(
        "Best Sampled Schedule: %s with Official Cost: %.2f (Frequency: %d shots)",
        best_sampled_sched,
        best_sampled_cost,
        qaoa_result.decoded_counts[best_sampled_sched],
    )

    hybrid_res = run_hybrid_refinement(scenario, best_sampled_sched)
    logger.info(
        "Hybrid Refinement Result: %s with Official Cost: %.2f (Refined from: %s)",
        hybrid_res.schedule,
        hybrid_res.official_cost,
        best_sampled_sched,
    )

    # Baselines summary dict
    baselines_dict = {
        "alternating": alt_res,
        "greedy": greedy_res,
        "qubo_minimizer": qubo_min_res,
        "random_expected": BaselineResult(
            name="Uniform Random Expected",
            schedule="N/A",
            official_cost=c_rand,
            delay=0,
            overflow=0,
            switches=0,
            additional_info={"cost": c_rand},
        ),
        "exact_global_optimum": BaselineResult(
            name="Exact Global Optimum",
            schedule=opt_sched,
            official_cost=c_star,
            delay=simulate_traffic(scenario, opt_sched).delay,
            overflow=simulate_traffic(scenario, opt_sched).overflow,
            switches=simulate_traffic(scenario, opt_sched).switches,
            additional_info={"worst_cost": worst_cost, "worst_schedule": worst_sched},
        ),
        "raw_qaoa_best": BaselineResult(
            name="Best Raw QAOA Sample",
            schedule=best_sampled_sched,
            official_cost=best_sampled_cost,
            delay=simulate_traffic(scenario, best_sampled_sched).delay,
            overflow=simulate_traffic(scenario, best_sampled_sched).overflow,
            switches=simulate_traffic(scenario, best_sampled_sched).switches,
            additional_info={"mean_sample_cost": sampling_metrics.e_q},
        ),
    }

    # 9. Multi-seed Evaluation (if requested or by default for full reproducibility)
    multi_seed_results = []
    if eval_multi_seeds:
        seeds_to_test = extra_seeds if extra_seeds else [42, 43, 44, 45, 46]
        logger.info("Evaluating robustness across %d fixed seeds: %s", len(seeds_to_test), seeds_to_test)
        for s in seeds_to_test:
            q_res = solve_qaoa(qubo, ising, p=p_depth, max_evaluations=max_evaluations, seed=s, final_shots=4096)
            s_metrics = compute_sampling_quality(all_costs, ranked_schedules, q_res.decoded_counts, 4096)
            multi_seed_results.append(
                {
                    "seed": s,
                    "e_q": s_metrics.e_q,
                    "lift": s_metrics.lift,
                    "lift_clipped": s_metrics.lift_clipped,
                    "elite_enrichment": s_metrics.elite_enrichment,
                    "elite_shots": s_metrics.elite_shots,
                }
            )
            logger.info(
                "  Seed %d: E_Q=%.2f, Lift=%.4f (Clipped: %.4f), Enrichment=%.2fx",
                s,
                s_metrics.e_q,
                s_metrics.lift,
                s_metrics.lift_clipped,
                s_metrics.elite_enrichment,
            )

        clipped_lifts = [m["lift_clipped"] for m in multi_seed_results]
        min_clipped_lift = min(clipped_lifts)
        mean_lift = float(np.mean(clipped_lifts))
        logger.info("Robustness Summary: Min Clipped Lift = %.4f, Mean Lift = %.4f", min_clipped_lift, mean_lift)

    # 10. Serialize all outputs
    output_path = Path(output_dir_str)
    out_files = save_experiment_outputs(
        output_dir=output_path,
        scenario=scenario,
        qubo=qubo,
        qaoa=qaoa_result,
        metrics=sampling_metrics,
        surrogate_metrics=surrogate_metrics,
        baselines=baselines_dict,
        hybrid_result=hybrid_res,
        log_content=log_buffer.getvalue(),
    )

    if multi_seed_results:
        with open(output_path / "multi_seed_results.json", "w", encoding="utf-8") as f:
            json.dump(multi_seed_results, f, indent=2)

    total_duration = time.perf_counter() - start_total_time
    logger.info("All outputs saved successfully to '%s':", output_path)
    for name, pth in out_files.items():
        logger.info("  - %s: %s", name, pth.name)
    logger.info("Total Pipeline Runtime: %.2f seconds", total_duration)
    logger.info("Execution completed successfully.")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Quantum Traffic Gridlock — Optimization Pipeline")
    parser.add_argument("--scenario", type=str, required=True, help="Path to scenario JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--out", type=str, required=True, help="Output directory path for artifacts")
    parser.add_argument("--p", type=int, default=3, help="QAOA depth p in [1, 4] (default: 3)")
    parser.add_argument("--max-evals", type=int, default=150, help="Maximum optimizer evaluations (default: 150)")
    parser.add_argument(
        "--eval-multi-seeds",
        action="store_true",
        help="Evaluate sampling across 5 fixed seeds [42, 43, 44, 45, 46]",
    )

    args = parser.parse_args()

    run_pipeline(
        scenario_path=args.scenario,
        seed=args.seed,
        output_dir_str=args.out,
        p_depth=args.p,
        max_evaluations=args.max_evals,
        eval_multi_seeds=args.eval_multi_seeds,
    )


if __name__ == "__main__":
    main()
