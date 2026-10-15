"""Serialization of experiment outputs, OpenQASM 3 circuits, and JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from baselines import BaselineResult
from metrics import SamplingQualityMetrics, SurrogateValidationMetrics
from qaoa_solver import QAOAResult
from qubo_builder import QUBOResult
from scenario import Scenario
from traffic_model import simulate_traffic


def save_experiment_outputs(
    output_dir: Path,
    scenario: Scenario,
    qubo: QUBOResult,
    qaoa: QAOAResult,
    metrics: SamplingQualityMetrics,
    surrogate_metrics: SurrogateValidationMetrics,
    baselines: Dict[str, BaselineResult],
    hybrid_result: BaselineResult,
    log_content: str,
) -> Dict[str, Path]:
    """Serialize all required competition artifacts into the specified output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: Dict[str, Path] = {}

    # 1. qubo.json
    qubo_path = output_dir / "qubo.json"
    with open(qubo_path, "w", encoding="utf-8") as f:
        json.dump(qubo.to_dict(), f, indent=2)
    generated_files["qubo"] = qubo_path

    # 2. circuit.qasm
    qasm_path = output_dir / "circuit.qasm"
    with open(qasm_path, "w", encoding="utf-8") as f:
        f.write(qaoa.qasm3_code)
    generated_files["circuit"] = qasm_path

    # 3. counts.json (Raw unmodified measurement counts summing to exactly 4096)
    counts_path = output_dir / "counts.json"
    with open(counts_path, "w", encoding="utf-8") as f:
        json.dump(qaoa.raw_counts, f, indent=2)
    generated_files["counts"] = counts_path

    # 4. schedule.json (Decoded schedules and best sampled results)
    # Find best sampled schedule among decoded counts
    best_sampled_sched = None
    min_sampled_cost = float("inf")
    for sched_str in qaoa.decoded_counts.keys():
        eval_res = simulate_traffic(scenario, sched_str)
        if eval_res.cost < min_sampled_cost:
            min_sampled_cost = eval_res.cost
            best_sampled_sched = sched_str

    assert best_sampled_sched is not None
    best_sampled_eval = simulate_traffic(scenario, best_sampled_sched)

    schedule_data = {
        "schedule_bit_order": [
            "A_slot1",
            "A_slot2",
            "A_slot3",
            "A_slot4",
            "B_slot1",
            "B_slot2",
            "B_slot3",
            "B_slot4",
        ],
        "qiskit_endianness_decoding": (
            "Qiskit raw counts are in little-endian qubit order [q7...q0]. "
            "Decoded schedule bits are obtained by reversing: bitstring[::-1], "
            "mapping qubit i directly to schedule bit i."
        ),
        "best_raw_qaoa_sample": {
            "schedule": best_sampled_sched,
            "cost": best_sampled_eval.cost,
            "delay": best_sampled_eval.delay,
            "overflow": best_sampled_eval.overflow,
            "switches": best_sampled_eval.switches,
            "shot_frequency": qaoa.decoded_counts[best_sampled_sched],
        },
        "hybrid_classical_refinement": {
            "method": hybrid_result.name,
            "schedule": hybrid_result.schedule,
            "cost": hybrid_result.official_cost,
            "delay": hybrid_result.delay,
            "overflow": hybrid_result.overflow,
            "switches": hybrid_result.switches,
            "label": "CLASSICALLY_REFINED_HYBRID (strictly separated from raw QAOA samples)",
        },
        "all_decoded_sample_counts": qaoa.decoded_counts,
    }
    sched_path = output_dir / "schedule.json"
    with open(sched_path, "w", encoding="utf-8") as f:
        json.dump(schedule_data, f, indent=2)
    generated_files["schedule"] = sched_path

    # 5. metrics.json
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    generated_files["metrics"] = metrics_path

    # 6. baselines.json
    baselines_path = output_dir / "baselines.json"
    baselines_data = {}
    for key, b in baselines.items():
        baselines_data[key] = {
            "name": b.name,
            "schedule": b.schedule,
            "cost": b.official_cost,
            "delay": b.delay,
            "overflow": b.overflow,
            "switches": b.switches,
            "additional_info": b.additional_info,
        }
    with open(baselines_path, "w", encoding="utf-8") as f:
        json.dump(baselines_data, f, indent=2)
    generated_files["baselines"] = baselines_path

    # 7. training.json
    training_path = output_dir / "training.json"
    training_data = {
        "p_depth": qaoa.p_depth,
        "evaluations_count": qaoa.num_evaluations,
        "training_time_sec": qaoa.training_time_sec,
        "sampling_time_sec": qaoa.sampling_time_sec,
        "optimal_gammas": qaoa.optimal_gammas,
        "optimal_betas": qaoa.optimal_betas,
        "best_energy": qaoa.best_energy,
        "history": qaoa.training_history,
    }
    with open(training_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2)
    generated_files["training"] = training_path

    # 8. surrogate_validation.json
    surrogate_path = output_dir / "surrogate_validation.json"
    with open(surrogate_path, "w", encoding="utf-8") as f:
        json.dump(surrogate_metrics.to_dict(), f, indent=2)
    generated_files["surrogate_validation"] = surrogate_path

    # 9. run.log
    log_path = output_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_content)
    generated_files["log"] = log_path

    return generated_files
