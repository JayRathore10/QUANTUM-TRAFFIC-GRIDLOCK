"""Performance metrics: Lift, Elite Enrichment, and Surrogate Validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from qubo_builder import QUBOResult
from scenario import Scenario
from traffic_model import simulate_traffic


@dataclass
class SamplingQualityMetrics:
    """Comprehensive sampling performance metrics for QAOA."""

    c_star: float
    c_rand: float
    e_q: float
    lift: float
    lift_clipped: float
    elite_enrichment: float
    elite_shots: int
    elite_schedules: List[str]
    total_shots: int = 4096

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "optimal_cost_c_star": float(self.c_star),
            "uniform_random_c_rand": float(self.c_rand),
            "mean_qaoa_cost_e_q": float(self.e_q),
            "raw_lift": float(self.lift),
            "lift_clipped_0_1": float(self.lift_clipped),
            "elite_enrichment_factor": float(self.elite_enrichment),
            "elite_shots_count": int(self.elite_shots),
            "total_shots": int(self.total_shots),
            "elite_set_size": len(self.elite_schedules),
        }


def compute_sampling_quality(
    all_costs: Dict[str, float],
    ranked_schedules: List[Tuple[float, str]],
    decoded_counts: Dict[str, int],
    total_shots: int = 4096,
) -> SamplingQualityMetrics:
    """Calculate Lift and Elite Enrichment exactly according to the challenge rubric.

    Lift formula:
      Lift = (C_rand - E_Q) / (C_rand - C*)

    Elite Enrichment:
      Select exactly 13 lowest-cost schedules (breaking ties by canonical ascending bitstring).
      Enrichment = (elite_shots / total_shots) / (13 / 256)
    """
    c_star = ranked_schedules[0][0]
    c_rand = float(np.mean(list(all_costs.values())))

    # Calculate E_Q (mean official cost over all raw samples)
    total_sampled_cost = 0.0
    actual_shots = 0
    for sched_str, count in decoded_counts.items():
        cost = all_costs[sched_str]
        total_sampled_cost += cost * count
        actual_shots += count

    if actual_shots != total_shots:
        raise ValueError(f"Decoded counts sum to {actual_shots}, expected {total_shots}")

    e_q = total_sampled_cost / total_shots

    # Compute Lift with zero-denominator safeguard
    denom = c_rand - c_star
    if abs(denom) < 1e-9:
        raw_lift = 0.0
    else:
        raw_lift = (c_rand - e_q) / denom

    lift_clipped = float(np.clip(raw_lift, 0.0, 1.0))

    # Elite Set: exactly the 13 lowest-cost schedules
    elite_13 = ranked_schedules[:13]
    elite_strings = set(item[1] for item in elite_13)

    elite_shots = sum(count for sched_str, count in decoded_counts.items() if sched_str in elite_strings)
    elite_fraction = elite_shots / float(total_shots)
    random_expected_fraction = 13.0 / 256.0
    elite_enrichment = elite_fraction / random_expected_fraction

    return SamplingQualityMetrics(
        c_star=c_star,
        c_rand=c_rand,
        e_q=e_q,
        lift=raw_lift,
        lift_clipped=lift_clipped,
        elite_enrichment=elite_enrichment,
        elite_shots=elite_shots,
        elite_schedules=[item[1] for item in elite_13],
        total_shots=total_shots,
    )


@dataclass
class SurrogateValidationMetrics:
    """Statistical evaluation of the QUBO surrogate against the exact official scorer."""

    correlation: float
    surrogate_regret_percent: float
    qubo_optimal_schedule: str
    qubo_optimal_official_cost: float
    true_optimal_schedule: str
    true_optimal_cost: float
    max_energy_cost_diff: float
    mean_absolute_difference: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert surrogate validation metrics to dictionary."""
        return {
            "pearson_correlation": float(self.correlation),
            "surrogate_regret_percent": float(self.surrogate_regret_percent),
            "qubo_optimal_schedule": self.qubo_optimal_schedule,
            "qubo_optimal_official_cost": float(self.qubo_optimal_official_cost),
            "true_optimal_schedule": self.true_optimal_schedule,
            "true_optimal_cost": float(self.true_optimal_cost),
            "max_difference": float(self.max_energy_cost_diff),
            "mean_absolute_difference": float(self.mean_absolute_difference),
        }


def validate_qubo_surrogate(
    scenario: Scenario,
    qubo: QUBOResult,
    all_costs: Dict[str, float],
    ranked_schedules: List[Tuple[float, str]],
) -> SurrogateValidationMetrics:
    """Validate QUBO surrogate fidelity across all 256 schedules."""
    schedules = list(all_costs.keys())
    exact_vals = np.array([all_costs[s] for s in schedules], dtype=float)

    qubo_vals = np.zeros(len(schedules), dtype=float)
    min_energy = float("inf")
    best_qubo_sched = ""

    for i, s in enumerate(schedules):
        x = [int(c) for c in s]
        energy = qubo.evaluate(x)
        qubo_vals[i] = energy
        if energy < min_energy:
            min_energy = energy
            best_qubo_sched = s

    # Pearson correlation
    corr = float(np.corrcoef(exact_vals, qubo_vals)[0, 1])

    # Regret relative to true optimum
    c_star = ranked_schedules[0][0]
    true_opt_sched = ranked_schedules[0][1]

    qubo_opt_cost = all_costs[best_qubo_sched]
    regret_pct = ((qubo_opt_cost - c_star) / c_star) * 100.0 if c_star > 0 else 0.0

    diffs = np.abs(exact_vals - qubo_vals)
    max_diff = float(np.max(diffs))
    mae = float(np.mean(diffs))

    return SurrogateValidationMetrics(
        correlation=corr,
        surrogate_regret_percent=regret_pct,
        qubo_optimal_schedule=best_qubo_sched,
        qubo_optimal_official_cost=qubo_opt_cost,
        true_optimal_schedule=true_opt_sched,
        true_optimal_cost=c_star,
        max_energy_cost_diff=max_diff,
        mean_absolute_difference=mae,
    )
