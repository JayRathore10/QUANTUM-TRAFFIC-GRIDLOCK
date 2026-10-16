# Quantum Traffic Gridlock — IBM Qiskit Fall Fest 2026 (QC@IITI)

A complete, scientifically grounded, reproducible quantum optimization solution for coordinating multi-intersection urban traffic signals using **QUBO** and **QAOA** (Quantum Approximate Optimization Algorithm) on Qiskit.

## Deliverables & Documentation

- **Technical Report**: [report.pdf](./report.pdf) (strictly <= 3 pages)
- **AI-Use Disclosure**: [AI_USE.pdf](./AI_USE.pdf) (1-page standalone disclosure)
- **Source Files**: [report.md](./report.md), [AI_USE.md](./AI_USE.md)

## Repository Structure

```text
QUANTUM-TRAFFIC-GRIDLOCK/
├── assets/
│   └── screenshots/
│       └── image.png                 # Pytest verification output screenshot
├── outputs/
│   └── R_seed42/                     # Generated deliverables for Scenario R (Seed 42)
│       ├── baselines.json            # 6 evaluated benchmark comparisons
│       ├── circuit.qasm              # Final bound QAOA circuit in OpenQASM 3.0
│       ├── counts.json               # Exactly 4,096 raw measurement shots
│       ├── metrics.json              # Lift, clipped lift, elite enrichment factor
│       ├── multi_seed_results.json   # Robustness evaluation across 5 fixed seeds
│       ├── qubo.json                 # Upper-triangular QUBO & Ising parameters
│       ├── run.log                   # Complete execution and training trace log
│       ├── schedule.json             # Decoded optimal 8-bit signal schedule
│       ├── surrogate_validation.json # QUBO energy vs official cost correlation
│       └── training.json             # COBYLA evaluation trace and optimal angles
├── scenarios/
│   ├── R.json                        # Public Reference Scenario R
│   ├── scenario_R.json               # Official scorer schema Reference Scenario R
│   ├── heavy_ns.json                 # Asymmetric heavy North-South arrivals
│   └── balanced.json                 # Symmetrical multi-approach load test
├── tests/
│   ├── test_baselines_and_metrics.py # Unit tests for benchmark calculators
│   ├── test_cli.py                   # CLI end-to-end regression test
│   ├── test_generalization.py        # QUBO validation across unseen traffic
│   ├── test_qaoa_solver.py           # Circuit depth, endianness, and shot bounds
│   ├── test_qubo_and_ising.py        # QUBO-Ising equivalence and regret verification
│   ├── test_scenario.py              # Scenario schema parser and validator tests
│   └── test_traffic_model.py         # Official scorer vector regression checks
├── AI_USE.md                         # Markdown source of AI assistance disclosure
├── AI_USE.pdf                        # Standalone 1-page AI disclosure document
├── baselines.py                      # Alternating, greedy, enumeration, and hybrid solvers
├── challenge_spec.txt                # Plaintext transcript of challenge guidelines
├── extensions.py                     # Ablation studies, minimum-green, & noise models
├── ising.py                          # QUBO to Ising transformation and validation
├── metrics.py                        # Lift, Elite Enrichment, and surrogate calculations
├── pytest.ini                        # Pytest configuration file
├── qaoa_solver.py                    # Multi-start QAOA parameter training & sampling
├── qubo_builder.py                   # Analytical queue and switching QUBO model
├── README.md                         # Project documentation and execution instructions
├── report.md                         # Technical report markdown source
├── report.pdf                        # Technical report PDF (strictly <= 3 pages)
├── requirements.txt                  # Pinned dependencies
├── run.py                            # Official CLI entrypoint pipeline
├── scenario.py                       # Robust multi-schema scenario parser
├── scenario_R.json                   # Official reference scenario from scorer package
├── scorer.py                         # Authoritative evaluation module
├── serialization.py                  # Standardized JSON and QASM 3 artifact writers
└── traffic_model.py                  # Deterministic traffic dynamics simulation
```

---

## 1. Problem Overview

Urban traffic lights at adjacent intersections strongly influence one another: clearing vehicles through upstream intersection A discharges traffic downstream to intersection B. The goal is to optimize the 8-bit signal schedule across two intersections (A, B) over four 30-second time slots.

### Decision variable representation

- `1` = North-South (NS) phase receives green.
- `0` = East-West (EW) phase receives green.
- The 8 schedule decision bits are strictly ordered as:

```text
A_slot1, A_slot2, A_slot3, A_slot4, B_slot1, B_slot2, B_slot3, B_slot4
```

### Objective (official cost)

```text
Cost = Total Delay + w_overflow · Total Overflow + w_switch · Total Switches
```

- **Delay**: Sum of residual queue lengths across all approaches and time slots after service.
- **Overflow**: Queue accumulation exceeding nominal storage `S`.
- **Switches**: Phase transitions between consecutive slots, `|x_{t+1} − x_t|`. No penalty before slot 1.
- **Transfers**: Inter-junction transfers on the directed link A→B: `⌊α · departures + 0.5⌋` vehicles enter the downstream approach at `t+1`.

---

## 2. Mathematical QUBO Derivation

Rather than fitting or hardcoding scenario answers, the QUBO is constructed from first-principles traffic dynamics. (Slots are indexed `t = 0..3` in the math below, i.e. slots 1–4 in the bit ordering above.)

**1. Exact switching**

```text
H_switch = w_switch · Σ_{I∈{A,B}} Σ_{t=0..2} ( x_{I,t} + x_{I,t+1} − 2·x_{I,t}·x_{I,t+1} )
```

**2. Effective queue evolution** (unclipped cumulative queue)

```text
q̃_{app,t} = Σ_{s=0..t} a_{app,s} + Σ_{s=0..t-1} τ_{s+1}(x) − Σ_{s=0..t} C_eff,s · g_{app,s}(x)
```

where `C_eff,0 = min(a_{app,0}, C)` models initial emptiness without unphysical negative-departure clipping, and `C_eff,s = C` for `s ≥ 1`.

**3. Calibrated quadratic queue barrier**

```text
H_queue = Σ_{t=0..3} Σ_{app} ( q̃_{app,t} + (w_overflow / (2·S·C)) · q̃_{app,t}² )
```

This ensures the marginal penalty at the nominal storage boundary `q = S` scales with `w_overflow`.

**4. Dynamic link coupling**

Transfers `τ_{t+1} = ⌊α·C + 0.5⌋ · g_{A,from,t}` dynamically couple upstream signal decisions to downstream queue arrivals, producing off-diagonal bilinear QUBO terms.

---

## 3. Installation & Requirements

Python 3.11+ is recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

Pinned packages:

- `qiskit>=2.0.0`
- `qiskit-aer>=0.17.0`
- `numpy>=2.0.0`
- `scipy>=1.14.0`
- `pytest>=8.0.0`

---

## 4. Execution Interface

Run the pipeline from the command line:

```bash
python run.py --scenario scenarios/R.json --seed 42 --out outputs/R_seed42
```

To evaluate robustness across 5 fixed seeds (42, 43, 44, 45, 46):

```bash
python run.py --scenario scenarios/R.json --seed 42 --out outputs/R_seed42 --eval-multi-seeds
```

---

## 5. Required Output Artifacts

For each run, the pipeline generates:

- `qubo.json`: QUBO upper-triangular matrix, offset, variable mappings, and derivation rationale.
- `circuit.qasm`: Final bound-parameter QAOA circuit in OpenQASM 3.0 format.
- `counts.json`: Unmodified raw 4,096 measurement counts from AerSimulator.
- `schedule.json`: Decoded schedules, bit ordering, best raw QAOA sample, and hybrid classical refinement.
- `metrics.json`: Lift, clipped lift, elite enrichment factor, and shot counts.
- `baselines.json`: Benchmark comparisons across 6 baselines.
- `training.json`: Optimizer evaluation history and optimal variational angles (γ*, β*).
- `surrogate_validation.json`: Statistical comparison of QUBO energy vs official cost over all 256 schedules.
- `run.log`: Full execution trace.

---

## 6. Official Verification Vectors & Test Suite

The traffic evaluator reproduces the official verification vectors exactly:

| Schedule (A \| B) | Official Cost |
|---|---:|
| `1010 \| 1010` | 116.0 |
| `1111 \| 1111` | 146.0 |
| `0000 \| 0000` | 237.0 |
| `0111 \| 1001` | 92.5 |

Run all unit and regression tests:

```bash
pytest -v
```

### Pytest Execution Output (16/16 Passed)

```text
==================================== test session starts ====================================
platform win32 -- Python 3.12.5, pytest-9.1.1, pluggy-1.7.0
rootdir: E:\QUANTUM-TRAFFIC-GRIDLOCK
configfile: pytest.ini
testpaths: tests
collected 16 items

tests/test_baselines_and_metrics.py::test_baselines PASSED                           [  6%]
tests/test_baselines_and_metrics.py::test_metrics_computation PASSED                 [ 12%]
tests/test_cli.py::test_cli_execution PASSED                                         [ 18%]
tests/test_generalization.py::test_unseen_scenarios_pipeline[scenarios/heavy_ns.json] PASSED [ 25%]
tests/test_generalization.py::test_unseen_scenarios_pipeline[scenarios/balanced.json] PASSED [ 31%]
tests/test_qaoa_solver.py::test_qiskit_endianness_decoding PASSED                    [ 37%]
tests/test_qaoa_solver.py::test_qaoa_execution_and_constraints PASSED                [ 43%]
tests/test_qubo_and_ising.py::test_qubo_structure PASSED                             [ 50%]
tests/test_qubo_and_ising.py::test_qubo_ising_equivalence_all_states PASSED          [ 56%]
tests/test_qubo_and_ising.py::test_qubo_regret_scenario_r PASSED                     [ 62%]
tests/test_scenario.py::test_valid_scenario_parsing PASSED                           [ 68%]
tests/test_scenario.py::test_scenario_validation_errors PASSED                       [ 75%]
tests/test_traffic_model.py::test_four_official_reference_vectors PASSED             [ 81%]
tests/test_traffic_model.py::test_schedule_bit_parsing PASSED                        [ 87%]
tests/test_traffic_model.py::test_switching_cost_exactness PASSED                    [ 93%]
tests/test_traffic_model.py::test_overflow_and_transfer_dynamics PASSED              [100%]

=================================== 16 passed in 20.01s ====================================
```

---

## 7. Performance & Resource Compliance

| Metric | Value | Limit |
|---|---|---|
| Qubit count | Exactly 8 | <= 16 |
| Circuit depth | p = 3 | p <= 4 |
| Optimizer evaluations | 150 | <= 300 |
| Shot count | Exactly 4,096 | 4,096 |
| Runtime | ~2.3 s per run | <= 10 min |
| Surrogate regret on R | 0.00% (QUBO minimizer finds the exact global optimum `11101010`) | <= 5% for full credit |
| Seed 42 sampling lift | 0.1795 (E_Q = 137.0223, Elite Enrichment 1.125x) | Clipped to [0, 1] |
| 5-seed robustness | Mean Lift: 0.1545, Min Clipped Lift: 0.0589 (Seeds 42–46) | >= 5 fixed seeds |