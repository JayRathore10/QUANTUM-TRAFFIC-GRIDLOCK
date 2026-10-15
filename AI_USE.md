# AI Assistance & Attribution Disclosure
**Quantum Traffic Gridlock — IBM Qiskit Fall Fest 2026 (QC@IITI, IIT Indore)**

In compliance with Section 10 ("Mathematical Integrity & AI Use") of the challenge specification, this document transparently describes the utilization of AI assistance, libraries, and independent verification throughout the development of this project.

---

## 1. Scope and Nature of AI Assistance

An agentic AI assistant was utilized as a pair-programming partner to assist with:
- **Code Structuring & Implementation**: Generating initial scaffolding for the modular project architecture (`scenario.py`, `traffic_model.py`, `qubo_builder.py`, `ising.py`, `qaoa_solver.py`, `baselines.py`, `metrics.py`, `serialization.py`, `run.py`).
- **Mathematical Algebra Double-Checking**: Verifying the analytical expansion of quadratic forms in the QUBO-to-Ising conversion matrix elements and confirming tensor sign conventions.
- **Unit Testing Scaffolding**: Drafting automated pytest test cases for edge conditions (zero arrivals, storage overflows, transfer rounding).

---

## 2. Independent Human Verification & Scientific Derivation

All critical scientific and mathematical components were independently validated:
1. **Traffic Dynamics Verification**:
   The four published reference vectors (`1010|1010 -> 116.0`, `1111|1111 -> 146.0`, `0000|0000 -> 237.0`, `0111|1001 -> 92.5`) were independently computed and unit-tested to guarantee zero deviation from official scoring dynamics.
2. **First-Principles QUBO Derivation**:
   No answers, optimal schedules, or scenario-specific parameters were hardcoded or back-propagated into the QUBO matrix. The quadratic model was derived from fundamental physical traffic equations (effective queue departure constraints and marginal overflow penalty scaling).
3. **Algebraic Invariance**:
   Exact state-by-state numerical equivalence between the QUBO matrix and the Ising Hamiltonian was verified across all 256 computational basis states, ensuring machine-precision agreement ($\le 1.71 \times 10^{-13}$).
4. **Quantum Sampling Authenticity**:
   All 4,096 measurement counts reported in `counts.json` were genuinely sampled from the final bound parameterized circuit executed on Qiskit's `AerSimulator`. No counts or benchmark results were synthesized, fabricated, or post-processed.
5. **Separation of Hybrid Methods**:
   Classical local refinement was kept strictly separate from the raw quantum sampling metrics.

---

## 3. Libraries & Environment

- **Python**: 3.12.5
- **Quantum Computing Framework**: Qiskit 2.5.2, Qiskit Aer 0.17.2
- **Scientific Computing**: NumPy 2.1.1, SciPy 1.16.1
- **Testing**: Pytest 9.1.1
