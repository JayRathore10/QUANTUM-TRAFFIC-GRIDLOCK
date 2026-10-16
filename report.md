# Quantum Traffic Gridlock — Technical Report

**IBM Qiskit Fall Fest 2026 | QC@IITI, IIT Indore**

## 1. Mathematical Formulation & Exact QUBO Derivation

The traffic network consists of two connected intersections, `A` and `B`, optimized over four discrete 30-second time slots ($t \in \{0, 1, 2, 3\}$). The system is modeled using 8 binary decision variables $x_{I,t} \in \{0,1\}$ ordered as `[A_slot1..4, B_slot1..4]`, where `1` denotes a North-South (NS) green phase and `0` denotes an East-West (EW) green phase. The objective is to minimize the official composite traffic cost:

$$\text{Cost}(x) = \text{Delay} + w_{\text{overflow}} \cdot \text{Overflow} + w_{\text{switch}} \cdot \text{Switches}$$

Rather than empirically tuning or hardcoding scenario parameters, our QUBO is derived analytically from first-principles traffic dynamics to ensure scenario-independent generalization.

### 1.1 Exact Switching Penalties

Phase switches between adjacent slots at each intersection are represented exactly as quadratic binary terms without approximation:

$$H_{\text{switch}} = w_{\text{switch}} \sum_{I \in \{A,B\}} \sum_{t=0}^{2} \vert x_{I,t+1} - x_{I,t} \vert = w_{\text{switch}} \sum_{I \in \{A,B\}} \sum_{t=0}^{2} \left(x_{I,t} + x_{I,t+1} - 2x_{I,t}x_{I,t+1}\right)$$

### 1.2 Effective Departure Capacity & Dynamic Link Coupling

To prevent unphysical negative queues without adding auxiliary slack qubits, the slot-0 departure capacity is bounded by initial arrivals: $C_{\text{eff},0} = \min(a_{app,0}, C)$, and $C_{\text{eff},s} = C$ for $s \ge 1$. For any approach, the cumulative unclipped queue at slot $t$ is linear in the phase variables:

$$\tilde{q}_{app,t}(x) = \sum_{s=0}^{t} a_{app,s} + \sum_{s=0}^{t-1} \tau_{s+1}(x) - \sum_{s=0}^{t} C_{\text{eff},s} \, g_{app,s}(x)$$

where $g_{NS,s} = x_{I,s}$, $g_{EW,s} = 1 - x_{I,s}$, and directed $A \to B$ link transfers $\tau_{s+1} = \lfloor \alpha C + 0.5 \rfloor \, g_{A,\text{src},s}$ introduce off-diagonal bilinear couplings $Q_{i,j}$ between upstream green decisions at $A$ (slot $s$) and downstream queue accumulation at $B$ (slots $t > s$).

### 1.3 Quadratic Queue Barrier & Surrogate Fidelity

Because the exact overflow $\max(0, q_t - S)$ is piecewise-linear, we approximate combined delay and overflow via a calibrated quadratic barrier:

$$H_{\text{queue}} = \sum_{t=0}^{3} \sum_{app} \left( w_{\text{lin}}\tilde{q}_{app,t} + w_{\text{quad}}\tilde{q}_{app,t}^{2} \right), \quad w_{\text{lin}} = 1.0, \quad w_{\text{quad}} = \frac{w_{\text{overflow}}}{2SC}$$

Setting $w_{\text{quad}} = w_{\text{overflow}} / (2SC)$ ensures the marginal penalty $\partial H_{\text{queue}}/\partial \tilde{q}$ at the storage threshold $\tilde{q} = S$ scales proportionally with $w_{\text{overflow}}$. On Reference Scenario R, the QUBO achieves a **Pearson correlation of 0.9738** across all 256 states and **0.00% surrogate regret**, with its exact minimizer matching the true global optimum (`11101010`, $C^* = 72.0$). Across unseen scenarios (`heavy_ns.json`, `balanced.json`), surrogate regret remains $\le 5\%$. Exact equivalence between the upper-triangular QUBO and Ising Hamiltonian ($x_i = (1 - Z_i)/2$) is verified across all 256 basis states to machine precision ($\le 1.71 \times 10^{-13}$).

## 2. QAOA Implementation, Multi-Start Training & Ablations

### 2.1 Resource Adherence & Circuit Architecture

- **Qubits & Depth:** Exactly 8 qubits (limit $\le 16$) and depth $p = 3$ (limit $p \le 4$), exported to OpenQASM 3.0 (`circuit.qasm`).
- **Evaluation & Shot Limits:** 150 COBYLA objective evaluations (limit $\le 300$) using exact statevector probabilities $\langle H_C \rangle = \sum_x p(x)E(x)$ during training, followed by **exactly 4,096 raw measurement shots** on `AerSimulator` (`counts.json`).
- **Endianness Decoding:** Qiskit little-endian bitstrings `[q7..q0]` are reversed (`bitstring[::-1]`) to map qubit $i$ to schedule bit $i$.

### 2.2 Spectrally Normalized Multi-Start Optimization

To mitigate barren plateaus and local traps in COBYLA, we normalize the Ising coefficients by $h_{\text{scale}} = \max(1, 0.5(\max_i \vert h_i \vert + \max_{i<j}\vert J_{ij} \vert))$ and divide the 150-evaluation budget across three deterministic starts (50 evaluations each): (1) a linear ramp, (2) a spectrally normalized adiabatic ramp, and (3) a harmonic-seeded ramp.

### 2.3 Systematic Ablation Studies

Implemented in `extensions.py`, our ablations on Scenario R demonstrate:

1. **Circuit Depth ($p \in \{1, 2, 3, 4\}$):** Increasing depth from $p=1$ to $p=3$ steadily reduces expected QUBO energy and improves sampling lift; $p=3$ offers the best accuracy-to-gate-count tradeoff before $p=4$ experiences classical optimizer saturation under the evaluation cap.
2. **Penalty Scaling ($\text{quad\_scale} \in \{0.25, 0.5, 1.0, 2.0, 4.0\}$):** The analytical calibration ($\text{scale} = 1.0$) maximizes Pearson correlation ($0.9738$) and yields $0.00\%$ regret, whereas extreme under-scaling ($0.25$) under-penalizes storage overflow and over-scaling ($4.0$) distorts linear delay tradeoffs.
3. **Initialization Strategy:** Multi-start and linear/spectral ramps consistently outperform unscaled random initialization in both final expected cost $E_Q$ and elite enrichment.

## 3. Baselines, Sampling Quality & Multi-Seed Robustness

### 3.1 Mandatory Baseline Comparison (Scenario R, Seed 42)

| Baseline / Method | Schedule | Official Cost | Delay / Overflow / Switches | Notes |
|---|---|---:|---|---|
| **Exact Enumeration ($C^*$)** | `11101010` | **72.00** | 66 / 1 / 4 | Global optimum over all 256 schedules (worst: `00001111` at 297.0). |
| **Exact QUBO Minimizer** | `11101010` | **72.00** | 66 / 1 / 4 | Surrogate ceiling; **0.00% regret** (QUBO energy: 89.26). |
| **Per-Slot Greedy Control** | `01111001` | **92.50** | 67 / 6 / 3 | Serves $\max(q_{NS}, q_{EW})$ online after arrivals (ties to NS). |
| **Best Alternating** | `10101010` | **116.00** | 73 / 10 / 6 | Best of 4 cyclic schedules (`10101010`, `10100101`, `01011010`, `01010101`). |
| **Uniform Random ($C_{\text{rand}}$)** | N/A | **151.25** | N/A | Mean official cost across all 256 schedules. |
| **QAOA Raw Sampling (Seed 42)** | `11101010` (best) | **137.02** ($E_Q$) | 66 / 1 / 4 (best) | Raw lift = **0.1795**, elite enrichment = **1.125x** (234 / 4,096 elite shots; `11101010` sampled 20 times). |
| **Hybrid QAOA + Local Search** | `11101010` | **72.00** | 66 / 1 / 4 | Separately labeled 1-bit-flip classical refinement from the best QAOA sample. |

### 3.2 Five-Seed Robustness Evaluation (Scenario R)

Evaluating across 5 fixed seeds (`42, 43, 44, 45, 46`) yields positive lift on every seed, with **Mean Clipped Lift = 0.1545** and **Worst-Seed Clipped Lift = 0.0589**:

| Seed | Mean Cost ($E_Q$) | Raw Lift | Clipped Lift $[0,1]$ | Elite Shots (of 4,096) | Elite Enrichment |
|:---:|---:|---:|---:|---:|---:|
| **42** | 137.0223 | 0.1795 | 0.1795 | 234 | 1.1250x |
| **43** | 145.3016 | 0.0751 | 0.0751 | 234 | 1.1250x |
| **44** | 133.2650 | 0.2269 | 0.2269 | 286 | 1.3750x |
| **45** | 132.8557 | 0.2321 | 0.2321 | 207 | 0.9952x |
| **46** | 146.5801 | 0.0589 | 0.0589 | 182 | 0.8750x |

## 4. Innovation, Extensions & Limitations

### 4.1 Mathematical Extensions (`extensions.py`)

1. **Minimum-Green Phase Hold ($\tau_{\min} = 2$):** Rapid 1-slot phase oscillations (`1 -> 0 -> 1` or `0 -> 1 -> 0`) create traffic safety hazards. We add a 3-slot quadratic penalty $H_{\text{mg}} = \lambda_{\text{mg}} \sum_{I, t \in \{1,2\}} (x_{I,t} - x_{I,t-1}x_{I,t} - x_{I,t}x_{I,t+1} + x_{I,t-1}x_{I,t+1})$ with $\lambda_{\text{mg}} = 2.5$ within the same 8-qubit space, suppressing isolated 1-slot phase flips.
2. **Gate-Level Depolarizing Noise Analysis:** We simulate noisy execution on `AerSimulator` by injecting 2-qubit depolarizing errors $\epsilon_{2q} \in \{0.002, 0.005, 0.01, 0.02\}$ on `rzz`/`cx` gates and $\epsilon_{1q} = \epsilon_{2q}/10$ on `h`/`rx`/`rz` gates, quantifying graceful degradation of lift and elite enrichment on NISQ hardware.

### 4.2 Limitations

- **Unclipped Surrogate Queues:** To avoid adding auxiliary slack qubits per slot/approach, $\tilde{q}_{app,t}$ is modeled linearly without $\max(0, \cdot)$ clipping for $t \ge 1$. Under extremely sparse arrival regimes (e.g., consecutive zero arrivals), $\tilde{q}$ can under-estimate true queues if green phases starve.
- **Smooth Overflow Approximation:** The quadratic barrier $w_{\text{quad}}\tilde{q}^2$ penalizes moderate queues ($0 < \tilde{q} \le S$) slightly more than the exact zero-below-storage hinge $\max(0, q - S)$, causing minor energy offset shifts while preserving state rankings.
- **Shallow-Depth QAOA Concentration:** At $p=3$ with standard transverse-field mixers, QAOA reliably shifts probability mass toward low-cost schedules (positive lift and global optimum sampling) but retains broad support across the 256-state space compared to classical exact enumeration.