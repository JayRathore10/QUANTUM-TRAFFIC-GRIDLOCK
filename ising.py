"""QUBO to Ising Hamiltonian conversion and bidirectional verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from qubo_builder import QUBOResult


@dataclass
class IsingHamiltonian:
    """Ising model specification: H = h0 * I + sum_i h_i * Z_i + sum_{i<j} J_ij * Z_i * Z_j.

    Spin convention:
      s_i in {+1, -1}
      x_i = (1 - s_i) / 2
      s_i = 1 - 2 * x_i
      |0> state -> s = +1 (x = 0)
      |1> state -> s = -1 (x = 1)
    """

    h0: float
    h: np.ndarray  # 1D array of length n
    J: np.ndarray  # 2D upper-triangular array of shape (n, n)
    num_qubits: int = 8

    def evaluate(self, spins: Sequence[int]) -> float:
        """Evaluate Ising energy for spin configuration s in {+1, -1}^n."""
        s = np.array(spins, dtype=float)
        energy = self.h0 + float(np.dot(self.h, s))
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                energy += self.J[i, j] * s[i] * s[j]
        return float(energy)

    def evaluate_binary(self, x: Sequence[int]) -> float:
        """Evaluate Ising energy directly for binary configuration x in {0, 1}^n."""
        spins = [1 - 2 * b for b in x]
        return self.evaluate(spins)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Ising Hamiltonian to JSON-serializable dictionary."""
        return {
            "h0_offset": float(self.h0),
            "h_linear": self.h.tolist(),
            "J_coupling": self.J.tolist(),
            "num_qubits": self.num_qubits,
            "spin_convention": "s_i in {+1, -1}, x_i = (1 - s_i) / 2",
        }


def qubo_to_ising(qubo: QUBOResult) -> IsingHamiltonian:
    """Convert an upper-triangular QUBO to an equivalent Ising Hamiltonian.

    Mathematical transformation:
      x_i = (1 - Z_i) / 2
      x_i * x_j = (1 - Z_i - Z_j + Z_i * Z_j) / 4
    """
    Q = qubo.Q
    offset = qubo.offset
    n = Q.shape[0]

    J = np.zeros((n, n), dtype=float)
    h = np.zeros(n, dtype=float)
    h0 = offset + 0.5 * float(np.sum(np.diag(Q)))

    for i in range(n):
        h[i] -= 0.5 * Q[i, i]
        for j in range(i + 1, n):
            q_ij = Q[i, j]
            if abs(q_ij) > 1e-15:
                J[i, j] = 0.25 * q_ij
                h[i] -= 0.25 * q_ij
                h[j] -= 0.25 * q_ij
                h0 += 0.25 * q_ij

    return IsingHamiltonian(h0=h0, h=h, J=J, num_qubits=n)


def verify_qubo_ising_equivalence(qubo: QUBOResult, ising: IsingHamiltonian) -> Tuple[bool, float]:
    """Verify exact equivalence of QUBO and Ising energies across all 2^n configurations.

    Returns (is_valid, max_absolute_error).
    """
    n = qubo.Q.shape[0]
    max_err = 0.0

    for state_int in range(1 << n):
        # Extract binary vector
        x = [(state_int >> (n - 1 - k)) & 1 for k in range(n)]
        e_qubo = qubo.evaluate(x)
        e_ising = ising.evaluate_binary(x)
        err = abs(e_qubo - e_ising)
        if err > max_err:
            max_err = err

    is_valid = max_err < 1e-9
    return is_valid, max_err
