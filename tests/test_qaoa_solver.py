"""Automated tests for QAOA solver, endianness, and QASM export."""

import pytest
from ising import qubo_to_ising
from qaoa_solver import decode_qiskit_bitstring, solve_qaoa
from qubo_builder import build_traffic_qubo
from scenario import parse_scenario


@pytest.fixture
def scenario_r():
    return parse_scenario("scenarios/R.json")


def test_qiskit_endianness_decoding():
    """Verify Qiskit little-endian decoding to canonical schedule bit order."""
    # Qubit 0 = 1 (A_slot1 = 1), rest 0 -> Qiskit bitstring has '1' at index 7: '00000001'
    # Decoded schedule bitstring must have '1' at index 0: '10000000'
    assert decode_qiskit_bitstring("00000001") == "10000000"

    # Qubit 7 = 1 (B_slot4 = 1), rest 0 -> Qiskit bitstring has '1' at index 0: '10000000'
    # Decoded schedule bitstring must have '1' at index 7: '00000001'
    assert decode_qiskit_bitstring("10000000") == "00000001"

    # General pattern
    assert decode_qiskit_bitstring("10101100") == "00110101"


def test_qaoa_execution_and_constraints(scenario_r):
    """Test QAOA execution respects depth, evaluation count, shot count, and QASM export."""
    qubo = build_traffic_qubo(scenario_r)
    ising = qubo_to_ising(qubo)

    result = solve_qaoa(
        qubo=qubo,
        ising=ising,
        p=2,
        max_evaluations=50,
        seed=123,
        final_shots=4096,
    )

    # 1. Total shots must be exactly 4096
    assert sum(result.raw_counts.values()) == 4096
    assert sum(result.decoded_counts.values()) == 4096

    # 2. Evaluations must not exceed limit
    assert result.num_evaluations <= 50

    # 3. Depth must match
    assert result.p_depth == 2

    # 4. Qubits <= 16
    assert result.num_qubits == 8

    # 5. QASM export validity
    assert "OPENQASM 3.0" in result.qasm3_code
    assert "rzz" in result.qasm3_code
