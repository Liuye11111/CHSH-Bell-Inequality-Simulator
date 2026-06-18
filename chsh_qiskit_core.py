"""
Qiskit Bell态制备与CHSH不等式蒙特卡洛仿真核心模块

运行方式：
    python chsh_qiskit_core.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity


def create_bell_state() -> Tuple[QuantumCircuit, np.ndarray]:
    """
    使用Qiskit量子电路制备Bell态 |Phi+>。

    电路：
        |00> -- H(q0) -- CNOT(q0 -> q1) -->
        (|00> + |11>) / sqrt(2)
    """
    circuit = QuantumCircuit(2, name="Bell state")
    circuit.h(0)
    circuit.cx(0, 1)

    statevector = Statevector.from_instruction(circuit)
    return circuit, np.asarray(statevector.data, dtype=complex)


BELL_CIRCUIT, BELL_PHI_PLUS = create_bell_state()


def bell_state_fidelity() -> float:
    ideal_state = np.array(
        [1.0, 0.0, 0.0, 1.0],
        dtype=complex
    ) / np.sqrt(2.0)

    return float(
        state_fidelity(
            Statevector(BELL_PHI_PLUS),
            Statevector(ideal_state)
        )
    )


def bell_state_details() -> Dict:
    basis_labels = ["|00>", "|01>", "|10>", "|11>"]

    return {
        "circuit_text": str(BELL_CIRCUIT.draw(output="text")),
        "statevector": BELL_PHI_PLUS.copy(),
        "amplitudes": {
            label: complex(amplitude)
            for label, amplitude in zip(
                basis_labels,
                BELL_PHI_PLUS
            )
        },
        "probabilities": {
            label: float(abs(amplitude) ** 2)
            for label, amplitude in zip(
                basis_labels,
                BELL_PHI_PLUS
            )
        },
        "fidelity": bell_state_fidelity(),
    }


def measurement_basis(angle_deg: float) -> Dict[int, np.ndarray]:
    theta = np.deg2rad(float(angle_deg))

    plus_state = np.array(
        [np.cos(theta), np.sin(theta)],
        dtype=complex
    )
    minus_state = np.array(
        [-np.sin(theta), np.cos(theta)],
        dtype=complex
    )

    return {1: plus_state, -1: minus_state}


def joint_probabilities(
    alice_angle: float,
    bob_angle: float,
    state: np.ndarray = BELL_PHI_PLUS,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    state = np.asarray(state, dtype=complex).reshape(-1)

    if state.size != 4:
        raise ValueError("双比特量子态必须包含4个复振幅。")

    norm = np.linalg.norm(state)
    if norm == 0:
        raise ValueError("量子态不能是零向量。")

    state = state / norm
    alice_basis = measurement_basis(alice_angle)
    bob_basis = measurement_basis(bob_angle)

    outcomes: List[Tuple[int, int]] = [
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]

    probabilities = []

    for alice_result, bob_result in outcomes:
        joint_basis = np.kron(
            alice_basis[alice_result],
            bob_basis[bob_result]
        )
        amplitude = np.vdot(joint_basis, state)
        probabilities.append(float(np.abs(amplitude) ** 2))

    probabilities_array = np.asarray(
        probabilities,
        dtype=float
    )
    probabilities_array = np.clip(
        probabilities_array,
        0.0,
        1.0
    )
    probabilities_array /= probabilities_array.sum()

    return outcomes, probabilities_array


def theoretical_correlation(
    alice_angle: float,
    bob_angle: float
) -> float:
    difference = float(alice_angle) - float(bob_angle)

    return float(
        np.cos(2.0 * np.deg2rad(difference))
    )


def simulate_measurement_pair(
    alice_angle: float,
    bob_angle: float,
    shots: int,
    rng: np.random.Generator,
) -> Dict:
    if not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError("shots必须为正整数。")

    outcomes, probabilities = joint_probabilities(
        alice_angle,
        bob_angle
    )

    sampled_indices = rng.choice(
        len(outcomes),
        size=shots,
        p=probabilities
    )

    alice_results = np.fromiter(
        (
            outcomes[index][0]
            for index in sampled_indices
        ),
        dtype=int,
        count=shots
    )

    bob_results = np.fromiter(
        (
            outcomes[index][1]
            for index in sampled_indices
        ),
        dtype=int,
        count=shots
    )

    counts = {
        outcome: int(
            np.count_nonzero(
                sampled_indices == index
            )
        )
        for index, outcome in enumerate(outcomes)
    }

    return {
        "alice_angle": float(alice_angle),
        "bob_angle": float(bob_angle),
        "shots": int(shots),
        "counts": counts,
        "simulated_correlation": float(
            np.mean(alice_results * bob_results)
        ),
        "theoretical_correlation": theoretical_correlation(
            alice_angle,
            bob_angle
        ),
    }


def simulate_chsh(
    a: float = 0.0,
    a_prime: float = 45.0,
    b: float = 22.5,
    b_prime: float = -22.5,
    shots: int = 10_000,
    seed: Optional[int] = 42,
) -> Dict:
    rng = np.random.default_rng(seed)

    result_ab = simulate_measurement_pair(
        a, b, shots, rng
    )
    result_ab_prime = simulate_measurement_pair(
        a, b_prime, shots, rng
    )
    result_a_prime_b = simulate_measurement_pair(
        a_prime, b, shots, rng
    )
    result_a_prime_b_prime = simulate_measurement_pair(
        a_prime,
        b_prime,
        shots,
        rng
    )

    simulated_s = (
        result_ab["simulated_correlation"]
        + result_ab_prime["simulated_correlation"]
        + result_a_prime_b["simulated_correlation"]
        - result_a_prime_b_prime["simulated_correlation"]
    )

    theoretical_s = (
        result_ab["theoretical_correlation"]
        + result_ab_prime["theoretical_correlation"]
        + result_a_prime_b["theoretical_correlation"]
        - result_a_prime_b_prime["theoretical_correlation"]
    )

    return {
        "angles": {
            "a": float(a),
            "a_prime": float(a_prime),
            "b": float(b),
            "b_prime": float(b_prime),
        },
        "shots_per_setting": int(shots),
        "seed": seed,
        "pair_results": {
            "E(a,b)": result_ab,
            "E(a,b')": result_ab_prime,
            "E(a',b)": result_a_prime_b,
            "E(a',b')": result_a_prime_b_prime,
        },
        "simulated_s": float(simulated_s),
        "theoretical_s": float(theoretical_s),
        "classical_bound": 2.0,
        "quantum_bound": float(2.0 * np.sqrt(2.0)),
    }


def self_check() -> None:
    assert np.isclose(
        bell_state_fidelity(),
        1.0,
        atol=1e-12
    )

    _, probabilities = joint_probabilities(
        0.0,
        22.5
    )
    assert np.isclose(
        probabilities.sum(),
        1.0,
        atol=1e-12
    )

    result = simulate_chsh(
        shots=100_000,
        seed=12345
    )
    assert abs(result["simulated_s"]) > 2.0


if __name__ == "__main__":
    self_check()

    details = bell_state_details()
    print(details["circuit_text"])
    print("Statevector:", details["statevector"])
    print("Fidelity:", details["fidelity"])

    result = simulate_chsh()
    print("Simulated S:", result["simulated_s"])
    print("Theoretical S:", result["theoretical_s"])
