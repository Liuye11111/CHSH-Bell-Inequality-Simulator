"""
CHSH 不等式蒙特卡洛仿真核心模块

功能：
1. 构造双比特 Bell 态 |Phi+>
2. 计算不同测量角度下的联合概率
3. 通过蒙特卡洛方法模拟重复测量
4. 计算关联期望值 E(a, b)
5. 计算 CHSH 参数 S

运行方式：
    python chsh_simulator.py
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# Bell 态：
# |Phi+> = (|00> + |11>) / sqrt(2)
BELL_PHI_PLUS = np.array(
    [1.0, 0.0, 0.0, 1.0],
    dtype=complex
) / np.sqrt(2.0)


def measurement_basis(angle_deg: float) -> Dict[int, np.ndarray]:
    """
    根据线偏振测量角 angle_deg 生成两个正交测量基矢。

    这里采用：
        |+_theta> = cos(theta)|0> + sin(theta)|1>
        |-_theta> = -sin(theta)|0> + cos(theta)|1>

    参数：
        angle_deg: 测量角度，单位为度。

    返回：
        字典 {+1: |+_theta>, -1: |-_theta>}
    """
    theta = np.deg2rad(float(angle_deg))

    plus_state = np.array(
        [np.cos(theta), np.sin(theta)],
        dtype=complex
    )

    minus_state = np.array(
        [-np.sin(theta), np.cos(theta)],
        dtype=complex
    )

    return {
        1: plus_state,
        -1: minus_state
    }


def joint_probabilities(
    alice_angle: float,
    bob_angle: float,
    state: np.ndarray = BELL_PHI_PLUS
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    计算 Alice 和 Bob 在给定测量角度下四种联合结果的概率。

    四种结果：
        (+1, +1)
        (+1, -1)
        (-1, +1)
        (-1, -1)
    """
    state = np.asarray(state, dtype=complex).reshape(-1)

    if state.size != 4:
        raise ValueError("双比特量子态必须包含 4 个复振幅。")

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
        (-1, -1)
    ]

    probabilities = []

    for alice_result, bob_result in outcomes:
        joint_basis = np.kron(
            alice_basis[alice_result],
            bob_basis[bob_result]
        )

        amplitude = np.vdot(joint_basis, state)
        probability = float(np.abs(amplitude) ** 2)
        probabilities.append(probability)

    probabilities_array = np.asarray(probabilities, dtype=float)

    # 修正浮点运算可能产生的极小负值，并重新归一化
    probabilities_array = np.clip(probabilities_array, 0.0, 1.0)
    total_probability = probabilities_array.sum()

    if total_probability <= 0:
        raise RuntimeError("联合概率计算失败。")

    probabilities_array /= total_probability

    return outcomes, probabilities_array


def theoretical_correlation(
    alice_angle: float,
    bob_angle: float
) -> float:
    """
    Bell 态 |Phi+> 在当前测量基定义下的理论关联：

        E(alpha, beta) = cos[2(alpha - beta)]
    """
    angle_difference = float(alice_angle) - float(bob_angle)

    return float(
        np.cos(2.0 * np.deg2rad(angle_difference))
    )


def simulate_measurement_pair(
    alice_angle: float,
    bob_angle: float,
    shots: int,
    rng: np.random.Generator
) -> Dict:
    """
    对一组测量角度进行蒙特卡洛重复测量。
    """
    if not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError("shots 必须为正整数。")

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
        (outcomes[index][0] for index in sampled_indices),
        dtype=int,
        count=shots
    )

    bob_results = np.fromiter(
        (outcomes[index][1] for index in sampled_indices),
        dtype=int,
        count=shots
    )

    products = alice_results * bob_results
    simulated_correlation = float(np.mean(products))

    counts = {
        outcome: int(np.count_nonzero(sampled_indices == index))
        for index, outcome in enumerate(outcomes)
    }

    return {
        "alice_angle": float(alice_angle),
        "bob_angle": float(bob_angle),
        "shots": int(shots),
        "counts": counts,
        "probabilities": {
            outcome: float(probability)
            for outcome, probability in zip(outcomes, probabilities)
        },
        "simulated_correlation": simulated_correlation,
        "theoretical_correlation": theoretical_correlation(
            alice_angle,
            bob_angle
        )
    }


def simulate_chsh(
    a: float = 0.0,
    a_prime: float = 45.0,
    b: float = 22.5,
    b_prime: float = -22.5,
    shots: int = 10_000,
    seed: Optional[int] = 42
) -> Dict:
    """
    完成四组测量并计算 CHSH 参数。

    采用：
        S = E(a,b) + E(a,b') + E(a',b) - E(a',b')

    默认角度：
        a  = 0°
        a' = 45°
        b  = 22.5°
        b' = -22.5°

    理论上可得到：
        |S| = 2*sqrt(2) ≈ 2.828
    """
    if not isinstance(shots, (int, np.integer)) or shots <= 0:
        raise ValueError("shots 必须为正整数。")

    rng = np.random.default_rng(seed)

    result_ab = simulate_measurement_pair(a, b, shots, rng)
    result_ab_prime = simulate_measurement_pair(a, b_prime, shots, rng)
    result_a_prime_b = simulate_measurement_pair(a_prime, b, shots, rng)
    result_a_prime_b_prime = simulate_measurement_pair(
        a_prime, b_prime, shots, rng
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
            "b_prime": float(b_prime)
        },
        "shots_per_setting": int(shots),
        "seed": seed,
        "pair_results": {
            "E(a,b)": result_ab,
            "E(a,b')": result_ab_prime,
            "E(a',b)": result_a_prime_b,
            "E(a',b')": result_a_prime_b_prime
        },
        "simulated_s": float(simulated_s),
        "theoretical_s": float(theoretical_s),
        "classical_bound": 2.0,
        "quantum_bound": float(2.0 * np.sqrt(2.0)),
        "violates_classical_bound": bool(abs(simulated_s) > 2.0)
    }


def print_results(results: Dict) -> None:
    """
    将仿真结果打印到终端。
    """
    print("=" * 62)
    print("CHSH Bell Inequality Monte Carlo Simulation")
    print("=" * 62)

    angles = results["angles"]
    print(
        "测量角度："
        f"a={angles['a']:.1f}°, "
        f"a'={angles['a_prime']:.1f}°, "
        f"b={angles['b']:.1f}°, "
        f"b'={angles['b_prime']:.1f}°"
    )
    print(f"每组测量次数：{results['shots_per_setting']}")
    print(f"随机种子：{results['seed']}")
    print("-" * 62)

    for name, pair_result in results["pair_results"].items():
        simulated = pair_result["simulated_correlation"]
        theoretical = pair_result["theoretical_correlation"]

        print(
            f"{name:<10} "
            f"模拟值={simulated:>8.4f}   "
            f"理论值={theoretical:>8.4f}"
        )

    print("-" * 62)
    print(f"模拟 CHSH 参数 S = {results['simulated_s']:.4f}")
    print(f"理论 CHSH 参数 S = {results['theoretical_s']:.4f}")
    print(f"经典极限          = {results['classical_bound']:.4f}")
    print(f"量子上限          = {results['quantum_bound']:.4f}")

    if results["violates_classical_bound"]:
        print("结论：|S| > 2，观察到 CHSH 不等式违背。")
    else:
        print("结论：本次有限样本模拟中未观察到 CHSH 不等式违背。")

    print("=" * 62)


def self_check() -> None:
    """
    简单自检：
    1. 四种联合概率之和应为 1；
    2. 默认理论 CHSH 值应接近 2*sqrt(2)；
    3. 大样本模拟应明显超过经典极限 2。
    """
    _, probabilities = joint_probabilities(0.0, 22.5)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-12)

    result = simulate_chsh(shots=100_000, seed=12345)

    assert np.isclose(
        abs(result["theoretical_s"]),
        2.0 * np.sqrt(2.0),
        atol=1e-12
    )

    assert abs(result["simulated_s"]) > 2.0


if __name__ == "__main__":
    self_check()
    simulation_results = simulate_chsh()
    print_results(simulation_results)
