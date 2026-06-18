"""
最终版：Qiskit Bell态制备 + CHSH蒙特卡洛仿真完整界面

运行方式：
    python -m streamlit run app_final.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import streamlit as st

from chsh_qiskit_core import (
    bell_state_details,
    simulate_chsh,
    theoretical_correlation,
)


st.set_page_config(
    page_title="CHSH不等式蒙特卡洛仿真",
    layout="wide",
)


def configure_plot_font() -> bool:
    """优先使用支持中文的字体；没有时图表自动改用英文标签。"""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
    ]
    installed_fonts = {
        font.name for font in font_manager.fontManager.ttflist
    }

    for candidate in candidates:
        if candidate in installed_fonts:
            plt.rcParams["font.sans-serif"] = [candidate]
            plt.rcParams["axes.unicode_minus"] = False
            return True

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return False


HAS_CHINESE_FONT = configure_plot_font()


def plot_text(chinese: str, english: str) -> str:
    return chinese if HAS_CHINESE_FONT else english


def build_angle_scan_data(shots: int, seed: int):
    """
    固定 a=0°、a'=45°，令 b=theta、b'=-theta。
    theta从0°变化到22.5°时，理论|S|从2增加到2sqrt(2)。
    """
    theta_theory = np.linspace(0.0, 22.5, 91)
    theory_s = []

    for theta in theta_theory:
        value = (
            theoretical_correlation(0.0, theta)
            + theoretical_correlation(0.0, -theta)
            + theoretical_correlation(45.0, theta)
            - theoretical_correlation(45.0, -theta)
        )
        theory_s.append(abs(value))

    theta_simulation = np.linspace(0.0, 22.5, 10)
    simulation_s = []
    scan_shots = max(500, min(int(shots), 10_000))

    for index, theta in enumerate(theta_simulation):
        result = simulate_chsh(
            a=0.0,
            a_prime=45.0,
            b=float(theta),
            b_prime=float(-theta),
            shots=scan_shots,
            seed=seed + 100 + index,
        )
        simulation_s.append(abs(result["simulated_s"]))

    return (
        theta_theory,
        theory_s,
        theta_simulation,
        simulation_s,
        scan_shots,
    )


def build_convergence_data(
    max_shots: int,
    a: float,
    a_prime: float,
    b: float,
    b_prime: float,
    seed: int,
):
    """计算不同Shots下的CHSH蒙特卡洛结果。"""
    shot_values = np.unique(
        np.geomspace(
            100,
            max(100, int(max_shots)),
            num=11,
        ).astype(int)
    )

    simulated_values = []

    for index, current_shots in enumerate(shot_values):
        result = simulate_chsh(
            a=a,
            a_prime=a_prime,
            b=b,
            b_prime=b_prime,
            shots=int(current_shots),
            seed=seed + 200 + index,
        )
        simulated_values.append(abs(result["simulated_s"]))

    return shot_values, simulated_values


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1350px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CHSH不等式蒙特卡洛仿真")
st.write(
    "使用Qiskit量子电路制备Bell态，并在不同测量基下进行重复测量，"
    "计算关联期望值与CHSH参数。"
)

with st.sidebar:
    st.header("仿真参数")

    shots = st.slider(
        "每组测量次数",
        min_value=100,
        max_value=50_000,
        value=10_000,
        step=100,
    )

    seed = st.number_input(
        "随机种子",
        min_value=0,
        max_value=1_000_000,
        value=42,
        step=1,
    )

    st.divider()
    st.subheader("Alice测量角度")

    a = st.slider(
        "a（°）",
        min_value=-90.0,
        max_value=90.0,
        value=0.0,
        step=0.5,
    )

    a_prime = st.slider(
        "a'（°）",
        min_value=-90.0,
        max_value=90.0,
        value=45.0,
        step=0.5,
    )

    st.subheader("Bob测量角度")

    b = st.slider(
        "b（°）",
        min_value=-90.0,
        max_value=90.0,
        value=22.5,
        step=0.5,
    )

    b_prime = st.slider(
        "b'（°）",
        min_value=-90.0,
        max_value=90.0,
        value=-22.5,
        step=0.5,
    )


# -------------------------------------------------------------------
# 1. Bell态制备
# -------------------------------------------------------------------

st.subheader("1. Bell态制备")

step_col1, arrow_col1, step_col2, arrow_col2, step_col3 = st.columns(
    [1, 0.12, 1, 0.12, 1]
)

with step_col1:
    with st.container(border=True):
        st.markdown("#### 步骤0：初始态")
        st.write("两个量子比特均初始化为0。")
        st.latex(r"|\psi_0\rangle=|00\rangle")

with arrow_col1:
    st.markdown(
        "<div style='text-align:center;font-size:2rem;padding-top:3.5rem;'>→</div>",
        unsafe_allow_html=True,
    )

with step_col2:
    with st.container(border=True):
        st.markdown("#### 步骤1：Hadamard门")
        st.write("对$q_0$施加H门，使其进入叠加态。")
        st.latex(
            r"|\psi_1\rangle="
            r"\frac{|00\rangle+|01\rangle}{\sqrt2}"
        )
        

with arrow_col2:
    st.markdown(
        "<div style='text-align:center;font-size:2rem;padding-top:3.5rem;'>→</div>",
        unsafe_allow_html=True,
    )

with step_col3:
    with st.container(border=True):
        st.markdown("#### 步骤2：CNOT门")
        st.write("$q_0$为控制位，$q_1$为目标位。")
        st.latex(
            r"|\psi_2\rangle="
            r"\frac{|00\rangle+|11\rangle}{\sqrt2}"
        )
        


details = bell_state_details()

circuit_col, state_col = st.columns([0.9, 1.1])

with circuit_col:
    st.markdown("#### Qiskit量子电路")
    st.code(
        details["circuit_text"],
        language="text",
    )

    with st.expander("查看电路代码"):
        st.code(
            """circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
state = Statevector.from_instruction(circuit)""",
            language="python",
        )

with state_col:
    st.markdown("#### 模拟器输出状态")

    state_rows = []

    for basis, amplitude in details["amplitudes"].items():
        state_rows.append(
            {
                "基态": basis,
                "振幅实部": round(amplitude.real, 6),
                "振幅虚部": round(amplitude.imag, 6),
                "测量概率": round(
                    details["probabilities"][basis],
                    6,
                ),
            }
        )

    st.dataframe(
        state_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        f"Bell态保真度：**{details['fidelity']:.6f}**"
    )
   


# -------------------------------------------------------------------
# 2. CHSH当前参数仿真
# -------------------------------------------------------------------

results = simulate_chsh(
    a=a,
    a_prime=a_prime,
    b=b,
    b_prime=b_prime,
    shots=int(shots),
    seed=int(seed),
)

simulated_s = abs(results["simulated_s"])
theoretical_s = abs(results["theoretical_s"])

st.subheader("2. CHSH仿真结果")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric(
    "模拟 |S|",
    f"{simulated_s:.4f}",
)

metric_col2.metric(
    "理论 |S|",
    f"{theoretical_s:.4f}",
)

metric_col3.metric(
    "经典极限",
    "2.0000",
)

metric_col4.metric(
    "量子上限",
    f"{results['quantum_bound']:.4f}",
)

if simulated_s > 2.0:
    st.success(
        f"当前参数下得到 |S|={simulated_s:.4f}>2，"
        "观察到CHSH不等式违背。"
    )
else:
    st.info(
        f"当前参数下得到 |S|={simulated_s:.4f}≤2。"
    )


# -------------------------------------------------------------------
# 3. 四组关联期望值
# -------------------------------------------------------------------

st.subheader("3. 四组关联期望值")

correlation_rows = []
correlation_labels = []
simulated_correlations = []
theoretical_correlations = []

for name, pair_result in results["pair_results"].items():
    simulated_value = pair_result["simulated_correlation"]
    theoretical_value = pair_result["theoretical_correlation"]

    correlation_rows.append(
        {
            "测量组合": name,
            "模拟值": round(simulated_value, 4),
            "理论值": round(theoretical_value, 4),
            "绝对误差": round(
                abs(simulated_value - theoretical_value),
                4,
            ),
        }
    )

    correlation_labels.append(name)
    simulated_correlations.append(simulated_value)
    theoretical_correlations.append(theoretical_value)

table_col, chart_col = st.columns([0.95, 1.25])

with table_col:
    st.dataframe(
        correlation_rows,
        use_container_width=True,
        hide_index=True,
    )



# -------------------------------------------------------------------
# 4. 单次测量结果统计
# -------------------------------------------------------------------

st.subheader("4. 单组测量结果统计")

selected_pair_name = st.selectbox(
    "选择一组测量角度",
    options=list(results["pair_results"].keys()),
)

selected_pair = results["pair_results"][
    selected_pair_name
]

outcome_order = [
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]

outcome_labels = [
    "(+1,+1)",
    "(+1,-1)",
    "(-1,+1)",
    "(-1,-1)",
]

outcome_counts = [
    selected_pair["counts"][outcome]
    for outcome in outcome_order
]

count_col, explanation_col = st.columns([1.25, 0.75])

with count_col:
    fig_counts, ax_counts = plt.subplots(
        figsize=(7.5, 4.3)
    )

    ax_counts.bar(
        outcome_labels,
        outcome_counts,
    )
    ax_counts.set_xlabel(
        plot_text(
            "联合测量结果",
            "Joint outcome",
        )
    )
    ax_counts.set_ylabel(
        plot_text(
            "出现次数",
            "Count",
        )
    )
    ax_counts.set_title(
        plot_text(
            f"{selected_pair_name}测量结果统计",
            f"Outcome counts: {selected_pair_name}",
        )
    )
    fig_counts.tight_layout()

    st.pyplot(fig_counts)
    plt.close(fig_counts)

with explanation_col:
    with st.container(border=True):
        st.markdown("#### 当前组合")
        st.write(
            f"Alice角度：**{selected_pair['alice_angle']:.1f}°**"
        )
        st.write(
            f"Bob角度：**{selected_pair['bob_angle']:.1f}°**"
        )
        st.write(
            f"模拟关联值：**{selected_pair['simulated_correlation']:.4f}**"
        )
        st.write(
            f"理论关联值：**{selected_pair['theoretical_correlation']:.4f}**"
        )
        st.write(
            f"总测量次数：**{selected_pair['shots']}**"
        )


# -------------------------------------------------------------------
# 5. 测量角扫描
# -------------------------------------------------------------------

st.subheader("5. 从经典极限到量子上限")

st.write(
    "固定$a=0°$、$a'=45°$，令$b=θ$、$b'=-θ$。"
    "当$θ$由0°增加到22.5°时，理论$|S|$由2增加到$2\\sqrt2$。"
)

(
    theta_theory,
    theory_scan_s,
    theta_simulation,
    simulation_scan_s,
    scan_shots,
) = build_angle_scan_data(
    int(shots),
    int(seed),
)

fig_angle, ax_angle = plt.subplots(
    figsize=(9.5, 4.8)
)

ax_angle.plot(
    theta_theory,
    theory_scan_s,
    linewidth=2.0,
    label=plot_text(
        "理论曲线",
        "Theory",
    ),
)

ax_angle.scatter(
    theta_simulation,
    simulation_scan_s,
    label=plot_text(
        f"蒙特卡洛结果（{scan_shots}次/组）",
        f"Monte Carlo ({scan_shots} shots)",
    ),
)

ax_angle.axhline(
    2.0,
    linestyle="--",
    label=plot_text(
        "经典极限2",
        "Classical bound 2",
    ),
)

ax_angle.axhline(
    2.0 * np.sqrt(2.0),
    linestyle=":",
    label=plot_text(
        "量子上限2√2",
        "Quantum bound 2sqrt(2)",
    ),
)

ax_angle.set_xlim(0.0, 22.5)
ax_angle.set_xlabel(
    plot_text(
        "测量角θ（°）",
        "Measurement angle theta (deg)",
    )
)
ax_angle.set_ylabel("|S|")
ax_angle.set_title(
    plot_text(
        "CHSH参数随测量角变化",
        "CHSH parameter versus angle",
    )
)
ax_angle.legend()
ax_angle.grid(alpha=0.25)
fig_angle.tight_layout()

st.pyplot(fig_angle)
plt.close(fig_angle)


# -------------------------------------------------------------------
# 6. Shots收敛
# -------------------------------------------------------------------

st.subheader("6. CHSH参数随测量次数的收敛")

st.write(
    "测量次数较少时，蒙特卡洛结果存在明显波动；"
    "增加Shots后，结果通常会逐渐稳定在当前角度对应的理论值附近。"
)

convergence_shots, convergence_values = build_convergence_data(
    max_shots=int(shots),
    a=a,
    a_prime=a_prime,
    b=b,
    b_prime=b_prime,
    seed=int(seed),
)

fig_convergence, ax_convergence = plt.subplots(
    figsize=(9.5, 4.8)
)

ax_convergence.plot(
    convergence_shots,
    convergence_values,
    marker="o",
    label=plot_text(
        "模拟 |S|",
        "Simulated |S|",
    ),
)

ax_convergence.axhline(
    2.0,
    linestyle="--",
    label=plot_text(
        "经典极限",
        "Classical bound",
    ),
)

ax_convergence.axhline(
    theoretical_s,
    linestyle=":",
    label=plot_text(
        "当前角度理论值",
        "Theory for current angles",
    ),
)

ax_convergence.set_xscale("log")
ax_convergence.set_xlabel(
    plot_text(
        "每组测量次数",
        "Shots per setting",
    )
)
ax_convergence.set_ylabel("|S|")
ax_convergence.set_title(
    plot_text(
        "CHSH参数的统计收敛",
        "Statistical convergence of CHSH",
    )
)
ax_convergence.legend()
ax_convergence.grid(alpha=0.25)
fig_convergence.tight_layout()

st.pyplot(fig_convergence)
plt.close(fig_convergence)


with st.expander("查看理论公式"):
    st.markdown(
        r"""
Bell态为

$$
|\Phi^+\rangle=
\frac{|00\rangle+|11\rangle}{\sqrt2}
$$

关联期望值为

$$
E(\alpha,\beta)=
\frac{1}{N}\sum_{i=1}^{N}A_iB_i
$$

本程序采用

$$
S=
E(a,b)+E(a,b')+E(a',b)-E(a',b')
$$

经典局域隐变量理论要求

$$
|S|\leq2
$$

在最优测量角度下，量子力学允许达到

$$
|S|=2\sqrt2
$$
"""
    )
