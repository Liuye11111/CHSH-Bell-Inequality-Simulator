# CHSH Bell Inequality Simulator

## 项目简介

本项目是一个基于Python和Streamlit开发的交互式量子仿真平台。

项目通过蒙特卡洛方法模拟Bell纠缠态在不同测量基下的大量重复测量，计算量子关联期望值与CHSH参数，直观展示量子力学对经典局域隐变量理论的违背。

## 项目目标

- 模拟双比特Bell纠缠态
- 设置Alice和Bob的测量角度
- 进行多次蒙特卡洛随机测量
- 统计不同测量结果出现的次数
- 计算四组关联期望值
- 计算CHSH参数
- 对比经典极限2与量子理论上限2√2
- 通过交互式网页展示仿真结果

## 技术路线

- Python
- NumPy
- Streamlit
- Matplotlib或Plotly
- Monte Carlo Simulation

## 理论背景

对于Bell态：

$$
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt{2}}
$$

Alice和Bob分别选择不同测量方向，并将测量结果表示为$+1$或$-1$。

两者的关联期望值为：

$$
E(\alpha,\beta)=\frac{1}{N}\sum_{i=1}^{N}A_iB_i
$$

CHSH参数由四组关联期望值组合得到：

$$
S=E(a,b)-E(a,b')+E(a',b)+E(a',b')
$$

局域隐变量理论要求：

$$
|S|\leq2
$$

而量子力学允许达到：

$$
|S|_{\max}=2\sqrt{2}\approx2.828
$$

## 项目状态

项目正在开发中，后续将逐步加入核心算法、交互界面、统计图表与自动化测试。
