# 随机工况实验 I：开发/验证集结果分析

## 1. 随机工况设置

论文给出 F1/X1/T1/T200 方差为 [2,1,8,5]，本实验使用平方根作为标准差。
Gaussian iid、逐步独立采样、1 s 采样、1000 steps、初态 B 属于
implementation choice for reproduction；相关精确实现 not specified in the paper。
当前仅使用反复参与 authority/reward 开发的 420000–420009 validation seeds。
两策略训练初始化均为 seed42，并非十个训练种子。430000–430049 仅预留，
审计范围内未发现运行证据。本轮未运行它们，也未训练。均值/95% CI 是给定
已选择策略下的配对验证 realization 描述统计，不是无偏独立 test 推断。

## 2. Baseline 结果

zero-residual H∞/RPI/QP 已具有较强经验扰动抑制。X2 均值
25.384924，平均轨迹内标准差 0.043027；
跨 realization 最小安全余量 0.212371。
所检验轨迹的物理状态/输入违约均为零。

## 3. Residual SAC 结果

暂推荐 A 作为验证集主展示候选，B 作为补充 Pareto 点。A 的 X2 退化比 B 小，
但经济收益也较小，而且 F200 TV 的代价较大；没有任何一方全面占优。
A/B 同时改变 authority 和 reward，不能归因为纯 α 消融。
α 在此指 stochastic residual authority scale，不是 SAC entropy 系数。

## 4. 内部严格配对定量分析

|Method|J_econ|Gain %|Worst gain %|X2 mean|Mean within-seed std|Global X2 range|X2 IAE ratio (B)|P2 IAE ratio (B)|P100 TV ratio|F200 TV ratio|
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
|baseline|6217912.952|0.000000|0.000000|25.384924|0.043027|25.212371–25.539580|1.000000|1.000000|1.000000|1.000000|
|A_alpha010|6216471.785|0.023161|0.009478|25.369345|0.046562|25.178676–25.530982|1.181462|0.989078|0.825742|1.315386|
|B_alpha020_rewardcal|6215268.649|0.042468|0.017342|25.350740|0.049341|25.160545–25.556139|1.538162|0.938511|1.048697|0.985507|

A mean gain 0.023161% (descriptive paired t95 interval 0.017547–0.028775%). X2 B-reference IAE +18.15%, ISE +39.01%, P100 TV -17.43%, F200 TV +31.54%.

两者共用初态、扰动 path、seed、horizon、nonlinear plant 和经济成本。
J_econ 不包含 shaping penalty；收益取每个 seed 的相对收益后再平均，
不采用 aggregate 均值的比值。主 IAE/ISE/peak 参考固定 B=[25.39,50.125]，
CSV 同时保留 *_nominal=[25,49.743] 的结果，避免事后切换 reference。
靠近 X2=25 可以减小 nominal error，却增加 B-reference error，二者并不矛盾。
性能积分使用 1000 个步前状态；安全审计含 terminal，共 1001 个状态。
输入 TV/RMS 使用 999 个相邻已执行输入差分。std 为各 seed 轨迹内总体 std 的
均值，正文极值为所有轨迹极值；CSV mean/min/max 区分两种聚合。
G_emp 是相对 B 的状态 L2 偏差除以四外扰按标准差归一化后的联合 L2 范数，
仅为 empirical disturbance amplification，并非 H∞ norm。
SAC 使用一部分抑扰余量换取小幅经济收益，不能把其解释为全面支配 baseline，
也不能忽略 A 的 F200 TV 增加。

## 5. ECC2019 定性文献比较

原文 PDF 第5页/印刷2262页报告质量约束仅罕见且小幅违反；Fig.2 位于
PDF第6页/印刷2263页。本文所显示轨迹没有 X2<25；相近宽纵轴下随机带较窄。
由于原作者 raw data、exact realization 和部分实现细节缺失，这只是不同模拟
下的定性视觉及报告行为比较，不构成受控数值优势证明。禁止计算本文相对
ECC2019 的 std/IAE/违约率/经济收益百分比。原文14%/12%的 baseline 是其
naive/nominal-economic NMPC，不是本文 robust baseline。
引用图注及正文措辞见 ecc2019_literature_comparison.md；没有重绘原图数据。

## 6. 安全结论边界

Gaussian 支撑无界，A 的 W 超界均值为
98.28%。因此这里只能称
empirical stochastic evaluation，不能宣称 formal Gaussian robust guarantee。
观测到的物理/QP/Omega/robust-region 异常为零不意味着连续域解析认证。
Omega/Gm/Bj 与 finite-jump supervisor 保留用于实验 II/III，不将其条件证书
移植为实验 I 的 Gaussian 安全证明。

## 7. 小结与论文条件

内部经济小幅改善、经验安全保持与明确性能代价同时成立。当前材料足以形成
诚实的论文开发/验证结果章节，但尚不能冒充独立最终测试。真正必要的下一项
是先锁定候选及分析方案，再进行预留独立种子的配对评价；本轮未自动启动。
若需跨训练种子可复现结论，单 seed42 不足。停止 ECC 数值拟合、盲目 reward/
authority/critic 调参；有限跳变实验独立保留。本轮无训练、无 commit、无 push。
