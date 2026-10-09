# 经济归因与有限搜索：详细结果

状态：COMPLETE。未完成的 witness 不计为失败，也不用于 headroom 结论。

## 真实经济成本

运行时代码：`model.py:108 / EvaporatorModel.economic_cost`。
`ell = 10.09*F2 + 10.09*F3 + 600*F100 + 0.6*F200`。
F3=50固定；P100通过T100/Q100/F100及F4/F2间接影响代价。600乘的是F100，不能标成600*P100。
代码未给成本系数明确货币/时间单位，本报告使用模型成本单位。J是原有1000步stage-cost之和。
成本在nonlinear next state、最终实际输入、当前扰动上计算。与SAC replay reward严格分开。
旧metrics helper的零权重return并非锁定训练reward，已从主经济CSV移除；receipt中的对应字段仅留作诊断来源。

## 各成本分量贡献

下表为10条相同DEV路径平均。正数表示baseline-SAC，即节省；负数表示额外成本。

|geometry|F2|F3|F100蒸汽|F200冷却|总节省|收益%|
|---|---:|---:|---:|---:|---:|---:|
|baseline|-81.779|0.000|5473.914|-13.047|5379.088|0.08650902|
|C2_8|-86.412|0.000|5781.975|-5.010|5690.554|0.09151897|
|A3|-88.361|0.000|5969.150|-372.165|5508.623|0.08859226|
|C2|-79.360|0.000|5369.525|-388.203|4901.961|0.07883575|
|C2_3|-73.459|0.000|5007.926|-603.013|4331.453|0.06965937|

Production的主要收益来自蒸汽成本下降，F2与F200成本上涨抵消一部分。F2是原经济代价项，不自行解释为产品销售收益。

## geometry与SAC分开

绝对量：DeltaJ_total = DeltaJ_geometry + DeltaJ_SAC_given_geometry。三种百分比有不同分母，不能直接相加。

|geometry|baseline额外成本|SAC相对自身baseline节省|最终对production收益%|自身baseline收益%|瞬时经济占优比例|
|---|---:|---:|---:|---:|---:|
|baseline|0.000|5379.088|0.08650902|0.08650902|59.04%|
|C2_8|0.000|5690.554|0.09151897|0.09151897|58.91%|
|A3|505.719|6014.342|0.08859226|0.09671759|58.06%|
|C2|505.719|5407.680|0.07883575|0.08696187|58.14%|
|C2_3|758.922|5090.375|0.06965937|0.08185469|57.69%|

C2/C2_3的自身zero-residual baseline已变贵。必须先扣除这部分，再比较actor效果。

|geometry|baseline F2额外成本|baseline蒸汽额外成本|baseline冷却额外成本|
|---|---:|---:|---:|
|baseline|0.000|0.000|0.000|
|C2_8|0.000|0.000|0.000|
|A3|-2.635|116.409|391.945|
|C2|-2.635|116.409|391.945|
|C2_3|-3.952|174.593|588.281|

## 精确代数 input/state channel identity

固定当步disturbance时，当前模型的cost对post-state X2/P2和实际P100/F200为仿射函数，逐步恒等式已核对。
这是代数分量拆解，不是完整动态因果或离线重新优化。P100直接项节省与state项损失不能混称净P100收益。

|geometry|post-X2贡献|post-P2贡献|direct-P100贡献|direct-F200贡献|
|---|---:|---:|---:|---:|
|baseline|-3071.437|-1766.756|10230.328|-13.047|
|C2_8|-3239.546|-1845.106|10780.215|-5.010|
|A3|-3343.245|-2622.291|11846.325|-372.165|
|C2|-3006.300|-2465.354|10761.818|-388.203|
|C2_3|-2797.900|-2775.351|10507.718|-603.013|

完整channel数据见economic_channel_decomposition.csv。

## actor方向与相关性

local residual相对于同一个SAC当前state/nominal context的zero-action输入；与配对闭环baseline输入差是两种量。

|channel|positive|negative|zero|mean|median|p5|p25|p75|p95|Pearson(g)|Spearman(g)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|P100|0.06%|99.94%|0.00%|-3.58893|-3.17431|-5.40925|-4.47180|-3.05155|-2.34940|-0.36285|-0.36073|
|F200|26.25%|73.75%|0.00%|-0.16282|-0.12996|-1.34641|-0.24324|0.04405|0.79745|-0.33428|-0.24091|

|经济分组|channel|local residual mean|配对闭环输入差mean|negative比例|
|---|---|---:|---:|---:|
|economic_positive|P100|-3.76898|-1.10575|100.00%|
|economic_negative|P100|-3.32939|0.54483|99.85%|
|economic_positive|F200|-0.27046|-0.16458|80.95%|
|economic_negative|F200|-0.00767|0.29032|63.38%|

correlation不等于causation。状态条件分箱、绝对residual相关性与authority utilization均在actor_behavior_summary.csv。

## 新geometry动作范围是否被使用

在每条新geometry轨迹的同一state/z/raw actor下查询原online mapping，并检查当前输入是否属于原alpha=0.1收缩动作集合。
这是局部映射干预，不传播干预后的plant，也不将local差值当成闭环经济因果效应。

|geometry|原动作集合外步数比例|同raw映射P100绝对变化mean|同raw映射F200绝对变化mean|原QP不可行|
|---|---:|---:|---:|---:|
|baseline|0.00%|0.00000|0.00000|0|
|C2_8|12.31%|0.18000|0.03504|0|
|A3|88.77%|0.36359|0.56530|0|
|C2|41.44%|0.00271|0.63095|0|
|C2_3|44.52%|0.18210|1.03709|0|

C2并非完全没有使用新增geometry：映射后的F200及可执行动作范围确实改变。不能仅归因于positive F200 authority；baseline投影、动作中心和随后state轨迹也改变。
C2_8的baseline成本与production一致，新增P100范围及映射被实际使用；其净增益不是baseline减弱制造，也不是CSV回放dtype的数值差异。

## 五个P100-only预注册候选

|factor|certificate|positive P100 authority|negative P100 authority|F200 positive authority|完成路径|收益mean%|收益worst%|win|X2最低margin|P100 TV|F200 TV|P100 RMS|F200 RMS|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1.0|CERTIFIED_PASS|3.10745|-3.25609|0.40482|10|0.08650902|0.07685794|59.04%|0.13230|3440.10|683.65|4.34122|0.90381|
|1.025|CERTIFIED_PASS|3.20443|-3.35307|0.40482|10|0.08905236|0.07846083|58.96%|0.12999|3439.55|684.86|4.34063|0.90632|
|1.05|CERTIFIED_PASS|3.30142|-3.45006|0.40482|10|0.09151897|0.08052576|58.91%|0.12762|3438.82|687.22|4.33947|0.91153|
|1.075|CERTIFIED_PASS|3.39840|-3.54704|0.40482|10|0.09396843|0.08266142|58.96%|0.12523|3437.52|689.50|4.33784|0.91610|
|1.1|CERTIFIED_PASS|3.49539|-3.64403|0.40482|10|0.09614896|0.08470604|58.88%|0.12446|3436.14|692.77|4.33665|0.92142|

证书JSON保存全部gate、10256个非线性bounded-W样本、完整QP probes。有限样本覆盖不声称连续域解析证明。
Gaussian DEV闭环的W超界单独记录；这些轨迹仅为empirical safety，不重新定义certified W。

## witness与下一阶段

{
  "status": "COMPLETE",
  "preregistered_budget": 512,
  "completed_candidates": 512,
  "safe_candidates": 512,
  "positive_mean_candidates": 39,
  "all_validation_positive_candidates": 19,
  "best_safe": {
    "candidate": 167,
    "safety_passed": true,
    "failure": null,
    "completed_paths": 10,
    "search": "finite_state_dependent_witness_search",
    "mean_improvement_pct": 0.0638590710789818,
    "worst_path_improvement_pct": 0.03849600781995038,
    "all_validation_positive": true,
    "minimum_X2_margin": 0.1538452735955147,
    "control_quality_pass": true,
    "economic_win_fraction": 0.5693,
    "J_econ": 6213944.696697863,
    "P100_TV": 3499.630022796736,
    "F200_TV": 634.0582721359737,
    "P100_RMS_du": 4.482360729771652,
    "F200_RMS_du": 0.8386797413753415,
    "P100_TV_ratio": 0.9120035643753281,
    "F200_TV_ratio": 0.7696667676133252,
    "P100_RMS_du_ratio": 0.9299998107037443,
    "F200_RMS_du_ratio": 0.8101246966943039,
    "X2_std": 0.04762153146571726,
    "centered_X2_MAE": 0.03790348648738566,
    "X2_IAE_ratio": 1.7013192917700273,
    "P2_IAE_ratio": 1.0422086619717674,
    "X2_ISE_ratio": 2.6655761849907633,
    "P2_ISE_ratio": 1.1487263338521727,
    "W_exceedance_rate": 0.9827999999999999,
    "P100_sign_change_ratio": 1.0215857958874222,
    "F200_sign_change_ratio": 1.0454732362910735,
    "physical_state_violation_steps": 0,
    "physical_input_violation_count": 0,
    "QP_infeasible_count": 0,
    "Omega_exit_count": 0,
    "robust_region_violation_count": 0
  },
  "best_control_quality": {
    "candidate": 167,
    "safety_passed": true,
    "failure": null,
    "completed_paths": 10,
    "search": "finite_state_dependent_witness_search",
    "mean_improvement_pct": 0.0638590710789818,
    "worst_path_improvement_pct": 0.03849600781995038,
    "all_validation_positive": true,
    "minimum_X2_margin": 0.1538452735955147,
    "control_quality_pass": true,
    "economic_win_fraction": 0.5693,
    "J_econ": 6213944.696697863,
    "P100_TV": 3499.630022796736,
    "F200_TV": 634.0582721359737,
    "P100_RMS_du": 4.482360729771652,
    "F200_RMS_du": 0.8386797413753415,
    "P100_TV_ratio": 0.9120035643753281,
    "F200_TV_ratio": 0.7696667676133252,
    "P100_RMS_du_ratio": 0.9299998107037443,
    "F200_RMS_du_ratio": 0.8101246966943039,
    "X2_std": 0.04762153146571726,
    "centered_X2_MAE": 0.03790348648738566,
    "X2_IAE_ratio": 1.7013192917700273,
    "P2_IAE_ratio": 1.0422086619717674,
    "X2_ISE_ratio": 2.6655761849907633,
    "P2_ISE_ratio": 1.1487263338521727,
    "W_exceedance_rate": 0.9827999999999999,
    "P100_sign_change_ratio": 1.0215857958874222,
    "F200_sign_change_ratio": 1.0454732362910735,
    "physical_state_violation_steps": 0,
    "physical_input_violation_count": 0,
    "QP_infeasible_count": 0,
    "Omega_exit_count": 0,
    "robust_region_violation_count": 0
  },
  "certification_scope": "Gaussian empirical only",
  "claim": "best found in finite state-only family, NOT global optimum or oracle upper bound",
  "frozen_SAC_reference_gain_pct": 0.08650902042768663
}

完整512个候选完成前，不能判断learning headroom、intrinsic economic upper bound或geometry方向最终优劣。
有限state-only family的best-found也不是global optimum或oracle upper bound。
未完成witness时不画虚构Pareto，也不填safe/positive数量为0。

## 固定与测试

production hashes unchanged: True
未训练SAC；未使用430000~430049；未修改生产U_R/alpha=0.1/Strong reference/K/W/Z/S/Omega/QP/actor/reward。
未commit、未push；旧Conservative partial结果保留。
测试结果见tests_receipt.json。可恢复执行命令见econ_attribution/README.md。