# Safe-SAC economic recovery experiment v1

这是独立开发实验，主对照是 `primary_strong_baseline` 与其 Safe-SAC。辅助对照是按 baseline-only 规则冻结的 `sensitivity_conservative_baseline` 与其 Safe-SAC。绝不替换主 baseline，不改经济成本，不挑弱 K。所有新输出在 `evaporation_safe_sac/economic_recovery_v1/`，旧 v1/v2、最终 Candidate A 保持只读。

## 执行顺序（本机 PowerShell）

```powershell
Set-Location "C:\Users\cushy\PycharmProjects"
$recoveryPython = "C:\Users\cushy\PycharmProjects\evaporation\.venv-cuda\Scripts\python.exe"
& $recoveryPython -m evaporation.zanon2019_economic_recovery prepare
& $recoveryPython -m evaporation.zanon2019_economic_recovery tests
& $recoveryPython -m evaporation.zanon2019_economic_recovery oracle --pair strong
# 已完成的 v2 seed42 自动作为只读历史源，不重新训练/覆盖。
& $recoveryPython -m evaporation.zanon2019_economic_recovery train --pair strong --seed 2027 --device cuda
& $recoveryPython -m evaporation.zanon2019_economic_recovery train --pair strong --seed 314159 --device cuda
& $recoveryPython -m evaporation.zanon2019_economic_recovery report
# 只有三个 Strong seed 均完成并通过 continuation gate 才允许以下阶段。
& $recoveryPython -m evaporation.zanon2019_economic_recovery sweep
& $recoveryPython -m evaporation.zanon2019_economic_recovery oracle --pair conservative
& $recoveryPython -m evaporation.zanon2019_economic_recovery audit
& $recoveryPython -m evaporation.zanon2019_economic_recovery train --pair conservative --seed 42 --device cuda
& $recoveryPython -m evaporation.zanon2019_economic_recovery train --pair conservative --seed 2027 --device cuda
& $recoveryPython -m evaporation.zanon2019_economic_recovery train --pair conservative --seed 314159 --device cuda
& $recoveryPython -m evaporation.zanon2019_economic_recovery freeze-figure --pair strong --device cuda
& $recoveryPython -m evaporation.zanon2019_economic_recovery freeze-figure --pair conservative --device cuda
& $recoveryPython -m evaporation.zanon2019_economic_recovery report
```

`prepare` 只执行一次；immutable protocol/selection/calibration 不允许覆盖。`tests` 可重跑。`report` 可随时刷新已有开发数据；缺失 seed/图会列为 pending。`status` 显示阶段状态。每次 oracle 512 个候选、每个候选十条 1000-step 路径，可能较耗时，不是训练，也不是在线 MPC。oracle 已完成证据不能覆盖；中止后保留目录，不静默重启。

本轮没有 final-independent-test 入口。仅开发扰动种子 420000..420009；430000..430049 禁用。训练种子 42/2027/314159、100×1000、每五回合固定评估。训练扰动种子仍 `training_seed*1000000+episode`，随机 action RNG 仍 `path_seed+77`，SAC 用同一 `set_seed`。所有模型、网络、学习率、23D observation、buffer、batch、alpha=.10 authority、interior-anchor mapping 与 v2 一致。

## Conservative reference 的边界

预注册 grid 是 X2=B+[0,.05,.10,.15,.20]，P2 固定 B 值。选择第一个正增量、全部十条 paired baseline minimum X2 margin 增加超过 1e-4 且全部 gate 通过的候选。1e-4 只是预先定义的可观测数值 guard，不是论文或工程安全裕量标准。成本只记录，不用于排序；无候选通过则停止，不扩大 grid。

精确 nonlinear `steady_input` 求得新 u_ref，检验 derivative≈0。保留 B 的归一化原点和 A/B Jacobian、K/W/Z、tightening、物理 S。仅仿射预测器 equilibrium intercept、nominal policy offset、z_ref/v_ref 和 reference reward target 自洽更新。Omega error vertices 平移以保留**同一个物理集合**。这是 `implementation choice for reproduction`，不是在新点 Jacobian relinearization。所有变化和不变数组在 reference_checks 列明。如果固定物理 S/Omega、H∞、RPI、verification QP、5% authority、nominal W finite-probe coverage 失败则拒绝。不重新优化 gain 或集合。有限 probes 不是 nonlinear 连续域证明。

Strong 使用已冻结 λ=17.613805207960148 和历史 69 个 paired excitation rollouts。Conservative 对**相同 raw actor-action excitation profiles**、相同 disturbance，重新通过本方 controller/plant 生成 paired rollout，用相同 `lambda=.05*mean(abs(r_econ))/mean(excess_applied_move)` 规则计算，冻结后不手调。excitation 不直接复用 physical q。这种跨 reference 的 deterministic calibration 是实现选择，非论文规定。

训练奖励仍是 `(ell_own_base-ell_policy)/200 - lambda_smooth*sum(max(0,(du_policy/100)^2-(du_own_base/100)^2))`，加原 v2 safety-only dual 项。actual-input move 初始 previous input 是本方 physical(v_ref)。replay 保存七个独立 components，minibatch 使用当前 safety multipliers。安全异常立即中止，W 超界单列，不伪称 Gaussian formal certificate。

## 指标与证据

锁定 instantaneous `100*(ell_base-ell_SAC)/ell_base`（positive=better）；raw difference `ell_SAC-ell_base`；strict win fraction；rolling20 为 trailing **完整**二十步（前十九步为空）；cumulative ratio of cost sums，终值等于 paired improvement，不能用 mean instantaneous percent 代替。baseline cost<=0 时报错，不填零。

Conservatism penalty=JC-JS，recovered=JC-JCSAC，recovery ratio=100*recovered/penalty 仅 penalty>0；可以负或>100%，不截断。跨 reference initial state 不同，仅是敏感性机制比较，不评四方法总冠军。每条 validation realization 单列；跨 training seed Student-t CI，只有一训练 seed 不报告 CI。

更密的 oracle 使用完整 actor->mapping->QP->nonlinear plant，constant grid、10s/5s分段、Pareto mutation；包括已学 Strong policy，以免新搜索比已知可行 policy 更差。所有 safe witness、失败 evidence、Pareto front 保存。文件按请求命名 `oracle_safe_economic_upper_bound_*.json`，**内容明确 best_found_not_certified_upper_bound**。有限搜索不能证明经济上界或“无 headroom”；utilization 只有相同十条路径/初值/协议才计算，可能>100%，不截断。不得拿旧三条路径 oracle 与十条 fixed validation 比 utilization。

主学习曲线是固定 paired evaluation，有每个训练 seed 原始点。raw training return 仅补充诊断。Fig.2-style 只用预声明 development episode100 actor：训练后复制冻结权重→重新 paired rollout 420000→绘图，检查权重不变。不按图像好看挑 disturbance；本协议不锁定 economic_first_final_v2 最终策略。所有六次训练完成后，另行预注册最终 selection 和 control sanity thresholds，再考虑独立测试。
