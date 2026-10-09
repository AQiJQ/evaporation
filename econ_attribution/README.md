# Economic attribution / P100 sensitivity / finite state-only witness

独立子包，不新增顶层Python文件，不修改正在运行的300回合训练的源码冻结清单。
生产U_R、K/W/Z/S/Omega/QP、alpha=0.1、Strong reference、reward和冻结actor不变。

有效输出目录：
`evaporation_safe_sac/economic_recovery_v1/economic_attribution_v1_replayfix/`。
初始`economic_attribution_v1/`保留诊断失败证据：CSV读取float64与冻结actor原float32
输出的回放dtype不一致。修复只恢复actor输出dtype，不放松任何QP/安全容差。

## 分阶段本机执行（不训练）

```powershell
Set-Location "C:\Users\cushy\PycharmProjects"
$auditPython = ".\evaporation\.venv-cuda\Scripts\python.exe"
& $auditPython -m evaporation.econ_attribution.study audit --workers 2
& $auditPython -m evaporation.econ_attribution.extras.channels
& $auditPython -m evaporation.econ_attribution.extras.authority_usage --workers 2
& $auditPython -m evaporation.econ_attribution.study p100 --workers 2
& $auditPython -m evaporation.econ_attribution.study witness --workers 2
& $auditPython -m evaporation.econ_attribution.study report
& $auditPython -m evaporation.econ_attribution.extras.details
& $auditPython -m evaporation.econ_attribution.extras.verify_outputs
```

`prepare`已运行后不要重复调用。上述阶段可以恢复：已完成receipt保留，未完成path重算。
参数/源码/hash改变则拒绝恢复，不能追加候选或悄悄调参数。最晚以完整512候选结果
生成最终decision；部分结果只标记PARTIAL，不能宣称经济上限或已证明headroom不足。
若中断使用Ctrl+C，保存已完成候选，再用原命令恢复。不恢复旧Conservative Oracle。

并行是CPU独立闭环，不是CUDA优化，Torch线程设置仅为计算资源控制。矩阵QP与
nonlinear plant主要在CPU，无需把这种搜索伪称GPU加速。Windows沙箱禁止进程
通信时可在正常本机终端运行，或使用`--workers 1`。不启动SAC、不接入在线控制器。

## 预注册

P100-only factor=[1.00,1.025,1.05,1.075,1.10]，F200窗口完全保持原生产值。
每候选重新执行既有finite bounded-W coverage、KZ tightening和所有gate；失败
不能因为Gaussian rollout没越界就称certified。仅CERTIFIED_PASS才跑冻结actor。

512个state-only affine actor candidates：零策略1个，预先固定constant controls 16个，
fixed-seed Latin-hypercube 495个。actor输出`clip(b+L*phi,-1,1)`，phi仅取现有
23D observation中的前2项×既有state_scale，无新传感器、clock或未来扰动信息。
参数、候选编号、预算、seed均在protocol.json冻结，不根据结果增补。
全部候选通过原interior-anchor/alpha/QP/nonlinear rollout；不直接优化q。

每候选最多10×1000步，首次安全异常终止该候选并保留evidence，不把partial cost
当健康的完整结果。输出Pareto，不新增综合加权评分。控制质量报告复用现有每路径
TV/RMS<=1.10数值带，另报IAE/ISE、X2波动、边界margin和sign changes。
该带是diagnostic，不改变safety constraints。仅development seeds420000..420009。

## 结果

stage cost采用真实runtime nonlinear post-state timing，拆成F2/F3/F100/F200四项。
绝对geometry+SAC effects严格相加；三个百分比使用不同denominator，不直接相加。
归因同时区分same-state local applied residual与closed-loop paired input difference。
额外channels输出验证固定disturbance下实际代价的精确代数恒等式，区分P100/F200
直接成本项变化与post-state X2/P2项变化，不把correlation声称为动态因果证明。

`extras.details`生成中文详细报告、P100汇总和字段语义说明。离线调用legacy
metrics helper时传入的零权重仅用于提取过程/控制指标，不是本轮reward改动。
因此helper产生的`total_training_reward`等字段从主经济CSV移除，不能将它们
画成当前锁定reward的学习曲线。per-path JSON保留原helper字段作为诊断来源，
其解释由`metric_field_semantics.json`明确限定。没有任何新的SAC return数据。

所有8个主CSV与8张主图由完整相应数据产生；未完成witness时不造图或填假0。
归档完整逐步cost components、cumulative components与actor action authority。
每个witness的完整1000步数组使用压缩NPZ，经济最佳/控制质量代表点另存CSV。
预留至少几GB空间。没有任何数据写回旧Strong/Conservative实验。

## 测试

```powershell
& $auditPython -m evaporation.econ_attribution.tests
```

全套包含现有300训练入口的测试。当前该旧测试的CUDA检查依赖默认输出目录为空；
如果300训练已开始，原测试会先被“目录非空”挡住。为避免修改正在冻结的顶层源码，
仅在测试进程内把默认路径隔离，既不改磁盘代码也不影响外部训练：

```powershell
& $auditPython -c 'import sys,uuid; from evaporation import zanon2019_seed42_300 as launcher; from evaporation.econ_attribution import tests; launcher.DEFAULT_OUTPUT=launcher.ROOT/("test_preflight_"+uuid.uuid4().hex); sys.argv=["tests","--all"]; tests.main()'
```

包含optional CasADi另环境检查与evaporation.test。测试receipt只写新诊断目录。
No SAC training, no final seeds, no production edits, no commit/push。
