# Paired Economic Safe-SAC v2 (Candidate C)

这是不同 reward formulation 的独立 development 实验。主 baseline 不削弱，旧
`paired_economic_v1/oracle10/` 的 145 candidates 和 v1 joint=0 永久保留。
所有新增数据仅写入 `evaporation_safe_sac/paired_economic_v2/`。

## 研究问题与冻结项

检查 Residual SAC 是否在保持观察到的物理安全的前提下，相比 paired zero-residual
H∞+RPI+QP baseline 改善经济性能，并避免增加剧烈实际控制活动。

B=[25.39,50.125]、H∞ K/W/Z/S/tightening、Omega、Gm/Bj、QP、interior-anchor
mapping、23D observation、nonlinear plant、经济成本、SAC 网络/LR/buffer/batch
均不修改。stochastic residual authority alpha=0.10 不变，不能与 SAC entropy alpha 混淆。
stochastic 工况没有 state jumps，因此保留原有不启用 Gm/Bj 的行为。

v1 joint=0 的直接原因是最坏 realization 的 X2 IAE/B 比值超过1.10。
该指标既包含相对 B 的经济运行点偏移，也包含随机波动，因此 v2 不再把
X2/P2 IAE/ISE 作为硬门槛或隐藏 reward penalty。但 reference **仍是 B**，不换成25。
完整报告 IAE_B/ISE_B、mean shift、std、centered MAE 与物理边界距离。
不从 ECC2019 Fig.2 估计 IAE/std 或阈值，不把 v2 witness 事后称为 v1 witness。

## 精确 reward 与量纲

对同初态、同一个完整 F1/X1/T1/T200 realization、同 timestep 的 paired rollout：

```text
econ[k] = (cost_baseline[k] - cost_SAC[k]) / 200
dP = (P100[k] - P100[k-1]) / 100
dF = (F200[k] - F200[k-1]) / 100
excessP = max(0, dP_SAC² - dP_baseline²)
excessF = max(0, dF_SAC² - dF_baseline²)
smooth_raw = excessP + excessF
reward = econ - lambda_smooth*smooth_raw - dot(current_safety_dual, safety_costs)
```

P100/F200 **等归一化权重**。100/100 来自冻结的 `cfg.input_scale`，与原 actual-input
move 及 normalized mapping 一致。不能用单条轨迹最大值重新缩放。
所有 move 均使用 mapping/QP 后 plant 实际看到的 physical input，不使用 raw actor。
first step 两条轨迹的 previous applied input 都为 `physical(design.v_ref)`，与原
`run_episode` reset 语义一致。不是从0起算，也不是把第一次 input difference 任意删除。

Replay 独立保存7列：economic_reward_raw、smoothness_penalty_raw，及
physical-state/input/QP/Omega/robust-region 的5个 safety event components。
旧固定 shaping scalar return 仅作 diagnostic，不能进入学习 reward。
sample 时重构固定 lambda_smooth 与当前 safety dual。v1 的 IAE、TV、boundary
performance dual 不继承。安全仍 immediate abort，所以成功轨迹 safety components
为0，不能声称这些乘子主动优化了恢复/平滑性。smoothness 不参与 dual update。

## 权重标定与锁定

先运行全部测试，再审计 development 归档，不重跑旧策略：30条锁定 Candidate A
轨迹、30条对应 training seeds 的 episode100 late trajectories、71/76/133 各3条 oracle。
每条1000步，69条轨迹共69000步等权汇总。paired baseline 为每个文件自身的baseline列。
文件路径、SHA256、角色、seed 与每条量级均记录，不挑选表现好的 calibration realization。

```text
lambda_smooth = 0.05 * mean(abs(econ)) / mean(smooth_raw)
regularizer_fraction = lambda_smooth * sum(smooth_raw) / sum(abs(econ))
```

这是 **implementation choice for reproduction**，不是论文规定的系数。
分母近零、非有限量或非零 move excitation 样本不足时停止。5%为唯一训练主值；
10%仅保存为离线尺度 sensitivity。后续不得修改锁定JSON或沿用同目录重新标定。
训练期间每回合校验配置hash。若5个连续 post-warmup evaluations 都非正经济收益
且 regularizer fraction>10%，中止并报告 `regularizer scale mismatch`，不现场调权重。
经济分母为0时 fraction=null（undefined），不伪装成0。所有 evaluation 使用相同定义。

### 本轮已完成的实测审计

112项测试（含独立ECC2019环境补跑3项CasADi检查）与原`evaporation.test`集成检查通过。
新增33项测试，未启动SAC训练。69000个development paired steps得到：

- mean abs paired economic reward：0.17730491252098654；
- median abs economic reward：0.14826537654100092；
- mean raw smoothness：0.0005033123462750052；
- median nonzero raw smoothness：0.00024904072763973597；
- p95 raw smoothness：0.0024030582006404847；
- locked lambda_smooth：17.613805207960148；
- mean weighted smoothness：0.008865245626049327，目标占比5%。

**注意净收益与绝对经济量级的区别。** 在固定该权重后，3条realization的平均
total reward：candidate71≈+0.294289，candidate76≈−0.466868，candidate133≈−1.189181。
76/133的经济收益仍正且TV仍优于baseline，但正负逐步经济项相消后，smooth penalty
已超过它们的净经济reward。只有71仍具有正的净objective witness。v2 joint criterion
不事后加入total reward硬门槛，三个v2 witness仍保留，同时完整公开这一reward trade-off。
这不能证明训练必然失败或成功；不改权重，只允许先在本机验证seed42。
训练若触发持续scale mismatch gate必须停止，不能中途改lambda继续跑。

## Oracle v2 重分类

145条候选从原各3条 per-realization metrics 重分类。每条 realization 要求：

- physical/input/QP/Omega/robust-region numerical safety pass；
- economic improvement >0；
- P100_TV_ratio<=1 且 F200_TV_ratio<=1。

X2/P2 IAE/ISE只是diagnostics。原 v1 标签不改，v2标签写到新CSV。
71/76/133的完整归档重新核对 nonlinear post-state、stage-cost timing、终端安全和
operating-point decomposition。只有 v2 witness>0 才允许训练。
oracle知道离线扰动路径，不能作为可部署策略或独立泛化结果。

## 本机执行

从项目父目录运行。以下 tests/audit/reclassify 已执行后，不重复覆盖其不可变输出。
新增源码后必须重新 tests；锁定后再改源码意味着需要一个新的实验版本。

```powershell
Set-Location "C:\Users\cushy\PycharmProjects"
$v2Python = ".\evaporation\.venv-cuda\Scripts\python.exe"
& $v2Python -m evaporation.zanon2019_paired_v2 tests
& $v2Python -m evaporation.zanon2019_paired_v2 audit
& $v2Python -m evaporation.zanon2019_paired_v2 reclassify
```

只先启动 seed42，100×1000、CUDA、每5回合固定 evaluation，保留 episode0：

```powershell
& $v2Python -m evaporation.zanon2019_paired_v2 train --seed 42 --device cuda
```

每回合 baseline 与 SAC 使用完全相同随机路径。路径seed为
`training_seed*1000000+episode`，local action RNG=`path_seed+77`，沿用现有实验。
literal variances F1/X1/T1/T200=[2,1,8,5]，std为平方根，不减扰动强度。
Gaussian iid、每秒独立采样、负F1/X1 clip沿用既有 **implementation choice for reproduction**。
不重新训练ECC2019 NMPC。

seed42完成100000步、无安全异常/NaN/Inf/action collapse、Replay核对成功，并至少
有一个 post-warmup positive-economic empirical-safe validation checkpoint 后，才可继续：

```powershell
& $v2Python -m evaporation.zanon2019_paired_v2 train --seed 2027 --device cuda
& $v2Python -m evaporation.zanon2019_paired_v2 train --seed 314159 --device cuda
& $v2Python -m evaporation.zanon2019_paired_v2 report
```

入口强制依次检查前一个种子的 continuation gate，不自动启动后续种子。partial/
aborted输出保留，不允许覆盖或静默恢复成新实验。安全异常立即保存 evidence并中止。

## Figures / learning / comparison

全部 checkpoint 使用固定10个 development paths `420000..420009`，不向actor提供
未来扰动。validation接口 no_grad + deterministic。每个checkpoint完整记录性能、
X2 operating point、reward decomposition、控制TV/RMS、模式/action collapse、安全/W。

完整21个evaluation checkpoints，不是只展示经济最好者。learning curves使用fixed paired
validation，保留各training seed原始点。training return/loss/entropy另列diagnostic图，
不强制曲线单调上升或接近0，未记录数据不补造。

`report` 自动输出所有 requested learning/Pareto PNG、三seed CSV、旧A vs全部v2
checkpoints表。未训练seed标记not_run，不填0。best-economic summary只是development
描述，不是final_v2 selection，不自动把它称为最佳综合策略。

Fig.2-type固定 validation_seed=420000、predeclared episode100。根figures使用seed42，
其他training seeds在figures/seed_<seed>分别保留。不是依据图形好看程度挑realization。
共轴X2/P2/P100/F200/instantaneous cost、完整响应和固定前200步X2放大图、累计成本。
X2线25和minimum（包含terminal）完整报告，P2物理边界和实际input robust bounds保留。
瞬时成本差 `l_SAC-l_baseline`，负值有利于SAC；不是百分比。
累计成本由原始 stage costs 直接cumsum，final gap与gain从同CSV核对。
`total_control_activity=(TV_P100/100+TV_F200/100)/1000`，不使用动态轨迹归一化。

训练完成后自动report，也可独立report。没有真实v2训练时不生成伪learning/仿真图。
Fig.2-type是问题表达方式，不是ECC2019 Fig.2 reproduction，也不声称baseline是论文NMPC。

## 认证限制与下一步

继续完整记录W_exceedance。约98%的Gaussian mismatch超原W，所有结论标记
`empirical_only_under_ECC2019_stochastic_disturbance`，不能写formal Gaussian robust guarantee。
finite-jump集合证书与本轮无jump工况不能混淆。

**本轮禁止执行430000..430049**。三training seeds完成并确认reward表现后，再单独
锁定final_v2 selection rule、检查独立测试授权，才可进行final independent test。
此处没有final-test CLI，不commit、不push。本轮准备代码、audit、reclassification，
正式训练由用户在本机执行。
