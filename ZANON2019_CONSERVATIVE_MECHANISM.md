# Conservative mechanism stage2

最新请求允许先做 Conservative mechanism，暂不等待 Strong 2027/314159。旧 `economic_recovery_v1/protocol.json` 和所有 Strong/v2 文件不改。新增阶段协议和输出位于 `economic_recovery_v1/conservative_mechanism_stage2/`。

本阶段入口不会训练 SAC，无云任务、commit 或 push。只有 reference/安全门禁、margin gate、明确正 economic penalty 和安全正收益 oracle 全通过，才有理由另行准备本机 seed42 训练。本轮如果 gate 失败则停止，不调用旧训练入口绕开。

## 本机命令

```powershell
Set-Location "C:\Users\cushy\PycharmProjects"
$mechanismPython = "C:\Users\cushy\PycharmProjects\evaporation\.venv-cuda\Scripts\python.exe"
& $mechanismPython -m unittest evaporation.zanon2019_conservative_mechanism_tests -v
# prepare/sweep/oracle 各仅运行一次，已有结果保持不覆盖。
& $mechanismPython -m evaporation.zanon2019_conservative_mechanism prepare
& $mechanismPython -m evaporation.zanon2019_conservative_mechanism sweep
& $mechanismPython -m evaporation.zanon2019_conservative_mechanism oracle
& $mechanismPython -m evaporation.zanon2019_conservative_mechanism status
```

## 比较口径

继承预注册 X2 increment `[0,.05,.10,.15,.20]` 和固定 P2 reference=50.125。每个 candidate 用精确 nonlinear `steady_input` 得到完整 u_ref，检验 derivative≈0、reference∈S、reference input∈U-KZ、H∞/RPI、Omega、verification QP、5% residual authority 与有限 nominal W coverage。沿用上一版本的固定 B Jacobian 和 reference-derived affine equilibrium reanchoring。不得为候选改 K/W/Z、物理 S/Omega、tightening、QP、mapping 或经济函数。

最新要求是全部候选相同 initial condition，故统一 physical reset `[25.39,50.125]`，不是每个 candidate 自己的新稳态。target 与 reset 分开记录。420000..420009 的相同 Gaussian iid disturbance trajectories、1000 steps/1 s、同 nonlinear plant。每个 Conservative oracle 也与本方 baseline 使用该 common reset。Gaussian safety scope 仍 empirical-only，W 超界单列。

选择第一个正 delta，使十条路径各自 minimum X2 margin 增加 > 已预注册 1e-4，整体 worst minimum 同时增加，且安全/reference/numerical checks 全通过。选择不看 SAC 或经济排序。整个固定 grid 都报告；即使后面更远点成本更大，也不替换第一个合法候选。

`baseline_conservatism_sweep.csv` 为五个 candidate 的 summary；`..._per_seed.csv` 为真正完成的各条 baseline 数据；JSON 有全部 gates/失败原因。minimum margin 是全部十条完整轨迹含 terminal state 的最低值，不是各 minimum 的平均。被 reference gate 拒绝的候选不做假 baseline simulation，mean J/margin 留空。其 `steady_stage_cost` 是 nominal equilibrium 的单步经济成本，不可当作 stochastic rollout mean J。

Conservatism penalty 由 mean paired baseline costs 差得到；数值显著性 guard 使用既有 stage-cost audit tolerance 1e-7 的两条 1000-step 求和误差量级 `2e-4`，不设期望 1% 收益。没有合法 reference 或没有明确正 penalty，则 oracle summary 标 `blocked_before_search`，不是搜索无收益或收益为0。

如果合法，有限 oracle 用512个 raw actor-action候选：constant grid、10s/5s segment profiles、Pareto mutation。每条都经过 unchanged mapping/QP/nonlinear plant；保留 safety counters、TV/RMS 与 Pareto witness。结果是 best found，非数学上界；失败候选不能用 partial rollout 成本评分。430000..430049 没有入口或开发数据通道。

本轮没有为了通过当前 gate 自动优化 P2。若固定 P2 的全部候选被 U-KZ 拒绝，应先报告这个实际瓶颈，而不是擅自换二维 reference selection 或放宽安全几何。
