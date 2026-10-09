# Alpha=0.2 stochastic reward-scale ablation

Only three existing shaping weights change, by explicit `--reward-calibration`.
Omitting that option preserves the original reward. Safety geometry, action
mapping, observation, entropy, SAC optimizer settings and disturbance protocol
remain unchanged. Never resume an old scalar-reward replay buffer: start fresh.

## Audit

From `C:\Users\cushy\PycharmProjects`, run with the CUDA virtualenv Python:

```powershell
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_reward_audit
```

The audit reads all 21 checkpoints x 10 validation realizations from the old
alpha=0.2 experiment. Calibration uses each zero-residual baseline once, not
repeated checkpoint copies. Inputs are FINAL APPLIED inputs, including the
first move from physical(v_ref). State loss uses next state, reconstructed
from the saved actual mismatch and checked against the next CSV row. Stored
stage costs and J_econ are independently reconciled. No plant rollout or SAC
training is performed by the audit. Existing output is never overwritten.

Implementation choice for reproduction; not specified in the paper:
baseline state/P100/F200 shaping contributions are calibrated to 10%/5%/5%
of mean absolute economic reward. State loss remains the existing X2+P2
normalized squared loss; economic reward scale remains 200. Saturation is
unexcited on this baseline, so its original weight is retained.

Baseline means: |economic reward|=0.9946118186,
state loss=1.9759964571e-5, P100 move loss=0.0023233085,
F200 move loss=0.0001086460. Weights:

| Component | Old | Recalibrated |
|---|---:|---:|
| State recovery | 5.50146635 | 5033.46964536 |
| P100 actual move | 108.11728454 | 21.40507392 |
| F200 actual move | 81.88044907 | 457.73071321 |
| Saturation | 3.66616387 | unchanged |

The large numerical state-weight multiplier corrects a very small normalized
state loss. It is not a claim of improved learned performance or a penalty
sweep. Historical economic-minus-four-shaping is a partial objective, not
full return: old CSVs lack other penalties. New logs preserve all reward
components and distinguish raw evaluation return (including RPI terms) from
replay-equivalent reward (excluding RPI event/excess). RPI remains audited.

## Local training only

```powershell
Set-Location "C:\Users\cushy\PycharmProjects"
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_train --disturbance-mode zanon2019_stochastic --stochastic-residual-scale 0.2 --authority-pilot --episodes 100 --steps 1000 --seed 42 --device cuda --reward-calibration ".\evaporation\evaporation_safe_sac\outputs_zanon2019_alpha020_reward_scale_audit_v1\reward_calibration.json" --output-dir ".\evaporation\evaporation_safe_sac\outputs_zanon2019_alpha020_rewardcal_seed42_100x1000"
```

Fresh initialization, identical training disturbance/action RNG rules,
evaluation every five episodes on 420000..420009. Test seeds 430000..430049
remain reserved. The opt-in loader rejects other authority values or changed
geometry/reference/scales. Physical/input/QP/Omega safety anomalies still
stop training. W exceedances remain diagnostics; no Gaussian formal claim.

Compare episode100, episodes55..100 evaluation mean and the complete
post-warmup Pareto front against the old alpha=0.2 experiment. Economics
comparability is the existing per-seed 0.5% gate, with 1% also reported. Check
X2 IAE/ISE, G_X2, std, peak and margins separately; F200 TV and RMS separately.
Do not hide either in a two-axis mean. Report all safety counters, raw action,
collapse, applied/requested ratio and occupancy. A joint candidate must pass
the unchanged per-seed quality gates, not merely have the highest economics.
If economics becomes slightly negative but remains comparable, report that
trade-off explicitly. Do not automatically extend training or increase weights.
