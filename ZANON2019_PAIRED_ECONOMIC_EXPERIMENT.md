# Independent paired-economic experiment

This is a NEW development experiment. The locked Experiment I, its policies,
final_v1 rule, reserved test seeds, controller sources, K/W/Z/S/Omega, 23D
observation, interior-anchor mapping and QP are not modified. No baseline is
weakened retrospectively to inflate the primary gain. No claim of ECC2019
RL-NMPC reproduction or formal Gaussian safety is made.

## Objective

Primary reward component is `(cost_zero_residual - cost_SAC) / 200`, with the
same initial B state and exact same stochastic disturbance path. Zero means
baseline economics. A successful learned policy should approach a POSITIVE
fixed-evaluation plateau, not an artificially forced zero reward. Economic
cost itself, and the Fig.2 difference SAC minus baseline, are unchanged.

Constraints use all 1000 pre-step states for X2/P2 IAE (no 40-second crop),
and actual final inputs for TV/occupancy. X2/P2 IAE and P100/F200 TV have the
existing 10% comprehensive-performance tolerance. TV constraints use excess
TV divided by horizon times robust input range; baseline TV=0 is safe to
evaluate and implies zero allowed extra TV. First-input movement is not TV,
consistent with the existing stochastic evaluation's `diff(control)` metric.

The old stochastic checkpoint rule had NO occupancy hard threshold. This new
experiment explicitly extends the same 10% tolerance to occupied-step count:
`g_boundary = (occupied_SAC - 1.1*occupied_base) / steps`. This is an
**implementation choice for reproduction**, not specified in the paper.
If baseline occupancy is zero, no extra occupancy is allowed by this
performance gate. Hard safety constraints and safety QPs are never relaxed.

Replay saves economic reward and five independent signed constraint
contributions. The sum of constraint contributions equals episode g. A fixed
positive scaling (90th percentile of positive development violations with
documented numerical floors) conditions the dual update without changing
constraint boundaries. Current multipliers are applied when sampling old
transitions, not baked into stored rewards. Dual starts at zero, learning
rate .1 and cap100 are initial implementation choices, not a claim of optimal
tuning. Reaching the cap while a constraint is violated aborts and reports a
trade-off, not a reason to raise the cap indefinitely.

Old fixed weighted reward is still computed by the immutable rollout and
saved ONLY as `legacy_weighted_return_diagnostic`. The replay adapter replaces
its learning signal using audited final input, pre-state for IAE, and actual
next-state economic cost. No original source needs to be patched.

## Run order (PowerShell, local machine)

Start in the repository parent; the CUDA environment is used for training:

```powershell
Set-Location "C:\Users\cushy\PycharmProjects"
$pyEvap = ".\evaporation\.venv-cuda\Scripts\python.exe"
& $pyEvap -B -m evaporation.zanon2019_paired_tests
```

### 1. Archived checkpoint constraint audit (no training)

```powershell
& $pyEvap -B -m evaporation.zanon2019_paired_experiment audit
```

Fresh output directories are mandatory. If audit has already been generated,
use a new `--output-dir` and pass its `constraint_audit.json` to training.
The audit reads the 630 existing seed/checkpoint/realization rows, provides
individual g and complete-checkpoint summaries, and preconditioning scales.
Five feasible realizations do not mean five eligible checkpoints.

### 2. Actor-action-space feasibility search (no SAC training)

```powershell
& $pyEvap -B -m evaporation.zanon2019_paired_experiment oracle --segment-steps 10 --population 12 --generations 12
```

This derivative-free Pareto archive uses block mutations, global offsets,
and constant-profile restarts. It is a finite heuristic search, NOT a global
optimizer. It evaluates common profiles on development seeds420000..420002.
Every candidate uses raw actor commands in[-1,1]^2, alpha=.10, the real
interior-anchor mapping, the real safety QP and nonlinear plant. It does not
optimize q directly and is not an online MPC. It saves all candidate metrics,
failed trajectories, the nondominated front, representative actions/traces,
and `oracle_summary.json`. Unsafe/partial candidates cannot count as witnesses.
The zero-action candidate is a compulsory baseline-identity check.

Only if search is productive, refine at5-second segments in a new directory:

```powershell
& $pyEvap -B -m evaporation.zanon2019_paired_experiment oracle --segment-steps 5 --population 16 --generations 20 --output-dir ".\evaporation\evaporation_safe_sac\paired_economic_v1\oracle5"
```

An offline oracle uses future path information during optimization. Its
trajectory witness is NOT a trained feedback policy, deployable controller,
unbiased gain estimate, or global upper bound. No witness means only that
this search did not find one, not that none mathematically exists.

### 3. Training ONLY after a joint witness is present

```powershell
& $pyEvap -B -m evaporation.zanon2019_paired_experiment train --device cuda --seed 42 --episodes 100 --steps 1000
```

Training refuses `joint_witness_count=0`, protocol mismatch, missing/changed
trajectory evidence, or changed audit sources. If using oracle5, explicitly
pass its `--oracle-summary` path. Do not fabricate a witness to bypass the gate.

Network/LR/buffer/batch/entropy settings are reused. Training disturbance RNG
is still `seed*1000000+episode`, action RNG `path_seed+77`; the baseline rollout
uses a separate local generator and no Torch/global action samples. Actor
never sees the future path or baseline trajectory. Each episode gets its own
exact paired baseline; no amplitude/path interpolation cache is used.

Fixed deterministic evaluation runs every5 episodes on420000..420009. It
records J_econ/IAE/ISE/TV/RMS/sign changes/occupancy, collapse, Z/Omega modes,
constraint g, actual objective and paired economics. Primary learning curves
are fixed paired economic gain/reward, with dual and constraint plots. Returns
under changing lambda are not directly comparable performance scores.

Selection requires post-warmup, zero physical/input/QP/Omega/robust-region
counts, positive mean and nonnegative worst-realization gain, and all five
performance constraints in every validation realization. Eligible joint
checkpoints are ranked by combined IAE/TV ratio, then economics, not economic
max alone. No joint checkpoint leaves `best_joint_checkpoint=null`; diagnostic
or final actor is never silently promoted to a successful joint policy.

All safety anomalies stop immediately with evidence. Nonlinear W exceedance
is still recorded, not deleted or relabeled as Gaussian certification.
Reserved independent test seeds430000..430049 are rejected by all phases.

## Optional supplementary baseline sensitivity

```powershell
& $pyEvap -B -m evaporation.zanon2019_paired_experiment baseline-sensitivity --supplemental-action 0.25 0
```

This predeclared fixed performance-bias actor command uses the SAME safety
filter and constraints. It does not redesign K or weaken physical gates.
It may or may not be more conservative/economically weaker; both safety and
cost must be checked. It is explicitly supplementary, not ECC2019 naive
NMPC, and never replaces the primary zero-residual reward/selection baseline.
To claim a paper-baseline comparison, an independently specified NMPC
reproduction must actually be run; these numbers cannot substitute for it.

No training or final test is automatically launched; no commit or push.
