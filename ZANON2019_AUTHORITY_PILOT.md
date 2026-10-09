# ECC2019 stochastic authority-specific retraining

This is an empirical stochastic experiment, not a Gaussian safety certificate.
The paired baseline is the frozen zero-residual controller, not ECC2019 NMPC.
New experimental choices below are **implementation choice for reproduction**;
authority scaling, seed split and checkpoint hierarchy are **not specified in the paper**.
The existing calibrated stochastic benchmark and paper specification are unchanged.

## Run locally

```powershell
Set-Location "C:\Users\cushy\PycharmProjects"
& "C:\Users\cushy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m evaporation.zanon2019_authority_pilot --run
```

This runs fresh alpha=0.10/0.20/0.30/0.50, each seed42, 100 episodes x1000
steps, evaluation every 5 episodes. It does not train alpha1, extend to300
episodes, run final tests, warm-start, or resume. Every target is checked before
starting any run. Existing evidence is never overwritten.
If a pilot safety-aborts, its evidence is preserved; only another independent
fresh alpha can start. Unexpected runtime/setup failures stop the launcher.
There is no automatic parameter change or restart.

Run roots are `evaporation_safe_sac/outputs_zanon2019_alpha010_seed42_100x1000`
through `alpha020`, `alpha030`, `alpha050`. A fresh `--run-root` may be used if
these directories already contain evidence. Use the same root for comparison.

To update comparison only, with no training:

```powershell
& "C:\Users\cushy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m evaporation.zanon2019_authority_pilot
```

For a single pilot, invoke `evaporation.zanon2019_train` with
`--authority-pilot --stochastic-residual-scale 0.30 --episodes 100 --steps 1000
--seed 42 --eval-seeds 10 --eval-seed-start 420000 --eval-every 5` and the matching
fresh `--output-dir`. Do not use the previous alpha1 output directory.

## Action/replay semantics

`a_raw` remains in [-1,1]^2. Warmup and stochastic actor exploration are
unchanged. The environment's AuthorityController passes `alpha*a_raw` to the
original interior-anchor mapping. The actor network, entropy/log-probability,
critic action and replay all continue to use `a_raw`, not `alpha*a_raw`.
This is authority-specific retraining, not post-hoc evaluation scaling.
alpha1 forwards the exact original action without an extra numerical operation.
The legacy command still uses alpha1, three evaluation seeds and legacy selection.
Non-unit scaling is rejected outside `zanon2019_stochastic`.

The controller/mapping/geometry/reward/23D observation are not redesigned.
Experiment I has no finite state jumps: Gm/Bj remain frozen and inactive, as in
the previous stochastic experiment. This adapter does not activate them.

## Validation and safety

Training disturbance seed: `42*1000000+episode`; action RNG: path seed+77.
Fresh independent runs reset SAC initialization and replay with seed42.
All pilot validation uses420000..420009. Final430000..430049 are reserved and
rejected by the training entry point. Evaluation is deterministic and paired
with cached same-initial-state, same-path, same-horizon zero residual.

Physical X2/P2/input violation, QP infeasibility or Omega exit stops training;
the failing actor, full checkpoint and diagnostic trajectories are preserved.
W exceedance remains logged but is not a checkpoint rejection criterion.
Model files record alpha in `models/action_environment.json`: use the same
environment alpha when evaluating an actor saved from that directory.

## Selection and evidence

Only post-warmup actors can be learned selections. Economics comparability is
checked separately for each seed: policy cost <=1.005 or1.01 baseline cost.
The0.5% gate is used for main selection;1% is reported, not silently substituted.
Mean economic improvement and worst-seed improvement are both reported.

Main selection is an explicit lexicographic hierarchy: empirical safety,
0.5% comparability, minimum/p1/p5 X2 margin and near-boundary fraction,
IAE/ISE/amplification/std, TV/RMS/occupancy, then economics.
Margin p1/p5 in the aggregate are worst per-seed quantiles, not pooled quantiles;
complete per-seed values are in fixed_evaluation.csv. No hidden weighted scalar
score and no automatic smallest-alpha rule are used.
Absolute boundary occupancy is reported, without inventing a cutoff.
Raw saturation uses the existing |action|>=0.99 convention, distinct from
scaled action and physical applied residual magnitude.

Saved selections: best_safety_economic_actor, best_economic_actor,
best_disturbance_rejection_actor, best_margin_actor, best_low_activity_actor,
final_actor (pre-warmup final remains diagnostic instead). A best-safety
checkpoint alone is not a claimed joint-success policy: inspect
authority_joint_candidate and all validation metrics.

Comparison outputs in `outputs_zanon2019_authority_comparison`:
stochastic_authority_comparison.csv, selected_checkpoints.csv,
all_checkpoints.csv, pareto.csv (the latter three have stochastic_authority_
prefix), all eight requested authority plots, authority_decision.json and summary.
Unavailable runs/checkpoints are blank, not zero. Aborted runs remain marked.
The original alpha1 episode5 is separately saved as a failed historical
three-seed reference; it is not a ten-seed retrained authority candidate.
Full cross-authority comparison verifies equality of frozen experiment/SAC
configuration, geometry hashes and validation protocol, apart from alpha.

No automatic300-episode promotion or final-test execution is supported.
Review the Pareto front and per-seed margins/control occupancy before deciding
whether any validation candidate warrants further training. Without completed
pilots, sweet-spot authority and reward-change conclusions remain unavailable.
