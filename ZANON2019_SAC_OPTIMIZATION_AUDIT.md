# Frozen SAC optimization audit

Only diagnostic modules were added. No training, optimizer update, replay mutation,
reward change, safety change, commit or push is performed.

Run the full offline audit from PowerShell:

```powershell
Set-Location "C:\Users\cushy\PycharmProjects"
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_sac_optimization_audit --device cuda
```

Defaults: original 10 validation disturbance seeds; actors 100/90/80; 20 stochastic
action realizations per disturbance seed; 512 evaluation states for critic probes;
12 states × 7 first actions × 20 Monte Carlo continuations of 1000 steps; 41×41 grids.
Output: `evaporation_safe_sac/outputs_zanon2019_sac_optimization_audit/`.
Existing nonempty output directories are never overwritten. Use a fresh
`--output-dir` for reruns. Partial CSVs are retained on interruption.

This is substantial offline evaluation, NOT SAC training. CUDA accelerates neural
network calls; nonlinear simulation and one-step geometry/QP remain CPU work.

## Evidence boundaries

- Episode100 has its matching current and target critics. Episode80/90 have actor
  checkpoints only: their policies can be evaluated, but final critics are never
  misrepresented as matching historical critics.
- No replay snapshot or per-transition training trace exists in the current run.
  Coverage, OOD distances and stored/recomputed replay rewards are unavailable.
  The corresponding CSVs and replay plot explicitly say unavailable.
- Diagnostic states preserve moving nominal z, previous FINAL applied input,
  previous residual, uncertainty EMA, mode history and observation. A first-action
  branch does not reset nominal state to actual state.
- Soft Q is compared against conditional soft Monte Carlo return. No first-action
  entropy bonus is included; subsequent stochastic continuation uses the saved
  checkpoint's entropy coefficient (not the residual scale 0.2).
- Economic/IAE/TV/environment return have NO entropy bonus. Raw evaluation RPI
  penalties are separated from replay-equivalent rewards.
- Full Monte Carlo uses common disturbances and action-sampling random numbers
  across candidate first actions. Safety-aborted trajectories are flagged and
  excluded from complete-horizon calibration, not presented as safe successes.
- gamma^1000 is approximately 4.32e-5, but this is a discount factor, not a bound
  on absolute truncation error. Branches stop at the original episode terminal
  (1000 minus source step), matching the stored done convention. Remaining time
  is not encoded in observation: time aliasing can also affect calibration.
  Short diagnostic branches stopping before that terminal retain truncation bias.
- A short-horizon smoke test is NOT evidence of miscalibration, stochastic mismatch,
  systematic overestimation, coverage deficiency, or any final Case A-F diagnosis.
- State bands absent from measured trajectories are marked unavailable, not
  replaced by fabricated states. Sampled actor actions are not replay actions.

All diagnostic budgets, state bands, sampling protocols, CI approximations and
finite differences are **implementation choices for reproduction**, not specified
in the paper. The audit does not provide a nonlinear continuous-domain safety proof.

Run the no-training unit tests:

```powershell
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_sac_optimization_audit_tests
```

After the full run, inspect `summary.json`, the nine requested CSVs, diagnostic
figures and `zanon2019_sac_optimization_audit.md`. Do not select a SAC modification
from missing replay evidence or from a smoke-test correlation.
