# ECC2019 reproduction and comparison

## Retired from the primary quantitative benchmark (2026-10-06)

**exploratory reproduction, not used as the primary quantitative benchmark**.
The historical commands below are retained for provenance, not recommended next
experiments. Do not continue fitting ECC2019 parameters to resemble Fig.2.
The current paper strategy is qualitative external literature comparison with
ECC2019, and strict quantitative paired comparison of our zero-residual robust
baseline with existing frozen Residual SAC checkpoints. See
`evaporation_safe_sac/paper_results_stochastic/`. No ECC2019 statistics inferred
from pixels or untrained controllers belong in the primary comparison table.

This implementation is **ECC2019_reproduction_with_documented_assumptions**,
not exact reproduction. Read the feasibility/assumption files before fitting.
Only new isolated modules were added; SAC and all frozen safety designs are unchanged.
There is NO supplied trained ECC theta. Naive solver checks are never an RL-tuned baseline.

## Completed locally without training

- Reread full supplied PDF and visually reviewed Numerical Example/Fig1/Fig2.
- Implemented non-condensed N=10 NMPC with learned quadratics/model bias/state bounds,
  soft state slack with ws=1/Ws=I, physical hard inputs, Q/V problems and envelope gradients.
- Implemented isolated Eq8 constrained TD fitting with full-convergence checks,
  positive stage/terminal Hessians, damping.01, frozen target and500-step policy refresh.
- Verified eight no-training tests, including identical symbolic/current nonlinear RK4,
  Q(s,greedy)=V(s), envelope-gradient finite difference and refusal of naive artifacts.
- Verified/reused all existing10-seed1000-step SAC/baseline paths against actual plant,
  economic cost, common B initial state and shared exogenous arrays. No new rollouts trained.
- Generated eight comparison figures, with ECC explicitly unavailable until fitted.

## Isolated solver environment

`.venv-ecc2019` contains CasADi/SciPy; current CUDA/SAC environment is not modified.
The solver environment reads existing CUDA benchmark dependencies without writing them.
If recreating it on another machine:

```powershell
Set-Location "C:\Users\cushy\PycharmProjects\evaporation"
& ".\.venv-cuda\Scripts\python.exe" -m venv --system-site-packages ".venv-ecc2019"
& ".\.venv-cuda\Scripts\python.exe" -m pip --isolated --python ".\.venv-ecc2019\Scripts\python.exe" install casadi scipy
```

## Review / tests / solver-only checks

```powershell
Set-Location "C:\Users\cushy\PycharmProjects"
& ".\evaporation\.venv-ecc2019\Scripts\python.exe" -m evaporation.ecc2019_reproduction review
& ".\evaporation\.venv-ecc2019\Scripts\python.exe" -m evaporation.ecc2019_reproduction_tests
& ".\evaporation\.venv-ecc2019\Scripts\python.exe" -m evaporation.ecc2019_reproduction solver-smoke
```

The smoke result uses NAIVE initialization and is excluded from comparison. A tiny
positive terminal Hessian resolves the paper's zero-initialization/strict-PD inconsistency.
Canonical slack usage reports required slack, not IPOPT's tiny barrier residual slack.

## ECC-only fitting, on your machine

No SAC training is needed. Training seed419001 is disjoint from development/final seeds.
The command below runs a100000-step ECC fit, NOT a SAC run. This is expensive: each
normal sample requests a converged Eq8 fit with multiple inner Q NLPs. Start only after
reviewing assumptions. A shorter500-step pilot can check execution but does not establish
convergence or reproduce paper results; use its own fresh output directory.

```powershell
& ".\evaporation\.venv-ecc2019\Scripts\python.exe" -m evaporation.ecc2019_reproduction train-ecc --training-steps 100000 --output-dir ".\evaporation\evaporation_safe_sac\ecc2019_parameter_fit"
```

The default fit-every=1 matches the Section-IV schedule. Changing it to500 is an
explicit computational approximation and is logged; do not relabel that as exact paper
learning. fit-window32, gamma.99, physical steady-deviation quadratic coordinates,
IPOPT/SLSQP,1s prediction discretization, training length and gauge are documented choices.
The Eq8 fitter is implemented, but no actual ECC learning/convergence experiment has
been run here. Its first pilot can fail; such evidence must be reported, not bypassed
or mislabeled as an authors' RL controller. Do not start a long fit before the pilot.
Exploration follows the printed physical-origin saturated Gaussian literally: it almost
always gives[100,100]. This is not silently changed to Gaussian noise around the greedy input.

No solver-failure fallback is presented as a successful ECC solution. Full-fit failure,
PD failure or objective increase marks training_failed and blocks comparison. Successful
fitting does NOT establish authors' parameter convergence or a nonlinear stability proof.

## Development comparison AFTER ECC fitting

```powershell
& ".\evaporation\.venv-ecc2019\Scripts\python.exe" -m evaporation.ecc2019_reproduction development --ecc-parameters ".\evaporation\evaporation_safe_sac\ecc2019_parameter_fit\ecc2019_learned_theta.json" --output-dir ".\evaporation\evaporation_safe_sac\outputs_ecc2019_development_trained"
```

Only420000..420009 are used. Frozen SAC episode100 candidates alpha.10 and rewardcal
alpha.20 are predeclared, no test-based checkpoint selection. Existing ours paths are
verified then reused; the trained ECC controller gets the EXACT same arrays and initial B.
All methods share plant,1s sampling,1000-step horizon and next-state stage cost convention.

- Main common IAE/ISE reference: nominal[25,49.743]; fixed-B diagnostics separate.
- Safety includes terminal state (1001 samples), performance uses1000 left endpoints.
- Control-bound occupancy in cross-method tables uses common physical100..400 bounds.
  Ours robust-bound occupancy/QP/Omega/W remain internal and are NOT fake ECC zeros.
- G_emp uses the same fixed-B error/standardized disturbance norm for every method,
  with an additional centered-fluctuation diagnostic. Neither is an H-infinity norm.
- Paired economic gap=(J_ours-J_ECC)/J_ECC*100, primary±.5%, sensitivity±1%.
- Aggregation uses seed-level mean/sample std/95%CI/min/max, not pooled pseudo-replication.
- ECC slack and actual physical violations are separate; learned bounds do not redefine
  the external physical X2>=25 metric.
- Economic tables exclude slack/shaping; J plus physical soft penalty is separate.
- Fig2 qualitative validation is manual review of trained traces; no digitization,
  gain-target fitting or numerical std/IAE inference from pixels.

No50-seed final-test command is implemented here. Approve the trained development
reproduction, qualitative behavior and ONE prelocked SAC candidate before final testing.
No commit or push.
