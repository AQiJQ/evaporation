# Stage3 2D reference feasibility — first-phase results

First phase completed. **Finite oracle not run; SAC not run; training gate
remains closed pending safe positive finite-search evidence.** No claim of
economic recovery by SAC is made from these baseline-only results.

## 1. Local steady sensitivity and fixed protocol

Strong physical reference: `[25.39, 50.125]`.
Exact nonlinear steady input: `[194.8614429683, 216.2766535052]`.

| Physical input derivative | X2 (percentage point) | P2 (kPa) |
|---|---:|---:|
| P100 | 4.2031634500 | 3.9177503253 |
| F200 | 11.3353972097 | -10.7384110153 |

Central-difference steps: `[1e-4, 1e-4]`, checked at half and double steps;
all sensitivity checks passed (rtol=1e-6, atol=1e-7).

Local compensation direction: `dP2/dX2 = +1.0555935318`.
Along this direction the local F200 change is zero while P100 increases
by approximately `8.33871535 * dX2`. Both input channels are checked.
The limiting current steady-input bound is **F200 upper**. Fixed-P2 linear
headroom permits only approximately `delta X2=0.00234906`, explaining the
previous 1D rejection without contradicting or replacing that evidence.

Frozen tightened physical input interval:

- P100: `[162.3005488378, 225.9359290949]`.
- F200: `[206.1828780169, 216.3032809876]`.

Prerecorded X2 targets: `[0.01,0.02,0.03,0.05,0.10,0.15,0.20]` above Strong.
Fixed local P2 range: `[49.7027625873,50.5472374127]`, obtained from the
Jacobian rule and intersected with existing physical/S bounds. No expansion
after looking at economics. Input-bound bisection tolerance=1e-10, full-gate
boundary tolerance=1e-8, numerical input inset=1e-8 physical input units.

For all seven targets, the exact nearest input-bound point also passed
**all 17 complete reference checks**, so the fallback grid was not used.
No economic objective or SAC result was used to choose P2. The input-feasible
local P2 corridor widths range from 0.414165 to 0.215253 kPa: fixed-P2
headroom is extremely small, but the tested 2D corridor is not a single line.

## 2. Complete preregistered candidates

All values below are baseline-only; mean J is across ten matched paths.

| delta X2 | nearest P2 | P100 | F200 | P100 nearest tight margin | mean J_econ | economic penalty % | minimum X2 margin | margin increase |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.01** | **50.133072198** | **194.935091497** | **216.303280978** | 31.0008376 | **6,218,788.848101** | **0.01408666** | **0.22258878** | **0.01021737** |
| 0.02 | 50.143615433 | 195.018405019 | 216.303280978 | 30.9175241 | 6,219,701.864529 | 0.02877031 | 0.23324627 | 0.02087486 |
| 0.03 | 50.154150373 | 195.101670149 | 216.303280978 | 30.8342589 | 6,220,614.150133 | 0.04344220 | 0.24390248 | 0.03153108 |
| 0.05 | 50.175195406 | 195.268055462 | 216.303280978 | 30.6678736 | 6,222,436.533241 | 0.07275080 | 0.26521112 | 0.05283971 |
| 0.10 | 50.227663505 | 195.683175866 | 216.303280977 | 30.2527532 | 6,226,979.777837 | 0.14581784 | 0.28781228 | 0.07544087 |
| 0.15 | 50.279926170 | 196.097097817 | 216.303280977 | 29.8388313 | 6,231,504.968744 | 0.21859452 | 0.29018382 | 0.07781242 |
| 0.20 | 50.331984603 | 196.509828342 | 216.303280977 | 29.4261008 | 6,236,012.239745 | 0.29108301 | 0.29255336 | 0.08018195 |

Each F200 tightened-input margin is approximately **1e-8**, not zero;
the exact individual margins and steady residuals are in the candidate CSV.
Steady derivative infinity norms are at most 4.62e-15. All seven candidates
are reference-feasible AND valid Conservative candidates under the specified
ten-path baseline selection rule. No candidate was silently removed.

## 3. Frozen Selected Conservative Reference

`[X2_ref,P2_ref,P100_ref,F200_ref] =`
`[25.4000000000,50.133072197983,194.935091496659,216.303280977570]`.

This is the **first ascending target** meeting all gates, not the largest
cost penalty and not the candidate with predicted maximum SAC gain.
Selection was frozen before any oracle/SAC and used no SAC results.

| Metric | Strong | Selected Conservative |
|---|---:|---:|
| mean J_econ | 6,217,912.951733 | 6,218,788.848101 |
| mean X2 | 25.38492377 | 25.39533989 |
| worst-path minimum X2 | 25.21237141 | 25.22258878 |
| worst-path minimum X2 margin | 0.21237141 | 0.22258878 |
| mean X2 std | 0.04302705 | 0.04303978 |
| mean centered X2 MAE | 0.03416478 | 0.03417669 |
| mean X2 IAE | 34.78991693 | 34.77541222 |
| mean P2 IAE | 53.19767962 | 51.93514316 |
| mean P100 TV | 3838.05811136 | 3839.80318279 |
| mean P100 RMS delta input | 4.82125969 | 4.82344878 |
| mean F200 TV | 823.57027475 | 823.93537873 |
| mean F200 RMS delta input | 1.03481861 | 1.03528050 |
| mean W exceedance rate | 0.9828 | 0.9828 |

Economic penalty: **875.89636797**, or **+0.0140866618%**; well above the
fixed 2e-4 absolute numerical guard, but still a small economic difference.
It must not be presented as a large SAC economic gain.

Every path's minimum X2 margin increased by 0.01021737–0.01089921 (>1e-4).
The reported aggregate minimum includes terminal states and is the worst
path, not the mean of ten minimum margins. IAE uses each design's own frozen
reference; common physical initial state for both is Strong B `[25.39,50.125]`.

There were 70 new candidate rollouts, each 1000 steps at 1 s, on only
420000..420009. The ten existing Strong traces were reused read-only after
regenerating and exactly checking their disturbances and independently
re-auditing nonlinear transition costs, terminal constraints and safety.

All 70 new paths had zero:

- physical state violation;
- physical input violation;
- QP infeasibility;
- Omega exit;
- robust-region anomaly.

W exceedance rates remained 0.9828–0.9833 across the grid. This is **empirical
safety under the stochastic reproduction distribution**, not a formal
Gaussian-disturbance safety certificate or continuous nonlinear-domain proof.

## 4. Geometry and numerical limitations

No K/W/Z/S/tightening/QP, physical constraints, Strong reference, reward,
residual authority, controller mapping, observation or SAC setting was changed.
Implementation choice for reproduction: as in the previous reference-only
module, fixed A/B and normalization are retained, the affine predictor is
re-anchored at the exact new equilibrium, and Omega error coordinates are
translated without changing its physical set.

Selected minimum residual authority=**0.0574845382** (floor remains 0.05).
Spectral radius=0.9980369585; sampled Hinf norm=2999.9974135453,
gamma=3000; sampled Hinf margin=0.0025864547.

The unchanged RPI infinite-support/tail certificate passes. The separately
reported plotted outer polygon has direct facet excess about 7.23077e-7 and
is not itself guaranteed RCI; no new facet tolerance was silently introduced
to reinterpret the original RPI gate. See each `reference_checks/*.json`.

**Input margin warning:** nearest-feasible P2 places the nonlinear F200
reference essentially at the frozen tightened upper boundary (only ~1e-8
numerical inset). This is not a high-input-margin reference search. The tested
closed-loop state margin did increase, but input-margin claims must remain
distinct from that result.

Stage2's 24 output files retained identical SHA256 hashes; its original
fixed-P2 failure result still holds. Strong frozen source/config and geometry
hash checks passed. No new 430000..430049 simulation, training, selection or
threshold tuning was performed. No commit or push.

Tests: 14 new reference tests passed; complete suite 162 tests passed with
the optional CasADi group separately passing 3 tests; `evaporation.test`
passed. The CUDA environment was used as interpreter; no SAC experiment ran.

## 5. Files and next gated phase

Output directory:
`C:\Users\cushy\PycharmProjects\evaporation\evaporation_safe_sac\economic_recovery_v1\conservative_mechanism_stage3_2d\`

- `protocol_2d.json`: immutable range/grid/tolerances and frozen hashes.
- `steady_input_local_jacobian.json`: four derivatives and sensitivity checks.
- `feasible_direction_analysis.json`: compensation and limiting bound.
- `conservative_2d_reference_candidates.csv`: all seven references and input margins.
- `conservative_2d_baseline_results.csv/json`: complete aggregate metrics.
- `conservative_2d_baseline_results_per_seed.csv`: all 70 path metrics.
- `conservative_2d_selection.json`: frozen first-eligible baseline.
- `conservative_2d_safety_economics_tradeoff.png`: inspected trade-off plot.
- `baseline_trajectories/`: full matched disturbance/state/input/cost traces.
- `tests_receipt.json`: complete test receipt.

Finite oracle status: **not run**, not failed and not an infeasibility result.
Candidate/safety-passing/positive-economic counts and gains are not available.
The Conservative-SAC training gate remains **closed** until this next phase
produces verified safe positive paired economic evidence. No pilot, other
training seed or independent test is allowed to bypass that gate.

The independent oracle entry is implemented and retains the previously
declared finite 512-candidate budget, 10/5-step profiles and seed 190019.
All ten paths use the selected Conservative baseline and common initial B,
through the real interior-anchor/QP/nonlinear plant. Only finite best-found
results may be reported, never a mathematical upper bound.

From `C:\Users\cushy\PycharmProjects`:

```powershell
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_reference2d oracle
```

This is a substantial offline simulation workload (up to 5,120,000 online
control steps), not SAC training. Do not use the old fixed-P2 Conservative
training entry for the new 2D reference. Local seed42 training must be gated
by the new selected-reference/oracle evidence and keep the frozen smoothness
weight and /200 paired economic objective. No training was launched here.
