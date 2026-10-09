# Candidate A: frozen reproducibility and independent paired test

This workflow adds orchestration/auditing/reporting only. It does not change
any controller, observation, reward, mapping, replay or network. Final policy
selection is now permanently `final_v1`, explicitly authorized before test.
No training or independent test was run while preparing this workflow.

## Permanent final_v1 checkpoint selection

The old `AuthoritySelection` margin-first aliases remain historical/diagnostic.
Its episode5 choice was negative and is not used as a final policy. All saved
post-warmup checkpoints are rescanned from their ten paired validation rows.
Final gates: zero physical-state/input/QP/Omega counts; mean paired economic
improvement >0 and worst validation seed >=0. W exceedance is diagnostic only.
No X2 IAE<=1.10 hard gate is introduced.

Ranking: X2 IAE first, with absolute 0.01 near-tie bands anchored at the smallest
remaining ratio; within each band, descending min/p1/p5 X2 margins, ascending
near-boundary fractions, P2 IAE/ISE/std, mean TV/RMS and occupancy; economics is
the last performance tie-break. Identical metrics use earlier episode as the
deterministic final tie-break. Other fields use exact comparison (zero tolerance).
No chained pairwise tolerance comparator or human checkpoint override is used.

`final_checkpoint_selection_rule.json` records the rules and selector source
SHA256 permanently. Old pre-final-v1 draft evidence is preserved under `history/`.
Seed42 was automatically selected at episode15, not manually assigned episode100.
`seed42_final_selection_audit.csv` contains all 21 checkpoints (18 eligible).
Candidate B remains a development Pareto point with different reward settings,
not a pure authority ablation or a three-seed training target.

## Frozen protocol

- Training initialization seeds: 42, 2027, 314159. Each exactly 100x1000.
- Candidate A authority=.10, original uncalibrated weighted reward, CUDA.
- Validation: 420000..420009, paired fixed evaluation every 5 episodes.
- Independent test: 430000..430049. No training or selection access.
- Representative test realization: 430000, predeclared, not best-case choice.
- Literal variances [F1,X1,T1,T200]=[2,1,8,5]. Gaussian iid, negative feed
  clipping, initial B, 1s sampling and 1000 steps are documented implementation
  choices, not exact author-specified ECC2019 implementation details.
- Each test seed: one baseline + three frozen deterministic policies.
- Final safety anomalies are retained; testing continues unless the program
  cannot continue a trajectory. Program failures get explicit missing metrics,
  not zero anomalies or omitted seeds. Training safety aborts remain unchanged.
- Local artifact auditing found no prior execution evidence for the reserved
  seeds. This cannot establish absence of unrecorded external/human use.

## Run locally

From PowerShell:

```powershell
Set-Location "C:\Users\cushy\PycharmProjects"
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_final_experiment --phase train
```

This trains **only 2027 and 314159** sequentially. Seed42 is strictly checked and
reused. Existing completed/failed runs are preserved, never overwritten,
automatically extended or retrained. A partial run without a run summary requires
manual evidence review. Do not delete it to obtain a better result.

After training has finished (do not change parameters for a failed seed):

```powershell
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_final_experiment --phase lock
```

This writes `locked_final_policies.json` and immutable copies of final_v1 eligible
actors only. Missing/in-progress runs block final locking. A completed run without
an eligible checkpoint is `no_stable_positive_economic_safe_checkpoint`, with
`selected_checkpoint=null`. Best empirical-safe/final actors remain diagnostic;
there is no forced fallback policy. Three-row validation results are written to
`training_seed_reproducibility_validation.csv`. Pending rows are not failures.

Stop here. No independent final tests in this stage.

```powershell
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_final_experiment --phase validation
```

This only regenerates validation screening. `--phase run` may be used locally
to train pending seeds, lock the policies and then **stop**. It never starts final
tests. Both `--phase test` and the test function reject access without a separate
later independent-test authorization tied to the immutable policy lock. No such
authorization is generated in this stage. Do not create it merely for smoke tests.

Case A: 3/3 completed runs eligible, ready to request independent testing.
Case B: 2/3 eligible, report the failed run and discuss testing separately.
Case C: 0/3 or 1/3 eligible, pause final testing; do not retune/retrain to hide failure.
While runs are pending, report `pending_training_completion`, not Case C.

Future final-test reports (not run in this stage) can be regenerated from
completed saved evidence without new rollouts:

```powershell
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_final_experiment --phase report
```

Output: `evaporation_safe_sac/final_stochastic_candidate_A/`.

## Local completion command (no final test)

```powershell
Set-Location "C:\Users\cushy\PycharmProjects"
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_final_experiment --phase run --device cuda
```

This preserves the already locked seed42 episode15 without rescanning it, trains
only pending seeds2027/314159, applies final_v1 after training, generates the
three-row validation CSV and `training_seed_reproducibility_summary.json`, then
locks eligible actors/null failures and stops. It never calls the final test.
If training safety-aborts, it records the abort and proceeds to the other seed
without retrying the failed seed. Partial runs without terminal evidence require
review, not automatic restart.

After all runs finish or legitimately abort, four `training_seed_validation_*.png`
figures are generated. Failed policies remain labeled `no eligible policy`.
Within-policy n=10 descriptive validation intervals and between-training-seed
sample stds remain separate. Missing policies are excluded from policy aggregates,
not filled with zero. With only one eligible policy, between-policy std is null.
Summary safety totals cover all available SAC validation histories (including
rejected checkpoints), not just the necessarily-safe selected policies. No
independent-test generalization claim follows from these reused validation seeds.

## Statistical interpretation

`final_test_per_policy.csv`: 150 training-seed x test-seed rows, including program
failures if any. Baseline evidence is separately saved once for each realization.
`final_test_training_seed_summary.csv`: three rows; per-policy mean/median/sample
std/t95 CI/min/max/p5/p95, positive fraction, safety counters and margins.
`final_test_cross_training_seed_summary.csv`: statistics of three policy means,
with between-training-seed sample std and t95(df=2), not an iid CI over 150 rows.
`final_test_metric_summary_long.csv`: all numerical metrics in transparent long
format, including raw state/control and empirical disturbance amplification.

IAE/ISE/G use the same prespecified fixed B reference, not each trajectory's final
mean. G_emp is empirical amplification, not Hinf norm. Within-trajectory std is
not a standard error. Input TV/RMS uses actual input. Safety includes terminal
state (1001 samples); performance uses 1000 pre-step samples. Repeated before/
after Omega or QP counts are step counters, not independent event samples.
All ratios and economic gains are paired per test realization before averaging.
W exceedance is retained as diagnostic, not a final empirical failure gate.

Gaussian support is unbounded. Even zero violations over all independent test
trajectories are **empirical_only_under_ECC2019_stochastic_disturbance**, not a
formal Gaussian robust guarantee. ECC2019 remains qualitative literature only.

No commit/push or new parameter tuning is part of this workflow.
