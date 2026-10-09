# Primary Conservative boundary audit and complete finite search

The frozen Primary reference is read at FULL precision from stage3 selection.
It is not reselected, rounded, clipped or moved inward. Strong and all prior
results remain read-only. Outputs are exclusively in
`evaporation_safe_sac/economic_recovery_v1/conservative_mechanism_stage4_oracle_training/`.

From `C:\Users\cushy\PycharmProjects`:

```powershell
$py = ".\evaporation\.venv-cuda\Scripts\python.exe"
& $py -m unittest evaporation.zanon2019_stage4_tests -v
& $py -m evaporation.zanon2019_stage4 tests
& $py -m evaporation.zanon2019_stage4 prepare
& $py -m evaporation.zanon2019_stage4 boundary
& $py -m evaporation.zanon2019_stage4 oracle --workers 8
& $py -m evaporation.zanon2019_stage4 sensitivity
```

Preparation and completed audit/search outputs are immutable. The full oracle
has exactly 512 candidates, no early positive-witness stop. Parallel workers
evaluate independent paths of the same candidate; candidate order, RNG,
constant grid, 10/5-second profiles and sequential Pareto updates are unchanged.
The original elite economic ranking is the mean per-path percentage. New
reported economic headline percentages are ratios of mean absolute costs;
both quantities are saved, never confused.

Interrupted oracle execution can be restarted with the same command:
completed path metrics/trajectory hashes and regenerated action profiles are
checked. It is not a new search and never changes the registered budget.

The oracle uses NumPy CPU plant simulation and the existing CPU 2D QP;
using `.venv-cuda` does not move these operations onto the GPU. There is no
oracle `--device cuda` option. For local continuation on the current
16-logical-CPU machine, use `--workers 10`. Completed caches are reused:

```powershell
Set-Location "C:\Users\cushy\PycharmProjects"
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_stage4 oracle --workers 10
```

CUDA is supported for the later SAC training command below, only after the
complete oracle gate passes. No GPU speedup is claimed for the oracle.

Full per-path metrics and trajectories are retained, including failures.
Safe full numerical trajectories are losslessly compressed float64 NPZ
(not downsampled/rounded); failures retain partial evidence. This avoids
multi-gigabyte CSV duplication on the nearly full C drive. Undefined nominal
F200 margins in Omega mode are stored as indexed Z-only values, not zeros.

Boundary repeat audit: 20 in-process full checks plus 3 new Windows Python
processes each performing 3 checks, and an exact JSON roundtrip. Backend is
the unchanged NumPy float64 2D active-set QP. Default QP tol=1e-9 normalized,
reference tol=1e-8 normalized. Physical F200 scale is 100. The reference raw
physical margin (~1e-8) is SMALLER than either converted tolerance. Repeat
stability therefore does NOT imply generous margin or formal Gaussian safety.
Any repeat gate flip closes oracle/training without tolerance/reference rescue.

F200 tightened margin is defined on the **Z nominal input**. In Omega mode,
the safe polytope constrains total actual input, not the shadow nominal input.
The latter tightened-margin metric is NA for Omega steps. The diagnostic
comparison of actual F200 to nominal tight upper is stored separately and
negative values do NOT signify actual-input constraint violations. Bound-active
rates distinguish Z tightened and Omega actual robust input bounds. The
near-boundary tolerance 1e-6 physical reuses the existing occupancy diagnostic;
no hard safety constraint is changed. Mapped-candidate F200 QP corrections
are counted; raw actor commands are not labelled rejected merely because
the mapping limits their physical range.

Economics always reports all three dimensions, with absolute costs/savings:

1. `100*(J_cons-J_candidate)/J_cons`.
2. `100*(J_cons-J_candidate)/(J_cons-J_strong)`; NA for nonpositive penalty,
   never clipped at 100%.
3. `100*(J_strong-J_candidate)/J_strong`, and absolute `J_candidate-J_strong`.

The small denominator (~875.9) is explicitly retained. Finite search is a
best-found privileged offline result, NOT a mathematical economic bound.
No observation, safety set, mapping, reward, network or optimizer is altered.

Break-even counts completed transitions (1..1000). Recovery break-even is
the first cumulative saving vs Conservative reaching the FULL horizon
conservatism penalty; Strong outperformance is the first cumulative candidate
cost below the SAME path's cumulative Strong cost. They may differ, may be
transient and may be NA. No requirement of sustained crossing is invented.

## Local training — only AFTER the complete oracle gate

Audit/oracle never start training. Before training, inspect
`oracle_conservative_512_summary.json`:
`conservative_seed42_training_allowed` must be true. A candidate must pass
all five safety counters, have positive mean gain, nonnegative worst-path
gain and all-path TV/RMS ratios <= the reused 1.10 activity band. Maximum
economics and the control-qualified training witness are reported separately.

Only then, explicitly on this local Windows machine:

```powershell
& $py -m evaporation.zanon2019_stage4_train train --seed 42 --device cuda
```

This has exactly 100 episodes x 1000 steps. No other seed entry exists, no
silent CPU fallback, no overwrite/resume of interrupted training. Replay,
smoothness lambda, safety duals, network, LR and entropy are the frozen v2
implementation. The same Conservative baseline and common initial Strong B
are paired every episode. Strong rollouts are performance diagnostics only,
not the reward baseline. Validation uses only 420000..420009 every 5 episodes.

The development presentation policy is preregistered terminal episode100,
not the highest-economics checkpoint. All checkpoint/Pareto trade-offs remain
available. Continuation to 2027 is only recommended if terminal post-warmup
quality/safety/economic gates pass; no continuation is automatically launched.
314159 is not recommended before independently authorizing/validating 2027.

After complete training a frozen episode100 policy is paired again on 420000
for Fig2-style and cumulative Strong cost plots; true replay-objective return
and fixed economic learning curves are saved. Curves are neither shifted nor
forced upward or toward zero. No new 430000..430049 paths, commit or push.
