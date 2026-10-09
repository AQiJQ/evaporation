# Independent 2D Conservative reference stage

This stage preserves the stage2 fixed-P2 failure results and frozen Strong
controller. Run from `C:\Users\cushy\PycharmProjects` using the existing CUDA
environment (baseline/set computations are CPU tasks; no SAC training here).

```powershell
$py = ".\evaporation\.venv-cuda\Scripts\python.exe"
& $py -m unittest evaporation.zanon2019_reference2d_tests -v
& $py -m evaporation.zanon2019_reference2d tests
& $py -m evaporation.zanon2019_reference2d prepare
& $py -m evaporation.zanon2019_reference2d search
& $py -m evaporation.zanon2019_reference2d baselines
# Only after a valid baseline is frozen:
& $py -m evaporation.zanon2019_reference2d oracle
```

`protocol_2d.json` is immutable. Central differences use 1e-4 in both
physical state dimensions, checked at half/double steps. P2's neighborhood is
Strong P2 +/- max(0.05, twice the linear compensation for delta X2=0.20),
intersected with physical/S reference bounds. No result-dependent expansion.

For each fixed X2, exact nonlinear monotone input-bound roots determine the
nearest P2 interval. A 1e-8 physical input inset is only a numerical interior
guard, not a changed constraint. Root tolerance is 1e-10. All original gates,
including Hinf, RPI support, Omega, nominal-W finite probes, verification QP
and 5% minimum authority, must pass. The unchanged online QP is additionally
initialized at both the candidate reference and the common Strong reset.
If this nearest input-bound point fails other gates, a fixed 401-point grid
and 1e-8 boundary refinement are used: this is nearest *detected* interval,
not a theorem excluding disconnected sub-grid feasible islands.

Implementation choice for reproduction: keep Strong normalization and A/B/K,
re-anchor only the affine equilibrium, reference and nominal-policy offset.
Omega vertices are translated coordinates of exactly the same physical set.
This is not gain/geometry redesign or nonlinear continuous-domain proof.

Baseline comparisons use common physical initial state [25.39, 50.125], ten
identical Gaussian literal-variance paths 420000..420009, 1000 steps at 1 s.
Read-only Strong stage2 traces are re-audited against the nonlinear plant,
terminal constraints and regenerated disturbances. Failed candidates retain
null costs/margins, not fabricated zeros. IAE uses each design's own reference;
state statistics and physical lower-bound margin remain directly comparable.

Selection is the first ascending delta with every path safe, every minimum
X2 margin improved by >1e-4, and mean J increased by >2e-4. Larger economic
penalty is never a ranking criterion. No SAC result enters selection.
If no candidate qualifies, oracle and training are blocked; do not weaken
Strong or change the frozen geometry. All stochastic safety is empirical.

The conditional finite oracle uses the previously declared 512 raw-action
profiles (10/5-second segments, deterministic seed 190019) and all ten paired
Conservative paths. It passes the real interior-anchor mapping/QP/plant and
saves Pareto witnesses. It is a finite best-found result, not a mathematical
upper bound. No SAC training is automatically started, even if the gate opens.
