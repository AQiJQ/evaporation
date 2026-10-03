# ECC 2019-motivated stochastic evaporation comparison

## Benchmark definition

Project simplified nonlinear evaporation plant, physical states X2/P2 and inputs P100/F200; stochastic exogenous F1/X1/T1/T200. ECC2019 omits model and cost equations and refers to earlier publications, so exact plant identity cannot be established from this PDF alone.
Implementation choice for reproduction: independent zero-mean Gaussian per step; reported paper variances are converted to standard deviations with square roots.
The paper does not specify the distribution, per-step independence, initial state, sampling time, exact seed, or exact gain formula.

## Our paired zero-residual and residual SAC results

Evaluated 1 paired seeds. Mean economic improvement vs *our own* zero-residual robust baseline: 5.930220845639041%.
Both controllers use identical initial state and saved disturbance sequence for each seed.

## Constraint/safety result

See per_seed_metrics.csv for X2/P2/input, Omega/Z/QP and strict normalized-facet W exceedance counts. An out-of-W rollout is not formally certified by the fixed W/Omega/Gm assumptions.

## Economic and control-quality result

See aggregate_summary.csv for J_econ, stage costs, X2/P2 IAE/ISE, P100/F200 TV, RMS moves, sign changes and robust-bound occupancy.

## Comparison scope with ECC 2019

Under a stochastic operating-uncertainty benchmark motivated by Zanon et al. (ECC 2019), the proposed controller is evaluated with explicit physical and robust-safety auditing.
This is not an implementation of their RL-NMPC, whose state constraints are softened. The paper's 14%/12% gains are against naive and nominal-economic NMPC, not our zero-residual baseline. No direct economic or formal-safety superiority claim is made.

## Other experiments

Experiment II (single finite jump) and Experiment III (unknown-time repeated jumps with dwell20 Gm/Bj) remain separate and unchanged.
