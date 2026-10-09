# ECC2019 development comparison

ECC controller label: ECC2019_reproduction_with_documented_assumptions
Trained ECC available: False. Final50seed test: NOT RUN.
This is development only, not final paper statistics. No SAC retraining or test-based selection.
Archives independently checked against identical disturbance arrays, initial B, existing plant and stage cost.
Main IAE/ISE reference=[25,49.743]; B-relative diagnostics separate. Safety includes terminal state.
If ECC is unavailable, economic gaps and better-safety conclusions are unavailable, NOT zero.
No inference from missing ECC about comparable economics or rare violations. No forced14%/12% gains.
Fig2 qualitative consistency requires the learned ECC trajectory; figures alone do not certify reproduction.
Do not proceed to50final seeds until learned ECC solver/parameter/qualitative review and one prelocked SAC candidate are approved.

| Method | J_econ mean | X2 mean | X2 std (mean within-seed) | X2 global range | X2 violation count |
|---|---:|---:|---:|---|---:|
| ecc2019_reproduction | unavailable | unavailable | unavailable | unavailable | unavailable |
| robust_zero_residual | 6217912.951733 | 25.384924 | 0.043027 | 25.212371 to 25.539580 | 0 |
| proposed_alpha010_ep100 | 6216471.784833 | 25.369345 | 0.046562 | 25.178676 to 25.530982 | 0 |
| proposed_alpha020_rewardcal_ep100 | 6215268.649165 | 25.350740 | 0.049341 | 25.160545 to 25.556139 | 0 |

SAC vs robust baseline is a performance-margin/economics trade-off, not a requirement to dominate every metric.
External success requires comparable J and empirical safety/rejection advantages over a validated TRAINED ECC controller.
Solver-smoke is NAIVE, not RL-tuned; it is excluded from every comparison above.