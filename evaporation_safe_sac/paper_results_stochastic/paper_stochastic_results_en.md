# Stochastic Experiment I — validation results

## 1. Stochastic settings

The paper reports variances [2,1,8,5] for [F1,X1,T1,T200]. We use their square
roots as standard deviations. Gaussian iid per-step draws, 1 s sampling,
1000 steps and initial B are implementation choices for reproduction, not
details specified in the paper. These are reused validation/development seeds 420000–420009, not
independent held-out test seeds or ten independently trained policies. Both
policies have training initialization seed 42. The reserved 430000–430049 have
no execution evidence in audited manifests/fixed evaluations/trajectory filenames;
no new evaluation was run. Confidence intervals describe variability across the
ten paired realizations conditional on the selected policy; adaptive tuning and
selection prevent interpreting them as unbiased final-test inference.


## 2. Robust baseline

The zero-residual H∞/RPI/QP controller provides strong empirical disturbance
attenuation under the tested stochastic conditions. Its X2 mean is
25.384924, mean within-seed std
0.043027, global minimum margin
0.212371. No observed state/input violation occurs.

## 3. Residual SAC

Candidate A is recommended provisionally for the validation main display: it
uses less X2 performance margin than B, despite smaller economic gain and a
larger F200 TV. B is a supplementary Pareto point, not a pure authority ablation:
reward calibration and authority both differ. Neither dominates the baseline.

## 4. Internally paired quantitative analysis

|Method|J_econ|Gain %|Worst gain %|X2 mean|Mean within-seed std|Global X2 range|X2 IAE ratio (B)|P2 IAE ratio (B)|P100 TV ratio|F200 TV ratio|
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
|baseline|6217912.952|0.000000|0.000000|25.384924|0.043027|25.212371–25.539580|1.000000|1.000000|1.000000|1.000000|
|A_alpha010|6216471.785|0.023161|0.009478|25.369345|0.046562|25.178676–25.530982|1.181462|0.989078|0.825742|1.315386|
|B_alpha020_rewardcal|6215268.649|0.042468|0.017342|25.350740|0.049341|25.160545–25.556139|1.538162|0.938511|1.048697|0.985507|

A mean gain 0.023161% (descriptive paired t95 interval 0.017547–0.028775%). X2 B-reference IAE +18.15%, ISE +39.01%, P100 TV -17.43%, F200 TV +31.54%.

J_econ is the sum of saved next-state/final-input stage costs;
shaping and relaxed-state costs are excluded. Gain is the mean of per-seed
100*(J_base-J_SAC)/J_base, not the ratio of aggregate means. IAE/ISE and ratios in
the primary table use the prespecified balanced reference B=[25.39,50.125].
The additional *_nominal columns use [25,49.743]. Moving nearer to X2=25 may
reduce nominal-reference error while increasing B-reference error; neither is
hidden or substituted post hoc. Performance uses 1000 pre-step states; safety
uses 1001 states including the reconstructed terminal state. Input TV/RMS use
999 successive saved-input differences. X2 std is the mean of ten within-seed
population stds; extrema in the text span all realizations. CSV mean/min/max
fields distinguish per-seed averaging from global extrema. Margin quantiles are
per-seed statistics, not pooled confidence bounds. Empirical G is the L2 norm of
state deviation from B divided by the L2 norm of four exogenous deviations
normalized by sqrt(variance); it is empirical disturbance amplification, not
an H-infinity norm. Raw state statistics do not depend on the chosen reference.

Residual SAC trades part of the robust baseline's disturbance-rejection margin
for a small positive economic gain, retaining zero observed physical violations.
Do not claim that the small X2 band is obtained without any control-activity
cost: A's F200 TV increases. No external matched economic comparison is made.

## 5. Qualitative ECC2019 comparison

See ecc2019_literature_comparison.md for source pages, original wording, caption
and citation text. ECC2019 reports rare/small X2 constraint violations with
relaxed state bounds; no such violation occurs in our displayed evaluations.
Our band appears narrow on a similar axis, but external numerical superiority
is not established because the protocols/realizations are not matched.

## 6. Safety claim boundary

Gaussian support is unbounded. Mean W exceedance is
98.28% for A. This is empirical
stochastic evaluation, not formal Gaussian robust certification. Zero observed
physical/QP/Omega/robust-region anomalies do not certify the continuous domain.
Finite-jump Omega/Gm/Bj and conditional dwell-time certificates remain unchanged
for Experiments II and III, not transplanted into Experiment I.

## 7. Summary and readiness

Improved internal economics and preserved empirical safety coexist with an
explicit performance trade-off. These artifacts support a thesis development
results section, not an unbiased final test claim. Lock the candidate and
analysis protocol before an independently reserved paired test; no such test or
training is initiated here. Multiple training seeds would be needed for claims
about learning reproducibility. Stop ECC2019 fitting and further blind
reward/authority/critic tuning. Preserve finite-jump experiments separately.
