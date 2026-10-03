# ECC2019-motivated stochastic evaporation calibration

Source: `C:/Users/cushy/Desktop/Practical_Reinforcement_Learning_of_Stabilizing_Economic_MPC.pdf`, Numerical Example pp. 2261-2262; Fig. 2 p. 2263.

## Scope and interpretation

The paper calls the four printed sigma values *variances*. Literal variance is the main mode. Gaussian iid sampling, balanced-B initial state, 1 s step, 1000-step horizon, and seeds are **implementation choice for reproduction**; the distribution, temporal independence, initial state, sampling time and fixed evaluation realization are **not specified in the paper**. Sigma-as-std is sensitivity only.
The stochastic path is passed unchanged into the existing nonlinear plant as `[F1,X1,T1,T200]`; the same path is verified in the resulting records. The plant's exogenous quantities enter `EvaporatorModel.algebraic` and `derivative`, then RK4 `step`. ECC2019 prose says flow F2 while its variance subscript says F1; the project model uses F1. No Fig. 2 curve matching or disturbance rescaling was performed.

## Disturbance input calibration

| variable | target_mean | mean | target_variance | variance | std | min | max | p1 | p5 | p50 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 10 | 10.2249 | 2 | 2.75672 | 1.66034 | 7.68998 | 13.117 | 7.71347 | 7.80742 | 10.9313 | 12.7012 | 13.0339 |
| X1 | 5 | 4.74767 | 1 | 1.141 | 1.06818 | 2.77867 | 7.25701 | 2.85772 | 3.17394 | 4.77975 | 6.24155 | 7.05392 |
| T1 | 40 | 39.9032 | 8 | 5.32568 | 2.30774 | 34.9232 | 43.5683 | 35.264 | 36.627 | 39.7658 | 43.4831 | 43.5513 |
| T200 | 25 | 24.1488 | 5 | 4.09421 | 2.02342 | 20.8213 | 27.7382 | 20.8523 | 20.9762 | 24.2501 | 27.2283 | 27.6362 |

## State response about balanced reference B

| state | reference | mean | std | error_std | min | max | p1 | p5 | p50 | p95 | p99 | peak_deviation | IAE | ISE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X2 | 25.39 | 25.3568 | 0.0207457 | 0.0207457 | 25.319 | 25.39 | 25.3199 | 25.3235 | 25.3596 | 25.386 | 25.3892 | 0.0709642 | 0.497526 | 0.0229579 |
| P2 | 50.125 | 50.1299 | 0.00297508 | 0.00297508 | 50.125 | 50.1359 | 50.1252 | 50.1262 | 50.1291 | 50.1352 | 50.1357 | 0.0108692 | 0.0740954 | 0.000498775 |

The state and error standard deviations are numerically equal for a fixed reference, but both are reported to make the reference explicit. Smaller fluctuations than a visually inspected Fig. 2 do not establish superior disturbance rejection: controllers, realizations and initial conditions are not matched, and no exact statistics were digitized from that figure.

## Empirical disturbance amplification

Pooled G_X2_emp=0.0190016, G_P2_emp=0.00280077. Input denominator uses the literal-variance standard deviations `[sqrt(2),1,sqrt(8),sqrt(5)]` for all groups. These are finite-trajectory ratios, **not** an H-infinity norm, induced L2 gain, or formal bound.

## Control effort and economics

| group | J_econ_mean | P100_TV_mean | F200_TV_mean | P100_delta_RMS_mean | F200_delta_RMS_mean | P100_upper_bound_steps_mean | F200_upper_bound_steps_mean | P100_outer_10pct_steps_mean | F200_outer_10pct_steps_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| literal_variance_20seed | 96149.2 | 63.1108 | 13.9896 | 5.0127 | 1.11376 | 0 | 0 | 0 | 0 |

TV=0 is not evidence of good control without the corresponding bound occupancy. There is no matched ECC2019 RL-NMPC rollout, so neither stronger overall performance nor a direct economic gain against that paper can be claimed.

## Fixed-W coverage and safety

W exceedance: 15/15 steps; pooled rate=1. Worst sample: seed 420000, step 11, facet 0, utilization 9.00878, excess 0.00253171 normalized state units. Full sample is in `w_worst_case_sample.json`; per-step facet values are in `w_facet_per_step.csv`.
Observed physical-state violation steps=0; QP infeasible steps=0; Omega exit steps=0. See `per_seed_baseline_metrics.csv` for X2/P2/input, robust-region, Z, and violation magnitudes.
`Z exit != physical violation`; `W exceedance != physical violation`; `Omega exit != automatically physical violation`; `zero observed physical violation != formal guarantee`.
Certification status: `empirical_only_under_ECC2019_stochastic_disturbance`. Gaussian support is unbounded, and the original formal certificate applies only under its e0-in-Z / w-in-W assumptions (plus separate bounded finite-jump/minimum-dwell conditions for Experiment III). No all-Gaussian formal safety guarantee is claimed.

## Disturbance component ablation

| group | W_exceedance_rate_pooled | W_max_facet_utilization_overall | X2_std_mean | P2_std_mean | X2_IAE_mean | P2_IAE_mean | G_X2_emp_mean | G_P2_emp_mean | physical_state_violation_steps_total | Omega_exit_count_total | QP_infeasible_count_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 0.933333 | 15.8589 | 0.0246722 | 0.00395334 | 0.409374 | 0.0743004 | 0.0300452 | 0.00534922 | 0 | 0 | 0 |
| X1 | 0.466667 | 4.67197 | 0.0112435 | 0.00204945 | 0.164522 | 0.054371 | 0.0132055 | 0.00379382 | 0 | 0 | 0 |
| T1 | 0.333333 | 3.53015 | 0.00130945 | 0.000403425 | 0.0181364 | 0.0139726 | 0.00174114 | 0.00124306 | 0 | 0 | 0 |
| T200 | 0.866667 | 26.3113 | 0.000110892 | 0.00236439 | 0.00160848 | 0.0427874 | 0.000147752 | 0.00337337 | 0 | 0 | 0 |
| all_four | 1 | 40.5812 | 0.0207457 | 0.00297508 | 0.497526 | 0.0740954 | 0.0190016 | 0.00280077 | 0 | 0 | 0 |

Largest *isolated* W exceedance rate: `F1`. This is a one-at-a-time diagnostic, not an additive causal decomposition of the all-four case.

## Sigma interpretation sensitivity

| group | W_exceedance_rate_pooled | X2_std_mean | P2_std_mean | G_X2_emp_mean | G_P2_emp_mean | physical_state_violation_steps_total | Omega_exit_count_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| literal_variance_20seed | 1 | 0.0207457 | 0.00297508 | 0.0190016 | 0.00280077 | 0 | 0 |
| sigma_as_std_audit_5seed | 1 | 0.0300502 | 0.00336049 | 0.01369 | 0.00122295 | 0 | 0 |

## Decision

Experiment I is suitable as an **empirical stochastic training/evaluation environment** only if physical/QP/Omega abort behavior remains acceptable; it is **not suitable for claims of formal safety under paper-scale Gaussian disturbance** with frozen W. Do not start formal SAC training on the assumption that the existing W certificate covers these draws. No implementation bug was inferred solely from W exceedance; it can arise because the paper-scale exogenous uncertainty lies outside the fixed certified domain. Original controller/safety code remains unchanged.

Experiment II (single state jump) and Experiment III (unknown-time repeated jumps with D=20) remain separate and unchanged.
