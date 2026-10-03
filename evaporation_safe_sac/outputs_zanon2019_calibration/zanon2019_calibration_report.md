# ECC2019-motivated stochastic evaporation calibration

Source: `C:/Users/cushy/Desktop/Practical_Reinforcement_Learning_of_Stabilizing_Economic_MPC.pdf`, Numerical Example pp. 2261-2262; Fig. 2 p. 2263.

## Scope and interpretation

The paper calls the four printed sigma values *variances*. Literal variance is the main mode. Gaussian iid sampling, balanced-B initial state, 1 s step, 1000-step horizon, and seeds are **implementation choice for reproduction**; the distribution, temporal independence, initial state, sampling time and fixed evaluation realization are **not specified in the paper**. Sigma-as-std is sensitivity only.
The stochastic path is passed unchanged into the existing nonlinear plant as `[F1,X1,T1,T200]`; the same path is verified in the resulting records. The plant's exogenous quantities enter `EvaporatorModel.algebraic` and `derivative`, then RK4 `step`. ECC2019 prose says flow F2 while its variance subscript says F1; the project model uses F1. No Fig. 2 curve matching or disturbance rescaling was performed.

## Disturbance input calibration

| variable | target_mean | mean | target_variance | variance | std | min | max | p1 | p5 | p50 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 10 | 9.9864 | 2 | 1.96743 | 1.40265 | 4.58105 | 15.6925 | 6.74988 | 7.66785 | 9.98845 | 12.2843 | 13.2641 |
| X1 | 5 | 5.00115 | 1 | 1.00087 | 1.00043 | 0.909318 | 9.06533 | 2.65705 | 3.3676 | 4.99669 | 6.65865 | 7.31935 |
| T1 | 40 | 39.9769 | 8 | 7.91388 | 2.81316 | 28.4506 | 50.4123 | 33.3554 | 35.3345 | 39.968 | 44.641 | 46.4939 |
| T200 | 25 | 24.9776 | 5 | 5.06849 | 2.25133 | 16.1916 | 33.3281 | 19.7801 | 21.2821 | 24.9585 | 28.6833 | 30.2447 |

## State response about balanced reference B

| state | reference | mean | std | error_std | min | max | p1 | p5 | p50 | p95 | p99 | peak_deviation | IAE | ISE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| X2 | 25.39 | 25.3831 | 0.0440999 | 0.0440999 | 25.2124 | 25.5531 | 25.2758 | 25.3095 | 25.3833 | 25.4553 | 25.4843 | 0.177629 | 707.388 | 39.8403 |
| P2 | 50.125 | 50.1626 | 0.0687067 | 0.0687067 | 49.9988 | 50.4491 | 50.0368 | 50.0628 | 50.1555 | 50.2846 | 50.368 | 0.324051 | 1168.4 | 122.737 |

The state and error standard deviations are numerically equal for a fixed reference, but both are reported to make the reference explicit. Smaller fluctuations than a visually inspected Fig. 2 do not establish superior disturbance rejection: controllers, realizations and initial conditions are not matched, and no exact statistics were digitized from that figure.

## Empirical disturbance amplification

Pooled G_X2_emp=0.0223502, G_P2_emp=0.039229. Input denominator uses the literal-variance standard deviations `[sqrt(2),1,sqrt(8),sqrt(5)]` for all groups. These are finite-trajectory ratios, **not** an H-infinity norm, induced L2 gain, or formal bound.

## Control effort and economics

| group | J_econ_mean | P100_TV_mean | F200_TV_mean | P100_delta_RMS_mean | F200_delta_RMS_mean | P100_upper_bound_steps_mean | F200_upper_bound_steps_mean | P100_outer_10pct_steps_mean | F200_outer_10pct_steps_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| literal_variance_20seed | 6.22258e+06 | 3845.84 | 824.31 | 4.81963 | 1.03332 | 0 | 0 | 0 | 0 |

TV=0 is not evidence of good control without the corresponding bound occupancy. There is no matched ECC2019 RL-NMPC rollout, so neither stronger overall performance nor a direct economic gain against that paper can be claimed.

## Fixed-W coverage and safety

W exceedance: 19661/20000 steps; pooled rate=0.98305. Worst sample: seed 420014, step 716, facet 0, utilization 20.9529, excess 0.00630746 normalized state units. Full sample is in `w_worst_case_sample.json`; per-step facet values are in `w_facet_per_step.csv`.
Observed physical-state violation steps=0; QP infeasible steps=0; Omega exit steps=0. See `per_seed_baseline_metrics.csv` for X2/P2/input, robust-region, Z, and violation magnitudes.
`Z exit != physical violation`; `W exceedance != physical violation`; `Omega exit != automatically physical violation`; `zero observed physical violation != formal guarantee`.
Certification status: `empirical_only_under_ECC2019_stochastic_disturbance`. Gaussian support is unbounded, and the original formal certificate applies only under its e0-in-Z / w-in-W assumptions (plus separate bounded finite-jump/minimum-dwell conditions for Experiment III). No all-Gaussian formal safety guarantee is claimed.

Legacy cross-product membership flags 17706/20000 while unit-normal facet membership flags 19661/20000; 1955 samples are outside by the strict test but inside by the legacy tolerance. This is a **diagnostic undercount**, not a change to the frozen controller or its sets. See `legacy_vs_facet_w_audit.csv`.

Largest relative W facet utilization is 64.5816 at seed 420004, step 762; this differs from the worst *absolute* facet excess sample. See `w_worst_relative_utilization_sample.json`.

## Disturbance component ablation

| group | W_exceedance_rate_pooled | W_max_facet_utilization_overall | X2_std_mean | P2_std_mean | X2_IAE_mean | P2_IAE_mean | G_X2_emp_mean | G_P2_emp_mean | physical_state_violation_steps_total | Omega_exit_count_total | QP_infeasible_count_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 0.8695 | 31.8979 | 0.040169 | 0.0398784 | 32.6873 | 51.2779 | 0.0414185 | 0.0617739 | 0 | 0 | 0 |
| X1 | 0.5955 | 8.41537 | 0.0161134 | 0.0297654 | 14.4788 | 43.8461 | 0.0179425 | 0.0519036 | 0 | 0 | 0 |
| T1 | 0.3502 | 7.59106 | 0.00581066 | 0.0269929 | 9.24663 | 49.8431 | 0.0110013 | 0.0577022 | 0 | 0 | 0 |
| T200 | 0.8815 | 51.1658 | 0.00523901 | 0.0257786 | 8.57251 | 43.0915 | 0.00990406 | 0.0493297 | 0 | 0 | 0 |
| all_four | 0.9828 | 64.5816 | 0.043027 | 0.0452878 | 34.7899 | 53.1977 | 0.0219463 | 0.0314343 | 0 | 0 | 0 |

Largest *isolated* W exceedance rate: `T200`. This is a one-at-a-time diagnostic, not an additive causal decomposition of the all-four case.

## Sigma interpretation sensitivity

Matched-seed comparison (same five disturbance seeds in both interpretations; no SAC training):

| mode | W_exceedance_rate_mean | X2_std_mean | P2_std_mean | G_X2_emp_mean | G_P2_emp_mean | P100_TV_mean | F200_TV_mean | physical_state_violation_steps_mean | Omega_exit_count_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| literal_variance_matched | 0.9834 | 0.0442079 | 0.0455281 | 0.0224244 | 0.0306137 | 3837.11 | 824.836 | 0 | 0 |
| sigma_as_std_sensitivity | 0.994 | 0.0607316 | 0.0682285 | 0.0153529 | 0.021698 | 5254.46 | 1135.91 | 0 | 0 |

The sigma-as-std input means and variances are saved in `sigma_as_std_disturbance_statistics.csv`. The older summary table below uses 20 vs 5 seeds and should not be treated as a paired sensitivity estimate.

| group | W_exceedance_rate_pooled | X2_std_mean | P2_std_mean | G_X2_emp_mean | G_P2_emp_mean | physical_state_violation_steps_total | Omega_exit_count_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| literal_variance_20seed | 0.98305 | 0.0432937 | 0.0461973 | 0.0222938 | 0.0343703 | 0 | 0 |
| sigma_as_std_audit_5seed | 0.994 | 0.0607316 | 0.0682285 | 0.0153529 | 0.021698 | 0 | 0 |

## Decision

Experiment I is suitable as an **empirical stochastic training/evaluation environment** only if physical/QP/Omega abort behavior remains acceptable; it is **not suitable for claims of formal safety under paper-scale Gaussian disturbance** with frozen W. Do not start formal SAC training on the assumption that the existing W certificate covers these draws. No implementation bug was inferred solely from W exceedance; it can arise because the paper-scale exogenous uncertainty lies outside the fixed certified domain. Original controller/safety code remains unchanged.

Experiment II (single state jump) and Experiment III (unknown-time repeated jumps with D=20) remain separate and unchanged.
