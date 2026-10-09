# ECC2019 episode-5 failure diagnosis

No SAC training or design changes. All scales are diagnostic-only autonomous closed loops.

Scope: empirical_only_under_ECC2019_stochastic_disturbance. Episode 5: unsafe_and_performance_degraded_checkpoint.

## First failure and cascade

```json
{
  "first": {
    "seed": 420000,
    "first_physical_transition_step": 12,
    "first_Omega_affected_transition_step": 12,
    "first_Omega_before_step": 13,
    "first_QP_infeasible_step": 13,
    "before_first_physical": {
      "physical": 0,
      "Omega_affected_steps": 0,
      "QP_infeasible": 0
    },
    "at_first_physical": {
      "physical": 1,
      "Omega_affected_steps": 1,
      "QP_infeasible": 0
    },
    "strictly_after_first_physical": {
      "physical": 26,
      "Omega_affected_steps": 48,
      "QP_infeasible": 27
    },
    "first_failure_type": "model/disturbance-domain mismatch induced safety failure",
    "first_failure_evidence": {
      "step": 12,
      "physical_before_safe": true,
      "physical_after_violation": true,
      "Omega_before": true,
      "Omega_after": false,
      "Z_before": false,
      "QP_feasible": true,
      "QP_modification": 0.0,
      "safety_mode": "Omega_safe_one_step_QP",
      "W_exceedance": true,
      "W_max_facet_excess": 0.0008020597085476507,
      "W_max_facet_utilization": 3.5372247456955046,
      "certified_W_next_Omega_max_excess": -2.4178869144627496e-05,
      "actual_next_Omega_max_excess": 0.0007775542110901694,
      "X2_minus_25": 0.03389212801285879,
      "state_0": 25.03389212801286,
      "state_1": 50.07439017289667,
      "control_0": 157.62058787298596,
      "control_1": 219.3866533151498,
      "actual_next_0": 24.988336686833648,
      "actual_next_1": 50.06821885857201,
      "w_actual_0": -0.0011180756616398294,
      "w_actual_1": -1.5543402755334532e-05,
      "raw_action_0": -0.9941441416740417,
      "raw_action_1": 0.07082480937242508,
      "predicted_nominal_next_0": 25.005107821758244,
      "predicted_nominal_next_1": 50.06852972662712,
      "predicted_W_next_min_0": 25.00036268303717,
      "predicted_W_next_min_1": 50.06832397801609,
      "predicted_W_next_max_0": 25.009126590199482,
      "predicted_W_next_max_1": 50.06893693781723,
      "baseline_input_at_same_state_0": 233.65538828079238,
      "baseline_input_at_same_state_1": 224.50642560788918,
      "alpha": 1.0,
      "seed": 420000,
      "actor_unscaled_0": -0.9941441416740417,
      "actor_unscaled_1": 0.07082480937242508,
      "residual_0": -0.7603480040780642,
      "residual_1": -0.05119772292739388,
      "applied_residual_0": -0.7603480040780641,
      "applied_residual_1": -0.05119772292739378,
      "disturbance_0": 10.980917736364011,
      "disturbance_1": 4.8137246690121005,
      "disturbance_2": 39.842974130452305,
      "disturbance_3": 24.20774347142913
    }
  },
  "cascades": [
    {
      "seed": 420000,
      "first_physical_transition_step": 12,
      "first_Omega_affected_transition_step": 12,
      "first_Omega_before_step": 13,
      "first_QP_infeasible_step": 13,
      "before_first_physical": {
        "physical": 0,
        "Omega_affected_steps": 0,
        "QP_infeasible": 0
      },
      "at_first_physical": {
        "physical": 1,
        "Omega_affected_steps": 1,
        "QP_infeasible": 0
      },
      "strictly_after_first_physical": {
        "physical": 26,
        "Omega_affected_steps": 48,
        "QP_infeasible": 27
      },
      "first_failure_type": "model/disturbance-domain mismatch induced safety failure",
      "first_failure_evidence": {
        "step": 12,
        "physical_before_safe": true,
        "physical_after_violation": true,
        "Omega_before": true,
        "Omega_after": false,
        "Z_before": false,
        "QP_feasible": true,
        "QP_modification": 0.0,
        "safety_mode": "Omega_safe_one_step_QP",
        "W_exceedance": true,
        "W_max_facet_excess": 0.0008020597085476507,
        "W_max_facet_utilization": 3.5372247456955046,
        "certified_W_next_Omega_max_excess": -2.4178869144627496e-05,
        "actual_next_Omega_max_excess": 0.0007775542110901694,
        "X2_minus_25": 0.03389212801285879,
        "state_0": 25.03389212801286,
        "state_1": 50.07439017289667,
        "control_0": 157.62058787298596,
        "control_1": 219.3866533151498,
        "actual_next_0": 24.988336686833648,
        "actual_next_1": 50.06821885857201,
        "w_actual_0": -0.0011180756616398294,
        "w_actual_1": -1.5543402755334532e-05,
        "raw_action_0": -0.9941441416740417,
        "raw_action_1": 0.07082480937242508,
        "predicted_nominal_next_0": 25.005107821758244,
        "predicted_nominal_next_1": 50.06852972662712,
        "predicted_W_next_min_0": 25.00036268303717,
        "predicted_W_next_min_1": 50.06832397801609,
        "predicted_W_next_max_0": 25.009126590199482,
        "predicted_W_next_max_1": 50.06893693781723,
        "baseline_input_at_same_state_0": 233.65538828079238,
        "baseline_input_at_same_state_1": 224.50642560788918,
        "alpha": 1.0,
        "seed": 420000,
        "actor_unscaled_0": -0.9941441416740417,
        "actor_unscaled_1": 0.07082480937242508,
        "residual_0": -0.7603480040780642,
        "residual_1": -0.05119772292739388,
        "applied_residual_0": -0.7603480040780641,
        "applied_residual_1": -0.05119772292739378,
        "disturbance_0": 10.980917736364011,
        "disturbance_1": 4.8137246690121005,
        "disturbance_2": 39.842974130452305,
        "disturbance_3": 24.20774347142913
      }
    },
    {
      "seed": 420001,
      "first_physical_transition_step": 15,
      "first_Omega_affected_transition_step": 15,
      "first_Omega_before_step": 16,
      "first_QP_infeasible_step": 16,
      "before_first_physical": {
        "physical": 0,
        "Omega_affected_steps": 0,
        "QP_infeasible": 0
      },
      "at_first_physical": {
        "physical": 1,
        "Omega_affected_steps": 1,
        "QP_infeasible": 0
      },
      "strictly_after_first_physical": {
        "physical": 27,
        "Omega_affected_steps": 51,
        "QP_infeasible": 28
      },
      "first_failure_type": "model/disturbance-domain mismatch induced safety failure",
      "first_failure_evidence": {
        "step": 15,
        "physical_before_safe": true,
        "physical_after_violation": true,
        "Omega_before": true,
        "Omega_after": false,
        "Z_before": false,
        "QP_feasible": true,
        "QP_modification": 0.0,
        "safety_mode": "Omega_safe_one_step_QP",
        "W_exceedance": true,
        "W_max_facet_excess": 0.0021421659152858806,
        "W_max_facet_utilization": 7.776498447342081,
        "certified_W_next_Omega_max_excess": -0.0009588869434459593,
        "actual_next_Omega_max_excess": 0.0011832025165413908,
        "X2_minus_25": 0.045851299191284056,
        "state_0": 25.045851299191284,
        "state_1": 50.04266087888789,
        "control_0": 160.09247085316687,
        "control_1": 215.17874413612478,
        "actual_next_0": 24.98225196225188,
        "actual_next_1": 50.037082067076156,
        "w_actual_0": -0.0024584320413923826,
        "w_actual_1": -1.290597293096031e-05,
        "raw_action_0": -0.9481236934661865,
        "raw_action_1": -0.031067099422216415,
        "predicted_nominal_next_0": 25.019128442872766,
        "predicted_nominal_next_1": 50.03734018653478,
        "predicted_W_next_min_0": 25.01438330415169,
        "predicted_W_next_min_1": 50.03713443792375,
        "predicted_W_next_max_0": 25.023147211314004,
        "predicted_W_next_max_1": 50.037747397724885,
        "baseline_input_at_same_state_0": 233.65538828079238,
        "baseline_input_at_same_state_1": 221.6640765479363,
        "alpha": 1.0,
        "seed": 420001,
        "actor_unscaled_0": -0.9481236934661865,
        "actor_unscaled_1": -0.031067099422216415,
        "residual_0": -0.7356291742762551,
        "residual_1": -0.0648533241181151,
        "applied_residual_0": -0.7356291742762551,
        "applied_residual_1": -0.06485332411811516,
        "disturbance_0": 12.187560991555964,
        "disturbance_1": 4.720217091580435,
        "disturbance_2": 39.83423941035709,
        "disturbance_3": 23.90416269147488
      }
    },
    {
      "seed": 420002,
      "first_physical_transition_step": 50,
      "first_Omega_affected_transition_step": 50,
      "first_Omega_before_step": 51,
      "first_QP_infeasible_step": 51,
      "before_first_physical": {
        "physical": 0,
        "Omega_affected_steps": 0,
        "QP_infeasible": 0
      },
      "at_first_physical": {
        "physical": 1,
        "Omega_affected_steps": 1,
        "QP_infeasible": 0
      },
      "strictly_after_first_physical": {
        "physical": 28,
        "Omega_affected_steps": 50,
        "QP_infeasible": 29
      },
      "first_failure_type": "model/disturbance-domain mismatch induced safety failure",
      "first_failure_evidence": {
        "step": 50,
        "physical_before_safe": true,
        "physical_after_violation": true,
        "Omega_before": true,
        "Omega_after": false,
        "Z_before": false,
        "QP_feasible": true,
        "QP_modification": 0.0,
        "safety_mode": "Omega_safe_one_step_QP",
        "W_exceedance": true,
        "W_max_facet_excess": 0.002154327252627264,
        "W_max_facet_utilization": 8.210033609446013,
        "certified_W_next_Omega_max_excess": -0.00013309971081380823,
        "actual_next_Omega_max_excess": 0.0020228734734194544,
        "X2_minus_25": 0.0189539393598821,
        "state_0": 25.018953939359882,
        "state_1": 50.03436185438205,
        "control_0": 177.86525068870102,
        "control_1": 225.27559355623998,
        "actual_next_0": 24.96965689789871,
        "actual_next_1": 50.0333198999651,
        "w_actual_0": -0.002472315765638295,
        "w_actual_1": 8.566321362334962e-05,
        "raw_action_0": -0.9576784372329712,
        "raw_action_1": 0.17044003307819366,
        "predicted_nominal_next_0": 25.006741634383282,
        "predicted_nominal_next_1": 50.03160663569263,
        "predicted_W_next_min_0": 25.00199649566221,
        "predicted_W_next_min_1": 50.0314008870816,
        "predicted_W_next_max_0": 25.01076040282452,
        "predicted_W_next_max_1": 50.03201384688274,
        "baseline_input_at_same_state_0": 233.65538828079238,
        "baseline_input_at_same_state_1": 218.27618329720391,
        "alpha": 1.0,
        "seed": 420002,
        "actor_unscaled_0": -0.9576784372329712,
        "actor_unscaled_1": 0.17044003307819366,
        "residual_0": -0.5579013759209136,
        "residual_1": 0.06999410259036054,
        "applied_residual_0": -0.5579013759209135,
        "applied_residual_1": 0.06999410259036057,
        "disturbance_0": 11.768173351238644,
        "disturbance_1": 3.899234115588442,
        "disturbance_2": 41.85721297246329,
        "disturbance_3": 26.835471905819947
      }
    }
  ]
}
```

## Residual scale frontier

|alpha|economics %|X2 IAE|P2 IAE|P100 TV|F200 TV|G X2|physical|QP|Omega|
|---|---|---|---|---|---|---|---|---|---|
|0|0.000000|1.0000|1.0000|1.0000|1.0000|1.0000|0|0|0|
|0.05|0.004119|1.0091|0.9846|0.9634|0.9789|1.0096|0|0|0|
|0.1|0.008299|1.0303|0.9699|0.9312|0.9707|1.0312|0|0|0|
|0.2|0.017998|1.1090|0.9432|0.8801|0.9821|1.1098|0|0|0|
|0.3|0.031205|1.2596|0.9187|0.8627|1.0509|1.2508|0|0|0|
|0.4|0.047300|1.4731|0.9003|0.8856|1.1671|1.4483|0|0|0|
|0.5|0.074669|1.7810|0.8986|0.9394|1.3740|1.7288|0|0|0|
|0.6|0.097723|2.2198|0.9124|0.9902|1.5676|2.1315|0|0|0|
|0.75|0.109058|2.6866|1.0646|1.2006|2.0016|2.6220|0|0|0|
|1|0.205184|4.7430|1.3386|1.6323|2.6687|4.6278|84|84|152|

Empirical-safe positive-economic scales on these paths: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75].
Level-C scales: [0.05, 0.1]. Representative audited scale only: 0.1.

## Reward source and scope

All components are recomputed by current train.run_episode. Its evaluation return includes RPI event/excess; stochastic SAC replay excludes these two, which remain separately audited. Component CSVs show signed contributions and magnitude ratios per seed, and per-step reward reconciliation. Physical-margin buffering penalty is not present; violation penalties activate only after violation. No new penalty is invented.

## Interpretation limitations

This is one early checkpoint and three fixed realizations, not a convergence result or proof about all Gaussian disturbances. Cascade counts are temporal associations, not proof that every later violation would disappear after repairing the first. Intermediate scales change the state fed to the frozen actor; no teacher forcing is used. A raw negative actor component need not imply a negative physical residual under the interior-anchor mapping; inspect actor and physical-residual sign statistics separately. Mean absolute economic reward versus penalty is not alone sufficient to prove reward dominance: examine incremental policy-vs-baseline reward differences as well.

## Files

residual_scale_sweep.csv, residual_scale_per_seed.csv, first_failure_trace.csv, cascade_analysis.json, reward_component_audit.csv, reward_components_per_step.csv, X2_margin_statistics.csv, actor_action_statistics.csv, diagnosis_results.json. All requested plots and full per-alpha/seed transition traces are saved alongside this report.

## Aggregated evidence and interpretation

Frontier classification: A. Positive-economic, empirically safe scales: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]; Level-C scales: [0.05, 0.1].
Episode-5 replay reward mean change versus baseline: -4.32149119. A negative value means the frozen actual replay reward does not rank this unsafe policy above baseline on these paths. Economic component magnitude alone must not be used to claim why SAC accepted it.

### Reward components (three-seed average per step, signed)

|component|baseline|alpha=1|frontier representative|
|---|---|---|---|
|economic_reward|0.017986006|0.081911311|0.02056539|
|projection_penalty|0|0|0|
|mapping_penalty|0|0|0|
|move_penalty|-2.6499872e-34|-0.00016451153|-9.0777994e-07|
|state_recovery_penalty|-0.00012729859|-0.0011405253|-0.00012780806|
|p100_move_penalty|-0.25065867|-1.6111403|-0.22125389|
|f200_move_penalty|-0.008878665|-0.082984493|-0.0085742796|
|saturation_penalty|0|-0.0096080913|-3.0696584e-08|
|state_violation_penalty|0|-2.8|0|
|input_violation_penalty|0|0|0|
|qp_infeasible_penalty|0|-0.14|0|
|state_excess_penalty|0|-4.3235344e-05|0|
|input_excess_penalty|0|0|0|
|actual_training_replay_reward_same_transition|-0.24167862|-4.5631698|-0.20939152|

### X2 physical margins (post-state, includes final transition)

|label|alpha|mean|min|p1|p5|fraction<0.05|fraction<0.10|fraction<0.20|
|---|---|---|---|---|---|---|---|---|
|baseline|0|0.381622|0.212371|0.274393|0.308239|0.0000%|0.0000%|0.0000%|
|episode5_actor|1|0.225497|-0.0451625|-0.0177505|0.0120398|12.3333%|21.5333%|40.9667%|
|frontier_representative|0.1|0.376277|0.204515|0.270206|0.302145|0.0000%|0.0000%|0.0000%|

### Actor signs and saturation

```json
[
  {
    "dimension": "P100",
    "kind": "raw_actor_action",
    "mean": -0.3116572895014542,
    "std": 0.5684647231515313,
    "min": -0.9945880770683289,
    "max": 0.5953409075737,
    "p1": -0.9929037988185883,
    "p5": -0.9902556002140045,
    "fraction_abs_gt_0p9": 0.2843333333333333,
    "negative_fraction": 0.6533333333333333,
    "positive_fraction": 0.3466666666666667,
    "raw_nonzero_sign_disagreement_with_requested_residual": 0.11733333333333333
  },
  {
    "dimension": "P100",
    "kind": "requested_residual_normalized",
    "mean": -0.21252628316707667,
    "std": 0.3014430808021511,
    "min": -0.7716799094050657,
    "max": 0.1306101649871927,
    "p1": -0.7698261431945138,
    "p5": -0.7667457632342121,
    "fraction_abs_gt_0p9": 0.0,
    "negative_fraction": 0.543,
    "positive_fraction": 0.299,
    "raw_nonzero_sign_disagreement_with_requested_residual": 0.11733333333333333
  },
  {
    "dimension": "F200",
    "kind": "raw_actor_action",
    "mean": 0.022487710555181062,
    "std": 0.1311590213015849,
    "min": -0.2975117564201355,
    "max": 0.5477275848388672,
    "p1": -0.2138655613362789,
    "p5": -0.15642673522233963,
    "fraction_abs_gt_0p9": 0.0,
    "negative_fraction": 0.4683333333333333,
    "positive_fraction": 0.5316666666666666,
    "raw_nonzero_sign_disagreement_with_requested_residual": 0.199
  },
  {
    "dimension": "F200",
    "kind": "requested_residual_normalized",
    "mean": 0.014661725316731723,
    "std": 0.04819165539323381,
    "min": -0.17222803016781152,
    "max": 0.18385769764844678,
    "p1": -0.10646226433429097,
    "p5": -0.05457015617651911,
    "fraction_abs_gt_0p9": 0.0,
    "negative_fraction": 0.37666666666666665,
    "positive_fraction": 0.5953333333333334,
    "raw_nonzero_sign_disagreement_with_requested_residual": 0.199
  }
]
```

### QP infeasibility context

```json
[
  {
    "seed": 420000,
    "QP_infeasible_total": 27,
    "QP_infeasible_with_Omega_before_false": 27,
    "QP_infeasible_while_Omega_before_true": 0,
    "fallback_mode_counts": {
      "outside_certified_domain": 27
    }
  },
  {
    "seed": 420001,
    "QP_infeasible_total": 28,
    "QP_infeasible_with_Omega_before_false": 28,
    "QP_infeasible_while_Omega_before_true": 0,
    "fallback_mode_counts": {
      "outside_certified_domain": 28
    }
  },
  {
    "seed": 420002,
    "QP_infeasible_total": 29,
    "QP_infeasible_with_Omega_before_false": 29,
    "QP_infeasible_while_Omega_before_true": 0,
    "fallback_mode_counts": {
      "outside_certified_domain": 29
    }
  }
]
```

Temporal ordering of later failures is recorded per seed. A later event after the first physical failure is not automatically attributed causally to that one event; repeated fresh Gaussian exceedances and re-entry can create separate excursions.

### Diagnostic decision

Priority: residual authority diagnostic/retraining discussion. Do not change alpha automatically. Reward magnitude imbalance can coexist with excessive authority or early actor saturation; this sweep does not isolate all causal mechanisms. A single early checkpoint cannot establish SAC convergence or policy optimality.

No positive-X2-buffer term exists in the frozen reward. State recovery uses fixed balanced B, move penalties use final actual inputs, and physical-violation event penalties act after violating the bound. RPI remains audited but is not a stochastic replay penalty.

### Same-state first-failure counterfactual

```json
{
  "seed": 420000,
  "step": 12,
  "same_state_same_disturbance": true,
  "zero_residual_safe_input_at_this_controller_state": [
    233.65538828079238,
    224.50642560788918
  ],
  "zero_residual_counterfactual_next_state": [
    25.049997184752083,
    50.080408629926495
  ],
  "actor_actual_next_state": [
    24.988336686833648,
    50.06821885857201
  ],
  "scope": "one-step causal action comparison only, separate from autonomous scale sweep; not a certificate"
}
```

### Reward imbalance versus optimization

Recovery penalty is 0.01256% of mean absolute economic reward for baseline and 0.03785% for alpha=1. P100 move penalty is not uniformly weak (24.73% / 53.47%). The full actor replay reward is worse than baseline, so these data do not show that the full weighted objective prefers the unsafe trajectory. There is a recovery-scale imbalance plus an early policy suboptimal under its own recorded objective; this alone does not identify a SAC implementation bug or convergence failure.

### Action mapping semantics

The interior-anchor command is a fraction of the current feasible polytope, not a fixed physical residual bound. Raw action signs need not equal physical residual signs because the boundary ray originates at the interior anchor before blending with the zero-action baseline. This is not evidence of a numerical mapper bug.

### Conclusion

Case A is a diagnostic frontier classification only. Joint scales [0.05, 0.1] pass the defined mean-economic and per-seed 10%-IAE/TV gates on these fixed realizations, not all Gaussian inputs. Inspect their worst-seed economics before describing a gain as stable. No alpha is applied to training and the original alpha=1 actor is not promoted as a paper candidate. Priority: residual authority diagnostic/retraining discussion. Weak recovery shaping remains a secondary audit concern, not an automatic weight change.
