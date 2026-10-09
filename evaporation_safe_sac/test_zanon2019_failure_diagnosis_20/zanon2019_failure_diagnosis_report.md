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
      "physical": 0,
      "Omega_affected_steps": 1,
      "QP_infeasible": 1
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
        "physical": 0,
        "Omega_affected_steps": 1,
        "QP_infeasible": 1
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
        "physical": 0,
        "Omega_affected_steps": 1,
        "QP_infeasible": 1
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
      "first_physical_transition_step": null,
      "first_Omega_affected_transition_step": null,
      "first_Omega_before_step": null,
      "first_QP_infeasible_step": null,
      "no_physical_failure": {
        "physical": 0,
        "Omega_affected_steps": 0,
        "QP_infeasible": 0
      }
    }
  ]
}
```

## Residual scale frontier

|alpha|economics %|X2 IAE|P2 IAE|P100 TV|F200 TV|G X2|physical|QP|Omega|
|---|---|---|---|---|---|---|---|---|---|
|0|0.000000|1.0000|1.0000|1.0000|1.0000|1.0000|0|0|0|
|0.05|0.047648|1.0091|1.0444|0.9514|0.9786|1.0107|0|0|0|
|0.1|0.095304|1.0188|1.0937|0.9052|0.9589|1.0228|0|0|0|
|0.2|0.182500|1.0515|1.1969|0.8526|0.9349|1.0589|0|0|0|
|0.3|0.279586|1.0880|1.3311|0.8352|0.9676|1.0993|0|0|0|
|0.4|0.357946|1.1569|1.4560|0.8177|1.1494|1.1605|0|0|0|
|0.5|0.614710|1.2972|1.6248|0.8644|1.3161|1.2891|0|0|0|
|0.6|0.882060|1.5277|2.0466|0.8236|1.4200|1.5206|0|0|0|
|0.75|1.372751|2.3903|3.7008|1.0855|1.7553|2.3306|0|0|0|
|1|4.881231|5.1923|8.6705|1.5848|2.5888|5.0874|2|2|4|

Empirical-safe positive-economic scales on these paths: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75].
Level-C scales: []. Representative audited scale only: 0.05.

## Reward source and scope

All components are recomputed by current train.run_episode. Its evaluation return includes RPI event/excess; stochastic SAC replay excludes these two, which remain separately audited. Component CSVs show signed contributions and magnitude ratios per seed, and per-step reward reconciliation. Physical-margin buffering penalty is not present; violation penalties activate only after violation. No new penalty is invented.

## Interpretation limitations

This is one early checkpoint and three fixed realizations, not a convergence result or proof about all Gaussian disturbances. Cascade counts are temporal associations, not proof that every later violation would disappear after repairing the first. Intermediate scales change the state fed to the frozen actor; no teacher forcing is used. A raw negative actor component need not imply a negative physical residual under the interior-anchor mapping; inspect actor and physical-residual sign statistics separately. Mean absolute economic reward versus penalty is not alone sufficient to prove reward dominance: examine incremental policy-vs-baseline reward differences as well.

## Files

residual_scale_sweep.csv, residual_scale_per_seed.csv, first_failure_trace.csv, cascade_analysis.json, reward_component_audit.csv, reward_components_per_step.csv, X2_margin_statistics.csv, actor_action_statistics.csv, diagnosis_results.json. All requested plots and full per-alpha/seed transition traces are saved alongside this report.

## Aggregated evidence and interpretation

Frontier classification: B. Positive-economic, empirically safe scales: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]; Level-C scales: [].
Episode-5 replay reward mean change versus baseline: -4.27504772. A negative value means the frozen actual replay reward does not rank this unsafe policy above baseline on these paths. Economic component magnitude alone must not be used to claim why SAC accepted it.

### Reward components (three-seed average per step, signed)

|component|baseline|alpha=1|frontier representative|
|---|---|---|---|
|economic_reward|0.0019292168|1.5178576|0.016676752|
|projection_penalty|0|0|0|
|mapping_penalty|0|0|0|
|move_penalty|-2.0978298e-34|-0.00025042133|-2.2598521e-07|
|state_recovery_penalty|-5.3171844e-05|-0.0010496649|-5.3290732e-05|
|p100_move_penalty|-0.25340679|-2.4747928|-0.22929977|
|f200_move_penalty|-0.015367626|-0.071270779|-0.014710886|
|saturation_penalty|0|-0.012406546|0|
|state_violation_penalty|0|-3.3333333|0|
|input_violation_penalty|0|0|0|
|qp_infeasible_penalty|0|-0.16666667|0|
|state_excess_penalty|0|-3.3409312e-05|0|
|input_excess_penalty|0|0|0|
|actual_training_replay_reward_same_transition|-0.26689837|-4.5419461|-0.22738742|

### X2 physical margins (post-state, includes final transition)

|label|alpha|mean|min|p1|p5|fraction<0.05|fraction<0.10|fraction<0.20|
|---|---|---|---|---|---|---|---|---|
|baseline|0|0.389208|0.307153|0.314164|0.328401|0.0000%|0.0000%|0.0000%|
|episode5_actor|1|0.223988|-0.017748|-0.0141581|0.0437215|13.3333%|18.3333%|41.6667%|
|frontier_representative|0.05|0.388075|0.305734|0.312736|0.327149|0.0000%|0.0000%|0.0000%|

### Actor signs and saturation

```json
[
  {
    "dimension": "P100",
    "kind": "raw_actor_action",
    "mean": -0.5903210001687209,
    "std": 0.5037713395027165,
    "min": -0.9941441416740417,
    "max": 0.5142608284950256,
    "p1": -0.9932235115766526,
    "p5": -0.9918792337179184,
    "fraction_abs_gt_0p9": 0.5,
    "negative_fraction": 0.85,
    "positive_fraction": 0.15,
    "raw_nonzero_sign_disagreement_with_requested_residual": 0.1
  },
  {
    "dimension": "P100",
    "kind": "requested_residual_normalized",
    "mean": -0.3116840504782513,
    "std": 0.2941888650249788,
    "min": -0.7698260927863885,
    "max": 0.04911268252878818,
    "p1": -0.7696569522525533,
    "p5": -0.7691153914012829,
    "fraction_abs_gt_0p9": 0.0,
    "negative_fraction": 0.7166666666666667,
    "positive_fraction": 0.1,
    "raw_nonzero_sign_disagreement_with_requested_residual": 0.1
  },
  {
    "dimension": "F200",
    "kind": "raw_actor_action",
    "mean": 0.053073505905922504,
    "std": 0.09093750894694444,
    "min": -0.13065381348133087,
    "max": 0.2933543920516968,
    "p1": -0.1291997566819191,
    "p5": -0.1177882730960846,
    "fraction_abs_gt_0p9": 0.0,
    "negative_fraction": 0.23333333333333334,
    "positive_fraction": 0.7666666666666667,
    "raw_nonzero_sign_disagreement_with_requested_residual": 0.16666666666666666
  },
  {
    "dimension": "F200",
    "kind": "requested_residual_normalized",
    "mean": 0.007978808668049503,
    "std": 0.03598286434613921,
    "min": -0.0648533241181151,
    "max": 0.09840677423304733,
    "p1": -0.060254785703991086,
    "p5": -0.05493665191200818,
    "fraction_abs_gt_0p9": 0.0,
    "negative_fraction": 0.36666666666666664,
    "positive_fraction": 0.6,
    "raw_nonzero_sign_disagreement_with_requested_residual": 0.16666666666666666
  }
]
```

### QP infeasibility context

```json
[
  {
    "seed": 420000,
    "QP_infeasible_total": 1,
    "QP_infeasible_with_Omega_before_false": 1,
    "QP_infeasible_while_Omega_before_true": 0,
    "fallback_mode_counts": {
      "outside_certified_domain": 1
    }
  },
  {
    "seed": 420001,
    "QP_infeasible_total": 1,
    "QP_infeasible_with_Omega_before_false": 1,
    "QP_infeasible_while_Omega_before_true": 0,
    "fallback_mode_counts": {
      "outside_certified_domain": 1
    }
  },
  {
    "seed": 420002,
    "QP_infeasible_total": 0,
    "QP_infeasible_with_Omega_before_false": 0,
    "QP_infeasible_while_Omega_before_true": 0,
    "fallback_mode_counts": {}
  }
]
```

Temporal ordering of later failures is recorded per seed. A later event after the first physical failure is not automatically attributed causally to that one event; repeated fresh Gaussian exceedances and re-entry can create separate excursions.

### Diagnostic decision

Priority: policy objective/optimization diagnosis before any retraining. Do not change alpha automatically. Reward magnitude imbalance can coexist with excessive authority or early actor saturation; this sweep does not isolate all causal mechanisms. A single early checkpoint cannot establish SAC convergence or policy optimality.

No positive-X2-buffer term exists in the frozen reward. State recovery uses fixed balanced B, move penalties use final actual inputs, and physical-violation event penalties act after violating the bound. RPI remains audited but is not a stochastic replay penalty.
