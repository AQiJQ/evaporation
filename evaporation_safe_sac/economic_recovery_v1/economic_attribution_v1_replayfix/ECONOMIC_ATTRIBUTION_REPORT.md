# Economic attribution and finite state-only witness study

Status: COMPLETE

Physical cost and reward are separate. Cost uses nonlinear post-state and applied input.
All percentages retain their specified denominators. Absolute geometry/SAC effects add exactly.
The P100/steam label is indirect; 600 multiplies F100, not P100.
Frozen actor comparisons include changed mapping and changed state trajectories; correlations are not causal proof.
No Gaussian trajectory is labeled formally bounded-W certified.

## Geometry comparison

| Geometry | Geometry baseline gain % | SAC vs own baseline % | Total vs production % | Win fraction |
|---|---:|---:|---:|---:|
|baseline|0.00000000|0.08650902|0.08650902|0.5904|
|C2_8|0.00000000|0.09151897|0.09151897|0.5891|
|A3|-0.00813319|0.09671759|0.08859226|0.5806|
|C2|-0.00813319|0.08696187|0.07883575|0.5814|
|C2_3|-0.01220532|0.08185469|0.06965937|0.5769|

## Physical component contributions (positive = saving)

baseline: {"flow_F2": -81.77907277698577, "recirculation_F3": 0.0, "steam_F100": 5473.9138506675135, "cooling_F200": -13.046983539703069}
C2_8: {"flow_F2": -86.41210974934401, "recirculation_F3": 0.0, "steam_F100": 5781.975397408661, "cooling_F200": -5.009737231236068}
A3: {"flow_F2": -88.36134683212659, "recirculation_F3": 0.0, "steam_F100": 5969.149626272265, "cooling_F200": -372.16528822347027}
C2: {"flow_F2": -79.36022399830654, "recirculation_F3": 0.0, "steam_F100": 5369.5245743433015, "cooling_F200": -388.20328298928945}
C2_3: {"flow_F2": -73.45934597189117, "recirculation_F3": 0.0, "steam_F100": 5007.925727437437, "cooling_F200": -603.0130874240771}

## P100-only scan

{"candidate": "P100_1000", "factor": 1.0, "certificate": "CERTIFIED_PASS", "completed_paths": 10, "positive_P100_authority": 3.107448612665206, "mean_gain_pct": 0.08650902042765084, "worst_gain_pct": 0.07685793668525148, "win_fraction": 0.5903999999999999, "minimum_X2_margin": 0.13229527485792403, "P100_TV": 3440.103263405856, "F200_TV": 683.6545091909168, "P100_RMS_du": 4.341219779163623, "F200_RMS_du": 0.9038050820361931, "X2_std": 0.046948445625689714, "centered_X2_MAE": 0.03733492033954312}
{"candidate": "P100_1025", "factor": 1.025, "certificate": "CERTIFIED_PASS", "completed_paths": 10, "positive_P100_authority": 3.2044334759464554, "mean_gain_pct": 0.0890523578868726, "worst_gain_pct": 0.07846083160470187, "win_fraction": 0.5896000000000001, "minimum_X2_margin": 0.12998900654807954, "P100_TV": 3439.5459888851206, "F200_TV": 684.8587409978153, "P100_RMS_du": 4.340629598279457, "F200_RMS_du": 0.9063186109516567, "X2_std": 0.047027225998441655, "centered_X2_MAE": 0.03740657510635558}
{"candidate": "P100_1050", "factor": 1.05, "certificate": "CERTIFIED_PASS", "completed_paths": 10, "positive_P100_authority": 3.301418339227706, "mean_gain_pct": 0.09151896987306787, "worst_gain_pct": 0.08052575907972918, "win_fraction": 0.5891, "minimum_X2_margin": 0.1276241431630183, "P100_TV": 3438.8169485386834, "F200_TV": 687.2211300664405, "P100_RMS_du": 4.339465949938124, "F200_RMS_du": 0.9115302032747454, "X2_std": 0.047144212829026484, "centered_X2_MAE": 0.0374864552168971}
{"candidate": "P100_1075", "factor": 1.075, "certificate": "CERTIFIED_PASS", "completed_paths": 10, "positive_P100_authority": 3.398403202508956, "mean_gain_pct": 0.09396843160988874, "worst_gain_pct": 0.08266142125244054, "win_fraction": 0.5896, "minimum_X2_margin": 0.12522989016080643, "P100_TV": 3437.519487730475, "F200_TV": 689.4951153699325, "P100_RMS_du": 4.337843310756849, "F200_RMS_du": 0.9160998863199381, "X2_std": 0.04723002786607732, "centered_X2_MAE": 0.03755033999771192}
{"candidate": "P100_1100", "factor": 1.1, "certificate": "CERTIFIED_PASS", "completed_paths": 10, "positive_P100_authority": 3.495388065790206, "mean_gain_pct": 0.09614896227871998, "worst_gain_pct": 0.08470604032103692, "win_fraction": 0.5888, "minimum_X2_margin": 0.12446119946208967, "P100_TV": 3436.1433488265443, "F200_TV": 692.7673745976798, "P100_RMS_du": 4.33665239873897, "F200_RMS_du": 0.9214163284268725, "X2_std": 0.0473335323963182, "centered_X2_MAE": 0.03763774450824936}

## Witness

{
  "status": "COMPLETE",
  "preregistered_budget": 512,
  "completed_candidates": 512,
  "safe_candidates": 512,
  "positive_mean_candidates": 39,
  "all_validation_positive_candidates": 19,
  "best_safe": {
    "candidate": 167,
    "safety_passed": true,
    "failure": null,
    "completed_paths": 10,
    "search": "finite_state_dependent_witness_search",
    "mean_improvement_pct": 0.0638590710789818,
    "worst_path_improvement_pct": 0.03849600781995038,
    "all_validation_positive": true,
    "minimum_X2_margin": 0.1538452735955147,
    "control_quality_pass": true,
    "economic_win_fraction": 0.5693,
    "J_econ": 6213944.696697863,
    "P100_TV": 3499.630022796736,
    "F200_TV": 634.0582721359737,
    "P100_RMS_du": 4.482360729771652,
    "F200_RMS_du": 0.8386797413753415,
    "P100_TV_ratio": 0.9120035643753281,
    "F200_TV_ratio": 0.7696667676133252,
    "P100_RMS_du_ratio": 0.9299998107037443,
    "F200_RMS_du_ratio": 0.8101246966943039,
    "X2_std": 0.04762153146571726,
    "centered_X2_MAE": 0.03790348648738566,
    "X2_IAE_ratio": 1.7013192917700273,
    "P2_IAE_ratio": 1.0422086619717674,
    "X2_ISE_ratio": 2.6655761849907633,
    "P2_ISE_ratio": 1.1487263338521727,
    "W_exceedance_rate": 0.9827999999999999,
    "P100_sign_change_ratio": 1.0215857958874222,
    "F200_sign_change_ratio": 1.0454732362910735,
    "physical_state_violation_steps": 0,
    "physical_input_violation_count": 0,
    "QP_infeasible_count": 0,
    "Omega_exit_count": 0,
    "robust_region_violation_count": 0
  },
  "best_control_quality": {
    "candidate": 167,
    "safety_passed": true,
    "failure": null,
    "completed_paths": 10,
    "search": "finite_state_dependent_witness_search",
    "mean_improvement_pct": 0.0638590710789818,
    "worst_path_improvement_pct": 0.03849600781995038,
    "all_validation_positive": true,
    "minimum_X2_margin": 0.1538452735955147,
    "control_quality_pass": true,
    "economic_win_fraction": 0.5693,
    "J_econ": 6213944.696697863,
    "P100_TV": 3499.630022796736,
    "F200_TV": 634.0582721359737,
    "P100_RMS_du": 4.482360729771652,
    "F200_RMS_du": 0.8386797413753415,
    "P100_TV_ratio": 0.9120035643753281,
    "F200_TV_ratio": 0.7696667676133252,
    "P100_RMS_du_ratio": 0.9299998107037443,
    "F200_RMS_du_ratio": 0.8101246966943039,
    "X2_std": 0.04762153146571726,
    "centered_X2_MAE": 0.03790348648738566,
    "X2_IAE_ratio": 1.7013192917700273,
    "P2_IAE_ratio": 1.0422086619717674,
    "X2_ISE_ratio": 2.6655761849907633,
    "P2_ISE_ratio": 1.1487263338521727,
    "W_exceedance_rate": 0.9827999999999999,
    "P100_sign_change_ratio": 1.0215857958874222,
    "F200_sign_change_ratio": 1.0454732362910735,
    "physical_state_violation_steps": 0,
    "physical_input_violation_count": 0,
    "QP_infeasible_count": 0,
    "Omega_exit_count": 0,
    "robust_region_violation_count": 0
  },
  "certification_scope": "Gaussian empirical only",
  "claim": "best found in finite state-only family, NOT global optimum or oracle upper bound",
  "frozen_SAC_reference_gain_pct": 0.08650902042768663
}

Report finite-family/geometry headroom without claiming a global upper bound; do not chase large gains by relaxing gates

No SAC training, no production changes, no final seeds, no commit/push. Existing Conservative partial results preserved.