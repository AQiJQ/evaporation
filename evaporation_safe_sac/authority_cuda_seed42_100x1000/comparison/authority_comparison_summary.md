# Stochastic residual authority pilot

{
  "certification_status": "empirical_only_under_ECC2019_stochastic_disturbance",
  "final_test_performed": false,
  "frozen_config_consistency": true,
  "all_pilots_completed": false,
  "all_pilots_resolved_including_early_stop": true,
  "sweet_spot_alpha": null,
  "candidate_episode": null,
  "selection_basis": "hierarchical validation candidate; review full Pareto trade-off and occupancy before extending",
  "reason": "unresolved pilots or no eligible completed comparable joint candidate; no authority recommendation",
  "automatically_extend_training": false,
  "reward_change_evidence": "not inferred automatically",
  "reference": "alpha=1 episode5 is failed 3-seed historical reference, not a 10-seed retrained pilot"
}

Validation only: 420000..420009. Final seeds 430000..430049 remain unused.
G is empirical amplification, not Hinf. Baseline is zero residual, not the ECC2019 controller.
Protocol and hierarchical tie-breaking: implementation choice for reproduction; not specified in the paper.
Missing runs/checkpoints are unavailable, not zero. An aborted run is not a completed safe pilot.
