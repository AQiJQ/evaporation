# SAC optimization audit

No training, no changes to reward/SAC/safety/replay.

## Availability
Episode100 critics/targets available; episode80/90 critics and all replay snapshots unavailable.
Evaluation states are not replay coverage. Case C / OOD conclusions cannot be inferred.

## Monte Carlo scope
Horizon=20, gamma^H=0.817907.
Soft Q is compared with stochastic-continuation soft MC. Environment performance contains NO entropy.
The deterministic/zero rows contain zero bonus placeholders, NOT comparable differential-entropy objectives.
Safety-aborted MC is not treated as a complete-horizon Q calibration sample.

## Results
{"MAE": 31.85870637540154, "RMSE": 32.06677569884371, "bias": -31.85870637540154, "Pearson": 0.9995953935508075, "Spearman": 0.9912087912087912}
{"pred_True_actual_True": 2, "pred_True_actual_False": 0, "pred_False_actual_True": 0, "pred_False_actual_False": 0}

Review deterministic_vs_stochastic_eval.csv by episode and disturbance seed before concluding Case B.
No automatic recommendation to tune SAC or add replay data.

## Paired deterministic / stochastic comparison
Action repetitions are averaged within each fixed disturbance seed before cross-seed comparison. CI is a normal approximation (implementation choice), not a training-seed confidence interval.
[
  {
    "episode": 100,
    "paired_disturbance_seed_count": 1,
    "environment_return_stochastic_minus_deterministic": {
      "mean": -0.5475031857062511,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": -0.5475031857062511,
      "max": -0.5475031857062511
    },
    "environment_return_deterministic_minus_zero": {
      "mean": 1.9125928673584056,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": 1.9125928673584056,
      "max": 1.9125928673584056
    },
    "environment_return_stochastic_minus_zero": {
      "mean": 1.3650896816521545,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": 1.3650896816521545,
      "max": 1.3650896816521545
    },
    "J_econ_stochastic_minus_deterministic": {
      "mean": 75.74067439767532,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": 75.74067439767532,
      "max": 75.74067439767532
    },
    "J_econ_deterministic_minus_zero": {
      "mean": -278.552986983399,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": -278.552986983399,
      "max": -278.552986983399
    },
    "J_econ_stochastic_minus_zero": {
      "mean": -202.8123125857237,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": -202.8123125857237,
      "max": -202.8123125857237
    },
    "X2_IAE_stochastic_minus_deterministic": {
      "mean": -0.025322681365677724,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": -0.025322681365677724,
      "max": -0.025322681365677724
    },
    "X2_IAE_deterministic_minus_zero": {
      "mean": 0.13767722501719248,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": 0.13767722501719248,
      "max": 0.13767722501719248
    },
    "X2_IAE_stochastic_minus_zero": {
      "mean": 0.11235454365151476,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": 0.11235454365151476,
      "max": 0.11235454365151476
    },
    "P2_IAE_stochastic_minus_deterministic": {
      "mean": 0.005538019944054895,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": 0.005538019944054895,
      "max": 0.005538019944054895
    },
    "P2_IAE_deterministic_minus_zero": {
      "mean": -0.027929052238782504,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": -0.027929052238782504,
      "max": -0.027929052238782504
    },
    "P2_IAE_stochastic_minus_zero": {
      "mean": -0.02239103229472761,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": -0.02239103229472761,
      "max": -0.02239103229472761
    },
    "P100_TV_stochastic_minus_deterministic": {
      "mean": -1.3702383015764212,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": -1.3702383015764212,
      "max": -1.3702383015764212
    },
    "P100_TV_deterministic_minus_zero": {
      "mean": -7.205588912520568,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": -7.205588912520568,
      "max": -7.205588912520568
    },
    "P100_TV_stochastic_minus_zero": {
      "mean": -8.57582721409699,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": -8.57582721409699,
      "max": -8.57582721409699
    },
    "F200_TV_stochastic_minus_deterministic": {
      "mean": 2.251029465245068,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": 2.251029465245068,
      "max": 2.251029465245068
    },
    "F200_TV_deterministic_minus_zero": {
      "mean": -8.374585122725364,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": -8.374585122725364,
      "max": -8.374585122725364
    },
    "F200_TV_stochastic_minus_zero": {
      "mean": -6.123555657480296,
      "sample_std": null,
      "normal_approx_95pct_CI": null,
      "min": -6.123555657480296,
      "max": -6.123555657480296
    }
  }
]

## Completion status
Full requested budget completed: False
A smoke/short-horizon run is an implementation test, not evidence of Q calibration or any Case A-F.

## Twelve diagnostic questions
1. Deterministic vs stochastic: see the paired environment-return differences above; entropy is separate.
2. Actor vs zero ranking: see four quadrants; incomplete MC excluded and sampling uncertainty remains.
3. Q/MC correlations: {'MAE': 31.85870637540154, 'RMSE': 32.06677569884371, 'bias': -31.85870637540154, 'Pearson': 0.9995953935508075, 'Spearman': 0.9912087912087912}
4. Overestimation: bias is reported; a truncated-MC positive bias alone is not a systematic-overestimation proof.
5. Actor exploiting error: compare gradient and actual local direction; single-CRN probes are preliminary only.
6. Q disagreement: {'zero': {'mean': 0.26274871826171875, 'median': 0.28502655029296875, 'p95': 0.32594909667968747, 'max': 0.33922576904296875}, 'actor': {'mean': 0.4800529479980469, 'median': 0.47127342224121094, 'p95': 0.8978322982788083, 'max': 1.0671272277832031}, 'sample': {'mean': 0.37747716903686523, 'median': 0.36890602111816406, 'p95': 0.6007179260253906, 'max': 0.6032867431640625}}; sampled actor action is NOT random replay action.
7. Zero-action replay coverage: unavailable (no snapshot).
8. Stored/recomputed replay reward: unavailable; independent official evaluation-path formula test passed.
9. Done mask: finite-terminal convention matches source; continuing-task truncation intent not specified.
10. OOD replay support: unavailable (no snapshot).
11. Cases: A/B require the numerical evidence above; C/D unsupported without replay, E not established, F not established.
12. Next modification: do not choose critic/extraction/replay/initialization/hyperparameters before the full audit.

## Frozen source hashes
{
  "C:\\Users\\cushy\\PycharmProjects\\evaporation\\evaporation_safe_sac\\outputs_zanon2019_alpha020_rewardcal_seed42_100x1000\\models\\final_checkpoint.pth": "a3667f8b21400051969abe0ccacbbac2d718fd825bea53f1de50bc1f2b21ae85",
  "C:\\Users\\cushy\\PycharmProjects\\evaporation\\evaporation_safe_sac\\outputs_certified_reference_margin_audit\\candidate_designs\\refine_1_X2_25.39000000_P2_50.12500000.npz": "6f2615d52f221a4de096760179e2f6750584f51ae131608235de4f96c2728af9",
  "C:\\Users\\cushy\\PycharmProjects\\evaporation\\evaporation_safe_sac\\outputs_controlled_invariant_error_B\\Omega_anchor_B_vertices.csv": "60f32ee9237056486bfe7751d36dfa8571e64db1ca5998c1b3fb6bb7c0892f7b",
  "C:\\Users\\cushy\\PycharmProjects\\evaporation\\evaporation_safe_sac\\outputs_zanon2019_alpha020_reward_scale_audit_v1\\reward_calibration.json": "43fa13a8dd79937da897ac5200ccbac57768adb123e10177cc0958a5566badab",
  "C:\\Users\\cushy\\PycharmProjects\\evaporation\\sac.py": "4de726380ae3d227a2153cb757365d3c53f322e325fc771fff23cf23a7ed69fb",
  "C:\\Users\\cushy\\PycharmProjects\\evaporation\\train.py": "0649ad8649a555015e145d4a3846dcdfd19fe179f588ac37ae09b9b23c24d993",
  "C:\\Users\\cushy\\PycharmProjects\\evaporation\\zanon2019_train.py": "38a074d761d369904446a2c4783766de0682e45c2775051b7a21eff6c74ae443",
  "C:\\Users\\cushy\\PycharmProjects\\evaporation\\zanon2019_authority.py": "74c17645323c85ae91f22d2f5a66b73b6d4978faacbd1cf61768b5967748bdbe",
  "C:\\Users\\cushy\\PycharmProjects\\evaporation\\evaporation_safe_sac\\outputs_zanon2019_alpha020_rewardcal_seed42_100x1000\\models\\evaluation\\episode_0100_actor.pth": "6fb6ffb359d7892b3a779ddfaaf008b7f9ae2b79bdfbcf3c94deeddd5282cc82"
}

All diagnostic budgets, state bands and finite differences are implementation choices for reproduction.