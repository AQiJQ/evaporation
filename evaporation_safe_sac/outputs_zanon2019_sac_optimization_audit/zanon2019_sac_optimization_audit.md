# SAC optimization audit

No training, no changes to reward/SAC/safety/replay.

## Availability
Episode100 critics/targets available; episode80/90 critics and all replay snapshots unavailable.
Evaluation states are not replay coverage. Case C / OOD conclusions cannot be inferred.

## Monte Carlo scope
Horizon=1000, gamma^H=4.31712e-05.
Soft Q is compared with stochastic-continuation soft MC. Environment performance contains NO entropy.
The deterministic/zero rows contain zero bonus placeholders, NOT comparable differential-entropy objectives.
Safety-aborted MC is not treated as a complete-horizon Q calibration sample.

## Results
{"MAE": 14.64100172413915, "RMSE": 25.711128015395754, "bias": -8.21847009129992, "Pearson": 0.21057145387601928, "Spearman": 0.29051331375923867}
{"pred_True_actual_True": 7, "pred_True_actual_False": 5, "pred_False_actual_True": 0, "pred_False_actual_False": 0}

Review deterministic_vs_stochastic_eval.csv by episode and disturbance seed before concluding Case B.
No automatic recommendation to tune SAC or add replay data.

## Paired deterministic / stochastic comparison
Action repetitions are averaged within each fixed disturbance seed before cross-seed comparison. CI is a normal approximation (implementation choice), not a training-seed confidence interval.
[
  {
    "episode": 80,
    "paired_disturbance_seed_count": 10,
    "environment_return_stochastic_minus_deterministic": {
      "mean": -2.899989687088703,
      "sample_std": 7.5665892744544925,
      "normal_approx_95pct_CI": [
        -7.5898103074390155,
        1.7898309332616096
      ],
      "min": -13.365650894940927,
      "max": 11.232013574334076
    },
    "environment_return_deterministic_minus_zero": {
      "mean": -57.31616515624612,
      "sample_std": 23.591447468580785,
      "normal_approx_95pct_CI": [
        -71.93829578722818,
        -42.69403452526406
      ],
      "min": -104.22108023430133,
      "max": -26.96164612722214
    },
    "environment_return_stochastic_minus_zero": {
      "mean": -60.21615484333481,
      "sample_std": 16.64801024705211,
      "normal_approx_95pct_CI": [
        -70.53469849787422,
        -49.89761118879541
      ],
      "min": -92.98906665996725,
      "max": -40.327297022163066
    },
    "J_econ_stochastic_minus_deterministic": {
      "mean": 570.4796669267118,
      "sample_std": 173.95882839063373,
      "normal_approx_95pct_CI": [
        462.65886803219377,
        678.30046582123
      ],
      "min": 252.8558386620134,
      "max": 786.4053606474772
    },
    "J_econ_deterministic_minus_zero": {
      "mean": -4032.4705437278376,
      "sample_std": 1483.5933559531609,
      "normal_approx_95pct_CI": [
        -4952.011232483601,
        -3112.9298549720743
      ],
      "min": -6430.452500746585,
      "max": -1697.0073796762154
    },
    "J_econ_stochastic_minus_zero": {
      "mean": -3461.9908768011255,
      "sample_std": 1325.9288269359868,
      "normal_approx_95pct_CI": [
        -4283.810078046141,
        -2640.17167555611
      ],
      "min": -5644.047140099108,
      "max": -1444.151541014202
    },
    "X2_IAE_stochastic_minus_deterministic": {
      "mean": -5.380102967624939,
      "sample_std": 1.783407557952551,
      "normal_approx_95pct_CI": [
        -6.485470424004744,
        -4.274735511245135
      ],
      "min": -8.47215418773905,
      "max": -2.854197407888236
    },
    "X2_IAE_deterministic_minus_zero": {
      "mean": 28.589026951395795,
      "sample_std": 13.278072301672449,
      "normal_approx_95pct_CI": [
        20.35919247509865,
        36.81886142769294
      ],
      "min": 14.781954571433275,
      "max": 54.599512445411406
    },
    "X2_IAE_stochastic_minus_zero": {
      "mean": 23.208923983770852,
      "sample_std": 11.574595123622547,
      "normal_approx_95pct_CI": [
        16.034915601125466,
        30.382932366416238
      ],
      "min": 11.927757163545039,
      "max": 46.127358257672356
    },
    "P2_IAE_stochastic_minus_deterministic": {
      "mean": 0.5281885612867739,
      "sample_std": 0.8355806618009979,
      "normal_approx_95pct_CI": [
        0.010290301510700872,
        1.046086821062847
      ],
      "min": -0.35626440530882064,
      "max": 1.8164496833865797
    },
    "P2_IAE_deterministic_minus_zero": {
      "mean": -5.014051246146132,
      "sample_std": 7.350133767055785,
      "normal_approx_95pct_CI": [
        -9.569711353064845,
        -0.4583911392274196
      ],
      "min": -17.061115074140346,
      "max": 2.1512561638155105
    },
    "P2_IAE_stochastic_minus_zero": {
      "mean": -4.485862684859358,
      "sample_std": 6.539572320654761,
      "normal_approx_95pct_CI": [
        -8.539131602365117,
        -0.4325937673535991
      ],
      "min": -15.244665390753767,
      "max": 1.79499175850669
    },
    "P100_TV_stochastic_minus_deterministic": {
      "mean": 168.71598107892555,
      "sample_std": 53.947713681683645,
      "normal_approx_95pct_CI": [
        135.2788417195469,
        202.1531204383042
      ],
      "min": 58.49206316098798,
      "max": 225.56446060847657
    },
    "P100_TV_deterministic_minus_zero": {
      "mean": 17.58736868014812,
      "sample_std": 208.94515729408892,
      "normal_approx_95pct_CI": [
        -111.91818152969911,
        147.09291888999536
      ],
      "min": -324.9810740578059,
      "max": 290.30158023603644
    },
    "P100_TV_stochastic_minus_zero": {
      "mean": 186.30334975907368,
      "sample_std": 195.49517348570004,
      "normal_approx_95pct_CI": [
        65.1341858812978,
        307.47251363684956
      ],
      "min": -101.40522720276931,
      "max": 503.82567039231526
    },
    "F200_TV_stochastic_minus_deterministic": {
      "mean": 130.54339529506464,
      "sample_std": 8.854952107203216,
      "normal_approx_95pct_CI": [
        125.05503911789259,
        136.0317514722367
      ],
      "min": 112.73723466566389,
      "max": 142.81910903115
    },
    "F200_TV_deterministic_minus_zero": {
      "mean": -103.99418930556035,
      "sample_std": 60.45074598106113,
      "normal_approx_95pct_CI": [
        -141.46194984261928,
        -66.52642876850143
      ],
      "min": -187.9233640409915,
      "max": -36.86184999063437
    },
    "F200_TV_stochastic_minus_zero": {
      "mean": 26.549205989504298,
      "sample_std": 60.58807557910741,
      "normal_approx_95pct_CI": [
        -11.003672314271025,
        64.10208429327962
      ],
      "min": -72.92284058806058,
      "max": 98.34423463919222
    }
  },
  {
    "episode": 90,
    "paired_disturbance_seed_count": 10,
    "environment_return_stochastic_minus_deterministic": {
      "mean": -13.652191376823946,
      "sample_std": 9.39241645430561,
      "normal_approx_95pct_CI": [
        -19.47367140759986,
        -7.830711346048035
      ],
      "min": -23.244625864522945,
      "max": 6.7199479355005
    },
    "environment_return_deterministic_minus_zero": {
      "mean": -25.465745487619124,
      "sample_std": 9.81605191437257,
      "normal_approx_95pct_CI": [
        -31.549797496874316,
        -19.381693478363932
      ],
      "min": -48.27030480210448,
      "max": -12.938945517865207
    },
    "environment_return_stochastic_minus_zero": {
      "mean": -39.11793686444307,
      "sample_std": 7.679804889427409,
      "normal_approx_95pct_CI": [
        -43.87792924995555,
        -34.357944478930584
      ],
      "min": -55.03453910928414,
      "max": -27.326945949985543
    },
    "J_econ_stochastic_minus_deterministic": {
      "mean": 637.1470608185977,
      "sample_std": 203.50247020930286,
      "normal_approx_95pct_CI": [
        511.0149230135336,
        763.2791986236617
      ],
      "min": 384.65923892054707,
      "max": 1014.157922747545
    },
    "J_econ_deterministic_minus_zero": {
      "mean": -1594.059480910655,
      "sample_std": 1439.7330912343143,
      "normal_approx_95pct_CI": [
        -2486.4152959496805,
        -701.7036658716298
      ],
      "min": -4305.7288492890075,
      "max": 616.6040268596262
    },
    "J_econ_stochastic_minus_zero": {
      "mean": -956.9124200920575,
      "sample_std": 1275.4213152480215,
      "normal_approx_95pct_CI": [
        -1747.4267412642898,
        -166.39809891982497
      ],
      "min": -3291.5709265414625,
      "max": 1034.6816105907783
    },
    "X2_IAE_stochastic_minus_deterministic": {
      "mean": -3.603551421554293,
      "sample_std": 3.440660159515293,
      "normal_approx_95pct_CI": [
        -5.7360946822529435,
        -1.471008160855643
      ],
      "min": -11.139647619974205,
      "max": -0.21446549766259437
    },
    "X2_IAE_deterministic_minus_zero": {
      "mean": 13.112566427258002,
      "sample_std": 9.629839285019335,
      "normal_approx_95pct_CI": [
        7.14393020142042,
        19.081202653095584
      ],
      "min": 5.852041426930356,
      "max": 34.89211747778765
    },
    "X2_IAE_stochastic_minus_zero": {
      "mean": 9.509015005703711,
      "sample_std": 6.280771374941992,
      "normal_approx_95pct_CI": [
        5.616152576213308,
        13.401877435194114
      ],
      "min": 4.364790060217949,
      "max": 23.752469857813445
    },
    "P2_IAE_stochastic_minus_deterministic": {
      "mean": 0.8925815998758765,
      "sample_std": 1.3053927196726107,
      "normal_approx_95pct_CI": [
        0.08349080978311707,
        1.701672389968636
      ],
      "min": -0.24221580652277197,
      "max": 3.3192658276967535
    },
    "P2_IAE_deterministic_minus_zero": {
      "mean": -4.1218395988761385,
      "sample_std": 6.036027357092878,
      "normal_approx_95pct_CI": [
        -7.863008114506137,
        -0.3806710832461402
      ],
      "min": -15.184590106132589,
      "max": 1.890114959471866
    },
    "P2_IAE_stochastic_minus_zero": {
      "mean": -3.229257999000262,
      "sample_std": 4.757872367503365,
      "normal_approx_95pct_CI": [
        -6.1782178445471745,
        -0.28029815345334974
      ],
      "min": -11.865324278435835,
      "max": 1.647899152949094
    },
    "P100_TV_stochastic_minus_deterministic": {
      "mean": 155.485019665535,
      "sample_std": 33.830909177783276,
      "normal_approx_95pct_CI": [
        134.51640491558146,
        176.45363441548852
      ],
      "min": 119.74287225816533,
      "max": 225.55806369817583
    },
    "P100_TV_deterministic_minus_zero": {
      "mean": -51.10774030823186,
      "sample_std": 298.5132684112077,
      "normal_approx_95pct_CI": [
        -236.12818094051127,
        133.91270032404756
      ],
      "min": -392.1845816518962,
      "max": 438.12336776781194
    },
    "P100_TV_stochastic_minus_zero": {
      "mean": 104.37727935730314,
      "sample_std": 282.844599901586,
      "normal_approx_95pct_CI": [
        -70.93161991803441,
        279.6861786326407
      ],
      "min": -247.7935537915314,
      "max": 570.9273259194388
    },
    "F200_TV_stochastic_minus_deterministic": {
      "mean": 133.4027995848203,
      "sample_std": 12.801652717418454,
      "normal_approx_95pct_CI": [
        125.46825302612115,
        141.33734614351943
      ],
      "min": 111.99589913798718,
      "max": 157.50303002093005
    },
    "F200_TV_deterministic_minus_zero": {
      "mean": -45.655285611058005,
      "sample_std": 80.58287906352005,
      "normal_approx_95pct_CI": [
        -95.60107150896346,
        4.290500286847447
      ],
      "min": -176.7062022430938,
      "max": 66.58187993075421
    },
    "F200_TV_stochastic_minus_zero": {
      "mean": 87.7475139737623,
      "sample_std": 73.12881244806088,
      "normal_approx_95pct_CI": [
        42.42180642960963,
        133.07322151791496
      ],
      "min": -19.552401453808784,
      "max": 178.5777790687414
    }
  },
  {
    "episode": 100,
    "paired_disturbance_seed_count": 10,
    "environment_return_stochastic_minus_deterministic": {
      "mean": -5.75124297597016,
      "sample_std": 8.641565448543638,
      "normal_approx_95pct_CI": [
        -11.107340731865406,
        -0.39514522007491326
      ],
      "min": -15.344205739342982,
      "max": 9.966018128363032
    },
    "environment_return_deterministic_minus_zero": {
      "mean": -49.53615558339482,
      "sample_std": 25.039343347595413,
      "normal_approx_95pct_CI": [
        -65.05570137769861,
        -34.016609789091035
      ],
      "min": -97.54485978669584,
      "max": -19.92601169524721
    },
    "environment_return_stochastic_minus_zero": {
      "mean": -55.28739855936499,
      "sample_std": 16.961620756351287,
      "normal_approx_95pct_CI": [
        -65.80032002138435,
        -44.774477097345624
      ],
      "min": -87.5788416583328,
      "max": -35.270217434590194
    },
    "J_econ_stochastic_minus_deterministic": {
      "mean": 485.1137076570652,
      "sample_std": 180.9207861611003,
      "normal_approx_95pct_CI": [
        372.977842630945,
        597.2495726831854
      ],
      "min": 274.87717297580093,
      "max": 839.209361041896
    },
    "J_econ_deterministic_minus_zero": {
      "mean": -2644.3025680025107,
      "sample_std": 1385.7292243929583,
      "normal_approx_95pct_CI": [
        -3503.1864395932107,
        -1785.4186964118107
      ],
      "min": -5484.280618593097,
      "max": -1074.187828041613
    },
    "J_econ_stochastic_minus_zero": {
      "mean": -2159.1888603454454,
      "sample_std": 1224.6029627514786,
      "normal_approx_95pct_CI": [
        -2918.2056403157076,
        -1400.1720803751832
      ],
      "min": -4645.071257551201,
      "max": -668.8885704213753
    },
    "X2_IAE_stochastic_minus_deterministic": {
      "mean": -4.390346791517323,
      "sample_std": 2.854545816311567,
      "normal_approx_95pct_CI": [
        -6.1596126186277615,
        -2.621080964406884
      ],
      "min": -9.258743691497699,
      "max": -1.1620437737775546
    },
    "X2_IAE_deterministic_minus_zero": {
      "mean": 18.616793359499912,
      "sample_std": 12.803216477950093,
      "normal_approx_95pct_CI": [
        10.681277571981745,
        26.55230914701808
      ],
      "min": 6.508443236657357,
      "max": 44.61493368486224
    },
    "X2_IAE_stochastic_minus_zero": {
      "mean": 14.226446567982588,
      "sample_std": 10.04502386156584,
      "normal_approx_95pct_CI": [
        8.000476275538148,
        20.45241686042703
      ],
      "min": 5.346399462879802,
      "max": 35.35618999336454
    },
    "P2_IAE_stochastic_minus_deterministic": {
      "mean": 0.6598471648051074,
      "sample_std": 0.7610128931756022,
      "normal_approx_95pct_CI": [
        0.18816648685200232,
        1.1315278427582125
      ],
      "min": -0.2854685739492311,
      "max": 1.7452189413831931
    },
    "P2_IAE_deterministic_minus_zero": {
      "mean": -3.8557859688268556,
      "sample_std": 4.540617418949783,
      "normal_approx_95pct_CI": [
        -6.670089802180852,
        -1.0414821354728594
      ],
      "min": -12.169305488954393,
      "max": 2.346939422931129
    },
    "P2_IAE_stochastic_minus_zero": {
      "mean": -3.1959388040217482,
      "sample_std": 3.845843300135652,
      "normal_approx_95pct_CI": [
        -5.579617177117081,
        -0.8122604309264161
      ],
      "min": -10.441277353400892,
      "max": 2.061470848981898
    },
    "P100_TV_stochastic_minus_deterministic": {
      "mean": 146.04863275020608,
      "sample_std": 28.96975152476578,
      "normal_approx_95pct_CI": [
        128.09299472899653,
        164.00427077141563
      ],
      "min": 89.64636464832984,
      "max": 189.05061871554017
    },
    "P100_TV_deterministic_minus_zero": {
      "mean": 184.25582149610653,
      "sample_std": 130.03430562001418,
      "normal_approx_95pct_CI": [
        103.65972387144157,
        264.8519191207715
      ],
      "min": -14.8921808118439,
      "max": 380.9941332456724
    },
    "P100_TV_stochastic_minus_zero": {
      "mean": 330.3044542463126,
      "sample_std": 124.21483213117044,
      "normal_approx_95pct_CI": [
        253.31530365915933,
        407.2936048334659
      ],
      "min": 132.04399914380065,
      "max": 504.57315878034706
    },
    "F200_TV_stochastic_minus_deterministic": {
      "mean": 108.22743514039772,
      "sample_std": 11.217594811984906,
      "normal_approx_95pct_CI": [
        101.27469784334464,
        115.18017243745079
      ],
      "min": 91.30582861889184,
      "max": 128.52255296703368
    },
    "F200_TV_deterministic_minus_zero": {
      "mean": -11.956182520475954,
      "sample_std": 45.228175046777736,
      "normal_approx_95pct_CI": [
        -39.98889584235555,
        16.076530801403642
      ],
      "min": -102.88772082270987,
      "max": 52.4785805858869
    },
    "F200_TV_stochastic_minus_zero": {
      "mean": 96.27125261992174,
      "sample_std": 35.74599152060426,
      "normal_approx_95pct_CI": [
        74.11565753639141,
        118.42684770345207
      ],
      "min": 25.63483214432381,
      "max": 143.78440920477874
    }
  }
]

## Completion status
Full requested budget completed: True
A smoke/short-horizon run is an implementation test, not evidence of Q calibration or any Case A-F.

## Twelve diagnostic questions
1. Deterministic vs stochastic: see the paired environment-return differences above; entropy is separate.
2. Actor vs zero ranking: see four quadrants; incomplete MC excluded and sampling uncertainty remains.
3. Q/MC correlations: {'MAE': 14.64100172413915, 'RMSE': 25.711128015395754, 'bias': -8.21847009129992, 'Pearson': 0.21057145387601928, 'Spearman': 0.29051331375923867}
4. Overestimation: bias is reported; a truncated-MC positive bias alone is not a systematic-overestimation proof.
5. Actor exploiting error: compare gradient and actual local direction; single-CRN probes are preliminary only.
6. Q disagreement: {'zero': {'mean': 0.4376186914741993, 'median': 0.3764495849609375, 'p95': 1.062505722045898, 'max': 2.032684326171875}, 'actor': {'mean': 0.27148314379155636, 'median': 0.22311973571777344, 'p95': 0.6669354438781737, 'max': 1.3363456726074219}, 'sample': {'mean': 0.29851346276700497, 'median': 0.24019622802734375, 'p95': 0.7984439849853515, 'max': 1.5584907531738281}}; sampled actor action is NOT random replay action.
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
  "C:\\Users\\cushy\\PycharmProjects\\evaporation\\evaporation_safe_sac\\outputs_zanon2019_alpha020_rewardcal_seed42_100x1000\\models\\evaluation\\episode_0100_actor.pth": "6fb6ffb359d7892b3a779ddfaaf008b7f9ae2b79bdfbcf3c94deeddd5282cc82",
  "C:\\Users\\cushy\\PycharmProjects\\evaporation\\evaporation_safe_sac\\outputs_zanon2019_alpha020_rewardcal_seed42_100x1000\\models\\evaluation\\episode_0090_actor.pth": "6921f98d201672280e39d2825a3cfbb9963d6eb7c31031a41ea2e47be8db3c8b",
  "C:\\Users\\cushy\\PycharmProjects\\evaporation\\evaporation_safe_sac\\outputs_zanon2019_alpha020_rewardcal_seed42_100x1000\\models\\evaluation\\episode_0080_actor.pth": "0804af5ab9748cf8599d837a6d65bbf0adfffaa4dd1c657020622a11478ac5ac"
}

All diagnostic budgets, state bands and finite differences are implementation choices for reproduction.