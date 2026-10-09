# Certificate-aware U_R sensitivity study

Status: COMPLETE. 

All values use the frozen balanced-B model/certificate. Passing means the existing bounded-W numerical protocol and sampled coverage, not a continuous nonlinear or Gaussian safety proof.
This study checks the active ECC2019 Z/Omega branch. Stored finite-jump Gm/Bj are hash-preserved and inactive here. Before adopting a P100-shrinking candidate for a finite-jump experiment, recheck its downstream Gm/Bj action-existence certificates. No production adoption is made.

Candidates: 31; passing: 19; failed: 12.

## Original rho replay

| attempt | rho | pass | failed gate |
|---:|---:|:---:|---|
|1|1|True||
|2|1.25|False|invariant_feasible,v_ref_feasible,verification_qp_feasible,residual_authority_feasible|
|3|1.125|False|invariant_feasible,v_ref_feasible,verification_qp_feasible,residual_authority_feasible|
|4|1.0625|True||
|5|1.09375|True||
|6|1.109375|False|invariant_feasible,v_ref_feasible,verification_qp_feasible,residual_authority_feasible|
|7|1.1015625|True||
|8|1.10546875|True||
|9|1.107421875|True||
|10|1.1083984375|True||

## Candidate comparison

| candidate | family | F200 U_R upper | nominal upper | certificate | positive applied F200 | frozen gain % |
|---|---|---:|---:|---|---:|---:|
|baseline|baseline|249.528606630|216.303280988|CERTIFIED_PASS|0.40481612279545515|0.08650902042768663|
|A0|shared_rho|246.203411318|212.978085675|CERTIFIED_FAIL|NOT_EVALUATED / no passing candidate|NOT_EVALUATED / no passing candidate|
|A1|shared_rho|249.528606630|216.303280988|CERTIFIED_PASS|0.40481612279545515|0.08650902042768663|
|A2|shared_rho|251.191204286|217.965878644|CERTIFIED_PASS|0.5378239352954551|0.08778135311920841|
|A3|shared_rho|252.853801943|219.628476300|CERTIFIED_PASS|0.6708317477954553|0.08859225672734908|
|A4|shared_rho|254.516399599|221.291073956|CERTIFIED_FAIL|NOT_EVALUATED / no passing candidate|NOT_EVALUATED / no passing candidate|
|A5|shared_rho|256.178997255|222.953671613|CERTIFIED_FAIL|NOT_EVALUATED / no passing candidate|NOT_EVALUATED / no passing candidate|
|B0|F200_half_width|249.528606630|216.303280988|CERTIFIED_PASS|0.40481612279545515|0.08650902042768663|
|B1|F200_half_width|252.299602724|219.074277081|CERTIFIED_PASS|0.6264958103187382|0.08013069623574469|
|B2|F200_half_width|255.070598818|221.845273175|CERTIFIED_FAIL|NOT_EVALUATED / no passing candidate|NOT_EVALUATED / no passing candidate|
|B3|F200_half_width|257.841594911|224.616269269|CERTIFIED_FAIL|NOT_EVALUATED / no passing candidate|NOT_EVALUATED / no passing candidate|
|B4|F200_half_width|260.612591005|227.387265363|CERTIFIED_FAIL|NOT_EVALUATED / no passing candidate|NOT_EVALUATED / no passing candidate|
|B5|F200_half_width|266.154583193|232.929257550|CERTIFIED_FAIL|NOT_EVALUATED / no passing candidate|NOT_EVALUATED / no passing candidate|
|B6|F200_half_width|271.696575380|238.471249738|CERTIFIED_FAIL|NOT_EVALUATED / no passing candidate|NOT_EVALUATED / no passing candidate|
|C0|independent_rho|249.528606630|216.303280988|CERTIFIED_PASS|0.40481612279545515|0.08650902042768663|
|C1|independent_rho|251.191204286|217.965878644|CERTIFIED_PASS|0.5378239352954551|0.08271561945164849|
|C2|independent_rho|252.853801943|219.628476300|CERTIFIED_PASS|0.6708317477954553|0.07883574823475638|
|C3|independent_rho|254.516399599|221.291073956|CERTIFIED_FAIL|NOT_EVALUATED / no passing candidate|NOT_EVALUATED / no passing candidate|
|C4|independent_rho|256.178997255|222.953671613|CERTIFIED_FAIL|NOT_EVALUATED / no passing candidate|NOT_EVALUATED / no passing candidate|
|C2_0|independent_2D|249.528606630|216.303280988|CERTIFIED_PASS|0.40481612279545515|0.08156498623628375|
|C2_1|independent_2D|251.191204286|217.965878644|CERTIFIED_PASS|0.5378239352954551|0.0777856016483934|
|C2_2|independent_2D|252.853801943|219.628476300|CERTIFIED_PASS|0.6708317477954553|0.07390171005004505|
|C2_3|independent_2D|254.516399599|221.291073956|CERTIFIED_PASS|0.8038395602954553|0.06965936580680034|
|C2_4|independent_2D|249.528606630|216.303280988|CERTIFIED_PASS|0.40481612279545515|0.08650902042768663|
|C2_5|independent_2D|251.191204286|217.965878644|CERTIFIED_PASS|0.5378239352954551|0.08271561945164849|
|C2_6|independent_2D|252.853801943|219.628476300|CERTIFIED_PASS|0.6708317477954553|0.07883574823475638|
|C2_7|independent_2D|254.516399599|221.291073956|CERTIFIED_FAIL|NOT_EVALUATED / no passing candidate|NOT_EVALUATED / no passing candidate|
|C2_8|independent_2D|249.528606630|216.303280988|CERTIFIED_PASS|0.40481612279545515|0.09151896987313066|
|C2_9|independent_2D|251.191204286|217.965878644|CERTIFIED_PASS|0.5378239352954551|0.08778135311920841|
|C2_10|independent_2D|252.853801943|219.628476300|CERTIFIED_PASS|0.6708317477954553|0.08391062548633578|
|C2_11|independent_2D|254.516399599|221.291073956|CERTIFIED_FAIL|NOT_EVALUATED / no passing candidate|NOT_EVALUATED / no passing candidate|

## Requested answers

1. rho search: initial=1.0, growth=1.25, bisection=8, tolerance=1e-08; each attempt rebuilds sampled W and optimizes local K/M/geometry. Finite numerical acceptance, not a globally maximal admissible domain.

2. original maximum accepted rho: 1.1083984375; selected=1.1083984375; exact archived-scale reproduction=True

3. Family A maximum shared rho: 1.21923828125

4. Family A F200 actual upper: 252.85380194272335

5. Family A F200 nominal upper: 219.62847630011106

6. Family B maximum hF: 32.5

7. Family B F200 actual upper: 252.29960272397335

8. Family C pass: True

9. Family C maximum rhoF: 1.274658203125

10. Corresponding rhoP: 1.052978515625

11. shared-rho coupling: At the same F200 +15% window, shared +15% fails W coverage, unchanged P100 also fails, but P100 -5% / F200 +15% passes. This demonstrates joint-domain coverage coupling in this preregistered finite test; it is not cost-free independent expansion.

12. current applied residual authority: P100=[-3.2560894130464355,3.107448612665206]; F200=[-0.6072241742754312,0.40481612279545515], alpha=.1, reset reference

13. maximum geometry applied authority: C2_3: F200=[-1.205759330525431,0.8038395602954553]

14. authority increment: 0.3990234375000002

15. P100 effect: C2_3 P100=[-3.062119686483935,2.913478886102706]; independent candidates preserve actual P100 bounds, but state-specific baseline projection can still change authority.

16. failed candidates: A0, A4, A5, B2, B3, B4, B5, B6, C3, C4, C2_7, C2_11

17. most common limiting gates: [('nonlinear_W_coverage', 11), ('reference_input', 1)]

18. certified expanded UR exists: True

19. reclaimable F200 actual headroom: 4.98779296875

20. reclaimable F200 nominal headroom: 4.98779296875

21. current frozen-policy economic gain (%): 0.08650902042768663

22. all passing candidate economics: See complete sensitivity CSV; NOT_EVALUATED values are never zero-imputed.

23. maximum frozen-policy gain (%): C2_8: 0.09151896987313066

24. win fraction: current=0.5903999999999999; maximum-gain candidate=0.5891

25. X2 minimum margin: current=0.13229527485792403; maximum-gain candidate=0.1276241431630183

26. control TV/RMS: current P100 TV/RMS=3440.103263405856/4.341219779163623, F200=683.6545091909168/0.9038050820361931; maximum-gain P100=3438.8169485386834/4.339465949938124, F200=687.2211300664405/0.9115302032747454

27. QP modifications: current=0.0, maximum-gain=0.0; per-seed counts retained.

28. new empirical anomalies: Any counterfactual exception is retained and suppresses aggregate gain; inspect frozen_policy_geometry_sensitivity.csv, never score incomplete/unsafe rollouts.

29. UR limits economic headroom: Compare frozen-policy and geometry-only zero-residual gains separately. An improvement driven by a changed zero-residual base is geometry sensitivity, not new learning.

30. alpha=.1 bottleneck: Alpha is held fixed. This experiment cannot causally identify alpha versus critic/reward/learning as the remaining bottleneck.

31. next priority: Finish frozen-policy counterfactual after disk resources are available, before choosing geometry versus SAC changes. No production expansion is selected here. Only passing candidates with useful authority and favorable empirical quality justify a separately preregistered follow-up.

32. enlarged_ur_v1: Not created; not automatically authorized by this sensitivity result.

33. sensitivity CSV: C:\Users\cushy\PycharmProjects\evaporation\evaporation_safe_sac\economic_recovery_v1\ur_certificate_sensitivity\ur_certificate_sensitivity.csv

34. failure CSV: C:\Users\cushy\PycharmProjects\evaporation\evaporation_safe_sac\economic_recovery_v1\ur_certificate_sensitivity\ur_certificate_failure_reasons.csv

35. certificate/authority figure: C:\Users\cushy\PycharmProjects\evaporation\evaporation_safe_sac\economic_recovery_v1\ur_certificate_sensitivity\ur_certificate_vs_authority.png

36. authority/economics figure: C:\Users\cushy\PycharmProjects\evaporation\evaporation_safe_sac\economic_recovery_v1\ur_certificate_sensitivity\ur_authority_vs_economic_improvement.png

37. channel-scaling comparison: C:\Users\cushy\PycharmProjects\evaporation\evaporation_safe_sac\economic_recovery_v1\ur_certificate_sensitivity\shared_vs_independent_ur_scaling.csv

38. tests: Latest full run: 221 tests, passed=True. Latest run had one disk-write OSError(28); all 14 new contract tests passed. Earlier clean run: 221 main + 3 optional CasADi + evaporation.test passed. Final full-suite clean rerun pending resource resolution.

39. production preserved: True

40. training: False

41. final seeds used: False

42. commit: False

43. push: False


## Resume after resource resolution

Stop the separately running stage4 Oracle yourself if appropriate and free disk space. This study never stops that historical process or deletes its results.
From the repository parent, run the following offline-only commands:
```powershell
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_ur_certificate_sensitivity --tests
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_ur_certificate_sensitivity --counterfactual --resume --workers 1
& ".\evaporation\.venv-cuda\Scripts\python.exe" -m evaporation.zanon2019_ur_certificate_report
```
The candidate protocol, alpha and production hashes are checked on resume. No training entry is added.