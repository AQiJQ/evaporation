# U_R region provenance and conservatism audit

## Definition and direct active source

U_R(rho,u_c) = U intersect product_i [u_c_i - rho h_i, u_c_i + rho h_i], h=[35,30].
rho=1.1083984375; u_c=[194.861442968292, 216.276653505223].
First builder: {'file': 'C:\\Users\\cushy\\PycharmProjects\\evaporation\\theta_learning.py', 'function': 'OnlineThetaLearner._robust_region_bounds', 'line': 334, 'expression': None}; downstream pre-KZ intersection: {'file': 'C:\\Users\\cushy\\PycharmProjects\\evaporation\\control.py', 'function': 'build_safety_design', 'line': 735, 'expression': 'physical_u_hi = np.minimum'}.
Actual-input U_R lower/upper: [156.067497655792, 183.024700380223] / [233.655388280792, 249.528606630223].
F200: min(400, 216.276653505223 + 1.1083984375 * 30) = 249.528606630223.
P100: min(400, 194.861442968292 + 1.1083984375 * 35) = 233.655388280792.
This is NOT an LP-generated maximal input set, U intersect Pre(S), or an all-input next-state invariance certificate. The box defines the declared nonlinear-mismatch/actual-input domain. Admissible online actions must additionally pass the corresponding state-dependent QP.
The active support-LP facet for F200 is local_window_F200_upper; for P100 it is local_window_P100_upper. The constructed static LP has multiplier 1 for that facet. The original builder has no LP dual. Other input can be interior, so incidental active corner facets need not be mistaken for the limiting F200 facet.
These facets mean reference-centered local operating-input limits, NOT a P2/X2/S/Omega/Gm/Bj state facet. They are state-independent and mode-independent. The construction is reference-dependent; both current references inherit the frozen Strong box, so runtime physical U_R does not shift with Conservative reference.

## Original scale selection and evidence limit

Original search: {'file': 'C:\\Users\\cushy\\PycharmProjects\\evaporation\\theta_learning.py', 'function': 'OnlineThetaLearner.build_certified_robust_operating_design', 'line': 637, 'expression': None}. The grow/bisect process resamples the nonlinear mismatch domain and can refine gains/W at each candidate scale, then checks Hinf/RPI/tightening/reference/S/QP/authority gates.
The exact balanced-B candidate CSV saves selected scale 1.1083984375 and its accepted gates. certified_reference_search saves final NPZ/row, not the failed-scale diagnostic trace. Therefore the specific gate that rejected the next larger scale cannot be recovered from these artifacts. Logs for the different [25.3825,50.1125] candidate are NOT substituted. No fresh redesign is performed to manufacture the missing history.
W/Z/S matter INDIRECTLY to scale acceptance. At frozen rho, removing a nonexistent direct W/P2/S/Omega term from the U_R formula changes its upper by exactly zero. This is NOT a claim that removing W/S in a new certification search would leave the selected rho unchanged.

## Addition/removal decomposition

| Stage | P100 upper | F200 upper |
|---|---:|---:|
| Physical U | 400.000000000 | 400.000000000 |
| Add local P100 window | 233.655388281 | 400.000000000 |
| Add local F200 window: U_R | 233.655388281 | 249.528606630 |
| Subtract directional KZ: nominal interval | 225.935929095 | 216.303280988 |

Direct local-window upper reduction: P100=166.344611719208; F200=150.471393369777.
Removing only F200 local upper restores max F200=400 in the standalone box audit. Removing P100 upper does not change F200 max. Perturbing only F200 upper RHS by diagnostic epsilon=1e-6 increases max F200 by approximately 1e-6 (unit sensitivity). Solver/gate tolerances and production inequalities remain unchanged.
There is no evidence of multiple state facets forming the direct U_R upper. W-off direct-box audit stays 249.528606630223 at frozen rho. The extra KZ subtraction is 33.225325642612; it is a separate downstream tube requirement, not part of the 400 -> U_R reduction.

## P100 versus F200

P100 is actually MORE reduced at the U_R layer (400 ->233.655388), compared with F200 (400 ->249.528607). The important asymmetry occurs after KZ tightening: P100 max support ~7.71946, F200 ~33.22533. F200 support nearly uses all its original upper half-window 33.251953125; P100 retains large steady-reference headroom. P2 dominates KZ, but does NOT identify the direct U_R active constraint.
Physical one-second B: [[ 8.097892294501e-04,  4.525091412688e-08],
 [ 1.595766150331e-04, -2.849668234717e-05]].
F200 -> Q200 = UA2*(T3-T200)/(1+UA2/(2 Cp F200)) -> F5 -> (F4-F5)/pressure_capacitance; at fixed x P100, F4 independent of F200
F200 increase directly increases F5 and lowers P2; small positive next-X2 effect is indirect via pressure/temperature coupling; this does not establish a P2-active U_R facet

## Reference margins

| Reference/channel | To physical upper | To U_R upper | To nominal tightened upper | Applied residual min/max at reset |
|---|---:|---:|---:|---|
| Strong/P100 | 205.138557032 | 38.7939453125 | 31.0744861267 | -3.256089413, 3.107448613 |
| Strong/F200 | 183.723346495 | 33.251953125 | 0.0266274823877 | -0.607224174, 0.404816123 |
| Primary_Conservative/P100 | 205.064908503 | 38.7202967841 | 31.0008375983 | -3.263454266, 3.100083760 |
| Primary_Conservative/F200 | 183.696719022 | 33.2253256527 | 1.00412762549e-08 | -0.607224174, 0.404816123 |

Neither Strong nor Conservative steady F200 is near the actual U_R upper: both retain about33.23 physical units. Conservative is near only the NOMINAL STEADY REFERENCE upper. Safe base projection gives reset F200≈212.25512 and retains bidirectional residual freedom.

## Z/Omega and finite-jump dependencies

Z-mode: actor -> interior-anchor nominal candidate -> nominal QP (U_R minus KZ and next-z in S) -> u_actual=v+Ke -> robust actual clip. Omega-mode: actor -> actual-q polytope candidate -> QP (u_ref+q in U_R, robust next-xi in Omega) -> applied actual u. There is no second KZ subtraction from Omega actual q.
U_R is independent of finite-jump machinery. U_R/q bounds feed Omega and later Gm/Bj; these sets do not construct or shrink the saved U_R. ECC2019 stochastic use has no active finite-jump automaton.

## Archived-state authority attribution

24 archived-state contexts, plus two reference reset contexts; 442 static action probes. Historical x/action replay recovers moving z without plant integration or sampling any new disturbance path. Replay source SHA and exact controls are checked.
Mode probe counts: {'Z_mode_existing_controller': 238, 'Omega_safe_one_step_QP': 204}. QP infeasible=0; QP modifications=0; actual clips=0; mapping alpha contractions=416.
See authority_target_facets.csv for FULL boundary-target active families, residual_authority_attribution.csv for per-probe actual clipping/projection, residual_authority_ranges.csv for exact ranges, and point_constraint_removal.csv for isolated S/Omega/W/local-window sensitivity.
The boundary target uses full safe-polytope authority. The frozen alpha=0.1 contracts the raw-action image toward the safe baseline before final QP. A limiting boundary target is not evidence that final applied action was clipped. Static frequencies are not 1000-step learned-policy clipping rates.

## Conservatism classification and conclusions

A Necessary under current construction: hard physical U, declared X_R/U_R mismatch domain, bounded W conditional tube certificate, directional KZ subtraction for v+Ke, and state-dependent S/Omega QP. Enlarging the declared domain invalidates relying on the old coverage/certificate without re-audit.
B Approximation: axis-aligned domain windows, box controlled-invariant S, finite W hull/inflation and eigenbasis-tail support enclosure. No globally maximal-domain comparison is available, so the certified recoverable F200 headroom is NOT quantified. The saved polygon/directional-support gap has essentially zero F200 magnitude; its P100 certificate-contract concern remains in the previous audit.
C Implementation choices: h=[35,30], shared rho for X_R/U_R channels, finite grow/bisect/gain search, baseline reserve and alpha=0.1 mapping. They are not universal necessities of Hinf/RPI theory. The physical-minus-window gap is a geometric cap, NOT evidence all of it can safely be recovered.
D No obvious duplicate mathematical subtraction was found. U_R and nominal U_R minus KZ apply to different variables; final input rows/clips enforce actual bounds. Do not confuse declared local domain and online feasible action set.
Economic headroom is geometrically restricted by U_R and downstream nominal tightening, but actual economics limitation cannot be causally assigned mainly to U_R from this static audit. Strong/Conservative difference is largely steady nominal admissibility; current effective authority is also strongly contracted by the frozen mapping scale. No safe gain from relaxing U_R is established.
If authorized in a later task, study domain/aspect-ratio and certificate-aware geometry feasibility before more blind SAC training. Do not automatically enlarge U_R, change W/sets, or alter mapping based on these counterfactuals. A new domain requires coverage validation and complete relevant re-certification, including downstream Omega/Gm/Bj when affected.

## Preservation and tests

All production sources, K/W/Z/S/Omega, references, previous audit outputs and finite-jump saved files were hash-checked unchanged. No optimization result was connected to the controller. No SAC training, Oracle continuation, new final seeds, commit or push. Test results are in tests_receipt.json.
