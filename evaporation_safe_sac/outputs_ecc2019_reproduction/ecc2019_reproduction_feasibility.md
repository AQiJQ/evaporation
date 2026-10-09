# ECC2019 reproduction feasibility (before controller implementation)

Source: Practical Reinforcement Learning of Stabilizing Economic MPC,
ECC2019 pp.2258-2263, especially Eq.(2),(3),(8), Numerical Example pp.2261-2262,
Fig.2 p.2263. The supplied PDF was reread, including visual review of Fig.1/2.

## specified_in_paper

- States X2/P2; inputs P100/F200; nominal predictor treats exogenous quantities as constant.
- Disturbance variances in F1/X1/T1/T200 order: [2,1,8,5], mean at nominal values.
  The prose says F2 once, while its variance subscript is F1: record this discrepancy.
- State bounds [25,40] to [100,80]; actual input bounds [100,100] to [400,400].
- NON-condensed nonlinear MPC, horizon N=10. Eq.(3) defines Q by fixing x0=s,u0=a;
  V/policy remove the u0=a constraint. Quadratic arrival lambda, stage ell_theta,
  terminal Vf_theta, constant additive predictor bias cf, learned state bounds.
- Slack objective sum gamma^k*(s_k^T I s_k + 1^T s_k), k=0..N;
  all state bounds relaxed, input bounds hard.
- Eq.(8) squared TD fit to frozen-target V, with positive stage/terminal Hessians;
  damped update theta <- theta + .01*(theta_star-theta); policy refresh every500 steps.
- Greedy90%, exploratory10%; printed exploration sat(e,ul,uu), e~N(0,sqrt(10)).
- Initial Hl=I, xl=[25,40], xu=[100,80], other parameters zero.
- 14% gain relative to naive NMPC and12% relative to nominal-economic NMPC.
  These numbers are NOT the safe robust baseline gain and NOT a substitute for simulated J.
- Fig.2 plots RL concentration, naive/nominal-economic concentration, instantaneous
  cost difference over axis0..1000. It supports qualitative inspection, not digitization.

## derived_from_cited_model/current_common_evaporation_model

- Same existing two-state nonlinear evaporator, algebraic flows and economic cost
  10.09*(F2+F3)+600*F100+.6*F200; nominal [F1,X1,T1,T200]=[10,5,40,25].
- ECC2019 explicitly omits equations/economic expression and refers to [26],[27].
  Current model correspondence is documented, not claimed as a verbatim ECC2019 equation.

## implementation_choice_for_reproduction / not specified in the paper

- Gaussianity, cross-variable independence, independent resampling each step,
  random seeds/realization, initial physical state and numeric sampling time.
- Use the frozen current stochastic generator and1s RK4 plant step; all methods
  start at balanced B=[25.39,50.125];1000 steps is a common protocol choice,
  not proof that the paper's axis units are seconds.
- Numeric discount gamma, numerical solver/tolerances, exact quadratic factor convention,
  training batch/sample window, training length, terminal constraint h_f and gauge handling.
- Quadratic coordinate origin is also not specified. Use physical-unit deviations
  from nominal x=[25,49.743] and its nonlinear steady input as one predeclared basis.
  Hl=I is interpreted in THIS basis, not presented as the authors' confirmed origin.
  A raw-origin quadratic I penalizes inputs towards100 even at the operating point;
  these origins span the same quadratic function class but differ in initialization.
  No basis is tuned against Fig2 or held-out observations.
- The paper writes h=(x-xl,xu-x) but h<=s cannot represent the stated bounds with
  nonnegative slack. Implement [xl-x,x-xu]<=s to match the physical bounds and report
  the printed sign discrepancy, not silently call it the original printed formula.
- Initial Vf=0 conflicts with strict Hessian Vf>0. Use an explicitly logged tiny
  positive Hessian initialization/floor, never pretend both printed conditions hold exactly.
- Physical-origin interpretation of sat(N(0,sqrt10),100,400) almost always yields100.
  Preserve and flag this literal interpretation; do NOT silently center noise at greedy input.
- Algebraic percentage-gain formula and cost-difference sign are not explicitly defined.
  Report raw paired J; chosen gain=(J_reference-J_RL)/J_reference*100 and difference
  cost_RL-cost_naive are implementation choices. Do not force12%/14% agreement.

## Algorithmically consequential missing artifacts

No authors' source, original sampling/discretization, learned theta, exact random array,
numeric gamma or solver setup is supplied. Current repository has SAC actors, NOT an
ECC2019 trained parameter artifact. Naive initialization MUST NOT be labeled RL-tuned.

Therefore exact reproduction cannot be claimed. Label trained-controller results:
**ECC2019_reproduction_with_documented_assumptions**.
Until independently fitted ECC parameters exist and pass solver/PD/qualitative checks,
ECC metrics and matched-economic conclusions are unavailable. No fabricated Fig.2 curves.

## Frozen comparison protocol

Keep robust baseline K/W/Z/tightening/B/QP unchanged. Keep SAC actors/weights unchanged.
Keep Gm/Bj inactive for stochastic Experiment I. Development seeds420000..420009.
Final seeds430000..430049 require explicit later approval and prelocked ONE SAC candidate.
ECC learner training belongs on the user's machine; no SAC training is requested.
No final-test access, commit or push in this phase.
