# ECC2019 qualitative literature comparison

## Source-supported observations

Source: Practical Reinforcement Learning of Stabilizing Economic MPC, supplied PDF.
Numerical Example: PDF page 4 / printed 2261. Fig.2: PDF page 6 / printed 2263.
The process uses X2/P2 states and P100/F200 inputs; the X2 lower quality bound is 25.
The exogenous quantities fluctuate about nominal values. The reported variances
are Var(F1)=2, Var(X1)=1, Var(T1)=8, Var(T200)=5. State bounds are relaxed
through slack variables in Eq.(3e), with linear and quadratic slack costs.
PDF page 5 / printed 2262 states:
> Indeed, the constraint is violated but only rarely and by small amounts.

The paper discusses the economics/constraint trade-off and reports 14% and 12%
gains relative to its naive and nominal-economic NMPC comparators, respectively.
These are NOT gains relative to our robust controller. Fig.2 displays stochastic
X2 responses and an instantaneous economic cost difference. Its explicit algebraic
sign convention and exact percentage-gain formula are not specified in the paper.

## Our observations

See the internally paired table below. Our curves and distributions use archived
validation realizations, not recovered author data. All three internal methods
have zero observed physical state/input violations in these archived evaluations.
The concentration bands in our presented plots are narrow on a 24–30 axis.

## Comparison scope

qualitative external comparison, quantitative internal ablation.
Gaussian distribution, independent per-step sampling, 1 s sampling, initial state
B and a 1000-step horizon are implementation choices for reproduction; the exact
distribution, sampling convention, initial state and seed/realization matching
are not specified in the paper. We do not assert matched external protocols.
Do not infer ECC2019 std, IAE, violation rate or maximum violation from pixels.
No quantitative economic or disturbance-rejection superiority over ECC2019 is
established. A visibly narrower plotted band across these separate simulations
is a qualitative observation, not a controlled numerical superiority result.

## Suggested citation caption

ECC2019 Fig.2 (printed p.2263; PDF p.6): reported stochastic concentration
responses and economic-cost comparison of RL-tuned and reference NMPC controllers.
Reproduced only if the thesis author chooses to quote the original figure with
appropriate attribution/permission; no author trajectory has been redrawn here.

## Suggested body reference

ECC2019 reports rare and small violations of the X2 quality constraint, whereas
no X2<25 violation was observed in our evaluated validation trajectories. The
present curves form a visibly narrow band on a similar axis scale, but differences
in unspecified implementation details and random realizations prevent a matched
quantitative comparison. Both studies illustrate an economics/constraint trade-off.

|Method|J_econ|Gain %|Worst gain %|X2 mean|Mean within-seed std|Global X2 range|X2 IAE ratio (B)|P2 IAE ratio (B)|P100 TV ratio|F200 TV ratio|
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
|baseline|6217912.952|0.000000|0.000000|25.384924|0.043027|25.212371–25.539580|1.000000|1.000000|1.000000|1.000000|
|A_alpha010|6216471.785|0.023161|0.009478|25.369345|0.046562|25.178676–25.530982|1.181462|0.989078|0.825742|1.315386|
|B_alpha020_rewardcal|6215268.649|0.042468|0.017342|25.350740|0.049341|25.160545–25.556139|1.538162|0.938511|1.048697|0.985507|
