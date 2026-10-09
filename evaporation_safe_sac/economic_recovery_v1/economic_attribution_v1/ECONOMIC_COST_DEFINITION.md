# Runtime economic cost

10.09*(F2+F3) + 600*F100 + 0.6*F200

Source: C:\Users\cushy\PycharmProjects\evaporation\model.py:108, EvaporatorModel.economic_cost.

Four additive components: 10.09 F2, 10.09 F3, 600 F100, 0.6 F200. F3=50 is fixed.

P100 is not directly priced: P100 -> T100 -> Q100 -> F100 and F4 -> F2.
At fixed x,d, increasing P100 raises steam cost and reduces F2 cost.
F200 has direct cooling cost; its indirect effects appear through subsequent state evolution.

not explicitly specified for the cost coefficients in runtime code; model cost units per stage. Do not assign an unverified currency/time unit.

economic_cost(next_state, final_applied_control, current_disturbance); existing post-state convention. J is the existing undiscounted stage-cost sum, with no added dt factor.

Training reward remains separate: paired cost gap/200 minus unchanged fixed excess-move penalty and safety dual.
