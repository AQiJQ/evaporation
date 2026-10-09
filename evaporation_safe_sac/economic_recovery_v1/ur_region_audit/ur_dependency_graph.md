# U_R dependency graph

```text
physical U + seed reference u_c + configured input half-window [35,30]
                          + scalar rho
                              |
                    direct rectangular U_R(rho)
                              |
 X_R(rho) + nonlinear mismatch samples over X_R x U_R x external domain
                              |
                        W hull/inflation
                              |
                  A/B, Hinf K -> RPI Z/support
                              |
           X_R minus Z + U_R minus KZ -> box invariant S
                              |
               verification QP + residual authority floor
                              |
        joint candidate gate -> scale scan/growth/bisection -> rho_selected

FROZEN RUNTIME
 U_R minus KZ + S -> Z nominal mapping/QP -> v + Ke -> actual u in U_R
 U_R -> q bounds + A/B/W -> Omega predecessor construction -> Omega QP
 Omega/A/B/W/q bounds -> later finite-jump Gm and Bj -> event supervisor
```

U_R is independent of finite-jump machinery. The code dependency is downstream
from U_R to Omega/Gm/Bj, not upstream. W/Z/S indirectly affect the original
accepted scale; they are not separate additive facets in the direct U_R box.
No failure gate for the exact original balanced-B scale boundary is inferred
from neighboring-reference logs. New region/gain/set certification is not run.
