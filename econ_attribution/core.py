"""Physical-cost and actor attribution, separate from all training rewards."""
from __future__ import annotations

import inspect
import numpy as np

from .. import zanon2019_paired_v2 as v2
from .. import zanon2019_paired_experiment as io
from .. import zanon2019_ur_certificate_sensitivity as sensitivity
from ..zanon2019_input_tightening_audit import polytope_extrema

COMPONENTS = ("flow_F2", "recirculation_F3", "steam_F100", "cooling_F200")


def cost_components(model, state, control, disturbance):
    flows = model.algebraic(state, control, disturbance)
    parts = np.array([10.09 * flows.f2, 10.09 * model.cfg.recirculation_f3,
                      600. * flows.f100, .6 * control[1]])
    np.testing.assert_allclose(parts.sum(), model.economic_cost(state, control, disturbance), rtol=0, atol=1e-9)
    return parts


def cost_definition(model):
    fn = type(model).economic_cost
    source, line = inspect.getsourcelines(fn)
    return dict(file=inspect.getsourcefile(fn), function=fn.__qualname__, line=line,
        runtime_source="".join(source), formula="10.09*(F2+F3) + 600*F100 + 0.6*F200",
        components=dict(zip(COMPONENTS, ("10.09*F2", "10.09*F3", "600*F100", "0.6*F200"))),
        coefficients={"flow": 10.09, "steam": 600., "cooling": .6}, F3=model.cfg.recirculation_f3,
        variables=dict(X2="actual product concentration", P2="actual evaporator pressure",
            P100="steam pressure; indirect through T100/Q100/F100/F4/F2", F200="cooling flow; direct 0.6*F200",
            F2="F1-F4", F3="fixed recirculation flow", F100="Q100/latent_steam"),
        interpretation="all three groups are positive operating costs; F2 term is not a product sales reward",
        unit="not explicitly specified for the cost coefficients in runtime code; model cost units per stage",
        timing="economic_cost(next_state, final_applied_control, current_disturbance); existing post-state convention",
        integration="J=sum of 1000 existing stage costs, no new dt factor", training_reward_is_distinct=True)


def economic_arrays(records, env):
    model, d = env[1:3]
    states = np.array([r["state"] for r in records])
    controls = np.array([r["control"] for r in records])
    disturbances = np.array([r["disturbance"] for r in records])
    mismatch = np.array([r["w_hat"] for r in records])
    nxt = model.physical_state(model.normalized_state(states) @ d.a.T +
                              model.normalized_input(controls) @ d.b.T + d.affine + mismatch)
    if not np.allclose(nxt[:-1], states[1:], rtol=0, atol=1e-8):
        raise ValueError("Post-state timing mismatch")
    parts = np.array([cost_components(model, x, u, w) for x, u, w in zip(nxt, controls, disturbances)])
    actual = np.array([r["economic_cost"] for r in records])
    np.testing.assert_allclose(parts.sum(1), actual, rtol=0, atol=1e-7)
    return parts, nxt


def effect_split(production, own, sac):
    p, b, s = float(production), float(own), float(sac)
    if min(p, b) <= 0 or not np.isfinite([p, b, s]).all():
        raise ValueError("Finite positive economic denominators required")
    dg, ds, dt = b-p, s-b, s-p
    np.testing.assert_allclose(dg+ds, dt, rtol=0, atol=1e-8)
    gg, gs, gt = 100*(p-b)/p, 100*(b-s)/b, 100*(p-s)/p
    np.testing.assert_allclose(gg+(1-gg/100)*gs, gt, rtol=0, atol=1e-10)
    return dict(J_base_prod=p, J_base_geometry=b, J_SAC_geometry=s,
        DeltaJ_geometry=dg, DeltaJ_SAC_given_geometry=ds, DeltaJ_total=dt,
        Gain_geometry_pct=gg, Gain_SAC_given_geometry_pct=gs, Gain_total_vs_prod_pct=gt)


def distribution(values, eps=1e-6):
    values = np.asarray(values, float)
    if not len(values):
        return dict(count=0, mean=None, median=None)
    return dict(count=len(values), positive_fraction=float(np.mean(values > eps)),
        negative_fraction=float(np.mean(values < -eps)), zero_fraction=float(np.mean(abs(values) <= eps)),
        mean=float(values.mean()), median=float(np.median(values)), mean_abs=float(abs(values).mean()),
        **{f"p{q}": float(np.percentile(values, q)) for q in (5, 25, 75, 95)})


def midranks(values):
    values = np.asarray(values)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts)-counts
    return (starts + (counts-1)/2)[inverse]


def correlation(a, b, spearman=False):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.std(a) <= 1e-14 or np.std(b) <= 1e-14:
        return None
    if spearman:
        a, b = midranks(a), midranks(b)
    return float(np.corrcoef(a, b)[0, 1])


def replay_attribution(env, records, baseline, geometry, seed):
    """Replay original online mapping at archived states, never bypass QP.

    Local residual (same state/nominal context) differs from the closed-loop
    SAC-minus-paired-baseline input difference. Both are explicitly recorded.
    """
    cfg, model, design, omega, domain = env[:5]
    disturbance = np.array([r["disturbance"] for r in records])
    ctrl = io.PairedEvidenceController(cfg, model, design, omega, domain, disturbance)
    ctrl.reset(records[0]["state"])
    cp, _ = economic_arrays(records, env)
    bp, _ = economic_arrays(baseline, env)
    rows = []
    for k, (r, b) in enumerate(zip(records, baseline)):
        # Frozen SAC.select_action returns float32. CSV readers return float64;
        # restore the original actor-output dtype before the unchanged mapping.
        # This is diagnostic replay fidelity, NOT a relaxed QP/safety tolerance.
        applied, info = ctrl.act(r["state"], np.asarray(r["raw_action"], dtype=np.float32))
        np.testing.assert_allclose(applied, r["control"], rtol=0, atol=1e-7)
        center = model.physical_input(info["action_center_actual_norm"])
        requested = np.asarray(info["requested_residual"]) * cfg.input_scale
        residual = np.asarray(info["applied_residual"]) * cfg.input_scale
        np.testing.assert_allclose(applied-center, residual, rtol=0, atol=1e-8)
        lo, hi, _ = polytope_extrema(info["action_safe_rows"], info["action_safe_bounds"])
        c = np.asarray(info["action_center_coordinate"])
        neg, pos = sensitivity.ALPHA*(lo-c)*cfg.input_scale, sensitivity.ALPHA*(hi-c)*cfg.input_scale
        cost_base, cost_sac = bp[k].sum(), cp[k].sum()
        row = dict(geometry=geometry, validation_seed=seed, step=k, mode=info["mode"],
            X2=float(r["state"][0]), P2=float(r["state"][1]),
            delta_ell=float(cost_sac-cost_base), g_t=float(100*(cost_base-cost_sac)/cost_base),
            raw_actor_rho=float(np.max(abs(r["raw_action"]))),
            mapping_saturation=bool(np.any(abs(r["raw_action"]) >= .99)),
            QP_modification=float(info["final_verification_gap"]) > 1e-8)
        for j, name in enumerate(("P100", "F200")):
            app = residual[j]
            available = pos[j] if app > 0 else -neg[j]
            row.update({f"raw_actor_{name}": float(r["raw_action"][j]),
                f"scaled_actor_{name}": float(info["scaled_actor_action"][j]),
                f"mapped_residual_{name}": float(requested[j]), f"applied_residual_{name}": float(app),
                f"QP_input_{name}": float(center[j]+requested[j]), f"final_input_{name}": float(applied[j]),
                f"local_zero_input_{name}": float(center[j]), f"paired_baseline_input_{name}": float(b["control"][j]),
                f"paired_input_difference_{name}": float(applied[j]-b["control"][j]),
                f"negative_authority_{name}": float(neg[j]), f"positive_authority_{name}": float(pos[j]),
                f"authority_utilization_{name}": float(abs(app)/available) if available > 1e-12 else None})
        row.update({name: float(r["disturbance"][j]) for j, name in enumerate(("F1", "X1", "T1", "T200"))})
        row.update({f"component_gain_{name}": float(bp[k, j]-cp[k, j]) for j, name in enumerate(COMPONENTS)})
        rows.append(row)
    return rows


class StateAffineWitness:
    """Observable state-only policy. No time, disturbance or future path input."""
    def __init__(self, bias, gain, state_scale):
        self.bias = np.asarray(bias, float)
        self.gain = np.asarray(gain, float).reshape(2, 2)
        self.state_scale = np.asarray(state_scale, float)

    def select_action(self, obs, deterministic=True):
        # obs[:2] is the clipped normalized actual x-reference, not moving z.
        # Match the SAC information set and float32 observation exactly.
        phi = np.asarray(obs[:2], float) * self.state_scale
        return np.clip(self.bias+self.gain @ phi, -1., 1.)


def witness_parameters(count=512, seed=1902019):
    if count != 512:
        raise ValueError("Only the preregistered 512-candidate budget is allowed")
    from itertools import product
    params = [dict(candidate=0, family="state_only_affine", bias=[0., 0.], gain=[[0., 0.], [0., 0.]])]
    params += [dict(candidate=i+1, family="state_only_affine_constant_control", bias=list(b),
                   gain=[[0., 0.], [0., 0.]]) for i, b in enumerate(product((-.75, -.25, .25, .75), repeat=2))]
    rng = np.random.default_rng(seed)
    n = count-len(params)
    lhs = np.column_stack([(rng.permutation(n)+rng.uniform(size=n))/n for _ in range(6)])
    # Fixed physical-deviation gains, preregistered before any result; no adaptive tuning.
    lower = np.array([-.95, -.95, -12., -3., -12., -3.])
    upper = -lower
    for row in lower+(upper-lower)*lhs:
        params.append(dict(candidate=len(params), family="state_only_affine", bias=row[:2].tolist(),
                           gain=row[2:].reshape(2, 2).tolist()))
    return params


def pareto_front(rows):
    safe = [r for r in rows if r.get("safety_passed")]
    keys = ("P100_TV_ratio", "F200_TV_ratio", "P100_RMS_du_ratio", "F200_RMS_du_ratio", "X2_std", "centered_X2_MAE")
    def vector(r):
        return np.r_[-r["mean_improvement_pct"], [r[k] for k in keys], -r["minimum_X2_margin"]]
    return [r for r in safe if not any(np.all(vector(t) <= vector(r)) and np.any(vector(t) < vector(r)) for t in safe)]
