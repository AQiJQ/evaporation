"""Candidate C components. Frozen controller and SAC sources are not edited.

Implementation choice for reproduction: safety costs are event indicators
for physical state/input, QP, Omega and robust-region failures. Immediate
abort remains primary. No IAE/TV performance multipliers survive from v1.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .sac import ReplayBuffer
from .zanon2019_training_report import relative
from .zanon2019_paired_experiment import SAFETY

SAFETY_KEYS = ("physical_state", "physical_input", "qp", "omega", "robust_region")
CRITERION = "economic_safety_smoothness_v2"


@dataclass
class SafetyDual:
    values: np.ndarray
    learning_rate: float = .1
    cap: float = 100.

    @classmethod
    def create(cls):
        return cls(np.zeros(5))

    def update(self, costs):
        g = np.asarray(costs, float)
        if g.shape != (5,) or not np.isfinite(g).all() or np.any(g < 0):
            raise ValueError("Invalid safety event costs")
        self.values = np.clip(self.values + self.learning_rate * g, 0, self.cap)


def paired_move(control, previous, base_control, base_previous, scales):
    scales = np.asarray(scales, float)
    if scales.shape != (2,) or np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("Two positive fixed physical input scales required")
    ds = (np.asarray(control) - previous) / scales
    db = (np.asarray(base_control) - base_previous) / scales
    return np.maximum(0., ds**2 - db**2)


def components(records, baseline, scales, initial_input):
    """Raw columns: economics, smoothness, five safety event indicators.

    Initial previous input is physical(v_ref), exactly run_episode semantics.
    Performance uses pre-step samples; safety includes terminal next state.
    """
    if not records or len(records) != len(baseline):
        raise ValueError("Incomplete paired trajectory")
    if not np.array_equal(records[0]["state"], baseline[0]["state"]):
        raise ValueError("Unpaired initial state")
    result = []
    for k, (r, b) in enumerate(zip(records, baseline)):
        if not np.array_equal(r["disturbance"], b["disturbance"]):
            raise ValueError(f"Unpaired disturbance at {k}")
        result.append(step_component(r, b, records[k-1]["control"] if k else initial_input,
                                     baseline[k-1]["control"] if k else initial_input, scales))
    result = np.asarray(result)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite reward components")
    return result


def step_component(r, b, previous, base_previous, scales):
    if not np.array_equal(r["disturbance"], b["disturbance"]):
        raise ValueError("Unpaired disturbance")
    excess = paired_move(r["control"], previous, b["control"], base_previous, scales)
    events = [float(bool(r.get(key, False))) for key in
              ("physical_state_violation", "physical_input_violation")]
    events += [float(not bool(r.get("qp_feasible", True))),
               float(not bool(r.get("in_Omega_before", True))),
               float(bool(r.get("robust_region_violation", False)))]
    return np.r_[(b["economic_cost"]-r["economic_cost"])/200., excess.sum(), events]


def decomposition(raw, weight, dual):
    raw = np.asarray(raw, float)
    if raw.ndim != 2 or raw.shape[1] != 7 or not np.isfinite(raw).all():
        raise ValueError("Expected finite seven-component transitions")
    econ = float(raw[:, 0].sum())
    smooth = -float(weight * raw[:, 1].sum())
    lagrange = -float(raw[:, 2:].sum(axis=0) @ dual)
    denom = float(np.abs(raw[:, 0]).sum())
    return dict(economic_reward_component=econ, smoothness_regularization_component=smooth,
                lagrangian_constraint_component=lagrange, total_training_reward=econ+smooth+lagrange,
                regularizer_fraction=abs(smooth)/denom if denom > 1e-12 else None,
                regularizer_fraction_defined=denom > 1e-12,
                sum_abs_economic_reward=denom, smoothness_penalty_raw_total=float(raw[:, 1].sum()))


def operating_metrics(records, terminal_minimum=None):
    x = np.array([r["state"][0] for r in records])
    mean = float(x.mean())
    minimum = float(x.min()) if terminal_minimum is None else min(float(x.min()), terminal_minimum)
    return dict(mean_X2=mean, std_X2=float(x.std()), centered_X2_MAE=float(np.abs(x-mean).mean()),
                minimum_X2=minimum, minimum_X2_margin=minimum-25.,
                distance_to_boundary_integral=float(np.abs(x-25.).sum()))


def audited_metrics(records, env):
    """Recompute archived/live metrics without fabricating missing legacy rewards."""
    from .zanon2019_benchmark import trajectory_metrics
    from .controlled_invariant_error_set import facets
    cfg, model, design, omega = env[:4]
    rows = [dict(r, omega_exit_event=not r["in_Omega_before"]) for r in records]
    result = trajectory_metrics(rows, cfg, model, design)
    x = np.array([r["state"] for r in rows])
    u = np.array([r["control"] for r in rows])
    xn, un = model.normalized_state(x), model.normalized_input(u)
    next_n = xn@design.a.T+un@design.b.T+design.affine+np.array([r["w_hat"] for r in rows])
    next_x = model.physical_state(next_n)
    if not np.allclose(next_x[:-1], x[1:], rtol=0, atol=1e-8):
        raise ValueError("Archived/live post-state timing mismatch")
    # Verify complete costs, not only a displayed aggregate.
    actual_cost = [model.economic_cost(y, ui, r["disturbance"]) for y, ui, r in zip(next_x, u, rows)]
    if not np.allclose(actual_cost, [r["economic_cost"] for r in rows], rtol=0, atol=1e-7):
        raise ValueError("Paired stage cost does not match nonlinear post-state timing")
    all_x = np.vstack([x[0], next_x])
    result["physical_state_violation_steps"] = int(np.sum(np.any(
        (all_x < cfg.state_lower-1e-8)|(all_x > cfg.state_upper+1e-8), axis=1)))
    result["X2_margin_min"] = float(all_x[:, 0].min()-25.)
    h, b = facets(omega)
    result["Omega_exit_count"] = int(np.sum(np.max(
        (np.vstack([xn[0], next_n])-design.z_ref)@h.T-b, axis=1) > 1e-8))
    robust_x = np.vstack([xn[0], next_n])
    result["robust_region_violation_count"] = int(np.sum(np.any(
        (robust_x < design.robust_state_lower-1e-8)|(robust_x > design.robust_state_upper+1e-8), axis=1)))
    result["robust_region_violation_count"] += int(np.sum(np.any(
        (un < design.robust_input_lower-1e-8)|(un > design.robust_input_upper+1e-8), axis=1)))
    raw = np.array([r["raw_action"] for r in rows])
    requested = np.array([r["residual"] for r in rows])
    applied = np.array([r["applied_residual"] for r in rows])
    nonzero = np.max(np.abs(raw), axis=1) > 1e-8
    displacement = np.array([r["applied_displacement_from_baseline"] for r in rows])
    collapse = nonzero & (displacement < 1e-6)
    nr, na = np.linalg.norm(requested, axis=1), np.linalg.norm(applied, axis=1)
    result.update(action_collapse_fraction=float(collapse.sum()/max(nonzero.sum(), 1)),
        nonzero_actor_steps=int(nonzero.sum()), collapsed_action_steps=int(collapse.sum()),
        raw_actor_saturation_fraction=float(np.mean(np.any(np.abs(raw) >= .99, axis=1))),
        Z_mode_fraction=float(np.mean([r["safety_mode"] == "Z_mode_existing_controller" for r in rows])),
        Omega_mode_fraction=float(np.mean([r["safety_mode"] == "Omega_safe_one_step_QP" for r in rows])),
        residual_requested_norm_mean=float(nr.mean()), residual_applied_norm_mean=float(na.mean()),
        residual_applied_ratio=relative(float(na.sum()), float(nr.sum())),
        QP_modification_rate=float(np.mean([float(r.get("final_verification_gap") or 0) > 1e-8 for r in rows])),
        supervisor_modification_rate=0., any_outer_10pct_fraction=result["any_outer_10pct_steps"]/len(rows))
    return result


def paired_metrics(records, base, env, weight, dual):
    cfg, model, design, omega = env[:4]
    metric = audited_metrics(records, env)
    bm = audited_metrics(base, env)
    if any(metric[k] != 0 or bm[k] != 0 for k in SAFETY):
        raise RuntimeError("v2 immediate safety abort: unsafe paired trajectory")
    op = operating_metrics(records, 25.+metric["X2_margin_min"])
    bop = operating_metrics(base, 25.+bm["X2_margin_min"])
    raw = components(records, base, cfg.input_scale, model.physical_input(design.v_ref))
    out = dict(**op, baseline_minimum_X2=bop["minimum_X2"], baseline_minimum_X2_margin=bop["minimum_X2_margin"],
        mean_X2_shift_vs_baseline=op["mean_X2"]-bop["mean_X2"],
        X2_std_ratio=relative(op["std_X2"], bop["std_X2"]),
        centered_X2_MAE_ratio=relative(op["centered_X2_MAE"], bop["centered_X2_MAE"]),
        J_econ=metric["J_econ"], baseline_J_econ=bm["J_econ"],
        economic_improvement_pct=100.*(bm["J_econ"]-metric["J_econ"])/bm["J_econ"],
        **decomposition(raw, weight, dual),
        **{key: metric[key] for key in SAFETY}, W_exceedance_rate=metric["W_exceedance_rate"],
        **{key: metric[key] for key in ("action_collapse_fraction", "nonzero_actor_steps", "collapsed_action_steps",
            "raw_actor_saturation_fraction", "Z_mode_fraction", "Omega_mode_fraction",
            "residual_requested_norm_mean", "residual_applied_norm_mean", "residual_applied_ratio",
            "QP_modification_rate", "supervisor_modification_rate", "any_outer_10pct_fraction")})
    for key in ("X2_IAE", "X2_ISE", "P2_IAE", "P2_ISE", "P100_TV", "F200_TV",
                "P100_delta_RMS", "F200_delta_RMS", "P100_input_sign_changes", "F200_input_sign_changes"):
        ratio_name = key.replace("delta_RMS", "RMS_du").replace("input_sign_changes", "sign_change")+"_ratio"
        out.update({key: metric[key], "baseline_"+key: bm[key], ratio_name: relative(metric[key], bm[key])})
    out["total_control_activity"] = sum(out[key]/cfg.input_scale[i]/len(records)
                                          for i, key in enumerate(("P100_TV", "F200_TV")))
    out["joint_v2"] = joint([out])
    return out, raw


def joint(rows):
    """Oracle criterion; tracking ratios never gate v2 eligibility."""
    if not rows:
        return False
    return bool(all(r["economic_improvement_pct"] > 0
                    and r["P100_TV_ratio"] <= 1. and r["F200_TV_ratio"] <= 1.
                    and (bool(r["safety_passed"]) if "safety_passed" in r
                         else all(r[k] == 0 for k in SAFETY)) for r in rows))


class EconomicReplay(ReplayBuffer):
    """Sample-time economics - FIXED smoothness - CURRENT safety multipliers."""
    def __init__(self, obs_dim, action_dim, capacity, device, dual, weight):
        super().__init__(obs_dim, action_dim, capacity, device)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Invalid immutable smoothness weight")
        self._weight = float(weight)
        self.dual = dual
        self.components = np.zeros((capacity, 7), dtype=np.float32)
        self.context = None

    @property
    def weight(self):
        return self._weight

    @property
    def allocated_bytes(self):
        return super().allocated_bytes+self.components.nbytes

    def bind(self, ctrl, base):
        self.context = ctrl, base
        self.episode_records, self.episode_components = [], []

    def add_components(self, obs, action, raw, next_obs, done, execution_mask=1.):
        raw = np.asarray(raw, np.float32)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise ValueError("Seven independent finite components required")
        position = self.ptr
        super().add(obs, action, 0., next_obs, done, execution_mask)
        self.components[position] = raw

    def add(self, obs, action, legacy_reward, next_obs, done, execution_mask=1.):
        if self.context is None:
            raise RuntimeError("Unbound paired replay")
        ctrl, base = self.context
        row = ctrl.evidence[-1]
        k = len(self.episode_records)
        if row["step"] != k or not row["next_state_observed"]:
            raise RuntimeError("Replay requires an audited actual transition")
        state, next_state, control = [np.array([row[f"{key}_{i}"] for i in range(2)])
                                      for key in ("state", "next_state", "control")]
        record = dict(state=state, control=control,
            disturbance=np.array([row[f"disturbance_{i}"] for i in range(4)]),
            economic_cost=ctrl.model.economic_cost(next_state, control, base[k]["disturbance"]))
        # Safety audit runs BEFORE this hook and aborts unsafe transitions.
        if k == 0 and not np.array_equal(record["state"], base[0]["state"]):
            raise ValueError("Unpaired initial training state")
        initial = ctrl.model.physical_input(ctrl.d.v_ref)
        raw = step_component(record, base[k], self.episode_records[-1]["control"] if k else initial,
                             base[k-1]["control"] if k else initial, ctrl.cfg.input_scale)
        self.episode_records.append(record)
        self.episode_components.append(raw)
        self.add_components(obs, action, raw, next_obs, done, execution_mask)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        raw = torch.as_tensor(self.components[idx], device=self.device)
        dual = torch.as_tensor(self.dual.values, dtype=raw.dtype, device=self.device)
        reward = raw[:, :1]-self.weight*raw[:, 1:2]-(raw[:, 2:]*dual).sum(1, keepdim=True)
        return tuple(torch.as_tensor(v[idx], device=self.device) for v in
                     (self.obs, self.actions))+(reward,)+tuple(torch.as_tensor(v[idx], device=self.device)
                     for v in (self.next_obs, self.dones, self.execution_masks))
