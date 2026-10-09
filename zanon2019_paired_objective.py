"""Paired economics and dimensionless performance constraints (new experiment).

No frozen controller/reward implementation is edited. Recovery uses every
pre-step sample, matching the stochastic evaluator (NOT the 2016 40s crop).
10% IAE/TV tolerances reuse the stochastic comprehensive-performance rule.
Boundary allowance is an explicit implementation choice: 10% more occupied
steps than baseline. It was not a hard gate in the previous stochastic run.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .omega_constrained_replay import ConstrainedReplayBuffer

KEYS = ("x2", "p2", "tv_p100", "tv_f200", "boundary")
TOLERANCE = .10
PRECONDITION_FLOORS = np.array([.05, .05, 1e-4, 1e-4, .01])


def occupied(control, lower, upper):
    return float(np.any(np.abs(np.asarray(control) - (lower + upper) / 2)
                        / ((upper - lower) / 2) > .9))


@dataclass
class Limits:
    steps: int
    iae: np.ndarray
    tv: np.ndarray
    boundary_steps: float
    input_ranges: np.ndarray

    def allowed(self):
        return np.r_[np.full(2, 1 + TOLERANCE),
                     TOLERANCE * self.tv / (self.steps * self.input_ranges),
                     TOLERANCE * self.boundary_steps / self.steps]


def limits_from_metrics(baseline, steps, ranges):
    if steps <= 0 or np.any(np.asarray(ranges) <= 0):
        raise ValueError("Nonpositive horizon/input range")
    return Limits(steps, np.array([baseline[k] for k in ("X2_IAE", "P2_IAE")]),
                  np.array([baseline[k] for k in ("P100_TV", "F200_TV")]),
                  float(baseline["any_outer_10pct_steps"]), np.asarray(ranges))


def violations(metric, limits):
    # No baseline TV ratio: zero-TV baseline remains well-defined.
    values = np.r_[np.array([metric[k] for k in ("X2_IAE", "P2_IAE")])
                   / np.maximum(limits.iae, 1e-12),
                   (np.array([metric[k] for k in ("P100_TV", "F200_TV")]) - limits.tv)
                   / (limits.steps * limits.input_ranges),
                   (metric["any_outer_10pct_steps"] - limits.boundary_steps) / limits.steps]
    return values - limits.allowed()


def step_components(step, state, control, cost, previous_control, baseline,
                    limits, reference, lower, upper, economic_scale):
    """Raw component sum[1:] is exactly g; no fixed reward weights.

Allowances are distributed uniformly across the fixed-length episode. These
action-independent offsets preserve inequality meanings and make zero-policy
constraint costs nonpositive. No entropy or audit-only RPI term is added.
    """
    if not np.isfinite(economic_scale) or economic_scale <= 0:
        raise ValueError("economic_scale must be positive")
    base = baseline[step]
    if not np.array_equal(np.asarray(baseline[0]["state"]), np.asarray(reference)):
        raise ValueError("Paired baseline must start at the common reference")
    recovery = np.abs(np.asarray(state) - reference) / np.maximum(limits.iae, 1e-12)
    tv = np.zeros(2)
    if step:
        tv = (np.abs(np.asarray(control) - previous_control)
              - np.abs(np.asarray(base["control"]) - baseline[step-1]["control"]))
        tv /= limits.steps * limits.input_ranges
    boundary = (occupied(control, lower, upper)
                - occupied(base["control"], lower, upper)) / limits.steps
    g_step = np.r_[recovery, tv, boundary] - limits.allowed() / limits.steps
    return np.r_[(base["economic_cost"] - cost) / economic_scale, g_step]


def component_series(records, baseline, limits, cfg, model, design, scale):
    if len(records) != len(baseline) or len(records) != limits.steps:
        raise ValueError("Incomplete paired trajectory")
    lo, hi = (model.physical_input(v) for v in
              (design.robust_input_lower, design.robust_input_upper))
    for k, (r, b) in enumerate(zip(records, baseline)):
        if not np.array_equal(r["disturbance"], b["disturbance"]):
            raise ValueError(f"Unpaired disturbance at step {k}")
    if not np.array_equal(records[0]["state"], baseline[0]["state"]):
        raise ValueError("Unpaired initial state")
    return np.array([step_components(k, r["state"], r["control"], r["economic_cost"],
                                    records[k-1]["control"] if k else None,
                                    baseline, limits, cfg.robust_economic_reference_state,
                                    lo, hi, scale) for k, r in enumerate(records)])


class PairedReplay(ConstrainedReplayBuffer):
    """Adapter to the immutable run_episode interface; rewrites replay ONLY.

The old scalar rollout reward is diagnostic. The source of learning reward is
independent components built from the actual audited transition, evaluated
with CURRENT dual multipliers by ConstrainedReplayBuffer.sample().
    """
    def __init__(self, *args, scales, economic_scale, **kwargs):
        super().__init__(*args, **kwargs)
        self.scales = np.asarray(scales, dtype=float)
        if self.scales.shape != (5,) or np.any(self.scales <= 0):
            raise ValueError("Expected five positive fixed component scales")
        self.economic_scale = economic_scale
        self.context = None

    def bind(self, ctrl, baseline, limits):
        self.context = (ctrl, baseline, limits)
        self.episode_components = []

    def add(self, obs, action, legacy_reward, next_obs, done, execution_mask=1.):
        if self.context is None:
            raise RuntimeError("Paired replay must be bound before an episode")
        ctrl, baseline, limits = self.context
        k = len(self.episode_components)
        row = ctrl.evidence[-1]
        if row["step"] != k or not row["next_state_observed"]:
            raise RuntimeError("Replay does not correspond to an audited transition")
        if not np.array_equal([row[f"disturbance_{i}"] for i in range(4)],
                              baseline[k]["disturbance"]):
            raise RuntimeError("Training and baseline disturbance paths differ")
        state = np.array([row[f"state_{i}"] for i in range(2)])
        next_state = np.array([row[f"next_state_{i}"] for i in range(2)])
        control = np.array([row[f"control_{i}"] for i in range(2)])
        previous = (np.array([ctrl.evidence[k-1][f"control_{i}"] for i in range(2)])
                    if k else None)
        lo, hi = (ctrl.model.physical_input(v) for v in
                  (ctrl.d.robust_input_lower, ctrl.d.robust_input_upper))
        cost = ctrl.model.economic_cost(next_state, control, baseline[k]["disturbance"])
        raw = step_components(k, state, control, cost, previous, baseline, limits,
                              ctrl.cfg.robust_economic_reference_state, lo, hi,
                              self.economic_scale)
        self.episode_components.append(raw)
        scaled = np.r_[raw[0], raw[1:] / self.scales]
        self.add_components(obs, action, scaled, next_obs, done, execution_mask)
