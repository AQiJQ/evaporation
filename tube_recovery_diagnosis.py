"""Separate certified-tube stress from out-of-tube Paper2016 shock recovery.

No SAC training and no changes to the controller or its certified geometry.
The formal-domain test uses the *affine certified model* with generated w in W;
the shock test uses the nonlinear plant and nominal external conditions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .control import SafeController, point_in_convex_polygon
from .model import EvaporatorModel
from .paper2016_compare import SCENARIOS, SHOCK_SECONDS, apply_state_shock
from .residual_action_space_diagnosis import (
    DEFAULT_DESIGN, DEFAULT_STRESS_ACTOR, FrozenActor, MODES,
    REPO_DIR, load_fixed_b, write_csv,
)
from .train import observation


FORMAL_MODES = (
    "current_5pct_box",
    "max_margin_symmetric_box",
    "max_margin_asymmetric_polytope",
)
SHOCK_MODES = (
    "zero_residual",
    "current_5pct_box",
    "max_margin_asymmetric_polytope",
    "polytope_outside_Z_gated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--actor", type=Path, default=DEFAULT_STRESS_ACTOR)
    parser.add_argument("--formal-trials", type=int, default=5)
    parser.add_argument("--formal-steps", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_tube_recovery_diagnosis_B"
    ))
    return parser.parse_args()


def configure_mode(cfg: ExperimentConfig, mode: str) -> None:
    mapped = "max_margin_asymmetric_polytope" if mode == "polytope_outside_Z_gated" else mode
    mapped = "current_5pct_box" if mode == "zero_residual" else mapped
    cfg.residual_reserve_mode, cfg.residual_parameterization = MODES[mapped]
    cfg.residual_reserve_fraction = 0.8


def inside(vertices: np.ndarray, point: np.ndarray) -> bool:
    return point_in_convex_polygon(point, vertices, tol=1e-8)


def bounds_flags(cfg, model, design, x: np.ndarray, u: np.ndarray | None):
    state = model.physical_state(x)
    physical_state_bad = bool(
        np.any(state < cfg.state_lower - 1e-8)
        or np.any(state > cfg.state_upper + 1e-8)
    )
    robust_state_bad = bool(
        np.any(x < design.robust_state_lower - 1e-8)
        or np.any(x > design.robust_state_upper + 1e-8)
    )
    if u is None:
        return physical_state_bad, False, robust_state_bad, False
    physical_input = model.physical_input(u)
    physical_input_bad = bool(
        np.any(physical_input < cfg.input_lower - 1e-8)
        or np.any(physical_input > cfg.input_upper + 1e-8)
    )
    robust_input_bad = bool(
        np.any(u < design.robust_input_lower - 1e-8)
        or np.any(u > design.robust_input_upper + 1e-8)
    )
    return (
        physical_state_bad, physical_input_bad,
        robust_state_bad, robust_input_bad,
    )


def formal_sequences(design, trials: int, steps: int):
    rng = np.random.default_rng(20260924)
    directions = np.array([
        [1.0, 0.0], [1.0, 0.5], [1.0, 1.0],
        [0.5, 1.0], [0.0, 1.0], [-0.5, 1.0],
        [-1.0, 1.0], [-1.0, 0.5], [-1.0, 0.0],
        [-1.0, -0.5], [-1.0, -1.0], [-0.5, -1.0],
        [0.0, -1.0], [0.5, -1.0], [1.0, -1.0], [1.0, -0.5],
    ])
    corners = np.array(list(__import__("itertools").product((-1.0, 1.0), repeat=2)))
    for family in ("random", "corners", "polytope_boundary"):
        for trial in range(trials):
            if trial == 0:
                e0 = np.zeros(2)
            elif trial % 2:
                e0 = 0.95 * design.rpi_boundary[(trial * 137) % len(design.rpi_boundary)]
            else:
                weights = rng.dirichlet(np.ones(3))
                selected = rng.choice(len(design.rpi_boundary), size=3, replace=False)
                e0 = 0.95 * (weights @ design.rpi_boundary[selected])
            if not inside(design.rpi_boundary, e0):
                raise RuntimeError("generated e0 is outside Z")
            actions = np.empty((steps, 2))
            disturbances = np.empty((steps, 2))
            for k in range(steps):
                if family == "random":
                    a = rng.uniform(-1.0, 1.0, 2)
                    if np.max(np.abs(a)) < 0.2:
                        a[np.argmax(np.abs(a))] = 0.2
                elif family == "corners":
                    a = corners[(k + trial) % len(corners)]
                else:
                    # At rho=1 the polytope mapping targets 0.999 of the
                    # available directional boundary; the other modes get
                    # exactly the same requested actor directions.
                    a = directions[(k + 3 * trial) % len(directions)]
                actions[k] = a
                if k % 4 == 0:
                    disturbances[k] = design.w_vertices[(k // 4 + trial) % len(design.w_vertices)]
                else:
                    indices = rng.choice(len(design.w_vertices), size=3)
                    disturbances[k] = rng.dirichlet(np.ones(3)) @ design.w_vertices[indices]
                if not inside(design.w_vertices, disturbances[k]):
                    raise RuntimeError("generated disturbance is outside W")
            yield family, trial, e0, actions, disturbances


def formal_domain_stress(cfg, model, design, trials, steps):
    sequences = list(formal_sequences(design, trials, steps))
    rows = []
    for mode in FORMAL_MODES:
        configure_mode(cfg, mode)
        print(f"formal-domain {mode}", flush=True)
        for family, trial, e0, actions, disturbances in sequences:
            controller = SafeController(cfg, model, design)
            x = design.z_ref + e0
            controller.reset(model.physical_state(x))
            # reset() deliberately sets z to x for ordinary episodes; the
            # formal tube test instead needs the specified nonzero e0=x-z.
            controller.z = design.z_ref.copy()
            if any(bounds_flags(cfg, model, design, x, None)):
                raise RuntimeError("generated initial tube state violates bounds")
            for k, (action, w) in enumerate(zip(actions, disturbances)):
                e_before = x - controller.z
                # The pattern is deliberately imposed; no actor is trained or queried.
                _, info = controller.act(
                    model.physical_state(x), action, action_is_normalized=True
                )
                u = np.asarray(info["actual_norm"])
                x_next = design.a @ x + design.b @ u + design.affine + w
                e_next = x_next - controller.z
                inferred_w = x_next - (
                    design.a @ x + design.b @ u + design.affine
                )
                ps, pi, rs, ri = bounds_flags(cfg, model, design, x_next, u)
                tube_bad = not inside(design.rpi_boundary, e_next)
                disturbance_bad = not inside(design.w_vertices, inferred_w)
                unclipped = info["nominal"] + design.k @ e_before
                rows.append({
                    "mode": mode, "action_family": family, "trial": trial,
                    "step": k, "a_P100": float(action[0]),
                    "a_F200": float(action[1]),
                    "e0_in_Z": bool(inside(design.rpi_boundary, e0)),
                    "e_before_in_Z": bool(inside(design.rpi_boundary, e_before)),
                    "e_next_X2": float(e_next[0]), "e_next_P2": float(e_next[1]),
                    "w_X2": float(w[0]), "w_P2": float(w[1]),
                    "w_in_W": bool(inside(design.w_vertices, w)),
                    "physical_state_violation": ps,
                    "physical_input_violation": pi,
                    "robust_state_violation": rs,
                    "robust_input_violation": ri,
                    "robust_region_violation": rs or ri,
                    "qp_infeasible": not bool(info["qp_feasible"]),
                    "tube_violation": tube_bad,
                    "disturbance_bound_exceedance": disturbance_bad,
                    "ancillary_clip_norm": float(np.linalg.norm(u - unclipped)),
                    "residual_applied_norm": float(np.linalg.norm(
                        info["nominal"] - info["base"]
                    )),
                })
                x = x_next
    return rows


def shock_rollout(cfg, model, design, actor, scenario, mode):
    configure_mode(cfg, mode)
    controller = SafeController(cfg, model, design)
    state = cfg.robust_economic_reference_state.copy()
    controller.reset(state)
    previous_u = design.v_ref.copy()
    w_est = np.zeros(2)
    rows = []
    for second in range(300):
        before_shock = state.copy()
        x_before_shock = model.normalized_state(before_shock)
        ps_before, _, rs_before, _ = bounds_flags(
            cfg, model, design, x_before_shock, None
        )
        e_before_shock = x_before_shock - controller.z
        state = apply_state_shock(state, scenario, second, scale=1.0)
        shock_applied = bool(np.linalg.norm(state - before_shock) > 0.0)
        x_shock = model.normalized_state(state)
        z = controller.z.copy()
        e_shock = x_shock - z
        ps_shock, _, rs_shock, _ = bounds_flags(cfg, model, design, x_shock, None)
        obs = observation(model, controller, state, previous_u, w_est)
        raw_action = (
            np.zeros(2) if mode == "zero_residual" else actor.action(obs)
        )
        gate_closed = bool(
            mode == "polytope_outside_Z_gated"
            and not inside(design.rpi_boundary, e_shock)
        )
        requested_action = raw_action.copy()
        if gate_closed:
            raw_action = np.zeros(2)
        control, info = controller.act(state, raw_action, action_is_normalized=True)
        next_state = model.step(state, control, cfg.disturbance_nominal)
        x_next = model.normalized_state(next_state)
        e_next = x_next - controller.z
        u = np.asarray(info["actual_norm"])
        ps_next, pi_next, rs_next, ri_next = bounds_flags(
            cfg, model, design, x_next, u
        )
        inferred_w = x_next - (
            design.a @ x_shock + design.b @ u + design.affine
        )
        w_est = cfg.disturbance_estimate_ema * w_est + (
            1.0 - cfg.disturbance_estimate_ema
        ) * inferred_w
        applied_residual = np.asarray(info["nominal"]) - np.asarray(info["base"])
        rows.append({
            "scenario": scenario, "mode": mode, "second": second,
            "shock_applied": shock_applied,
            "shock_X2": float(state[0] - before_shock[0]),
            "shock_P2": float(state[1] - before_shock[1]),
            "e_before_shock_X2": float(e_before_shock[0]),
            "e_before_shock_P2": float(e_before_shock[1]),
            "e_before_shock_in_Z": inside(design.rpi_boundary, e_before_shock),
            "x_after_shock_X2": float(state[0]),
            "x_after_shock_P2": float(state[1]),
            "z_before_X2": float(model.physical_state(z)[0]),
            "z_before_P2": float(model.physical_state(z)[1]),
            "e_after_shock_X2": float(e_shock[0]),
            "e_after_shock_P2": float(e_shock[1]),
            "e_after_shock_in_Z": inside(design.rpi_boundary, e_shock),
            "physical_state_violation_at_shock": ps_shock,
            "new_physical_violation_caused_by_shock": bool(
                shock_applied and ps_shock and not ps_before
            ),
            "robust_state_violation_at_shock": rs_shock,
            "new_robust_violation_caused_by_shock": bool(
                shock_applied and rs_shock and not rs_before
            ),
            "requested_actor_P100": float(requested_action[0]),
            "requested_actor_F200": float(requested_action[1]),
            "gate_closed": gate_closed,
            "sac_residual_norm": float(np.linalg.norm(applied_residual)),
            "sac_residual_P100": float(applied_residual[0] * cfg.input_scale[0]),
            "sac_residual_F200": float(applied_residual[1] * cfg.input_scale[1]),
            "ancillary_P100": float(info["ancillary"][0] * cfg.input_scale[0]),
            "ancillary_F200": float(info["ancillary"][1] * cfg.input_scale[1]),
            "nominal_P100": float(model.physical_input(info["nominal"])[0]),
            "nominal_F200": float(model.physical_input(info["nominal"])[1]),
            "actual_input_P100": float(control[0]),
            "actual_input_F200": float(control[1]),
            "x_next_X2": float(next_state[0]),
            "x_next_P2": float(next_state[1]),
            "z_next_X2": float(model.physical_state(controller.z)[0]),
            "z_next_P2": float(model.physical_state(controller.z)[1]),
            "e_next_X2": float(e_next[0]),
            "e_next_P2": float(e_next[1]),
            "e_next_in_Z": inside(design.rpi_boundary, e_next),
            "physical_state_violation_next": ps_next,
            "physical_input_violation": pi_next,
            "robust_state_violation_next": rs_next,
            "robust_input_violation": ri_next,
            "qp_infeasible": not bool(info["qp_feasible"]),
            "nonlinear_w_exceeds_W": not inside(design.w_vertices, inferred_w),
        })
        state = next_state
        previous_u = u
    return rows


def first_true_time(rows, key, *, shock_only=False, next_state=False):
    for row in rows:
        if (not shock_only or row["shock_applied"]) and row[key]:
            return int(row["second"] + (1 if next_state else 0))
    return None


def reentry_by_shock(rows):
    events = []
    for index, shock in enumerate(SHOCK_SECONDS):
        end = SHOCK_SECONDS[index + 1] if index + 1 < len(SHOCK_SECONDS) else 300
        segment = [r for r in rows if shock <= r["second"] < end]
        first = None
        for row in segment:
            if row["e_after_shock_in_Z"]:
                first = int(row["second"])
                break
            if row["e_next_in_Z"]:
                first = int(row["second"] + 1)
                break
        events.append({
            "shock_second": shock,
            "e_in_Z_immediately_after_shock": bool(segment[0]["e_after_shock_in_Z"]),
            "e_in_Z_immediately_before_shock": bool(segment[0]["e_before_shock_in_Z"]),
            "physical_violation_at_shock": bool(segment[0]["physical_state_violation_at_shock"]),
            "new_physical_violation_caused_by_shock": bool(
                segment[0]["new_physical_violation_caused_by_shock"]
            ),
            "robust_region_violation_at_shock": bool(segment[0]["robust_state_violation_at_shock"]),
            "new_robust_violation_caused_by_shock": bool(
                segment[0]["new_robust_violation_caused_by_shock"]
            ),
            "first_reentry_second_before_next_shock": first,
            "reentry_delay_seconds": None if first is None else first - shock,
        })
    return events


def shock_summary(rows):
    result = {}
    for mode in SHOCK_MODES:
        result[mode] = {}
        for scenario in SCENARIOS:
            subset = [r for r in rows if r["mode"] == mode and r["scenario"] == scenario]
            result[mode][scenario] = {
                "shock_events": reentry_by_shock(subset),
                "first_physical_violation_at_shock_second": first_true_time(
                    subset, "physical_state_violation_at_shock", shock_only=True
                ),
                "first_physical_violation_after_step_second": first_true_time(
                    subset, "physical_state_violation_next", next_state=True
                ),
                "first_robust_region_violation_at_shock_second": first_true_time(
                    subset, "robust_state_violation_at_shock", shock_only=True
                ),
                "first_robust_region_violation_after_step_second": first_true_time(
                    subset, "robust_state_violation_next", next_state=True
                ),
                "physical_state_violation_steps": sum(
                    bool(r["physical_state_violation_next"]) for r in subset
                ),
                "physical_input_violation_steps": sum(
                    bool(r["physical_input_violation"]) for r in subset
                ),
                "robust_region_violation_steps": sum(
                    bool(r["robust_state_violation_next"] or r["robust_input_violation"])
                    for r in subset
                ),
                "qp_infeasible_steps": sum(bool(r["qp_infeasible"]) for r in subset),
                "tube_outside_steps_after_step": sum(
                    not bool(r["e_next_in_Z"]) for r in subset
                ),
                "nonlinear_w_exceedance_steps": sum(
                    bool(r["nonlinear_w_exceeds_W"]) for r in subset
                ),
                "sac_active_fraction": float(np.mean([
                    r["sac_residual_norm"] > 1e-10 for r in subset
                ])),
                "gate_closed_fraction": float(np.mean([
                    r["gate_closed"] for r in subset
                ])),
            }
    return result


def formal_summary(rows):
    metrics = (
        "physical_state_violation", "physical_input_violation",
        "robust_region_violation", "qp_infeasible", "tube_violation",
        "disturbance_bound_exceedance",
    )
    result = {}
    for mode in FORMAL_MODES:
        subset = [r for r in rows if r["mode"] == mode]
        result[mode] = {
            "steps": len(subset),
            "nonzero_action_steps": sum(
                max(abs(r["a_P100"]), abs(r["a_F200"])) > 1e-12
                for r in subset
            ),
            **{key + "_count": sum(bool(r[key]) for r in subset) for key in metrics},
            "ancillary_clip_max": float(max(r["ancillary_clip_norm"] for r in subset)),
            "first_failure": next((r for r in subset if any(r[k] for k in metrics)), None),
        }
        result[mode]["formal_domain_passed"] = bool(
            all(result[mode][key + "_count"] == 0 for key in metrics)
            and result[mode]["ancillary_clip_max"] <= 1e-8
        )
    return result


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed")
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    formal_rows = formal_domain_stress(
        cfg, model, design, args.formal_trials, args.formal_steps
    )
    write_csv(args.output_dir / "formal_domain_steps.csv", formal_rows)
    formal = formal_summary(formal_rows)
    with (args.output_dir / "formal_domain_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(formal, stream, indent=2, allow_nan=False)
    print(json.dumps({mode: entry["formal_domain_passed"] for mode, entry in formal.items()}), flush=True)
    if not formal["max_margin_asymmetric_polytope"]["formal_domain_passed"]:
        print("polytope failed inside certified tube; skip shock study until bug isolated", flush=True)
        return
    actor = FrozenActor(args.actor, cfg)
    if actor.scale <= 0.0:
        raise ValueError("recovery diagnosis requires a nonzero frozen actor")
    shock_rows = []
    for mode in SHOCK_MODES:
        for scenario in SCENARIOS:
            print(f"Paper2016 shock {mode} {scenario}", flush=True)
            shock_rows.extend(shock_rollout(cfg, model, design, actor, scenario, mode))
    write_csv(args.output_dir / "paper2016_shock_recovery_steps.csv", shock_rows)
    summary = {
        "scope": "diagnosis only; out-of-tube recovery is not formally certified",
        "reference_state": cfg.robust_economic_reference_state.tolist(),
        "actor_checkpoint": str(args.actor),
        "actor_output_scale": actor.scale,
        "formal_domain": formal,
        "paper2016_shock_recovery": shock_summary(shock_rows),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
