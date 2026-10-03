"""Offline one-step tube-outside recovery QP for certified reference B.

No MPC horizon, no SAC training, and no change to the online controller.
``du_rec`` augments the actual input only: z advances with v_base, so the
certified affine error dynamics contain B @ du_rec as intended.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .control import SafeController
from .model import EvaporatorModel
from .paper2016_compare import SCENARIOS, apply_state_shock
from .residual_action_space_diagnosis import (
    DEFAULT_DESIGN, DEFAULT_STRESS_ACTOR, FrozenActor, REPO_DIR,
    load_fixed_b, write_csv,
)
from .train import observation
from .tube_recovery_diagnosis import (
    bounds_flags, configure_mode, inside, reentry_by_shock,
)


MODES = ("zero_residual", "Z_outside_residual_zero", "one_step_recovery_QP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--actor", type=Path, default=DEFAULT_STRESS_ACTOR)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--simulation-seconds", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_one_step_recovery_B"
    ))
    return parser.parse_args()


def z_halfspaces(vertices: np.ndarray) -> np.ndarray:
    """Irredundant nonzero-edge facets H with Z={e:H e<=1}."""
    v = np.asarray(vertices, dtype=float)
    edges = np.roll(v, -1, axis=0) - v
    keep = np.linalg.norm(edges, axis=1) > 1e-9
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])[keep]
    offsets = np.sum(normals * v[keep], axis=1)
    if np.any(offsets <= 0.0):
        raise RuntimeError("RPI polygon does not contain the origin strictly")
    h = normals / offsets[:, None]
    if np.max(h @ v.T) > 1.0 + 1e-7:
        raise RuntimeError("derived Z halfspaces do not contain stored vertices")
    return h


def project_origin_2d_fast(a: np.ndarray, b: np.ndarray, tol=1e-9):
    """Vectorized equivalent of the existing 2D active-set projection."""
    if np.all(b >= -tol):
        return np.zeros(2), True
    n = len(b)
    candidate_points = []
    norms = np.sum(a * a, axis=1)
    usable = norms > 1e-14
    row_points = np.zeros((np.sum(usable), 2))
    row_points[:] = (b[usable] / norms[usable])[:, None] * a[usable]
    if len(row_points):
        good = np.all(a @ row_points.T <= b[:, None] + tol, axis=0)
        candidate_points.append(row_points[good])
    i, j = np.triu_indices(n, k=1)
    det = a[i, 0] * a[j, 1] - a[i, 1] * a[j, 0]
    useful = np.abs(det) > 1e-12
    i, j, det = i[useful], j[useful], det[useful]
    if len(det):
        intersections = np.column_stack([
            (b[i] * a[j, 1] - a[i, 1] * b[j]) / det,
            (a[i, 0] * b[j] - b[i] * a[j, 0]) / det,
        ])
        good = np.all(a @ intersections.T <= b[:, None] + tol, axis=0)
        candidate_points.append(intersections[good])
    points = np.vstack(candidate_points) if candidate_points else np.empty((0, 2))
    if len(points) == 0:
        return np.zeros(2), False
    return points[np.argmin(np.sum(points * points, axis=1))], True


def recovery_hard_constraints(cfg, model, design, info, x: np.ndarray):
    """Linear du constraints on actual input and next physical state.

    The existing verification QP constrains *v_base* and z_next, not du_rec:
    du_rec is ancillary and deliberately does not enter the nominal z update.
    """
    base = np.asarray(info["base"])
    ancillary = np.asarray(info["ancillary"])
    actual_zero = base + ancillary
    x_center = design.a @ x + design.b @ actual_zero + design.affine
    x_lo = model.normalized_state(cfg.state_lower)
    x_hi = model.normalized_state(cfg.state_upper)
    u_phys_lo = model.normalized_input(cfg.input_lower)
    u_phys_hi = model.normalized_input(cfg.input_upper)
    rows = [np.eye(2), -np.eye(2), np.eye(2), -np.eye(2)]
    bounds = [
        design.robust_input_upper - actual_zero,
        -(design.robust_input_lower - actual_zero),
        u_phys_hi - actual_zero,
        -(u_phys_lo - actual_zero),
    ]
    # Require the *certified affine* one-step physical state constraints for
    # every W vertex. No constraint is silently softened by alpha.
    for w in design.w_vertices:
        rows.extend((design.b, -design.b))
        bounds.extend((x_hi - x_center - w, -(x_lo - x_center - w)))
    return np.vstack(rows), np.concatenate(bounds)


def solve_recovery_qp(cfg, model, design, info, x, h_z, rho):
    """Solve min alpha+rho||du||² via 2D projection at fixed alpha.

    The alpha-feasible sets are nested. Projection onto each is solved exactly
    by 2D active sets with separation over all Z facets; a scalar convex
    search then minimizes the full one-step QP objective. No new dependency.
    """
    e = np.asarray(info["e"])
    if (
        not info["qp_feasible"]
        or np.any(info["a_q"] @ info["base"] > info["b_q"] + 1e-8)
    ):
        return {
            "feasible": False, "reason": "nominal_verification_qp",
            "du": np.zeros(2), "alpha": float("nan"),
            "alpha_min_feasible": float("nan"),
        }
    acl = design.a + design.b @ design.k
    hard_a, hard_b = recovery_hard_constraints(cfg, model, design, info, x)
    hard_point, hard_feasible = project_origin_2d_fast(hard_a, hard_b)
    if not hard_feasible:
        return {
            "feasible": False, "reason": "hard_input_or_physical_state",
            "du": np.zeros(2), "alpha": float("nan"),
            "alpha_min_feasible": float("nan"),
        }
    ray_a = h_z @ design.b
    worst_w = np.max(h_z @ design.w_vertices.T, axis=1)
    ray_offset = h_z @ (acl @ e) + worst_w
    active_cuts: list[int] = []

    def fixed_alpha(alpha):
        bound = alpha - ray_offset
        point = hard_point
        if np.max(ray_a @ point - bound) <= 1e-8:
            return point, True
        for _ in range(len(bound) + 1):
            violation = ray_a @ point - bound
            maximum = float(np.max(violation))
            if maximum <= 1e-8:
                return point, True
            candidates = np.argsort(violation)[-4:]
            added = False
            for index in candidates:
                if violation[index] > 1e-8 and int(index) not in active_cuts:
                    active_cuts.append(int(index))
                    added = True
            if not added:
                return point, False
            point, feasible = project_origin_2d_fast(
                np.vstack([hard_a, ray_a[active_cuts]]),
                np.concatenate([hard_b, bound[active_cuts]]),
            )
            if not feasible:
                return point, False
        return point, False

    upper = max(0.0, float(np.max(ray_a @ hard_point + ray_offset)))
    upper_point, upper_ok = fixed_alpha(upper + 1e-8)
    if not upper_ok:
        return {
            "feasible": False, "reason": "numeric_upper_alpha",
            "du": np.zeros(2), "alpha": float("nan"),
            "alpha_min_feasible": float("nan"),
        }
    low, high = 0.0, upper + 1e-8
    for _ in range(21):
        middle = 0.5 * (low + high)
        _, feasible = fixed_alpha(middle)
        if feasible:
            high = middle
        else:
            low = middle
    alpha_min = high
    best_alpha, best_du = high, fixed_alpha(high)[0]
    best_value = best_alpha + rho * float(best_du @ best_du)
    upper_value = upper + rho * float(upper_point @ upper_point)
    if upper_value < best_value:
        best_alpha, best_du, best_value = upper, upper_point, upper_value
    left, right = alpha_min, upper
    if right > left + 1e-8:
        ratio = (np.sqrt(5.0) - 1.0) / 2.0
        c = right - ratio * (right - left)
        d = left + ratio * (right - left)
        for _ in range(23):
            pc, fc = fixed_alpha(c)
            pd, fd = fixed_alpha(d)
            vc = c + rho * float(pc @ pc) if fc else float("inf")
            vd = d + rho * float(pd @ pd) if fd else float("inf")
            if vc < best_value:
                best_alpha, best_du, best_value = c, pc, vc
            if vd < best_value:
                best_alpha, best_du, best_value = d, pd, vd
            if vc <= vd:
                right, d = d, c
                c = right - ratio * (right - left)
            else:
                left, c = c, d
                d = left + ratio * (right - left)
    certified_alpha = max(0.0, float(np.max(ray_a @ best_du + ray_offset)))
    hard_excess = float(np.max(hard_a @ best_du - hard_b))
    if hard_excess > 1e-7 or certified_alpha > best_alpha + 1e-5:
        return {
            "feasible": False, "reason": "solver_postcheck_failed",
            "du": np.zeros(2), "alpha": float("nan"),
            "alpha_min_feasible": alpha_min,
            "hard_excess": hard_excess,
            "alpha_excess": certified_alpha - best_alpha,
        }
    return {
        "feasible": True, "reason": "optimal_one_step_qp",
        "du": best_du, "alpha": certified_alpha,
        "alpha_min_feasible": alpha_min,
        "objective": certified_alpha + rho * float(best_du @ best_du),
        "hard_excess": hard_excess,
    }


def robust_alpha(h_z, design, e, du):
    acl = design.a + design.b @ design.k
    return max(0.0, float(np.max(
        h_z @ (acl @ e + design.b @ du)
        + np.max(h_z @ design.w_vertices.T, axis=1)
    )))


def rollout(cfg, model, design, actor, h_z, rho, scenario, mode, seconds):
    configure_mode(
        cfg, "zero_residual" if mode == "zero_residual"
        else "max_margin_asymmetric_polytope"
    )
    controller = SafeController(cfg, model, design)
    state = cfg.robust_economic_reference_state.copy()
    controller.reset(state)
    previous_u = design.v_ref.copy()
    w_est = np.zeros(2)
    rows = []
    for second in range(seconds):
        before_shock = state.copy()
        x_before = model.normalized_state(before_shock)
        e_before = x_before - controller.z
        ps_before, _, rs_before, _ = bounds_flags(
            cfg, model, design, x_before, None
        )
        state = apply_state_shock(state, scenario, second, scale=1.0)
        shock_applied = bool(np.linalg.norm(state - before_shock) > 0.0)
        x = model.normalized_state(state)
        e = x - controller.z
        in_z = inside(design.rpi_boundary, e)
        ps_shock, _, rs_shock, _ = bounds_flags(cfg, model, design, x, None)
        obs = observation(model, controller, state, previous_u, w_est)
        actor_action = actor.action(obs)
        sac_action = (
            np.zeros(2) if mode == "zero_residual" or not in_z
            else actor_action
        )
        nominal_z_before = controller.z.copy()
        _, info = controller.act(state, sac_action, action_is_normalized=True)
        du = np.zeros(2)
        attempt = mode == "one_step_recovery_QP" and not in_z
        if attempt:
            solution = solve_recovery_qp(cfg, model, design, info, x, h_z, rho)
            if solution["feasible"]:
                du = np.asarray(solution["du"])
                actual_u = np.asarray(info["base"]) + np.asarray(info["ancillary"]) + du
                # controller.act() already advanced z using v_base and zero
                # SAC residual. Do not add du to z: it belongs in e dynamics.
            else:
                actual_u = np.asarray(info["actual_norm"])
        else:
            solution = {
                "feasible": None, "reason": "not_attempted",
                "alpha_min_feasible": float("nan"),
            }
            actual_u = np.asarray(info["actual_norm"])
        existing_clipped_u = np.asarray(info["actual_norm"])
        control = model.physical_input(actual_u)
        next_state = model.step(state, control, cfg.disturbance_nominal)
        next_x = model.normalized_state(next_state)
        next_e = next_x - controller.z
        ps_next, pi_next, rs_next, ri_next = bounds_flags(
            cfg, model, design, next_x, actual_u
        )
        inferred_w = next_x - (
            design.a @ x + design.b @ actual_u + design.affine
        )
        w_est = cfg.disturbance_estimate_ema * w_est + (
            1.0 - cfg.disturbance_estimate_ema
        ) * inferred_w
        rows.append({
            "mode": mode, "scenario": scenario, "second": second,
            "shock_applied": shock_applied,
            "shock_X2": float(state[0] - before_shock[0]),
            "shock_P2": float(state[1] - before_shock[1]),
            "e_before_shock_in_Z": inside(design.rpi_boundary, e_before),
            "e_after_shock_in_Z": in_z,
            "e_after_shock_X2": float(e[0]),
            "e_after_shock_P2": float(e[1]),
            "physical_state_violation_at_shock": ps_shock,
            "new_physical_violation_caused_by_shock": bool(
                shock_applied and ps_shock and not ps_before
            ),
            "robust_state_violation_at_shock": rs_shock,
            "new_robust_violation_caused_by_shock": bool(
                shock_applied and rs_shock and not rs_before
            ),
            "actor_request_P100": float(actor_action[0]),
            "actor_request_F200": float(actor_action[1]),
            "sac_active": bool(np.linalg.norm(info["nominal"] - info["base"]) > 1e-10),
            "sac_residual_P100": float(
                (info["nominal"][0] - info["base"][0]) * cfg.input_scale[0]
            ),
            "sac_residual_F200": float(
                (info["nominal"][1] - info["base"][1]) * cfg.input_scale[1]
            ),
            "recovery_qp_attempted": attempt,
            "recovery_qp_feasible": solution["feasible"],
            "recovery_qp_reason": solution["reason"],
            "alpha_current": max(0.0, float(np.max(h_z @ e))),
            "alpha_predicted_robust": robust_alpha(h_z, design, e, du),
            "alpha_min_feasible": solution["alpha_min_feasible"],
            "alpha_actual_next": max(0.0, float(np.max(h_z @ next_e))),
            "du_rec_P100": float(du[0] * cfg.input_scale[0]),
            "du_rec_F200": float(du[1] * cfg.input_scale[1]),
            "du_rec_norm_normalized": float(np.linalg.norm(du)),
            "du_rec_norm_physical": float(np.linalg.norm(du * cfg.input_scale)),
            "actual_input_change_vs_existing_clip": float(np.linalg.norm(
                (actual_u - existing_clipped_u) * cfg.input_scale
            )),
            "unclipped_Ke_input_outside_robust_norm": float(np.linalg.norm(
                (np.asarray(info["base"]) + np.asarray(info["ancillary"])
                - existing_clipped_u) * cfg.input_scale
            )),
            "ancillary_P100": float(info["ancillary"][0] * cfg.input_scale[0]),
            "ancillary_F200": float(info["ancillary"][1] * cfg.input_scale[1]),
            "v_base_P100": float(model.physical_input(info["base"])[0]),
            "v_base_F200": float(model.physical_input(info["base"])[1]),
            "P100": float(control[0]), "F200": float(control[1]),
            "x_after_shock_X2": float(state[0]),
            "x_after_shock_P2": float(state[1]),
            "z_before_X2": float(model.physical_state(nominal_z_before)[0]),
            "z_before_P2": float(model.physical_state(nominal_z_before)[1]),
            "x_next_X2": float(next_state[0]),
            "x_next_P2": float(next_state[1]),
            "z_next_X2": float(model.physical_state(controller.z)[0]),
            "z_next_P2": float(model.physical_state(controller.z)[1]),
            "e_next_X2": float(next_e[0]),
            "e_next_P2": float(next_e[1]),
            "e_next_in_Z": inside(design.rpi_boundary, next_e),
            "physical_state_violation_next": ps_next,
            "physical_input_violation": pi_next,
            "robust_state_violation_next": rs_next,
            "robust_input_violation": ri_next,
            "verification_qp_feasible": bool(info["qp_feasible"]),
            "verification_qp_nominal_base_feasible": bool(np.all(
                info["a_q"] @ info["base"] <= info["b_q"] + 1e-8
            )),
            "nonlinear_w_exceeds_W": not inside(design.w_vertices, inferred_w),
        })
        state = next_state
        previous_u = actual_u
    return rows


def fixed_common_reference_metrics(rows, zero_rows, reference, key):
    # One common nominal B target, and one zero-residual 5% band for all
    # controllers within the scenario. The window starts at the final shock.
    signal = np.array([float(r[key]) for r in rows if r["second"] >= 40])
    baseline = np.array([float(r[key]) for r in zero_rows if r["second"] >= 40])
    tolerance = max(0.05 * float(np.max(np.abs(baseline - reference))), 1e-9)
    inside_band = np.abs(signal - reference) <= tolerance
    settle = next((
        i for i in range(len(signal)) if bool(np.all(inside_band[i:]))
    ), None)
    return {
        "common_reference": float(reference),
        "common_tolerance": tolerance,
        "last_shock_second": 40,
        "settling_time_after_last_shock_seconds": settle,
        "IAE_after_last_shock": float(np.sum(np.abs(signal - reference))),
        "ISE_after_last_shock": float(np.sum((signal - reference) ** 2)),
        "peak_deviation_after_last_shock": float(np.max(np.abs(signal - reference))),
    }


def summarize(rows, cfg, design):
    result = {}
    robust_lower = cfg.linearization_input + design.robust_input_lower * cfg.input_scale
    robust_upper = cfg.linearization_input + design.robust_input_upper * cfg.input_scale
    for scenario in SCENARIOS:
        zero = [r for r in rows if r["scenario"] == scenario and r["mode"] == "zero_residual"]
        result[scenario] = {}
        for mode in MODES:
            subset = [r for r in rows if r["scenario"] == scenario and r["mode"] == mode]
            attempted = [r for r in subset if r["recovery_qp_attempted"]]
            result[scenario][mode] = {
                "shock_reentry": reentry_by_shock(subset),
                "first_physical_violation_second": next((
                    int(r["second"] + 1) for r in subset
                    if r["physical_state_violation_next"]
                ), None),
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
                "verification_qp_infeasible_steps": sum(
                    not bool(r["verification_qp_feasible"]) for r in subset
                ),
                "nominal_base_verification_qp_infeasible_steps": sum(
                    not bool(r["verification_qp_nominal_base_feasible"]) for r in subset
                ),
                "recovery_qp_attempts": len(attempted),
                "recovery_qp_infeasible_steps": sum(
                    not bool(r["recovery_qp_feasible"]) for r in attempted
                ),
                "recovery_qp_failure_reasons": sorted(set(
                    r["recovery_qp_reason"] for r in attempted
                    if not r["recovery_qp_feasible"]
                )),
                "nonlinear_w_bound_exceedance_steps": sum(
                    bool(r["nonlinear_w_exceeds_W"]) for r in subset
                ),
                "sac_active_fraction": float(np.mean([r["sac_active"] for r in subset])),
                "du_rec_norm_physical_mean": float(np.mean([
                    r["du_rec_norm_physical"] for r in subset
                ])),
                "du_rec_norm_physical_max": float(max(
                    r["du_rec_norm_physical"] for r in subset
                )),
                "actual_input_change_vs_existing_clip_max": float(max(
                    r["actual_input_change_vs_existing_clip"] for r in subset
                )),
                "actual_input_change_vs_existing_clip_mean": float(np.mean([
                    r["actual_input_change_vs_existing_clip"] for r in subset
                ])),
                "F200_at_robust_upper_steps": int(sum(
                    abs(r["F200"] - robust_upper[1])
                    <= 1e-6 for r in subset
                )),
                "F200_at_robust_lower_steps": int(sum(
                    abs(r["F200"] - robust_lower[1])
                    <= 1e-6 for r in subset
                )),
                "P100_at_robust_lower_steps": int(sum(
                    abs(r["P100"] - robust_lower[0])
                    <= 1e-6 for r in subset
                )),
                "alpha_predicted_min": float(min(
                    r["alpha_predicted_robust"] for r in subset
                )),
                "alpha_predicted_max": float(max(
                    r["alpha_predicted_robust"] for r in subset
                )),
                "X2": fixed_common_reference_metrics(
                    subset, zero, cfg.robust_economic_reference_state[0],
                    "x_after_shock_X2",
                ),
                "P2": fixed_common_reference_metrics(
                    subset, zero, cfg.robust_economic_reference_state[1],
                    "x_after_shock_P2",
                ),
                "P100": {
                    "min": float(min(r["P100"] for r in subset)),
                    "max": float(max(r["P100"] for r in subset)),
                    "total_variation": float(np.sum(np.abs(np.diff([
                        r["P100"] for r in subset
                    ])))),
                },
                "F200": {
                    "min": float(min(r["F200"] for r in subset)),
                    "max": float(max(r["F200"] for r in subset)),
                    "total_variation": float(np.sum(np.abs(np.diff([
                        r["F200"] for r in subset
                    ])))),
                },
            }
        recovery = result[scenario]["one_step_recovery_QP"]
        gated_rows = [r for r in rows if r["scenario"] == scenario
                      and r["mode"] == "Z_outside_residual_zero"]
        recovery_rows = [r for r in rows if r["scenario"] == scenario
                         and r["mode"] == "one_step_recovery_QP"]
        recovery["max_input_difference_vs_gated"] = float(max(
            np.hypot(a["P100"] - b["P100"], a["F200"] - b["F200"])
            for a, b in zip(recovery_rows, gated_rows)
        ))
        recovery["max_state_difference_vs_gated"] = float(max(
            np.hypot(a["x_next_X2"] - b["x_next_X2"],
                     a["x_next_P2"] - b["x_next_P2"])
            for a, b in zip(recovery_rows, gated_rows)
        ))
    return result


def main() -> None:
    args = parse_args()
    if args.rho <= 0.0:
        raise ValueError("rho must be positive")
    if args.simulation_seconds < 41:
        raise ValueError("simulation must include all three original shocks")
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed")
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    actor = FrozenActor(args.actor, cfg)
    if actor.scale <= 0.0:
        raise ValueError("diagnosis requires a nonzero frozen actor")
    h_z = z_halfspaces(design.rpi_boundary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        for mode in MODES:
            print(f"{scenario}: {mode}", flush=True)
            rows.extend(rollout(
                cfg, model, design, actor, h_z, args.rho, scenario, mode,
                args.simulation_seconds,
            ))
    write_csv(args.output_dir / "one_step_recovery_trajectories.csv", rows)
    summary = {
        "scope": "offline one-step recovery QP; no MPC or SAC training",
        "recovery_dynamics": "z_next=A*z+B*v_base+c; e_next=(A+B*K)*e+B*du_rec+w",
        "verification_qp_semantics": (
            "The existing nominal verification QP constrains v_base and "
            "z_next. du_rec augments actual input only, so actual input is "
            "separately constrained by robust and physical input bounds; "
            "one-step physical state bounds hold at every certified W vertex."
        ),
        "solver": (
            "dependency-free 2D active-set projection with separation over "
            "all Z facets and scalar convex search in alpha; no horizon"
        ),
        "reference_state": cfg.robust_economic_reference_state.tolist(),
        "reference_input": cfg.robust_economic_reference_input.tolist(),
        "rho": args.rho,
        "Z_facets": len(h_z),
        "W_vertices": len(design.w_vertices),
        "actor_checkpoint": str(args.actor),
        "actor_output_scale": actor.scale,
        "common_reference_definition": (
            "B steady state for all methods and scenarios; per-scenario 5% "
            "band from zero-residual peak after t=40; IAE/ISE over t=40..299"
        ),
        "scenarios": summarize(rows, cfg, design),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
