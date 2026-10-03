"""Offline finite-count Paper2016 state-jump survivability in fixed B/Omega.

Regular uncertainty w belongs to certified W in xi+=A xi+B q+w.
Paper2016's instantaneous xi^+=xi^-+delta is a separate jump operator;
delta is never inserted into W.  No shock clock or SAC training is used.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .controlled_invariant_error_set import (
    area, bounds_and_domains, certificate, contains, facets,
    feasible_control_polygon, hull, intersect, iterate_invariant,
    polygon_distance, set_properties,
)
from .model import EvaporatorModel
from .omega_action_realizability import DEFAULT_OMEGA
from .omega_online_closed_loop import OmegaSafeOnlineController
from .paper2016_compare import SCENARIOS, apply_state_shock
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b
from .train import ZeroResidualPolicy, run_episode


DEFAULT_ABORTED = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_safe_B_interior_anchor_seed42_300x300/"
    "training_action_steps.csv"
)
DEFAULT_OUTPUT = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_finite_state_shock_B"
)
SHOCK_SECONDS = (0, 20, 40)


def write_vertices(path, vertices, cfg):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("xi_X2_normalized", "xi_P2_normalized",
                         "error_X2_physical", "error_P2_physical"))
        for e in np.asarray(vertices):
            writer.writerow((*e, *(e * cfg.state_scale)))


def write_history(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def margin(poly, point, cfg):
    if len(poly) < 3:
        return None
    return -polygon_distance(poly, point, cfg.state_scale)


def jump_pre(poly, delta):
    return hull(np.asarray(poly) - delta) if len(poly) >= 3 else np.empty((0, 2))


def full_dimensional(poly):
    return len(poly) >= 3 and area(poly) > 1e-12


def maximum_facet_excess(points, poly):
    if len(points) == 0 or not full_dimensional(poly):
        return None
    normals, bounds = facets(poly)
    return float(np.max(np.asarray(points) @ normals.T - bounds))


def control_at(e, target, design, domain, model, cfg):
    if not full_dimensional(target):
        return {"feasible": False, "reason": "target_empty_or_lower_dimensional"}
    q = hull(feasible_control_polygon(
        e, target, design.a, design.b, design.w_vertices,
        domain["q_lower"], domain["q_upper"],
    ))
    if len(q) == 0:
        return {"feasible": False, "reason": "empty_admissible_q_polygon"}
    physical = model.physical_input(design.v_ref + q)
    normals, bounds = facets(target)
    center = np.mean(q, axis=0)
    next_vertices = design.a @ e + design.b @ center + design.w_vertices
    max_excess = float(np.max(next_vertices @ normals.T - bounds))
    return {
        "feasible": bool(max_excess <= 2e-7),
        "q_vertices_normalized": q.tolist(),
        "q_min_normalized": np.min(q, axis=0).tolist(),
        "q_max_normalized": np.max(q, axis=0).tolist(),
        "P100_F200_min_physical": np.min(physical, axis=0).tolist(),
        "P100_F200_max_physical": np.max(physical, axis=0).tolist(),
        "q_polygon_area_normalized": area(q),
        "max_regular_W_next_target_facet_excess": max_excess,
    }


def representative_controls(poly, design, domain, model, cfg):
    if not full_dimensional(poly):
        return {"representatives": [], "all_feasible": False}
    points = [("centroid", np.mean(poly, axis=0))]
    points += [(f"vertex_{i}", e) for i, e in enumerate(poly)]
    points += [(f"edge_midpoint_{i}", 0.5 * (e + poly[(i + 1) % len(poly)]))
               for i, e in enumerate(poly)]
    reference = np.zeros(2)
    if contains(poly, reference):
        points.append(("balanced_reference", reference))
    rows = []
    for name, e in points:
        rows.append({
            "name": name, "xi_normalized": np.asarray(e).tolist(),
            "physical_state": model.physical_state(design.z_ref + e).tolist(),
            "control": control_at(e, poly, design, domain, model, cfg),
        })
    return {
        "representatives": rows,
        "all_feasible": all(row["control"]["feasible"] for row in rows),
        "minimum_q_polygon_area_normalized": min(
            row["control"]["q_polygon_area_normalized"] for row in rows
            if row["control"]["feasible"]
        ) if all(row["control"]["feasible"] for row in rows) else None,
    }


def zero_baseline_shock_membership(cfg, model, design, omega, domain,
                                   scenario, sets):
    controller = OmegaSafeOnlineController(cfg, model, design, omega, domain)
    _, records, _ = run_episode(
        cfg, model, controller, ZeroResidualPolicy(), None,
        np.random.default_rng(cfg.seed + 31000 + SCENARIOS.index(scenario)),
        training=False, global_step=0, paper_scenario=scenario,
    )
    result = []
    for index, second in enumerate(SHOCK_SECONDS):
        e = model.normalized_state(records[second]["state"]) - design.z_ref
        remaining = 2 - index
        target = sets[remaining]
        result.append({
            "second": second,
            "remaining_shocks_after_jump": remaining,
            "state_after_jump": np.asarray(records[second]["state"]).tolist(),
            "xi_after_jump_normalized": e.tolist(),
            "in_C_remaining": contains(target, e),
            "margin_to_C_remaining_physical": margin(target, e, cfg),
            "in_Omega": contains(omega, e),
            "margin_to_Omega_physical": margin(omega, e, cfg),
        })
    return result


def aborted_m2_prejump(cfg, model, design, omega, omega_pre_x, path):
    if not path.is_file():
        return {"status": "aborted_M2_step_log_unavailable"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["episode"] == "3"]
    if len(rows) <= 40 or rows[40]["scenario_type"] != "concentration_positive":
        return {"status": "aborted_M2_episode_3_not_present"}
    x = cfg.robust_economic_reference_state.copy()
    for second, row in enumerate(rows):
        before = x.copy()
        x = apply_state_shock(x, "concentration_positive", second, scale=1.0)
        if second == 40:
            e_minus = model.normalized_state(before) - design.z_ref
            e_plus = model.normalized_state(x) - design.z_ref
            delta = e_plus - e_minus
            expected = np.array([1.0 / cfg.state_scale[0], 0.0])
            if np.linalg.norm(delta - expected) > 1e-10:
                raise RuntimeError("Episode-3 t=40 state jump differs from Paper2016")
            return {
                "status": "reconstructed_from_saved_applied_inputs",
                "state_pre_jump": before.tolist(),
                "state_post_jump": x.tolist(),
                "xi_pre_jump_normalized": e_minus.tolist(),
                "xi_post_jump_normalized": e_plus.tolist(),
                "pre_in_Omega": contains(omega, e_minus),
                "pre_in_Omega_pre_concentration_positive": contains(
                    omega_pre_x, e_minus
                ),
                "post_in_Omega": contains(omega, e_plus),
                "pre_margin_to_Omega_physical": margin(omega, e_minus, cfg),
                "pre_margin_to_Omega_pre_physical": margin(
                    omega_pre_x, e_minus, cfg
                ),
                "post_margin_to_Omega_physical": margin(omega, e_plus, cfg),
                "logged_mode_at_jump": row["safety_mode"],
                "logged_qp_feasible_at_jump": row["qp_feasible"],
            }
        u = np.array([
            float(row["final_applied_P100"]), float(row["final_applied_F200"])
        ])
        x = model.step(x, u, cfg.disturbance_nominal)
    raise RuntimeError("Episode-3 t=40 jump missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--aborted-action-log", type=Path, default=DEFAULT_ABORTED)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.max_iterations <= 1000:
        raise ValueError("max-iterations must be 1..1000")
    cfg = ExperimentConfig(
        benchmark_profile="zanon2016", experiment_mode="proposed",
        steps_per_episode=300, seed=args.seed,
        residual_parameterization="state_dependent_polytope",
    )
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    domain, _ = bounds_and_domains(cfg, model, design)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    omega_certificate = certificate(omega, domain, design)
    if not omega_certificate["passed"]:
        raise RuntimeError(f"Fixed Omega certificate failed: {omega_certificate}")
    deltas = {
        "pressure_positive": np.array([0.0, 1.0 / cfg.state_scale[1]]),
        "pressure_negative": np.array([0.0, -1.0 / cfg.state_scale[1]]),
        "concentration_positive": np.array([1.0 / cfg.state_scale[0], 0.0]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    omega_area = area(omega)
    one_jump = {}
    pre_polys = {}
    pre_all = omega.copy()
    for name, delta in deltas.items():
        poly = hull(intersect(omega, jump_pre(omega, delta)))
        pre_polys[name] = poly
        pre_all = hull(intersect(pre_all, poly)) if full_dimensional(pre_all) else pre_all
        write_vertices(args.output_dir / f"Omega_pre_{name}_vertices.csv", poly, cfg)
        one_jump[name] = {
            **set_properties(poly, cfg.state_scale),
            "area_over_Omega": area(poly) / omega_area,
            "balanced_reference_in_set": contains(poly, np.zeros(2)),
            "balanced_reference_margin_physical": margin(poly, np.zeros(2), cfg),
        }
    write_vertices(args.output_dir / "Omega_pre_all_vertices.csv", pre_all, cfg)
    finite = {}
    for name, delta in deltas.items():
        sets = [omega]
        levels = {}
        for count in (1, 2, 3):
            jump_domain = hull(intersect(omega, jump_pre(sets[-1], delta)))
            write_vertices(args.output_dir /
                           f"{name}_C{count}_jump_domain_vertices.csv",
                           jump_domain, cfg)
            if full_dimensional(jump_domain):
                current, history, converged = iterate_invariant(
                    jump_domain, design.a, design.b, design.w_vertices,
                    domain["q_lower"], domain["q_upper"], args.max_iterations,
                )
                current = hull(current) if full_dimensional(current) else np.empty((0, 2))
            else:
                current, history, converged = np.empty((0, 2)), [], False
            sets.append(current)
            write_vertices(args.output_dir / f"{name}_C{count}_vertices.csv",
                           current, cfg)
            write_history(args.output_dir / f"{name}_C{count}_area_history.csv",
                          history)
            controls = representative_controls(current, design, domain, model, cfg)
            cert = (certificate(current, domain, design)
                    if full_dimensional(current) else
                    {"passed": False, "reason": "empty_or_lower_dimensional"})
            regular_excess = maximum_facet_excess(current, omega)
            jump_excess = maximum_facet_excess(current + delta, sets[-2])
            certified = bool(
                full_dimensional(current) and converged and cert["passed"]
                and controls["all_feasible"]
                and regular_excess is not None and regular_excess <= 2e-7
                and jump_excess is not None and jump_excess <= 2e-7
            )
            if full_dimensional(current) and not certified:
                raise RuntimeError(
                    f"{name} C{count} failed a finite-shock safety gate: "
                    f"converged={converged}, regular_cert={cert['passed']}, "
                    f"controls={controls['all_feasible']}, "
                    f"Omega_excess={regular_excess}, jump_excess={jump_excess}"
                )
            levels[f"C{count}"] = {
                **set_properties(current, cfg.state_scale),
                "nonempty_full_dimensional": full_dimensional(current),
                "area_over_Omega": area(current) / omega_area,
                "jump_domain_area_physical": area(jump_domain) * float(
                    np.prod(cfg.state_scale)
                ),
                "predecessor_iterations": len(history),
                "numerically_converged": converged,
                "fully_certified_at_numerical_tolerance": certified,
                "C_m_max_facet_excess_over_Omega": regular_excess,
                "jumped_C_m_max_facet_excess_over_C_m_minus_1": jump_excess,
                "reference_in_set": contains(current, np.zeros(2)),
                "reference_margin_physical": margin(current, np.zeros(2), cfg),
                "regular_step_vertex_certificate": cert,
                "representative_action_feasibility": controls,
            }
            print(name, f"C{count}", "area_physical=",
                  levels[f"C{count}"]["area_physical"],
                  "iterations=", len(history), "converged=", converged,
                  flush=True)
        # Algebraic immediate jump check, independent of a particular policy.
        immediate = []
        for jump_number in (1, 2, 3):
            e = jump_number * delta
            target = sets[3 - jump_number]
            immediate.append({
                "jump_number": jump_number,
                "remaining_shocks": 3 - jump_number,
                "xi_if_no_regular_transition": e.tolist(),
                "in_required_C": contains(target, e),
                "margin_to_required_C_physical": margin(target, e, cfg),
            })
        finite[name] = {
            "delta_normalized": delta.tolist(),
            "delta_physical": (delta * cfg.state_scale).tolist(),
            "levels": levels,
            "initial_reference_in_C3": contains(sets[3], np.zeros(2)),
            "initial_reference_margin_to_C3_physical": margin(
                sets[3], np.zeros(2), cfg
            ),
            "immediate_repeated_jump_membership": immediate,
            "paired_zero_residual_0_20_40_shock_membership": (
                zero_baseline_shock_membership(
                    cfg, model, design, omega, domain, name, sets
                )
            ),
        }
    q_rows = []
    for scenario, scenario_data in finite.items():
        for level, level_data in scenario_data["levels"].items():
            for sample in level_data["representative_action_feasibility"]["representatives"]:
                control = sample["control"]
                q_rows.append({
                    "scenario": scenario,
                    "set": level,
                    "representative": sample["name"],
                    "xi_X2_normalized": sample["xi_normalized"][0],
                    "xi_P2_normalized": sample["xi_normalized"][1],
                    "q_P100_min_normalized": control["q_min_normalized"][0],
                    "q_P100_max_normalized": control["q_max_normalized"][0],
                    "q_F200_min_normalized": control["q_min_normalized"][1],
                    "q_F200_max_normalized": control["q_max_normalized"][1],
                    "P100_min_physical": control["P100_F200_min_physical"][0],
                    "P100_max_physical": control["P100_F200_max_physical"][0],
                    "F200_min_physical": control["P100_F200_min_physical"][1],
                    "F200_max_physical": control["P100_F200_max_physical"][1],
                    "max_regular_W_next_target_facet_excess": control[
                        "max_regular_W_next_target_facet_excess"
                    ],
                })
    write_history(args.output_dir / "representative_q_ranges.csv", q_rows)
    output = {
        "scope": "offline fixed-B linearized set analysis; no SAC training",
        "regular_uncertainty": "xi_next=A xi+B q+w, w in unchanged certified W",
        "jump_operator": "xi_plus=xi_minus+delta; delta is not part of W",
        "shock_clock_used_in_set_computation": False,
        "quantifier_interpretation": (
            "At every regular step a feasible q must keep the state in C_m "
            "for all w in W; an unpredictable instantaneous scenario-specific "
            "jump maps C_m into C_(m-1). Consecutive jumps are permitted."
        ),
        "linear_model_caveat": (
            "Numerical 2D polytope computation under the fixed affine model; "
            "nonlinear continuous-domain coverage is not proved here."
        ),
        "reference_state_physical": model.physical_state(design.z_ref).tolist(),
        "reference_input_physical": model.physical_input(design.v_ref).tolist(),
        "Omega": set_properties(omega, cfg.state_scale),
        "Omega_physical_state_extent": {
            "lower": model.physical_state(
                design.z_ref + np.min(omega, axis=0)
            ).tolist(),
            "upper": model.physical_state(
                design.z_ref + np.max(omega, axis=0)
            ).tolist(),
            "width": (np.ptp(omega, axis=0) * cfg.state_scale).tolist(),
        },
        "Omega_certificate": omega_certificate,
        "Omega_pre_by_scenario": one_jump,
        "Omega_pre_all": {
            **set_properties(pre_all, cfg.state_scale),
            "area_over_Omega": area(pre_all) / omega_area,
            "balanced_reference_in_set": contains(pre_all, np.zeros(2)),
            "balanced_reference_margin_physical": margin(
                pre_all, np.zeros(2), cfg
            ),
        },
        "aborted_M2_episode_3_t40": aborted_m2_prejump(
            cfg, model, design, omega, pre_polys["concentration_positive"],
            args.aborted_action_log,
        ),
        "finite_shock_sets": finite,
    }
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
