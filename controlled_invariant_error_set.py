"""Offline 2D robust controlled-invariant error sets at certified reference B.

This is predecessor-set iteration, not MPC or an online optimizer.  Two
quantifier scopes are reported: fixed B anchor, and a conservative common
error domain valid for every z in S and every v_base in tightened U.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .control import point_in_convex_polygon
from .model import EvaporatorModel
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--reachability-iterations", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_controlled_invariant_error_B"
    ))
    return parser.parse_args()


def hull(points: np.ndarray, tol: float = 1e-11) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) == 0:
        return points
    unique = np.unique(np.round(points / tol).astype(np.int64), axis=0) * tol
    if len(unique) <= 2:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def cross(o, a, b):
        u, v = a - o, b - o
        return float(u[0] * v[1] - u[1] * v[0])

    lower = []
    for p in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-14:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-14:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def rectangle(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    if np.any(upper < lower):
        return np.empty((0, 2))
    return np.array([
        [lower[0], lower[1]], [upper[0], lower[1]],
        [upper[0], upper[1]], [lower[0], upper[1]],
    ], dtype=float)


def facets(vertices: np.ndarray):
    v = np.asarray(vertices, dtype=float)
    if len(v) < 3:
        return np.empty((0, 2)), np.empty(0)
    edges = np.roll(v, -1, axis=0) - v
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 1e-10
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])[keep]
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, None]
    bounds = np.sum(normals * v[keep], axis=1)
    return normals, bounds


def clip_halfspace(vertices: np.ndarray, normal: np.ndarray, bound: float,
                   tol: float = 1e-11) -> np.ndarray:
    poly = np.asarray(vertices, dtype=float)
    if len(poly) == 0:
        return poly
    output = []
    for start, end in zip(poly, np.roll(poly, -1, axis=0)):
        ds = float(normal @ start - bound)
        de = float(normal @ end - bound)
        if ds <= tol:
            output.append(start)
        if (ds < -tol and de > tol) or (ds > tol and de < -tol):
            fraction = ds / (ds - de)
            output.append(start + fraction * (end - start))
    # Sutherland-Hodgman preserves cyclic convex order. Re-hulling after every
    # facet dominates the runtime on the many-facet certified RPI target.
    return np.asarray(output).reshape(-1, 2) if output else np.empty((0, 2))


def intersect(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    current = np.asarray(first, dtype=float)
    normals, bounds = facets(second)
    if len(current) == 0 or len(normals) == 0:
        return np.empty((0, 2))
    for n, b in zip(normals, bounds):
        current = clip_halfspace(current, n, b)
        if len(current) == 0:
            break
    return current


def area(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    return 0.5 * abs(float(np.sum(
        vertices[:, 0] * np.roll(vertices[:, 1], -1)
        - vertices[:, 1] * np.roll(vertices[:, 0], -1)
    )))


def contains(vertices: np.ndarray, point: np.ndarray, tol=1e-8) -> bool:
    return len(vertices) >= 3 and point_in_convex_polygon(point, vertices, tol=tol)


def predecessor(target, a, b, w_vertices, q_lower, q_upper):
    """Pre(target) for e+=A e+B q+w, q in a 2D input rectangle."""
    if len(target) < 3:
        return np.empty((0, 2))
    eroded = np.asarray(target, dtype=float)
    normals, bounds = facets(target)
    for normal, bound in zip(normals, bounds):
        support_w = float(np.max(w_vertices @ normal))
        eroded = clip_halfspace(eroded, normal, bound - support_w)
        if len(eroded) < 3:
            return np.empty((0, 2))
    q_vertices = rectangle(q_lower, q_upper)
    negative_input = -(q_vertices @ b.T)
    summed = hull((eroded[:, None, :] + negative_input[None, :, :]).reshape(-1, 2))
    return hull(summed @ np.linalg.inv(a).T)


def iterate_invariant(domain, a, b, w_vertices, q_lower, q_upper, max_iterations):
    current = np.asarray(domain, dtype=float)
    history = []
    converged = False
    for iteration in range(1, max_iterations + 1):
        pre = predecessor(current, a, b, w_vertices, q_lower, q_upper)
        next_set = intersect(current, pre)
        old_area, next_area = area(current), area(next_set)
        inclusion_gap = 0.0
        if len(next_set) >= 3:
            normals, bounds = facets(next_set)
            inclusion_gap = max(0.0, float(np.max(normals @ current.T - bounds[:, None])))
        else:
            inclusion_gap = float("inf")
        history.append({
            "iteration": iteration, "vertices": len(next_set),
            "area_normalized": next_area,
            "area_removed": old_area - next_area,
            "old_to_new_max_facet_excess": inclusion_gap,
        })
        current = next_set
        if len(current) < 3:
            break
        if inclusion_gap <= 1e-9 and abs(old_area - next_area) <= 1e-12:
            converged = True
            break
    return current, history, converged


def feasible_control_polygon(e, omega, a, b, w_vertices, q_lower, q_upper):
    q_poly = rectangle(q_lower, q_upper)
    normals, bounds = facets(omega)
    if len(normals) == 0:
        return np.empty((0, 2))
    for normal, bound in zip(normals, bounds):
        q_poly = clip_halfspace(
            q_poly, normal @ b,
            bound - normal @ (a @ e) - np.max(w_vertices @ normal),
        )
        if len(q_poly) == 0:
            break
    return q_poly


def polygon_distance(vertices: np.ndarray, point: np.ndarray, scale):
    physical_v = np.asarray(vertices) * scale
    p = np.asarray(point) * scale
    edges = np.roll(physical_v, -1, axis=0) - physical_v
    starts = physical_v
    denom = np.sum(edges * edges, axis=1)
    useful = denom > 1e-18
    fractions = np.zeros(len(edges))
    fractions[useful] = np.clip(
        np.sum((p - starts[useful]) * edges[useful], axis=1) / denom[useful],
        0.0, 1.0,
    )
    nearest = starts + fractions[:, None] * edges
    distance = float(np.min(np.linalg.norm(nearest - p, axis=1)))
    return -distance if contains(vertices, point) else distance


def set_properties(vertices, state_scale):
    return {
        "vertices_normalized": np.asarray(vertices).tolist(),
        "vertices_physical_error": (np.asarray(vertices) * state_scale).tolist(),
        "vertex_count": len(vertices),
        "area_normalized": area(vertices),
        "area_physical": area(vertices) * float(np.prod(state_scale)),
    }


def bounds_and_domains(cfg, model, design):
    x_lo = np.maximum(model.normalized_state(cfg.state_lower), design.robust_state_lower)
    x_hi = np.minimum(model.normalized_state(cfg.state_upper), design.robust_state_upper)
    u_lo = np.maximum(model.normalized_input(cfg.input_lower), design.robust_input_lower)
    u_hi = np.minimum(model.normalized_input(cfg.input_upper), design.robust_input_upper)
    anchor = {
        "e_lower": x_lo - design.z_ref,
        "e_upper": x_hi - design.z_ref,
        "q_lower": u_lo - design.v_ref,
        "q_upper": u_hi - design.v_ref,
    }
    # A deliberately conservative common correction q for every z in S and
    # every base in tightened U. This is an inner approximation, not a claim
    # that a state/base-dependent feedback cannot work outside it.
    uniform = {
        "e_lower": x_lo - design.invariant_lower,
        "e_upper": x_hi - design.invariant_upper,
        "q_lower": u_lo - design.u_lower_tight,
        "q_upper": u_hi - design.u_upper_tight,
    }
    return anchor, uniform


def certificate(omega, domain, design, tol=2e-7):
    if len(omega) < 3:
        return {"passed": False, "reason": "empty_or_degenerate_set"}
    tested = []
    max_next_excess = -float("inf")
    min_q_area = float("inf")
    min_state_margin = float("inf")
    normals, bounds = facets(omega)
    for e in omega:
        q_poly = feasible_control_polygon(
            e, omega, design.a, design.b, design.w_vertices,
            domain["q_lower"], domain["q_upper"],
        )
        if len(q_poly) < 3:
            return {"passed": False, "reason": "vertex_has_no_2d_admissible_control",
                    "failed_vertex": e.tolist(), "control_vertices": q_poly.tolist()}
        q = np.mean(q_poly, axis=0)
        next_states = design.a @ e + design.b @ q + design.w_vertices
        max_next_excess = max(max_next_excess, float(np.max(next_states @ normals.T - bounds)))
        min_q_area = min(min_q_area, area(q_poly))
        min_state_margin = min(min_state_margin, float(np.min(np.concatenate([
            e - domain["e_lower"], domain["e_upper"] - e,
        ]))))
        tested.append([*e, *q])
    return {
        "passed": bool(max_next_excess <= tol and min_state_margin >= -tol),
        "reason": "vertex_feedback_certificate" if max_next_excess <= tol else "next_state_excess",
        "tested_vertices": len(tested),
        "max_next_facet_excess_normalized": max_next_excess,
        "minimum_state_box_margin_normalized": min_state_margin,
        "minimum_feasible_q_polygon_area": min_q_area,
    }


def control_report(e, omega, domain, design, model, cfg):
    q_poly = feasible_control_polygon(
        e, omega, design.a, design.b, design.w_vertices,
        domain["q_lower"], domain["q_upper"],
    )
    if len(q_poly) < 3:
        return {"robust_one_step_feasible": False, "vertices": q_poly.tolist()}
    du_poly = q_poly - design.k @ e
    physical_u = model.physical_input(design.v_ref + q_poly)
    q_mid = np.mean(q_poly, axis=0)
    central_slack = np.minimum(q_mid - domain["q_lower"],
                               domain["q_upper"] - q_mid)
    return {
        "robust_one_step_feasible": True,
        "feasible_q_vertices": q_poly.tolist(),
        "feasible_du_vertices": du_poly.tolist(),
        "q_range_normalized": [np.min(q_poly, axis=0).tolist(), np.max(q_poly, axis=0).tolist()],
        "du_range_normalized": [np.min(du_poly, axis=0).tolist(), np.max(du_poly, axis=0).tolist()],
        "physical_input_range_P100_F200": [
            np.min(physical_u, axis=0).tolist(), np.max(physical_u, axis=0).tolist(),
        ],
        "q_polygon_area_normalized": area(q_poly),
        "central_input_box_slack_normalized": central_slack.tolist(),
        "central_input_box_slack_physical": (central_slack * cfg.input_scale).tolist(),
        "feasible_input_width_physical": (
            (np.max(q_poly, axis=0) - np.min(q_poly, axis=0)) * cfg.input_scale
        ).tolist(),
        "central_q_normalized": q_mid.tolist(),
        "central_du_normalized": (q_mid - design.k @ e).tolist(),
        "input_saturation_forced": [
            bool(np.isclose(np.min(q_poly[:, j]), domain["q_upper"][j], atol=1e-7)
                 or np.isclose(np.max(q_poly[:, j]), domain["q_lower"][j], atol=1e-7))
            for j in range(2)
        ],
    }


def reachability(omega, z, design, domain, shocks, max_iterations):
    reachable = intersect(omega, hull(z))
    history = []
    first = {name: (0 if contains(reachable, e) else None) for name, e in shocks.items()}
    for iteration in range(1, max_iterations + 1):
        pre = predecessor(reachable, design.a, design.b, design.w_vertices,
                          domain["q_lower"], domain["q_upper"])
        new = hull(np.vstack([reachable, intersect(omega, pre)])) if len(pre) >= 3 else reachable
        # If R_0=Z∩Ω is not robust invariant, a union with its predecessor
        # would not by itself certify robust reachability. Explicitly audit it.
        growth = area(new) - area(reachable)
        history.append({"iteration": iteration, "area_normalized": area(new),
                        "area_growth": growth})
        reachable = new
        for name, e in shocks.items():
            if first[name] is None and contains(reachable, e):
                first[name] = iteration
        if growth <= 1e-12:
            break
    return reachable, history, first


def write_csv(path, vertices, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(np.asarray(vertices).tolist())


def main():
    args = parse_args()
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed")
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    domains = dict(zip(("anchor_B", "uniform_S_conservative"),
                       bounds_and_domains(cfg, model, design)))
    z = hull(design.rpi_boundary)
    shocks = {
        "pressure_positive": np.array([0.0, 1.0 / cfg.state_scale[1]]),
        "pressure_negative": np.array([0.0, -1.0 / cfg.state_scale[1]]),
        "concentration_positive": np.array([1.0 / cfg.state_scale[0], 0.0]),
    }
    result = {
        "scope_note": (
            "anchor_B certifies fixed z_ref/v_ref; uniform_S_conservative is a "
            "common-q inner approximation for all z in S and all v_base in tightened U. "
            "Neither is a joint (z,e) reachability certificate for actual nonlinear state shocks."
        ),
        "linearized_dynamics_note": (
            "q=K e+du, hence e_next=A e+B q+w; without an independent bound on du, "
            "the existential controlled-invariant set is independent of K."
        ),
        "reference_state_physical": model.physical_state(design.z_ref).tolist(),
        "reference_input_physical": model.physical_input(design.v_ref).tolist(),
        "A": design.a.tolist(), "B": design.b.tolist(), "K": design.k.tolist(),
        "W_vertices_normalized": design.w_vertices.tolist(),
        "Z": set_properties(z, cfg.state_scale),
        "domains": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "Z_vertices.csv", z, ("e_X2_normalized", "e_P2_normalized"))
    for label, domain in domains.items():
        initial = rectangle(domain["e_lower"], domain["e_upper"])
        iteration_limit = args.max_iterations if label == "anchor_B" else min(args.max_iterations, 300)
        omega, history, converged = iterate_invariant(
            initial, design.a, design.b, design.w_vertices,
            domain["q_lower"], domain["q_upper"], iteration_limit,
        )
        cert = certificate(omega, domain, design)
        shock_rows = {}
        for name, e in shocks.items():
            margins = np.concatenate([e - domain["e_lower"], domain["e_upper"] - e])
            shock_rows[name] = {
                "e_normalized": e.tolist(),
                "e_physical_error": (e * cfg.state_scale).tolist(),
                "in_Z": contains(z, e), "in_Omega": contains(omega, e),
                "signed_distance_to_Z_physical": polygon_distance(z, e, cfg.state_scale),
                "signed_distance_to_Omega_physical": (
                    polygon_distance(omega, e, cfg.state_scale) if len(omega) >= 3 else None
                ),
                "minimum_state_domain_margin_normalized": float(np.min(margins)),
                "state_domain_margins_physical": (margins * np.tile(cfg.state_scale, 2)).tolist(),
                "control": control_report(e, omega, domain, design, model, cfg),
            }
        rep = {}
        if len(omega) >= 3:
            for name, e in {"reference": np.zeros(2), "centroid": np.mean(omega, axis=0),
                            **shocks}.items():
                rep[name] = control_report(e, omega, domain, design, model, cfg)
            rep["Omega_vertices"] = [
                {"e_normalized": e.tolist(),
                 "control": control_report(e, omega, domain, design, model, cfg)}
                for e in omega
            ]
        r0 = intersect(omega, z)
        target_cert = certificate(r0, domain, design) if len(r0) >= 3 else {
            "passed": False, "reason": "Z_intersection_empty"
        }
        reachable, reach_history, first = reachability(
            omega, z, design, domain, shocks, args.reachability_iterations,
        ) if converged and cert["passed"] and target_cert["passed"] else (
            r0, [], {name: None for name in shocks}
        )
        result["domains"][label] = {
            "domain_bounds_normalized": {key: val.tolist() for key, val in domain.items()},
            "Omega": set_properties(omega, cfg.state_scale),
            "Omega_vertices_absolute_state_physical": (
                model.physical_state(design.z_ref)[None, :]
                + omega * cfg.state_scale[None, :]
            ).tolist(),
            "Omega_over_Z_area_ratio": area(omega) / area(z) if area(z) else None,
            "predecessor_iterations": len(history),
            "predecessor_converged": converged,
            "fully_robust_controlled_invariant": bool(converged and cert["passed"]),
            "iteration_status": (
                "certified_fixed_point" if converged and cert["passed"]
                else "uncertified_iteration_set"
            ),
            "vertex_certificate": cert,
            "shock_states": shock_rows,
            "representative_controls": rep,
            "Z_intersection_target_invariant": target_cert,
            "robust_backward_reachability": {
                "evaluated": bool(converged and cert["passed"] and target_cert["passed"]),
                "status": (
                    "bounded_diagnosis_only" if reach_history and
                    reach_history[-1]["area_growth"] > 1e-12
                    else "converged" if reach_history else "not_evaluated_uncertified_domain"
                ),
                "first_iteration_from_shock_to_Z": first,
                "iterations": len(reach_history),
                "final_area_normalized": area(reachable),
                "converged": bool(reach_history and reach_history[-1]["area_growth"] <= 1e-12),
            },
        }
        write_csv(args.output_dir / f"Omega_{label}_vertices.csv", omega,
                  ("e_X2_normalized", "e_P2_normalized"))
        write_csv(args.output_dir / f"predecessor_{label}_history.csv",
                  [[row[k] for k in ("iteration", "vertices", "area_normalized", "area_removed",
                                     "old_to_new_max_facet_excess")] for row in history],
                  ("iteration", "vertices", "area_normalized", "area_removed",
                   "old_to_new_max_facet_excess"))
        write_csv(args.output_dir / f"reachability_{label}_history.csv",
                  [[row[k] for k in ("iteration", "area_normalized", "area_growth")]
                   for row in reach_history],
                  ("iteration", "area_normalized", "area_growth"))
        print(label, "area", area(omega), "iterations", len(history),
              "converged", converged, "certificate", cert["passed"], flush=True)
        for name, row in shock_rows.items():
            print(" ", name, "in Z", row["in_Z"], "in Omega", row["in_Omega"],
                  "control", row["control"]["robust_one_step_feasible"], flush=True)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved", output)


if __name__ == "__main__":
    main()
