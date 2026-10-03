"""Offline robust backward reachability from the certified RPI Z inside B's Omega.

Uses only the total normalized error correction q in e+ = A e + B q + w.
The actual physical input is u = v_ref + input_scale * q; there is no
unbounded ancillary/correction decomposition or online optimization here.
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
    feasible_control_polygon, hull, intersect, polygon_distance, predecessor,
    set_properties,
)
from .model import EvaporatorModel
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b


DEFAULT_OMEGA = REPO_DIR / (
    "evaporation_safe_sac/outputs_controlled_invariant_error_B/"
    "Omega_anchor_B_vertices.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_robust_backward_reachability_B"
    ))
    return parser.parse_args()


def maximum_facet_excess(points, poly):
    if len(points) == 0 or len(poly) < 3:
        return float("inf")
    normals, bounds = facets(poly)
    return float(np.max(points @ normals.T - bounds))


def q_report(e, target, design, domain, model, cfg):
    polygon = hull(feasible_control_polygon(
        e, target, design.a, design.b, design.w_vertices,
        domain["q_lower"], domain["q_upper"],
    ))
    if len(polygon) < 3:
        return {"feasible": False, "q_vertices_normalized": polygon.tolist()}
    u_vertices = model.physical_input(design.v_ref + polygon)
    q_center = np.mean(polygon, axis=0)
    next_vertices = design.a @ e + design.b @ q_center + design.w_vertices
    next_excess = maximum_facet_excess(next_vertices, target)
    return {
        "feasible": bool(next_excess <= 2e-7),
        "q_vertices_normalized": polygon.tolist(),
        "q_range_normalized": [
            np.min(polygon, axis=0).tolist(), np.max(polygon, axis=0).tolist(),
        ],
        "actual_input_range_P100_F200": [
            np.min(u_vertices, axis=0).tolist(), np.max(u_vertices, axis=0).tolist(),
        ],
        "central_q_normalized": q_center.tolist(),
        "central_actual_input_P100_F200": model.physical_input(
            design.v_ref + q_center
        ).tolist(),
        "max_next_target_facet_excess_normalized": next_excess,
        "polygon_area_normalized": area(polygon),
    }


def main():
    args = parse_args()
    if not 0 <= args.max_iterations <= 1000:
        raise ValueError("max-iterations must be between 0 and 1000")
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed")
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    domain, _ = bounds_and_domains(cfg, model, design)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    z = hull(design.rpi_boundary)
    if not certificate(omega, domain, design)["passed"]:
        raise RuntimeError("Loaded Omega failed its vertex invariance check")
    z_in_omega = maximum_facet_excess(z, omega)
    if z_in_omega > 1e-8:
        raise RuntimeError(f"R0=Z is not contained in Omega: excess={z_in_omega}")
    z_cert = certificate(z, domain, design)
    if not z_cert["passed"]:
        raise RuntimeError("R0=Z is not robust controlled invariant under admissible q")

    shocks = {
        "pressure_positive": np.array([0.0, 1.0 / cfg.state_scale[1]]),
        "pressure_negative": np.array([0.0, -1.0 / cfg.state_scale[1]]),
        "concentration_positive": np.array([1.0 / cfg.state_scale[0], 0.0]),
    }
    first_hit = {name: (0 if contains(z, e) else None) for name, e in shocks.items()}
    first_action = {name: None for name in shocks}
    current = z.copy()  # R0 is exactly Z, not Z intersect Omega.
    history = [{"k": 0, "vertices": len(current),
                "area_normalized": area(current),
                "area_physical": area(current) * float(np.prod(cfg.state_scale)),
                "area_gain_normalized": 0.0,
                "old_to_new_facet_excess_normalized": 0.0,
                "new_to_old_facet_excess_normalized": 0.0}]
    converged = False
    tolerance = {"inclusion": 1e-8, "hausdorff": 1e-8, "area": 1e-10}
    for k in range(1, args.max_iterations + 1):
        pre = predecessor(current, design.a, design.b, design.w_vertices,
                          domain["q_lower"], domain["q_upper"])
        next_set = hull(intersect(omega, pre)) if len(pre) >= 3 else np.empty((0, 2))
        if len(next_set) < 3:
            raise RuntimeError(f"R_{k} is empty despite invariant R0")
        old_to_new = maximum_facet_excess(current, next_set)
        new_to_old = maximum_facet_excess(next_set, current)
        if old_to_new > tolerance["inclusion"]:
            raise RuntimeError(f"Nested reachable sets failed at k={k}: {old_to_new}")
        gain = area(next_set) - area(current)
        history.append({
            "k": k, "vertices": len(next_set),
            "area_normalized": area(next_set),
            "area_physical": area(next_set) * float(np.prod(cfg.state_scale)),
            "area_gain_normalized": gain,
            "old_to_new_facet_excess_normalized": old_to_new,
            "new_to_old_facet_excess_normalized": new_to_old,
        })
        for name, e in shocks.items():
            if first_hit[name] is None and contains(next_set, e, tol=1e-9):
                first_hit[name] = k
                first_action[name] = q_report(e, current, design, domain, model, cfg)
                if not first_action[name]["feasible"]:
                    raise RuntimeError(f"First-hit control check failed: {name} at k={k}")
        current = next_set
        if k % 100 == 0 or any(step == k for step in first_hit.values()):
            print(f"k={k} area={history[-1]['area_physical']:.9g} "
                  f"vertices={len(current)} first_hit={first_hit}", flush=True)
        if max(0.0, new_to_old) <= tolerance["hausdorff"] and abs(gain) <= tolerance["area"]:
            converged = True
            break

    final = current
    shocks_result = {}
    for name, e in shocks.items():
        shocks_result[name] = {
            "e_normalized": e.tolist(),
            "e_physical_error": (e * cfg.state_scale).tolist(),
            "in_Omega": contains(omega, e),
            "first_R_k": first_hit[name],
            "in_final_R": contains(final, e, tol=1e-9),
            "signed_distance_to_final_R_physical": polygon_distance(
                final, e, cfg.state_scale
            ),
            "signed_distance_to_R_inf_physical": (
                polygon_distance(final, e, cfg.state_scale) if converged else None
            ),
            "admissible_q_box_normalized": [
                domain["q_lower"].tolist(), domain["q_upper"].tolist(),
            ],
            "admissible_actual_input_box_P100_F200": [
                model.physical_input(design.v_ref + domain["q_lower"]).tolist(),
                model.physical_input(design.v_ref + domain["q_upper"]).tolist(),
            ],
            "robust_one_step_q_to_Omega": q_report(
                e, omega, design, domain, model, cfg
            ),
            "first_step_q_toward_Z": first_action[name],
            "robust_path_to_Z_found_within_scan": first_hit[name] is not None,
        }

    output = {
        "model": "e_next=A e+B q+w; u=v_ref+q (normalized input), w in W",
        "scope": "Fixed balanced-B reference and certified anchor Omega; offline linearized reachability only",
        "convergence": {
            "numerically_converged": converged, "iterations": len(history) - 1,
            "maximum_iterations": args.max_iterations, "tolerances": tolerance,
            "not_covered_interpretation": (
                "Not covered by R_k within the scan; not an absolute unreachable proof."
            ),
        },
        "R0_Z_vertex_certificate": z_cert,
        "R0_Z_max_facet_excess_over_Omega": z_in_omega,
        "reference_state_physical": model.physical_state(design.z_ref).tolist(),
        "reference_input_physical": model.physical_input(design.v_ref).tolist(),
        "Omega": set_properties(omega, cfg.state_scale),
        "Z": set_properties(z, cfg.state_scale),
        "R_final": set_properties(final, cfg.state_scale),
        "R_final_index": len(history) - 1,
        "R_inf_area_physical": (
            area(final) * float(np.prod(cfg.state_scale)) if converged else None
        ),
        "R_inf_status": (
            "numerical_fixed_point_estimate" if converged
            else "not_established_at_iteration_limit"
        ),
        "R_final_over_Omega_area_ratio": area(final) / area(omega),
        "R_inf_over_Omega_area_ratio": (
            area(final) / area(omega) if converged else None
        ),
        "shock_states": shocks_result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "R_area_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (args.output_dir / "R_final_vertices.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["e_X2_normalized", "e_P2_normalized"])
        writer.writerows(final.tolist())
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", path, "converged", converged, "first_hit", first_hit, flush=True)


if __name__ == "__main__":
    main()
