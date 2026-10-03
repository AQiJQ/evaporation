"""Offline one-step Omega safety projection and R_k rank-decrease diagnosis.

This is a diagnostic prototype, not an integrated controller or MPC.  All
claims are conditional on w_actual belonging to the fixed certified W.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .control import project_qp_2d
from .controlled_invariant_error_set import (
    bounds_and_domains, contains, facets, hull, intersect, predecessor,
    rectangle,
)
from .model import EvaporatorModel
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b


OUTPUT_ROOT = REPO_DIR / "evaporation_safe_sac"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=OUTPUT_ROOT / (
        "outputs_controlled_invariant_error_B/Omega_anchor_B_vertices.csv"
    ))
    parser.add_argument("--coverage", type=Path, default=OUTPUT_ROOT / (
        "outputs_omega_nonlinear_mismatch_B/coverage_summary.json"
    ))
    parser.add_argument("--reachability", type=Path, default=OUTPUT_ROOT / (
        "outputs_robust_backward_reachability_B/summary.json"
    ))
    parser.add_argument("--random-states", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=420017)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / (
        "outputs_omega_safe_filter_B"
    ))
    return parser.parse_args()


def qp_rows(xi, target, design, domain):
    normals, bounds = facets(target)
    support = np.max(design.w_vertices @ normals.T, axis=0)
    rows = np.vstack([np.eye(2), -np.eye(2), normals @ design.b])
    rhs = np.concatenate([
        domain["q_upper"], -domain["q_lower"],
        bounds - normals @ (design.a @ xi) - support,
    ])
    return rows, rhs


def project_action(xi, candidate, target, design, domain, model, cfg):
    rows, rhs = qp_rows(xi, target, design, domain)
    safe_q, feasible = project_qp_2d(candidate, rows, rhs, tol=1e-9)
    if not feasible:
        return {"feasible": False}
    inequality_excess = float(np.max(rows @ safe_q - rhs))
    next_states = design.a @ xi + design.b @ safe_q + design.w_vertices
    normals, bounds = facets(target)
    next_excess = float(np.max(next_states @ normals.T - bounds))
    actual_u = model.physical_input(design.v_ref + safe_q)
    actual_next = model.normalized_state(model.step(
        model.physical_state(xi), actual_u, cfg.disturbance_nominal
    ))
    predicted = design.a @ xi + design.b @ safe_q + design.affine
    mismatch = actual_next - predicted
    w_normals, w_bounds = facets(hull(design.w_vertices))
    w_max_util = float(np.max((w_normals @ mismatch) / w_bounds))
    actual_next_excess = float(np.max(normals @ actual_next - bounds))
    return {
        "feasible": bool(inequality_excess <= 2e-7 and next_excess <= 2e-7),
        "q_candidate_normalized": np.asarray(candidate).tolist(),
        "q_safe_normalized": safe_q.tolist(),
        "q_safe_physical_P100_F200": (safe_q * cfg.input_scale).tolist(),
        "projection_distance_normalized": float(np.linalg.norm(safe_q - candidate)),
        "actual_input_P100_F200": actual_u.tolist(),
        "max_QP_inequality_excess_normalized": inequality_excess,
        "max_W_vertex_next_target_excess_normalized": next_excess,
        "nonlinear_W_max_facet_utilization": w_max_util,
        "nonlinear_next_target_excess_normalized": actual_next_excess,
    }


def rank_targets(omega, z, design, domain, ranks):
    desired = {int(k) - 1 for k in ranks.values() if k is not None and k > 0}
    if not desired:
        return {}
    current = z.copy()
    saved = {}
    for k in range(0, max(desired) + 1):
        if k in desired:
            saved[k] = current.copy()
        if k == max(desired):
            break
        pre = predecessor(current, design.a, design.b, design.w_vertices,
                          domain["q_lower"], domain["q_upper"])
        current = hull(intersect(omega, pre))
        if len(current) < 3:
            raise RuntimeError(f"R_{k+1} unexpectedly empty")
    return saved


def main():
    args = parse_args()
    with args.coverage.open(encoding="utf-8") as handle:
        coverage = json.load(handle)
    if coverage["W_membership_violation_count"] != 0 or coverage["random_samples"] < 20000:
        raise RuntimeError("W coverage sample gate did not pass")
    with args.reachability.open(encoding="utf-8") as handle:
        reachable = json.load(handle)
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed")
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    domain, _ = bounds_and_domains(cfg, model, design)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    z = hull(design.rpi_boundary)
    shocks = {
        "pressure_positive": np.array([0.0, 1.0 / cfg.state_scale[1]]),
        "pressure_negative": np.array([0.0, -1.0 / cfg.state_scale[1]]),
        "concentration_positive": np.array([1.0 / cfg.state_scale[0], 0.0]),
    }
    points = []
    for i, xi in enumerate(omega):
        points.append((f"Omega_vertex_{i}", xi))
    for name, xi in shocks.items():
        points.append((f"shock_{name}", xi))
    points.append(("reference", np.zeros(2)))
    rng = np.random.default_rng(args.seed)
    lo, hi = np.min(omega, axis=0), np.max(omega, axis=0)
    for i in range(args.random_states):
        while True:
            xi = rng.uniform(lo, hi)
            if contains(omega, xi):
                break
        points.append((f"random_{i}", xi))
    candidates = [
        np.zeros(2),
        *rectangle(domain["q_lower"], domain["q_upper"]),
    ]
    rows = []
    max_qp_excess = -float("inf")
    max_nonlinear_excess = -float("inf")
    max_nonlinear_w_util = -float("inf")
    failures = 0
    for label, xi in points:
        mode = ("retain_existing_Z_controller" if contains(z, xi)
                else "Omega_one_step_QP" if contains(omega, xi)
                else "unsupported_outside_Omega")
        for candidate in candidates:
            result = project_action(xi, candidate, omega, design, domain, model, cfg)
            if not result["feasible"]:
                failures += 1
                continue
            max_qp_excess = max(max_qp_excess,
                                result["max_W_vertex_next_target_excess_normalized"])
            max_nonlinear_excess = max(max_nonlinear_excess,
                                       result["nonlinear_next_target_excess_normalized"])
            max_nonlinear_w_util = max(max_nonlinear_w_util,
                                       result["nonlinear_W_max_facet_utilization"])
            rows.append({
                "label": label, "mode": mode,
                "xi_X2_normalized": xi[0], "xi_P2_normalized": xi[1],
                "xi_X2_physical": xi[0] * cfg.state_scale[0],
                "xi_P2_physical": xi[1] * cfg.state_scale[1],
                "q_candidate_X2_normalized": candidate[0],
                "q_candidate_P2_normalized": candidate[1],
                "q_safe_X2_normalized": result["q_safe_normalized"][0],
                "q_safe_P2_normalized": result["q_safe_normalized"][1],
                "q_safe_P100_physical": result["q_safe_physical_P100_F200"][0],
                "q_safe_F200_physical": result["q_safe_physical_P100_F200"][1],
                "projection_distance_normalized": result["projection_distance_normalized"],
                "P100": result["actual_input_P100_F200"][0],
                "F200": result["actual_input_P100_F200"][1],
                "W_vertex_next_excess": result["max_W_vertex_next_target_excess_normalized"],
                "nonlinear_W_utilization": result["nonlinear_W_max_facet_utilization"],
                "nonlinear_next_excess": result["nonlinear_next_target_excess_normalized"],
            })

    ranks = {
        name: reachable["shock_states"][name]["first_R_k"] for name in shocks
    }
    targets = rank_targets(omega, z, design, domain, ranks)
    rank_result = {}
    for name, xi in shocks.items():
        k = ranks[name]
        if k is None:
            rank_result[name] = {
                "first_R_k": None, "rank_decreasing_QP_feasible": None,
                "interpretation": "No R_k coverage within prior 1000-step scan",
            }
            continue
        target = targets[k - 1]
        projected = project_action(xi, np.zeros(2), target, design, domain, model, cfg)
        rank_result[name] = {
            "first_R_k": k, "target_rank": k - 1,
            "rank_decreasing_QP_feasible": projected["feasible"],
            "projected_action": projected,
            "inductive_return_bound_steps": k if projected["feasible"] else None,
            "conditional_on": "Every future model mismatch belongs to fixed W",
        }

    summary = {
        "scope": "fixed B coordinates; offline one-step QP; no controller integration or training",
        "W_coverage_tested_samples_passed": True,
        "continuous_domain_W_coverage_proved": False,
        "Omega_QP_probe_states": len(points),
        "Omega_QP_candidate_actions_per_state": len(candidates),
        "Omega_QP_attempts": len(points) * len(candidates),
        "Omega_QP_infeasible_count": failures,
        "maximum_W_vertex_next_Omega_excess_normalized": max_qp_excess,
        "maximum_tested_nonlinear_next_Omega_excess_normalized": max_nonlinear_excess,
        "maximum_tested_nonlinear_W_utilization": max_nonlinear_w_util,
        "mode_logic": {
            "inside_Z": "retain existing Hinf-RPI residual controller (not modified)",
            "outside_Z_inside_Omega": "Omega one-step projection (diagnostic only)",
            "outside_Omega": "no claim or filter action",
        },
        "rank_decreasing": rank_result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "Omega_QP_probes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("saved", path, "QP failures", failures, "rank", ranks, flush=True)


if __name__ == "__main__":
    main()
