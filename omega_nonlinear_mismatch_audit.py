"""Fixed-B nonlinear one-step mismatch audit over the anchor Omega domain.

No moving nominal state/input, controller mutation, SAC or MPC is involved.
The exogenous input stays at the Paper2016 nominal value (zero half-range).
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .controlled_invariant_error_set import (
    area, bounds_and_domains, contains, facets, hull, rectangle,
)
from .model import EvaporatorModel
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b


DEFAULT_OMEGA = REPO_DIR / (
    "evaporation_safe_sac/outputs_controlled_invariant_error_B/"
    "Omega_anchor_B_vertices.csv"
)
TRAJECTORIES = REPO_DIR / (
    "evaporation_safe_sac/outputs_one_step_recovery_B/"
    "one_step_recovery_trajectories.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--random-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=420016)
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_omega_nonlinear_mismatch_B"
    ))
    return parser.parse_args()


def representative_recovery_points(model, omega):
    if not TRAJECTORIES.exists():
        return []
    indices = {0, 1, 2, 5, 10, 20, 40, 80, 160, 299}
    scenarios = {"pressure_positive", "pressure_negative", "concentration_positive"}
    points = []
    with TRAJECTORIES.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] != "zero_residual" or row["scenario"] not in scenarios:
                continue
            if int(row["second"]) not in indices:
                continue
            x = np.array([float(row["x_after_shock_X2"]),
                          float(row["x_after_shock_P2"])])
            xi = model.normalized_state(x)
            if not contains(omega, xi):
                continue
            u = np.array([float(row["P100"]), float(row["F200"])])
            points.append((f"recovery_{row['scenario']}_t{row['second']}", xi,
                           model.normalized_input(u)))
    return points


def main():
    args = parse_args()
    if args.random_samples < 20000:
        raise ValueError("At least 20000 random state/input samples are required")
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed")
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    domain, _ = bounds_and_domains(cfg, model, design)
    u_lower = design.v_ref + domain["q_lower"]
    u_upper = design.v_ref + domain["q_upper"]
    input_vertices = rectangle(u_lower, u_upper)
    w = hull(design.w_vertices)
    h_w, b_w = facets(w)
    if np.any(b_w <= 0):
        raise RuntimeError("W must contain the origin strictly for facet utilization")
    points = []
    for i, xi in enumerate(omega):
        for j, un in enumerate(input_vertices):
            points.append((f"Omega_vertex_{i}_input_corner_{j}", xi, un))
    for name, xi in {
        "pressure_positive": np.array([0.0, 1.0 / cfg.state_scale[1]]),
        "pressure_negative": np.array([0.0, -1.0 / cfg.state_scale[1]]),
        "concentration_positive": np.array([1.0 / cfg.state_scale[0], 0.0]),
    }.items():
        for j, un in enumerate(input_vertices):
            points.append((f"shock_{name}_input_corner_{j}", xi, un))
        points.append((f"shock_{name}_reference_input", xi, design.v_ref))
    for name, xi, un in representative_recovery_points(model, omega):
        points.append((name, xi, un))
        for j, corner in enumerate(input_vertices):
            points.append((f"{name}_input_corner_{j}", xi, corner))
    rng = np.random.default_rng(args.seed)
    lo, hi = np.min(omega, axis=0), np.max(omega, axis=0)
    accepted = 0
    while accepted < args.random_samples:
        xi = rng.uniform(lo, hi)
        if contains(omega, xi):
            un = rng.uniform(u_lower, u_upper)
            points.append((f"random_{accepted}", xi, un))
            accepted += 1

    residuals = np.empty((len(points), 2))
    utilization = np.empty((len(points), len(b_w)))
    worst_index = 0
    rows = []
    for index, (label, xi, un) in enumerate(points):
        x = model.physical_state(xi)
        u = model.physical_input(un)
        actual_next = model.normalized_state(
            model.step(x, u, cfg.disturbance_nominal)
        )
        predicted = design.a @ xi + design.b @ un + design.affine
        mismatch = actual_next - predicted
        residuals[index] = mismatch
        utilization[index] = (h_w @ mismatch) / b_w
        max_util = float(np.max(utilization[index]))
        if max_util > np.max(utilization[worst_index]):
            worst_index = index
        rows.append({
            "label": label, "X2": x[0], "P2": x[1],
            "P100": u[0], "F200": u[1],
            "xi_X2_physical": x[0] - cfg.linearization_state[0],
            "xi_P2_physical": x[1] - cfg.linearization_state[1],
            "q_P100_physical": u[0] - cfg.linearization_input[0],
            "q_F200_physical": u[1] - cfg.linearization_input[1],
            "w_X2_normalized": mismatch[0], "w_P2_normalized": mismatch[1],
            "maximum_W_facet_utilization": max_util,
            "in_W": bool(max_util <= 1.0 + 1e-9),
        })
        if (index + 1) % 5000 == 0:
            print(f"audited {index+1}/{len(points)}", flush=True)

    violations = np.max(utilization, axis=1) > 1.0 + 1e-9
    worst = dict(rows[worst_index])
    worst["facet_utilization_all"] = utilization[worst_index].tolist()
    sample_groups = {}
    for group, prefix in (
        ("Omega_vertex_input_corners", "Omega_vertex_"),
        ("Paper2016_shocks", "shock_"),
        ("representative_recovery", "recovery_"),
        ("random_interior", "random_"),
    ):
        selected = np.array([row["label"].startswith(prefix) for row in rows])
        sample_groups[group] = {
            "count": int(np.sum(selected)),
            "violation_count": int(np.sum(violations[selected])),
            "maximum_W_facet_utilization": (
                float(np.max(utilization[selected])) if np.any(selected) else None
            ),
        }
    # Only propose a replacement disturbance set when the existing W fails.
    # An observed outer hull is not itself a continuum-domain certificate.
    w_omega = (
        hull(1.05 * hull(np.vstack([np.zeros((1, 2)), w, residuals])))
        if np.any(violations) else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "mismatch_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if w_omega is not None:
        with (args.output_dir / "W_omega_observed_vertices.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["w_X2_normalized", "w_P2_normalized"])
            writer.writerows(w_omega.tolist())
    result = {
        "reference_state_physical": model.physical_state(design.z_ref).tolist(),
        "reference_input_physical": model.physical_input(design.v_ref).tolist(),
        "coordinate_note": "xi=x-x_ref and q=u-u_ref, normalized by state/input scale for A,B,W",
        "disturbance": cfg.disturbance_nominal.tolist(),
        "disturbance_half_range": cfg.disturbance_half_range.tolist(),
        "Omega_area_physical": area(omega) * float(np.prod(cfg.state_scale)),
        "admissible_input_lower_physical": model.physical_input(u_lower).tolist(),
        "admissible_input_upper_physical": model.physical_input(u_upper).tolist(),
        "samples": len(points), "random_samples": args.random_samples,
        "sample_groups": sample_groups,
        "W_membership_violation_count": int(np.sum(violations)),
        "W_membership_violation_rate": float(np.mean(violations)),
        "w_actual_min_normalized": np.min(residuals, axis=0).tolist(),
        "w_actual_max_normalized": np.max(residuals, axis=0).tolist(),
        "w_actual_min_physical_state_units": (
            np.min(residuals, axis=0) * cfg.state_scale
        ).tolist(),
        "w_actual_max_physical_state_units": (
            np.max(residuals, axis=0) * cfg.state_scale
        ).tolist(),
        "W_vertices_normalized": w.tolist(),
        "W_facet_normals": h_w.tolist(), "W_facet_bounds": b_w.tolist(),
        "W_facet_max_utilization": np.max(utilization, axis=0).tolist(),
        "W_worst_sample": worst,
        "W_covers_tested_samples": bool(not np.any(violations)),
        "W_covers_entire_continuous_domain": None,
        "W_omega_observed_vertices_normalized": (
            w_omega.tolist() if w_omega is not None else None
        ),
        "W_omega_note": (
            "No replacement constructed: current W covered all tested samples"
            if w_omega is None else
            "1.05 times hull of old W, zero and observed mismatch; sample-covering only"
        ),
    }
    path = args.output_dir / "coverage_summary.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", path, "violations", result["W_membership_violation_count"],
          "of", len(points), "max_util", max(result["W_facet_max_utilization"]), flush=True)


if __name__ == "__main__":
    main()
