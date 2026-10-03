"""Offline fixed-B finite-jump safety sets with a minimum unknown dwell D.

The D-step predecessor uses only regular xi+=A xi+B q+w, w in certified W.
Paper2016 state jumps xi^+=xi^-+delta are a separate set translation. Neither
the set construction nor its admissible control uses a known shock clock.
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
    area, bounds_and_domains, certificate, contains, facets,
    feasible_control_polygon, hull, intersect, iterate_invariant,
    predecessor, set_properties,
)
from .model import EvaporatorModel
from .omega_action_realizability import DEFAULT_OMEGA
from .omega_finite_state_shock_sets import (
    full_dimensional, jump_pre, margin, maximum_facet_excess,
    write_vertices,
)
from .omega_interior_anchor import chebyshev_center_2d
from .omega_online_closed_loop import OmegaSafeOnlineController
from .omega_safe_action_filter_diagnosis import qp_rows
from .paper2016_compare import SCENARIOS
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b
from .train import ZeroResidualPolicy, run_episode


DEFAULT_OUTPUT = REPO_DIR / "evaporation_safe_sac/outputs_omega_dwell_shock_B"
DISPLAY_D = (0, 5, 10, 15, 20, 30)
SHOCK_SECONDS = (0, 20, 40)
NEAR_SINGLETON_CHEBYSHEV_RADIUS = 1e-4


def write_table(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_finite(value):
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_finite(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def backward_reachable(target, dwell, omega, design, domain):
    """B_D(target), restricting every predecessor state to Omega."""
    if not full_dimensional(target):
        return np.empty((0, 2)), [{"k": 0, "area_normalized": 0.0}], False
    current = hull(target)
    history = [{"k": 0, "area_normalized": area(current),
                "vertices": len(current), "area_gain": 0.0}]
    plateau = False
    for k in range(1, dwell + 1):
        pre = predecessor(current, design.a, design.b, design.w_vertices,
                          domain["q_lower"], domain["q_upper"])
        nxt = hull(intersect(omega, pre)) if full_dimensional(pre) else np.empty((0, 2))
        if not full_dimensional(nxt):
            raise RuntimeError(f"D-step predecessor unexpectedly empty at k={k}")
        old_outside_new = maximum_facet_excess(current, nxt)
        if old_outside_new is None or old_outside_new > 2e-7:
            raise RuntimeError(
                f"B_k lost a previously reachable state at k={k}: "
                f"excess={old_outside_new}"
            )
        gain = area(nxt) - area(current)
        history.append({"k": k, "area_normalized": area(nxt),
                        "vertices": len(nxt), "area_gain": gain})
        new_outside_old = maximum_facet_excess(nxt, current)
        current = nxt
        if gain <= 1e-12 and new_outside_old is not None and new_outside_old <= 1e-9:
            plateau = True
            break
    return current, history, plateau


def closed_set_vertex_certificate(poly, domain, design, tol=2e-7):
    """Certify existence of a q at each vertex, including singleton q sets.

    Unlike the older Omega diagnostic, finite-jump RCI need not have a
    full-dimensional q polytope at every boundary vertex. A unique feasible
    control still proves the one-step vertex condition for a convex set.
    """
    normals, bounds = facets(poly)
    if len(normals) == 0:
        return {"passed": False, "reason": "empty_set"}
    worst_next = -float("inf")
    worst_input = -float("inf")
    worst_state = -float("inf")
    dimension_counts = {"point": 0, "line": 0, "area": 0}
    for vertex in poly:
        q_poly = hull(feasible_control_polygon(
            vertex, poly, design.a, design.b, design.w_vertices,
            domain["q_lower"], domain["q_upper"],
        ))
        if len(q_poly) == 0:
            H, h = qp_rows(vertex, poly, design, domain)
            projected, feasible = project_qp_2d(
                0.5 * (domain["q_lower"] + domain["q_upper"]), H, h
            )
            if feasible and float(np.max(H @ projected - h)) <= tol:
                q_poly = np.asarray([projected])
            else:
                return {"passed": False, "reason": "empty_q_at_vertex",
                        "failed_vertex": vertex.tolist(),
                        "QP_feasible": feasible,
                        "QP_candidate": projected.tolist(),
                        "QP_max_excess": float(np.max(H @ projected - h))}
        dimension = "area" if area(q_poly) > 1e-12 else (
            "line" if len(q_poly) > 1 else "point"
        )
        dimension_counts[dimension] += 1
        q = np.mean(q_poly, axis=0)
        H, h = qp_rows(vertex, poly, design, domain)
        worst_input = max(worst_input, float(np.max(np.concatenate((
            domain["q_lower"] - q, q - domain["q_upper"]
        )))))
        worst_next = max(worst_next, float(np.max(
            (design.a @ vertex + design.b @ q + design.w_vertices)
            @ normals.T - bounds
        )))
        worst_state = max(worst_state, float(np.max(np.concatenate((
            domain["e_lower"] - vertex, vertex - domain["e_upper"]
        )))))
        if float(np.max(H @ q - h)) > tol:
            return {"passed": False, "reason": "q_row_excess",
                    "failed_vertex": vertex.tolist(),
                    "q_row_excess": float(np.max(H @ q - h))}
    return {
        "passed": bool(max(worst_next, worst_input, worst_state) <= tol),
        "reason": "convex_vertex_feedback_certificate",
        "tested_vertices": len(poly),
        "q_dimension_counts": dimension_counts,
        "max_next_facet_excess_normalized": worst_next,
        "max_input_excess_normalized": worst_input,
        "max_state_excess_normalized": worst_state,
    }


def build_one_dwell(dwell, delta, omega, design, domain, max_iterations):
    sets = [omega]
    ready = [None]
    backward = [None]
    levels = []
    for m in (1, 2, 3):
        reach, reach_history, reach_plateau = backward_reachable(
            sets[m - 1], dwell, omega, design, domain
        )
        backward.append(reach)
        jump_ready = (hull(intersect(omega, jump_pre(reach, delta)))
                      if full_dimensional(reach) else np.empty((0, 2)))
        ready.append(jump_ready)
        if full_dimensional(jump_ready):
            invariant, history, converged = iterate_invariant(
                jump_ready, design.a, design.b, design.w_vertices,
                domain["q_lower"], domain["q_upper"], max_iterations,
            )
            invariant = (hull(invariant) if full_dimensional(invariant)
                         else np.empty((0, 2)))
        else:
            invariant, history, converged = np.empty((0, 2)), [], False
        cert = (closed_set_vertex_certificate(invariant, domain, design)
                if full_dimensional(invariant) else
                {"passed": False, "reason": "empty_or_lower_dimensional"})
        jump_excess = maximum_facet_excess(invariant + delta, reach)
        subset_excess = maximum_facet_excess(invariant, jump_ready)
        passed = bool(
            full_dimensional(invariant) and converged and cert["passed"]
            and jump_excess is not None and jump_excess <= 2e-7
            and subset_excess is not None and subset_excess <= 2e-7
        )
        # A nonconverged finite iterate is not an RCI certificate. Keep its
        # diagnostics, but never feed it into the next jump level as G_m.
        candidate_area = area(invariant)
        if not passed:
            invariant = np.empty((0, 2))
        sets.append(invariant)
        levels.append({
            "m": m, "G": invariant, "JumpReady": jump_ready,
            "B_D_previous": reach,
            "B_D_area_history": reach_history,
            "B_D_early_plateau": reach_plateau,
            "RCI_iteration_count": len(history),
            "RCI_numerically_converged": converged,
            "uncertified_final_iterate_area_normalized": (
                candidate_area if not passed else None
            ),
            "RCI_last_iteration": history[-1] if history else None,
            "regular_vertex_certificate": cert,
            "jumped_G_max_facet_excess_over_B_D_previous": jump_excess,
            "G_max_facet_excess_over_JumpReady": subset_excess,
            "fully_certified_at_numerical_tolerance": passed,
        })
    return {"G": sets, "JumpReady": ready, "B_D_previous": backward,
            "levels": levels}


def point_samples(poly, rng, grid_size=7, random_count=80):
    if not full_dimensional(poly):
        return []
    points = [(f"vertex_{i}", v) for i, v in enumerate(poly)]
    for i, start in enumerate(poly):
        end = poly[(i + 1) % len(poly)]
        for fraction in (0.25, 0.5, 0.75):
            points.append((f"edge_{i}_{fraction}",
                           (1.0 - fraction) * start + fraction * end))
    low, high = np.min(poly, axis=0), np.max(poly, axis=0)
    normals, bounds = facets(poly)
    for ix, x in enumerate(np.linspace(low[0], high[0], grid_size)):
        for iy, y in enumerate(np.linspace(low[1], high[1], grid_size)):
            candidate = np.array([x, y])
            if np.min(bounds - normals @ candidate) > 1e-7:
                points.append((f"grid_{ix}_{iy}", candidate))
    for i in range(random_count):
        # Convex combinations sample strictly inside the existing polygon.
        weights = rng.dirichlet(np.ones(len(poly)))
        points.append((f"random_{i}", weights @ poly))
    return points


def authority_audit(poly, design, domain, model, cfg, rng,
                    scenario, dwell, m):
    rows = []
    for sample_name, xi in point_samples(poly, rng):
        H, h = qp_rows(xi, poly, design, domain)
        q_poly = hull(feasible_control_polygon(
            xi, poly, design.a, design.b, design.w_vertices,
            domain["q_lower"], domain["q_upper"],
        ))
        if len(q_poly) == 0:
            projected, feasible = project_qp_2d(
                0.5 * (domain["q_lower"] + domain["q_upper"]), H, h
            )
            if feasible and np.max(H @ projected - h) <= 2e-7:
                q_poly = np.asarray([projected])
            else:
                raise RuntimeError(
                    f"Admissible q empty: {scenario}, D={dwell}, G{m}, "
                    f"{sample_name}, xi={xi.tolist()}"
                )
        if area(q_poly) <= 1e-12:
            center, radius = np.mean(q_poly, axis=0), 0.0
        else:
            try:
                center, radius = chebyshev_center_2d(H, h)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Chebyshev LP failed: {scenario}, D={dwell}, G{m}, "
                    f"{sample_name}, xi={xi.tolist()}, q_vertices={q_poly.tolist()}, "
                    f"max_q_excess={float(np.max(q_poly @ H.T - h))}"
                ) from exc
        projected, feasible = project_qp_2d(center, H, h)
        if not feasible or np.max(H @ projected - h) > 2e-7 or len(q_poly) == 0:
            raise RuntimeError(
                f"Admissible q empty: {scenario}, D={dwell}, G{m}, {sample_name}"
            )
        physical = model.physical_input(design.v_ref + q_poly)
        widths = np.ptp(physical, axis=0)
        rows.append({
            "scenario": scenario, "D": dwell, "set": f"G{m}",
            "sample": sample_name,
            "xi_X2_normalized": float(xi[0]),
            "xi_P2_normalized": float(xi[1]),
            "q_area_normalized": area(q_poly),
            "q_chebyshev_radius_normalized": radius,
            "P100_min": float(np.min(physical[:, 0])),
            "P100_max": float(np.max(physical[:, 0])),
            "F200_min": float(np.min(physical[:, 1])),
            "F200_max": float(np.max(physical[:, 1])),
            "P100_width": float(widths[0]),
            "F200_width": float(widths[1]),
            "near_singleton_q": bool(radius < NEAR_SINGLETON_CHEBYSHEV_RADIUS),
        })
    summary = {"sample_count": len(rows),
               "near_singleton_q_fraction": float(np.mean([
                   row["near_singleton_q"] for row in rows
               ])) if rows else None,
               "near_singleton_threshold_chebyshev_radius_normalized":
                   NEAR_SINGLETON_CHEBYSHEV_RADIUS}
    for subset, selected in (
        ("boundary", [row for row in rows if row["sample"].startswith((
            "vertex_", "edge_"))]),
        ("interior", [row for row in rows if row["sample"].startswith((
            "grid_", "random_"))]),
    ):
        summary[f"{subset}_sample_count"] = len(selected)
        summary[f"{subset}_near_singleton_fraction"] = (
            float(np.mean([row["near_singleton_q"] for row in selected]))
            if selected else None
        )
    for key in ("q_area_normalized", "q_chebyshev_radius_normalized",
                "P100_min", "P100_max", "F200_min", "F200_max",
                "P100_width", "F200_width"):
        values = [row[key] for row in rows]
        summary[key + "_min_median_max"] = (
            [float(np.min(values)), float(np.median(values)), float(np.max(values))]
            if values else None
        )
    return summary, rows


def baseline_records(cfg, model, design, omega, domain, scenario):
    controller = OmegaSafeOnlineController(cfg, model, design, omega, domain)
    _, records, _ = run_episode(
        cfg, model, controller, ZeroResidualPolicy(), None,
        np.random.default_rng(cfg.seed + 31000 + SCENARIOS.index(scenario)),
        training=False, global_step=0, paper_scenario=scenario,
    )
    return records


def protocol_validation(records, result, delta, design, model, cfg):
    validation = []
    sets, ready, reach = result["G"], result["JumpReady"], result["B_D_previous"]
    for index, second in enumerate(SHOCK_SECONDS):
        m = 3 - index
        post_state = np.asarray(records[second]["state"])
        pre_state = post_state - delta * cfg.state_scale
        pre = model.normalized_state(pre_state) - design.z_ref
        post = model.normalized_state(post_state) - design.z_ref
        next_shock = SHOCK_SECONDS[index + 1] if index + 1 < 3 else len(records)
        target = sets[m - 1]
        entered = next((k for k in range(second + 1, next_shock)
                        if contains(target, model.normalized_state(
                            records[k]["state"]
                        ) - design.z_ref)), None)
        validation.append({
            "shock_second": second, "shocks_remaining_before": m,
            "pre_state_physical": pre_state.tolist(),
            "post_state_physical": post_state.tolist(),
            "pre_in_G_m": contains(sets[m], pre),
            "pre_margin_to_G_m_physical": margin(sets[m], pre, cfg),
            "pre_in_JumpReady_m": contains(ready[m], pre),
            "post_in_B_D_G_m_minus_1": contains(reach[m], post),
            "post_margin_to_B_D_physical": margin(reach[m], post, cfg),
            "first_subsequent_step_in_G_m_minus_1": entered,
            "entered_G_m_minus_1_before_next_shock": entered is not None,
        })
    return validation


def plot_area(path, sensitivity):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    colors = {"concentration_positive": "#1a8174",
              "pressure_positive": "#c97829",
              "pressure_negative": "#5767a2"}
    for scenario in SCENARIOS:
        rows = [row for row in sensitivity if row["scenario"] == scenario]
        ax.plot([row["D"] for row in rows],
                [row["G3_area_over_Omega"] for row in rows],
                "o-", color=colors[scenario], label=scenario)
    ax.set(xlabel="Minimum normal steps between state jumps, D",
           ylabel="area(G3) / area(Omega)",
           title="No-clock finite-jump robust-safe area")
    ax.set_xticks(DISPLAY_D)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--scan-max-D", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0 <= args.scan_max_D <= 100:
        raise ValueError("scan-max-D must be 0..100")
    if not 1 <= args.max_iterations <= 1000:
        raise ValueError("max-iterations must be 1..1000")
    cfg = ExperimentConfig(
        benchmark_profile="zanon2016", experiment_mode="proposed",
        seed=args.seed, steps_per_episode=300,
        residual_parameterization="state_dependent_polytope",
    )
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    domain, _ = bounds_and_domains(cfg, model, design)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    omega_cert = certificate(omega, domain, design)
    if not omega_cert["passed"]:
        raise RuntimeError("Loaded Omega failed its regular-W vertex certificate")
    delta_by_scenario = {
        "pressure_positive": np.array([0.0, 1.0 / cfg.state_scale[1]]),
        "pressure_negative": np.array([0.0, -1.0 / cfg.state_scale[1]]),
        "concentration_positive": np.array([1.0 / cfg.state_scale[0], 0.0]),
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    omega_area = area(omega)
    rng = np.random.default_rng(args.seed + 74001)
    baselines = {
        scenario: baseline_records(cfg, model, design, omega, domain, scenario)
        for scenario in SCENARIOS
    }
    sensitivity, all_samples, detailed, minima = [], [], {}, {}
    for scenario in SCENARIOS:
        delta = delta_by_scenario[scenario]
        first_nonempty = first_reference = None
        inconclusive_d = []
        for dwell in range(args.scan_max_D + 1):
            result = build_one_dwell(
                dwell, delta, omega, design, domain, args.max_iterations
            )
            if any(level["uncertified_final_iterate_area_normalized"] is not None
                   and level["uncertified_final_iterate_area_normalized"] > 0
                   for level in result["levels"]):
                inconclusive_d.append(dwell)
            g3 = result["G"][3]
            if full_dimensional(g3) and first_nonempty is None:
                first_nonempty = dwell
            if contains(g3, np.zeros(2)) and first_reference is None:
                first_reference = dwell
            if dwell not in DISPLAY_D:
                continue
            row = {"scenario": scenario, "D": dwell}
            level_details = {}
            for level in result["levels"]:
                m = level["m"]
                poly = level["G"]
                row[f"G{m}_nonempty"] = full_dimensional(poly)
                row[f"G{m}_area_physical"] = area(poly) * float(np.prod(cfg.state_scale))
                row[f"G{m}_area_over_Omega"] = area(poly) / omega_area
                row[f"G{m}_reference_in_set"] = contains(poly, np.zeros(2))
                if m == 3:
                    row["G3_reference_margin_physical"] = margin(
                        poly, np.zeros(2), cfg
                    )
                write_vertices(output / f"{scenario}_D{dwell}_G{m}_vertices.csv",
                               poly, cfg)
                write_vertices(output / f"{scenario}_D{dwell}_JumpReady{m}_vertices.csv",
                               level["JumpReady"], cfg)
                write_vertices(output / f"{scenario}_D{dwell}_B{dwell}_G{m-1}_vertices.csv",
                               level["B_D_previous"], cfg)
                authority = None
                if full_dimensional(poly):
                    authority, samples = authority_audit(
                        poly, design, domain, model, cfg, rng,
                        scenario, dwell, m
                    )
                    all_samples.extend(samples)
                level_details[f"G{m}"] = {
                    **set_properties(poly, cfg.state_scale),
                    "reference_in_set": contains(poly, np.zeros(2)),
                    "reference_margin_physical": margin(poly, np.zeros(2), cfg),
                    "JumpReady_area_physical": area(level["JumpReady"]) * float(
                        np.prod(cfg.state_scale)
                    ),
                    "B_D_previous_area_physical": area(level["B_D_previous"]) * float(
                        np.prod(cfg.state_scale)
                    ),
                    "B_D_area_history": level["B_D_area_history"],
                    "B_D_early_plateau": level["B_D_early_plateau"],
                    "RCI_iteration_count": level["RCI_iteration_count"],
                    "RCI_numerically_converged": level["RCI_numerically_converged"],
                    "RCI_last_iteration": level["RCI_last_iteration"],
                    "uncertified_final_iterate_area_normalized": level[
                        "uncertified_final_iterate_area_normalized"
                    ],
                    "regular_vertex_certificate": level["regular_vertex_certificate"],
                    "jumped_G_max_facet_excess_over_B_D_previous": level[
                        "jumped_G_max_facet_excess_over_B_D_previous"
                    ],
                    "fully_certified_at_numerical_tolerance": level[
                        "fully_certified_at_numerical_tolerance"
                    ],
                    "control_authority": authority,
                }
            sensitivity.append(row)
            level_details["Paper2016_zero_residual_validation"] = protocol_validation(
                baselines[scenario], result, delta, design, model, cfg
            )
            detailed[f"{scenario}_D{dwell}"] = level_details
            print(scenario, "D", dwell, "G3_area_ratio",
                  round(row["G3_area_over_Omega"], 6),
                  "reference_in_G3", row["G3_reference_in_set"], flush=True)
        minima[scenario] = {
            "first_integer_D_G3_certified_nonempty_in_0_to_scan_max": first_nonempty,
            "first_integer_D_reference_certified_in_G3_in_0_to_scan_max": first_reference,
            "inconclusive_D_due_to_nonconverged_or_failed_RCI_gate": inconclusive_d,
            "scan_max_D": args.scan_max_D,
            "not_found_interpretation": (
                "Minima are for numerically certified sets under the iteration "
                "cap. Earlier inconclusive D prevent claiming an absolute "
                "minimum; no result outside the finite scan is implied."
            ),
        }
    write_table(output / "dwell_sensitivity.csv", sensitivity)
    write_table(output / "control_authority_samples.csv", all_samples)
    plot_area(output / "G3_area_ratio_vs_D.png", sensitivity)
    summary = {
        "scope": "offline fixed-B linearized polytope analysis; no SAC training",
        "regular_model_uncertainty": "xi_next=A xi+B q+w, w in unchanged certified W",
        "instantaneous_state_jump": "xi_plus=xi_minus+delta, separate from W",
        "shock_clock_used_in_set_construction": False,
        "all_intermediate_backward_states_constrained_to_Omega": True,
        "actual_input_constraints": "unchanged physical/robust input intersection",
        "Omega": set_properties(omega, cfg.state_scale),
        "Omega_regular_vertex_certificate": omega_cert,
        "reference_state_physical": model.physical_state(design.z_ref).tolist(),
        "display_D": list(DISPLAY_D),
        "integer_D_minimum_scan": minima,
        "near_singleton_q_threshold": (
            "Chebyshev radius < 1e-4 in normalized q coordinates; "
            "diagnostic only, not a safety gate"
        ),
        "set_details": detailed,
        "interpretation_caveat": (
            "Numerical set computation for the fixed affine model and W; "
            "does not establish analytic nonlinear continuous-domain coverage. "
            "Paper2016 0/20/40 trajectories are post-hoc validation only."
        ),
    }
    path = output / "summary.json"
    path.write_text(json.dumps(json_finite(summary), indent=2, allow_nan=False),
                    encoding="utf-8")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
