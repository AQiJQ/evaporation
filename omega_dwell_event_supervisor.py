"""Independent event-driven G_m/B_j one-step supervisor and nonlinear stress test.

The supervisor sees the measured state, not a shock schedule. It detects an
instantaneous jump only after it appears in the innovation relative to the
previous affine prediction and certified regular W. No SAC training or online
multi-step optimization is performed.
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
    area, bounds_and_domains, contains, facets, feasible_control_polygon, hull,
    intersect, predecessor,
)
from .model import EvaporatorModel
from .omega_dwell_shock_sets import DEFAULT_OUTPUT as DWELL_OUTPUT, json_finite
from .omega_finite_state_shock_sets import maximum_facet_excess
from .omega_interior_anchor import InteriorAnchorController, chebyshev_center_2d
from .omega_online_closed_loop import policy_action
from .omega_safe_action_filter_diagnosis import qp_rows
from .paper2016_compare import SCENARIOS
from .residual_action_space_diagnosis import (
    DEFAULT_DESIGN, DEFAULT_STRESS_ACTOR, REPO_DIR, FrozenActor, load_fixed_b,
)
from .train import observation


D = 20
LOW_RADIUS = 1e-4
POLICIES = ("zero_residual", "frozen_old_actor", "random_bounded_nonzero")
DEFAULT_OUTPUT = REPO_DIR / "evaporation_safe_sac/outputs_omega_dwell_event_supervisor_B"


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_poly(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    if len(rows) < 3:
        raise ValueError(f"Expected full-dimensional stored set: {path}")
    if not header[0].endswith("X2_normalized") or not header[1].endswith("P2_normalized"):
        raise ValueError(f"Unexpected normalized set columns: {path}")
    return hull(np.asarray([[float(row[0]), float(row[1])] for row in rows]))


def validate_source_summary(source: Path, omega):
    saved = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if saved["shock_clock_used_in_set_construction"]:
        raise RuntimeError("Dwell source used a shock clock")
    stored_omega = hull(np.asarray(saved["Omega"]["vertices_normalized"]))
    if (maximum_facet_excess(omega, stored_omega) > 2e-7 or
            maximum_facet_excess(stored_omega, omega) > 2e-7):
        raise RuntimeError("Loaded Omega differs from certified dwell source")
    for scenario in SCENARIOS:
        detail = saved["set_details"][f"{scenario}_D20"]
        for m in (1, 2, 3):
            level = detail[f"G{m}"]
            if (not level["fully_certified_at_numerical_tolerance"] or
                    not level["regular_vertex_certificate"]["passed"]):
                raise RuntimeError(f"Stored G{m} lacks certification: {scenario}")
            poly = load_poly(source / f"{scenario}_D20_G{m}_vertices.csv")
            if abs(area(poly) - level["area_normalized"]) > 2e-7:
                raise RuntimeError(f"Stored G{m} area changed: {scenario}")
    return saved


def make_rank_sets(scenario, omega, design, domain, source: Path):
    """Recreate B_0..B_20 from stored certified G sets; check stored B_20."""
    sets = [omega]
    for m in (1, 2, 3):
        sets.append(load_poly(source / f"{scenario}_D20_G{m}_vertices.csv"))
    ranks = {}
    for m in (0, 1, 2, 3):
        family = [sets[m]]
        for _ in range(D):
            pre = predecessor(family[-1], design.a, design.b, design.w_vertices,
                              domain["q_lower"], domain["q_upper"])
            next_set = hull(intersect(omega, pre))
            if len(next_set) < 3:
                raise RuntimeError(f"B_j unexpectedly empty: {scenario}, m={m}")
            if maximum_facet_excess(family[-1], next_set) > 2e-7:
                raise RuntimeError(f"B_j nesting failed: {scenario}, m={m}")
            family.append(next_set)
        ranks[m] = family
        if m < 3:
            stored = load_poly(source / f"{scenario}_D20_B20_G{m}_vertices.csv")
            if (maximum_facet_excess(family[-1], stored) > 2e-7 or
                    maximum_facet_excess(stored, family[-1]) > 2e-7):
                raise RuntimeError(f"Recomputed B20 differs from stored: {scenario}, G{m}")
    return sets, ranks


def minimum_rank(family, xi):
    for j, poly in enumerate(family):
        if contains(poly, xi, tol=1e-8):
            return j
    return None


def feasible_q_audit(xi, target, design, domain):
    rows, bounds = qp_rows(xi, target, design, domain)
    q_poly = hull(feasible_control_polygon(
        xi, target, design.a, design.b, design.w_vertices,
        domain["q_lower"], domain["q_upper"],
    ))
    if len(q_poly) == 0:
        q0 = 0.5 * (domain["q_lower"] + domain["q_upper"])
        q_one, feasible = project_qp_2d(q0, rows, bounds)
        if not feasible or np.max(rows @ q_one - bounds) > 2e-7:
            return None
        q_poly = np.asarray([q_one])
    if area(q_poly) <= 1e-12:
        center, radius = np.mean(q_poly, axis=0), 0.0
    else:
        center, radius = chebyshev_center_2d(rows, bounds)
    if np.max(rows @ center - bounds) > 2e-7:
        return None
    return {
        "rows": rows, "bounds": bounds, "poly": q_poly,
        "center": center, "radius": float(radius), "area": float(area(q_poly)),
        "q_min": np.min(q_poly, axis=0), "q_max": np.max(q_poly, axis=0),
    }


class EventDrivenDwellSupervisor:
    def __init__(self, cfg, model, design, omega, domain, scenario, source,
                 *, precomputed=None, shock_scale_lower=1.0):
        self.cfg, self.model, self.design = cfg, model, design
        self.omega, self.domain, self.scenario = omega, domain, scenario
        self.sets, self.ranks = (
            make_rank_sets(scenario, omega, design, domain, source)
            if precomputed is None else precomputed
        )
        if not 0.0 < shock_scale_lower <= 1.0:
            raise ValueError("shock_scale_lower must be in (0,1]")
        self.shock_scale_lower = float(shock_scale_lower)
        if not contains(self.sets[3], np.zeros(2)):
            raise RuntimeError(f"Reference not in certified G3: {scenario}")
        self.delta = {
            "pressure_positive": np.array([0.0, 1.0 / cfg.state_scale[1]]),
            "pressure_negative": np.array([0.0, -1.0 / cfg.state_scale[1]]),
            "concentration_positive": np.array([1.0 / cfg.state_scale[0], 0.0]),
        }[scenario]
        self.w = hull(design.w_vertices)
        self.remaining_shocks = 3
        self.predicted_xi = np.zeros(2)
        self.recovery_steps = 0
        self.events = []
        self.failed = False
        self.failure_reason = None

    def fail(self, reason):
        self.failed = True
        self.failure_reason = reason
        return {"certificate_failure": True, "failure_reason": reason}

    def observe(self, state, second):
        """Classify the *observed* innovation, then choose current rank target."""
        if self.failed:
            return self.fail(self.failure_reason)
        xi = self.model.normalized_state(state) - self.design.z_ref
        innovation = xi - self.predicted_xi
        normal = contains(self.w, innovation, tol=2e-7)
        inferred_scale = float(np.clip(
            innovation @ self.delta / (self.delta @ self.delta),
            self.shock_scale_lower, 1.0,
        ))
        jump = contains(self.w, innovation - inferred_scale * self.delta,
                        tol=2e-7)
        if normal == jump:
            return self.fail(
                "ambiguous_or_uncovered_measurement_innovation"
                if normal else "measurement_innovation_outside_W_and_W_plus_delta"
            )
        event = bool(jump)
        if event:
            if self.remaining_shocks <= 0:
                return self.fail("unexpected_extra_state_jump")
            self.remaining_shocks -= 1
            self.recovery_steps = 0
            self.events.append({
                "detected_second": second,
                "remaining_shocks_after": self.remaining_shocks,
                "initial_rank": None, "recovery_steps": None,
                "completed_within_D": None,
            })
            if not contains(self.ranks[self.remaining_shocks][D], xi):
                return self.fail("post_jump_outside_B20")
        if not contains(self.omega, xi):
            return self.fail("omega_exit")
        m = self.remaining_shocks
        rank = minimum_rank(self.ranks[m], xi)
        if rank is None:
            return self.fail("state_outside_B20")
        if event:
            self.events[-1]["initial_rank"] = rank
            if rank == 0:
                self.events[-1].update(recovery_steps=0, completed_within_D=True)
        if rank == 0:
            mode, target, target_rank = "wait_Gm", self.sets[m], 0
        else:
            mode, target_rank = "recover_Bj", rank - 1
            target = self.ranks[m][target_rank]
        return {
            "certificate_failure": False, "event_detected": event,
            "inferred_jump_scale": inferred_scale if event else None,
            "innovation": innovation, "xi": xi, "remaining_shocks": m,
            "rank": rank, "mode": mode, "target": target,
            "target_rank": target_rank, "in_Gm": rank == 0,
        }

    def filter_candidate(self, state, candidate_input, context):
        q_candidate = self.model.normalized_input(candidate_input) - self.design.v_ref
        audit = feasible_q_audit(context["xi"], context["target"],
                                 self.design, self.domain)
        if audit is None:
            return self.fail("one_step_QP_infeasible")
        scale = self.cfg.input_scale
        q_physical, feasible = project_qp_2d(
            q_candidate * scale, audit["rows"] / scale[np.newaxis, :],
            audit["bounds"],
        )
        q_safe = q_physical / scale
        if not feasible or np.max(audit["rows"] @ q_safe - audit["bounds"]) > 2e-7:
            return self.fail("one_step_QP_infeasible")
        control = self.model.physical_input(self.design.v_ref + q_safe)
        return {
            "certificate_failure": False, "q_safe": q_safe,
            "control": control, "candidate_input": candidate_input,
            "projection_distance_physical": float(np.linalg.norm(
                control - candidate_input)),
            "q_area_normalized": audit["area"],
            "q_chebyshev_radius_normalized": audit["radius"],
            "low_authority": bool(audit["radius"] < LOW_RADIUS),
            "q_min_normalized": audit["q_min"],
            "q_max_normalized": audit["q_max"],
            "q_center_normalized": audit["center"],
        }

    def complete_step(self, xi_next, context):
        m, rank = context["remaining_shocks"], context["rank"]
        next_rank = minimum_rank(self.ranks[m], xi_next)
        if next_rank is None:
            return self.fail("normal_step_outside_B20")
        if rank == 0 and next_rank != 0:
            return self.fail("normal_step_Gm_exit")
        if rank > 0 and next_rank > rank - 1:
            return self.fail("recovery_rank_did_not_decrease")
        if rank > 0:
            self.recovery_steps += 1
            if next_rank == 0:
                self.events[-1].update(
                    recovery_steps=self.recovery_steps,
                    completed_within_D=self.recovery_steps <= D,
                )
        return {"certificate_failure": False, "next_rank": next_rank}


def random_schedule(rng, seconds):
    if seconds < 95:
        raise ValueError("Need at least 95 steps for three random dwell-20 jumps")
    first = int(rng.integers(0, 16))
    gaps = rng.integers(20, 36, size=2)
    schedule = [first, first + int(gaps[0]), first + int(gaps[0] + gaps[1])]
    if schedule[-1] >= seconds:
        raise RuntimeError("Generated shock beyond trajectory")
    return schedule


def boundary_action_probes(cfg, model, design, omega, domain, source):
    """Test extreme candidate q on G3 vertices/edges, notably pressure+."""
    rows = []
    for scenario in SCENARIOS:
        g3 = load_poly(source / f"{scenario}_D20_G3_vertices.csv")
        points = [(f"vertex_{i}", vertex) for i, vertex in enumerate(g3)]
        points += [
            (f"edge_mid_{i}", 0.5 * (vertex + g3[(i + 1) % len(g3)]))
            for i, vertex in enumerate(g3)
        ]
        candidate_q = [np.zeros(2)] + [
            np.array([x, y]) for x in (domain["q_lower"][0], domain["q_upper"][0])
            for y in (domain["q_lower"][1], domain["q_upper"][1])
        ]
        for name, xi in points:
            audit = feasible_q_audit(xi, g3, design, domain)
            if audit is None:
                rows.append({
                    "scenario": scenario, "point": name,
                    "candidate": "unavailable", "QP_feasible": False,
                    "low_authority": None, "q_radius": None,
                    "projection_distance_physical": None,
                    "linear_robust_next_in_G3": None,
                    "nonlinear_mismatch_in_W": None,
                    "nonlinear_next_in_G3": None,
                })
                continue
            normals, bounds = facets(g3)
            w_normals, w_bounds = facets(hull(design.w_vertices))
            for index, q in enumerate(candidate_q):
                physical_q, feasible = project_qp_2d(
                    q * cfg.input_scale,
                    audit["rows"] / cfg.input_scale[np.newaxis, :],
                    audit["bounds"],
                )
                safe_q = physical_q / cfg.input_scale
                if feasible:
                    next_vertices = (design.a @ xi + design.b @ safe_q
                                     + design.w_vertices)
                    robust_next = bool(np.max(next_vertices @ normals.T - bounds)
                                       <= 2e-7)
                    state = model.physical_state(design.z_ref + xi)
                    control = model.physical_input(design.v_ref + safe_q)
                    nonlinear_next = (model.normalized_state(model.step(
                        state, control, cfg.disturbance_nominal)) - design.z_ref)
                    mismatch = nonlinear_next - (design.a @ xi + design.b @ safe_q)
                    in_w = bool(np.max(mismatch @ w_normals.T - w_bounds) <= 2e-7)
                    in_g3 = contains(g3, nonlinear_next, tol=2e-7)
                else:
                    robust_next = in_w = in_g3 = None
                rows.append({
                    "scenario": scenario, "point": name,
                    "candidate": "zero" if index == 0 else f"input_corner_{index}",
                    "QP_feasible": bool(feasible),
                    "low_authority": audit["radius"] < LOW_RADIUS,
                    "q_radius": audit["radius"],
                    "projection_distance_physical": float(np.linalg.norm(
                        (safe_q - q) * cfg.input_scale)) if feasible else None,
                    "linear_robust_next_in_G3": robust_next,
                    "nonlinear_mismatch_in_W": in_w,
                    "nonlinear_next_in_G3": in_g3,
                })
    return rows


def stress_rollout(cfg, model, design, omega, domain, scenario, policy,
                   actor, source, seed, seconds, trajectory,
                   observation_variant="base19"):
    supervisor = EventDrivenDwellSupervisor(
        cfg, model, design, omega, domain, scenario, source
    )
    candidate_controller = InteriorAnchorController(cfg, model, design, omega, domain)
    state = model.physical_state(design.z_ref)
    candidate_controller.reset(state)
    previous_u = design.v_ref.copy()
    w_est = np.zeros(2)
    rng = np.random.default_rng(seed)
    schedule = random_schedule(rng, seconds)
    rows = []
    failure = None
    for second in range(seconds):
        # The plant-side event generator alone knows the schedule. The
        # supervisor receives only the resulting measured state.
        if second in schedule:
            state = state + supervisor.delta * cfg.state_scale
        context = supervisor.observe(state, second)
        if context["certificate_failure"]:
            failure = context["failure_reason"]
            break
        obs = observation(model, candidate_controller, state, previous_u, w_est)
        if observation_variant == "supervisor23":
            from .omega_supervisor_observation import supervisor_observation_extra
            obs = np.concatenate((obs, supervisor_observation_extra(
                context, design, domain
            )))
        action = policy_action(policy, obs, actor, rng)
        try:
            candidate_input, candidate_info = candidate_controller.act(state, action)
        except (RuntimeError, ValueError) as exc:
            failure = f"candidate_controller_error:{type(exc).__name__}:{exc}"
            break
        filtered = supervisor.filter_candidate(state, candidate_input, context)
        if filtered["certificate_failure"]:
            failure = filtered["failure_reason"]
            break
        control = filtered["control"]
        q_safe = filtered["q_safe"]
        feasible_u_min = model.physical_input(
            design.v_ref + filtered["q_min_normalized"]
        )
        feasible_u_max = model.physical_input(
            design.v_ref + filtered["q_max_normalized"]
        )
        # In Z mode, the nominal state must follow the final nominal input,
        # not the candidate nominal input discarded by this supervisor.
        if candidate_info["mode"] == "Z_mode_existing_controller":
            ancillary = np.asarray(candidate_info["ancillary"])
            final_nominal = model.normalized_input(control) - ancillary
            candidate_controller.inner.z = (
                design.a @ np.asarray(candidate_info["z_before"])
                + design.b @ final_nominal + design.affine
            )
        next_state = model.step(state, control, cfg.disturbance_nominal)
        xi_next = model.normalized_state(next_state) - design.z_ref
        linear_next = design.a @ context["xi"] + design.b @ q_safe
        mismatch = xi_next - linear_next
        w_ok = contains(supervisor.w, mismatch, tol=2e-7)
        supervisor.predicted_xi = linear_next
        complete = supervisor.complete_step(xi_next, context)
        u_norm = model.normalized_input(control)
        x_norm = model.normalized_state(state)
        next_norm = model.normalized_state(next_state)
        rows.append({
            "scenario": scenario, "policy": policy, "trajectory": trajectory,
            "second": second, "shock_injected": second in schedule,
            "shock_detected": context["event_detected"],
            "remaining_shocks": context["remaining_shocks"],
            "mode": context["mode"], "rank": context["rank"],
            "target_rank": context["target_rank"],
            "in_Gm_before": context["in_Gm"],
            "in_Gm_after": contains(supervisor.sets[context["remaining_shocks"]], xi_next),
            "in_Omega_after": contains(omega, xi_next),
            "X2": state[0], "P2": state[1],
            "X2_next": next_state[0], "P2_next": next_state[1],
            "P100": control[0], "F200": control[1],
            "candidate_P100": candidate_input[0],
            "candidate_F200": candidate_input[1],
            "actor_a_P100": action[0], "actor_a_F200": action[1],
            "q_area_normalized": filtered["q_area_normalized"],
            "q_chebyshev_radius_normalized": filtered[
                "q_chebyshev_radius_normalized"],
            "low_authority": filtered["low_authority"],
            "q_P100_min_physical": feasible_u_min[0],
            "q_P100_max_physical": feasible_u_max[0],
            "q_F200_min_physical": feasible_u_min[1],
            "q_F200_max_physical": feasible_u_max[1],
            "projection_distance_physical": filtered["projection_distance_physical"],
            "nonlinear_mismatch_outside_W": not w_ok,
            "mismatch_X2_normalized": mismatch[0],
            "mismatch_P2_normalized": mismatch[1],
            "physical_state_violation": bool(np.any(state < cfg.state_lower - 1e-8)
                or np.any(state > cfg.state_upper + 1e-8)
                or np.any(next_state < cfg.state_lower - 1e-8)
                or np.any(next_state > cfg.state_upper + 1e-8)),
            "physical_input_violation": bool(np.any(control < cfg.input_lower - 1e-8)
                or np.any(control > cfg.input_upper + 1e-8)),
            "robust_region_violation": bool(
                np.any(x_norm < design.robust_state_lower - 1e-8)
                or np.any(x_norm > design.robust_state_upper + 1e-8)
                or np.any(next_norm < design.robust_state_lower - 1e-8)
                or np.any(next_norm > design.robust_state_upper + 1e-8)
                or np.any(u_norm < design.robust_input_lower - 1e-8)
                or np.any(u_norm > design.robust_input_upper + 1e-8)),
            "Gm_exit": context["mode"] == "wait_Gm" and not contains(
                supervisor.sets[context["remaining_shocks"]], xi_next),
            "B_k_recovery_failure": context["mode"] == "recover_Bj"
                and complete["certificate_failure"],
            "QP_infeasible": False,
        })
        if rows[-1]["physical_state_violation"] or rows[-1]["physical_input_violation"]:
            failure = "physical_constraint_violation"
            break
        if rows[-1]["robust_region_violation"]:
            failure = "robust_region_violation"
            break
        if not w_ok:
            failure = "nonlinear_mismatch_outside_W"
            break
        if complete["certificate_failure"]:
            failure = complete["failure_reason"]
            break
        w_est = cfg.disturbance_estimate_ema * w_est + (
            1.0 - cfg.disturbance_estimate_ema) * mismatch
        previous_u = u_norm
        state = next_state
    events = supervisor.events
    radii = [row["q_chebyshev_radius_normalized"] for row in rows]
    projections = [row["projection_distance_physical"] for row in rows]
    result = {
        "scenario": scenario, "policy": policy, "trajectory": trajectory,
        "seed": seed, "shock_seconds_plant_side_only": schedule,
        "shock_detected_seconds": [e["detected_second"] for e in events],
        "steps_completed": len(rows), "certificate_failure": failure is not None,
        "failure_reason": failure,
        "G_m_exit_count": sum(row["Gm_exit"] for row in rows),
        "B_k_recovery_failure_count": sum(row["B_k_recovery_failure"] for row in rows)
            + int(failure in {"post_jump_outside_B20", "state_outside_B20",
                              "recovery_rank_did_not_decrease"}),
        "Omega_exit_count": sum(not row["in_Omega_after"] for row in rows)
            + int(failure == "omega_exit"),
        "QP_infeasible_count": sum(row["QP_infeasible"] for row in rows)
            + int(failure == "one_step_QP_infeasible"),
        "physical_state_violation_count": sum(row["physical_state_violation"] for row in rows),
        "physical_input_violation_count": sum(row["physical_input_violation"] for row in rows),
        "robust_region_violation_count": sum(row["robust_region_violation"] for row in rows),
        "nonlinear_mismatch_outside_W_count": sum(row["nonlinear_mismatch_outside_W"] for row in rows),
        "low_authority_fraction": float(np.mean([row["low_authority"] for row in rows]))
            if rows else None,
        "q_radius_min": float(np.min(radii)) if radii else None,
        "q_radius_median": float(np.median(radii)) if radii else None,
        "projection_distance_mean_physical": float(np.mean(projections))
            if projections else None,
        "projection_distance_max_physical": float(np.max(projections))
            if projections else None,
        "shock_events": events,
        "max_recovery_steps": max((e["recovery_steps"] for e in events
                                   if e["recovery_steps"] is not None), default=None),
        "all_three_recovered_within_D": (
            len(events) == 3 and all(e["completed_within_D"] for e in events)
        ),
    }
    return result, rows


def aggregate(results, all_steps):
    grouped = {}
    counts = ("G_m_exit_count", "B_k_recovery_failure_count", "Omega_exit_count",
              "QP_infeasible_count", "physical_state_violation_count",
              "physical_input_violation_count", "robust_region_violation_count",
              "nonlinear_mismatch_outside_W_count")
    for scenario in SCENARIOS:
        for policy in POLICIES:
            selected = [row for row in results if row["scenario"] == scenario
                        and row["policy"] == policy]
            steps = [row for row in all_steps if row["scenario"] == scenario
                     and row["policy"] == policy]
            radii = [row["q_chebyshev_radius_normalized"] for row in steps]
            projections = [row["projection_distance_physical"] for row in steps]
            grouped[f"{scenario}/{policy}"] = {
                "trajectories": len(selected),
                "all_three_recovered_within_D_count": sum(
                    row["all_three_recovered_within_D"] for row in selected),
                "certificate_failure_count": sum(row["certificate_failure"] for row in selected),
                **{key: sum(row[key] for row in selected) for key in counts},
                "max_recovery_steps": max((row["max_recovery_steps"] for row in selected
                                           if row["max_recovery_steps"] is not None), default=None),
                "low_authority_fraction": float(np.mean([
                    row["low_authority"] for row in steps
                ])) if steps else None,
                "min_q_radius": min(radii, default=None),
                "median_q_radius": float(np.median(radii)) if radii else None,
                "mean_candidate_projection_distance_physical": float(np.mean(projections))
                    if projections else None,
                "max_candidate_projection_distance_physical": max(projections, default=None),
            }
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_controlled_invariant_error_B/"
        "Omega_anchor_B_vertices.csv"))
    parser.add_argument("--dwell-sets", type=Path, default=DWELL_OUTPUT)
    parser.add_argument("--actor", type=Path, default=DEFAULT_STRESS_ACTOR)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.trajectories < 1:
        raise ValueError("trajectories must be positive")
    cfg = ExperimentConfig(
        benchmark_profile="zanon2016", experiment_mode="proposed",
        seed=args.seed, steps_per_episode=args.seconds,
        residual_parameterization="state_dependent_polytope",
    )
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    domain, _ = bounds_and_domains(cfg, model, design)
    omega = load_poly(args.omega_vertices)
    validate_source_summary(args.dwell_sets, omega)
    actor = FrozenActor(args.actor, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_files = {}
    for scenario in SCENARIOS:
        sets, ranks = make_rank_sets(
            scenario, omega, design, domain, args.dwell_sets
        )
        path = args.output_dir / f"{scenario}_G_and_B0_to_B20.npz"
        np.savez_compressed(path, **{
            **{f"G{m}": sets[m] for m in range(4)},
            **{f"B{j}_G{m}": ranks[m][j]
               for m in range(4) for j in range(D + 1)},
        })
        rank_files[scenario] = str(path)
    all_results, all_steps = [], []
    for scenario in SCENARIOS:
        for policy in POLICIES:
            for trajectory in range(args.trajectories):
                # Pair policies on exactly the same realized jump schedule.
                seed = args.seed + 100000 * SCENARIOS.index(scenario) + trajectory
                result, rows = stress_rollout(
                    cfg, model, design, omega, domain, scenario, policy,
                    actor, args.dwell_sets, seed, args.seconds, trajectory,
                )
                all_results.append(result)
                all_steps.extend(rows)
                write_csv(args.output_dir / f"{scenario}_{policy}_{trajectory}_steps.csv", rows)
                print(scenario, policy, trajectory, "steps", result["steps_completed"],
                      "recovered", result["all_three_recovered_within_D"],
                      "failure", result["failure_reason"], flush=True)
    probes = boundary_action_probes(
        cfg, model, design, omega, domain, args.dwell_sets
    )
    write_csv(args.output_dir / "G3_boundary_action_probes.csv", probes)
    summary = {
        "scope": "independent offline-tested event-driven one-step supervisor prototype",
        "D": D, "shock_clock_available_to_supervisor": False,
        "event_detection": "measurement innovation in W+delta but not W",
        "sets_unchanged": True, "online_multi_step_optimization": False,
        "candidate_controller": "existing interior-anchor mapping",
        "candidate_actor": str(args.actor),
        "recomputed_rank_set_files": rank_files,
        "low_authority_radius_threshold_normalized": LOW_RADIUS,
        "trajectories": all_results,
        "by_scenario_policy": aggregate(all_results, all_steps),
        "G3_boundary_action_probe_summary": {
            scenario: {
                "point_count": len({row["point"] for row in probes
                                    if row["scenario"] == scenario}),
                "low_authority_point_count": len({row["point"] for row in probes
                    if row["scenario"] == scenario and row["low_authority"]}),
                "QP_infeasible_count": sum(not row["QP_feasible"] for row in probes
                    if row["scenario"] == scenario),
                "linear_robust_next_failure_count": sum(
                    row["linear_robust_next_in_G3"] is False for row in probes
                    if row["scenario"] == scenario),
                "nonlinear_mismatch_outside_W_count": sum(
                    row["nonlinear_mismatch_in_W"] is False for row in probes
                    if row["scenario"] == scenario),
                "nonlinear_next_outside_G3_count": sum(
                    row["nonlinear_next_in_G3"] is False for row in probes
                    if row["scenario"] == scenario),
            } for scenario in SCENARIOS
        },
        "certificate_scope": (
            "fixed affine model, certified regular W, minimum dwell >=20, "
            "one of the three scenario-specific repeated jumps. Nonlinear "
            "plant stress tests are numerical validation only."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(json_finite(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_csv(args.output_dir / "trajectory_summary.csv", all_results)


if __name__ == "__main__":
    main()
