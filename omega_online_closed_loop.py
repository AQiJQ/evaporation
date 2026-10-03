"""Online Omega one-step QP prototype and 300 s Paper2016 closed-loop audit.

Mode A delegates unchanged to SafeController.act. Mode B projects the actual
total-input correction around fixed balanced reference B. No MPC or training.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .control import SafeController, project_qp_2d
from .controlled_invariant_error_set import contains, hull, intersect, predecessor
from .model import EvaporatorModel
from .omega_safe_action_filter_diagnosis import qp_rows
from .paper2016_compare import SCENARIOS, SHOCK_SECONDS, apply_state_shock
from .residual_action_space_diagnosis import (
    DEFAULT_STRESS_ACTOR, DEFAULT_DESIGN, REPO_DIR, FrozenActor, load_fixed_b,
)
from .train import observation
from .controlled_invariant_error_set import bounds_and_domains


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_controlled_invariant_error_B/"
        "Omega_anchor_B_vertices.csv"
    ))
    parser.add_argument("--actor", type=Path, default=DEFAULT_STRESS_ACTOR)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--rank-limit", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_omega_online_closed_loop_B"
    ))
    return parser.parse_args()


class OmegaSafeOnlineController:
    def __init__(self, cfg, model, design, omega, domain, rank_sets=None):
        self.cfg, self.model, self.d = cfg, model, design
        self.inner = SafeController(cfg, model, design)
        self.omega = omega
        self.domain = domain
        self.rank_sets = rank_sets
        self.was_in_omega = True

    @property
    def z(self):
        return self.inner.z

    def reset(self, state):
        self.inner.reset(state)
        self.was_in_omega = contains(
            self.omega, self.model.normalized_state(state) - self.d.z_ref
        )

    def rank_of(self, omega_error):
        return (
            minimum_rank(self.rank_sets, omega_error)
            if self.rank_sets is not None else None
        )

    def act(self, state, actor_action, *, action_is_normalized=True):
        if not action_is_normalized:
            raise ValueError("Omega controller expects normalized SAC action")
        x = self.model.normalized_state(state)
        rpi_error = x - self.inner.z
        omega_error = x - self.d.z_ref
        in_z = contains(self.d.rpi_boundary, rpi_error)
        in_omega = contains(self.omega, omega_error)
        omega_exit_event = bool(self.was_in_omega and not in_omega)
        self.was_in_omega = in_omega
        requested = self.cfg.residual_action_scale * np.asarray(actor_action)
        if in_z:
            # Preserve every step of the existing certified Z-inner path.
            control, info = self.inner.act(
                state, actor_action, action_is_normalized=True
            )
            applied = np.asarray(info["nominal"]) - np.asarray(info["base"])
            mode = "Z_mode_existing_controller"
            omega_qp_feasible = None
            q_base = np.asarray(info["base"]) + np.asarray(info["ancillary"]) - self.d.v_ref
            q_candidate = q_base + requested
        else:
            # Advance nominal z with its existing conservative zero-residual
            # base. The Omega QP acts on the actual plant input only.
            fallback_control, info = self.inner.act(
                state, np.zeros(2), action_is_normalized=True
            )
            q_base = np.asarray(info["base"]) + np.asarray(info["ancillary"]) - self.d.v_ref
            q_candidate = q_base + requested
            applied = np.zeros(2)
            control = fallback_control
            mode = "outside_certified_domain"
            omega_qp_feasible = None
            if in_omega:
                mode = "Omega_safe_one_step_QP"
                aq, bq = qp_rows(omega_error, self.omega, self.d, self.domain)
                q_safe, feasible = project_qp_2d(q_candidate, aq, bq)
                q_base_safe, base_feasible = project_qp_2d(q_base, aq, bq)
                omega_qp_feasible = bool(
                    feasible and base_feasible
                    and np.max(aq @ q_safe - bq) <= 2e-7
                )
                if omega_qp_feasible:
                    control = self.model.physical_input(self.d.v_ref + q_safe)
                    # Actor contribution is the paired-filter action change;
                    # it excludes the correction needed by the zero actor.
                    applied = q_safe - q_base_safe
                else:
                    mode = "Omega_QP_infeasible_fallback"
        merged = dict(info)
        merged.update({
            "mode": mode, "in_Z": in_z, "in_Omega": in_omega,
            "omega_exit_event": omega_exit_event,
            "rpi_error": rpi_error.copy(), "omega_error": omega_error.copy(),
            "reachable_rank_before": self.rank_of(omega_error),
            "q_base": q_base.copy(), "q_candidate": q_candidate.copy(),
            "requested_residual": requested.copy(), "applied_residual": applied.copy(),
            "omega_qp_feasible": omega_qp_feasible,
            "base_verification_qp_feasible": bool(info["qp_feasible"]),
            "z_before": np.asarray(info["z"]).copy(),
            "z_next": self.inner.z.copy(),
        })
        merged["actual_norm"] = self.model.normalized_input(control)
        # The Z verification QP is only a certificate for Mode A. Outside Z,
        # its failure must not invalidate an independently feasible Omega QP.
        merged["qp_feasible"] = (
            bool(info["qp_feasible"]) if in_z else
            bool(omega_qp_feasible) if in_omega else False
        )
        merged["execution_mask"] = float(merged["qp_feasible"])
        if mode != "Z_mode_existing_controller":
            merged["requested_candidate"] = np.asarray(info["base"]) + requested
            merged["candidate"] = merged["requested_candidate"].copy()
            merged["requested_residual"] = requested.copy()
            merged["parameterized_residual"] = applied.copy()
            merged["projection_gap"] = (
                float(np.linalg.norm(self.model.normalized_input(control)
                                     - self.d.v_ref - q_candidate))
                if omega_qp_feasible else float(info["projection_gap"])
            )
            merged["feasible_action_mapping_scale"] = (
                min(1.0, float(np.linalg.norm(applied)) /
                    max(float(np.linalg.norm(requested)), 1e-12))
            )
            merged["feasible_action_mapping_gap"] = float(
                np.linalg.norm(applied - requested)
            )
        return control, merged


def build_reachable_sets(design, omega, domain, limit):
    sets = [hull(design.rpi_boundary)]
    for _ in range(limit):
        pre = predecessor(sets[-1], design.a, design.b, design.w_vertices,
                          domain["q_lower"], domain["q_upper"])
        if len(pre) < 3:
            break
        sets.append(hull(intersect(omega, pre)))
    return sets


def minimum_rank(sets, xi):
    if not contains(sets[-1], xi, tol=1e-8):
        return None
    lo, hi = 0, len(sets) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if contains(sets[mid], xi, tol=1e-8):
            hi = mid
        else:
            lo = mid + 1
    return lo


def policy_action(mode, obs, actor, rng):
    if mode == "zero_residual":
        return np.zeros(2)
    if mode == "frozen_old_actor":
        return np.asarray(actor.action(obs), dtype=float)
    action = rng.uniform(-1.0, 1.0, 2)
    if np.max(np.abs(action)) < 0.2:
        action[int(np.argmax(np.abs(action)))] = 0.2
    return action


def rollout(cfg, model, design, omega, domain, scenario, policy, actor, seed, seconds):
    controller = OmegaSafeOnlineController(cfg, model, design, omega, domain)
    state = model.physical_state(design.z_ref)
    controller.reset(state)
    previous_u = design.v_ref.copy()
    w_est = np.zeros(2)
    rng = np.random.default_rng(seed)
    rows = []
    for second in range(seconds):
        before_shock = state.copy()
        state = apply_state_shock(state, scenario, second, scale=1.0)
        obs = observation(model, controller, state, previous_u, w_est)
        action = policy_action(policy, obs, actor, rng)
        control, info = controller.act(state, action)
        x = model.normalized_state(state)
        un = model.normalized_input(control)
        next_state = model.step(state, control, cfg.disturbance_nominal)
        next_x = model.normalized_state(next_state)
        next_rpi_error = next_x - controller.z
        next_omega_error = next_x - design.z_ref
        mismatch = next_x - (design.a @ x + design.b @ un + design.affine)
        w_est = cfg.disturbance_estimate_ema * w_est + (
            1.0 - cfg.disturbance_estimate_ema
        ) * mismatch
        req = np.asarray(info["requested_residual"])
        app = np.asarray(info["applied_residual"])
        rows.append({
            "scenario": scenario, "policy": policy, "second": second,
            "shock_applied": bool(second in SHOCK_SECONDS),
            "shock_X2": float(state[0] - before_shock[0]),
            "shock_P2": float(state[1] - before_shock[1]),
            "mode": info["mode"],
            "in_Z_before": bool(info["in_Z"]),
            "in_Omega_before": bool(info["in_Omega"]),
            "in_Z_next": contains(design.rpi_boundary, next_rpi_error),
            "in_Omega_next": contains(omega, next_omega_error),
            "omega_exit_event": bool(info["omega_exit_event"]),
            "rpi_error_X2_physical": info["rpi_error"][0] * cfg.state_scale[0],
            "rpi_error_P2_physical": info["rpi_error"][1] * cfg.state_scale[1],
            "omega_error_X2_physical": info["omega_error"][0] * cfg.state_scale[0],
            "omega_error_P2_physical": info["omega_error"][1] * cfg.state_scale[1],
            "omega_error_X2_normalized": info["omega_error"][0],
            "omega_error_P2_normalized": info["omega_error"][1],
            "next_omega_error_X2_normalized": next_omega_error[0],
            "next_omega_error_P2_normalized": next_omega_error[1],
            "X2": state[0], "P2": state[1],
            "X2_next": next_state[0], "P2_next": next_state[1],
            "z_X2": model.physical_state(info["z_before"])[0],
            "z_P2": model.physical_state(info["z_before"])[1],
            "P100": control[0], "F200": control[1],
            "q_base_P100": info["q_base"][0] * cfg.input_scale[0],
            "q_base_F200": info["q_base"][1] * cfg.input_scale[1],
            "q_candidate_P100": info["q_candidate"][0] * cfg.input_scale[0],
            "q_candidate_F200": info["q_candidate"][1] * cfg.input_scale[1],
            "actor_a_P100": action[0], "actor_a_F200": action[1],
            "requested_residual_norm_normalized": float(np.linalg.norm(req)),
            "applied_residual_norm_normalized": float(np.linalg.norm(app)),
            "requested_residual_P100": req[0] * cfg.input_scale[0],
            "requested_residual_F200": req[1] * cfg.input_scale[1],
            "applied_residual_P100": app[0] * cfg.input_scale[0],
            "applied_residual_F200": app[1] * cfg.input_scale[1],
            "physical_state_violation": bool(np.any(next_state < cfg.state_lower - 1e-8)
                or np.any(next_state > cfg.state_upper + 1e-8)),
            "physical_state_violation_at_shock": bool(
                second in SHOCK_SECONDS and (
                    np.any(state < cfg.state_lower - 1e-8)
                    or np.any(state > cfg.state_upper + 1e-8)
                )
            ),
            "physical_input_violation": bool(np.any(control < cfg.input_lower - 1e-8)
                or np.any(control > cfg.input_upper + 1e-8)),
            "robust_state_violation": bool(np.any(next_x < design.robust_state_lower - 1e-8)
                or np.any(next_x > design.robust_state_upper + 1e-8)),
            "robust_state_violation_at_shock": bool(
                second in SHOCK_SECONDS and (
                    np.any(x < design.robust_state_lower - 1e-8)
                    or np.any(x > design.robust_state_upper + 1e-8)
                )
            ),
            "robust_input_violation": bool(np.any(un < design.robust_input_lower - 1e-8)
                or np.any(un > design.robust_input_upper + 1e-8)),
            "qp_infeasible": bool(not info["base_verification_qp_feasible"] or
                info["omega_qp_feasible"] is False),
            "omega_qp_feasible": info["omega_qp_feasible"],
            "nonlinear_w_in_W": contains(design.w_vertices, mismatch),
            "nonlinear_w_X2_normalized": mismatch[0],
            "nonlinear_w_P2_normalized": mismatch[1],
            "economic_stage_cost": model.economic_cost(
                state, control, cfg.disturbance_nominal
            ),
        })
        state = next_state
        previous_u = un
    return rows


def state_metrics(rows, baseline, reference, key):
    values = np.asarray([row[key] for row in rows if row["second"] >= 40])
    base = np.asarray([row[key] for row in baseline if row["second"] >= 40])
    tolerance = max(0.05 * float(np.max(np.abs(base - reference))), 1e-9)
    inside = np.abs(values - reference) <= tolerance
    settling = next((i for i in range(len(values)) if np.all(inside[i:])), None)
    return {
        "reference": float(reference), "common_tolerance": tolerance,
        "window_start_second": 40,
        "IAE_after_last_shock": float(np.sum(np.abs(values - reference))),
        "ISE_after_last_shock": float(np.sum((values - reference) ** 2)),
        "settling_seconds_after_last_shock": settling,
    }


def summarize(rows, cfg, model, design, baseline):
    requested = np.asarray([r["requested_residual_norm_normalized"] for r in rows])
    applied = np.asarray([r["applied_residual_norm_normalized"] for r in rows])
    post = [r for r in rows if r["second"] >= 40]
    steady_cost = model.economic_cost(
        model.physical_state(design.z_ref), model.physical_input(design.v_ref),
        cfg.disturbance_nominal,
    )
    in_z = [bool(r["in_Z_before"]) for r in rows]
    return {
        "physical_state_violation_steps": sum(r["physical_state_violation"] for r in rows),
        "physical_state_violation_at_shock_count": sum(
            r["physical_state_violation_at_shock"] for r in rows
        ),
        "physical_input_violation_steps": sum(r["physical_input_violation"] for r in rows),
        "robust_region_violation_steps": sum(
            r["robust_state_violation"] or r["robust_input_violation"] for r in rows
        ),
        "robust_state_violation_at_shock_count": sum(
            r["robust_state_violation_at_shock"] for r in rows
        ),
        "Omega_exit_count": (
            sum(r["omega_exit_event"] for r in rows)
            + int(rows[-1]["in_Omega_before"] and not rows[-1]["in_Omega_next"])
        ),
        "Omega_outside_steps": sum(not r["in_Omega_before"] for r in rows),
        "QP_infeasible_steps": sum(r["qp_infeasible"] for r in rows),
        "Z_outside_steps": sum(not r["in_Z_before"] for r in rows),
        "Z_reentry_events": sum(
            not in_z[i-1] and in_z[i] for i in range(1, len(in_z))
        ),
        "Z_reentry_after_last_shock_second": next((
            r["second"] - 40 for r in post if r["in_Z_before"]
        ), None),
        "Omega_safe_mode_fraction": float(np.mean([
            r["mode"] == "Omega_safe_one_step_QP" for r in rows
        ])),
        "Z_mode_fraction": float(np.mean([
            r["mode"] == "Z_mode_existing_controller" for r in rows
        ])),
        "outside_certified_domain_fraction": float(np.mean([
            r["mode"] == "outside_certified_domain" for r in rows
        ])),
        "requested_residual_norm_mean": float(np.mean(requested)),
        "applied_residual_norm_mean": float(np.mean(applied)),
        "applied_over_requested_sum_ratio": float(
            np.sum(applied) / np.sum(requested)
        ) if np.sum(requested) > 1e-12 else None,
        "nonlinear_W_exceedance_steps": sum(not r["nonlinear_w_in_W"] for r in rows),
        "X2": state_metrics(rows, baseline, design.z_ref[0] * cfg.state_scale[0]
            + cfg.linearization_state[0], "X2"),
        "P2": state_metrics(rows, baseline, design.z_ref[1] * cfg.state_scale[1]
            + cfg.linearization_state[1], "P2"),
        "P100_total_variation": float(np.sum(np.abs(np.diff([r["P100"] for r in rows])))),
        "F200_total_variation": float(np.sum(np.abs(np.diff([r["F200"] for r in rows])))),
        "J_econ": float(sum(r["economic_stage_cost"] for r in rows)),
        "J_transient": float(sum(r["economic_stage_cost"] - steady_cost for r in rows)),
        "steady_stage_cost": steady_cost,
        "rank_decrease_steps": sum(r["rank_decreased"] is True for r in rows),
        "rank_increase_steps": sum(r["rank_decreased"] is False and
            r["rank_next"] is not None and r["rank_before"] is not None
            and r["rank_next"] > r["rank_before"] for r in rows),
        "rank_unknown_steps": sum(r["rank_before"] is None for r in rows),
    }


def main():
    args = parse_args()
    if args.seconds < 41:
        raise ValueError("300 s endpoint diagnosis requires all three shocks")
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed")
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    cfg.residual_parameterization = "state_dependent_polytope"
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    domain, _ = bounds_and_domains(cfg, model, design)
    actor = FrozenActor(args.actor, cfg)
    if actor.scale <= 0.0:
        raise ValueError("Frozen actor checkpoint has zero policy_output_scale")
    policies = ("zero_residual", "frozen_old_actor", "random_bounded_nonzero")
    all_rows = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for policy_index, policy in enumerate(policies):
            rows = rollout(cfg, model, design, omega, domain, scenario, policy,
                           actor, args.seed + 100 * scenario_index + policy_index,
                           args.seconds)
            all_rows.extend(rows)
            print(scenario, policy, "Omega exits", sum(r["omega_exit_event"] for r in rows),
                  "QP failures", sum(r["qp_infeasible"] for r in rows), flush=True)
    sets = build_reachable_sets(design, omega, domain, args.rank_limit)
    for row in all_rows:
        xi = np.array([row["omega_error_X2_normalized"],
                       row["omega_error_P2_normalized"]])
        next_xi = np.array([row["next_omega_error_X2_normalized"],
                            row["next_omega_error_P2_normalized"]])
        before, after = minimum_rank(sets, xi), minimum_rank(sets, next_xi)
        row["rank_before"] = before
        row["rank_next"] = after
        row["rank_decreased"] = (
            after < before if before is not None and after is not None else None
        )
    summary = {
        "scope": "online prototype only; existing Z controller unchanged; no training",
        "coordinates": {
            "rpi_error": "x-z; only for Z membership",
            "omega_error": "x-x_ref; only for Omega/R_k membership",
        },
        "reference_state_physical": model.physical_state(design.z_ref).tolist(),
        "reference_input_physical": model.physical_input(design.v_ref).tolist(),
        "rank_sets_computed": len(sets),
        "rank_hard_constraint_enabled": False,
        "continuous_domain_W_coverage_proved": False,
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        summary["scenarios"][scenario] = {}
        baseline = [r for r in all_rows if r["scenario"] == scenario
                    and r["policy"] == "zero_residual"]
        for policy in policies:
            subset = [r for r in all_rows if r["scenario"] == scenario
                      and r["policy"] == policy]
            summary["scenarios"][scenario][policy] = summarize(
                subset, cfg, model, design, baseline
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trajectories.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
