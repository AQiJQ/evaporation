"""Balanced-B Omega-safe SAC training with paired fixed Paper2016 evaluation.

Uses the existing SAC, replay and reward loop, with fixed B/K/W/Z/S/Omega.
The default run is the seed-42, 300x300 reward-shaping short experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .controlled_invariant_error_set import bounds_and_domains, certificate, hull
from .model import EvaporatorModel
from .omega_feasible_normalized import FeasibleSetNormalizedController
from .omega_interior_anchor import InteriorAnchorController
from .omega_dwell_shock_sets import DEFAULT_OUTPUT as DWELL_SET_OUTPUT
from .omega_dwell_sac_controller import DwellSupervisedInteriorController
from .omega_supervisor_observation import DwellObservationController
from .omega_online_closed_loop import OmegaSafeOnlineController, build_reachable_sets
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b
from .sac import ReplayBuffer, SACAgent, SACConfig, set_seed
from .train import (
    ACTION_DIM, OBS_DIM, ZeroResidualPolicy,
    balanced_random_paper2016_schedule, run_episode,
)
from .paper2016_compare import SCENARIOS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--action-mapping", choices=(
        "feasible_set_normalized", "legacy_fixed_residual", "interior_anchor"
    ), default="feasible_set_normalized")
    parser.add_argument("--supervisor", choices=("none", "dwell20"), default="none")
    parser.add_argument("--observation-variant", choices=("base19", "supervisor23"),
                        default="base19")
    parser.add_argument("--dwell-sets", type=Path, default=DWELL_SET_OUTPUT)
    parser.add_argument("--frozen-reward-calibration", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_omega_safe_B_interior_anchor_seed42_300x300/"
        "baseline_reward_calibration.json"))
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_controlled_invariant_error_B/"
        "Omega_anchor_B_vertices.csv"
    ))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--calibrate-only", action="store_true")
    return parser.parse_args()


def controller(cfg, model, design, omega, domain, rank_sets=None,
               action_mapping="legacy_fixed_residual", supervisor="none",
               dwell_sets=None, observation_variant="base19"):
    if supervisor == "dwell20":
        if action_mapping != "interior_anchor":
            raise ValueError("Dwell supervisor requires interior_anchor mapping")
        controller_type = (DwellObservationController
                           if observation_variant == "supervisor23"
                           else DwellSupervisedInteriorController)
        return controller_type(
            cfg, model, design, omega, domain, dwell_sets
        )
    if observation_variant != "base19":
        raise ValueError("supervisor23 requires dwell20 supervisor")
    controller_type = {
        "feasible_set_normalized": FeasibleSetNormalizedController,
        "legacy_fixed_residual": OmegaSafeOnlineController,
        "interior_anchor": InteriorAnchorController,
    }[action_mapping]
    return controller_type(
        cfg, model, design, omega, domain, rank_sets=rank_sets
    )


def evaluate_three(cfg, model, design, omega, domain, policy, rank_sets=None,
                   action_mapping="legacy_fixed_residual", supervisor="none",
                   dwell_sets=None, observation_variant="base19"):
    results = {}
    for index, scenario in enumerate(SCENARIOS):
        stat, records, _ = run_episode(
            cfg, model, controller(cfg, model, design, omega, domain, rank_sets,
                                   action_mapping, supervisor, dwell_sets,
                                   observation_variant),
            policy, None, np.random.default_rng(cfg.seed + 31000 + index),
            training=False, global_step=0, paper_scenario=scenario,
        )
        results[scenario] = {
            "stat": stat, "records": records,
            "robust_input_lower": design.robust_input_lower,
            "robust_input_upper": design.robust_input_upper,
        }
    return results


def baseline_calibration(cfg, model, design, omega, domain, output_dir,
                         action_mapping="legacy_fixed_residual"):
    baseline = evaluate_three(
        cfg, model, design, omega, domain, ZeroResidualPolicy(),
        action_mapping=action_mapping,
    )
    safety_keys = (
        "omega_exit_count", "qp_infeasible_rate", "violation_rate",
        "robust_operating_region_violation_rate",
        "disturbance_bound_exceedance_rate",
    )
    baseline_safety = {
        scenario: {
            key: baseline[scenario]["stat"][key] for key in safety_keys
        } for scenario in SCENARIOS
    }
    failures = {
        scenario: {key: value for key, value in values.items() if value > 0}
        for scenario, values in baseline_safety.items()
    }
    failures = {key: value for key, value in failures.items() if value}
    if failures:
        raise RuntimeError(f"Zero-residual baseline safety audit failed: {failures}")
    records = [r for scenario in SCENARIOS for r in baseline[scenario]["records"]]
    mean_economic = float(np.mean([abs(r["economic_reward"]) for r in records]))
    loss_keys = {
        "state_recovery": ("state_recovery_loss", 0.10),
        "P100_move": ("p100_move_loss", 0.05),
        "F200_move": ("f200_move_loss", 0.05),
        "saturation": ("saturation_loss", 0.05),
    }
    means = {
        name: float(np.mean([r[key] for r in records]))
        for name, (key, _) in loss_keys.items()
    }
    if mean_economic <= 1e-12 or any(value <= 1e-12 for value in means.values()):
        raise RuntimeError(
            f"Reward calibration requires nonzero baseline losses: "
            f"economic={mean_economic}, losses={means}"
        )
    weights = {
        name: target * mean_economic / means[name]
        for name, (_, target) in loss_keys.items()
    }
    cfg.paper2016_state_recovery_penalty_weight = weights["state_recovery"]
    cfg.paper2016_p100_move_penalty_weight = weights["P100_move"]
    cfg.paper2016_f200_move_penalty_weight = weights["F200_move"]
    cfg.paper2016_saturation_penalty_weight = weights["saturation"]
    result = {
        "action_mapping": action_mapping,
        "mean_absolute_economic_reward": mean_economic,
        "mean_losses": means,
        "calibrated_weights": weights,
        "target_penalty_fractions": {
            name: target for name, (_, target) in loss_keys.items()
        },
        "realized_baseline_penalty_means": {
            name: weights[name] * means[name] for name in loss_keys
        },
        "realized_baseline_penalty_fractions": {
            name: weights[name] * means[name] / mean_economic
            for name in loss_keys
        },
        "scenario_loss_means": {
            scenario: {
                name: float(np.mean([
                    record[key] for record in baseline[scenario]["records"]
                ]))
                for name, (key, _) in loss_keys.items()
            } for scenario in SCENARIOS
        },
        "scenario_J_econ": {
            scenario: baseline[scenario]["stat"]["J_econ"]
            for scenario in SCENARIOS
        },
        "scenario_omega_exit_count": {
            scenario: baseline[scenario]["stat"]["omega_exit_count"]
            for scenario in SCENARIOS
        },
        "scenario_safety_audit": baseline_safety,
        "baseline_step_count": len(records),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_reward_calibration.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("baseline reward calibration:", json.dumps(result, indent=2), flush=True)
    return baseline, result


def frozen_supervised_baseline(cfg, model, design, omega, domain, output_dir,
                               dwell_sets, calibration_path,
                               observation_variant="base19"):
    """Pair on the new supervisor while preserving all prior reward weights."""
    frozen = json.loads(calibration_path.read_text(encoding="utf-8"))
    weights = frozen["calibrated_weights"]
    cfg.paper2016_state_recovery_penalty_weight = float(weights["state_recovery"])
    cfg.paper2016_p100_move_penalty_weight = float(weights["P100_move"])
    cfg.paper2016_f200_move_penalty_weight = float(weights["F200_move"])
    cfg.paper2016_saturation_penalty_weight = float(weights["saturation"])
    baseline = evaluate_three(
        cfg, model, design, omega, domain, ZeroResidualPolicy(),
        action_mapping="interior_anchor", supervisor="dwell20",
        dwell_sets=dwell_sets, observation_variant=observation_variant,
    )
    safety_keys = (
        "omega_exit_count", "qp_infeasible_rate", "violation_rate",
        "robust_operating_region_violation_rate",
        "disturbance_bound_exceedance_rate",
        "supervisor_Gm_exit_count", "supervisor_Bk_recovery_failure_count",
        "supervisor_omega_exit_count", "supervisor_QP_infeasible_count",
    )
    failures = {
        scenario: {key: baseline[scenario]["stat"][key] for key in safety_keys
                   if baseline[scenario]["stat"][key] > 0}
        for scenario in SCENARIOS
    }
    failures = {key: value for key, value in failures.items() if value}
    if failures:
        raise RuntimeError(f"Supervised zero-residual baseline failed: {failures}")
    if not all(baseline[s]["stat"]["supervisor_all_three_recovered_within_D"]
               for s in SCENARIOS):
        raise RuntimeError("Supervised zero-residual baseline missed D=20 recovery")
    records = [r for scenario in SCENARIOS for r in baseline[scenario]["records"]]
    result = {
        "mode": "frozen_previous_interior_anchor_reward_weights",
        "source": str(calibration_path), "weights": weights,
        "paired_supervised_baseline_J_econ": {
            s: baseline[s]["stat"]["J_econ"] for s in SCENARIOS
        },
        "mean_absolute_economic_reward": float(np.mean([
            abs(r["economic_reward"]) for r in records
        ])),
        "realized_penalty_means": {
            name: float(np.mean([r[key] for r in records]))
            for name, key in (
                ("state_recovery", "state_recovery_penalty"),
                ("P100_move", "p100_move_penalty"),
                ("F200_move", "f200_move_penalty"),
                ("saturation", "saturation_penalty"),
            )
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_reward_calibration.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return baseline, result


def recovery_metrics(records, baseline, reference, state_index):
    signal = np.asarray([
        r["state"][state_index] for r in records[40:]
    ])
    baseline_signal = np.asarray([
        r["state"][state_index] for r in baseline[40:]
    ])
    tolerance = max(0.05 * float(np.max(np.abs(baseline_signal - reference))), 1e-9)
    inside = np.abs(signal - reference) <= tolerance
    settling = next((i for i in range(len(signal)) if np.all(inside[i:])), None)
    return {
        "IAE": float(np.sum(np.abs(signal - reference))),
        "ISE": float(np.sum((signal - reference) ** 2)),
        "common_reference": float(reference),
        "common_tolerance": tolerance,
        "settling_time_seconds_after_last_shock": settling,
    }


def action_collapse_metrics(records):
    """Condition collapse on a nonzero actor command, also by safety mode."""
    selected = [r for r in records if r["nonzero_actor"]]
    collapsed = sum(r["collapsed_action"] for r in selected)
    result = {
        "nonzero_actor_steps": len(selected),
        "collapsed_action_steps": int(collapsed),
        "action_collapse_fraction": (
            float(collapsed / len(selected)) if selected else None
        ),
        "applied_displacement_from_baseline_mean": float(np.mean([
            r["applied_displacement_from_baseline"] for r in records
        ])),
        "applied_displacement_from_baseline_max": float(np.max([
            r["applied_displacement_from_baseline"] for r in records
        ])),
    }
    for mode, prefix in (
        ("Z_mode_existing_controller", "Z_mode"),
        ("Omega_safe_one_step_QP", "Omega_mode"),
    ):
        subset = [r for r in selected if r["safety_mode"] == mode]
        count = sum(r["collapsed_action"] for r in subset)
        result[f"{prefix}_nonzero_actor_steps"] = len(subset)
        result[f"{prefix}_collapsed_action_steps"] = int(count)
        result[f"{prefix}_action_collapse_fraction"] = (
            float(count / len(subset)) if subset else None
        )
    return result


def supervisor_record_metrics(records, stat):
    enabled = "supervisor_Gm_normal_fraction" in stat
    events = [{
        "step": step,
        "post_jump_rank": record["supervisor_rank_before"],
        "inferred_jump_scale": record["supervisor_inferred_jump_scale"],
    } for step, record in enumerate(records)
        if record["supervisor_event_detected"]]
    completions = [record["supervisor_recovery_completed_steps"]
                   for record in records
                   if record["supervisor_recovery_completed_steps"] is not None]
    for event, steps in zip(events, completions):
        event["recovery_steps"] = steps
    return {
        "supervisor_enabled": enabled,
        "Gm_normal_mode_fraction": stat.get("supervisor_Gm_normal_fraction"),
        "Bj_recovery_mode_fraction": stat.get("supervisor_Bj_recovery_fraction"),
        "shock_ranks_and_recovery_json": json.dumps(events),
        "max_recovery_steps": stat.get("supervisor_recovery_max_steps"),
        "all_three_recovered_within_D": stat.get(
            "supervisor_all_three_recovered_within_D"),
        "recovery_deadline_failure_count": int(
            enabled and not stat["supervisor_all_three_recovered_within_D"]
        ),
        "supervisor_projection_distance_mean_physical": stat.get(
            "supervisor_projection_distance_mean_physical"),
        "supervisor_projection_distance_max_physical": stat.get(
            "supervisor_projection_distance_max_physical"),
        "supervisor_low_authority_fraction": stat.get(
            "supervisor_low_authority_fraction"),
        "supervisor_q_radius_min": stat.get("supervisor_q_radius_min"),
        "supervisor_q_radius_median": stat.get("supervisor_q_radius_median"),
        "G3_low_authority_steps": sum(
            r["supervisor_low_authority"] and r["supervisor_remaining_shocks"] == 3
            for r in records
        ),
        "Gm_exit_count": stat.get("supervisor_Gm_exit_count", 0),
        "Bk_recovery_failure_count": stat.get(
            "supervisor_Bk_recovery_failure_count", 0),
        "supervisor_Omega_exit_count": stat.get("supervisor_omega_exit_count", 0),
        "supervisor_QP_infeasible_count": stat.get(
            "supervisor_QP_infeasible_count", 0),
    }


def eval_row(episode, scenario, result, baseline, cfg):
    stat, records = result["stat"], result["records"]
    baseline_stat, baseline_records = baseline["stat"], baseline["records"]
    x2 = recovery_metrics(records, baseline_records,
                          cfg.robust_economic_reference_state[0], 0)
    p2 = recovery_metrics(records, baseline_records,
                          cfg.robust_economic_reference_state[1], 1)
    baseline_x2 = recovery_metrics(baseline_records, baseline_records,
                                   cfg.robust_economic_reference_state[0], 0)
    baseline_p2 = recovery_metrics(baseline_records, baseline_records,
                                   cfg.robust_economic_reference_state[1], 1)
    controls = np.asarray([r["control"] for r in records])
    baseline_controls = np.asarray([r["control"] for r in baseline_records])
    requested = np.asarray([r["residual_requested_norm"] for r in records])
    applied = np.asarray([r["residual_applied_norm"] for r in records])
    raw_actions = np.asarray([r["raw_action"] for r in records])
    robust_f200_upper = (
        cfg.linearization_input[1]
        + result["design_robust_input_upper_F200"] * cfg.input_scale[1]
    )
    robust_lower = (
        cfg.linearization_input + result["robust_input_lower"] * cfg.input_scale
    )
    robust_upper = (
        cfg.linearization_input + result["robust_input_upper"] * cfg.input_scale
    )
    robust_mid = 0.5 * (robust_lower + robust_upper)
    robust_half_range = 0.5 * (robust_upper - robust_lower)
    eta = np.abs(controls - robust_mid) / robust_half_range
    baseline_eta = np.abs(baseline_controls - robust_mid) / robust_half_range
    saturation_steps = int(np.sum(np.any(eta > 0.9, axis=1)))
    baseline_saturation_steps = int(np.sum(np.any(baseline_eta > 0.9, axis=1)))
    at_f200_upper = np.abs(controls[:, 1] - robust_f200_upper) <= 1e-6
    longest_f200_upper_run = 0
    current_upper_run = 0
    for at_upper in at_f200_upper:
        current_upper_run = current_upper_run + 1 if at_upper else 0
        longest_f200_upper_run = max(longest_f200_upper_run, current_upper_run)
    return {
        "episode": episode, "scenario": scenario,
        "paired_economic_improvement_percent": float(
            100.0 * (baseline_stat["J_econ"] - stat["J_econ"])
            / max(abs(baseline_stat["J_econ"]), 1e-12)
        ),
        "J_econ": stat["J_econ"], "baseline_J_econ": baseline_stat["J_econ"],
        "J_transient": stat["J_transient"],
        "total_evaluation_return": stat["total_reward_total"],
        "economic_reward_total": stat["economic_reward_total"],
        "state_recovery_penalty_total": stat["state_recovery_penalty_total"],
        "P100_move_penalty_total": stat["p100_move_penalty_total"],
        "F200_move_penalty_total": stat["f200_move_penalty_total"],
        "saturation_penalty_total": stat["saturation_penalty_total"],
        "X2_IAE": x2["IAE"], "X2_ISE": x2["ISE"],
        "baseline_X2_IAE": baseline_x2["IAE"],
        "baseline_X2_ISE": baseline_x2["ISE"],
        "X2_common_reference_settling_seconds": x2["settling_time_seconds_after_last_shock"],
        "baseline_X2_common_reference_settling_seconds": (
            baseline_x2["settling_time_seconds_after_last_shock"]
        ),
        "X2_common_tolerance": x2["common_tolerance"],
        "P2_IAE": p2["IAE"], "P2_ISE": p2["ISE"],
        "baseline_P2_IAE": baseline_p2["IAE"],
        "baseline_P2_ISE": baseline_p2["ISE"],
        "P2_common_reference_settling_seconds": p2["settling_time_seconds_after_last_shock"],
        "baseline_P2_common_reference_settling_seconds": (
            baseline_p2["settling_time_seconds_after_last_shock"]
        ),
        "P2_common_tolerance": p2["common_tolerance"],
        "normalized_mean_IAE_ratio": float(0.5 * (
            x2["IAE"] / max(baseline_x2["IAE"], 1e-12)
            + p2["IAE"] / max(baseline_p2["IAE"], 1e-12)
        )),
        "P100_TV": float(np.sum(np.abs(np.diff(controls[:, 0])))),
        "F200_TV": float(np.sum(np.abs(np.diff(controls[:, 1])))),
        "baseline_P100_TV": float(np.sum(np.abs(np.diff(baseline_controls[:, 0])))),
        "baseline_F200_TV": float(np.sum(np.abs(np.diff(baseline_controls[:, 1])))),
        "F200_min": float(np.min(controls[:, 1])),
        "F200_max": float(np.max(controls[:, 1])),
        "F200_at_robust_upper_steps": int(np.sum(at_f200_upper)),
        "F200_longest_robust_upper_run_steps": longest_f200_upper_run,
        "robust_input_saturation_steps": saturation_steps,
        "baseline_robust_input_saturation_steps": baseline_saturation_steps,
        "P100_outer_10pct_steps": int(np.sum(eta[:, 0] > 0.9)),
        "F200_outer_10pct_steps": int(np.sum(eta[:, 1] > 0.9)),
        "requested_residual_norm_mean": stat["residual_requested_norm_mean"],
        "applied_residual_norm_mean": stat["residual_applied_norm_mean"],
        "applied_over_requested_sum_ratio": float(
            np.sum(applied) / np.sum(requested)
        ) if np.sum(requested) > 1e-12 else None,
        "actor_raw_action_saturation_fraction": float(np.mean(
            np.max(np.abs(raw_actions), axis=1) >= 0.99
        )),
        "actor_raw_P100_saturation_fraction": float(np.mean(
            np.abs(raw_actions[:, 0]) >= 0.99
        )),
        "actor_raw_F200_saturation_fraction": float(np.mean(
            np.abs(raw_actions[:, 1]) >= 0.99
        )),
        "Z_mode_fraction": stat["z_mode_fraction"],
        "Omega_mode_fraction": stat["omega_mode_fraction"],
        "Omega_exit_count": stat["omega_exit_count"],
        "Omega_outside_steps": stat["omega_outside_steps"],
        "QP_infeasible_rate": stat["qp_infeasible_rate"],
        "physical_violation_rate": stat["violation_rate"],
        "physical_state_violation_steps": int(sum(
            r["physical_state_violation"] for r in records
        )),
        "physical_input_violation_steps": int(sum(
            r["physical_input_violation"] for r in records
        )),
        "robust_region_violation_rate": stat["robust_operating_region_violation_rate"],
        "nonlinear_W_exceedance_rate": stat["disturbance_bound_exceedance_rate"],
        "nonlinear_W_exceedance_steps": int(round(
            stat["disturbance_bound_exceedance_rate"] * len(records)
        )),
        **action_collapse_metrics(records),
        **supervisor_record_metrics(records, stat),
    }


PERFORMANCE_METRICS = (
    "X2_IAE", "X2_ISE", "P2_IAE", "P2_ISE", "P100_TV", "F200_TV",
)
SAFETY_METRICS = (
    "Omega_exit_count", "Omega_outside_steps", "QP_infeasible_rate",
    "physical_violation_rate", "physical_state_violation_steps",
    "physical_input_violation_steps", "robust_region_violation_rate",
    "nonlinear_W_exceedance_steps",
    "Gm_exit_count", "Bk_recovery_failure_count",
    "supervisor_Omega_exit_count", "supervisor_QP_infeasible_count",
    "recovery_deadline_failure_count",
)


def assess_checkpoint(rows, global_step, warmup_steps):
    """Audit all fixed scenarios; report trade-offs before selecting an actor.

    A 10% baseline tolerance defines 'not clearly worse'.  When baseline TV
    is exactly zero, allow at most one physical input unit of TV rather than
    dividing by zero.  The nearest-tradeoff candidate is explicitly labelled
    diagnostic if no checkpoint satisfies every performance tolerance.
    """
    if len(rows) != len(SCENARIOS):
        raise ValueError("Checkpoint audit requires all three fixed scenarios")
    mean_economic = float(np.mean([
        row["paired_economic_improvement_percent"] for row in rows
    ]))
    worst_economic = float(min(
        row["paired_economic_improvement_percent"] for row in rows
    ))
    safety_passed = all(
        float(row[key]) == 0.0 for row in rows for key in SAFETY_METRICS
    )
    relative_changes = {}
    degradations = []
    tolerance_passed = True
    for row in rows:
        scenario = row["scenario"]
        relative_changes[scenario] = {}
        for key in PERFORMANCE_METRICS:
            value = float(row[key])
            baseline = float(row[f"baseline_{key}"])
            denominator = max(baseline, 1.0) if key.endswith("_TV") else max(baseline, 1e-12)
            change = (value - baseline) / denominator
            relative_changes[scenario][key] = change
            degradations.append(max(0.0, change))
            allowed_increase = (1.0 if key.endswith("_TV") and baseline < 1e-12
                                else 0.10 * baseline)
            if value > baseline + allowed_increase + 1e-12:
                tolerance_passed = False
    eligible = bool(global_step >= warmup_steps and mean_economic > 0 and safety_passed)
    return {
        "episode": rows[0]["episode"], "global_step": global_step,
        "post_warmup": bool(global_step >= warmup_steps),
        "mean_economic_improvement_percent": mean_economic,
        "worst_scenario_economic_improvement_percent": worst_economic,
        "mean_normalized_IAE_ratio": float(np.mean([
            row["normalized_mean_IAE_ratio"] for row in rows
        ])),
        "safety_passed": safety_passed,
        "eligible_positive_economic_safe": eligible,
        "performance_within_10pct_baseline": tolerance_passed,
        "strict_constrained_candidate": bool(eligible and tolerance_passed),
        "mean_positive_performance_degradation": float(np.mean(degradations)),
        "worst_positive_performance_degradation": float(max(degradations)),
        "relative_changes_json": json.dumps(relative_changes, sort_keys=True),
        "common_settling_json": json.dumps({
            row["scenario"]: {
                state: {
                    "SAC": row[f"{state}_common_reference_settling_seconds"],
                    "baseline": row[f"baseline_{state}_common_reference_settling_seconds"],
                } for state in ("X2", "P2")
            } for row in rows
        }, sort_keys=True),
        "saturation_steps_json": json.dumps({
            row["scenario"]: {
                "SAC": row["robust_input_saturation_steps"],
                "baseline": row["baseline_robust_input_saturation_steps"],
                "F200_upper_steps": row["F200_at_robust_upper_steps"],
            } for row in rows
        }, sort_keys=True),
        "scenario_metrics_json": json.dumps({
            row["scenario"]: {
                key: row[key] for key in (
                    "paired_economic_improvement_percent", "J_econ",
                    "X2_IAE", "X2_ISE", "P2_IAE", "P2_ISE",
                    "P100_TV", "F200_TV", "F200_at_robust_upper_steps",
                    "robust_input_saturation_steps", "Omega_exit_count",
                    "actor_raw_action_saturation_fraction",
                    "normalized_mean_IAE_ratio",
                    "action_collapse_fraction",
                    "applied_displacement_from_baseline_mean",
                    "applied_displacement_from_baseline_max",
                    "requested_residual_norm_mean",
                    "applied_residual_norm_mean",
                    "applied_over_requested_sum_ratio",
                    "Z_mode_fraction", "Omega_mode_fraction",
                    "QP_infeasible_rate", "physical_violation_rate",
                    "robust_region_violation_rate",
                    "nonlinear_W_exceedance_steps",
                    "Gm_normal_mode_fraction", "Bj_recovery_mode_fraction",
                    "shock_ranks_and_recovery_json", "max_recovery_steps",
                    "supervisor_projection_distance_mean_physical",
                    "supervisor_low_authority_fraction",
                    "supervisor_q_radius_min", "supervisor_q_radius_median",
                    "G3_low_authority_steps", "Gm_exit_count",
                    "Bk_recovery_failure_count", "supervisor_Omega_exit_count",
                    "supervisor_QP_infeasible_count",
                )
            } for row in rows
        }, sort_keys=True),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_action_steps(path, episode, records):
    """Keep the full M2 action audit without retaining 90k steps in memory."""
    rows = []
    for second, record in enumerate(records):
        q_base = record["action_q_base"]
        center = record["action_center"]
        anchor = record["interior_anchor"]
        boundary = record["boundary_target"]
        mapped = record["mapped_action"]
        u_base = record["baseline_control"]
        applied = record["control"]
        action = record["raw_action"]
        rows.append({
            "episode": episode, "second": second,
            "scenario_type": record["scenario_type"],
            "shock_scale_lambda": record["shock_scale_lambda"],
            "safety_mode": record["safety_mode"],
            "raw_actor_a_P100": float(action[0]),
            "raw_actor_a_F200": float(action[1]),
            "rho": record["actor_rho"],
            "q_base_P100_normalized": float(q_base[0]),
            "q_base_F200_normalized": float(q_base[1]),
            "u_base_P100": float(u_base[0]),
            "u_base_F200": float(u_base[1]),
            "mapping_center_P100_normalized": float(center[0]),
            "mapping_center_F200_normalized": float(center[1]),
            "interior_anchor_P100_normalized": float(anchor[0]),
            "interior_anchor_F200_normalized": float(anchor[1]),
            "boundary_target_P100_normalized": float(boundary[0]),
            "boundary_target_F200_normalized": float(boundary[1]),
            "interior_tau_max": record["interior_tau_max"],
            "interior_chebyshev_radius": record["interior_chebyshev_radius"],
            "mapped_action_P100_normalized": float(mapped[0]),
            "mapped_action_F200_normalized": float(mapped[1]),
            "final_applied_P100": float(applied[0]),
            "final_applied_F200": float(applied[1]),
            "applied_displacement_from_baseline": record[
                "applied_displacement_from_baseline"
            ],
            "nonzero_actor": record["nonzero_actor"],
            "collapsed_action": record["collapsed_action"],
            "qp_feasible": record["qp_feasible"],
            "final_verification_gap": record["final_verification_gap"],
            "physical_violation": record["violation"],
            "robust_region_violation": record["robust_region_violation"],
            "supervisor_mode": record["supervisor_mode"],
            "supervisor_remaining_shocks": record["supervisor_remaining_shocks"],
            "supervisor_rank_before": record["supervisor_rank_before"],
            "supervisor_rank_next": record["supervisor_rank_next"],
            "supervisor_q_radius": record["supervisor_q_radius"],
            "supervisor_low_authority": record["supervisor_low_authority"],
            "supervisor_projection_distance_physical": record[
                "supervisor_projection_distance_physical"],
        })
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_fixed_trajectories(output_dir, episode, evaluation):
    rows = []
    for scenario in SCENARIOS:
        for step, record in enumerate(evaluation[scenario]["records"]):
            state = np.asarray(record["state"])
            control = np.asarray(record["control"])
            rows.append({
                "episode": episode, "scenario": scenario, "second": step,
                "X2": float(state[0]), "P2": float(state[1]),
                "P100": float(control[0]), "F200": float(control[1]),
                "safety_mode": record["safety_mode"],
                "in_Z_before": record["in_Z_before"],
                "in_Omega_before": record["in_Omega_before"],
                "reachable_rank_before": record["reachable_rank_before"],
                "requested_residual_norm": record["residual_requested_norm"],
                "applied_residual_norm": record["residual_applied_norm"],
                "applied_over_requested_step_ratio": record["residual_execution_ratio"],
                "actor_raw_a_P100": float(record["raw_action"][0]),
                "actor_raw_a_F200": float(record["raw_action"][1]),
                "actor_rho": record["actor_rho"],
                "q_base_P100_normalized": float(record["action_q_base"][0]),
                "q_base_F200_normalized": float(record["action_q_base"][1]),
                "u_base_P100": float(record["baseline_control"][0]),
                "u_base_F200": float(record["baseline_control"][1]),
                "interior_anchor_P100_normalized": float(record["interior_anchor"][0]),
                "interior_anchor_F200_normalized": float(record["interior_anchor"][1]),
                "boundary_target_P100_normalized": float(record["boundary_target"][0]),
                "boundary_target_F200_normalized": float(record["boundary_target"][1]),
                "mapped_action_P100_normalized": float(record["mapped_action"][0]),
                "mapped_action_F200_normalized": float(record["mapped_action"][1]),
                "applied_displacement_from_baseline": record[
                    "applied_displacement_from_baseline"
                ],
                "nonzero_actor": record["nonzero_actor"],
                "collapsed_action": record["collapsed_action"],
                "interior_tau_max": record["interior_tau_max"],
                "final_verification_gap": record["final_verification_gap"],
                "economic_reward": record["economic_reward"],
                "state_recovery_loss": record["state_recovery_loss"],
                "P100_move_loss": record["p100_move_loss"],
                "F200_move_loss": record["f200_move_loss"],
                "saturation_loss": record["saturation_loss"],
                "omega_exit_event": record["omega_exit_event"],
                "physical_violation": record["violation"],
                "robust_region_violation": record["robust_region_violation"],
                "qp_feasible": record["qp_feasible"],
                "execution_mask": record["execution_mask"],
                "paper_state_shock_X2": float(record["paper_state_shock"][0]),
                "paper_state_shock_P2": float(record["paper_state_shock"][1]),
                "supervisor_mode": record["supervisor_mode"],
                "supervisor_remaining_shocks": record["supervisor_remaining_shocks"],
                "supervisor_rank_before": record["supervisor_rank_before"],
                "supervisor_rank_next": record["supervisor_rank_next"],
                "supervisor_event_detected": record["supervisor_event_detected"],
                "supervisor_recovery_completed_steps": record[
                    "supervisor_recovery_completed_steps"],
                "supervisor_q_area": record["supervisor_q_area"],
                "supervisor_q_radius": record["supervisor_q_radius"],
                "supervisor_low_authority": record["supervisor_low_authority"],
                "supervisor_projection_distance_physical": record[
                    "supervisor_projection_distance_physical"],
            })
    trajectory_dir = output_dir / "fixed_trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    write_csv(trajectory_dir / f"episode_{episode:04d}.csv", rows)


def plot_curve(path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for scenario in SCENARIOS:
        subset = [r for r in rows if r["scenario"] == scenario]
        ax.plot([r["episode"] for r in subset],
                [r["paired_economic_improvement_percent"] for r in subset],
                label=scenario)
    episodes = sorted({r["episode"] for r in rows})
    mean = [np.mean([r["paired_economic_improvement_percent"] for r in rows
                     if r["episode"] == episode]) for episode in episodes]
    ax.plot(episodes, mean, color="black", linewidth=2.2, label="three-scenario mean")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set(xlabel="Episode", ylabel="Paired economic improvement [%]",
           title="Fixed deterministic Paper2016 evaluation (lambda=1)")
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def random_dwell_final_evaluation(cfg, model, design, omega, domain,
                                  dwell_sets, actor_path, output_dir,
                                  observation_variant="base19"):
    from .omega_dwell_event_supervisor import json_finite, stress_rollout
    from .residual_action_space_diagnosis import FrozenActor
    actor = FrozenActor(actor_path, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for scenario in SCENARIOS:
        for policy in ("zero_residual", "frozen_old_actor"):
            for trajectory in range(3):
                seed = cfg.seed + 970000 + 100000 * SCENARIOS.index(scenario) + trajectory
                result, steps = stress_rollout(
                    cfg, model, design, omega, domain, scenario, policy,
                    actor, dwell_sets, seed, 120, trajectory,
                    observation_variant=observation_variant,
                )
                result["policy"] = (
                    "final_checkpoint" if policy == "frozen_old_actor"
                    else "paired_zero_residual"
                )
                results.append(result)
                write_csv(output_dir / (
                    f"{scenario}_{result['policy']}_{trajectory}_steps.csv"
                ), steps)
    (output_dir / "summary.json").write_text(
        json.dumps(json_finite({
            "checkpoint": str(actor_path),
            "shock_clock_available_to_supervisor": False,
            "minimum_dwell": 20,
            "trajectories": results,
            "all_safe_and_recovered": all(
                not row["certificate_failure"]
                and row["all_three_recovered_within_D"]
                and all(row[key] == 0 for key in (
                    "G_m_exit_count", "B_k_recovery_failure_count",
                    "Omega_exit_count", "QP_infeasible_count",
                    "physical_state_violation_count",
                    "physical_input_violation_count",
                    "robust_region_violation_count",
                    "nonlinear_mismatch_outside_W_count",
                )) for row in results
            ),
        }), indent=2, allow_nan=False), encoding="utf-8"
    )
    return results


def main():
    args = parse_args()
    if args.observation_variant == "supervisor23" and args.supervisor != "dwell20":
        raise ValueError("supervisor23 observation requires --supervisor dwell20")
    if args.supervisor == "dwell20" and args.action_mapping != "interior_anchor":
        raise ValueError("--supervisor dwell20 requires --action-mapping interior_anchor")
    if args.output_dir is None:
        if args.supervisor == "dwell20":
            if args.observation_variant == "supervisor23":
                args.output_dir = REPO_DIR / "evaporation_safe_sac" / (
                    f"outputs_omega_safe_B_dwell20_interior_obs23_"
                    f"seed{args.seed}_{args.episodes}x{args.steps}_local_full"
                )
            else:
                args.output_dir = REPO_DIR / "evaporation_safe_sac" / (
                    f"outputs_omega_safe_B_dwell20_interior_seed{args.seed}_"
                    f"{args.episodes}x{args.steps}"
                )
        else:
            directory = {
                "feasible_set_normalized":
                    "outputs_omega_safe_B_feasible_normalized_seed42_300x300",
                "legacy_fixed_residual":
                    "outputs_omega_safe_B_legacy_mapping_recheck_seed42_300x300",
                "interior_anchor":
                    "outputs_omega_safe_B_interior_anchor_seed42_300x300",
            }[args.action_mapping]
            args.output_dir = REPO_DIR / "evaporation_safe_sac" / directory
    if (args.action_mapping == "interior_anchor" and not args.calibrate_only
            and any((args.output_dir / name).exists() for name in (
                "training_action_steps.csv", "training_log.csv", "run_summary.json"
            ))):
        raise RuntimeError(
            f"Interior-anchor output directory already contains a run: "
            f"{args.output_dir}. Keep the partial safety-aborted evidence "
            "and select a fresh --output-dir for any new experiment."
        )
    if args.steps < 41 or args.episodes < 1:
        raise ValueError("training must include all 0/20/40 s shocks")
    cfg = ExperimentConfig(
        benchmark_profile="zanon2016", experiment_mode="proposed",
        episodes=args.episodes, steps_per_episode=args.steps, seed=args.seed,
        residual_parameterization="state_dependent_polytope",
    )
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    cfg.omega_performance_shaping_enabled = True
    cfg.omega_exit_abort_count = 3
    cfg.omega_outside_abort_steps = 10
    # A mapping-only ablation must not proceed after leaving its certified
    # online domain. This changes monitoring, not the controller or reward.
    cfg.abort_on_first_uncertified_step = args.action_mapping == "interior_anchor"
    cfg.output_dir = args.output_dir
    set_seed(cfg.seed)
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    domain, _ = bounds_and_domains(cfg, model, design)
    cert = certificate(omega, domain, design)
    if not cert["passed"]:
        raise RuntimeError("Fixed B Omega failed its vertex invariance audit")
    if args.supervisor == "dwell20":
        baseline, calibration = frozen_supervised_baseline(
            cfg, model, design, omega, domain, args.output_dir,
            args.dwell_sets, args.frozen_reward_calibration,
            args.observation_variant,
        )
    else:
        baseline, calibration = baseline_calibration(
            cfg, model, design, omega, domain, args.output_dir,
            action_mapping=args.action_mapping,
        )
    if args.calibrate_only:
        return
    rank_sets = build_reachable_sets(design, omega, domain, 1000)
    sac_cfg = SACConfig(
        gamma=cfg.gamma_rl, tau=cfg.tau,
        actor_lr=cfg.actor_learning_rate,
        critic_lr=cfg.critic_learning_rate,
        alpha_lr=cfg.entropy_learning_rate,
        hidden_dim=cfg.hidden_dim, alpha=cfg.alpha_initial,
        alpha_min=cfg.alpha_min,
        entropy_tuning_warmup_updates=cfg.entropy_tuning_warmup_updates,
        auto_entropy_tuning=cfg.auto_entropy_tuning,
        target_entropy=cfg.target_entropy,
        actor_ema_decay=cfg.actor_ema_decay,
    )
    obs_dim = OBS_DIM + (4 if args.observation_variant == "supervisor23" else 0)
    agent = SACAgent(obs_dim, ACTION_DIM, sac_cfg, device=args.device)
    agent.zero_initialize_residual_mean()
    replay = ReplayBuffer(obs_dim, ACTION_DIM, cfg.replay_capacity, agent.device)
    train_controller = controller(
        cfg, model, design, omega, domain, rank_sets, args.action_mapping,
        args.supervisor, args.dwell_sets, args.observation_variant,
    )
    schedule = balanced_random_paper2016_schedule(cfg.episodes, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    global_step = 0
    training_rows, evaluation_rows, checkpoint_rows = [], [], []
    best_economic = None
    best_constrained = None
    nearest_tradeoff = None
    cumulative_omega_exits = 0
    cumulative_outside_steps = 0
    cumulative_fixed_eval_exits = 0
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    agent.save_actor(model_dir / "initial_diagnostic_actor.pth")
    initial_evaluation = evaluate_three(
        cfg, model, design, omega, domain, agent, rank_sets,
        action_mapping=args.action_mapping,
        supervisor=args.supervisor, dwell_sets=args.dwell_sets,
        observation_variant=args.observation_variant,
    )
    for scenario in SCENARIOS:
        initial_evaluation[scenario]["design_robust_input_upper_F200"] = (
            design.robust_input_upper[1]
        )
        evaluation_rows.append(eval_row(
            0, scenario, initial_evaluation[scenario], baseline[scenario], cfg
        ))
    write_csv(args.output_dir / "fixed_evaluation.csv", evaluation_rows)
    checkpoint_rows.append(assess_checkpoint(
        evaluation_rows[-len(SCENARIOS):], 0, cfg.warmup_steps
    ))
    write_csv(args.output_dir / "checkpoint_assessments.csv", checkpoint_rows)
    write_fixed_trajectories(args.output_dir, 0, initial_evaluation)
    plot_curve(args.output_dir / "paired_economic_improvement.png", evaluation_rows)
    if any(row["Omega_exit_count"] or row["QP_infeasible_rate"]
           for row in evaluation_rows):
        raise RuntimeError("Initial fixed evaluation left Omega or had an "
                           "infeasible QP; see fixed_evaluation.csv")
    if args.action_mapping == "interior_anchor" and any(
        float(row[key]) > 0 for row in evaluation_rows
        for key in SAFETY_METRICS
    ):
        raise RuntimeError("Interior-anchor initial evaluation has a safety "
                           "anomaly; see fixed_evaluation.csv")
    for episode in range(1, cfg.episodes + 1):
        try:
            stat, training_records, global_step = run_episode(
                cfg, model, train_controller, agent, replay, rng,
                training=True, global_step=global_step,
                paper_scenario=schedule[episode - 1],
            )
        except RuntimeError as exc:
            write_csv(args.output_dir / "training_log.csv", training_rows)
            (args.output_dir / "training_abort.json").write_text(json.dumps({
                "episode": episode, "global_step_before_episode": global_step,
                "scenario": schedule[episode - 1], "reason": str(exc),
                "cumulative_omega_exits_before_episode": cumulative_omega_exits,
                "cumulative_outside_steps_before_episode": cumulative_outside_steps,
            }, indent=2), encoding="utf-8")
            raise
        agent.update_actor_ema()
        training_rows.append({
            "episode": episode, "global_step": global_step,
            "scenario_type": schedule[episode - 1],
            "shock_scale_lambda": stat["shock_scale_lambda"],
            "actor_raw_action_saturation_fraction": float(np.mean([
                np.max(np.abs(record["raw_action"])) >= 0.99
                for record in training_records
            ])),
            "actor_raw_P100_saturation_fraction": float(np.mean([
                abs(record["raw_action"][0]) >= 0.99
                for record in training_records
            ])),
            "actor_raw_F200_saturation_fraction": float(np.mean([
                abs(record["raw_action"][1]) >= 0.99
                for record in training_records
            ])),
            **action_collapse_metrics(training_records),
            **stat,
        })
        if args.action_mapping == "interior_anchor":
            append_action_steps(
                args.output_dir / "training_action_steps.csv",
                episode, training_records,
            )
            anomaly = {
                key: stat[key] for key in (
                    "physical_state_violation_steps",
                    "physical_input_violation_steps",
                    "robust_operating_region_violation_rate",
                    "qp_infeasible_rate",
                    "disturbance_bound_exceedance_rate",
                ) if stat[key] > 0
            }
            if args.supervisor == "dwell20":
                anomaly.update({
                    key: stat[key] for key in (
                        "supervisor_Gm_exit_count",
                        "supervisor_Bk_recovery_failure_count",
                        "supervisor_omega_exit_count",
                        "supervisor_QP_infeasible_count",
                    ) if stat[key] > 0
                })
                if not stat["supervisor_all_three_recovered_within_D"]:
                    anomaly["supervisor_recovery_deadline_failure"] = 1
            if anomaly:
                write_csv(args.output_dir / "training_log.csv", training_rows)
                (args.output_dir / "training_abort.json").write_text(
                    json.dumps({
                        "episode": episode, "global_step": global_step,
                        "reason": "Interior-anchor safety anomaly",
                        "anomaly": anomaly,
                    }, indent=2), encoding="utf-8"
                )
                raise RuntimeError(f"Interior-anchor safety anomaly: {anomaly}")
        cumulative_omega_exits += stat["omega_exit_count"]
        cumulative_outside_steps += stat["omega_outside_steps"]
        if stat["omega_exit_count"]:
            print(f"WARNING episode={episode}: Omega exit events="
                  f"{stat['omega_exit_count']} cumulative={cumulative_omega_exits}; "
                  "outside-certified-domain control is uncertified", flush=True)
        if (cumulative_omega_exits >= cfg.omega_exit_abort_count
                or cumulative_outside_steps >= cfg.omega_outside_abort_steps):
            write_csv(args.output_dir / "training_log.csv", training_rows)
            (args.output_dir / "training_abort.json").write_text(json.dumps({
                "episode": episode, "global_step": global_step,
                "reason": "Frequent Omega exits across training episodes",
                "cumulative_omega_exits": cumulative_omega_exits,
                "cumulative_outside_steps": cumulative_outside_steps,
            }, indent=2), encoding="utf-8")
            raise RuntimeError("Omega training abort: frequent exits across "
                               f"episodes; count={cumulative_omega_exits}, "
                               f"outside_steps={cumulative_outside_steps}")
        if episode % cfg.evaluation_every == 0 or episode == cfg.episodes:
            evaluation = evaluate_three(
                cfg, model, design, omega, domain, agent, rank_sets,
                action_mapping=args.action_mapping,
                supervisor=args.supervisor, dwell_sets=args.dwell_sets,
                observation_variant=args.observation_variant,
            )
            for scenario in SCENARIOS:
                evaluation[scenario]["design_robust_input_upper_F200"] = (
                    design.robust_input_upper[1]
                )
                evaluation_rows.append(eval_row(
                    episode, scenario, evaluation[scenario], baseline[scenario], cfg
                ))
            latest = evaluation_rows[-len(SCENARIOS):]
            assessment = assess_checkpoint(latest, global_step, cfg.warmup_steps)
            checkpoint_rows.append(assessment)
            mean_improvement = assessment["mean_economic_improvement_percent"]
            if assessment["eligible_positive_economic_safe"]:
                if (best_economic is None or mean_improvement >
                        best_economic["mean_economic_improvement_percent"]):
                    best_economic = assessment.copy()
                    agent.save_actor(model_dir / "best_economic_diagnostic_actor.pth")
                tradeoff_key = (
                    assessment["mean_positive_performance_degradation"],
                    assessment["worst_positive_performance_degradation"],
                    -mean_improvement,
                )
                if (nearest_tradeoff is None or tradeoff_key <
                        nearest_tradeoff[0]):
                    nearest_tradeoff = (tradeoff_key, assessment.copy())
                    agent.save_actor(model_dir / "nearest_tradeoff_diagnostic_actor.pth")
                if assessment["strict_constrained_candidate"]:
                    if (best_constrained is None or tradeoff_key <
                            best_constrained[0]):
                        best_constrained = (tradeoff_key, assessment.copy())
                        agent.save_actor(
                            model_dir / "best_constrained_performance_actor.pth"
                        )
            write_csv(args.output_dir / "fixed_evaluation.csv", evaluation_rows)
            write_csv(args.output_dir / "checkpoint_assessments.csv", checkpoint_rows)
            write_fixed_trajectories(args.output_dir, episode, evaluation)
            plot_curve(args.output_dir / "paired_economic_improvement.png", evaluation_rows)
            new_exits = sum(row["Omega_exit_count"] for row in latest)
            cumulative_fixed_eval_exits += new_exits
            if new_exits:
                print(f"WARNING episode={episode}: fixed-evaluation Omega "
                      f"exits={new_exits}, cumulative="
                      f"{cumulative_fixed_eval_exits}", flush=True)
            if (cumulative_fixed_eval_exits >= cfg.omega_exit_abort_count
                    or any(row["QP_infeasible_rate"] for row in latest)):
                (args.output_dir / "training_abort.json").write_text(json.dumps({
                    "episode": episode, "global_step": global_step,
                    "reason": "Frequent fixed-evaluation Omega exits or QP failure",
                    "cumulative_fixed_eval_exits": cumulative_fixed_eval_exits,
                }, indent=2), encoding="utf-8")
                raise RuntimeError("Omega training abort: fixed evaluation "
                                   "left certified domain repeatedly or QP failed")
            if args.action_mapping == "interior_anchor":
                anomaly = {
                    row["scenario"]: {
                        key: row[key] for key in SAFETY_METRICS
                        if float(row[key]) > 0
                    } for row in latest
                }
                anomaly = {key: value for key, value in anomaly.items() if value}
                if anomaly:
                    (args.output_dir / "training_abort.json").write_text(
                        json.dumps({
                            "episode": episode, "global_step": global_step,
                            "reason": "Interior-anchor fixed-evaluation safety anomaly",
                            "anomaly": anomaly,
                        }, indent=2), encoding="utf-8"
                    )
                    raise RuntimeError("Interior-anchor fixed evaluation has "
                                       f"a safety anomaly: {anomaly}")
            print(f"episode={episode}/{cfg.episodes} step={global_step} "
                  f"fixed_mean_improvement={mean_improvement:.4f}% "
                  f"strict_candidate={assessment['strict_constrained_candidate']} "
                  f"omega_exits={stat['omega_exit_count']}", flush=True)
        if episode % cfg.save_every == 0 or episode == cfg.episodes:
            agent.save_actor(model_dir / "last_actor.pth")
            agent.save_checkpoint(model_dir / "last_checkpoint.pth")
            write_csv(args.output_dir / "training_log.csv", training_rows)
    write_csv(args.output_dir / "training_log.csv", training_rows)
    if args.supervisor == "dwell20":
        random_dwell_final_evaluation(
            cfg, model, design, omega, domain, args.dwell_sets,
            model_dir / "last_actor.pth",
            args.output_dir / "random_dwell_final_evaluation",
            args.observation_variant,
        )
    (args.output_dir / "run_summary.json").write_text(json.dumps({
        "episodes": cfg.episodes, "steps_per_episode": cfg.steps_per_episode,
        "seed": cfg.seed, "global_step": global_step,
        "action_mapping": args.action_mapping,
        "supervisor": args.supervisor,
        "dwell_set_source": str(args.dwell_sets) if args.supervisor == "dwell20" else None,
        "observation_dimension": obs_dim,
        "observation_variant": args.observation_variant,
        "raw_action_saturation_threshold": 0.99,
        "sac_hyperparameters_frozen": {
            "actor_lr": cfg.actor_learning_rate,
            "critic_lr": cfg.critic_learning_rate,
            "entropy_lr": cfg.entropy_learning_rate,
            "tau": cfg.tau,
            "batch_size": cfg.batch_size,
            "replay_capacity": cfg.replay_capacity,
            "hidden_dim": cfg.hidden_dim,
            "alpha_initial": cfg.alpha_initial,
        },
        "requested_applied_residual_definition": (
            "normalized final-input displacement from the safe zero-action "
            "center, before/after final verification QP"
            if args.action_mapping in {
                "feasible_set_normalized", "interior_anchor"
            } else
            "legacy fixed-scale requested residual and paired-filter applied residual"
        ),
        "action_collapse_thresholds": {
            "eps_a_infinity_norm": 1e-8,
            "eps_u_physical_input_l2": 1e-6,
        },
        "evaluation_every_episodes": cfg.evaluation_every,
        "fixed_scenarios": list(SCENARIOS),
        "training_schedule_counts": {
            s: schedule.count(s) for s in SCENARIOS
        },
        "shock_amplitude_distribution": {
            "uniform_probability": cfg.paper2016_random_shock_episode_probability,
            "uniform_bounds": [cfg.paper2016_training_shock_scale_lower,
                               cfg.paper2016_training_shock_scale_upper],
            "endpoint_probability": cfg.paper2016_endpoint_shock_episode_probability,
        },
        "calibration": calibration,
        "best_economic_diagnostic_checkpoint": best_economic,
        "best_constrained_performance_checkpoint": (
            best_constrained[1] if best_constrained is not None else None
        ),
        "nearest_tradeoff_diagnostic_checkpoint": (
            nearest_tradeoff[1] if nearest_tradeoff is not None else None
        ),
        "primary_checkpoint_status": (
            "strict_constrained_performance_available"
            if best_constrained is not None else
            "unavailable_no_positive_safe_checkpoint_within_performance_tolerance"
        ),
        "performance_tolerance_definition": (
            "All X2/P2 IAE/ISE and P100/F200 TV <= 110% of paired baseline; "
            "if baseline TV=0, absolute increase <=1 physical input unit."
        ),
        "cumulative_omega_exit_count": cumulative_omega_exits,
        "cumulative_omega_outside_steps": cumulative_outside_steps,
        "cumulative_fixed_eval_omega_exit_count": cumulative_fixed_eval_exits,
        "K_W_Z_S_Omega_fixed": True,
        "rank_constraint_enabled": args.supervisor == "dwell20",
        "reachable_rank_sets_computed": len(rank_sets),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
