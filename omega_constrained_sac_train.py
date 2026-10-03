"""Paired-baseline constrained SAC on frozen balanced-B dwell20/23D control.

Run locally for the full 300x300 experiment. This module changes only the
training objective/replay; it reuses the existing controller, supervisor,
interior-anchor mapping, network and deterministic evaluation protocol.
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
from .omega_constrained_objective import (
    KEYS, episode_violations, limits_from_baseline_records,
    limits_from_fixed_row, paired_step_components, physical_robust_ranges,
    violations_from_fixed_row,
)
from .omega_constrained_replay import ConstrainedReplayBuffer, DualState
from .omega_dwell_shock_sets import DEFAULT_OUTPUT as DWELL_SETS
from .omega_sac_train import (
    SAFETY_METRICS, action_collapse_metrics, assess_checkpoint, controller,
    eval_row, evaluate_three,
    frozen_supervised_baseline, plot_curve, random_dwell_final_evaluation,
    write_csv, write_fixed_trajectories,
)
from .paper2016_compare import SCENARIOS
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b
from .sac import SACAgent, SACConfig, set_seed
from .train import (
    ACTION_DIM, OBS_DIM, ZeroResidualPolicy,
    balanced_random_paper2016_schedule, run_episode,
    sample_paper2016_training_shock_scale,
)


DEFAULT_OUTPUT = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_safe_B_dwell20_obs23_constrained_seed42_300x300"
)
DEFAULT_AUDIT = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_constraint_audit_obs23/"
    "all_checkpoint_constraints.csv"
)
DEFAULT_REWARD_CALIBRATION = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_safe_B_interior_anchor_seed42_300x300/"
    "baseline_reward_calibration.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dual-lr", type=float, default=1.0)
    parser.add_argument("--lambda-max", type=float, default=1000.0)
    parser.add_argument("--constraint-audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--frozen-reward-calibration", type=Path,
                        default=DEFAULT_REWARD_CALIBRATION)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_controlled_invariant_error_B/"
        "Omega_anchor_B_vertices.csv"))
    parser.add_argument("--dwell-sets", type=Path, default=DWELL_SETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-random-dwell", action="store_true",
                        help="Only for short smoke checks; full run evaluates dwell.")
    return parser.parse_args()


def fixed_dual_learning_rates(audit_path: Path, base_rate: float):
    if base_rate <= 0 or not np.isfinite(base_rate):
        raise ValueError("--dual-lr must be positive and finite")
    with audit_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 183:
        raise RuntimeError("Dual preconditioning requires all 183 offline audit rows")
    positive_p90 = np.array([
        float(np.quantile([max(0.0, float(row[f"g_{key}"]))
                           for row in rows], 0.90)) for key in KEYS
    ])
    # Fixed before learning, solely from the preceding offline audit. This
    # keeps the user-defined raw g equations intact while making the five
    # componentwise dual step sizes numerically comparable.
    scales = np.maximum(positive_p90, 1e-4)
    return base_rate / scales, positive_p90


def add_fixed_constraint_fields(rows, ranges, steps):
    for row in rows:
        limits = limits_from_fixed_row(row, ranges, steps)
        g = violations_from_fixed_row(row, limits)
        row.update({f"g_{key}": float(value) for key, value in zip(KEYS, g)})
        row["five_constraints_feasible"] = bool(np.all(g <= 1e-10))
    return rows


def check_safety(stat, records):
    keys = (
        "supervisor_Gm_exit_count", "supervisor_Bk_recovery_failure_count",
        "supervisor_omega_exit_count", "supervisor_QP_infeasible_count",
        "physical_state_violation_steps", "physical_input_violation_steps",
        "robust_operating_region_violation_rate", "qp_infeasible_rate",
        "disturbance_bound_exceedance_rate",
    )
    anomaly = {key: stat[key] for key in keys if float(stat[key]) > 0}
    if not stat["supervisor_all_three_recovered_within_D"]:
        anomaly["recovery_deadline_failure"] = True
    if anomaly:
        raise RuntimeError(f"Constrained training safety anomaly: {anomaly}")
    if any(not r["qp_feasible"] for r in records):
        raise RuntimeError("Constrained training encountered infeasible QP")


def main():
    args = parse_args()
    if args.steps < 41 or args.episodes < 1:
        raise ValueError("Need >=41 steps and >=1 episode")
    if any((args.output_dir / name).exists() for name in (
        "training_log.csv", "fixed_evaluation.csv", "run_summary.json",
    )):
        raise RuntimeError(
            f"Output directory contains a run: {args.output_dir}; use a fresh --output-dir"
        )
    cfg = ExperimentConfig(
        benchmark_profile="zanon2016", experiment_mode="proposed",
        episodes=args.episodes, steps_per_episode=args.steps, seed=args.seed,
        residual_parameterization="state_dependent_polytope",
    )
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    cfg.omega_performance_shaping_enabled = True
    cfg.abort_on_first_uncertified_step = True
    cfg.omega_exit_abort_count = 3
    cfg.omega_outside_abort_steps = 10
    cfg.output_dir = args.output_dir
    if args.steps != cfg.paper2016_simulation_seconds:
        # Smoke-only runs pair on their actual horizon. The formal 300-step
        # protocol is unchanged when --steps=300.
        cfg.paper2016_simulation_seconds = args.steps
    set_seed(cfg.seed)
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    domain, _ = bounds_and_domains(cfg, model, design)
    if not certificate(omega, domain, design)["passed"]:
        raise RuntimeError("Fixed B Omega vertex certificate failed")
    baseline_fixed, calibration = frozen_supervised_baseline(
        cfg, model, design, omega, domain, args.output_dir,
        args.dwell_sets, args.frozen_reward_calibration,
        observation_variant="supervisor23",
    )
    ranges = physical_robust_ranges(cfg, design)
    dual_rates, audit_p90 = fixed_dual_learning_rates(
        args.constraint_audit_csv, args.dual_lr
    )
    dual = DualState.create(dual_rates, args.lambda_max)
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
    agent = SACAgent(OBS_DIM + 4, ACTION_DIM, sac_cfg, device=args.device)
    agent.zero_initialize_residual_mean()
    replay = ConstrainedReplayBuffer(
        OBS_DIM + 4, ACTION_DIM, cfg.replay_capacity, agent.device, dual
    )
    train_controller = controller(
        cfg, model, design, omega, domain,
        action_mapping="interior_anchor", supervisor="dwell20",
        dwell_sets=args.dwell_sets, observation_variant="supervisor23",
    )
    schedule = balanced_random_paper2016_schedule(cfg.episodes, cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    global_step = 0
    training_rows, evaluation_rows, checkpoint_rows = [], [], []
    endpoint_cache = {}
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    agent.save_actor(model_dir / "initial_diagnostic_actor.pth")
    best_economic = None
    best_joint = None

    def fixed_evaluation(episode):
        evaluation = evaluate_three(
            cfg, model, design, omega, domain, agent,
            action_mapping="interior_anchor", supervisor="dwell20",
            dwell_sets=args.dwell_sets, observation_variant="supervisor23",
        )
        latest = []
        for scenario in SCENARIOS:
            evaluation[scenario]["design_robust_input_upper_F200"] = (
                design.robust_input_upper[1]
            )
            row = eval_row(
                episode, scenario, evaluation[scenario],
                baseline_fixed[scenario], cfg
            )
            row["legacy_fixed_weight_return_diagnostic"] = row.pop(
                "total_evaluation_return", None
            )
            fixed_records = evaluation[scenario]["records"]
            baseline_records = baseline_fixed[scenario]["records"]
            limits = limits_from_baseline_records(
                baseline_records, cfg, model, design
            )
            component_sum = np.zeros(6, dtype=float)
            for step, record in enumerate(fixed_records):
                component_sum += paired_step_components(
                    step, record["state"], record["control"],
                    record["economic_cost"],
                    fixed_records[step - 1]["control"] if step > 0 else None,
                    baseline_records, limits, cfg, model, design,
                )
            row["objective_evaluation_return"] = float(
                component_sum[0] - np.dot(dual.values, component_sum[1:])
            )
            row["paired_economic_reward_total"] = float(component_sum[0])
            if np.max(np.abs(
                episode_violations(component_sum, limits)
                - violations_from_fixed_row(row, limits)
            )) > 1e-5:
                raise RuntimeError("Fixed-evaluation component sum disagrees with metrics")
            latest.append(row)
        add_fixed_constraint_fields(latest, ranges, cfg.paper2016_simulation_seconds)
        anomaly = {row["scenario"]: {
            key: row[key] for key in SAFETY_METRICS if float(row[key]) > 0
        } for row in latest}
        anomaly = {key: value for key, value in anomaly.items() if value}
        if anomaly:
            raise RuntimeError(f"Constrained fixed evaluation safety anomaly: {anomaly}")
        evaluation_rows.extend(latest)
        assessment = assess_checkpoint(latest, global_step, cfg.warmup_steps)
        assessment["all_five_constraints_feasible"] = all(
            row["five_constraints_feasible"] for row in latest
        )
        assessment["strict_joint_candidate"] = bool(
            assessment["strict_constrained_candidate"]
            and assessment["all_five_constraints_feasible"]
            and assessment["worst_scenario_economic_improvement_percent"] > 0
        )
        checkpoint_rows.append(assessment)
        write_csv(args.output_dir / "fixed_evaluation.csv", evaluation_rows)
        write_csv(args.output_dir / "checkpoint_assessments.csv", checkpoint_rows)
        write_fixed_trajectories(args.output_dir, episode, evaluation)
        plot_curve(args.output_dir / "paired_economic_improvement.png", evaluation_rows)
        return assessment

    fixed_evaluation(0)
    for episode in range(1, cfg.episodes + 1):
        scenario = schedule[episode - 1]
        shock_scale = sample_paper2016_training_shock_scale(cfg, rng)
        cache_key = scenario if shock_scale == 1.0 else None
        try:
            if cache_key is not None and cache_key in endpoint_cache:
                baseline_stat, baseline_records = endpoint_cache[cache_key]
            else:
                baseline_stat, baseline_records, _ = run_episode(
                    cfg, model, controller(
                        cfg, model, design, omega, domain,
                        action_mapping="interior_anchor", supervisor="dwell20",
                        dwell_sets=args.dwell_sets,
                        observation_variant="supervisor23",
                    ),
                    ZeroResidualPolicy(), None,
                    np.random.default_rng(cfg.seed + 600000 + episode),
                    training=False, global_step=0, paper_scenario=scenario,
                    paper_shock_scale_override=shock_scale,
                )
                check_safety(baseline_stat, baseline_records)
                if cache_key is not None:
                    endpoint_cache[cache_key] = (baseline_stat, baseline_records)
            limits = limits_from_baseline_records(
                baseline_records, cfg, model, design
            )
            before = dual.values.copy()
            stat, records, global_step = run_episode(
                cfg, model, train_controller, agent, replay, rng,
                training=True, global_step=global_step,
                paper_scenario=scenario,
                paper_shock_scale_override=shock_scale,
                paired_baseline_records=baseline_records,
                paired_limits=limits, constrained_dual=dual,
            )
            check_safety(stat, records)
            if len(records) != len(baseline_records):
                raise RuntimeError("Paired trajectories have different lengths")
            initial_difference = float(np.max(np.abs(
                np.asarray(records[0]["state"])
                - np.asarray(baseline_records[0]["state"])
            )))
            disturbance_difference = max(float(np.max(np.abs(
                np.asarray(current["disturbance"])
                - np.asarray(reference["disturbance"])
            ))) for current, reference in zip(records, baseline_records))
            if initial_difference > 1e-10 or disturbance_difference > 1e-10:
                raise RuntimeError("Paired initial state or exogenous disturbance mismatch")
            shock_difference = max(float(np.max(np.abs(
                np.asarray(current["paper_state_shock"])
                - np.asarray(reference["paper_state_shock"])
            ))) for current, reference in zip(records, baseline_records))
            if shock_difference > 1e-10:
                raise RuntimeError("Paired state-shock sequence mismatch")
            for index in (0, 20, 40):
                if not np.allclose(records[index]["paper_state_shock"],
                                   baseline_records[index]["paper_state_shock"],
                                   atol=1e-10):
                    raise RuntimeError("Paired state-shock sequence mismatch")
            if abs(float(stat["shock_scale_lambda"]) - shock_scale) > 1e-12:
                raise RuntimeError("Training shock scale differs from paired baseline")
            agent.update_actor_ema()
            components = np.asarray(stat["paired_component_totals"], dtype=float)
            states = np.asarray([record["state"] for record in records])
            controls = np.asarray([record["control"] for record in records])
            robust_lower = model.physical_input(design.robust_input_lower)
            robust_upper = model.physical_input(design.robust_input_upper)
            robust_mid = 0.5 * (robust_lower + robust_upper)
            robust_half = 0.5 * (robust_upper - robust_lower)
            occupied_steps = int(np.sum(np.any(
                np.abs(controls - robust_mid) / robust_half > 0.9,
                axis=1,
            )))
            realized = np.array([
                np.sum(np.abs(states[40:, 0] - cfg.robust_economic_reference_state[0]))
                / max(limits.baseline_x2_iae, 1e-12),
                np.sum(np.abs(states[40:, 1] - cfg.robust_economic_reference_state[1]))
                / max(limits.baseline_p2_iae, 1e-12),
                (np.sum(np.abs(np.diff(controls[:, 0]))) - limits.baseline_p100_tv)
                / (limits.steps * limits.robust_p100_range),
                (np.sum(np.abs(np.diff(controls[:, 1]))) - limits.baseline_f200_tv)
                / (limits.steps * limits.robust_f200_range),
                (occupied_steps - limits.baseline_boundary_steps) / limits.steps,
            ])
            metric_identity_error = float(np.max(np.abs(
                components[1:] - realized
            )))
            if metric_identity_error > 1e-5:
                raise RuntimeError(
                    f"Replay component sum disagrees with episode metrics: "
                    f"{metric_identity_error}"
                )
            g = episode_violations(components, limits)
            after = dual.update(g)
            row = {
                "episode": episode, "global_step": global_step,
                "scenario_type": scenario, "shock_scale_lambda": shock_scale,
                "paired_baseline_J_econ": baseline_stat["J_econ"],
                "J_econ": stat["J_econ"],
                "paired_economic_improvement_percent": 100.0 * (
                    baseline_stat["J_econ"] - stat["J_econ"]
                ) / max(abs(baseline_stat["J_econ"]), 1e-12),
                "paired_economic_reward_total": float(components[0]),
                "paired_initial_state_max_difference": initial_difference,
                "paired_exogenous_disturbance_max_difference": disturbance_difference,
                "paired_state_shock_max_difference": shock_difference,
                "paired_metric_identity_max_error": metric_identity_error,
                **{f"component_{key}_total": float(value)
                   for key, value in zip(KEYS, components[1:])},
                **{f"g_{key}": float(value) for key, value in zip(KEYS, g)},
                **{f"lambda_{key}_before": float(value)
                   for key, value in zip(KEYS, before)},
                **{f"lambda_{key}_after": float(value)
                   for key, value in zip(KEYS, after)},
                "constraints_feasible": bool(np.all(g <= 1e-10)),
                "actor_raw_action_saturation_fraction": float(np.mean([
                    np.max(np.abs(record["raw_action"])) >= 0.99
                    for record in records
                ])),
                **action_collapse_metrics(records),
                **{key: value for key, value in stat.items()
                   if key not in ("paired_component_totals",)},
            }
            training_rows.append(row)
            if episode % cfg.evaluation_every == 0 or episode == cfg.episodes:
                assessment = fixed_evaluation(episode)
                economic = assessment["mean_economic_improvement_percent"]
                if assessment["eligible_positive_economic_safe"]:
                    if best_economic is None or economic > best_economic[
                        "mean_economic_improvement_percent"
                    ]:
                        best_economic = assessment.copy()
                        agent.save_actor(model_dir / "best_economic_diagnostic_actor.pth")
                    if assessment["strict_joint_candidate"] and (
                        best_joint is None or economic > best_joint[
                            "mean_economic_improvement_percent"
                        ]
                    ):
                        best_joint = assessment.copy()
                        agent.save_actor(model_dir / "best_joint_candidate_actor.pth")
                print(
                    f"episode={episode}/{cfg.episodes} "
                    f"fixed_mean_improvement={economic:.4f}% "
                    f"strict_joint_candidate={assessment['strict_joint_candidate']}",
                    flush=True,
                )
            if episode % cfg.save_every == 0 or episode == cfg.episodes:
                agent.save_actor(model_dir / "last_actor.pth")
                agent.save_checkpoint(model_dir / "last_checkpoint.pth")
                write_csv(args.output_dir / "training_log.csv", training_rows)
        except Exception as exc:
            if training_rows:
                write_csv(args.output_dir / "training_log.csv", training_rows)
            (args.output_dir / "training_abort.json").write_text(
                json.dumps({
                    "episode": episode, "global_step": global_step,
                    "scenario": scenario, "shock_scale_lambda": shock_scale,
                    "reason": str(exc),
                }, indent=2), encoding="utf-8"
            )
            raise
    write_csv(args.output_dir / "training_log.csv", training_rows)
    if not args.skip_random_dwell:
        random_dwell_final_evaluation(
            cfg, model, design, omega, domain, args.dwell_sets,
            model_dir / "last_actor.pth",
            args.output_dir / "random_dwell_final_evaluation",
            observation_variant="supervisor23",
        )
    (args.output_dir / "run_summary.json").write_text(
        json.dumps({
            "episodes": cfg.episodes, "steps_per_episode": cfg.steps_per_episode,
            "seed": cfg.seed, "global_step": global_step,
            "observation_dimension": OBS_DIM + 4,
            "controller_and_supervisor_frozen": True,
            "objective": "paired_economic_minus_current_dual_times_five_components",
            "economic_scale": cfg.reward_cost_scale,
            "constraint_keys": KEYS,
            "dual_lr_base": args.dual_lr,
            "dual_learning_rates": dual_rates.tolist(),
            "offline_positive_g_p90": audit_p90.tolist(),
            "lambda_max": args.lambda_max,
            "final_lambdas": dual.values.tolist(),
            "best_economic_diagnostic_checkpoint": best_economic,
            "best_joint_candidate_checkpoint": best_joint,
            "strict_joint_checkpoint_count": sum(
                r["strict_joint_candidate"] for r in checkpoint_rows
            ),
            "paired_shock_protocol": "same initial state/scenario/lambda/0-20-40 shocks",
            "fixed_evaluation_every_episodes": cfg.evaluation_every,
            "formal_random_dwell_test_included": not args.skip_random_dwell,
        }, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
