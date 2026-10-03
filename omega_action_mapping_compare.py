"""Compare two completed 300x300 Omega-SAC runs without training or tuning.

Use after the feasible-set-normalized run finishes. The comparison keeps all
fixed-evaluation checkpoints and exposes the Pareto trade-off; it does not
replace checkpoint selection with a largest-economic-improvement shortcut.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .controlled_invariant_error_set import bounds_and_domains, hull
from .model import EvaporatorModel
from .omega_sac_train import evaluate_three
from .paper2016_compare import SCENARIOS
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b
from .sac import SACAgent, SACConfig
from .train import ACTION_DIM, OBS_DIM


OLD_DEFAULT = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_safe_B_rewardshape_seed42_300x300_local"
)
NEW_DEFAULT = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_safe_B_feasible_normalized_seed42_300x300"
)
PERFORMANCE = ("X2_IAE", "X2_ISE", "P2_IAE", "P2_ISE", "P100_TV", "F200_TV")
SAFETY = (
    "Omega_exit_count", "Omega_outside_steps", "QP_infeasible_rate",
    "physical_violation_rate", "physical_state_violation_steps",
    "physical_input_violation_steps", "robust_region_violation_rate",
    "nonlinear_W_exceedance_steps",
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(row, key):
    value = row.get(key)
    return None if value in (None, "", "nan", "NaN") else float(value)


def checkpoints(rows, warmup_steps, steps_per_episode):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["episode"]), {})[row["scenario"]] = row
    output = []
    for episode, scenarios in sorted(grouped.items()):
        if set(scenarios) != set(SCENARIOS):
            raise RuntimeError(f"Incomplete fixed evaluation at episode {episode}")
        items = list(scenarios.values())
        improvements = [number(row, "paired_economic_improvement_percent")
                        for row in items]
        safety_passed = all(number(row, key) == 0 for row in items for key in SAFETY)
        metric_ratios = {key: [] for key in PERFORMANCE}
        degradation = []
        within_tolerance = True
        for row in items:
            for key in PERFORMANCE:
                value = number(row, key)
                baseline = number(row, "baseline_" + key)
                denominator = max(baseline, 1.0) if key.endswith("_TV") else max(
                    baseline, 1e-12
                )
                ratio = value / denominator
                metric_ratios[key].append(ratio)
                degradation.append(max(0.0, (value - baseline) / denominator))
                tolerance = (1.0 if key.endswith("_TV") and baseline < 1e-12
                             else 0.1 * baseline)
                within_tolerance &= value <= baseline + tolerance + 1e-12
        mean_econ = float(np.mean(improvements))
        raw_sats = [number(row, "actor_raw_action_saturation_fraction")
                    for row in items]
        raw_sat = (float(np.mean(raw_sats))
                   if all(value is not None for value in raw_sats) else None)
        output.append({
            "episode": episode,
            "post_warmup": bool(episode * steps_per_episode >= warmup_steps),
            "mean_economic_improvement_percent": mean_econ,
            "worst_scenario_economic_improvement_percent": min(improvements),
            "mean_X2_IAE_ratio": float(np.mean(metric_ratios["X2_IAE"])),
            "mean_P2_IAE_ratio": float(np.mean(metric_ratios["P2_IAE"])),
            "mean_X2_ISE_ratio": float(np.mean(metric_ratios["X2_ISE"])),
            "mean_P2_ISE_ratio": float(np.mean(metric_ratios["P2_ISE"])),
            "mean_P100_TV_ratio": float(np.mean(metric_ratios["P100_TV"])),
            "mean_F200_TV_ratio": float(np.mean(metric_ratios["F200_TV"])),
            "mean_robust_outer_10pct_steps": float(np.mean([
                number(row, "robust_input_saturation_steps") for row in items
            ])),
            "mean_actor_raw_action_saturation_fraction": raw_sat,
            "mean_positive_performance_degradation": float(np.mean(degradation)),
            "worst_positive_performance_degradation": float(max(degradation)),
            "safety_passed": bool(safety_passed),
            "strict_constrained_candidate": bool(
                episode * steps_per_episode >= warmup_steps and
                mean_econ > 0 and safety_passed and within_tolerance
            ),
            "scenario_metrics": {
                scenario: {key: number(scenarios[scenario], key) for key in (
                    "paired_economic_improvement_percent", "J_econ",
                    "X2_IAE", "X2_ISE", "P2_IAE", "P2_ISE",
                    "P100_TV", "F200_TV", "F200_at_robust_upper_steps",
                    "F200_longest_robust_upper_run_steps",
                    "P100_outer_10pct_steps", "F200_outer_10pct_steps",
                    "robust_input_saturation_steps",
                    "actor_raw_action_saturation_fraction",
                    "requested_residual_norm_mean", "applied_residual_norm_mean",
                    "applied_over_requested_sum_ratio",
                    "Z_mode_fraction", "Omega_mode_fraction",
                    "Omega_exit_count", "QP_infeasible_rate",
                    "physical_state_violation_steps",
                    "physical_input_violation_steps",
                    "robust_region_violation_rate",
                    "nonlinear_W_exceedance_steps",
                )} for scenario in SCENARIOS
            },
        })
    return output


def pareto(records):
    eligible = [row for row in records if row["post_warmup"] and
                row["safety_passed"]]
    if not eligible:
        return []
    vectors = np.asarray([[
        -row["mean_economic_improvement_percent"],
        -row["worst_scenario_economic_improvement_percent"],
        row["mean_X2_IAE_ratio"], row["mean_P2_IAE_ratio"],
        row["mean_X2_ISE_ratio"], row["mean_P2_ISE_ratio"],
        row["mean_P100_TV_ratio"], row["mean_F200_TV_ratio"],
        row["mean_robust_outer_10pct_steps"],
    ] for row in eligible])
    selected = []
    for index, value in enumerate(vectors):
        dominates = np.all(vectors <= value + 1e-10, axis=1) & np.any(
            vectors < value - 1e-10, axis=1
        )
        if not np.any(dominates):
            selected.append(eligible[index]["episode"])
    return selected


def selections(records):
    valid = [row for row in records if row["post_warmup"] and
             row["safety_passed"] and
             row["mean_economic_improvement_percent"] > 0]
    economic = max(valid, key=lambda row: row["mean_economic_improvement_percent"],
                   default=None)
    near = min(valid, key=lambda row: (
        row["mean_positive_performance_degradation"],
        row["worst_positive_performance_degradation"],
        -row["mean_economic_improvement_percent"],
    ), default=None)
    strict = [row for row in valid if row["strict_constrained_candidate"]]
    constrained = min(strict, key=lambda row: (
        row["mean_positive_performance_degradation"],
        row["worst_positive_performance_degradation"],
        -row["mean_economic_improvement_percent"],
    ), default=None)
    return {"final": records[-1], "best_economic": economic,
            "nearest_tradeoff": near, "strict_constrained": constrained}


def validate_protocol(old, new):
    for key in ("seed", "episodes", "steps_per_episode",
                "evaluation_every_episodes", "fixed_scenarios",
                "training_schedule_counts", "shock_amplitude_distribution"):
        if old.get(key) != new.get(key):
            raise RuntimeError(f"Runs differ beyond action mapping: {key}")
    if new.get("action_mapping") != "feasible_set_normalized":
        raise RuntimeError("New run is not feasible-set normalized mapping")
    if new.get("observation_dimension") != OBS_DIM:
        raise RuntimeError("New run changed observation dimension")
    old_weights = old["calibration"]["calibrated_weights"]
    new_weights = new["calibration"]["calibrated_weights"]
    for key, old_value in old_weights.items():
        if not np.isclose(new_weights[key], old_value, atol=1e-9, rtol=0):
            raise RuntimeError(f"Reward calibration changed: {key}")


def replay_old_last_actor(old_dir, old_final_rows, design_path, omega_path):
    """Recover final old raw-action saturation absent from the old CSV."""
    actor_path = old_dir / "models/last_actor.pth"
    if not actor_path.is_file():
        return {"status": "unavailable_old_last_actor_missing"}
    import torch
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed",
                           seed=42, steps_per_episode=300,
                           residual_parameterization="state_dependent_polytope")
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    cfg.omega_performance_shaping_enabled = True
    model = EvaporatorModel(cfg)
    design = load_fixed_b(design_path, cfg, model)
    omega = hull(np.loadtxt(omega_path, delimiter=",", skiprows=1))
    domain, _ = bounds_and_domains(cfg, model, design)
    agent = SACAgent(OBS_DIM, ACTION_DIM,
                     SACConfig(hidden_dim=cfg.hidden_dim), device="cpu")
    try:
        payload = torch.load(actor_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(actor_path, map_location="cpu")
    if payload.get("obs_dim") != OBS_DIM or payload.get("action_dim") != ACTION_DIM:
        raise RuntimeError("Old actor dimensions differ from current protocol")
    agent.actor.load_state_dict(payload["actor"])
    agent.policy_output_scale = float(payload.get("policy_output_scale", 1.0))
    evaluation = evaluate_three(
        cfg, model, design, omega, domain, agent,
        action_mapping="legacy_fixed_residual",
    )
    result = {}
    for scenario in SCENARIOS:
        records = evaluation[scenario]["records"]
        J = evaluation[scenario]["stat"]["J_econ"]
        stored_J = number(old_final_rows[scenario], "J_econ")
        if abs(J - stored_J) > 1e-4:
            raise RuntimeError(
                f"Old actor replay differs from saved final J_econ: {scenario}, "
                f"replayed={J}, stored={stored_J}"
            )
        actions = np.asarray([row["raw_action"] for row in records])
        result[scenario] = {
            "J_econ_replay_verified": J,
            "actor_raw_action_saturation_fraction": float(np.mean(
                np.max(np.abs(actions), axis=1) >= 0.99
            )),
            "actor_raw_P100_saturation_fraction": float(np.mean(
                np.abs(actions[:, 0]) >= 0.99
            )),
            "actor_raw_F200_saturation_fraction": float(np.mean(
                np.abs(actions[:, 1]) >= 0.99
            )),
        }
    return {"status": "replayed_and_J_econ_verified", "scenarios": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, default=OLD_DEFAULT)
    parser.add_argument("--new-dir", type=Path, default=NEW_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_controlled_invariant_error_B/"
        "Omega_anchor_B_vertices.csv"
    ))
    parser.add_argument("--skip-old-actor-replay", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.new_dir / "mapping_ablation_comparison"
    old_summary = json.loads((args.old_dir / "run_summary.json").read_text(
        encoding="utf-8"
    ))
    new_summary = json.loads((args.new_dir / "run_summary.json").read_text(
        encoding="utf-8"
    ))
    validate_protocol(old_summary, new_summary)
    old_rows = read_csv(args.old_dir / "fixed_evaluation.csv")
    new_rows = read_csv(args.new_dir / "fixed_evaluation.csv")
    old = checkpoints(old_rows, 5000, old_summary["steps_per_episode"])
    new = checkpoints(new_rows, 5000, new_summary["steps_per_episode"])
    old_by_episode = {row["episode"]: row for row in old}
    new_by_episode = {row["episode"]: row for row in new}
    if set(old_by_episode) != set(new_by_episode):
        raise RuntimeError("Fixed-evaluation episode schedules do not match")
    old_front, new_front = set(pareto(old)), set(pareto(new))
    curve = []
    for episode in sorted(old_by_episode):
        first, second = old_by_episode[episode], new_by_episode[episode]
        curve.append({
            "episode": episode,
            "old_mean_economic_improvement_percent": first[
                "mean_economic_improvement_percent"
            ],
            "new_mean_economic_improvement_percent": second[
                "mean_economic_improvement_percent"
            ],
            "delta_mean_economic_improvement_percent": (
                second["mean_economic_improvement_percent"] -
                first["mean_economic_improvement_percent"]
            ),
            "old_worst_scenario_economic_improvement_percent": first[
                "worst_scenario_economic_improvement_percent"
            ],
            "new_worst_scenario_economic_improvement_percent": second[
                "worst_scenario_economic_improvement_percent"
            ],
            "old_strict_constrained_candidate": first[
                "strict_constrained_candidate"
            ],
            "new_strict_constrained_candidate": second[
                "strict_constrained_candidate"
            ],
            "old_pareto": episode in old_front,
            "new_pareto": episode in new_front,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "matched_fixed_economic_curve.csv", curve)
    for label, records, front in (("old", old, old_front), ("new", new, new_front)):
        write_csv(output_dir / f"{label}_checkpoint_pareto.csv", [{
            key: value for key, value in row.items() if key != "scenario_metrics"
        } | {"pareto_nondominated": row["episode"] in front}
            for row in records])
    old_final = {row["scenario"]: row for row in old_rows
                 if int(row["episode"]) == old[-1]["episode"]}
    old_replay = ({"status": "not_requested"} if args.skip_old_actor_replay else
                  replay_old_last_actor(args.old_dir, old_final,
                                        args.design, args.omega_vertices))
    if old_replay["status"] == "replayed_and_J_econ_verified":
        for scenario in SCENARIOS:
            old[-1]["scenario_metrics"][scenario][
                "actor_raw_action_saturation_fraction"
            ] = old_replay["scenarios"][scenario][
                "actor_raw_action_saturation_fraction"
            ]
    final_comparison = {}
    for scenario in SCENARIOS:
        old_metrics = old[-1]["scenario_metrics"][scenario]
        new_metrics = new[-1]["scenario_metrics"][scenario]
        final_comparison[scenario] = {
            "old": old_metrics,
            "new": new_metrics,
            "new_minus_old": {
                key: new_metrics[key] - value
                for key, value in old_metrics.items()
                if value is not None and new_metrics.get(key) is not None
            },
        }
    result = {
        "protocol_validation": "same seed/length/schedule/shocks/calibrated reward/19D new observation",
        "old_mapping": "legacy_fixed_residual",
        "new_mapping": "feasible_set_normalized",
        "old_checkpoint_selections": selections(old),
        "new_checkpoint_selections": selections(new),
        "old_pareto_episodes": sorted(old_front),
        "new_pareto_episodes": sorted(new_front),
        "old_final_actor_raw_action_replay": old_replay,
        "old_fixed_curve_raw_action_saturation_status": (
            "unavailable_not_logged_in_original_300x300_run"
        ),
        "requested_applied_ratio_note": (
            "New mapping requests a fraction of full safe ray authority; "
            "old mapping requests a fixed-scale physical residual. Compare "
            "closed-loop metrics and raw actor saturation directly, not raw "
            "requested residual magnitude across mappings."
        ),
        "final_scenario_comparison": final_comparison,
    }
    path = output_dir / "mapping_ablation_comparison.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False),
                    encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
