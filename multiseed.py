"""Formal 500 x 300 x 3 Paper2016 launcher and aggregate report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


DEFAULT_SEEDS = (42, 2027, 314159)
T_975_DF2 = 4.302652729911275


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal Paper2016 nominal-exogenous, unscaled-state-shock "
            "safe-SAC comparison on three independent seeds"
        )
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--benchmark-profile",
        choices=("default", "zanon2016"),
        default="zanon2016",
    )
    parser.add_argument(
        "--experiment-mode",
        choices=("proposed", "joint_theta"),
        default="proposed",
    )
    parser.add_argument(
        "--disturbance-mode",
        choices=("iid", "piecewise_constant"),
        default="piecewise_constant",
    )
    parser.add_argument("--disturbance-hold-steps", type=int, default=50)
    parser.add_argument(
        "--residual-parameterization",
        choices=("state_dependent_box", "legacy_ray"),
        default="state_dependent_box",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaporation_safe_sac/outputs_paper2016_main_500x300"),
    )
    return parser.parse_args()


def _moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    result[window - 1:] = np.convolve(
        values, np.ones(window, dtype=float) / window, mode="valid"
    )
    return result


def _write_aggregate(root: Path, seeds: list[int]) -> None:
    rows: list[dict[str, float]] = []
    raw_reward_curves: list[np.ndarray] = []
    reward_curves: list[np.ndarray] = []
    gap_curves: list[np.ndarray] = []
    reduction_curves: list[np.ndarray] = []
    nominal_improvement_curves: list[np.ndarray] = []
    episode_axis: np.ndarray | None = None
    for seed in seeds:
        seed_dir = root / f"seed_{seed}"
        with (seed_dir / "metrics.json").open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        comparison = metrics["rl_vs_no_rl"]
        certification = metrics["final_checkpoint_certification"]
        state_dependence = metrics["sac_state_dependence"]
        rows.append({
            "seed": float(seed),
            "best_episode": float(metrics["best_evaluation_episode"]),
            "selected_policy_scale": float(metrics["selected_policy_output_scale"]),
            "training_final_return": float(metrics["training_final_return"]),
            "evaluation_return": float(metrics["evaluation"]["return"]),
            "evaluation_return_per_step": float(metrics["evaluation"]["return_per_step"]),
            "evaluation_economic_cost_mean": float(metrics["evaluation"]["economic_cost_mean"]),
            "sac_incremental_cost_reduction_percent": float(
                comparison["sac_incremental_cost_reduction_vs_theta_only_percent"]
            ),
            "holdout_sac_incremental_cost_reduction_percent": float(
                100.0
                * (
                    comparison["theta_only_no_sac_holdout"]["economic_cost_mean"]
                    - comparison["safe_sac_holdout"]["economic_cost_mean"]
                )
                / max(
                    abs(comparison["theta_only_no_sac_holdout"]["economic_cost_mean"]),
                    1e-12,
                )
            ),
            "constraint_violation_rate": float(metrics["evaluation"]["violation_rate"]),
            "holdout_constraint_violation_rate": float(
                metrics["robustness_evaluation_best_policy"]["violation_rate"]
            ),
            "robust_region_scale": float(metrics["robust_region_scale"]),
            "first_feasible_scale": float(metrics["first_feasible_scale"]),
            "minimum_certified_scale": float(
                metrics["minimum_certified_scale"]
            ),
            "maximum_certified_scale": float(
                metrics["maximum_certified_scale"]
            ),
            "last_lower_infeasible_scale": float(
                metrics["last_lower_infeasible_scale"]
            ),
            "first_upper_infeasible_scale": float(
                metrics["first_upper_infeasible_scale"]
            ),
            "robust_operating_region_violation_rate": float(
                metrics["robust_operating_region_violation_rate"]
            ),
            "holdout_robust_operating_region_violation_rate": float(
                metrics["holdout_robust_operating_region_violation_rate"]
            ),
            "qp_intervention_rate": float(metrics["evaluation"]["intervention_rate"]),
            "safe_action_mapping_rate": float(
                metrics["evaluation"]["feasible_action_mapping_rate"]
            ),
            "requested_residual_state_dependence_norm": float(np.linalg.norm(
                state_dependence["requested_residual_peak_to_peak_physical"]
            )),
            "applied_residual_state_dependence_norm": float(np.linalg.norm(
                state_dependence["applied_residual_peak_to_peak_physical"]
            )),
            "requested_residual_disturbance_estimate_dependence_norm": float(
                metrics["requested_residual_disturbance_estimate_dependence_norm"]
            ),
            "applied_residual_disturbance_estimate_dependence_norm": float(
                metrics["applied_residual_disturbance_estimate_dependence_norm"]
            ),
            "residual_feasible_scale_mean": float(
                metrics["residual_feasible_scale_mean"]
            ),
            "residual_feasible_scale_min": float(
                metrics["residual_feasible_scale_min"]
            ),
            "residual_execution_ratio_mean": float(
                metrics["residual_execution_ratio_mean"]
            ),
            "rpi_area": float(metrics["rpi_polygon_area_percent_kpa"]),
            "x_minus_z_area": float(metrics["x_minus_z_area_percent_kpa"]),
            "uncovered_final_hull_vertices": float(
                certification["uncovered_final_hull_vertices"]
            ),
            "formal_safety_certification_passed": float(
                metrics["formal_safety_certification_passed"]
            ),
            "sac_positive_increment_learned": float(
                certification["sac_positive_increment_learned"]
            ),
            "training_seconds": float(metrics["training_seconds"]),
            "initial_rpi_area": float(metrics["initial_rpi_area"]),
            "optimized_rpi_area": float(metrics["optimized_rpi_area"]),
            "rpi_area_reduction_percent": float(metrics["rpi_area_reduction_percent"]),
            "initial_x_minus_z_area": float(metrics["initial_x_minus_z_area"]),
            "optimized_x_minus_z_area": float(metrics["optimized_x_minus_z_area"]),
            "x_minus_z_area_increase_percent": float(metrics["x_minus_z_area_increase_percent"]),
            "initial_invariant_area": float(metrics["initial_invariant_area"]),
            "optimized_invariant_area": float(metrics["optimized_invariant_area"]),
            "hinf_sampled_norm": float(metrics["hinf_sampled_norm"]),
            "hinf_gamma": float(metrics["hinf_gamma"]),
            "theta_only_economic_cost_mean": float(metrics["theta_only_economic_cost_mean"]),
            "safe_sac_economic_cost_mean": float(metrics["safe_sac_economic_cost_mean"]),
            "holdout_theta_only_economic_cost_mean": float(
                metrics["holdout_theta_only_economic_cost_mean"]
            ),
            "holdout_safe_sac_economic_cost_mean": float(
                metrics["holdout_safe_sac_economic_cost_mean"]
            ),
            "safe_reference_cost": float(metrics["safe_reference_cost"]),
            "normalized_safe_performance_gap": float(metrics["normalized_safe_performance_gap"]),
            "rpi_violation_rate": float(metrics["rpi_violation_rate"]),
            "qp_infeasible_rate": float(metrics["qp_infeasible_rate"]),
            "disturbance_bound_exceedance_rate": float(
                metrics["disturbance_bound_exceedance_rate"]
            ),
            "best_nonzero_policy_episode": float(metrics["best_nonzero_policy_episode"]),
            "best_nonzero_policy_scale": float(metrics["best_nonzero_policy_scale"]),
            "best_nonzero_cost_reduction_percent": float(
                metrics["best_nonzero_cost_reduction_percent"]
            ),
            "best_nonzero_incremental_return": float(
                metrics["best_nonzero_incremental_return"]
            ),
            "best_nonzero_violation_rate": float(
                metrics["best_nonzero_violation_rate"]
            ),
            "best_nonzero_intervention_rate": float(
                metrics["best_nonzero_intervention_rate"]
            ),
            "best_nonzero_mapping_rate": float(
                metrics["best_nonzero_mapping_rate"]
            ),
        })
        with (seed_dir / "training_log.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            records = list(csv.DictReader(stream))
        episodes = np.asarray([float(row["episode"]) for row in records])
        returns = np.asarray([float(row["return"]) for row in records])
        mask = episodes >= 1.0
        if episode_axis is None:
            episode_axis = episodes[mask]
        raw_reward_curves.append(returns[mask])
        reward_curves.append(_moving_average(returns[mask]))
        gap_curves.append(np.asarray([
            float(row["evaluation_normalized_safe_performance_gap"])
            for row in records
        ])[mask])
        reduction_curves.append(np.asarray([
            float(row["evaluation_sac_incremental_cost_reduction_percent"])
            for row in records
        ])[mask])
        paper_improvement_key = (
            "evaluation_paper2016_sac_improvement_percent"
            if "evaluation_paper2016_sac_improvement_percent" in records[0]
            else "evaluation_nominal_sac_improvement_percent"
        )
        nominal_improvement_curves.append(np.asarray([
            float(row[paper_improvement_key]) for row in records
        ])[mask])

    header = list(rows[0])
    with (root / "multi_seed_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [key for key in header if key != "seed"]
    aggregate = {
        "protocol": (
            "Paper2016 original nominal exogenous conditions and unscaled "
            "state shocks; three independent trainings with paired final-theta "
            "SAC/no-SAC evaluation"
        ),
        "benchmark_profile": "zanon2016",
        "uses_rho_d_scaling": False,
        "rho_d_scope": "separate four-exogenous-disturbance applicability analysis only",
        "seeds": seeds,
        "runs": rows,
        "mean": {
            key: float(np.mean([row[key] for row in rows])) for key in numeric_keys
        },
        "sample_std": {
            key: float(np.std([row[key] for row in rows], ddof=1))
            if len(rows) > 1 else 0.0
            for key in numeric_keys
        },
        "all_final_hulls_certified": bool(all(
            row["uncovered_final_hull_vertices"] == 0.0 for row in rows
        )),
        "all_fixed_designs_certified": bool(all(
            row["formal_safety_certification_passed"] > 0.5 for row in rows
        )),
        "seeds_with_positive_sac_increment": int(sum(
            row["sac_positive_increment_learned"] > 0.5 for row in rows
        )),
        "learning_curve_confidence_interval": (
            "pointwise two-sided 95% Student-t interval of the three "
            "20-episode moving-average curves (df=2)"
        ),
    }
    with (root / "multi_seed_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(aggregate, stream, indent=2, ensure_ascii=False)

    curve_summaries = []
    for seed, values in zip(seeds, raw_reward_curves):
        window = min(20, len(values))
        early = float(np.mean(values[:window]))
        late = float(np.mean(values[-window:]))
        enough_episodes_for_trend = len(values) >= 40
        curve_summaries.append({
            "seed": int(seed),
            "first_20_mean": early,
            "last_20_mean": late,
            "absolute_improvement": late - early,
            "relative_improvement": float(
                100.0 * (late - early) / max(abs(early), 1e-12)
            ),
            "first_20_episode_mean_return": early,
            "last_20_episode_mean_return": late,
            "return_improvement_absolute": late - early,
            "return_improvement_percent": float(
                100.0 * (late - early) / max(abs(early), 1e-12)
            ),
            "training_trend_status": (
                "positive" if enough_episodes_for_trend and late > early
                else "non_positive" if enough_episodes_for_trend
                else "insufficient_episodes"
            ),
            "positive_training_trend": (
                bool(late > early) if enough_episodes_for_trend else None
            ),
            "episodes_in_window": int(window),
        })
    early_values = np.asarray([
        item["first_20_episode_mean_return"] for item in curve_summaries
    ])
    late_values = np.asarray([
        item["last_20_episode_mean_return"] for item in curve_summaries
    ])
    early_mean = float(np.mean(early_values))
    late_mean = float(np.mean(late_values))
    enough_episodes_for_trend = bool(
        curve_summaries
        and all(item["episodes_in_window"] == 20 for item in curve_summaries)
        and len(raw_reward_curves[0]) >= 40
    )
    learning_summary = {
        "seeds": seeds,
        "moving_average_episodes": 20,
        "per_seed": curve_summaries,
        "aggregate": {
            "training_trend_status": (
                "positive" if enough_episodes_for_trend and late_mean > early_mean
                else "non_positive" if enough_episodes_for_trend
                else "insufficient_episodes"
            ),
            "first_20_mean": early_mean,
            "last_20_mean": late_mean,
            "absolute_improvement": late_mean - early_mean,
            "relative_improvement": float(
                100.0 * (late_mean - early_mean) / max(abs(early_mean), 1e-12)
            ),
            "first_20_episode_mean_return": early_mean,
            "last_20_episode_mean_return": late_mean,
            "return_improvement_absolute": late_mean - early_mean,
            "return_improvement_percent": float(
                100.0 * (late_mean - early_mean) / max(abs(early_mean), 1e-12)
            ),
            "positive_training_trend": (
                bool(late_mean > early_mean)
                if enough_episodes_for_trend else None
            ),
        },
        "interpretation": (
            "Unmodified episode cumulative training return. At least 40 "
            "episodes are required before comparing disjoint first/last-20 "
            "windows; shorter runs are labelled insufficient_episodes."
        ),
    }
    with (root / "learning_curve_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(learning_summary, stream, indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = np.vstack(reward_curves)
    valid = np.all(np.isfinite(curves), axis=0)
    plot_episodes = episode_axis[valid]
    plot_curves = curves[:, valid]
    mean = np.mean(plot_curves, axis=0)
    std = np.std(plot_curves, axis=0, ddof=1)
    ci95_half_width = T_975_DF2 * std / np.sqrt(len(seeds))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for seed, raw_curve, curve in zip(seeds, raw_reward_curves, plot_curves):
        ax.plot(
            episode_axis, raw_curve, linewidth=0.45, alpha=0.16,
            color="#777777",
        )
        ax.plot(plot_episodes, curve, linewidth=0.9, alpha=0.55, label=f"seed {seed}")
    ax.plot(plot_episodes, mean, color="#2f6fb3", linewidth=2.1, label="mean")
    ax.fill_between(
        plot_episodes, mean - ci95_half_width, mean + ci95_half_width,
        color="#2f6fb3", alpha=0.16, label="pointwise 95% CI",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode cumulative training return")
    ax.set_title("Three-seed SAC learning curve")
    ax.grid(True, color="#dddddd", linewidth=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(root / "multi_seed_learning_curve.png", dpi=180)
    plt.close(fig)

    def plot_evaluation_curves(
        curves_in: list[np.ndarray], ylabel: str, title: str, filename: str
    ) -> None:
        curves_array = np.vstack(curves_in)
        valid_columns = np.all(np.isfinite(curves_array), axis=0)
        x = episode_axis[valid_columns]
        y = curves_array[:, valid_columns]
        if y.shape[1] == 0:
            figure, axis = plt.subplots(figsize=(8.4, 4.8))
            axis.set_xlabel("Episode")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.text(
                0.5, 0.5,
                "Metric undefined: J_base - J_safe_ref <= 0",
                ha="center", va="center", transform=axis.transAxes,
            )
            axis.grid(True, color="#dddddd", linewidth=0.6)
            figure.tight_layout()
            figure.savefig(root / filename, dpi=180)
            plt.close(figure)
            return
        curve_mean = np.mean(y, axis=0)
        curve_std = np.std(y, axis=0, ddof=1)
        half_width = T_975_DF2 * curve_std / np.sqrt(len(seeds))
        figure, axis = plt.subplots(figsize=(8.4, 4.8))
        for seed, curve in zip(seeds, y):
            axis.plot(x, curve, linewidth=0.9, alpha=0.45, label=f"seed {seed}")
        axis.plot(x, curve_mean, color="#2f6fb3", linewidth=2.1, label="mean")
        axis.fill_between(
            x, curve_mean - half_width, curve_mean + half_width,
            color="#2f6fb3", alpha=0.16, label="pointwise 95% CI",
        )
        axis.set_xlabel("Episode")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.legend(frameon=True)
        figure.tight_layout()
        figure.savefig(root / filename, dpi=180)
        plt.close(figure)

    plot_evaluation_curves(
        gap_curves,
        "Normalized safe performance gap",
        "Paired deterministic SAC performance gap",
        "multi_seed_sac_performance_gap.png",
    )
    plot_evaluation_curves(
        reduction_curves,
        "SAC economic improvement over H∞–RPI [%]",
        "Paper2016 unscaled state-shock SAC economic improvement",
        "multi_seed_sac_economic_improvement_curve.png",
    )
    plot_evaluation_curves(
        nominal_improvement_curves,
        "SAC economic improvement over H∞–RPI [%]",
        "Paper2016 unscaled state-shock SAC economic improvement",
        "multi_seed_paper2016_economic_improvement.png",
    )


def _metric_summary(values: list[float], seeds: list[int]) -> dict[str, object]:
    """Five-number cross-seed summary plus a two-sided Student-t 95% CI."""
    pairs = [
        (int(seed), float(value)) for seed, value in zip(seeds, values)
        if value is not None and np.isfinite(float(value))
    ]
    if not pairs:
        return {
            "n": 0, "mean": None, "sample_std": None,
            "ci95_lower": None, "ci95_upper": None,
            "ci95_half_width": None, "min": None, "max": None,
            "values_by_seed": {},
        }
    array = np.asarray([value for _, value in pairs], dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    if len(array) == 3:
        half_width = float(T_975_DF2 * std / np.sqrt(3.0))
    elif len(array) == 2:
        half_width = float(12.706204736432095 * std / np.sqrt(2.0))
    else:
        half_width = None
    return {
        "n": int(len(array)),
        "mean": mean,
        "sample_std": std,
        "ci95_lower": None if half_width is None else mean - half_width,
        "ci95_upper": None if half_width is None else mean + half_width,
        "ci95_half_width": half_width,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values_by_seed": {str(seed): value for seed, value in pairs},
    }


def _write_formal_aggregate(root: Path, seeds: list[int]) -> None:
    """Aggregate only fixed deterministic learned-policy evidence."""
    scenarios = (
        "pressure_positive", "pressure_negative", "concentration_positive"
    )
    rows: list[dict[str, object]] = []
    evaluation_curves: dict[str, list[np.ndarray]] = {}
    evaluation_episodes: np.ndarray | None = None
    g_statuses: list[str] = []

    for seed in seeds:
        seed_dir = root / f"seed_{seed}"
        with (seed_dir / "metrics.json").open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        with (seed_dir / "paper2016_control_performance_metrics.json").open(
            encoding="utf-8"
        ) as stream:
            control = json.load(stream)
        learned = metrics.get("best_post_warmup_learned_checkpoint")
        if learned is None or not bool(learned["post_warmup_eligible"]):
            raise RuntimeError(
                f"seed {seed} has no post-warmup learned checkpoint"
            )
        comparison = metrics[
            "paper2016_best_post_warmup_scenario_comparison"
        ]
        if int(comparison["episode"]) != int(learned["episode"]):
            raise RuntimeError(
                f"seed {seed} paper scenario output does not use its learned checkpoint"
            )
        if metrics["residual_action_scale_normalized"] != [0.36, 0.3]:
            raise RuntimeError(f"seed {seed} did not use frozen 3x authority")
        if not metrics["paper2016_proposed_training_reward_rpi_terms_excluded"]:
            raise RuntimeError(f"seed {seed} did not use the Paper2016 reward fix")
        if int(metrics["episodes"]) != 500 or int(metrics["steps_per_episode"]) != 300:
            raise RuntimeError(f"seed {seed} does not satisfy 500x300")

        scenario_values = comparison["scenarios"]
        improvements = [
            float(scenario_values[scenario]["economic_cost_reduction_percent"])
            for scenario in scenarios
        ]
        sac_stats = [scenario_values[scenario]["safe_sac"] for scenario in scenarios]
        row: dict[str, object] = {
            "seed": int(seed),
            "best_diagnostic_episode": int(
                metrics["best_diagnostic_nonzero_checkpoint"]["episode"]
            ),
            "best_post_warmup_episode": int(learned["episode"]),
            "best_post_warmup_global_step": int(learned["global_step"]),
            "best_post_warmup_improvement_percent": float(
                learned["economic_cost_reduction_percent"]
            ),
            "economic_improvement_percent": float(np.mean(improvements)),
            "J_econ": float(np.mean([
                stat["economic_cost_total"] for stat in sac_stats
            ])),
            "requested_residual_norm": float(np.mean([
                stat["residual_requested_norm_mean"] for stat in sac_stats
            ])),
            "applied_residual_norm": float(np.mean([
                stat["residual_applied_norm_mean"] for stat in sac_stats
            ])),
            "execution_ratio": float(np.mean([
                stat["residual_execution_ratio_mean"] for stat in sac_stats
            ])),
            "requested_policy_state_dependence": float(
                comparison["requested_residual_state_dependence_norm"]
            ),
            "applied_policy_state_dependence": float(
                comparison["applied_residual_state_dependence_norm"]
            ),
            "physical_constraint_violation_rate": float(max(
                stat["violation_rate"] for stat in sac_stats
            )),
            "robust_operating_region_violation_rate": float(max(
                stat["robust_operating_region_violation_rate"]
                for stat in sac_stats
            )),
            "qp_infeasible_rate": float(max(
                stat["qp_infeasible_rate"] for stat in sac_stats
            )),
            "disturbance_bound_exceedance_rate": float(max(
                stat["disturbance_bound_exceedance_rate"] for stat in sac_stats
            )),
            "execution_mask_mean": float(min(
                stat["execution_mask_mean"] for stat in sac_stats
            )),
            "rpi_violation_rate": float(np.mean([
                stat["rpi_violation_rate"] for stat in sac_stats
            ])),
            "rpi_utilization_peak": float(max(
                stat["rpi_utilization_peak"] for stat in sac_stats
            )),
            "formal_safety_certification_passed": bool(
                metrics["formal_safety_certification_passed"]
            ),
        }
        for scenario in scenarios:
            scenario_result = scenario_values[scenario]
            stat = scenario_result["safe_sac"]
            scenario_control = control["scenarios"][scenario]
            row[f"{scenario}_economic_improvement_percent"] = float(
                scenario_result["economic_cost_reduction_percent"]
            )
            row[f"{scenario}_J_econ"] = float(stat["economic_cost_total"])
            row[f"{scenario}_requested_residual_norm"] = float(
                stat["residual_requested_norm_mean"]
            )
            row[f"{scenario}_applied_residual_norm"] = float(
                stat["residual_applied_norm_mean"]
            )
            for state in ("X2", "P2"):
                sac_state = scenario_control["safe_sac"][state]
                baseline_state = scenario_control["zero_residual_baseline"][state]
                prefix = f"{scenario}_{state}"
                for metric_name in ("peak_deviation", "IAE", "ISE"):
                    sac_value = float(sac_state[metric_name])
                    baseline_value = float(baseline_state[metric_name])
                    row[f"{prefix}_{metric_name}"] = sac_value
                    row[f"{prefix}_{metric_name}_baseline"] = baseline_value
                    row[f"{prefix}_{metric_name}_change"] = sac_value - baseline_value
                row[f"{prefix}_settling_time_empirical"] = sac_state[
                    "settling_time_after_last_shock_seconds"
                ]
                row[f"{prefix}_settling_time_empirical_baseline"] = baseline_state[
                    "settling_time_after_last_shock_seconds"
                ]
                row[f"{prefix}_settling_time_common"] = sac_state[
                    "common_reference"
                ]["settling_time_after_last_shock_seconds"]
                row[f"{prefix}_settling_time_common_baseline"] = baseline_state[
                    "common_reference"
                ]["settling_time_after_last_shock_seconds"]
            for actuator in ("P100", "F200"):
                sac_actuator = scenario_control["safe_sac"][actuator]
                baseline_actuator = scenario_control[
                    "zero_residual_baseline"
                ][actuator]
                paired = scenario_control[
                    "paired_changes_safe_sac_minus_baseline"
                ][actuator]
                prefix = f"{scenario}_{actuator}"
                row[f"{prefix}_min"] = float(sac_actuator["min"])
                row[f"{prefix}_max"] = float(sac_actuator["max"])
                row[f"{prefix}_peak_difference"] = float(
                    paired["peak_deviation_from_paired_zero_residual_baseline"]
                )
                row[f"{prefix}_TV"] = float(sac_actuator["total_variation"])
                row[f"{prefix}_TV_baseline"] = float(
                    baseline_actuator["total_variation"]
                )
                row[f"{prefix}_TV_change"] = float(
                    paired["total_variation_change"]
                )

        with (seed_dir / "training_log.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            records = list(csv.DictReader(stream))
        eval_records = [
            record for record in records
            if np.isfinite(float(record[
                "evaluation_sac_incremental_cost_reduction_percent"
            ]))
        ]
        seed_eval_episodes = np.asarray([
            float(record["episode"]) for record in eval_records
        ])
        if evaluation_episodes is None:
            evaluation_episodes = seed_eval_episodes
        elif not np.array_equal(evaluation_episodes, seed_eval_episodes):
            raise RuntimeError("fixed-evaluation episode grids differ across seeds")
        curve_fields = {
            "economic_improvement": (
                "evaluation_sac_incremental_cost_reduction_percent"
            ),
            "J_econ": "evaluation_economic_cost_mean",
            "requested_residual": "evaluation_residual_requested_norm_mean",
            "applied_residual": "evaluation_residual_applied_norm_mean",
            "execution_ratio": "evaluation_residual_execution_ratio_mean",
            "requested_state_dependence": (
                "evaluation_requested_residual_state_dependence_norm"
            ),
            "applied_state_dependence": (
                "evaluation_applied_residual_state_dependence_norm"
            ),
        }
        for scenario in scenarios:
            curve_fields[f"{scenario}_improvement"] = (
                f"evaluation_{scenario}_improvement_percent"
            )
        for output_name, field_name in curve_fields.items():
            evaluation_curves.setdefault(output_name, []).append(np.asarray([
                float(record[field_name]) for record in eval_records
            ]))
        post_warmup = [
            record for record in eval_records
            if float(record["global_step"]) >= float(metrics["final_checkpoint_certification"]["warmup_steps"])
        ]
        post_values = np.asarray([
            float(record["evaluation_sac_incremental_cost_reduction_percent"])
            for record in post_warmup
        ])
        row["early_post_warmup_mean_improvement"] = float(
            np.mean(post_values[:5])
        )
        row["late_5_mean_improvement"] = float(np.mean(post_values[-5:]))
        row["late_10_mean_improvement"] = float(np.mean(post_values[-10:]))
        row["post_warmup_positive_fraction"] = float(np.mean(post_values > 0.0))
        row["late_5_all_positive"] = bool(np.all(post_values[-5:] > 0.0))
        rows.append(row)
        g_statuses.append(metrics["paper2016_G"]["G_status"])

    header = list(rows[0])
    with (root / "aggregate_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = [
        key for key in header
        if key != "seed"
        and not isinstance(rows[0][key], bool)
        and any(
            row[key] is not None and isinstance(row[key], (int, float))
            for row in rows
        )
    ]
    statistics = {
        key: _metric_summary([row[key] for row in rows], seeds)
        for key in numeric_keys
    }
    improvement_values = [
        float(row["economic_improvement_percent"]) for row in rows
    ]
    aggregate = {
        "protocol": {
            "benchmark_profile": "zanon2016",
            "experiment_mode": "proposed",
            "episodes": 500,
            "steps_per_episode": 300,
            "seeds": seeds,
            "residual_action_scale": [0.36, 0.30],
            "rpi_reward_fix_scope": (
                "Paper2016 proposed replay reward only; complete RPI monitoring "
                "and formal safety audit retained"
            ),
            "primary_learning_evidence": (
                "fixed deterministic three-scenario economic evaluation; raw "
                "mixed-scenario training return is not used as primary evidence"
            ),
            "learned_checkpoint_requirement": "global_step >= warmup_steps",
        },
        "runs": rows,
        "statistics": statistics,
        "improvement_direction_consistent": bool(all(
            value > 0.0 for value in improvement_values
        )),
        "all_late_5_means_positive": bool(all(
            float(row["late_5_mean_improvement"]) > 0.0 for row in rows
        )),
        "all_late_5_evaluations_positive": bool(all(
            bool(row["late_5_all_positive"]) for row in rows
        )),
        "policy_state_dependence_present_all_seeds": bool(all(
            float(row["applied_policy_state_dependence"]) > 0.0 for row in rows
        )),
        "all_physical_qp_robust_region_metrics_zero": bool(all(
            float(row["physical_constraint_violation_rate"]) == 0.0
            and float(row["robust_operating_region_violation_rate"]) == 0.0
            and float(row["qp_infeasible_rate"]) == 0.0
            and float(row["disturbance_bound_exceedance_rate"]) == 0.0
            for row in rows
        )),
        "G_status": (
            "unavailable_empc_baseline"
            if any(status == "unavailable_empc_baseline" for status in g_statuses)
            else "available"
        ),
        "G_interface": {
            "formula": "G=(P_eco-P_method)/sum(P_s)",
            "P_eco": None,
            "P_method_by_seed_and_scenario": {
                str(row["seed"]): {
                    scenario: row[f"{scenario}_J_econ"]
                    for scenario in scenarios
                }
                for row in rows
            },
            "sum_P_s": None,
            "G": None,
        },
        "settling_time_definitions_file": (
            "seed_<seed>/paper2016_control_performance_metrics.json"
        ),
    }
    for filename in ("aggregate_summary.json", "multi_seed_summary.json"):
        with (root / filename).open("w", encoding="utf-8") as stream:
            json.dump(aggregate, stream, indent=2, ensure_ascii=False)
    with (root / "aggregate_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        columns = [
            "metric", "n", "mean", "sample_std", "ci95_lower",
            "ci95_upper", "ci95_half_width", "min", "max",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for metric, summary in statistics.items():
            writer.writerow({
                "metric": metric,
                **{key: summary[key] for key in columns if key != "metric"},
            })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw_curve(axis, name: str, ylabel: str, title: str) -> None:
        data = np.vstack(evaluation_curves[name])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0, ddof=1)
        half_width = T_975_DF2 * std / np.sqrt(len(seeds))
        for seed, values in zip(seeds, data):
            axis.plot(
                evaluation_episodes, values, linewidth=0.9, alpha=0.5,
                label=f"seed {seed}",
            )
        axis.plot(
            evaluation_episodes, mean, color="#2f6fb3", linewidth=2.1,
            label="mean",
        )
        axis.fill_between(
            evaluation_episodes, mean - half_width, mean + half_width,
            color="#2f6fb3", alpha=0.16, label="95% CI",
        )
        axis.set_xlabel("Episode")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.legend(frameon=True)

    figure, axis = plt.subplots(figsize=(8.6, 4.9))
    draw_curve(
        axis, "economic_improvement", "Economic improvement [%]",
        "Three-seed fixed deterministic economic improvement",
    )
    axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(root / "three_seed_economic_improvement.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)
    for axis, scenario in zip(axes, scenarios):
        draw_curve(
            axis, f"{scenario}_improvement", "Improvement [%]",
            scenario.replace("_", " ").title(),
        )
        axis.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8)
    figure.savefig(root / "three_scenario_improvement_mean_ci.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.6, 4.9))
    draw_curve(
        axis, "J_econ", "Mean stage economic cost",
        "Three-seed fixed deterministic J_econ",
    )
    figure.tight_layout()
    figure.savefig(root / "three_seed_J_econ_learning_curve.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    draw_curve(
        axes[0], "requested_residual", "Normalized residual norm",
        "Requested residual",
    )
    draw_curve(
        axes[1], "applied_residual", "Normalized residual norm",
        "Applied residual",
    )
    figure.savefig(root / "three_seed_residual_evolution.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    draw_curve(
        axes[0], "requested_state_dependence", "Physical peak-to-peak norm",
        "Requested policy state-dependence",
    )
    draw_curve(
        axes[1], "applied_state_dependence", "Physical peak-to-peak norm",
        "Applied policy state-dependence",
    )
    figure.savefig(root / "three_seed_policy_state_dependence.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    package_dir = Path(__file__).resolve().parent
    package_name = package_dir.name
    args.output_dir = args.output_dir.resolve()
    if len(args.seeds) != 3:
        raise ValueError("The 95% confidence-interval protocol requires exactly three seeds")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain distinct integers")
    if 42 not in args.seeds:
        raise ValueError("The requested three-seed protocol must include seed 42")
    if args.benchmark_profile != "zanon2016":
        raise ValueError(
            "The formal multiseed entrypoint is reserved for the zanon2016 "
            "original-scenario comparison"
        )
    if args.episodes != 500 or args.steps != 300:
        raise ValueError(
            "The formal multiseed protocol requires exactly --episodes 500 "
            "--steps 300; use evaporation.train for short debug runs"
        )
    if tuple(args.seeds) != DEFAULT_SEEDS:
        raise ValueError(
            "The formal protocol requires --seeds 42 2027 314159 in that order"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "formal_protocol.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(
            {
                "benchmark_profile": "zanon2016",
                "episodes": 500,
                "steps_per_episode": 300,
                "seeds": list(DEFAULT_SEEDS),
                "external_conditions": "nominal",
                "state_shocks": {
                    "scenarios": [
                        "pressure_positive",
                        "pressure_negative",
                        "concentration_positive",
                    ],
                    "times_seconds": [0, 20, 40],
                    "scaling": 1.0,
                },
                "training_scenario_schedule": (
                    "seeded balanced random permutation in three-episode "
                    "blocks; scenario counts differ by at most one"
                ),
                "evaluation_horizon_seconds_per_scenario": 300,
                "residual_action_scale": [0.36, 0.30],
                "reward_fix": {
                    "scope": (
                        "zanon2016 AND proposed AND "
                        "paper2016_original_state_shocks"
                    ),
                    "excluded_from_replay_reward_only": [
                        "rpi_violation_event_penalty",
                        "rpi_excess_penalty",
                    ],
                    "rpi_monitoring_and_formal_audit_retained": True,
                },
                "learned_checkpoint_requirement": (
                    "global_step >= warmup_steps; pre-warmup checkpoints are "
                    "diagnostic only"
                ),
                "uses_rho_d_scaling": False,
                "rho_d_scope": (
                    "separate four-exogenous-disturbance robustness/"
                    "applicability analysis only"
                ),
            },
            stream,
            indent=2,
            ensure_ascii=False,
        )
    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        command = [
            sys.executable,
            "-m",
            f"{package_name}.train",
            "--episodes", str(args.episodes),
            "--steps", str(args.steps),
            "--seed", str(seed),
            "--device", str(args.device),
            "--benchmark-profile", str(args.benchmark_profile),
            "--experiment-mode", str(args.experiment_mode),
            "--disturbance-mode", str(args.disturbance_mode),
            "--disturbance-hold-steps", str(args.disturbance_hold_steps),
            "--residual-parameterization", str(args.residual_parameterization),
            "--output-dir", str(seed_dir),
        ]
        print(f"\n=== evaporator safe-SAC seed {seed} ===", flush=True)
        subprocess.run(command, check=True, cwd=package_dir.parent)
    _write_formal_aggregate(args.output_dir, list(args.seeds))
    print(f"\nThree-seed summary: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
