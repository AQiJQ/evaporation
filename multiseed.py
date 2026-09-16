"""Sequential three-seed launcher and aggregate report for evaporator safe-SAC."""
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
        description="Train evaporator safe-SAC on three independent seeds"
    )
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--experiment-mode",
        choices=("proposed", "joint_theta"),
        default="proposed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "evaporation_safe_sac/outputs_state_dependent_safe_sac_300x2000"
        ),
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
    reward_curves: list[np.ndarray] = []
    gap_curves: list[np.ndarray] = []
    reduction_curves: list[np.ndarray] = []
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
            "rpi_area": float(metrics["rpi_polygon_area_percent_kpa"]),
            "x_minus_z_area": float(metrics["x_minus_z_area_percent_kpa"]),
            "uncovered_final_hull_vertices": float(
                certification["uncovered_final_hull_vertices"]
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
            "theta_only_economic_cost_mean": float(metrics["theta_only_economic_cost_mean"]),
            "safe_sac_economic_cost_mean": float(metrics["safe_sac_economic_cost_mean"]),
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
        reward_curves.append(_moving_average(returns[mask]))
        gap_curves.append(np.asarray([
            float(row["evaluation_normalized_safe_performance_gap"])
            for row in records
        ])[mask])
        reduction_curves.append(np.asarray([
            float(row["evaluation_sac_incremental_cost_reduction_percent"])
            for row in records
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
        "protocol": "three independent trainings; paired final-theta SAC/no-SAC evaluation",
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
    for seed, curve in zip(seeds, plot_curves):
        ax.plot(plot_episodes, curve, linewidth=0.9, alpha=0.45, label=f"seed {seed}")
    ax.plot(plot_episodes, mean, color="#2f6fb3", linewidth=2.1, label="mean")
    ax.fill_between(
        plot_episodes, mean - ci95_half_width, mean + ci95_half_width,
        color="#2f6fb3", alpha=0.16, label="pointwise 95% CI",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("20-episode moving-average reward")
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
        "SAC incremental economic cost reduction [%]",
        "Paired deterministic SAC cost reduction",
        "multi_seed_sac_cost_reduction.png",
    )


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
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
            "--experiment-mode", str(args.experiment_mode),
            "--output-dir", str(seed_dir),
        ]
        print(f"\n=== evaporator safe-SAC seed {seed} ===", flush=True)
        subprocess.run(command, check=True, cwd=package_dir.parent)
    _write_aggregate(args.output_dir, list(args.seeds))
    print(f"\nThree-seed summary: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
