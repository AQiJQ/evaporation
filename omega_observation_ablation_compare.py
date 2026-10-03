"""Paired fixed-evaluation comparison of 19D and supervisor-aware 23D runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .residual_action_space_diagnosis import REPO_DIR


DEFAULT_19D = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_safe_B_dwell20_interior_seed42_300x300_local"
)
DEFAULT_23D = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_safe_B_dwell20_interior_obs23_seed42_300x300_local_full"
)


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summary(run):
    fixed = csv_rows(run / "fixed_evaluation.csv")
    training = csv_rows(run / "training_log.csv")
    checkpoints = csv_rows(run / "checkpoint_assessments.csv")
    if int(training[-1]["episode"]) != 300 or len(fixed) != 183:
        raise RuntimeError(f"Run is incomplete: {run}")
    safety = (
        "Gm_exit_count", "Bk_recovery_failure_count", "supervisor_Omega_exit_count",
        "supervisor_QP_infeasible_count", "physical_state_violation_steps",
        "physical_input_violation_steps", "robust_region_violation_rate",
        "QP_infeasible_rate", "nonlinear_W_exceedance_steps",
        "recovery_deadline_failure_count",
    )
    metrics = (
        "paired_economic_improvement_percent", "normalized_mean_IAE_ratio",
        "X2_IAE", "X2_ISE", "P2_IAE", "P2_ISE", "P100_TV", "F200_TV",
        "robust_input_saturation_steps", "F200_at_robust_upper_steps",
        "action_collapse_fraction", "supervisor_projection_distance_mean_physical",
        "max_recovery_steps", "supervisor_low_authority_fraction",
    )
    result = {
        "run_directory": str(run),
        "positive_safe_checkpoints": sum(
            row["eligible_positive_economic_safe"] == "True" for row in checkpoints
        ),
        "strict_constrained_checkpoints": sum(
            row["strict_constrained_candidate"] == "True" for row in checkpoints
        ),
        "fixed_evaluation_safety_sums": {
            key: float(sum(float(row[key] or 0) for row in fixed)) for key in safety
        },
        "scenarios": {},
    }
    for scenario in sorted({row["scenario"] for row in fixed}):
        subset = [row for row in fixed if row["scenario"] == scenario]
        late = [row for row in subset if int(row["episode"]) >= 255]
        final = subset[-1]
        result["scenarios"][scenario] = {
            "episode_300": {key: float(final[key] or 0) for key in metrics},
            "late_255_300_mean": {
                key: float(np.mean([float(row[key] or 0) for row in late]))
                for key in metrics
            },
            "episode_300_IAE_ratios": {
                "X2": float(final["X2_IAE"]) / float(final["baseline_X2_IAE"]),
                "P2": float(final["P2_IAE"]) / float(final["baseline_P2_IAE"]),
            },
            "episode_300_common_settling_seconds": {
                key: (float(final[key]) if final[key] else None)
                for key in (
                    "X2_common_reference_settling_seconds",
                    "P2_common_reference_settling_seconds",
                )
            },
        }
    result["episode_300_mean_economic_improvement_percent"] = float(np.mean([
        values["episode_300"]["paired_economic_improvement_percent"]
        for values in result["scenarios"].values()
    ]))
    result["episode_300_worst_economic_improvement_percent"] = float(min(
        values["episode_300"]["paired_economic_improvement_percent"]
        for values in result["scenarios"].values()
    ))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base19", type=Path, default=DEFAULT_19D)
    parser.add_argument("--supervisor23", type=Path, default=DEFAULT_23D)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    base_training = csv_rows(args.base19 / "training_log.csv")
    new_training = csv_rows(args.supervisor23 / "training_log.csv")
    if len(base_training) != 300 or len(new_training) != 300:
        raise RuntimeError("Both ablation runs must have all 300 episodes")
    shock_mismatches = sum(
        old["scenario_type"] != new["scenario_type"]
        or abs(float(old["shock_scale_lambda"])
               - float(new["shock_scale_lambda"])) > 1e-12
        for old, new in zip(base_training, new_training)
    )
    if shock_mismatches:
        raise RuntimeError(
            f"The training shock sequences differ at {shock_mismatches} episodes"
        )
    result = {"base19": summary(args.base19),
              "supervisor23": summary(args.supervisor23)}
    result["paired_training_shock_sequence_mismatches"] = shock_mismatches
    output = args.output_dir or args.supervisor23 / "observation_ablation_comparison"
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, run in (("19D", args.base19), ("23D", args.supervisor23)):
        rows = csv_rows(run / "fixed_evaluation.csv")
        episodes = sorted({int(row["episode"]) for row in rows})
        axes[0].plot(episodes, [np.mean([
            float(row["paired_economic_improvement_percent"]) for row in rows
            if int(row["episode"]) == episode
        ]) for episode in episodes], label=label)
        axes[1].plot(episodes, [np.mean([
            float(row["normalized_mean_IAE_ratio"]) for row in rows
            if int(row["episode"]) == episode
        ]) for episode in episodes], label=label)
    axes[0].set(xlabel="Episode", ylabel="Mean paired economic improvement [%]")
    axes[1].set(xlabel="Episode", ylabel="Mean normalized IAE ratio")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison_learning_curves.png", dpi=160)


if __name__ == "__main__":
    main()
