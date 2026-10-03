"""Offline paired-performance constraint audit; never trains a policy."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .model import EvaporatorModel
from .omega_constrained_objective import (
    KEYS, limits_from_fixed_row, physical_robust_ranges,
    violations_from_fixed_row,
)
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b


DEFAULT_RUN = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_safe_B_dwell20_interior_obs23_seed42_300x300_local_full"
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.run_dir.parent / "outputs_omega_constraint_audit_obs23"
    output.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed")
    model = EvaporatorModel(cfg)
    design = load_fixed_b(DEFAULT_DESIGN, cfg, model)
    ranges = physical_robust_ranges(cfg, design)
    fixed = read_csv(args.run_dir / "fixed_evaluation.csv")
    assessments = {int(r["episode"]): r for r in
                   read_csv(args.run_dir / "checkpoint_assessments.csv")}
    if len(fixed) != 183 or len(assessments) != 61:
        raise RuntimeError("Expected all 61 checkpoints and 183 scenario rows")
    rows = []
    violation_counts = Counter()
    for row in fixed:
        limits = limits_from_fixed_row(row, ranges)
        g = violations_from_fixed_row(row, limits)
        episode = int(row["episode"])
        violated = [key for key, value in zip(KEYS, g) if value > 1e-10]
        violation_counts.update(violated)
        rows.append({
            "episode": episode, "scenario": row["scenario"],
            "economic_improvement_percent": float(
                row["paired_economic_improvement_percent"]),
            **{f"g_{key}": float(value) for key, value in zip(KEYS, g)},
            **{f"allowed_{key}": float(value) for key, value in
               zip(KEYS, limits.allowed())},
            "constraints_feasible": not violated,
            "joint_economic_and_constraints_feasible": (
                not violated
                and float(row["paired_economic_improvement_percent"]) > 0
            ),
            "violated_constraints": ";".join(violated),
            "existing_comprehensive_performance_passed": (
                assessments[episode]["performance_within_10pct_baseline"] == "True"
            ),
        })
    write_csv(output / "all_checkpoint_constraints.csv", rows)
    by_episode = []
    for episode in sorted({r["episode"] for r in rows}):
        subset = [r for r in rows if r["episode"] == episode]
        by_episode.append({
            "episode": episode,
            "mean_economic_improvement_percent": float(np.mean([
                r["economic_improvement_percent"] for r in subset])),
            "worst_economic_improvement_percent": float(min(
                r["economic_improvement_percent"] for r in subset)),
            "all_five_constraints_feasible": all(
                r["constraints_feasible"] for r in subset),
            "scenario_violations": sum(not r["constraints_feasible"] for r in subset),
            **{f"max_g_{key}": float(max(r[f"g_{key}"] for r in subset))
               for key in KEYS},
        })
    write_csv(output / "checkpoint_constraint_summary.csv", by_episode)
    late = [r for r in rows if r["episode"] >= 255]
    ep100 = [r for r in rows if r["episode"] == 100]
    report = {
        "checkpoint_count": 61,
        "scenario_rows": len(rows),
        "robust_input_physical_ranges": ranges.tolist(),
        "threshold_rule": (
            "IAE <= 1.10*paired_baseline; extra TV <= 10% baseline TV, "
            "or <=1 physical unit when baseline TV=0; extra boundary steps "
            "<=10% paired baseline occupancy steps (explicit extension, "
            "because prior comprehensive check had no occupancy threshold)"
        ),
        "constraint_violation_frequency_by_scenario_row": dict(violation_counts),
        "all_five_feasible_checkpoint_count": sum(
            r["all_five_constraints_feasible"] for r in by_episode),
        "positive_economic_all_five_feasible_checkpoint_count": sum(
            r["all_five_constraints_feasible"]
            and r["worst_economic_improvement_percent"] > 0
            for r in by_episode),
        "episode_100": ep100,
        "late_255_300_mean": {
            key: float(np.mean([r[f"g_{key}"] for r in late])) for key in KEYS
        },
        "late_255_300_feasible_rows": sum(r["constraints_feasible"] for r in late),
        "late_255_300_row_count": len(late),
    }
    (output / "audit_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for scenario in sorted({r["scenario"] for r in rows}):
        subset = [r for r in rows if r["scenario"] == scenario]
        axes[0].plot([r["episode"] for r in subset],
                     [r["economic_improvement_percent"] for r in subset],
                     label=scenario)
        axes[1].plot([r["episode"] for r in subset],
                     [max(r["g_x2"], r["g_p2"]) for r in subset],
                     label=scenario)
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[0].set(xlabel="Episode", ylabel="Economic improvement [%]")
    axes[1].set(xlabel="Episode", ylabel="Max recovery constraint g")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "economic_vs_recovery_constraints.png", dpi=160)
    plt.close(fig)
    print(json.dumps({
        "all_five_feasible_checkpoint_count": report[
            "all_five_feasible_checkpoint_count"],
        "violation_frequency": report[
            "constraint_violation_frequency_by_scenario_row"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
