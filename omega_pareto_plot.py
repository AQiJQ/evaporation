"""Plot the finite offline Omega-safe Pareto sweep as a static scientific figure."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .residual_action_space_diagnosis import REPO_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pressure-dir", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_omega_offline_pareto_B_pressure_refined"
    ))
    parser.add_argument("--concentration-dir", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_omega_offline_pareto_B_500"
    ))
    parser.add_argument("--output", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_omega_offline_pareto_B_500/"
        "pareto_economy_state_TV_occupancy.png"
    ))
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, scenario in zip(axes, (
        "pressure_positive", "pressure_negative", "concentration_positive"
    )):
        directory = (args.concentration_dir if scenario == "concentration_positive"
                     else args.pressure_dir)
        summary = json.loads((directory / "oracle_summary.json").read_text(encoding="utf-8"))
        baseline = summary["scenarios"][scenario]["paired_zero_residual_baseline"]
        with (directory / f"{scenario}_pareto_front.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        economy = np.array([
            100.0 * (baseline["J_econ"] - float(row["J_econ"])) / baseline["J_econ"]
            for row in rows
        ])
        state = np.array([float(row["objective_state_IAE_ratio_mean"]) for row in rows])
        total_tv = np.array([float(row["P100_TV"]) + float(row["F200_TV"])
                             for row in rows])
        occupancy = np.array([float(row["robust_outer_10pct_steps"]) for row in rows])
        scatter = ax.scatter(economy, state, c=np.log1p(total_tv),
                             s=12 + occupancy / 8, cmap="viridis", alpha=0.7)
        ax.scatter([0], [1], marker="x", s=90, color="red", linewidth=2,
                   label="zero-residual baseline")
        selected = summary["scenarios"][scenario]["selected_candidates"]
        full = selected["economy_recovery_reasonable_activity"]
        if full is not None:
            ax.scatter([
                100.0 * (baseline["J_econ"] - full["J_econ"]) / baseline["J_econ"]
            ], [full["objective_state_IAE_ratio_mean"]], marker="*", s=160,
                color="orange", edgecolor="black", label="joint witness")
        ax.axvline(0, color="gray", linewidth=0.7)
        ax.axhline(1, color="gray", linewidth=0.7)
        ax.set(title=scenario, xlabel="Economic improvement vs baseline [%]",
               ylabel="Mean X2/P2 IAE ratio to baseline")
        ax.legend(fontsize=8, loc="best")
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.85)
        colorbar.set_label("log(1 + P100 TV + F200 TV)")
    fig.suptitle("Fixed-B offline Omega-safe Pareto sweep; point size = robust-bound occupancy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
