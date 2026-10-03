"""Static scientific comparison of direct-q and SAC-action oracle sweeps."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .residual_action_space_diagnosis import REPO_DIR


OUTPUT = REPO_DIR / "evaporation_safe_sac/outputs_omega_action_realizability_B"
DIRECT = REPO_DIR / "evaporation_safe_sac/outputs_omega_offline_pareto_B_pressure_refined"
DIRECT_CONC = REPO_DIR / "evaporation_safe_sac/outputs_omega_offline_pareto_B_500"


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    summary = json.loads((OUTPUT / "action_realizability_summary.json").read_text(
        encoding="utf-8"
    ))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=False)
    for axis, scenario in zip(axes, summary["scenarios"]):
        data = summary["scenarios"][scenario]
        base = data["paired_zero_residual_baseline"]
        actor = rows(OUTPUT / f"{scenario}_pareto.csv")
        direct_dir = DIRECT_CONC if scenario == "concentration_positive" else DIRECT
        direct = rows(direct_dir / f"{scenario}_pareto_front.csv")
        for group, label, marker, color in (
            (direct, "direct-q", "o", "#3568a8"),
            (actor, "SAC-action", "^", "#e07726"),
        ):
            improvement = np.array([100 * (
                base["J_econ"] - float(row["J_econ"])
            ) / base["J_econ"] for row in group])
            iae_ratio = np.array([0.5 * (
                float(row["X2_IAE"]) / base["X2_IAE"] +
                float(row["P2_IAE"]) / base["P2_IAE"]
            ) for row in group])
            axis.scatter(improvement, iae_ratio, s=19, alpha=0.5,
                         marker=marker, color=color, label=label)
        chosen = data["actor_action_oracle"]["selected"].get("level_3")
        if chosen is not None:
            axis.scatter([100 * (base["J_econ"] - chosen["J_econ"]) / base["J_econ"]],
                         [chosen["mean_normalized_IAE_ratio"]], s=150,
                         color="gold", edgecolors="black", marker="*",
                         label="Level 3 actor witness", zorder=5)
        axis.axvline(0, color="gray", linewidth=0.8)
        axis.axhline(1, color="gray", linewidth=0.8)
        axis.scatter([0], [1], marker="x", color="red", s=60,
                     label="zero-residual baseline", zorder=5)
        axis.set_title(scenario)
        axis.set_xlabel("Economic improvement vs baseline [%]")
        axis.set_ylabel("Mean normalized X2/P2 IAE ratio")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="best")
    fig.suptitle("Fixed B: direct-q vs true SAC-action offline Pareto search")
    fig.tight_layout()
    path = OUTPUT / "direct_q_vs_actor_action_pareto.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
