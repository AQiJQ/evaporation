"""Replay saved deterministic actions to audit exact 19D evaluation reward.

No actor weights from intermediate checkpoints are needed: each saved trace
contains the deterministic raw action at every step. Replayed state/input
trajectories are checked against those traces before using reward totals.
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
from .omega_dwell_sac_controller import DwellSupervisedInteriorController
from .omega_dwell_shock_sets import DEFAULT_OUTPUT as DWELL_SETS
from .paper2016_compare import SCENARIOS
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b
from .train import run_episode


DEFAULT_RUN = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_safe_B_dwell20_interior_seed42_300x300_local"
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RecordedActions:
    def __init__(self, rows):
        self.actions = np.array([
            [float(row["actor_raw_a_P100"]), float(row["actor_raw_a_F200"])]
            for row in rows
        ], dtype=np.float32)
        self.index = 0

    def select_action(self, obs, deterministic=True):
        if not deterministic or self.index >= len(self.actions):
            raise RuntimeError("Recorded action sequence exhausted or stochastic")
        action = self.actions[self.index]
        self.index += 1
        return action


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit-checkpoints", type=int)
    args = parser.parse_args()
    output = args.output_dir or args.run_dir.parent / "outputs_omega_reward_alignment_audit_19D"
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run_dir / "run_summary.json").read_text(encoding="utf-8"))
    calibration = json.loads((args.run_dir / "baseline_reward_calibration.json").read_text(encoding="utf-8"))
    cfg = ExperimentConfig(
        benchmark_profile="zanon2016", experiment_mode="proposed",
        episodes=300, steps_per_episode=300, seed=summary["seed"],
        residual_parameterization="state_dependent_polytope",
    )
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    cfg.omega_performance_shaping_enabled = True
    for name, value in calibration["weights"].items():
        attr = {
            "state_recovery": "paper2016_state_recovery_penalty_weight",
            "P100_move": "paper2016_p100_move_penalty_weight",
            "F200_move": "paper2016_f200_move_penalty_weight",
            "saturation": "paper2016_saturation_penalty_weight",
        }[name]
        setattr(cfg, attr, float(value))
    model = EvaporatorModel(cfg)
    design = load_fixed_b(DEFAULT_DESIGN, cfg, model)
    omega = hull(np.loadtxt(REPO_DIR / (
        "evaporation_safe_sac/outputs_controlled_invariant_error_B/"
        "Omega_anchor_B_vertices.csv"), delimiter=",", skiprows=1))
    domain, _ = bounds_and_domains(cfg, model, design)
    if not certificate(omega, domain, design)["passed"]:
        raise RuntimeError("Fixed safety design certificate failed")
    fixed = {(int(row["episode"]), row["scenario"]): row for row in
             read_csv(args.run_dir / "fixed_evaluation.csv")}
    episodes = sorted({key[0] for key in fixed})
    if args.limit_checkpoints is not None:
        episodes = episodes[:args.limit_checkpoints]
    rows = []
    for episode in episodes:
        saved = read_csv(args.run_dir / "fixed_trajectories" / f"episode_{episode:04d}.csv")
        for scenario_index, scenario in enumerate(SCENARIOS):
            trace = [row for row in saved if row["scenario"] == scenario]
            policy = RecordedActions(trace)
            stat, records, _ = run_episode(
                cfg, model,
                DwellSupervisedInteriorController(
                    cfg, model, design, omega, domain, DWELL_SETS
                ),
                policy, None,
                np.random.default_rng(cfg.seed + 31000 + scenario_index),
                training=False, global_step=0, paper_scenario=scenario,
            )
            if policy.index != len(trace) or len(records) != len(trace):
                raise RuntimeError("Replay length mismatch")
            state_error = max(float(np.max(np.abs(
                np.asarray(record["state"]) - np.array([
                    float(t["X2"]), float(t["P2"])
                ])
            ))) for record, t in zip(records, trace))
            input_error = max(float(np.max(np.abs(
                np.asarray(record["control"]) - np.array([
                    float(t["P100"]), float(t["F200"])
                ])
            ))) for record, t in zip(records, trace))
            if max(state_error, input_error) > 1e-5:
                raise RuntimeError(
                    f"Replay mismatch episode={episode} scenario={scenario}: "
                    f"state={state_error}, input={input_error}"
                )
            f = fixed[(episode, scenario)]
            row = {
                "episode": episode, "scenario": scenario,
                "total_evaluation_return": stat["total_reward_total"],
                "replay_equivalent_return": (
                    stat["total_reward_total"]
                    + stat["rpi_violation_event_penalty_total"]
                    + stat["rpi_excess_penalty_total"]
                ),
                "formal_RPI_penalty_total": (
                    stat["rpi_violation_event_penalty_total"]
                    + stat["rpi_excess_penalty_total"]
                ),
                "economic_reward": stat["economic_reward_total"],
                "J_econ": stat["J_econ"],
                "state_recovery_penalty": stat["state_recovery_penalty_total"],
                "P100_move_penalty": stat["p100_move_penalty_total"],
                "F200_move_penalty": stat["f200_move_penalty_total"],
                "saturation_penalty": stat["saturation_penalty_total"],
                "other_replay_penalties": (stat["economic_reward_total"]
                                    - stat["total_reward_total"]
                                    - stat["rpi_violation_event_penalty_total"]
                                    - stat["rpi_excess_penalty_total"]
                                    - sum(stat[key] for key in (
                                        "state_recovery_penalty_total",
                                        "p100_move_penalty_total",
                                        "f200_move_penalty_total",
                                        "saturation_penalty_total"))),
                "economic_improvement_percent": float(f["paired_economic_improvement_percent"]),
                "X2_IAE": float(f["X2_IAE"]),
                "X2_IAE_ratio": float(f["X2_IAE"]) / float(f["baseline_X2_IAE"]),
                "P2_IAE": float(f["P2_IAE"]),
                "P2_IAE_ratio": float(f["P2_IAE"]) / float(f["baseline_P2_IAE"]),
                "X2_ISE": float(f["X2_ISE"]), "P2_ISE": float(f["P2_ISE"]),
                "P100_TV": float(f["P100_TV"]), "F200_TV": float(f["F200_TV"]),
                "P100_TV_ratio": float(f["P100_TV"]) / max(float(f["baseline_P100_TV"]), 1e-12),
                "F200_TV_ratio": (float(f["F200_TV"]) / float(f["baseline_F200_TV"])
                                  if float(f["baseline_F200_TV"]) > 1e-9 else None),
                "robust_bound_occupancy_steps": int(f["robust_input_saturation_steps"]),
                "F200_robust_upper_steps": int(f["F200_at_robust_upper_steps"]),
                "max_replay_state_error": state_error,
                "max_replay_input_error": input_error,
            }
            if abs(row["J_econ"] - float(f["J_econ"])) > 1e-4:
                raise RuntimeError("Replayed economic cost differs from fixed evaluation")
            rows.append(row)
        write_csv(output / "checkpoint_reward_alignment.csv", rows)
        print(f"audited checkpoint {episode}/{episodes[-1]}", flush=True)
    if not rows:
        return
    by_scenario = {}
    for scenario in SCENARIOS:
        subset = [r for r in rows if r["scenario"] == scenario]
        by_scenario[scenario] = {
            "formal_evaluation_return_vs_economic_improvement": pearson(
                [r["total_evaluation_return"] for r in subset],
                [r["economic_improvement_percent"] for r in subset]),
            "return_vs_economic_improvement": pearson(
                [r["replay_equivalent_return"] for r in subset],
                [r["economic_improvement_percent"] for r in subset]),
            "return_vs_X2_IAE_ratio": pearson(
                [r["replay_equivalent_return"] for r in subset],
                [r["X2_IAE_ratio"] for r in subset]),
            "return_vs_P2_IAE_ratio": pearson(
                [r["replay_equivalent_return"] for r in subset],
                [r["P2_IAE_ratio"] for r in subset]),
            "highest_replay_return_checkpoint": {
                key: max(subset, key=lambda r: r["replay_equivalent_return"])[key]
                for key in (
                    "episode", "replay_equivalent_return",
                    "economic_improvement_percent", "X2_IAE_ratio", "P2_IAE_ratio",
                    "P100_TV_ratio", "F200_TV_ratio",
                    "robust_bound_occupancy_steps",
                )
            },
        }
    components = ("economic_reward", "state_recovery_penalty",
                  "P100_move_penalty", "F200_move_penalty",
                  "saturation_penalty", "other_replay_penalties")
    rank_contribution = {}
    for scenario in SCENARIOS:
        subset = [r for r in rows if r["scenario"] == scenario]
        return_rank = np.argsort(np.argsort([
            r["replay_equivalent_return"] for r in subset
        ]))
        rank_contribution[scenario] = {
            key: {
                "range": float(np.ptp([r[key] for r in subset])),
                "rank_correlation_with_total_return": pearson(
                    return_rank, np.argsort(np.argsort([
                        r[key] if key == "economic_reward" else -r[key]
                        for r in subset
                    ]))),
            } for key in components
        }
    report = {
        "checkpoint_count": len(episodes), "scenario_rows": len(rows),
        "exact_replay_verified": True,
        "max_state_error": max(r["max_replay_state_error"] for r in rows),
        "max_input_error": max(r["max_replay_input_error"] for r in rows),
        "correlations_by_scenario": by_scenario,
        "ranking_component_contribution": rank_contribution,
    }
    (output / "reward_alignment_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for scenario in SCENARIOS:
        subset = [r for r in rows if r["scenario"] == scenario]
        x = [r["replay_equivalent_return"] for r in subset]
        axes[0].scatter(x, [r["economic_improvement_percent"] for r in subset],
                        label=scenario, alpha=0.75)
        axes[1].scatter(x, [0.5 * (r["X2_IAE_ratio"] + r["P2_IAE_ratio"])
                            for r in subset], label=scenario, alpha=0.75)
    axes[0].set(xlabel="Replay-equivalent evaluation return", ylabel="Economic improvement [%]")
    axes[1].set(xlabel="Replay-equivalent evaluation return", ylabel="Mean normalized IAE ratio")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "reward_alignment_scatter.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
