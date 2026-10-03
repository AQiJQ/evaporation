"""Read-only matched seed-42 comparison of M0/M1/M2 action mappings.

Run only after the interior-anchor 300x300 experiment finishes. Existing
training outputs are read, never changed; new comparison files are written
under the M2 output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .paper2016_compare import SCENARIOS
from .residual_action_space_diagnosis import REPO_DIR


ROOT = REPO_DIR / "evaporation_safe_sac"
DEFAULT_RUNS = {
    "M0_original_residual": ROOT / "outputs_omega_safe_B_rewardshape_seed42_300x300_local",
    "M1_boundary_ray": ROOT / "outputs_omega_safe_B_feasible_normalized_seed42_300x300",
    "M2_interior_anchor": ROOT / "outputs_omega_safe_B_interior_anchor_seed42_300x300",
}
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def optional_number(value):
    return None if value in (None, "", "nan", "NaN") else float(value)


def trajectory_action_metrics(path):
    rows = read_csv(path)
    result = {}
    for scenario in SCENARIOS:
        selected = [r for r in rows if r["scenario"] == scenario]
        raw_available = "actor_raw_a_P100" in selected[0]
        if raw_available:
            actions = np.array([[
                float(r["actor_raw_a_P100"]), float(r["actor_raw_a_F200"])
            ] for r in selected])
            rho = np.max(np.abs(actions), axis=1)
            nonzero = rho > 1e-8
            raw_saturation = float(np.mean(rho >= 0.99))
        else:
            # The original M0 trajectory format omitted raw actor actions.
            # A requested fixed-scale residual >0 implies a nonzero command,
            # but action saturation is not recoverable for all checkpoints.
            nonzero = np.array([
                float(r["requested_residual_norm"]) > 1e-8
                for r in selected
            ])
            raw_saturation = None
        if "collapsed_action" in selected[0]:
            collapsed = np.array([
                r["collapsed_action"].lower() == "true" for r in selected
            ])
            method = "physical_input_displacement_l2_lt_1e-6"
        else:
            # Historical M0/M1 traces lack the center input vector. Their
            # exactly-zero applied normalized residual is the available proxy.
            collapsed = np.array([
                float(r["applied_residual_norm"]) <= 1e-8 for r in selected
            ])
            method = (
                "historical_normalized_residual_norm_le_1e-8_proxy"
                if raw_available else
                "M0_requested_and_applied_residual_norm_proxy_no_raw_actor"
            )
        result[scenario] = {
            "raw_actor_saturation_fraction": raw_saturation,
            "nonzero_actor_steps": int(np.sum(nonzero)),
            "collapsed_action_steps": int(np.sum(nonzero & collapsed)),
            "action_collapse_fraction": (
                float(np.sum(nonzero & collapsed) / np.sum(nonzero))
                if np.any(nonzero) else None
            ),
            "collapse_measurement": method,
        }
    return result


def select_checkpoints(summary):
    return {
        "episode_300": 300,
        "best_economic": (
            summary["best_economic_diagnostic_checkpoint"] or {}
        ).get("episode"),
        "nearest_tradeoff": (
            summary["nearest_tradeoff_diagnostic_checkpoint"] or {}
        ).get("episode"),
        "strict_constrained": (
            summary["best_constrained_performance_checkpoint"] or {}
        ).get("episode"),
    }


def comparable_row(mapping, row, action):
    x2_ratio = float(row["X2_IAE"]) / float(row["baseline_X2_IAE"])
    p2_ratio = float(row["P2_IAE"]) / float(row["baseline_P2_IAE"])
    return {
        "mapping": mapping,
        "episode": int(row["episode"]),
        "scenario": row["scenario"],
        "economic_improvement_percent": float(
            row["paired_economic_improvement_percent"]
        ),
        "J_econ": float(row["J_econ"]),
        "X2_IAE": float(row["X2_IAE"]),
        "X2_ISE": float(row["X2_ISE"]),
        "P2_IAE": float(row["P2_IAE"]),
        "P2_ISE": float(row["P2_ISE"]),
        "normalized_mean_IAE_ratio": 0.5 * (x2_ratio + p2_ratio),
        "X2_common_reference_settling_seconds": optional_number(
            row["X2_common_reference_settling_seconds"]
        ),
        "P2_common_reference_settling_seconds": optional_number(
            row["P2_common_reference_settling_seconds"]
        ),
        "P100_TV": float(row["P100_TV"]),
        "F200_TV": float(row["F200_TV"]),
        "P100_plus_F200_TV": float(row["P100_TV"]) + float(row["F200_TV"]),
        "robust_bound_occupancy_steps": int(row["robust_input_saturation_steps"]),
        "raw_actor_saturation_fraction": action["raw_actor_saturation_fraction"],
        "action_collapse_fraction": action["action_collapse_fraction"],
        "collapsed_action_steps": action["collapsed_action_steps"],
        "nonzero_actor_steps": action["nonzero_actor_steps"],
        "collapse_measurement": action["collapse_measurement"],
        "Z_mode_fraction": float(row["Z_mode_fraction"]),
        "Omega_mode_fraction": float(row["Omega_mode_fraction"]),
        "Omega_exit_count": int(row["Omega_exit_count"]),
        "QP_infeasible_rate": float(row["QP_infeasible_rate"]),
        "physical_state_violation_steps": int(row["physical_state_violation_steps"]),
        "physical_input_violation_steps": int(row["physical_input_violation_steps"]),
        "robust_region_violation_rate": float(row["robust_region_violation_rate"]),
        "nonlinear_W_exceedance_steps": int(row["nonlinear_W_exceedance_steps"]),
    }


def checkpoint_mean(rows):
    collapse_values = [
        r["action_collapse_fraction"] for r in rows
        if r["action_collapse_fraction"] is not None
    ]
    return {
        "economic_improvement_percent": float(np.mean([
            r["economic_improvement_percent"] for r in rows
        ])),
        "worst_scenario_economic_improvement_percent": float(min(
            r["economic_improvement_percent"] for r in rows
        )),
        "normalized_mean_IAE_ratio": float(np.mean([
            r["normalized_mean_IAE_ratio"] for r in rows
        ])),
        "P100_plus_F200_TV": float(np.mean([
            r["P100_plus_F200_TV"] for r in rows
        ])),
        "robust_bound_occupancy_steps": float(np.mean([
            r["robust_bound_occupancy_steps"] for r in rows
        ])),
        "action_collapse_fraction": (
            float(np.mean(collapse_values)) if collapse_values else None
        ),
        "safety_passed": all(
            float(r["Omega_exit_count"]) == 0 and
            float(r["QP_infeasible_rate"]) == 0 and
            float(r["physical_state_violation_steps"]) == 0 and
            float(r["physical_input_violation_steps"]) == 0 and
            float(r["robust_region_violation_rate"]) == 0 and
            float(r["nonlinear_W_exceedance_steps"]) == 0
            for r in rows
        ),
    }


def plot_pareto(path, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {
        "M0_original_residual": "#606c78",
        "M1_boundary_ray": "#c97829",
        "M2_interior_anchor": "#1a8174",
    }
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for mapping, color in colors.items():
        episodes = sorted(set(r["episode"] for r in table if r["mapping"] == mapping))
        means = [checkpoint_mean([
            r for r in table if r["mapping"] == mapping and r["episode"] == ep
        ]) for ep in episodes]
        size = np.array([m["P100_plus_F200_TV"] for m in means])
        size = 30.0 + 3.0 * np.sqrt(np.maximum(size, 0.0))
        ax.scatter(
            [m["economic_improvement_percent"] for m in means],
            [m["normalized_mean_IAE_ratio"] for m in means],
            s=size, alpha=0.65, color=color, label=mapping,
            edgecolor="white", linewidth=0.4,
        )
    ax.axvline(0, color="gray", linewidth=0.9)
    ax.axhline(1, color="gray", linewidth=0.9)
    ax.set(
        xlabel="Three-scenario mean paired economic improvement [%]",
        ylabel="Three-scenario mean normalized IAE ratio",
        title="M0/M1/M2 fixed deterministic evaluation checkpoints",
    )
    ax.text(0.01, 0.99,
            "Marker area represents P100 TV + F200 TV; occupancy in CSV",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for mapping, path in DEFAULT_RUNS.items():
        parser.add_argument("--" + mapping.replace("_", "-"), type=Path,
                            default=path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    runs = {
        mapping: getattr(args, mapping) for mapping in DEFAULT_RUNS
    }
    output = args.output_dir or runs["M2_interior_anchor"] / "three_mapping_comparison"
    incomplete = []
    for mapping, path in runs.items():
        if (path / "run_summary.json").is_file():
            continue
        abort_path = path / "training_abort.json"
        if abort_path.is_file():
            abort = json.loads(abort_path.read_text(encoding="utf-8"))
            incomplete.append(
                f"{mapping}: stopped at episode {abort.get('episode')} "
                f"({abort.get('reason')}); anomaly={abort.get('anomaly')}"
            )
        else:
            incomplete.append(f"{mapping}: run_summary.json is absent at {path}")
    if incomplete:
        raise SystemExit(
            "Three-mapping comparison requires all completed 300x300 runs. "
            + " | ".join(incomplete)
        )
    summaries = {
        mapping: json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
        for mapping, path in runs.items()
    }
    common = (
        "seed", "episodes", "steps_per_episode", "evaluation_every_episodes",
        "fixed_scenarios", "training_schedule_counts", "shock_amplitude_distribution",
    )
    reference = summaries["M0_original_residual"]
    for mapping, summary in summaries.items():
        for key in common:
            if summary[key] != reference[key]:
                raise RuntimeError(f"Protocol differs for {mapping}: {key}")
        old_weights = reference["calibration"]["calibrated_weights"]
        for key, value in old_weights.items():
            if not np.isclose(summary["calibration"]["calibrated_weights"][key],
                              value, atol=1e-9, rtol=0):
                raise RuntimeError(f"Reward weight differs for {mapping}: {key}")
    for key in ("sac_hyperparameters_frozen", "observation_dimension"):
        if summaries["M1_boundary_ray"][key] != summaries["M2_interior_anchor"][key]:
            raise RuntimeError(f"M1/M2 differ in {key}")
    logs = {mapping: read_csv(path / "training_log.csv")
            for mapping, path in runs.items()}
    shock_matches = {}
    for mapping, records in logs.items():
        paired = list(zip(logs["M0_original_residual"], records))
        shock_matches[mapping] = {
            "scenario_exact_match": len(paired) == 300 and all(
                a["scenario_type"] == b["scenario_type"] for a, b in paired
            ),
            "lambda_exact_match": len(paired) == 300 and all(
                a["shock_scale_lambda"] == b["shock_scale_lambda"]
                for a, b in paired
            ),
        }
    table = []
    for mapping, path in runs.items():
        action_cache = {}
        for row in read_csv(path / "fixed_evaluation.csv"):
            episode = int(row["episode"])
            trajectory = path / "fixed_trajectories" / f"episode_{episode:04d}.csv"
            if not trajectory.is_file():
                raise RuntimeError(f"Missing fixed trajectory: {trajectory}")
            # Small per-episode cache avoids re-reading three scenarios.
            if episode not in action_cache:
                action_cache[episode] = trajectory_action_metrics(trajectory)
            table.append(comparable_row(
                mapping, row, action_cache[episode][row["scenario"]]
            ))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "three_mapping_checkpoint_table.csv", table)
    comparison = {}
    for mapping, summary in summaries.items():
        selection = select_checkpoints(summary)
        selected = {}
        for label, episode in selection.items():
            if episode is None:
                selected[label] = None
                continue
            rows = [r for r in table if r["mapping"] == mapping
                    and r["episode"] == episode]
            selected[label] = {
                "episode": episode,
                "three_scenario_mean": checkpoint_mean(rows),
                "scenarios": {r["scenario"]: r for r in rows},
            }
        late = [r for r in table if r["mapping"] == mapping
                and 250 < r["episode"] <= 300]
        comparison[mapping] = {
            "selected_checkpoints": selected,
            "late_50_episode_fixed_evaluation_mean": {
                "episodes": sorted(set(r["episode"] for r in late)),
                "three_scenario_mean": checkpoint_mean(late),
                "scenario_mean_economic_improvement_percent": {
                    scenario: float(np.mean([
                        r["economic_improvement_percent"] for r in late
                        if r["scenario"] == scenario
                    ])) for scenario in SCENARIOS
                },
            },
        }
    result = {
        "protocol_fields_equal": True,
        "reward_weights_equal": True,
        "sac_and_observation_metadata_note": (
            "M1/M2 metadata match exactly; historical M0 run_summary did not "
            "record these fields, so they cannot be asserted from that JSON."
        ),
        "training_shock_sequence_match_vs_M0": shock_matches,
        "late_window_definition": "250 < episode <= 300 (10 fixed evaluations)",
        "historical_collapse_note": (
            "M0/M1 historical trajectories record applied normalized residual "
            "norm, not center input; exactly-zero residual is used as a proxy. "
            "M2 uses physical final-minus-baseline input norm < 1e-6."
        ),
        "checkpoint_selection_note": (
            "Existing run_summary selections are reused unchanged; highest "
            "economic checkpoint is diagnostic, not automatically best."
        ),
        "mappings": comparison,
    }
    (output / "three_mapping_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    plot_pareto(output / "economic_vs_recovery_pareto.png", table)
    print(output / "three_mapping_summary.json")


if __name__ == "__main__":
    main()
