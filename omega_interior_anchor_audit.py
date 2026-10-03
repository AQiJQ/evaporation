"""Offline zero-tau diagnosis and interior-anchor prototype validation.

No SAC training or safety-design mutation is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .controlled_invariant_error_set import bounds_and_domains, hull
from .model import EvaporatorModel
from .omega_action_realizability import (
    DEFAULT_CONCENTRATION, DEFAULT_DIRECT, DEFAULT_OMEGA, actor_rollout,
    choose_direct_candidates, inverse_summary, load_reference_profiles,
    peek, read_csv, recreate_direct_trajectory, write_csv,
)
from .omega_feasible_normalized import FeasibleSetNormalizedController
from .omega_feasible_normalized_audit import PROBE_TIMES
from .omega_interior_anchor import (
    InteriorAnchorController, action_for_interior_point,
    chebyshev_center_2d,
)
from .omega_offline_pareto import baseline_metrics
from .omega_sac_train import evaluate_three
from .paper2016_compare import SCENARIOS, apply_state_shock
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b
from .sac import SACAgent, SACConfig
from .train import ACTION_DIM, OBS_DIM, ZeroResidualPolicy, run_episode


RUN_DIR = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_safe_B_feasible_normalized_seed42_300x300"
)
OLD_AUDIT = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_feasible_normalized_B/"
    "feasible_normalized_audit_summary.json"
)
DEFAULT_OUTPUT = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_interior_anchor_B"
)


def load_last_actor(path, cfg):
    import torch
    agent = SACAgent(OBS_DIM, ACTION_DIM,
                     SACConfig(hidden_dim=cfg.hidden_dim), device="cpu")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("obs_dim") != OBS_DIM or payload.get("action_dim") != ACTION_DIM:
        raise RuntimeError("Saved actor observation/action dimensions differ")
    agent.actor.load_state_dict(payload["actor"])
    agent.policy_output_scale = float(payload.get("policy_output_scale", 1.0))
    return agent


def facet_meaning(index, mode):
    if mode == "Omega_safe_one_step_QP":
        labels = (
            "actual P100 robust upper", "actual F200 robust upper",
            "actual P100 robust lower", "actual F200 robust lower",
            "robust next-state Omega facet 0", "robust next-state Omega facet 1",
            "robust next-state Omega facet 2", "robust next-state Omega facet 3",
        )
    else:
        labels = (
            "nominal P100 tightened upper", "nominal F200 tightened upper",
            "nominal P100 tightened lower", "nominal F200 tightened lower",
            "next nominal X2 invariant upper", "next nominal P2 invariant upper",
            "next nominal X2 invariant lower", "next nominal P2 invariant lower",
        )
    return labels[index] if index < len(labels) else f"additional facet {index}"


def diagnose_existing_actor(cfg, model, design, omega, domain, actor, run_dir,
                            output_dir):
    evaluations = evaluate_three(
        cfg, model, design, omega, domain, actor,
        action_mapping="feasible_set_normalized",
    )
    fixed_rows = read_csv(run_dir / "fixed_evaluation.csv")
    details, summary = [], {}
    for scenario in SCENARIOS:
        stored = next(row for row in fixed_rows if row["scenario"] == scenario
                      and int(row["episode"]) == 300)
        records = evaluations[scenario]["records"]
        if abs(evaluations[scenario]["stat"]["J_econ"] -
               float(stored["J_econ"])) > 1e-4:
            raise RuntimeError(f"Episode-300 actor replay J_econ mismatch: {scenario}")
        controller = FeasibleSetNormalizedController(
            cfg, model, design, omega, domain
        )
        controller.reset(cfg.robust_economic_reference_state.copy())
        counts = Counter()
        zero_tau_12 = zero_tau_8 = collapse = nonzero_raw = 0
        for second, record in enumerate(records):
            state = np.asarray(record["state"])
            action = np.asarray(record["raw_action"])
            control, info = controller.act(state, action,
                                           action_is_normalized=True)
            if np.linalg.norm(control - record["control"]) > 1e-6:
                raise RuntimeError(f"Replay control mismatch: {scenario}, {second}")
            rows = info["action_safe_rows"]
            bounds = info["action_safe_bounds"]
            center = info["action_center_coordinate"]
            direction = info["action_direction"]
            slack = bounds - rows @ center
            outward = rows @ direction
            eligible = np.flatnonzero(outward > 1e-12)
            limiting = (int(eligible[np.argmin(slack[eligible] / outward[eligible])])
                        if len(eligible) else -1)
            tau = float(info["action_tau_max"])
            raw_nonzero = bool(np.max(np.abs(action)) > 1e-8)
            mapped_zero = bool(np.linalg.norm(info["applied_residual"]) <= 1e-8)
            zero_tau_12 += tau <= 1e-12
            zero_tau_8 += tau <= 1e-8
            nonzero_raw += raw_nonzero
            collapse += raw_nonzero and mapped_zero
            counts[facet_meaning(limiting, info["mode"])] += 1
            near = {str(i): {
                "meaning": facet_meaning(i, info["mode"]),
                "slack": float(slack[i]),
                "outward_direction_dot": float(outward[i]),
            } for i in range(len(rows)) if slack[i] <= 1e-8}
            details.append({
                "scenario": scenario, "second": second,
                "mode": info["mode"],
                "raw_a_P100": float(action[0]),
                "raw_a_F200": float(action[1]),
                "rho": float(info["action_rho"]),
                "direction_P100": float(direction[0]),
                "direction_F200": float(direction[1]),
                "q_base_P100": float(info["q_base"][0]),
                "q_base_F200": float(info["q_base"][1]),
                "center_P100": float(center[0]),
                "center_F200": float(center[1]),
                "tau_max": tau,
                "limiting_facet_index": limiting,
                "limiting_facet_meaning": facet_meaning(limiting, info["mode"]),
                "limiting_facet_slack": (
                    float(slack[limiting]) if limiting >= 0 else None
                ),
                "limiting_facet_outward_dot": (
                    float(outward[limiting]) if limiting >= 0 else None
                ),
                "near_active_facets_json": json.dumps(near),
                "mapped_residual_P100": float(info["applied_residual"][0]),
                "mapped_residual_F200": float(info["applied_residual"][1]),
                "applied_P100": float(control[0]),
                "applied_F200": float(control[1]),
                "baseline_center_projected": bool(info["action_center_was_projected"]),
                "qp_feasible": bool(info["qp_feasible"]),
            })
        summary[scenario] = {
            "replay_matches_saved_J_econ_and_every_applied_input": True,
            "steps": len(records), "tau_le_1e_12": int(zero_tau_12),
            "tau_le_1e_8": int(zero_tau_8),
            "nonzero_raw_steps": int(nonzero_raw),
            "nonzero_raw_but_zero_mapped_steps": int(collapse),
            "nonzero_raw_but_zero_mapped_fraction": float(collapse / len(records)),
            "limiting_facet_frequency": dict(counts),
        }
    write_csv(output_dir / "episode_300_actor_active_facets.csv", details)
    return summary


def baseline_and_probes(cfg, model, design, omega, domain, seed, output_dir):
    rng = np.random.default_rng(seed + 55001)
    actions = [np.array(a, dtype=float) for a in (
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1), (0.5, 0), (0, 0.5),
    )]
    probes, baseline_summary = [], {}
    for scenario in SCENARIOS:
        old = FeasibleSetNormalizedController(cfg, model, design, omega, domain)
        new = InteriorAnchorController(cfg, model, design, omega, domain)
        _, old_records, _ = run_episode(
            cfg, model, old, ZeroResidualPolicy(), None,
            np.random.default_rng(seed + 31000), training=False,
            global_step=0, paper_scenario=scenario,
        )
        _, new_records, _ = run_episode(
            cfg, model, new, ZeroResidualPolicy(), None,
            np.random.default_rng(seed + 31000), training=False,
            global_step=0, paper_scenario=scenario,
        )
        max_state_gap = max(np.linalg.norm(a["state"] - b["state"])
                            for a, b in zip(old_records, new_records))
        max_input_gap = max(np.linalg.norm(a["control"] - b["control"])
                            for a, b in zip(old_records, new_records))
        if max_state_gap > 1e-10 or max_input_gap > 1e-10:
            raise RuntimeError(f"Zero action changed baseline: {scenario}")
        old_stat = baseline_metrics(old_records, cfg, model, design)
        new_stat = baseline_metrics(new_records, cfg, model, design)
        for key in ("J_econ", "X2_IAE", "P2_IAE", "P100_TV", "F200_TV"):
            if abs(old_stat[key] - new_stat[key]) > 1e-10:
                raise RuntimeError(f"Baseline metric changed: {scenario}, {key}")
        baseline_summary[scenario] = {
            "exact": True, "max_state_gap": max_state_gap,
            "max_input_gap": max_input_gap,
            "metrics": {key: float(new_stat[key]) for key in
                        ("J_econ", "X2_IAE", "P2_IAE", "P100_TV", "F200_TV")},
        }
        controller = InteriorAnchorController(cfg, model, design, omega, domain)
        state = cfg.robust_economic_reference_state.copy()
        controller.reset(state)
        for second in range(300):
            state = apply_state_shock(state, scenario, second, scale=1.0)
            if second in PROBE_TIMES:
                for action in actions + [rng.uniform(-1, 1, 2) for _ in range(8)]:
                    control, info = peek(controller, state, action)
                    rows = info["action_safe_rows"]
                    bounds = info["action_safe_bounds"]
                    final = info["action_final_coordinate"]
                    residual = np.asarray(info["applied_residual"])
                    probes.append({
                        "scenario": scenario, "second": second,
                        "mode": info["mode"],
                        "actor_a_P100": float(action[0]),
                        "actor_a_F200": float(action[1]),
                        "rho": float(np.max(np.abs(action))),
                        "interior_tau_max": float(info.get("interior_tau_max", 0.0)),
                        "interior_radius": float(info.get("interior_chebyshev_radius", 0.0)),
                        "mapped_displacement_norm": float(np.linalg.norm(residual)),
                        "inside_polytope": bool(
                            len(rows) and np.max(rows @ final - bounds) <= 2e-7
                        ),
                        "qp_feasible": bool(info["qp_feasible"]),
                        "final_qp_gap": float(info["final_verification_gap"]),
                        "final_qp_modified": bool(
                            info["final_verification_gap"] > 1e-7
                        ),
                        "physical_clip_gap": float(info["final_physical_clip_gap"]),
                        "interior_target_gap": float(info.get("interior_target_gap", 0.0)),
                        "applied_P100": float(control[0]),
                        "applied_F200": float(control[1]),
                    })
            control, info = controller.act(state, np.zeros(2),
                                           action_is_normalized=True)
            if not info["qp_feasible"]:
                raise RuntimeError(f"Zero baseline QP infeasible: {scenario}, {second}")
            state = model.step(state, control, cfg.disturbance_nominal)
    write_csv(output_dir / "interior_action_semantics_probes.csv", probes)
    if any(not row["inside_polytope"] or not row["qp_feasible"] or
           row["final_qp_modified"] or row["physical_clip_gap"] > 1e-7 or
           row["interior_target_gap"] > 1e-7 for row in probes):
        raise RuntimeError("Interior-anchor action probe failed safety verification")
    return baseline_summary, probes


def inverse_oracles(cfg, model, design, omega, domain, seed, output_dir):
    baseline, profiles = {}, {}
    for scenario in SCENARIOS:
        baseline[scenario], _, profiles[scenario] = load_reference_profiles(
            scenario, cfg, model, design, omega, domain, seed
        )
    choices = choose_direct_candidates(
        DEFAULT_DIRECT, DEFAULT_CONCENTRATION, baseline
    )
    old_summary = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))
    summary = {}
    for scenario in SCENARIOS:
        choice = choices[scenario]
        if choice is None:
            raise RuntimeError(f"No direct-q oracle candidate: {scenario}")
        if isinstance(choice, Path):
            oracle = read_csv(choice)
            direct_id = choice.name
        else:
            _, oracle = recreate_direct_trajectory(
                choice, scenario, profiles[scenario], cfg, model,
                design, omega, domain
            )
            direct_id = choice["id"]
        controller = InteriorAnchorController(cfg, model, design, omega, domain)
        controller.reset(cfg.robust_economic_reference_state.copy())
        rows = []
        for row in oracle:
            state = np.array([float(row["X2"]), float(row["P2"])])
            target = np.array([float(row["P100"]), float(row["F200"])])
            _, zero = peek(controller, state, np.zeros(2))
            base = zero["action_center_coordinate"]
            safe_rows = zero["action_safe_rows"]
            safe_bounds = zero["action_safe_bounds"]
            anchor, radius = chebyshev_center_2d(safe_rows, safe_bounds)
            if zero["mode"] == "Z_mode_existing_controller":
                desired = model.normalized_input(target) - zero["ancillary"]
            elif zero["mode"] == "Omega_safe_one_step_QP":
                desired = model.normalized_input(target) - design.v_ref
            else:
                raise RuntimeError(f"Oracle state outside certified domain: {zero['mode']}")
            action, projected = action_for_interior_point(
                base, anchor, desired, safe_rows, safe_bounds, cfg.input_scale
            )
            control, info = controller.act(state, action,
                                           action_is_normalized=True)
            rows.append({
                "scenario": scenario, "second": int(row["second"]),
                "actor_a_P100": float(action[0]),
                "actor_a_F200": float(action[1]),
                "reconstruction_error_physical": float(np.linalg.norm(
                    control - target
                )),
                "mode": info["mode"],
                "qp_feasible": bool(info["qp_feasible"]),
                "anchor_radius": radius,
                "interior_target_gap": float(info.get("interior_target_gap", 0.0)),
                "target_projection_error_physical": float(np.linalg.norm(
                    (projected - desired) * cfg.input_scale
                )),
            })
        write_csv(output_dir / "inverse_action_trajectories" /
                  f"{scenario}.csv", rows)
        current = inverse_summary(rows)
        previous = old_summary["scenarios"][scenario]["new_mapping_inverse"]
        summary[scenario] = {
            "direct_candidate": direct_id,
            "old_ray_exact_fraction": previous["exact_realizable_step_fraction"],
            "interior_anchor_inverse": current,
        }
        if (current["exact_realizable_step_fraction"] < 0.999 or
                current["qp_infeasible_count"]):
            raise RuntimeError(f"Interior anchor lost oracle realizability: {scenario}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed",
                           seed=args.seed, steps_per_episode=300,
                           residual_parameterization="state_dependent_polytope")
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    cfg.omega_performance_shaping_enabled = True
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    domain, _ = bounds_and_domains(cfg, model, design)
    actor = load_last_actor(args.run_dir / "models/last_actor.pth", cfg)
    collapse = diagnose_existing_actor(
        cfg, model, design, omega, domain, actor, args.run_dir, args.output_dir
    )
    baseline, probes = baseline_and_probes(
        cfg, model, design, omega, domain, args.seed, args.output_dir
    )
    oracle = inverse_oracles(
        cfg, model, design, omega, domain, args.seed, args.output_dir
    )
    nonzero = [row for row in probes if row["rho"] > 1e-8]
    report = {
        "scope": "offline only; no SAC training or safety-design change",
        "actor_checkpoint": str(args.run_dir / "models/last_actor.pth"),
        "existing_ray_episode_300": collapse,
        "zero_action_baseline": baseline,
        "interior_probes": {
            "count": len(probes),
            "feasible_fraction": float(np.mean([r["inside_polytope"] for r in probes])),
            "final_qp_modified_count": sum(r["final_qp_modified"] for r in probes),
            "nonzero_action_zero_displacement_count": sum(
                r["mapped_displacement_norm"] <= 1e-8 for r in nonzero
            ),
            "nonzero_action_zero_displacement_fraction": float(np.mean([
                r["mapped_displacement_norm"] <= 1e-8 for r in nonzero
            ])),
            "minimum_positive_anchor_radius": float(min(
                r["interior_radius"] for r in nonzero
            )),
        },
        "oracle_reconstruction": oracle,
        "ablation_training_authorized_by_this_script": False,
    }
    path = args.output_dir / "interior_anchor_audit_summary.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
