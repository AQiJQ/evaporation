"""Offline-only validation of full-feasible-set normalized actor mapping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .controlled_invariant_error_set import bounds_and_domains, hull
from .model import EvaporatorModel
from .omega_action_realizability import (
    DEFAULT_CONCENTRATION, DEFAULT_DIRECT, DEFAULT_OMEGA,
    actor_rollout, choose_direct_candidates, inverse_summary,
    load_reference_profiles, read_csv, recreate_direct_trajectory,
    search_actor, write_csv, peek,
)
from .omega_feasible_normalized import (
    FeasibleSetNormalizedController, action_for_feasible_point,
)
from .paper2016_compare import SCENARIOS, apply_state_shock
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b


DEFAULT_OUTPUT = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_feasible_normalized_B"
)
OLD_SUMMARY = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_action_realizability_B/"
    "action_realizability_summary.json"
)
PROBE_TIMES = (0, 1, 19, 20, 39, 40, 60, 100, 180, 299)


def action_probes(cfg, model, design, omega, domain, seed):
    rng = np.random.default_rng(seed + 55001)
    corners = [np.array(a, dtype=float) for a in (
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
        (0.5, 0), (0, 0.5),
    )]
    records = []
    for scenario in SCENARIOS:
        controller = FeasibleSetNormalizedController(cfg, model, design, omega, domain)
        state = cfg.robust_economic_reference_state.copy()
        controller.reset(state)
        for second in range(300):
            state = apply_state_shock(state, scenario, second, scale=1.0)
            if second in PROBE_TIMES:
                actions = corners + [rng.uniform(-1, 1, 2) for _ in range(8)]
                for action in actions:
                    control, info = peek(controller, state, action)
                    rows = info["action_safe_rows"]
                    bounds = info["action_safe_bounds"]
                    center = info["action_center_coordinate"]
                    final_point = info["action_final_coordinate"]
                    tau = float(info["action_tau_max"])
                    rho = float(info["action_rho"])
                    displacement = float(np.max(np.abs(final_point - center)))
                    records.append({
                        "scenario": scenario, "second": second,
                        "mode": info["mode"],
                        "a_0": float(action[0]), "a_1": float(action[1]),
                        "rho": rho,
                        "direction_0": float(info["action_direction"][0]),
                        "direction_1": float(info["action_direction"][1]),
                        "center_coordinate_0": float(center[0]),
                        "center_coordinate_1": float(center[1]),
                        "center_actual_P100": float(model.physical_input(
                            info["action_center_actual_norm"]
                        )[0]),
                        "center_actual_F200": float(model.physical_input(
                            info["action_center_actual_norm"]
                        )[1]),
                        "tau_max": tau,
                        "applied_P100": float(control[0]),
                        "applied_F200": float(control[1]),
                        "polytope_max_excess": float(info["action_polytope_max_excess"]),
                        "inside_polytope": bool(
                            len(rows) > 0 and np.max(rows @ final_point - bounds) <= 2e-7
                        ),
                        "final_qp_gap_normalized": float(info["final_verification_gap"]),
                        "final_qp_modified": bool(info["final_verification_gap"] > 1e-7),
                        "physical_clip_gap_normalized": float(info["final_physical_clip_gap"]),
                        "fraction_of_ray_used": (
                            displacement / tau if tau > 1e-12 else 0.0
                        ),
                        "fraction_error": (
                            abs(displacement / tau - rho) if tau > 1e-12 else 0.0
                        ),
                        "qp_feasible": bool(info["qp_feasible"]),
                        "center_was_projected": bool(info["action_center_was_projected"]),
                    })
            control, info = controller.act(state, np.zeros(2),
                                           action_is_normalized=True)
            if not info["qp_feasible"]:
                raise RuntimeError(f"New zero-residual baseline QP failed: {scenario}, {second}")
            state = model.step(state, control, cfg.disturbance_nominal)
    return records


def inverse_direct_trajectory(oracle, cfg, model, design, omega, domain):
    controller = FeasibleSetNormalizedController(cfg, model, design, omega, domain)
    controller.reset(cfg.robust_economic_reference_state.copy())
    output = []
    for row in oracle:
        state = np.array([float(row["X2"]), float(row["P2"])])
        target = np.array([float(row["P100"]), float(row["F200"])])
        _, zero_info = peek(controller, state, np.zeros(2))
        mode = zero_info["mode"]
        center = zero_info["action_center_coordinate"]
        if mode == "Z_mode_existing_controller":
            target_coordinate = model.normalized_input(target) - zero_info["ancillary"]
        elif mode == "Omega_safe_one_step_QP":
            target_coordinate = model.normalized_input(target) - design.v_ref
        else:
            raise RuntimeError(f"Oracle state outside certified domain: {mode}")
        action, projected = action_for_feasible_point(
            center, target_coordinate, zero_info["action_safe_rows"],
            zero_info["action_safe_bounds"], cfg.input_scale,
        )
        control, info = controller.act(state, action, action_is_normalized=True)
        error = float(np.linalg.norm(control - target))
        output.append({
            "scenario": row["scenario"], "second": int(row["second"]),
            "X2_oracle": float(state[0]), "P2_oracle": float(state[1]),
            "P100_oracle": float(target[0]), "F200_oracle": float(target[1]),
            "actor_a_P100": float(action[0]), "actor_a_F200": float(action[1]),
            "P100_reconstructed": float(control[0]),
            "F200_reconstructed": float(control[1]),
            "reconstruction_error_physical": error,
            "reconstruction_error_normalized": float(np.linalg.norm(
                model.normalized_input(control) - model.normalized_input(target)
            )),
            "mode": info["mode"], "qp_feasible": bool(info["qp_feasible"]),
            "in_Z": bool(info["in_Z"]), "in_Omega": bool(info["in_Omega"]),
            "tau_max": float(info["action_tau_max"]),
            "center_coordinate_0": float(center[0]),
            "center_coordinate_1": float(center[1]),
            "target_projection_error_physical": float(np.linalg.norm(
                (projected - target_coordinate) * cfg.input_scale
            )),
            "final_verification_gap": float(info["final_verification_gap"]),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--direct-dir", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--concentration-trajectory", type=Path,
                        default=DEFAULT_CONCENTRATION)
    parser.add_argument("--old-summary", type=Path, default=OLD_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--actor-budget", type=int, default=80)
    parser.add_argument("--skip-action-search", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed",
                           seed=args.seed, steps_per_episode=300)
    cfg.residual_parameterization = "state_dependent_polytope"
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    domain, _ = bounds_and_domains(cfg, model, design)
    old = json.loads(args.old_summary.read_text(encoding="utf-8"))
    probes = action_probes(cfg, model, design, omega, domain, args.seed)
    write_csv(args.output_dir / "action_semantics_probes.csv", probes)
    if any(not row["inside_polytope"] or not row["qp_feasible"] for row in probes):
        raise RuntimeError("New mapping violated a probed safety polytope/QP")
    if any(row["final_qp_modified"] or row["physical_clip_gap_normalized"] > 1e-7
           for row in probes):
        raise RuntimeError("Final verification or physical clip changed a mapped action")
    if max(row["fraction_error"] for row in probes) > 1e-6:
        raise RuntimeError("Actor rho does not match used ray-authority fraction")

    baseline_by_scenario, profiles_by_scenario = {}, {}
    for scenario in SCENARIOS:
        baseline_by_scenario[scenario], _, profiles_by_scenario[scenario] = (
            load_reference_profiles(
                scenario, cfg, model, design, omega, domain, args.seed
            )
        )
    chosen = choose_direct_candidates(
        args.direct_dir, args.concentration_trajectory, baseline_by_scenario
    )
    summary = {
        "scope": "offline prototype only; no SAC training or design change",
        "mapping": "ray to entire existing feasible Z/Ω polygon from projected zero-residual center",
        "actor_dimension": 2,
        "observation_dimension_unchanged": 19,
        "exact_tolerance_physical_input_l2": 1e-3,
        "actor_search_budget_per_scenario": (
            0 if args.skip_action_search else args.actor_budget
        ),
        "search_is_global_proof": False,
        "probe_count": len(probes),
        "probe_modes": {mode: sum(row["mode"] == mode for row in probes)
                        for mode in sorted(set(row["mode"] for row in probes))},
        "probe_max_polytope_excess": float(max(
            row["polytope_max_excess"] for row in probes
        )),
        "probe_max_fraction_error": float(max(
            row["fraction_error"] for row in probes
        )),
        "probe_final_qp_modified_count": int(sum(
            row["final_qp_modified"] for row in probes
        )),
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        choice = chosen[scenario]
        if choice is None:
            raise RuntimeError(f"No direct-q Level-1 candidate for {scenario}")
        if isinstance(choice, Path):
            oracle = read_csv(choice)
            direct_id = choice.name
        else:
            _, oracle = recreate_direct_trajectory(
                choice, scenario, profiles_by_scenario[scenario], cfg, model,
                design, omega, domain
            )
            direct_id = choice["id"]
        inverse = inverse_direct_trajectory(
            oracle, cfg, model, design, omega, domain
        )
        write_csv(args.output_dir / "inverse_action_trajectories" /
                  f"{scenario}.csv", inverse)
        current_inverse = inverse_summary(inverse)
        old_scenario = old["scenarios"][scenario]
        baseline = baseline_by_scenario[scenario]
        zero_result = actor_rollout(
            scenario, np.zeros((300, 2)), cfg, model, design, omega, domain,
            FeasibleSetNormalizedController
        )
        if zero_result is None:
            raise RuntimeError(f"New zero-residual rollout unsafe: {scenario}")
        zero_metrics, _ = zero_result
        for key in ("J_econ", "X2_IAE", "P2_IAE", "P100_TV", "F200_TV"):
            if abs(zero_metrics[key] - baseline[key]) > 1e-5:
                raise RuntimeError(f"Zero-residual baseline changed: {scenario}, {key}")
        scenario_summary = {
            "direct_candidate": direct_id,
            "paired_zero_residual_baseline": baseline,
            "old_mapping_inverse": old_scenario["inverse_action_audit"],
            "new_mapping_inverse": current_inverse,
            "exact_fraction_increase": float(
                current_inverse["exact_realizable_step_fraction"] -
                old_scenario["inverse_action_audit"]["exact_realizable_step_fraction"]
            ),
            "baseline_unchanged": True,
        }
        if not args.skip_action_search:
            actions = np.array([[
                row["actor_a_P100"], row["actor_a_F200"]
            ] for row in inverse])
            actor_result = search_actor(
                scenario, {"inverse_direct_q": actions}, baseline,
                cfg, model, design, omega, domain,
                np.random.default_rng(args.seed + 1000 * len(summary["scenarios"])),
                args.actor_budget, args.output_dir,
                FeasibleSetNormalizedController
            )
            scenario_summary["old_mapping_actor_oracle"] = old_scenario[
                "actor_action_oracle"
            ]["level_counts"]
            scenario_summary["new_mapping_actor_oracle"] = actor_result
        summary["scenarios"][scenario] = scenario_summary
        print(scenario, "exact old/new",
              f"{old_scenario['inverse_action_audit']['exact_realizable_step_fraction']:.3f}/"
              f"{current_inverse['exact_realizable_step_fraction']:.3f}",
              "actor levels", scenario_summary.get("new_mapping_actor_oracle", {}).get(
                  "level_counts", "skipped"), flush=True)
    path = args.output_dir / "feasible_normalized_audit_summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False),
                    encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
