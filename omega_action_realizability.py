"""Offline inverse-action audit and SAC-action direct-shooting Pareto search.

No training, safety-design mutation, online optimizer, or MPC is introduced.
Every proposed actor action is passed through OmegaSafeOnlineController.act.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .controlled_invariant_error_set import bounds_and_domains, contains, facets, hull
from .model import EvaporatorModel
from .omega_offline_pareto import (
    PHASE_EDGES, baseline_metrics, direct_omega_rollout,
    profile_from_trajectory, write_csv,
)
from .omega_online_closed_loop import OmegaSafeOnlineController
from .paper2016_compare import SCENARIOS, apply_state_shock
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b
from .train import ZeroResidualPolicy, run_episode


DEFAULT_OMEGA = REPO_DIR / (
    "evaporation_safe_sac/outputs_controlled_invariant_error_B/"
    "Omega_anchor_B_vertices.csv"
)
DEFAULT_DIRECT = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_offline_pareto_B_pressure_refined"
)
DEFAULT_CONCENTRATION = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_offline_pareto_B_500/"
    "selected_oracle_trajectories/"
    "concentration_positive_economy_recovery_reasonable_activity.csv"
)
DEFAULT_OUTPUT = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_action_realizability_B"
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_reference_profiles(scenario, cfg, model, design, omega, domain, seed):
    controller = OmegaSafeOnlineController(cfg, model, design, omega, domain)
    _, baseline_records, _ = run_episode(
        cfg, model, controller, ZeroResidualPolicy(), None,
        np.random.default_rng(seed + 31000), training=False,
        global_step=0, paper_scenario=scenario,
    )
    baseline = baseline_metrics(baseline_records, cfg, model, design)
    zero_q = model.normalized_input(np.array([
        record["control"] for record in baseline_records
    ])) - design.v_ref
    profiles = {"zero_residual_profile": zero_q,
                "reference_q_zero": np.zeros((300, 2))}
    sources = {
        "old_2000_last": REPO_DIR / (
            "evaporation_safe_sac/outputs_omega_safe_B_seed42_2000x300/"
            "fixed_trajectories/episode_2000.csv"
        ),
        "new_300_last": REPO_DIR / (
            "evaporation_safe_sac/outputs_omega_safe_B_rewardshape_seed42_300x300_local/"
            "fixed_trajectories/episode_0300.csv"
        ),
        "new_best_econ": REPO_DIR / (
            "evaporation_safe_sac/outputs_omega_safe_B_rewardshape_seed42_300x300_local/"
            "fixed_trajectories/episode_0070.csv"
        ),
    }
    for name, path in sources.items():
        q = profile_from_trajectory(path, scenario, model, design)
        if q is not None:
            profiles[name] = q
            for fraction in (0.05, 0.10, 0.20, 0.40, 0.60, 0.80):
                profiles[f"blend_zero_{name}_{fraction:.2f}"] = (
                    (1 - fraction) * zero_q + fraction * q
                )
    return baseline, baseline_records, profiles


def recreate_direct_trajectory(row, scenario, profiles, cfg, model, design,
                               omega, domain):
    normals, bounds = facets(omega)
    support = np.max(design.w_vertices @ normals.T, axis=0)
    fixed_rows = np.vstack([np.eye(2), -np.eye(2), normals @ design.b])
    fixed_head = np.concatenate([domain["q_upper"], -domain["q_lower"]])
    bias = np.array(json.loads(row["phase_bias_json"]), dtype=float)
    gain = np.array(json.loads(row["state_feedback_gain_json"]), dtype=float)
    result = direct_omega_rollout(
        scenario, profiles[row["profile"]], bias, gain, cfg, model, design,
        omega, domain, fixed_rows, fixed_head, normals, bounds, support,
    )
    if result is None:
        raise RuntimeError(f"Previously saved direct-q candidate no longer safe: {row['id']}")
    return result


def choose_direct_candidates(direct_dir, concentration_path, baselines):
    chosen = {}
    for scenario in ("pressure_positive", "pressure_negative"):
        rows = read_csv(direct_dir / f"{scenario}_pareto_front.csv")
        baseline = baselines[scenario]
        eligible = [row for row in rows if
                    float(row["J_econ"]) < baseline["J_econ"] and
                    0.5 * (float(row["X2_IAE"]) / baseline["X2_IAE"] +
                           float(row["P2_IAE"]) / baseline["P2_IAE"]) < 1.0]
        if not eligible:
            chosen[scenario] = None
            continue
        # Representative: smallest mean normalized IAE among economic-positive
        # Pareto candidates. This is Level 1, not necessarily strict Level 2.
        chosen[scenario] = min(eligible, key=lambda row: 0.5 * (
            float(row["X2_IAE"]) / baseline["X2_IAE"] +
            float(row["P2_IAE"]) / baseline["P2_IAE"]
        ))
    chosen["concentration_positive"] = concentration_path
    return chosen


def peek(controller, state, action):
    z_saved = controller.inner.z.copy()
    was_saved = controller.was_in_omega
    control, info = controller.act(state, np.asarray(action), action_is_normalized=True)
    controller.inner.z = z_saved
    controller.was_in_omega = was_saved
    return control, info


def invert_at_state(controller, state, target, model):
    target_n = model.normalized_input(target)
    zero_control, _ = peek(controller, state, np.zeros(2))
    delta = target_n - model.normalized_input(zero_control)
    raw_direction = delta / np.asarray(controller.cfg.residual_action_scale)
    size = float(np.max(np.abs(raw_direction)))
    direction = raw_direction / size if size > 1e-12 else np.zeros(2)
    end_control, _ = peek(controller, state, direction)
    end_delta = model.normalized_input(end_control) - model.normalized_input(zero_control)
    denominator = float(end_delta @ end_delta)
    ray_fraction = (float(np.clip((delta @ end_delta) / denominator, 0, 1))
                    if denominator > 1e-15 else 0.0)
    ray_action = ray_fraction * direction
    seeds = np.array([
        [0, 0], ray_action, direction,
        [-1, -1], [-1, 1], [1, -1], [1, 1],
    ], dtype=float)
    def loss(action):
        control, _ = peek(controller, state, action)
        delta = model.normalized_input(control) - target_n
        return float(delta @ delta)
    values = np.array([loss(seed) for seed in seeds])
    best = seeds[int(np.argmin(values))].copy()
    best_value = float(np.min(values))
    # Ray inversion is exact where the current map is unsaturated and radial.
    # For off-ray QP projections, use a small dependency-free local correction.
    if best_value > 1e-12:
        point = best.copy()
        for step in (0.10, 0.02, 0.004):
            for _ in range(2):
                probes = [np.clip(point + step * np.asarray(d), -1, 1) for d in
                          ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1))]
                probe_values = np.array([loss(probe) for probe in probes])
                index = int(np.argmin(probe_values))
                if probe_values[index] >= best_value - 1e-13:
                    break
                point, best_value = probes[index], float(probe_values[index])
        best = point
    applied, info = controller.act(state, best, action_is_normalized=True)
    error = float(np.linalg.norm(applied - target))
    return best, applied, info, error


def inverse_trajectory(oracle_rows, cfg, model, design, omega, domain):
    controller = OmegaSafeOnlineController(cfg, model, design, omega, domain)
    controller.reset(cfg.robust_economic_reference_state.copy())
    output = []
    for row in oracle_rows:
        state = np.array([float(row["X2"]), float(row["P2"])])
        target = np.array([float(row["P100"]), float(row["F200"])])
        action, applied, info, error = invert_at_state(
            controller, state, target, model
        )
        output.append({
            "second": int(row["second"]), "scenario": row["scenario"],
            "X2_oracle": state[0], "P2_oracle": state[1],
            "P100_oracle": target[0], "F200_oracle": target[1],
            "actor_a_P100": float(action[0]), "actor_a_F200": float(action[1]),
            "P100_reconstructed": float(applied[0]),
            "F200_reconstructed": float(applied[1]),
            "reconstruction_error_physical": error,
            "reconstruction_error_normalized": float(np.linalg.norm(
                model.normalized_input(applied) - model.normalized_input(target)
            )),
            "mode": info["mode"], "qp_feasible": bool(info["qp_feasible"]),
            "in_Omega": bool(info["in_Omega"]),
            "in_Z": bool(info["in_Z"]),
            "z_X2": float(model.physical_state(info["z_before"])[0]),
            "z_P2": float(model.physical_state(info["z_before"])[1]),
        })
    return output


def inverse_summary(rows, tolerance=1e-3):
    errors = np.array([float(row["reconstruction_error_physical"]) for row in rows])
    actions = np.array([[float(row["actor_a_P100"]), float(row["actor_a_F200"])]
                        for row in rows])
    result = {
        "exact_tolerance_physical_input_l2": tolerance,
        "exact_realizable_step_fraction": float(np.mean(errors <= tolerance)),
        "mean_reconstruction_error_physical": float(np.mean(errors)),
        "max_reconstruction_error_physical": float(np.max(errors)),
        "actor_action_mean": np.mean(actions, axis=0).tolist(),
        "actor_action_std": np.std(actions, axis=0).tolist(),
        "actor_action_quantiles_05_50_95": np.quantile(
            actions, [0.05, 0.5, 0.95], axis=0
        ).tolist(),
        "actor_action_saturation_fraction_any_abs_ge_0_99": float(np.mean(
            np.max(np.abs(actions), axis=1) >= 0.99
        )),
        "mode_errors": {},
        "qp_infeasible_count": int(sum(
            str(row["qp_feasible"]).lower() == "false" for row in rows
        )),
    }
    for mode in sorted(set(row["mode"] for row in rows)):
        selected = np.array([float(row["reconstruction_error_physical"]) for row in rows
                             if row["mode"] == mode])
        result["mode_errors"][mode] = {
            "count": len(selected), "mean": float(np.mean(selected)),
            "max": float(np.max(selected)),
            "exact_fraction": float(np.mean(selected <= tolerance)),
        }
    return result


def actor_rollout(scenario, actions, cfg, model, design, omega, domain,
                  controller_factory=OmegaSafeOnlineController):
    controller = controller_factory(cfg, model, design, omega, domain)
    state = cfg.robust_economic_reference_state.copy()
    controller.reset(state)
    lower = model.physical_input(design.robust_input_lower)
    upper = model.physical_input(design.robust_input_upper)
    mid, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
    rows = []
    for second in range(300):
        state = apply_state_shock(state, scenario, second, scale=1.0)
        if (np.any(state < cfg.state_lower - 1e-7)
                or np.any(state > cfg.state_upper + 1e-7)):
            return None
        action = np.clip(actions[second], -1, 1)
        control, info = controller.act(state, action, action_is_normalized=True)
        next_state = model.step(state, control, cfg.disturbance_nominal)
        xn = model.normalized_state(next_state)
        mismatch = xn - (design.a @ model.normalized_state(state)
                         + design.b @ model.normalized_input(control)
                         + design.affine)
        if (not info["qp_feasible"] or not info["in_Omega"]
                or info["mode"] == "outside_certified_domain"
                or np.any(next_state < cfg.state_lower - 1e-7)
                or np.any(next_state > cfg.state_upper + 1e-7)
                or np.any(control < cfg.input_lower - 1e-7)
                or np.any(control > cfg.input_upper + 1e-7)
                or np.any(model.normalized_input(control) < design.robust_input_lower - 1e-7)
                or np.any(model.normalized_input(control) > design.robust_input_upper + 1e-7)
                or not contains(design.w_vertices, mismatch, tol=2e-7)
                or not contains(omega, xn - design.z_ref, tol=2e-7)):
            return None
        eta = np.abs(control - mid) / half
        rows.append({
            "scenario": scenario, "second": second,
            "X2": float(state[0]), "P2": float(state[1]),
            "P100": float(control[0]), "F200": float(control[1]),
            "a_P100": float(action[0]), "a_F200": float(action[1]),
            "mode": info["mode"], "in_Z": bool(info["in_Z"]),
            "in_Omega": bool(info["in_Omega"]),
            "qp_feasible": bool(info["qp_feasible"]),
            "robust_outer_10pct": bool(np.any(eta > 0.9)),
            "F200_robust_upper": bool(abs(control[1] - upper[1]) <= 1e-6),
            "economic_stage_cost": float(model.economic_cost(
                next_state, control, cfg.disturbance_nominal
            )),
        })
        state = next_state
    states = np.array([[row["X2"], row["P2"]] for row in rows])
    controls = np.array([[row["P100"], row["F200"]] for row in rows])
    metrics = {
        "J_econ": float(sum(row["economic_stage_cost"] for row in rows)),
        "X2_IAE": float(np.sum(np.abs(states[40:, 0] - cfg.robust_economic_reference_state[0]))),
        "P2_IAE": float(np.sum(np.abs(states[40:, 1] - cfg.robust_economic_reference_state[1]))),
        "P100_TV": float(np.sum(np.abs(np.diff(controls[:, 0])))),
        "F200_TV": float(np.sum(np.abs(np.diff(controls[:, 1])))),
        "robust_outer_10pct_steps": int(sum(r["robust_outer_10pct"] for r in rows)),
        "F200_robust_upper_steps": int(sum(r["F200_robust_upper"] for r in rows)),
        "Z_mode_steps": int(sum(r["mode"] == "Z_mode_existing_controller" for r in rows)),
        "Omega_mode_steps": int(sum(r["mode"] == "Omega_safe_one_step_QP" for r in rows)),
    }
    return metrics, rows


def classify_levels(metrics, baseline):
    economy = metrics["J_econ"] < baseline["J_econ"] - 1e-5
    ratios = [metrics[key] / baseline[key] for key in ("X2_IAE", "P2_IAE")]
    level1 = economy and np.mean(ratios) < 1.0
    level2 = economy and all(ratio <= 1.0 + 1e-8 for ratio in ratios)
    tv = all(metrics[key] <= baseline[key] * 1.1 + (
        1.0 if baseline[key] < 1e-9 else 0.0
    ) for key in ("P100_TV", "F200_TV"))
    occupancy_limit = (baseline["robust_outer_10pct_steps"] - 30
                       if baseline["robust_outer_10pct_steps"] >= 270
                       else baseline["robust_outer_10pct_steps"] + 30)
    return {
        "economy_better": bool(economy),
        "mean_normalized_IAE_ratio": float(np.mean(ratios)),
        "level_1": bool(level1), "level_2": bool(level2),
        "level_3": bool(level2 and tv and
                        metrics["robust_outer_10pct_steps"] <= occupancy_limit),
    }


def nondominated(rows, baseline):
    if not rows:
        return []
    values = np.array([[
        row["J_econ"] / baseline["J_econ"],
        row["X2_IAE"] / baseline["X2_IAE"],
        row["P2_IAE"] / baseline["P2_IAE"],
        row["P100_TV"] / max(baseline["P100_TV"], 1.0),
        row["F200_TV"] / max(baseline["F200_TV"], 1.0),
        row["robust_outer_10pct_steps"] / 300,
    ] for row in rows])
    selected = []
    for index, value in enumerate(values):
        dominates = np.all(values <= value + 1e-10, axis=1) & np.any(
            values < value - 1e-10, axis=1
        )
        if not np.any(dominates):
            selected.append(rows[index])
    return selected


def search_actor(scenario, seeds, baseline, cfg, model, design, omega, domain,
                 rng, budget, output_dir,
                 controller_factory=OmegaSafeOnlineController):
    evaluated = []
    trajectories = {}
    action_arrays = {}
    seen = set()
    def evaluate(name, actions):
        if len(evaluated) >= budget:
            return
        actions = np.clip(actions, -1, 1)
        key = actions.tobytes()
        if key in seen:
            return
        seen.add(key)
        result = actor_rollout(scenario, actions, cfg, model, design, omega, domain,
                               controller_factory)
        if result is None:
            return
        metrics, trajectory = result
        status = classify_levels(metrics, baseline)
        row = {"id": f"{scenario}_actor_{len(evaluated):04d}",
               "source": name, **metrics, **status}
        evaluated.append(row)
        trajectories[row["id"]] = trajectory
        action_arrays[row["id"]] = actions.copy()
    evaluate("zero_residual", np.zeros((300, 2)))
    for name, actions in seeds.items():
        evaluate(name, actions)
        for fraction in (0.25, 0.5, 0.75):
            evaluate(f"blend_zero_{name}_{fraction}", fraction * actions)
    names = list(seeds)
    trials = 0
    coarse_limit = min(budget, max(20, budget // 2))
    while len(evaluated) < coarse_limit and trials < budget * 4:
        trials += 1
        name = names[int(rng.integers(len(names)))]
        base = seeds[name]
        mix = rng.uniform(0.05, 1.0)
        actions = mix * base
        bias = rng.normal(0.0, rng.choice([0.05, 0.12, 0.25]), size=(4, 2))
        for phase in range(4):
            actions[PHASE_EDGES[phase]:PHASE_EDGES[phase + 1]] += bias[phase]
        evaluate(f"piecewise_{name}_{trials:04d}", actions)
    # Direct-shooting refinement changes individual time-indexed a_k values
    # (or short windows) in the full 300x2 action tensor, not direct q.
    while len(evaluated) < budget and trials < budget * 12:
        trials += 1
        if not evaluated:
            break
        weights = rng.dirichlet(np.ones(5))
        parent = min(evaluated, key=lambda row: (
            weights[0] * row["J_econ"] / baseline["J_econ"] +
            weights[1] * row["X2_IAE"] / baseline["X2_IAE"] +
            weights[2] * row["P2_IAE"] / baseline["P2_IAE"] +
            weights[3] * row["P100_TV"] / max(baseline["P100_TV"], 1.0) +
            weights[4] * row["robust_outer_10pct_steps"] / 300
        ))
        actions = action_arrays[parent["id"]].copy()
        start = int(rng.integers(0, 300))
        length = int(rng.choice([1, 2, 5, 10, 20, 40]))
        axis = int(rng.integers(0, 2))
        perturb = float(rng.choice([-1, 1]) * rng.choice([0.04, 0.12, 0.30]))
        actions[start:min(300, start + length), axis] += perturb
        evaluate(f"time_local_{parent['id']}_{start}_{length}_{axis}", actions)
    pareto = nondominated(evaluated, baseline)
    write_csv(output_dir / f"{scenario}_all_feasible.csv", evaluated)
    write_csv(output_dir / f"{scenario}_pareto.csv", pareto)
    selected = {}
    for level in (1, 2, 3):
        candidates = [row for row in evaluated if row[f"level_{level}"]]
        if not candidates:
            selected[f"level_{level}"] = None
            continue
        best = min(candidates, key=lambda row: (
            row["mean_normalized_IAE_ratio"], row["J_econ"]
        ))
        selected[f"level_{level}"] = best
        write_csv(output_dir / "selected_actor_trajectories" /
                  f"{scenario}_level_{level}.csv", trajectories[best["id"]])
    return {
        "safe_candidate_count": len(evaluated),
        "trial_count": trials,
        "pareto_count": len(pareto),
        "level_counts": {f"level_{level}": sum(
            row[f"level_{level}"] for row in evaluated
        ) for level in (1, 2, 3)},
        "selected": selected,
        "best_mean_IAE_economy_positive": min(
            (row for row in evaluated if row["economy_better"]),
            key=lambda row: row["mean_normalized_IAE_ratio"],
            default=None,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--direct-dir", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--concentration-trajectory", type=Path,
                        default=DEFAULT_CONCENTRATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--actor-budget", type=int, default=150)
    parser.add_argument("--recompute-inverse", action="store_true")
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
    baselines, profiles = {}, {}
    for scenario in SCENARIOS:
        baselines[scenario], _, profiles[scenario] = load_reference_profiles(
            scenario, cfg, model, design, omega, domain, args.seed
        )
    chosen = choose_direct_candidates(
        args.direct_dir, args.concentration_trajectory, baselines
    )
    summary = {
        "scope": "offline only; current online controller act() used for both audits",
        "reference": cfg.robust_economic_reference_state.tolist(),
        "exact_input_tolerance_physical_l2": 1e-3,
        "action_search_budget_per_scenario": args.actor_budget,
        "search_is_global_proof": False,
        "level_3_rule": "Level2 + each TV <=110% baseline (+1 absolute if zero) + robust outer-10% occupancy reduced >=30 steps if baseline>=270, else no more than baseline+30",
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        baseline = baselines[scenario]
        choice = chosen[scenario]
        if choice is None:
            summary["scenarios"][scenario] = {"direct_candidate": None}
            continue
        if isinstance(choice, Path):
            oracle = read_csv(choice)
            direct_id = choice.name
        else:
            _, oracle = recreate_direct_trajectory(
                choice, scenario, profiles[scenario], cfg, model, design,
                omega, domain
            )
            direct_id = choice["id"]
            write_csv(args.output_dir / "selected_direct_trajectories" /
                      f"{scenario}_{direct_id}.csv", oracle)
        inverse_path = args.output_dir / "inverse_action_trajectories" / f"{scenario}.csv"
        if inverse_path.is_file() and not args.recompute_inverse:
            inverse = read_csv(inverse_path)
        else:
            inverse = inverse_trajectory(oracle, cfg, model, design, omega, domain)
            write_csv(inverse_path, inverse)
        inverse_metrics = inverse_summary(inverse)
        seeds = {"inverse_direct_q": np.array([
            [float(row["actor_a_P100"]), float(row["actor_a_F200"])]
            for row in inverse
        ])}
        # Frozen actor requested action is read from a previously saved
        # deterministic evaluation; it is an offline seed, never retrained.
        action_search = search_actor(
            scenario, seeds, baseline, cfg, model, design, omega, domain,
            np.random.default_rng(args.seed + 1000 * len(summary["scenarios"])),
            args.actor_budget, args.output_dir
        )
        summary["scenarios"][scenario] = {
            "paired_zero_residual_baseline": baseline,
            "selected_direct_candidate": direct_id,
            "inverse_action_audit": inverse_metrics,
            "actor_action_oracle": action_search,
        }
        print(scenario, direct_id, "inverse exact",
              f"{inverse_metrics['exact_realizable_step_fraction']:.3f}",
              "actor levels", action_search["level_counts"], flush=True)
    (args.output_dir / "action_realizability_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(args.output_dir / "action_realizability_summary.json")


if __name__ == "__main__":
    main()
