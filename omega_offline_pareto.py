"""Offline, finite-budget B/Omega performance oracle; never trains or controls online.

This searches piecewise candidate input trajectories and projects every step
through the fixed Omega one-step QP. It is a numerical Pareto feasibility-witness
search, not a global nonexistence proof or an MPC controller.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .control import project_qp_2d
from .controlled_invariant_error_set import bounds_and_domains, contains, facets, hull
from .model import EvaporatorModel
from .paper2016_compare import SCENARIOS, apply_state_shock
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b
from .omega_safe_action_filter_diagnosis import qp_rows
from .train import ZeroResidualPolicy, run_episode
from .omega_online_closed_loop import OmegaSafeOnlineController


PHASE_EDGES = (0, 20, 40, 100, 300)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile_from_trajectory(path: Path, scenario: str, model, design) -> np.ndarray | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["scenario"] == scenario]
    if len(rows) != 300:
        return None
    controls = np.array([[float(row["P100"]), float(row["F200"])] for row in rows])
    return model.normalized_input(controls) - design.v_ref


def baseline_metrics(records, cfg, model, design):
    states = np.asarray([record["state"] for record in records])
    controls = np.asarray([record["control"] for record in records])
    lower = model.physical_input(design.robust_input_lower)
    upper = model.physical_input(design.robust_input_upper)
    eta = np.abs(controls - 0.5 * (lower + upper)) / (0.5 * (upper - lower))
    return {
        "J_econ": float(sum(record["economic_cost"] for record in records)),
        "X2_IAE": float(np.sum(np.abs(states[40:, 0] - cfg.robust_economic_reference_state[0]))),
        "P2_IAE": float(np.sum(np.abs(states[40:, 1] - cfg.robust_economic_reference_state[1]))),
        "X2_ISE": float(np.sum((states[40:, 0] - cfg.robust_economic_reference_state[0]) ** 2)),
        "P2_ISE": float(np.sum((states[40:, 1] - cfg.robust_economic_reference_state[1]) ** 2)),
        "P100_TV": float(np.sum(np.abs(np.diff(controls[:, 0])))),
        "F200_TV": float(np.sum(np.abs(np.diff(controls[:, 1])))),
        "robust_outer_10pct_steps": int(np.sum(np.any(eta > 0.9, axis=1))),
        "F200_robust_upper_steps": int(np.sum(np.abs(controls[:, 1] - upper[1]) <= 1e-6)),
    }


def direct_omega_rollout(scenario, profile, bias, gain, cfg, model, design, omega,
                         domain, fixed_rows, fixed_rhs_head, normals, bounds, support):
    state = cfg.robust_economic_reference_state.copy()
    lower = model.physical_input(design.robust_input_lower)
    upper = model.physical_input(design.robust_input_upper)
    middle, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
    trajectory = []
    for second in range(300):
        state = apply_state_shock(state, scenario, second, scale=1.0)
        x = model.normalized_state(state)
        xi = x - design.z_ref
        if (not contains(omega, xi, tol=2e-7)
                or np.any(state < cfg.state_lower - 1e-7)
                or np.any(state > cfg.state_upper + 1e-7)
                or np.any(x < design.robust_state_lower - 1e-7)
                or np.any(x > design.robust_state_upper + 1e-7)):
            return None
        phase = min(int(np.searchsorted(PHASE_EDGES[1:], second, side="right")), 3)
        candidate = profile[second] + bias[phase] + gain @ xi
        rhs = np.concatenate([
            fixed_rhs_head, bounds - normals @ (design.a @ xi) - support
        ])
        q, feasible = project_qp_2d(candidate, fixed_rows, rhs)
        if not feasible or np.max(fixed_rows @ q - rhs) > 2e-7:
            return None
        control = model.physical_input(design.v_ref + q)
        next_state = model.step(state, control, cfg.disturbance_nominal)
        xn = model.normalized_state(next_state)
        mismatch = xn - (design.a @ x + design.b @ (design.v_ref + q) + design.affine)
        if (np.any(next_state < cfg.state_lower - 1e-7)
                or np.any(next_state > cfg.state_upper + 1e-7)
                or np.any(control < cfg.input_lower - 1e-7)
                or np.any(control > cfg.input_upper + 1e-7)
                or np.any(xn < design.robust_state_lower - 1e-7)
                or np.any(xn > design.robust_state_upper + 1e-7)
                or np.any(design.v_ref + q < design.robust_input_lower - 1e-7)
                or np.any(design.v_ref + q > design.robust_input_upper + 1e-7)
                or not contains(design.w_vertices, mismatch, tol=2e-7)
                or not contains(omega, xn - design.z_ref, tol=2e-7)):
            return None
        eta = np.abs(control - middle) / half
        trajectory.append({
            "scenario": scenario, "second": second,
            "X2": float(state[0]), "P2": float(state[1]),
            "P100": float(control[0]), "F200": float(control[1]),
            "q_candidate_P100": float(candidate[0]),
            "q_candidate_F200": float(candidate[1]),
            "q_safe_P100": float(q[0]), "q_safe_F200": float(q[1]),
            "Omega_member_before": True,
            "Omega_member_next": True,
            "robust_outer_10pct": bool(np.any(eta > 0.9)),
            "F200_robust_upper": bool(abs(control[1] - upper[1]) <= 1e-6),
            "economic_stage_cost": float(model.economic_cost(
                next_state, control, cfg.disturbance_nominal
            )),
        })
        state = next_state
    states = np.asarray([[row["X2"], row["P2"]] for row in trajectory])
    controls = np.asarray([[row["P100"], row["F200"]] for row in trajectory])
    metrics = {
        "J_econ": float(sum(row["economic_stage_cost"] for row in trajectory)),
        "X2_IAE": float(np.sum(np.abs(states[40:, 0] - cfg.robust_economic_reference_state[0]))),
        "P2_IAE": float(np.sum(np.abs(states[40:, 1] - cfg.robust_economic_reference_state[1]))),
        "X2_ISE": float(np.sum((states[40:, 0] - cfg.robust_economic_reference_state[0]) ** 2)),
        "P2_ISE": float(np.sum((states[40:, 1] - cfg.robust_economic_reference_state[1]) ** 2)),
        "P100_TV": float(np.sum(np.abs(np.diff(controls[:, 0])))),
        "F200_TV": float(np.sum(np.abs(np.diff(controls[:, 1])))),
        "robust_outer_10pct_steps": int(sum(row["robust_outer_10pct"] for row in trajectory)),
        "F200_robust_upper_steps": int(sum(row["F200_robust_upper"] for row in trajectory)),
    }
    return metrics, trajectory


def classify(metrics, baseline):
    econ = metrics["J_econ"] < baseline["J_econ"] - 1e-5
    recovery = all(metrics[key] <= baseline[key] + 1e-6 for key in ("X2_IAE", "P2_IAE"))
    tv_exact = all(metrics[key] <= baseline[key] + 1e-6 for key in ("P100_TV", "F200_TV"))
    tv_reasonable = all(
        metrics[key] <= baseline[key] * 1.10 + (1.0 if baseline[key] < 1e-9 else 0.0)
        for key in ("P100_TV", "F200_TV")
    )
    occupancy_limit = (
        baseline["robust_outer_10pct_steps"] - 30
        if baseline["robust_outer_10pct_steps"] >= 270
        else baseline["robust_outer_10pct_steps"] + 30
    )
    occupancy_relief = metrics["robust_outer_10pct_steps"] <= occupancy_limit
    return {
        "economy_better": econ,
        "recovery_no_worse_both_IAE": recovery,
        "TV_no_worse_exact_both": tv_exact,
        "reasonable_TV": tv_reasonable,
        "meaningful_or_nonworsening_occupancy": occupancy_relief,
        "economy_and_recovery": econ and recovery,
        "economy_and_TV": econ and tv_exact,
        "economy_recovery_reasonable_activity": (
            econ and recovery and tv_reasonable and occupancy_relief
        ),
    }


def objective_vector(metrics, baseline):
    return np.array([
        metrics["J_econ"] / baseline["J_econ"],
        0.5 * (metrics["X2_IAE"] / baseline["X2_IAE"]
               + metrics["P2_IAE"] / baseline["P2_IAE"]),
        0.5 * (metrics["P100_TV"] / max(baseline["P100_TV"], 1.0)
               + metrics["F200_TV"] / max(baseline["F200_TV"], 1.0)),
        metrics["robust_outer_10pct_steps"] / 300.0,
    ])


def nondominated_indices(vectors):
    values = np.asarray(vectors)
    keep = []
    for index, vector in enumerate(values):
        dominated = np.any(
            np.all(values <= vector + 1e-10, axis=1)
            & np.any(values < vector - 1e-10, axis=1)
        )
        if not dominated:
            keep.append(index)
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_controlled_invariant_error_B/Omega_anchor_B_vertices.csv"
    ))
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_omega_offline_pareto_B"
    ))
    parser.add_argument("--max-candidates-per-scenario", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                        default=list(SCENARIOS))
    args = parser.parse_args()
    if args.max_candidates_per_scenario < 4:
        raise ValueError("Need at least four candidates per scenario")
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed",
                           seed=args.seed, steps_per_episode=300)
    cfg.residual_parameterization = "state_dependent_polytope"
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    domain, _ = bounds_and_domains(cfg, model, design)
    normals, bounds = facets(omega)
    support = np.max(design.w_vertices @ normals.T, axis=0)
    rows = np.vstack([np.eye(2), -np.eye(2), normals @ design.b])
    rhs_head = np.concatenate([domain["q_upper"], -domain["q_lower"]])
    check_rows, check_rhs = qp_rows(np.zeros(2), omega, design, domain)
    if not (np.allclose(rows, check_rows) and np.allclose(
        np.concatenate([rhs_head, bounds - support]), check_rhs
    )):
        raise RuntimeError("Offline QP rows differ from verified Omega QP")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": "offline direct-q Omega-safe oracle; no SAC training or online change",
        "reference": cfg.robust_economic_reference_state.tolist(),
        "search_is_global_proof": False,
        "candidate_budget_per_scenario": args.max_candidates_per_scenario,
        "policy_family": "four-phase additive direct-q plus optional static state feedback over baseline/reference/old/new profiles; projected by fixed Omega one-step QP",
        "phase_edges_seconds": list(PHASE_EDGES),
        "classification": (
            "economy=J_econ strictly below paired baseline; recovery=both X2/P2 IAE no worse; "
            "TV_no_worse=both exact; reasonable_activity=both TV within 10% (+1 absolute "
            "when baseline TV=0), plus >=30 s occupancy relief if baseline occupies "
            ">=270 s, otherwise <=baseline+30 s"
        ),
        "scenarios": {},
    }
    rng = np.random.default_rng(args.seed)
    for scenario in args.scenarios:
        controller = OmegaSafeOnlineController(cfg, model, design, omega, domain)
        _, baseline_records, _ = run_episode(
            cfg, model, controller, ZeroResidualPolicy(), None,
            np.random.default_rng(args.seed + 31000), training=False,
            global_step=0, paper_scenario=scenario,
        )
        baseline = baseline_metrics(baseline_records, cfg, model, design)
        baseline_profile = model.normalized_input(np.asarray([
            record["control"] for record in baseline_records
        ])) - design.v_ref
        profiles = {
            "zero_residual_profile": baseline_profile,
            "reference_q_zero": np.zeros((300, 2)),
        }
        trajectory_sources = {
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
        for name, path in trajectory_sources.items():
            profile = profile_from_trajectory(path, scenario, model, design)
            if profile is not None:
                profiles[name] = profile
                for fraction in (0.05, 0.10, 0.20, 0.40, 0.60, 0.80):
                    profiles[f"blend_zero_{name}_{fraction:.2f}"] = (
                        (1.0 - fraction) * baseline_profile + fraction * profile
                    )
        evaluated = []
        trajectory_by_id = {}
        trial_count = 0
        seen = set()

        def evaluate(name, profile_name, bias, gain=None):
            nonlocal trial_count
            if gain is None:
                gain = np.zeros((2, 2))
            key = (profile_name, tuple(np.round(bias.ravel(), 8)),
                   tuple(np.round(gain.ravel(), 8)))
            if trial_count >= args.max_candidates_per_scenario:
                return
            trial_count += 1
            if key in seen:
                return
            seen.add(key)
            output = direct_omega_rollout(
                scenario, profiles[profile_name], bias, gain, cfg, model, design,
                omega, domain, rows, rhs_head, normals, bounds, support,
            )
            if output is None:
                return
            metrics, trajectory = output
            status = classify(metrics, baseline)
            candidate_id = f"{scenario}_{trial_count:04d}"
            record = {
                "id": candidate_id, "source": name, "profile": profile_name,
                "phase_bias_json": json.dumps(bias.tolist()),
                "state_feedback_gain_json": json.dumps(gain.tolist()),
                **metrics, **status,
            }
            objective = objective_vector(metrics, baseline)
            record.update({
                "objective_economy_ratio": objective[0],
                "objective_state_IAE_ratio_mean": objective[1],
                "objective_input_TV_ratio_mean": objective[2],
                "objective_bound_occupancy_fraction": objective[3],
            })
            evaluated.append(record)
            trajectory_by_id[candidate_id] = trajectory

        zero_bias = np.zeros((4, 2))
        for name in profiles:
            evaluate("unmodified_profile", name, zero_bias)
        # A reproducible coarse 2D×time Pareto sweep, not a single scalarized
        # optimizer.  Candidate profiles include zero, reference and frozen
        # actor trajectories; each gets symmetric coordinate and random probes.
        profile_names = list(profiles)
        for scale in (0.02, 0.05, 0.10):
            for phase in range(4):
                for axis in range(2):
                    for sign in (-1.0, 1.0):
                        if trial_count >= args.max_candidates_per_scenario:
                            break
                        bias = np.zeros((4, 2))
                        bias[phase, axis] = sign * scale
                        profile_name = profile_names[(phase + axis) % len(profile_names)]
                        evaluate("coordinate_sweep", profile_name, bias)
        random_limit = max(trial_count, int(0.75 * args.max_candidates_per_scenario))
        while trial_count < random_limit:
            bias = rng.normal(0.0, rng.choice([0.02, 0.05, 0.10]), size=(4, 2))
            profile_name = profile_names[int(rng.integers(len(profile_names)))]
            gain = (rng.normal(0.0, rng.choice([0.0, 0.5, 1.5]), size=(2, 2))
                    if rng.random() < 0.7 else np.zeros((2, 2)))
            evaluate("random_pareto_sweep", profile_name, bias, gain)
        # Epsilon-style local search: reduce constraint violations before
        # economic cost.  Separate recovery, TV, and occupancy priorities keep
        # this from being a single weighted-sum optimization.
        priorities = ("recovery", "TV", "joint", "economy")
        while trial_count < args.max_candidates_per_scenario:
            priority = priorities[trial_count % len(priorities)]
            def score(record):
                econ_gap = max(0.0, record["J_econ"] / baseline["J_econ"] - 1.0)
                recovery_gap = sum(max(0.0, record[key] / baseline[key] - 1.0)
                                   for key in ("X2_IAE", "P2_IAE"))
                tv_gap = sum(max(0.0, record[key] / max(baseline[key], 1.0) - 1.0)
                             for key in ("P100_TV", "F200_TV"))
                activity_gap = max(0.0, (
                    record["robust_outer_10pct_steps"]
                    - max(0, baseline["robust_outer_10pct_steps"] - 30)
                ) / 300.0)
                if priority == "recovery":
                    return (recovery_gap, econ_gap, record["J_econ"])
                if priority == "TV":
                    return (tv_gap, econ_gap, record["J_econ"])
                if priority == "joint":
                    return (recovery_gap + tv_gap + activity_gap, econ_gap,
                            record["J_econ"])
                return (econ_gap, record["J_econ"], recovery_gap)
            seed = min(evaluated, key=score)
            bias = np.asarray(json.loads(seed["phase_bias_json"]), dtype=float)
            gain = np.asarray(json.loads(seed["state_feedback_gain_json"]), dtype=float)
            if rng.random() < 0.8:
                phase, axis = int(rng.integers(4)), int(rng.integers(2))
                bias[phase, axis] += rng.choice([-1.0, 1.0]) * rng.choice([0.005, 0.015, 0.03])
            else:
                axis, state_axis = int(rng.integers(2)), int(rng.integers(2))
                gain[axis, state_axis] += rng.choice([-1.0, 1.0]) * rng.choice([0.1, 0.3, 0.7])
            evaluate("epsilon_local_refinement", seed["profile"], bias, gain)
        if not evaluated:
            raise RuntimeError(f"No Omega-safe candidate rollout for {scenario}")
        objectives = [objective_vector(record, baseline) for record in evaluated]
        pareto = [evaluated[index] for index in nondominated_indices(objectives)]
        write_csv(args.output_dir / f"{scenario}_all_feasible_candidates.csv", evaluated)
        write_csv(args.output_dir / f"{scenario}_pareto_front.csv", pareto)
        selected = {}
        for criterion in (
            "economy_and_recovery", "economy_and_TV",
            "economy_recovery_reasonable_activity",
        ):
            feasible = [record for record in evaluated if record[criterion]]
            selected[criterion] = min(feasible, key=lambda item: item["J_econ"]) if feasible else None
        selected["lowest_economic_cost"] = min(evaluated, key=lambda item: item["J_econ"])
        selected["lowest_state_IAE_ratio"] = min(evaluated, key=lambda item: item["objective_state_IAE_ratio_mean"])
        selected["lowest_input_TV_ratio"] = min(evaluated, key=lambda item: item["objective_input_TV_ratio_mean"])
        selected["lowest_bound_occupancy"] = min(evaluated, key=lambda item: item["robust_outer_10pct_steps"])
        trajectory_dir = args.output_dir / "selected_oracle_trajectories"
        for criterion, record in selected.items():
            if record is not None:
                write_csv(
                    trajectory_dir / f"{scenario}_{criterion}.csv",
                    trajectory_by_id[record["id"]],
                )
        summary["scenarios"][scenario] = {
            "paired_zero_residual_baseline": baseline,
            "attempted_candidate_count": trial_count,
            "fully_safe_candidate_count": len(evaluated),
            "pareto_candidate_count": len(pareto),
            "criteria_witness_count": {
                key: sum(record[key] for record in evaluated)
                for key in ("economy_and_recovery", "economy_and_TV",
                            "economy_recovery_reasonable_activity")
            },
            "selected_candidates": selected,
            "interpretation_if_no_witness": (
                "No witness in this finite direct-q policy family and budget; "
                "not a proof of global infeasibility."
            ),
        }
        (args.output_dir / "oracle_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"{scenario}: attempted={trial_count} safe={len(evaluated)} "
              f"pareto={len(pareto)} witnesses="
              f"{summary['scenarios'][scenario]['criteria_witness_count']}", flush=True)
    print(f"Wrote {args.output_dir / 'oracle_summary.json'}")


if __name__ == "__main__":
    main()
