"""Offline segmented-actor Pareto feasibility audit for frozen dwell20 safety.

This is direct shooting, not an online controller or SAC training. Every trial
uses the production interior-anchor -> Gm/Bj -> one-step QP -> nonlinear plant
path through train.run_episode. State jumps remain plant-side only.
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
from .omega_constrained_objective import allowed_extra_tv
from .omega_dwell_shock_sets import DEFAULT_OUTPUT as DWELL_SETS
from .omega_sac_train import controller
from .paper2016_compare import SCENARIOS
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b
from .train import run_episode


DEFAULT_OUTPUT = REPO_DIR / "evaporation_safe_sac/outputs_omega_segment_pareto_B"
DEFAULT_OMEGA = REPO_DIR / (
    "evaporation_safe_sac/outputs_controlled_invariant_error_B/"
    "Omega_anchor_B_vertices.csv"
)
DEFAULT_TRAJECTORY = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_safe_B_dwell20_obs23_constrained_seed42_300x300/"
    "fixed_trajectories/episode_0300.csv"
)
SAFETY_KEYS = (
    "omega_exit_count", "qp_infeasible_rate", "violation_rate",
    "robust_operating_region_violation_rate", "disturbance_bound_exceedance_rate",
    "supervisor_Gm_exit_count", "supervisor_Bk_recovery_failure_count",
    "supervisor_omega_exit_count", "supervisor_QP_infeasible_count",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class SegmentedPolicy:
    def __init__(self, segments: np.ndarray, segment_seconds: int):
        self.segments = np.asarray(segments, dtype=np.float32)
        self.segment_seconds = segment_seconds
        self.step = 0

    def select_action(self, _observation, deterministic=True):
        action = self.segments[min(self.step // self.segment_seconds,
                                   len(self.segments) - 1)]
        self.step += 1
        return action.copy()


def metrics(stat, records, cfg, lower, upper):
    states = np.asarray([r["state"] for r in records], dtype=float)
    controls = np.asarray([r["control"] for r in records], dtype=float)
    delta = np.diff(controls, axis=0)
    reference = np.asarray(cfg.robust_economic_reference_state)
    error = states[40:] - reference
    mid, half = (lower + upper) / 2, (upper - lower) / 2
    eta = np.abs(controls - mid) / half
    signs = np.sign(delta)
    # Deadband avoids counting floating-point jitter as a physical reversal.
    signs[np.abs(delta) <= 1e-6] = 0
    sign_changes = []
    for axis in range(2):
        active = signs[:, axis][signs[:, axis] != 0]
        sign_changes.append(int(np.sum(active[1:] != active[:-1])))
    result = {
        "J_econ": float(stat["J_econ"]),
        "X2_IAE": float(np.sum(np.abs(error[:, 0]))),
        "P2_IAE": float(np.sum(np.abs(error[:, 1]))),
        "X2_ISE": float(np.sum(error[:, 0] ** 2)),
        "P2_ISE": float(np.sum(error[:, 1] ** 2)),
        "P100_TV": float(np.sum(np.abs(delta[:, 0]))),
        "F200_TV": float(np.sum(np.abs(delta[:, 1]))),
        "P100_delta_RMS": float(np.sqrt(np.mean(delta[:, 0] ** 2))),
        "F200_delta_RMS": float(np.sqrt(np.mean(delta[:, 1] ** 2))),
        "P100_input_sign_changes": sign_changes[0],
        "F200_input_sign_changes": sign_changes[1],
        "P100_upper_steps": int(np.sum(np.abs(controls[:, 0] - upper[0]) <= 1e-6)),
        "P100_lower_steps": int(np.sum(np.abs(controls[:, 0] - lower[0]) <= 1e-6)),
        "F200_upper_steps": int(np.sum(np.abs(controls[:, 1] - upper[1]) <= 1e-6)),
        "F200_lower_steps": int(np.sum(np.abs(controls[:, 1] - lower[1]) <= 1e-6)),
        "P100_outer_10pct_steps": int(np.sum(eta[:, 0] > 0.9)),
        "F200_outer_10pct_steps": int(np.sum(eta[:, 1] > 0.9)),
        "any_outer_10pct_steps": int(np.sum(np.any(eta > 0.9, axis=1))),
        "recovery_max_steps": int(stat["supervisor_recovery_max_steps"]),
        "all_three_recovered_within_D": bool(stat["supervisor_all_three_recovered_within_D"]),
        "nonlinear_W_outside_steps": int(sum(r["supervisor_nonlinear_W_exceedance"]
                                              for r in records)),
        "physical_state_violation_steps": int(sum(r["physical_state_violation"]
                                                    for r in records)),
        "physical_input_violation_steps": int(sum(r["physical_input_violation"]
                                                    for r in records)),
        "robust_region_violation_steps": int(sum(r["robust_region_violation"]
                                                 for r in records)),
        **{key: float(stat[key]) for key in SAFETY_KEYS},
    }
    result["safety_passed"] = bool(
        result["all_three_recovered_within_D"]
        and result["nonlinear_W_outside_steps"] == 0
        and result["physical_state_violation_steps"] == 0
        and result["physical_input_violation_steps"] == 0
        and result["robust_region_violation_steps"] == 0
        and all(result[key] == 0 for key in SAFETY_KEYS)
    )
    return result


def classify(m, baseline):
    econ = 100 * (baseline["J_econ"] - m["J_econ"]) / abs(baseline["J_econ"])
    ratios = {key + "_ratio": m[key] / max(baseline[key], 1e-12)
              for key in ("X2_IAE", "P2_IAE", "X2_ISE", "P2_ISE")}
    a = econ > 1e-9 and ratios["X2_IAE_ratio"] <= 1 + 1e-9 and ratios["P2_IAE_ratio"] <= 1 + 1e-9
    recovery_10pct = (ratios["X2_IAE_ratio"] <= 1.1 + 1e-9
                      and ratios["P2_IAE_ratio"] <= 1.1 + 1e-9)
    activity_ok = all(
        m[key] <= baseline[key] + allowed_extra_tv(baseline[key]) + 1e-8
        for key in ("P100_TV", "F200_TV")
    )
    # A constantly saturated zero-residual baseline must not make Level C
    # vacuous: reuse the prior offline Pareto rule in that special case.
    boundary_limit = (baseline["any_outer_10pct_steps"] - 30
                      if baseline["any_outer_10pct_steps"] >= 270
                      else baseline["any_outer_10pct_steps"] + 30)
    boundary_ok = m["any_outer_10pct_steps"] <= boundary_limit + 1e-8
    return {"economic_improvement_percent": float(econ), **ratios,
            "level_A": bool(a),
            "level_B": bool(econ > 1e-9 and recovery_10pct and activity_ok),
            "level_C": bool(econ > 1e-9 and recovery_10pct and activity_ok and boundary_ok),
            "activity_within_comprehensive_tolerance": bool(activity_ok),
            "boundary_within_level_C_tolerance": bool(boundary_ok),
            "level_C_boundary_limit_steps": int(boundary_limit)}


def pareto_vectors(rows, baseline):
    columns = ("J_econ", "X2_IAE", "P2_IAE", "X2_ISE", "P2_ISE",
               "P100_TV", "F200_TV", "any_outer_10pct_steps")
    return np.array([[row[k] / max(abs(baseline[k]), 1.0) for k in columns]
                     for row in rows], dtype=float)


def front_indices(values):
    result = []
    for i, value in enumerate(values):
        dominates = (np.all(values <= value + 1e-10, axis=1)
                     & np.any(values < value - 1e-10, axis=1))
        if not np.any(dominates):
            result.append(i)
    return result


def plot_front(path: Path, rows: list[dict], front: list[dict]):
    """Economic/recovery trade-off; marker size reflects total input TV."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    x = [float(r["economic_improvement_percent"]) for r in rows]
    y = [(float(r["X2_IAE_ratio"]) + float(r["P2_IAE_ratio"])) / 2
         for r in rows]
    tv = np.array([float(r["P100_TV"]) + float(r["F200_TV"])
                   for r in rows])
    sizes = 15 + 90 * np.minimum(tv / max(np.quantile(tv, 0.9), 1), 2)
    ax.scatter(x, y, s=sizes, alpha=0.22, color="gray", label="safe candidates")
    for label, color, marker in (("A", "tab:red", "o"),
                                  ("B", "tab:blue", "s"),
                                  ("C", "tab:green", "^")):
        selected = [r for r in front
                    if r[f"level_{label}"] is True
                    or r[f"level_{label}"] == "True"]
        if selected:
            ax.scatter([float(r["economic_improvement_percent"]) for r in selected],
                       [(float(r["X2_IAE_ratio"]) + float(r["P2_IAE_ratio"])) / 2
                        for r in selected], marker=marker, color=color,
                       edgecolors="black", s=70, label=f"Pareto Level {label}")
    ax.axvline(0, color="black", lw=0.8)
    ax.axhline(1, color="black", lw=0.8)
    ax.set(xlabel="economic improvement vs paired zero residual (%)",
           ylabel="mean normalized X2/P2 IAE ratio",
           title="Segmented actor-action offline Pareto search")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def actor_seed(path: Path, scenario: str, segment_seconds: int):
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["scenario"] == scenario]
    if len(rows) != 300:
        return None
    actions = np.array([[float(r["actor_raw_a_P100"]),
                         float(r["actor_raw_a_F200"])] for r in rows])
    return np.array([np.mean(actions[i:i + segment_seconds], axis=0)
                     for i in range(0, 300, segment_seconds)])


def search_one(scenario, segment_seconds, budget, cfg, model, design, omega,
               domain, baseline, lower, upper, seed_actions, rng, output):
    nseg = 300 // segment_seconds
    actor = controller(cfg, model, design, omega, domain,
                       action_mapping="interior_anchor", supervisor="dwell20",
                       dwell_sets=DWELL_SETS, observation_variant="supervisor23")
    rows, segments_by_id, trajectories, failures, seen = [], {}, {}, [], set()

    def evaluate(name, segments):
        if len(rows) + len(failures) >= budget:
            return
        segments = np.clip(np.asarray(segments, dtype=np.float32), -1, 1)
        key = segments.tobytes()
        if key in seen:
            return
        seen.add(key)
        policy = SegmentedPolicy(segments, segment_seconds)
        try:
            stat, records, _ = run_episode(
                cfg, model, actor, policy, None, np.random.default_rng(42),
                training=False, global_step=0, paper_scenario=scenario,
            )
            m = metrics(stat, records, cfg, lower, upper)
            if not m["safety_passed"]:
                raise RuntimeError("safety counters or recovery failed")
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failures.append({"source": name, "reason": str(exc)[:240]})
            return
        row_id = f"{scenario}_{segment_seconds}s_{len(rows):05d}"
        row = {"id": row_id, "source": name, "segment_seconds": segment_seconds,
               **m, **classify(m, baseline)}
        rows.append(row)
        segments_by_id[row_id] = segments.copy()
        trajectories[row_id] = [{"second": i, "X2": r["state"][0],
                                 "P2": r["state"][1],
                                 "P100": r["control"][0],
                                 "F200": r["control"][1],
                                 "a_P100": segments[i // segment_seconds, 0],
                                 "a_F200": segments[i // segment_seconds, 1],
                                 "supervisor_mode": r["supervisor_mode"]}
                                for i, r in enumerate(records)]

    zero = np.zeros((nseg, 2))
    evaluate("zero_residual", zero)
    if seed_actions is not None:
        evaluate("constrained_ep300_segment_mean", seed_actions)
        for factor in (0.25, 0.5, 0.75):
            evaluate(f"constrained_blend_{factor}", factor * seed_actions)
    for axis in range(2):
        for sign in (-1, 1):
            for strength in (0.15, 0.4, 0.8):
                candidate = zero.copy()
                candidate[:, axis] = sign * strength
                evaluate(f"constant_{axis}_{sign}_{strength}", candidate)
    # Multiobjective archive search: no scalar weighted-sum objective. Parents
    # come from the nondominated set, with tournament pressure for uncovered
    # level-A/B/C conditions and per-objective extremes.
    attempts = 0
    while len(rows) + len(failures) < budget and attempts < budget * 15:
        attempts += 1
        if not rows:
            parent = zero
        else:
            values = pareto_vectors(rows, baseline)
            front = front_indices(values)
            if rng.random() < 0.35:
                chosen = int(np.argmin(values[:, int(rng.integers(values.shape[1]))]))
            elif rng.random() < 0.35:
                chosen = int(rng.integers(len(rows)))
            else:
                chosen = int(rng.choice(front))
            parent = segments_by_id[rows[chosen]["id"]]
        candidate = parent.copy()
        if rng.random() < 0.15:
            candidate *= float(rng.uniform(0.3, 1.0))
        if rng.random() < 0.20 and len(rows) > 1:
            other = segments_by_id[rows[int(rng.integers(len(rows)))]["id"]]
            cut = int(rng.integers(1, nseg))
            candidate[cut:] = other[cut:]
        width = int(rng.choice([1, 2, 3, 5, 10, nseg]))
        start = int(rng.integers(0, nseg - min(width, nseg) + 1))
        axis = int(rng.integers(0, 3))
        magnitude = float(rng.choice([0.04, 0.10, 0.25, 0.5]))
        perturb = rng.normal(0.0, magnitude, size=(1, 2))
        if axis < 2:
            perturb[0, 1 - axis] = 0
        candidate[start:start + width] += perturb
        evaluate(f"archive_mutation_{attempts}", candidate)

    values = pareto_vectors(rows, baseline)
    front = [rows[i] for i in front_indices(values)]
    prefix = output / f"{scenario}_{segment_seconds}s"
    write_csv(prefix.with_name(prefix.name + "_all.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_pareto.csv"), front)
    write_csv(prefix.with_name(prefix.name + "_failures.csv"), failures)
    plot_front(prefix.with_name(prefix.name + "_pareto.png"), rows, front)
    representatives = {}
    for level in ("A", "B", "C"):
        eligible = [r for r in rows if r[f"level_{level}"]]
        if eligible:
            chosen = min(eligible, key=lambda r: (
                r["X2_IAE_ratio"] + r["P2_IAE_ratio"],
                -r["economic_improvement_percent"],
                r["P100_TV"] + r["F200_TV"],
            ))
            representatives[level] = chosen["id"]
            write_csv(output / "witness_trajectories" / f"{chosen['id']}.csv",
                      trajectories[chosen["id"]])
            write_csv(output / "witness_actions" / f"{chosen['id']}.csv", [
                {"segment": i, "start_second": i * segment_seconds,
                 "a_P100": a[0], "a_F200": a[1]}
                for i, a in enumerate(segments_by_id[chosen["id"]])
            ])
        else:
            representatives[level] = None
    return {"scenario": scenario, "segment_seconds": segment_seconds,
            "attempted": len(rows) + len(failures), "safe_candidates": len(rows),
            "failed_candidates": len(failures), "pareto_count": len(front),
            "level_counts": {level: sum(r[f"level_{level}"] for r in rows)
                             for level in ("A", "B", "C")},
            "representative_witnesses": representatives,
            "best_economy_percent": max(r["economic_improvement_percent"] for r in rows),
            "best_X2_ratio_among_economic_positive": min(
                (r["X2_IAE_ratio"] for r in rows
                 if r["economic_improvement_percent"] > 0), default=None),
            "best_P2_ratio_among_economic_positive": min(
                (r["P2_IAE_ratio"] for r in rows
                 if r["economic_improvement_percent"] > 0), default=None)}


def oscillation_diagnosis(path: Path, output: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["scenario"] == "pressure_positive"]
    if len(rows) != 300:
        raise RuntimeError("Expected 300 pressure+ rows in episode-300 trajectory")
    u = np.array([float(r["P100"]) for r in rows])
    delta = np.diff(u)
    active = np.sign(delta[np.abs(delta) > 1e-6])
    centered = u - np.mean(u)
    fft = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(u), d=1.0)
    fft[0] = 0
    dominant = int(np.argmax(fft))
    high = frequencies >= 0.25
    result = {
        "P100_TV": float(np.sum(np.abs(delta))),
        "RMS_delta_P100": float(np.sqrt(np.mean(delta ** 2))),
        "max_abs_delta_P100": float(np.max(np.abs(delta))),
        "sign_flip_count": int(np.sum(active[1:] != active[:-1])),
        "lag1_P100_correlation": float(np.corrcoef(u[:-1], u[1:])[0, 1]),
        "lag1_delta_P100_correlation": float(np.corrcoef(delta[:-1], delta[1:])[0, 1]),
        "dominant_frequency_hz": float(frequencies[dominant]),
        "dominant_period_seconds": float(1 / frequencies[dominant]),
        "high_frequency_power_fraction_ge_0p25_hz": float(np.sum(fft[high]) / np.sum(fft)),
        "TV_from_steps_abs_delta_gt_20_fraction": float(
            np.sum(np.abs(delta)[np.abs(delta) > 20]) / np.sum(np.abs(delta))),
    }
    write_csv(output / "pressure_positive_ep300_P100_delta.csv", [
        {"second": i + 1, "P100": u[i + 1], "delta_P100": delta[i]}
        for i in range(len(delta))
    ])
    (output / "pressure_positive_ep300_oscillation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].plot(np.arange(1, 300), delta, lw=0.7)
        axes[0].set(ylabel="Delta P100", xlabel="second")
        axes[1].plot(frequencies[1:], fft[1:], lw=0.8)
        axes[1].set(xlabel="frequency (Hz)", ylabel="FFT power")
        fig.tight_layout()
        fig.savefig(output / "pressure_positive_ep300_oscillation.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-10s", type=int, default=250)
    parser.add_argument("--budget-5s", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                        default=list(SCENARIOS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--constrained-trajectory", type=Path,
                        default=DEFAULT_TRAJECTORY)
    args = parser.parse_args()
    if args.budget_10s < 1 or args.budget_5s < 0:
        raise ValueError("Search budgets must be positive/nonnegative")
    if 300 % 10 or 300 % 5:
        raise AssertionError("Segment length must divide Paper2016 horizon")
    if (args.output_dir / "summary.json").exists():
        raise RuntimeError("Output already contains a completed search; use a fresh directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed",
                           episodes=1, steps_per_episode=300, seed=args.seed,
                           residual_parameterization="state_dependent_polytope")
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    cfg.omega_performance_shaping_enabled = True
    cfg.abort_on_first_uncertified_step = True
    cfg.omega_exit_abort_count = 3
    cfg.omega_outside_abort_steps = 10
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    domain, _ = bounds_and_domains(cfg, model, design)
    if not certificate(omega, domain, design)["passed"]:
        raise RuntimeError("Frozen Omega certificate failed")
    lower = model.physical_input(design.robust_input_lower)
    upper = model.physical_input(design.robust_input_upper)
    rng = np.random.default_rng(args.seed)
    diagnostics = oscillation_diagnosis(args.constrained_trajectory, args.output_dir)
    summaries = []
    for scenario in args.scenarios:
        baseline_controller = controller(
            cfg, model, design, omega, domain,
            action_mapping="interior_anchor", supervisor="dwell20",
            dwell_sets=DWELL_SETS, observation_variant="supervisor23")
        baseline_policy = SegmentedPolicy(np.zeros((30, 2)), 10)
        stat, records, _ = run_episode(
            cfg, model, baseline_controller, baseline_policy, None,
            np.random.default_rng(42), training=False, global_step=0,
            paper_scenario=scenario,
        )
        baseline = metrics(stat, records, cfg, lower, upper)
        if not baseline["safety_passed"]:
            raise RuntimeError(f"Unsafe paired baseline: {scenario}")
        (args.output_dir / f"{scenario}_baseline.json").write_text(
            json.dumps(baseline, indent=2), encoding="utf-8")
        seed10 = actor_seed(args.constrained_trajectory, scenario, 10)
        summary10 = search_one(scenario, 10, args.budget_10s, cfg, model,
                               design, omega, domain, baseline, lower, upper,
                               seed10, rng, args.output_dir)
        summaries.append(summary10)
        print(json.dumps(summary10), flush=True)
        if args.budget_5s and summary10["safe_candidates"] > 1:
            seed5 = actor_seed(args.constrained_trajectory, scenario, 5)
            summary5 = search_one(scenario, 5, args.budget_5s, cfg, model,
                                   design, omega, domain, baseline, lower, upper,
                                   seed5, rng, args.output_dir)
            summaries.append(summary5)
            print(json.dumps(summary5), flush=True)
        (args.output_dir / "summary.json").write_text(json.dumps({
            "frozen_class": "balanced B, Hinf/RPI/Omega/dwell20/interior-anchor/supervisor23",
            "protocol": "Paper2016 endpoint 0/20/40 state jumps, 300 s",
            "optimization": "offline segmented actor actions, archive multiobjective derivative-free",
            "conclusion_limit": "finite search absence is not mathematical infeasibility",
            "oscillation_diagnosis": diagnostics,
            "searches": summaries,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
