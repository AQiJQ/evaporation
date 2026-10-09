"""Reporting and checkpoint selection for the frozen stochastic SAC experiment.

No controller, observation, disturbance, or optimization rule is changed
here. Reward components are reported without changing their computation.
Performance thresholds are empirical selection references.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .controlled_invariant_error_set import facets, hull
from .train import run_episode
from .zanon2019_benchmark import (
    StochasticInterior23, paired_rollouts, sample_disturbance_path,
    trajectory_metrics, write_csv,
)
from .zanon2019_authority import AuthorityController, authority_assessment


CERTIFICATION_STATUS = "empirical_only_under_ECC2019_stochastic_disturbance"
RATIO_METRICS = {
    "X2_IAE_ratio": "X2_IAE", "P2_IAE_ratio": "P2_IAE",
    "X2_ISE_ratio": "X2_ISE", "P2_ISE_ratio": "P2_ISE",
    "P100_TV_ratio": "P100_TV", "F200_TV_ratio": "F200_TV",
    "P100_RMS_du_ratio": "P100_delta_RMS",
    "F200_RMS_du_ratio": "F200_delta_RMS",
    "P100_sign_change_ratio": "P100_input_sign_changes",
    "F200_sign_change_ratio": "F200_input_sign_changes",
    "G_X2_ratio": "G_X2_emp", "G_P2_ratio": "G_P2_emp",
    "X2_std_ratio": "X2_std", "P2_std_ratio": "P2_std",
    "X2_peak_deviation_ratio": "X2_peak_deviation",
    "P2_peak_deviation_ratio": "P2_peak_deviation",
}
SELECTION_RULES = {
    "hard_gate": "physical X2/P2/input and QP counts all zero; W is diagnostic only",
    "learned_gate": "global_step >= warmup_steps; pre-warmup actors are diagnostic only",
    "level_B": "hard gate, positive mean economics, every seed X2/P2 IAE ratios <=1.10",
    "level_C": "level_B, every seed P100/F200 TV ratios <=1.10, Omega exits zero",
    "best_empirical_joint_actor": "level_C; minimize mean of X2/P2 IAE and P100/F200 TV ratios, then maximize economics",
    "best_economic_safe_actor": "hard gate; maximize paired mean economics",
    "best_recovery_safe_actor": "hard gate; minimize mean IAE ratio, then mean ISE ratio",
    "best_low_activity_safe_actor": "hard gate and mean economics >=0; minimize mean TV ratio, then mean RMS move ratio",
    "closest_pareto_actor": "hard gate and Omega zero; minimize violation distance to level_C; diagnostic trade-off when no joint exists",
    "pareto_axes": "maximize mean economics; minimize mean IAE, mean TV, outer-10%-occupancy fraction",
    "boundary_occupancy": "reported against paired stochastic baseline; no inherited finite-jump absolute TV or occupancy cutoff",
    "zero_denominator": "ratio=1 if both values are <=1e-12, otherwise divide by max(baseline,1e-12)",
}


def relative(value, baseline):
    if abs(value) <= 1e-12 and abs(baseline) <= 1e-12:
        return 1.0
    return float(value / max(abs(baseline), 1e-12))


def evaluation_safety_anomalies(rows):
    """Empirical stop gate for BOTH members of a paired evaluation.

    A W miss alone is expected under this Gaussian benchmark and never stops
    evaluation/training. Omega exit stops continuation without claiming that
    Omega is a Gaussian safety certificate. Robust-region counts stay audited.
    """
    counters = ("physical_state_violation_steps", "X2_violation_count",
                "P2_physical_violation_count", "physical_input_violation_count",
                "QP_infeasible_count", "Omega_exit_count", "Omega_exit_events")
    failures = []
    for row in rows:
        for controller in ("baseline", "SAC"):
            found = {key: int(row[f"{controller}_{key}"]) for key in counters
                     if row[f"{controller}_{key}"] > 0}
            if found:
                failures.append({"seed": row["seed"], "controller": controller,
                                 "counts": found})
    return failures


def metrics(records, cfg, model, design, omega):
    result = trajectory_metrics(records, cfg, model, design)
    from .zanon2019_reward_audit import reward_metrics
    result.update(reward_metrics(records))
    states = np.asarray([r["state"] for r in records])
    inputs = np.asarray([r["control"] for r in records])
    w = np.asarray([r["w_hat"] for r in records])
    # Reconstruct the already simulated next state, including the final step.
    next_n = (model.normalized_state(states) @ design.a.T
              + model.normalized_input(inputs) @ design.b.T + design.affine + w)
    next_x = model.physical_state(next_n)
    for axis, name in enumerate(("X2", "P2")):
        loss = float(np.mean(((next_x[:, axis] - cfg.robust_economic_reference_state[axis])
                             / cfg.state_scale[axis]) ** 2))
        result[f"reward_{name}_recovery_loss_mean"] = loss
        result[f"reward_{name}_recovery_penalty_mean"] = cfg.paper2016_state_recovery_penalty_weight * loss
    hx, bx = facets(omega)
    omega_after = np.max((next_n - design.z_ref) @ hx.T - bx, axis=1) > 1e-8
    omega_before = np.array([not bool(r["in_Omega_before"]) for r in records])
    error = states - cfg.robust_economic_reference_state
    disturbance = np.asarray([r["disturbance"] for r in records])
    dn = (disturbance - cfg.disturbance_nominal) / np.sqrt([2., 1., 8., 5.])
    energy = float(np.sum(dn ** 2))
    for axis, name in enumerate(("X2", "P2")):
        result[f"{name}_std"] = float(np.std(states[:, axis]))
        result[f"{name}_peak_deviation"] = float(np.max(np.abs(error[:, axis])))
        result[f"G_{name}_emp"] = float(np.linalg.norm(error[:, axis])
                                              / max(np.sqrt(energy), 1e-12))
    next_bad = np.any((next_x < cfg.state_lower - 1e-8)
                     | (next_x > cfg.state_upper + 1e-8), axis=1)
    result["physical_state_violation_steps"] = int(np.sum(next_bad))
    # Include the final transition in safety-margin diagnostics. Performance
    # IAE/G definitions above remain unchanged for historical comparability.
    margin = next_x[:, 0] - 25.
    result.update({"X2_margin_mean": float(np.mean(margin)),
        "X2_margin_min": float(np.min(margin)),
        **{f"X2_margin_p{p}": float(np.percentile(margin, p)) for p in (1, 5, 50)},
        **{f"X2_margin_fraction_lt_{label}": float(np.mean(margin < threshold))
           for label, threshold in (("0p05", .05), ("0p10", .10), ("0p20", .20))}})
    result["X2_violation_count"] = int(np.sum(
        (states[:, 0] < 25. - 1e-8) | (next_x[:, 0] < 25. - 1e-8)))
    result["P2_physical_violation_count"] = int(np.sum(
        (states[:, 1] < 40. - 1e-8) | (states[:, 1] > 80. + 1e-8)
        | (next_x[:, 1] < 40. - 1e-8) | (next_x[:, 1] > 80. + 1e-8)))
    result["Omega_exit_count"] = int(np.sum(omega_before | omega_after))
    result["Omega_exit_after_count"] = int(np.sum(omega_after))
    modes = np.asarray([r["safety_mode"] for r in records])
    result["Z_mode_fraction"] = float(np.mean(modes == "Z_mode_existing_controller"))
    result["Omega_mode_fraction"] = float(np.mean(modes == "Omega_safe_one_step_QP"))
    result["outside_Omega_fraction"] = float(np.mean(omega_before))
    raw = np.asarray([r["raw_action"] for r in records])
    result["stochastic_residual_scale"] = float(getattr(cfg, "stochastic_residual_scale", 1.))
    result["scaled_action_norm_mean"] = float(np.mean(np.linalg.norm(
        result["stochastic_residual_scale"] * raw, axis=1)))
    for axis in range(2):
        result[f"raw_actor_mean_{axis}"] = float(np.mean(raw[:, axis]))
        result[f"raw_actor_std_{axis}"] = float(np.std(raw[:, axis]))
        result[f"raw_actor_saturation_fraction_{axis}"] = float(np.mean(np.abs(raw[:, axis]) >= .99))
    rho = np.max(np.abs(raw), axis=1)
    requested = np.asarray([r["residual"] for r in records])
    applied = np.asarray([r["applied_residual"] for r in records])
    nr, na = np.linalg.norm(requested, axis=1), np.linalg.norm(applied, axis=1)
    nonzero = rho > 1e-8
    displacement = np.asarray([r["applied_displacement_from_baseline"] for r in records])
    collapsed = nonzero & (displacement < 1e-6)
    result.update({
        "residual_requested_norm_mean": float(np.mean(nr)),
        "residual_applied_norm_mean": float(np.mean(na)),
        "residual_applied_ratio": relative(float(np.sum(na)), float(np.sum(nr))),
        "residual_requested_applied_ratio": relative(float(np.sum(nr)), float(np.sum(na))),
        "residual_ratio_defined": bool(np.sum(nr) > 1e-12 and np.sum(na) > 1e-12),
        "zero_applied_residual_rate": float(np.mean(na <= 1e-8)),
        "nonzero_actor_steps": int(np.sum(nonzero)),
        "collapsed_action_steps": int(np.sum(collapsed)),
        "action_collapse_fraction": float(np.sum(collapsed) / max(np.sum(nonzero), 1)),
        "raw_actor_saturation_fraction": float(np.mean(np.any(np.abs(raw) >= .99, axis=1))),
        "applied_displacement_physical_mean": float(np.mean(displacement)),
        "applied_displacement_physical_max": float(np.max(displacement)),
        "QP_modification_rate": float(np.mean([
            float(r.get("final_verification_gap") or 0.) > 1e-8 for r in records])),
        "supervisor_active_fraction": 0.,
        "supervisor_modification_rate": 0.,
        "any_outer_10pct_fraction": result["any_outer_10pct_steps"] / len(records),
    })
    radii = [float(r["interior_chebyshev_radius"]) for r in records
             if r.get("interior_chebyshev_radius") is not None
             and np.isfinite(r["interior_chebyshev_radius"])]
    result["interior_radius_normalized_mean"] = float(np.mean(radii)) if radii else None
    result["interior_radius_normalized_min"] = float(min(radii)) if radii else None
    for name, mask in (("Z", modes == "Z_mode_existing_controller"),
                       ("Omega", modes == "Omega_safe_one_step_QP")):
        result[f"action_collapse_fraction_{name}"] = float(
            np.sum(collapsed & mask) / max(np.sum(nonzero & mask), 1))
    return result


def assessment(rows, episode, global_step, warmup):
    if not rows:
        raise ValueError("Empty paired assessment")
    mean = lambda key: float(np.mean([r[key] for r in rows]))
    count = lambda key: int(sum(r[f"SAC_{key}"] for r in rows))
    physical = count("physical_state_violation_steps") + count("physical_input_violation_count")
    safe = physical == 0 and count("X2_violation_count") == 0 \
        and count("P2_physical_violation_count") == 0 and count("QP_infeasible_count") == 0
    post = global_step >= warmup
    econ = mean("economic_improvement_percent")
    omega = count("Omega_exit_count")
    recovery = all(r[key] <= 1.10 for r in rows
                   for key in ("X2_IAE_ratio", "P2_IAE_ratio"))
    activity = all(r[key] <= 1.10 for r in rows
                   for key in ("P100_TV_ratio", "F200_TV_ratio"))
    result = {
        "episode": episode, "global_step": global_step, "post_warmup": post,
        "economic_improvement_pct": econ,
        "worst_economic_improvement_pct": float(min(r["economic_improvement_percent"] for r in rows)),
        **{key: mean(key) for key in RATIO_METRICS},
        "mean_IAE_ratio": .5 * (mean("X2_IAE_ratio") + mean("P2_IAE_ratio")),
        "mean_ISE_ratio": .5 * (mean("X2_ISE_ratio") + mean("P2_ISE_ratio")),
        "mean_TV_ratio": .5 * (mean("P100_TV_ratio") + mean("F200_TV_ratio")),
        "mean_RMS_du_ratio": .5 * (mean("P100_RMS_du_ratio") + mean("F200_RMS_du_ratio")),
        "physical_violation": physical, "QP_infeasible": count("QP_infeasible_count"),
        "Omega_exit": omega, "W_exceedance_rate": mean("SAC_W_exceedance_rate"),
        "baseline_J_econ": mean("baseline_J_econ"), "SAC_J_econ": mean("SAC_J_econ"),
        "empirical_safe": safe, "positive_economic": econ > 0,
        "level_B": safe and econ > 0 and recovery,
        "level_C": safe and econ > 0 and recovery and activity and omega == 0,
        "eligible_empirical_joint_checkpoint": post and safe and econ > 0
            and recovery and activity and omega == 0,
        "pareto_nondominated": False,
        "certification_status": CERTIFICATION_STATUS,
    }
    # Component diagnostics do not change selection or the evaluation protocol.
    # Replay-equivalent excludes RPI terms, while raw eval return retains them.
    for label in ("baseline", "SAC"):
        for key in rows[0]:
            if key.startswith(label + "_reward_"):
                result[key] = mean(key)
    for key in (
        "residual_applied_ratio", "residual_requested_applied_ratio",
        "residual_requested_norm_mean", "residual_applied_norm_mean",
        "action_collapse_fraction", "raw_actor_saturation_fraction",
        "QP_modification_rate", "Z_mode_fraction", "Omega_mode_fraction",
        "outside_Omega_fraction", "supervisor_modification_rate",
        "applied_displacement_physical_mean", "any_outer_10pct_fraction",
    ):
        result[key] = mean("SAC_" + key)
    result["joint_performance_score"] = .5 * (
        result["mean_IAE_ratio"] + result["mean_TV_ratio"])
    result["distance_to_joint"] = (
        max(0., -econ / 100.) + sum(max(0., r[k] - 1.10)
            for r in rows for k in ("X2_IAE_ratio", "P2_IAE_ratio",
                                    "P100_TV_ratio", "F200_TV_ratio")) / len(rows))
    return result


def update_pareto(statuses):
    eligible = [s for s in statuses if s["post_warmup"]
                and s["empirical_safe"] and s["Omega_exit"] == 0]
    for s in statuses:
        s["pareto_nondominated"] = False
    def vector(s):
        return np.array([-s["economic_improvement_pct"], s["mean_IAE_ratio"],
                         s["mean_TV_ratio"], s["any_outer_10pct_fraction"]])
    for s in eligible:
        v = vector(s)
        s["pareto_nondominated"] = not any(
            np.all(vector(t) <= v + 1e-10) and np.any(vector(t) < v - 1e-10)
            for t in eligible if t is not s)


class CheckpointSelection:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.best = {}

    def consider(self, agent, status):
        if not status["post_warmup"] or not status["empirical_safe"]:
            return
        criteria = {
            "best_economic_safe_actor": (-status["economic_improvement_pct"],),
            "best_recovery_safe_actor": (status["mean_IAE_ratio"], status["mean_ISE_ratio"]),
        }
        if status["economic_improvement_pct"] >= 0:
            criteria["best_low_activity_safe_actor"] = (
                status["mean_TV_ratio"], status["mean_RMS_du_ratio"])
        if status["eligible_empirical_joint_checkpoint"]:
            criteria["best_empirical_joint_actor"] = (
                status["joint_performance_score"], -status["economic_improvement_pct"])
        if status["Omega_exit"] == 0:
            criteria["closest_pareto_actor"] = (status["distance_to_joint"],
                                                 -status["economic_improvement_pct"])
        for name, score in criteria.items():
            previous = self.best.get(name)
            if previous is None or tuple(score) < tuple(previous["selection_score"]):
                agent.save_actor(self.model_dir / (name + ".pth"))
                self.best[name] = {**status, "selection_score": list(score),
                                   "checkpoint": str(self.model_dir / (name + ".pth"))}


def trace_rows(baseline, policy, alpha=1.):
    rows, cumulative = [], 0.
    for step, (b, p) in enumerate(zip(baseline, policy)):
        delta = float(p["economic_cost"] - b["economic_cost"])
        cumulative += delta
        row = {"step": step, "delta_l_SAC_minus_baseline": delta,
               "cumulative_delta_SAC_minus_baseline": cumulative,
               **{name: float(p["disturbance"][i])
                  for i, name in enumerate(("F1", "X1", "T1", "T200"))}}
        for prefix, r in (("baseline", b), ("SAC", p)):
            row[f"{prefix}_stochastic_residual_scale"] = alpha
            for i, v in enumerate(r["raw_action"]):
                row[f"{prefix}_scaled_action_{i}"] = float(alpha * v)
            for key in ("state", "control", "raw_action", "residual",
                        "applied_residual", "interior_anchor", "boundary_target", "w_hat"):
                for i, v in enumerate(r[key]):
                    row[f"{prefix}_{key}_{i}"] = float(v)
            for key in ("economic_cost", "in_Z_before", "in_Omega_before",
                        "qp_feasible", "safety_mode", "collapsed_action",
                        "interior_chebyshev_radius", "final_verification_gap",
                        "applied_displacement_from_baseline", "physical_state_violation",
                        "physical_input_violation", "robust_region_violation",
                        "economic_reward", "state_recovery_loss", "state_recovery_penalty",
                        "p100_move_loss", "p100_move_penalty", "f200_move_loss", "f200_move_penalty",
                        "saturation_loss", "saturation_penalty", "projection_penalty",
                        "mapping_penalty", "move_penalty", "rpi_violation_event_penalty",
                        "rpi_excess_penalty", "total_reward", "safety_penalty", "reward_safety_penalty",
                        "state_violation_penalty", "input_violation_penalty", "qp_infeasible_penalty",
                        "state_excess_penalty", "input_excess_penalty",
                        "rpi_penalty_excluded_from_training_reward"):
                row[f"{prefix}_{key}"] = r[key]
        rows.append(row)
    return rows


class FixedPairedEvaluator:
    def __init__(self, cfg, model, design, omega, domain, mode, variances,
                 hold_min, hold_max, seeds, output_dir):
        self.cfg, self.model, self.design, self.omega, self.domain = cfg, model, design, omega, domain
        self.output_dir = Path(output_dir)
        self.cache = {}
        for seed in seeds:
            path, metadata = sample_disturbance_path(
                cfg, mode, seed, cfg.steps_per_episode, variances, hold_min, hold_max)
            baseline, _ = paired_rollouts(cfg, model, design, omega, domain, None, path, seed)
            bm = metrics(baseline[1], cfg, model, design, omega)
            self.cache[seed] = (path, baseline[1], bm, metadata)
            write_csv(self.output_dir / "paired_baselines" / f"seed_{seed}.csv",
                      trace_rows(baseline[1], baseline[1]))

    def evaluate(self, agent, episode, global_step, label=None):
        rows = []
        for seed, (path, base, bm, metadata) in self.cache.items():
            ctrl = AuthorityController(self.cfg, self.model, self.design, self.omega, self.domain)
            _, records, _ = run_episode(
                self.cfg, self.model, ctrl, agent, None,
                np.random.default_rng(seed + 900000), training=False, global_step=0,
                disturbance_trajectory=path,
                initial_state_override=self.cfg.robust_economic_reference_state)
            if (len(records) != len(base) or not np.array_equal(records[0]["state"], base[0]["state"])
                    or any(not np.array_equal(r["disturbance"], path[i])
                           or np.any(r["paper_state_shock"]) for i, r in enumerate(records))):
                raise RuntimeError("Fixed paired initial state/disturbance/shock identity failed")
            pm = metrics(records, self.cfg, self.model, self.design, self.omega)
            row = {
                "episode": episode, "global_step": global_step, "seed": seed,
                "scenario": "stochastic_operating_uncertainty", "checkpoint_label": label,
                "economic_improvement_percent": 100. * (bm["J_econ"] - pm["J_econ"]) / bm["J_econ"],
                **{k: relative(pm[v], bm[v]) for k, v in RATIO_METRICS.items()},
                **{f"baseline_{k}": v for k, v in bm.items()},
                **{f"SAC_{k}": v for k, v in pm.items()},
                "paired_disturbance_identity_max_error": 0.,
                "disturbance_clip_count_F1": metadata["physical_clip_counts_F1_X1"][0],
                "disturbance_clip_count_X1": metadata["physical_clip_counts_F1_X1"][1],
                "certification_status": CERTIFICATION_STATUS,
            }
            row.update({"economic_degradation_pct": -row["economic_improvement_percent"],
                "economic_comparable_0p5pct": pm["J_econ"] <= 1.005 * bm["J_econ"],
                "economic_comparable_1pct": pm["J_econ"] <= 1.01 * bm["J_econ"],
                "stochastic_residual_scale": float(getattr(self.cfg, "stochastic_residual_scale", 1.))})
            rows.append(row)
            folder = label or f"episode_{episode:04d}"
            write_csv(self.output_dir / "evaluation_trajectories" / folder / f"seed_{seed}.csv",
                      trace_rows(base, records, getattr(self.cfg, "stochastic_residual_scale", 1.)))
        status = assessment(rows, episode, global_step, self.cfg.warmup_steps)
        if getattr(self.cfg, "authority_pilot", False):
            status = authority_assessment(rows, status)
        return rows, status


def plot_learning(rows, output_dir):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = (
        ("economic_improvement_percent", "paired economic improvement (%)"),
        ("X2_IAE_ratio", "X2 IAE ratio"), ("P2_IAE_ratio", "P2 IAE ratio"),
        ("mean_IAE_ratio", "mean IAE ratio"),
        ("P100_TV_ratio", "P100 TV ratio"), ("F200_TV_ratio", "F200 TV ratio"),
        ("G_X2_ratio", "X2 empirical amplification ratio"),
        ("G_P2_ratio", "P2 empirical amplification ratio"),
        ("SAC_physical_state_violation_steps", "physical state violation count"),
        ("SAC_physical_input_violation_count", "physical input violation count"),
        ("SAC_QP_infeasible_count", "QP infeasible count"),
        ("SAC_Omega_exit_count", "Omega exit count"),
        ("SAC_residual_applied_norm_mean", "applied residual magnitude"),
        ("SAC_residual_requested_applied_ratio", "requested / applied residual ratio"),
        ("SAC_action_collapse_fraction", "nonzero-actor collapse fraction"),
        ("SAC_QP_modification_rate", "final QP modification rate"),
    )
    def value(row, key):
        return .5 * (row["X2_IAE_ratio"] + row["P2_IAE_ratio"]) if key == "mean_IAE_ratio" else row[key]
    fig, axes = plt.subplots(4, 4, figsize=(17, 12))
    for ax, (key, title) in zip(axes.ravel(), series):
        episodes = sorted({r["episode"] for r in rows})
        for seed in sorted({r["seed"] for r in rows}):
            subset = sorted((r for r in rows if r["seed"] == seed), key=lambda r: r["episode"])
            ax.plot([r["episode"] for r in subset], [value(r, key) for r in subset],
                    lw=.6, alpha=.35)
        ax.plot(episodes, [np.mean([value(r, key) for r in rows if r["episode"] == ep])
                           for ep in episodes], lw=1.5, color="black")
        if key.endswith("ratio") and "residual" not in key:
            ax.axhline(1., color="gray", ls="--", lw=.6)
        ax.set(title=title, xlabel="episode")
        ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "fixed_paired_learning_curves.png", dpi=160)
    plt.close(fig)


def classify(statuses):
    learned = [s for s in statuses if s["post_warmup"]]
    if any(not s["empirical_safe"] or s["Omega_exit"] > 0 for s in statuses):
        return "D"
    if any(s["eligible_empirical_joint_checkpoint"] for s in learned):
        return "A"
    if any(s["positive_economic"] for s in learned):
        return "B"
    return "C" if learned else "not_assessed_no_post_warmup_checkpoint"


class EvidenceController(AuthorityController):
    """Read-only audit hook; never changes the candidate, input or observation.

    Capture before run_episode can raise, and audit the final next state too.
    Gaussian W misses remain diagnostic, not an empirical stopping gate.
    """
    def __init__(self, cfg, model, design, omega, domain, disturbance):
        super().__init__(cfg, model, design, omega, domain)
        self.disturbance_path = disturbance
        self.evidence = []
        self.oh, self.ob = facets(omega)
        self.wh, self.wb = facets(hull(design.w_vertices))

    def act(self, state, actor_action, *, action_is_normalized=True):
        control, info = super().act(state, actor_action,
                                    action_is_normalized=action_is_normalized)
        index = len(self.evidence)
        row = {"step": index, "safety_mode": info.get("mode"),
               "QP_infeasible": int(not info["qp_feasible"]),
               "Omega_exit_before": int(not info["in_Omega"]),
               "next_state_observed": False}
        for name, values in (("state", state), ("control", control),
                             ("raw_action", actor_action),
                             ("scaled_action", info["scaled_actor_action"]),
                             ("disturbance", self.disturbance_path[index]),
                             ("requested_residual", info["requested_residual"]),
                             ("applied_residual", info["applied_residual"])):
            row.update({f"{name}_{i}": float(v) for i, v in enumerate(values)})
        for name in ("interior_chebyshev_radius", "final_verification_gap",
                     "applied_displacement_from_baseline", "omega_exit_event"):
            row[name] = info.get(name)
        # This value is normally derived by run_episode, not returned in info.
        # Compute it here so an early audit abort still has complete evidence.
        center_input = self.model.physical_input(info["action_center_actual_norm"])
        row["applied_displacement_from_baseline"] = float(
            np.linalg.norm(np.asarray(control) - center_input))
        row.update({f"zero_residual_safe_input_{i}": float(v)
                    for i, v in enumerate(center_input)})
        self.evidence.append(row)
        self.last_state, self.last_control = np.array(state), np.array(control)
        self.last_info = info
        return control, info

    def audit_next_state(self, next_state):
        row = self.evidence[-1]
        xn = self.model.normalized_state(next_state)
        w = xn - (self.d.a @ self.model.normalized_state(self.last_state)
                  + self.d.b @ self.model.normalized_input(self.last_control)
                  + self.d.affine)
        omega_bad = np.max(self.oh @ (xn - self.d.z_ref) - self.ob) > 1e-8
        physical_bad = np.any((next_state < self.cfg.state_lower - 1e-8)
                              | (next_state > self.cfg.state_upper + 1e-8))
        input_bad = np.any((self.last_control < self.cfg.input_lower - 1e-8)
                           | (self.last_control > self.cfg.input_upper + 1e-8))
        row.update({"next_state_observed": True,
                    "physical_state_violation": int(physical_bad),
                    "physical_input_violation": int(input_bad),
                    "Omega_exit_after": int(omega_bad),
                    "W_exceedance": int(np.max(self.wh @ w - self.wb) > 1e-8)})
        row.update({f"next_state_{i}": float(v) for i, v in enumerate(next_state)})
        row.update({f"w_actual_{i}": float(v) for i, v in enumerate(w)})
        if physical_bad or input_bad or omega_bad or row["QP_infeasible"]:
            raise RuntimeError(f"ECC2019 empirical safety anomaly at step {row['step']}: {row}")
