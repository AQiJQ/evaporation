"""Offline frozen-actor failure/scale/reward audit; no replay or SAC updates.

All trajectories are autonomous closed loops through StochasticInterior23.
Scale is a diagnostic wrapper BEFORE the unchanged interior-anchor mapper.
Only no-learning evaluation permits continuing beyond an unsafe transition,
to inspect the requested cascading failures. Never used by the trainer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .controlled_invariant_error_set import facets, hull
from .train import ZeroResidualPolicy, run_episode
from .zanon2019_benchmark import (
    DEFAULT_DESIGN, DEFAULT_OMEGA, REPO_DIR, StochasticInterior23,
    make_agent, make_setup, load_actor, sample_disturbance_path, write_csv,
)
from .zanon2019_training_report import CERTIFICATION_STATUS, assessment, metrics, RATIO_METRICS, relative
from .zanon2019_train import save_json


ALPHAS = (0., .05, .10, .20, .30, .40, .50, .60, .75, 1.)
DEFAULT_RUN = REPO_DIR / "evaporation_safe_sac/outputs_zanon2019_stochastic_seed42_300x1000"
PENALTIES = ("projection_penalty", "mapping_penalty", "move_penalty",
             "state_recovery_penalty", "p100_move_penalty", "f200_move_penalty",
             "saturation_penalty", "state_violation_penalty", "input_violation_penalty",
             "qp_infeasible_penalty", "state_excess_penalty", "input_excess_penalty")
RPI_AUDITS = ("rpi_violation_event_penalty", "rpi_excess_penalty")


class ScaledFrozenActor:
    def __init__(self, actor, alpha):
        self.actor, self.alpha, self.unscaled = actor, float(alpha), []

    def select_action(self, obs, deterministic=True):
        raw = self.actor.select_action(obs, deterministic=True).astype(np.float32)
        self.unscaled.append(raw.copy())
        return (self.alpha * raw).astype(np.float32)


class AuditController(StochasticInterior23):
    """Read-only transition diagnostics, deliberately NOT a safety exception hook."""
    def __init__(self, cfg, model, design, omega, domain):
        super().__init__(cfg, model, design, omega, domain)
        self.transitions = []
        self.hw, self.bw = facets(hull(design.w_vertices))
        self.ho, self.bo = facets(omega)

    def act(self, state, actor_action, *, action_is_normalized=True):
        control, info = super().act(state, actor_action, action_is_normalized=action_is_normalized)
        self.last_state, self.last_control = np.array(state), np.array(control)
        self.last_info = info
        return control, info

    def audit_next_state(self, next_state):
        i = self.last_info
        pred = (self.d.a @ self.model.normalized_state(self.last_state)
                + self.d.b @ self.model.normalized_input(self.last_control) + self.d.affine)
        actual = self.model.normalized_state(next_state)
        w = actual - pred
        possible = pred + self.d.w_vertices
        possible_x = self.model.physical_state(possible)
        facet_values = self.hw @ w
        # Raw h-relative facet utilization is meaningful only for h>0;
        # signed excess is always emitted, including any nonpositive bound.
        utilization = np.divide(facet_values, self.bw,
            out=np.full_like(self.bw, np.nan), where=self.bw > 1e-15)
        row = {"step": len(self.transitions),
            "physical_before_safe": bool(np.all(self.last_state >= self.cfg.state_lower - 1e-8)
                                          and np.all(self.last_state <= self.cfg.state_upper + 1e-8)),
            "physical_after_violation": bool(np.any(next_state < self.cfg.state_lower - 1e-8)
                                            or np.any(next_state > self.cfg.state_upper + 1e-8)),
            "Omega_before": bool(i["in_Omega"]),
            "Omega_after": bool(np.max(self.ho @ (actual - self.d.z_ref) - self.bo) <= 1e-8),
            "Z_before": bool(i["in_Z"]),
            "QP_feasible": bool(i["qp_feasible"]),
            "QP_modification": float(i.get("final_verification_gap") or 0.),
            "safety_mode": i["mode"],
            "W_exceedance": bool(np.max(facet_values - self.bw) > 1e-8),
            "W_max_facet_excess": float(np.max(facet_values - self.bw)),
            "W_max_facet_utilization": float(np.nanmax(utilization)),
            "certified_W_next_Omega_max_excess": float(np.max((possible - self.d.z_ref) @ self.ho.T - self.bo)),
            "actual_next_Omega_max_excess": float(np.max(self.ho @ (actual - self.d.z_ref) - self.bo)),
            "X2_minus_25": float(self.last_state[0] - 25.),
        }
        for key, array in (("state", self.last_state), ("control", self.last_control),
                           ("actual_next", next_state), ("w_actual", w),
                           ("raw_action", i.get("diagnostic_raw_action", np.zeros(2))),
                           ("predicted_nominal_next", self.model.physical_state(pred)),
                           ("predicted_W_next_min", np.min(possible_x, axis=0)),
                           ("predicted_W_next_max", np.max(possible_x, axis=0)),
                           ("baseline_input_at_same_state", self.model.physical_input(i["action_center_actual_norm"]))):
            row.update({f"{key}_{k}": float(v) for k, v in enumerate(array)})
        for k in range(len(self.bw)):
            row[f"W_facet_{k}_utilization"] = float(utilization[k])
            row[f"W_facet_{k}_excess"] = float(facet_values[k] - self.bw[k])
        self.transitions.append(row)


def rollout(cfg, model, design, omega, domain, policy, path, seed):
    ctrl = AuditController(cfg, model, design, omega, domain)
    stat, records, _ = run_episode(cfg, model, ctrl, policy, None,
        np.random.default_rng(seed + 900000), training=False, global_step=0,
        disturbance_trajectory=path, initial_state_override=cfg.robust_economic_reference_state)
    assert len(records) == len(path) == len(ctrl.transitions)
    for k, r in enumerate(records):
        np.testing.assert_array_equal(r["disturbance"], path[k])
        assert not np.any(r["paper_state_shock"])
        # Each transition follows its OWN preceding plant state, not oracle forcing.
        if k + 1 < len(records):
            np.testing.assert_allclose(records[k+1]["state"],
                [ctrl.transitions[k]["actual_next_0"], ctrl.transitions[k]["actual_next_1"]], atol=1e-12, rtol=0)
    return stat, records, ctrl.transitions


def first_index(flags):
    ids = np.flatnonzero(flags)
    return int(ids[0]) if len(ids) else None


def cascade_for(seed, transitions):
    physical = np.array([t["physical_after_violation"] for t in transitions])
    omega = np.array([not (t["Omega_before"] and t["Omega_after"]) for t in transitions])
    qp = np.array([not t["QP_feasible"] for t in transitions])
    p = first_index(physical)
    result = {"seed": seed, "first_physical_transition_step": p,
        "first_Omega_affected_transition_step": first_index(omega),
        "first_Omega_before_step": first_index([not t["Omega_before"] for t in transitions]),
        "first_QP_infeasible_step": first_index(qp)}
    masks = {"before_first_physical": np.arange(len(physical)) < p,
             "at_first_physical": np.arange(len(physical)) == p,
             "strictly_after_first_physical": np.arange(len(physical)) > p} if p is not None else {"no_physical_failure": np.ones(len(physical), bool)}
    for name, mask in masks.items():
        result[name] = {"physical": int(np.sum(physical & mask)),
                        "Omega_affected_steps": int(np.sum(omega & mask)),
                        "QP_infeasible": int(np.sum(qp & mask))}
    if p is not None:
        t = transitions[p]
        result["first_failure_type"] = (
            "model/disturbance-domain mismatch induced safety failure"
            if t["QP_feasible"] and t["Omega_before"] and t["physical_before_safe"]
            and t["W_exceedance"] and t["certified_W_next_Omega_max_excess"] <= 2e-7
            else "requires further diagnosis")
        result["first_failure_evidence"] = {k: v for k, v in t.items() if not k.startswith("W_facet_")}
    return result


def replay_components(record):
    """Audit CURRENT train.run_episode components, not a new reward formula.

    run_episode recomputes every real component during deterministic rollout.
    Its eval branch includes RPI penalties. For the same frozen stochastic
    proposed protocol, its training/replay branch excludes exactly two logged
    RPI terms. Expose that existing difference; never count audited RPI as
    reward seen by SAC. No weights or component definitions are changed.
    """
    components = {"economic_reward": float(record["economic_reward"])}
    components.update({key: -float(record[key]) for key in PENALTIES})
    real_eval_reward = sum(components.values()) - sum(float(record[k]) for k in RPI_AUDITS)
    np.testing.assert_allclose(real_eval_reward, record["reward"], atol=1e-7, rtol=1e-11)
    return components


def reward_audit(selected, data, out):
    rows, step_rows = [], []
    for label, alpha in selected.items():
        for seed, (_, records, _) in data[alpha].items():
            c = [replay_components(r) for r in records]
            econ = np.array([r["economic_reward"] for r in records])
            reference = float(np.mean(np.abs(econ)))
            for key in c[0]:
                v = np.array([r[key] for r in c])
                rows.append({"label": label, "alpha": alpha, "seed": seed,
                    "component": key, "mean_signed": float(np.mean(v)),
                    "median_signed": float(np.median(v)), "p95_abs": float(np.percentile(np.abs(v), 95)),
                    "mean_abs": float(np.mean(np.abs(v))), "episode_cumulative_signed": float(np.sum(v)),
                    "mean_abs_over_mean_abs_economic": relative(float(np.mean(np.abs(v))), reference),
                    "role": "active replay reward component (zero allowed)"})
            for key in RPI_AUDITS:
                v = -np.array([r[key] for r in records])
                rows.append({"label": label, "alpha": alpha, "seed": seed,
                    "component": key, "mean_signed": float(np.mean(v)),
                    "median_signed": float(np.median(v)), "p95_abs": float(np.percentile(np.abs(v), 95)),
                    "mean_abs": float(np.mean(np.abs(v))), "episode_cumulative_signed": float(np.sum(v)),
                    "mean_abs_over_mean_abs_economic": relative(float(np.mean(np.abs(v))), reference),
                    "role": "AUDIT ONLY: excluded from stochastic proposed SAC replay reward"})
            for k, (r, components) in enumerate(zip(records, c)):
                step_rows.append({"label": label, "alpha": alpha, "seed": seed, "step": k,
                    **components, **{"audited_"+key: r[key] for key in RPI_AUDITS},
                    "actual_evaluation_reward": r["reward"],
                    "actual_training_replay_reward_same_transition": sum(components.values()),
                    "positive_physical_margin_penalty": "not present in current reward",
                    "state_recovery_loss": r["state_recovery_loss"],
                    "p100_move_loss": r["p100_move_loss"], "f200_move_loss": r["f200_move_loss"],
                    "saturation_loss": r["saturation_loss"]})
    write_csv(out / "reward_component_audit.csv", rows)
    write_csv(out / "reward_components_per_step.csv", step_rows)
    return rows


def margin_audit(selected, data, out):
    rows = []
    for label, alpha in selected.items():
        for seed, (_, records, transitions) in data[alpha].items():
            # Report x_0...x_{T-1} for comparability, and x_1...x_T so final
            # violations cannot disappear through a pre-state convention.
            for scope in ("pre_state", "post_state"):
                x = np.array([r["state"][0] for r in records]) if scope == "pre_state" else np.array([t["actual_next_0"] for t in transitions])
                v = x - 25.
                rows.append({"label": label, "alpha": alpha, "seed": seed, "scope": scope,
                    "mean_margin_X2": float(np.mean(v)), "min_margin_X2": float(np.min(v)),
                    "p1_margin_X2": float(np.percentile(v, 1)), "p5_margin_X2": float(np.percentile(v, 5)),
                    **{f"fraction_margin_lt_{limit:.2f}": float(np.mean(v < limit)) for limit in (.05, .1, .2)}})
    write_csv(out / "X2_margin_statistics.csv", rows)
    return rows


def action_audit(data, out):
    rows = []
    for seed, (_, records, _) in data[1.].items():
        for kind, v in (("raw_actor_action", np.array([r["actor_unscaled"] for r in records])),
                        ("requested_residual_normalized", np.array([r["residual"] for r in records])),
                        ("requested_residual_physical", np.array([r["residual_physical"] for r in records]))):
            for axis, name in enumerate(("P100", "F200")):
                x = v[:, axis]
                rows.append({"seed": seed, "kind": kind, "dimension": name,
                    "mean": float(np.mean(x)), "std": float(np.std(x)), "min": float(min(x)), "max": float(max(x)),
                    "abs_gt_0p9_fraction": float(np.mean(np.abs(x) > .9)),
                    "negative_fraction": float(np.mean(x < -1e-8)), "positive_fraction": float(np.mean(x > 1e-8)),
                    "zero_fraction": float(np.mean(np.abs(x) <= 1e-8)),
                    "threshold_note": "0.9 is actor saturation only for raw_actor_action; residual threshold is descriptive in its own units"})
    write_csv(out / "actor_action_statistics.csv", rows)
    return rows


def plot_outputs(sweep, selected, data, first, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["alpha"] for r in sweep]
    plots = {
        "residual_scale_vs_economics.png": (("economic_improvement_pct", "mean"), ("worst_economic_improvement_pct", "worst seed")),
        "residual_scale_vs_X2_IAE.png": (("X2_IAE_ratio", "X2 IAE ratio"), ("P2_IAE_ratio", "P2 IAE ratio")),
        "residual_scale_vs_G_X2.png": (("G_X2_ratio", "X2 empirical G ratio"), ("G_P2_ratio", "P2 empirical G ratio")),
        "residual_scale_vs_TV.png": (("P100_TV_ratio", "P100 TV ratio"), ("F200_TV_ratio", "F200 TV ratio")),
        "residual_scale_vs_safety.png": (("physical_violation", "physical next-state violations"), ("QP_infeasible", "QP infeasible"), ("Omega_exit", "Omega affected steps")),
    }
    for filename, specs in plots.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, name in specs:
            ax.plot(x, [r[key] for r in sweep], "o-", label=name)
        ax.axhline(0 if "economics" in filename or "safety" in filename else 1., color="gray", lw=.7)
        ax.set(xlabel="diagnostic raw actor action scale alpha", ylabel="paired metric")
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out / filename, dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, alpha in selected.items():
        v = np.concatenate([[r["state"][0]-25. for r in records] for _, records, _ in data[alpha].values()])
        ax.hist(v, bins=70, density=True, histtype="step", label=f"{label} alpha={alpha:g}")
    ax.axvline(0, color="red", lw=.8); ax.set(xlabel="X2 - 25 (percentage points)", ylabel="density")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "X2_margin_distribution.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, alpha in selected.items():
        v = [r for _, records, _ in data[alpha].values() for r in records]
        ax.scatter([r["actual_next_0"]-25. for r in v], [r["economic_cost"] for r in v], s=3, alpha=.2, label=f"{label} alpha={alpha:g}")
    ax.set(xlabel="next X2 - 25 (stage cost uses next state)", ylabel="economic stage cost")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "X2_margin_vs_economic_stage_cost.png", dpi=160); plt.close(fig)
    if first:
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
        t = [r["step"] for r in first]
        axes[0].plot(t, [r["state_0"] for r in first], label="pre X2")
        axes[0].plot(t, [r["actual_next_0"] for r in first], label="actual next X2")
        axes[0].plot(t, [r["predicted_W_next_min_0"] for r in first], label="W-predicted min X2")
        axes[0].axhline(25., color="red"); axes[0].legend()
        axes[1].plot(t, [r["control_0"] for r in first], label="P100")
        axes[1].plot(t, [r["control_1"] for r in first], label="F200"); axes[1].legend()
        axes[2].plot(t, [r["W_max_facet_utilization"] for r in first], label="W max facet utilization")
        axes[2].axhline(1., color="red"); axes[2].legend()
        axes[3].step(t, [int(r["QP_feasible"]) for r in first], label="QP feasible")
        axes[3].step(t, [int(r["Omega_before"]) for r in first], label="Omega before")
        axes[3].step(t, [int(r["physical_after_violation"]) for r in first], label="physical next violation"); axes[3].legend()
        axes[3].set_xlabel("transition step (zero-based)")
        fig.tight_layout(); fig.savefig(out / "first_failure_trace.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--actor", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "evaporation_safe_sac/outputs_zanon2019_failure_diagnosis")
    parser.add_argument("--steps", type=int, help="smoke-only override; full audit defaults to saved horizon")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Preserve existing diagnosis; choose a fresh --output-dir")
    manifest = json.loads((args.run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    steps = args.steps or manifest["steps"]
    actor_path = args.actor or args.run_dir / "models/evaluation/episode_0005_actor.pth"
    cfg, model, design, omega, domain, weights = make_setup(steps, manifest["seed"])
    cfg.episodes = manifest["episodes"]
    cfg.abort_on_first_uncertified_step = True  # inert in training=False; frozen trainer still aborts
    assert manifest["observation_dim"] == 23
    assert weights == manifest["weights"]
    tracked = [Path(__file__).with_name(p) for p in ("config.py", "control.py", "model.py", "sac.py", "train.py", "omega_interior_anchor.py", "omega_feasible_normalized.py", "zanon2019_benchmark.py")]
    tracked += [Path(p) for p in manifest["frozen_geometry_sources"]]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    for p, expected in manifest["frozen_geometry_sources"].items():
        assert hashes[p] == expected, f"Frozen geometry changed: {p}"
    actor = make_agent(cfg, "cpu")
    load_actor(actor, actor_path)
    actor_hash = hashlib.sha256(actor_path.read_bytes()).hexdigest()
    out.mkdir(parents=True)
    save_json(out / "diagnosis_manifest.json", {"run_dir": str(args.run_dir), "actor": str(actor_path),
        "actor_sha256": actor_hash, "steps": steps, "seeds": manifest["evaluation_seeds"],
        "alphas": list(ALPHAS), "weights_unchanged": weights, "certification_status": CERTIFICATION_STATUS,
        "training": False, "supervisor": "unchanged Gm/Bj inactive in stochastic no-jump protocol, as in saved run",
        "closed_loop": "same frozen neural parameters evaluated on each alpha's own state; not identical raw action time series across different states",
        "safety_failures": "continued ONLY in isolated no-learning diagnosis to observe cascades; trainer unchanged and still stops",
        "component_source": "current train.run_episode actual component records; replay RPI exclusion explicitly reconciled",
        "source_hashes_before": hashes})
    paths, bases, data, sweep, per_seed = {}, {}, {}, [], []
    for seed in manifest["evaluation_seeds"]:
        path, _ = sample_disturbance_path(cfg, manifest["mode"], seed, steps,
            manifest["variance_F1_X1_T1_T200"], 20, 50)
        # Compare saved realization exactly rather than relying only on seeds.
        saved = args.run_dir / "paired_baselines" / f"seed_{seed}.csv"
        with saved.open(encoding="utf-8-sig", newline="") as f:
            records = list(csv.DictReader(f))[:steps]
        np.testing.assert_array_equal(path, [[float(r[k]) for k in ("F1", "X1", "T1", "T200")] for r in records])
        paths[seed] = path
        bases[seed] = rollout(cfg, model, design, omega, domain, ZeroResidualPolicy(), path, seed)
        np.testing.assert_allclose([r["control"] for r in bases[seed][1]],
            [[float(r["baseline_control_0"]), float(r["baseline_control_1"])] for r in records], rtol=0, atol=1e-9)
    for alpha in ALPHAS:
        data[alpha], latest = {}, []
        for seed, path in paths.items():
            scaled = ScaledFrozenActor(actor, alpha)
            result = rollout(cfg, model, design, omega, domain, scaled, path, seed)
            stat, records, transitions = result
            bm = metrics(bases[seed][1], cfg, model, design, omega)
            pm = metrics(records, cfg, model, design, omega)
            for step, (r, t, raw) in enumerate(zip(records, transitions, scaled.unscaled)):
                t["raw_action_0"], t["raw_action_1"] = map(float, r["raw_action"])
                t.update({"alpha": alpha, "seed": seed, "actor_unscaled_0": float(raw[0]), "actor_unscaled_1": float(raw[1])})
                for key in ("residual", "applied_residual", "disturbance"):
                    t.update({f"{key}_{k}": float(v) for k, v in enumerate(r[key])})
                r.update(t)
                r["actor_unscaled"] = raw
                r["residual_physical"] = r["residual"] * cfg.input_scale
            if alpha == 0:
                np.testing.assert_array_equal([r["control"] for r in records], [r["control"] for r in bases[seed][1]])
                np.testing.assert_array_equal([r["state"] for r in records], [r["state"] for r in bases[seed][1]])
            if alpha == 1.:
                with (args.run_dir / "evaluation_trajectories/episode_0005" / f"seed_{seed}.csv").open(encoding="utf-8-sig", newline="") as f:
                    old = list(csv.DictReader(f))[:steps]
                np.testing.assert_allclose([r["control"] for r in records], [[float(r["SAC_control_0"]), float(r["SAC_control_1"])] for r in old], atol=1e-7, rtol=0)
            data[alpha][seed] = result
            row = {"alpha": alpha, "seed": seed, "economic_improvement_percent": 100.*(bm["J_econ"]-pm["J_econ"])/bm["J_econ"],
                **{k: relative(pm[v], bm[v]) for k,v in RATIO_METRICS.items()},
                **{"baseline_"+k: v for k,v in bm.items()}, **{"SAC_"+k: v for k,v in pm.items()}}
            latest.append(row); per_seed.append(row)
            write_csv(out / "trajectories" / f"alpha_{alpha:.2f}" / f"seed_{seed}.csv", transitions)
            print(f"alpha={alpha:.2f} seed={seed} improvement={row['economic_improvement_percent']:.6f}% physical={pm['physical_state_violation_steps']} Omega={pm['Omega_exit_count']} QP={pm['QP_infeasible_count']}", flush=True)
        status = assessment(latest, 5, 5000, cfg.warmup_steps)
        status.update({"alpha": alpha, "Z_exit": sum(r["SAC_local_RPI_Z_exit_count"] for r in latest),
            "maximum_X2_violation_magnitude": max(max(0., 25.-t["actual_next_0"]) for _,_,ts in data[alpha].values() for t in ts),
            "interior_radius_mean": float(np.mean([r["SAC_interior_radius_normalized_mean"] for r in latest])),
            "empirical_safe_positive_economic": status["empirical_safe"] and status["Omega_exit"] == 0 and status["positive_economic"],
            "checkpoint_status": "unsafe_and_performance_degraded_checkpoint" if alpha == 1. else "diagnostic_only_not_a_new_policy"})
        sweep.append(status)
        write_csv(out / "residual_scale_sweep.csv", sweep)
        write_csv(out / "residual_scale_per_seed.csv", per_seed)
    cascades = [cascade_for(seed, result[2]) for seed, result in data[1.].items()]
    save_json(out / "cascade_analysis.json", {"step_convention": "zero-based transition step; physical_after at step k means x_(k+1); Omega affected count unions before/after, not number of independent exit events",
        "seeds": cascades, "totals_strictly_after_each_seed_first_physical": {key: sum(c.get("strictly_after_first_physical", {}).get(key,0) for c in cascades) for key in ("physical", "Omega_affected_steps", "QP_infeasible")}})
    earliest = min((c for c in cascades if c["first_physical_transition_step"] is not None), key=lambda c:(c["first_physical_transition_step"],c["seed"]), default=None)
    first = []
    if earliest:
        p, seed = earliest["first_physical_transition_step"], earliest["seed"]
        first = data[1.][seed][2][max(0,p-10):min(steps,p+11)]
        write_csv(out / "first_failure_trace.csv", first)
    safe = [r for r in sweep if r["alpha"]>0 and r["empirical_safe_positive_economic"]]
    joint = [r for r in safe if r["level_C"]]
    eligible = joint or safe or [r for r in sweep if r["empirical_safe"] and r["Omega_exit"]==0]
    chosen = min(eligible, key=lambda r:(r["distance_to_joint"],-r["economic_improvement_pct"])) if eligible else min(sweep, key=lambda r:(r["physical_violation"]+r["Omega_exit"]+r["QP_infeasible"],r["distance_to_joint"]))
    selected = {"baseline": 0., "episode5_actor": 1., "frontier_representative": chosen["alpha"]}
    reward = reward_audit(selected, data, out)
    margins = margin_audit(selected, data, out)
    actions = action_audit(data, out)
    plot_outputs(sweep, selected, data, first, out)
    assert actor.total_updates == 0
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in hashes.items())
    save_json(out / "diagnosis_results.json", {"first_failure": earliest, "sweep": sweep,
        "safe_positive_alpha": [r["alpha"] for r in safe], "joint_alpha": [r["alpha"] for r in joint],
        "selected_for_audit_only": selected, "frozen_files_verified_unchanged": True,
        "actor_total_updates": actor.total_updates, "certification_status": CERTIFICATION_STATUS})
    lines = ["# ECC2019 episode-5 failure diagnosis", "", "No SAC training or design changes. All scales are diagnostic-only autonomous closed loops.",
        "", f"Scope: {CERTIFICATION_STATUS}. Episode 5: unsafe_and_performance_degraded_checkpoint.",
        "", "## First failure and cascade", "", "```json", json.dumps({"first":earliest,"cascades":cascades},indent=2), "```",
        "", "## Residual scale frontier", "", "|alpha|economics %|X2 IAE|P2 IAE|P100 TV|F200 TV|G X2|physical|QP|Omega|", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sweep:
        lines.append(f"|{r['alpha']:g}|{r['economic_improvement_pct']:.6f}|{r['X2_IAE_ratio']:.4f}|{r['P2_IAE_ratio']:.4f}|{r['P100_TV_ratio']:.4f}|{r['F200_TV_ratio']:.4f}|{r['G_X2_ratio']:.4f}|{r['physical_violation']}|{r['QP_infeasible']}|{r['Omega_exit']}|")
    lines += ["", f"Empirical-safe positive-economic scales on these paths: {[r['alpha'] for r in safe]}.",
        f"Level-C scales: {[r['alpha'] for r in joint]}. Representative audited scale only: {chosen['alpha']}.",
        "", "## Reward source and scope", "", "All components are recomputed by current train.run_episode. Its evaluation return includes RPI event/excess; stochastic SAC replay excludes these two, which remain separately audited. Component CSVs show signed contributions and magnitude ratios per seed, and per-step reward reconciliation. Physical-margin buffering penalty is not present; violation penalties activate only after violation. No new penalty is invented.",
        "", "## Interpretation limitations", "", "This is one early checkpoint and three fixed realizations, not a convergence result or proof about all Gaussian disturbances. Cascade counts are temporal associations, not proof that every later violation would disappear after repairing the first. Intermediate scales change the state fed to the frozen actor; no teacher forcing is used. A raw negative actor component need not imply a negative physical residual under the interior-anchor mapping; inspect actor and physical-residual sign statistics separately. Mean absolute economic reward versus penalty is not alone sufficient to prove reward dominance: examine incremental policy-vs-baseline reward differences as well.",
        "", "## Files", "", "residual_scale_sweep.csv, residual_scale_per_seed.csv, first_failure_trace.csv, cascade_analysis.json, reward_component_audit.csv, reward_components_per_step.csv, X2_margin_statistics.csv, actor_action_statistics.csv, diagnosis_results.json. All requested plots and full per-alpha/seed transition traces are saved alongside this report."]
    (out / "zanon2019_failure_diagnosis_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
