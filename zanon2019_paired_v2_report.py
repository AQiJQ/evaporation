"""Development-only figures; never pick a flattering disturbance realization.

Matplotlib scientific plots, unchanged raw validation points, no fabricated
learning/late seed data. Episode100 is a predeclared presentation checkpoint,
NOT final_v2 selection. All three individual training seeds remain visible.
"""
from pathlib import Path
import numpy as np
from . import zanon2019_paired_experiment as v1
from .zanon2019_benchmark import write_csv
from .zanon2019_paired_v2_objective import paired_metrics, components, operating_metrics


def figure_data(rows):
    """Single source for plots and numeric Fig.2 summary; no rescaling costs."""
    step = np.array([int(r["step"]) for r in rows])
    if not np.array_equal(step, np.arange(len(rows))):
        raise ValueError("Plot input must be a complete paired time axis")
    result = {"step": step}
    for name in ("baseline", "SAC"):
        for key in ("state_0", "state_1", "control_0", "control_1", "economic_cost"):
            result[name+"_"+key] = np.array([float(r[name+"_"+key]) for r in rows])
    result["delta"] = result["SAC_economic_cost"]-result["baseline_economic_cost"]
    if not np.allclose(result["delta"], [float(r["delta_l_SAC_minus_baseline"]) for r in rows], atol=1e-8):
        raise ValueError("Plot cost-difference sign/input mismatch")
    return result


def fixed_curve(rows, key):
    ordered = sorted(rows, key=lambda r: int(r["episode"]))
    if any(r.get("metric_source") != "fixed_paired_validation" for r in ordered):
        raise ValueError("Main learning curves require fixed paired validation")
    return [r["episode"] for r in ordered], [r.get(key) for r in ordered]


def pareto(rows):
    eligible = [r for r in rows if r["post_warmup"] and r["empirical_safe"]]
    def vec(r):
        return np.array([-r["mean_economic_improvement_pct"], r["X2_std_ratio"],
                         r["P100_TV_ratio"], r["F200_TV_ratio"], r["any_outer_10pct_fraction"]])
    for r in rows:
        r["pareto_nondominated"] = r in eligible and not any(
            np.all(vec(t) <= vec(r)) and np.any(vec(t) < vec(r)) for t in eligible)
    return rows


def report(root, env):
    from .zanon2019_paired_v2 import load, locked_config, archived_pair, calibration_sources, TRAIN_SEEDS
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})
    cfg, model, design = env[:3]
    config = locked_config(root)
    weight = config["lambda_smooth"]
    figures = root/"figures"; figures.mkdir(exist_ok=True)
    curves, training, summaries, all_rows = {}, {}, [], []
    for seed in TRAIN_SEEDS:
        directory = root/f"seed_{seed}"
        if not (directory/"checkpoints.json").exists():
            summaries.append(dict(training_seed=seed, run_state="not_run", checkpoint=None))
            continue
        manifest = load(directory/"experiment_manifest.json")
        if manifest["smoothness_config_sha256"] != v1.digest(root/"smoothness_regularization_config.json"):
            raise RuntimeError("Report config differs from immutable training reward")
        rows = load(directory/"checkpoints.json")
        for r in rows:
            r.update(training_seed=seed, metric_source="fixed_paired_validation")
        curves[seed] = pareto(rows)
        all_rows.extend(rows)
        if (directory/"training_log.csv").exists():
            training[seed] = v1.read_csv(directory/"training_log.csv")
        eligible = [r for r in rows if r["positive_economic_safe_learned"]]
        best = max(eligible, key=lambda r: r["mean_economic_improvement_pct"]) if eligible else {}
        summary = load(directory/"run_summary.json")
        summaries.append(dict(best, training_seed=seed, run_state=summary["run_state"],
            presentation="best-economic DEVELOPMENT descriptor; NOT final_v2 selection"))
    write_csv(root/"three_seed_training_summary.csv", summaries)
    write_csv(root/"three_seed_checkpoint_pareto.csv", all_rows)

    old_rows = []
    lock = load(v1.LOCK_ROOT/"locked_final_policies.json")
    for role, seed, dev, path in calibration_sources():
        if role != "Candidate_A": continue
        base, records = archived_pair(path)
        metric, _ = paired_metrics(records, base, env, weight, np.zeros(5))
        old_rows.append(dict(training_seed=seed, **metric))
    comparison = []
    for seed in TRAIN_SEEDS:
        old = [r for r in old_rows if r["training_seed"] == seed]
        selected = next(p for p in lock["policies"] if p["training_seed"] == seed)
        comparison.append(dict(training_seed=seed, experiment="locked_Candidate_A",
            episode=selected["selected_episode"], **{k: float(np.mean([r[k] for r in old]))
            for k in old[0] if isinstance(old[0][k], (int, float)) and k != "training_seed"}))
        for r in curves.get(seed, []):
            comparison.append(dict(experiment="Candidate_C_v2_all_development_checkpoints", **r))
    write_csv(root/"old_vs_v2_reward_comparison.csv", comparison)
    v1.save(root/"report_status.json", dict(training_seeds_available=list(curves),
        training_seeds_not_run=[s for s in TRAIN_SEEDS if s not in curves], representative_seed=420000,
        presentation_checkpoint="episode100 for each seed, not chosen by performance",
        final_selection_rule_locked=False, final_test_performed=False,
        figures_pending=not bool(curves), total_control_activity_definition="(TV_P100/100+TV_F200/100)/1000",
        performance_sampling="pre-step 1000 states; minimum margin includes terminal next-state",
        certification_status="empirical_only_under_ECC2019_stochastic_disturbance"))
    if not curves:
        print("No v2 training: comparison prepared; policy/learning figures pending real local training.", flush=True)
        return

    def savefig(fig, name, target=figures):
        target.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(target/(name+".png"), dpi=180)
        plt.close(fig)

    def draw(ax, keys, title, reference=None):
        for seed, rows in curves.items():
            for i, key in enumerate(keys):
                x, y = fixed_curve(rows, key)
                good = [(a, b) for a, b in zip(x, y) if b is not None]
                ax.plot([p[0] for p in good], [p[1] for p in good], marker="o", ms=2.5,
                        ls=("-", "--", ":", "-.")[i % 4], label=f"seed {seed}: {key}")
        if reference is not None: ax.axhline(reference, color="grey", lw=.8, ls="--")
        ax.set(xlabel="Training episode", ylabel=title)
        ax.legend(fontsize=7)

    spec = {
        "economic_improvement": (["mean_economic_improvement_pct", "worst_seed_economic_improvement_pct"], "Paired economic improvement (%)", 0),
        "X2_IAE_ratio": (["X2_IAE_ratio"], "X2 IAE relative to B / baseline", 1),
        "X2_centered_fluctuation": (["centered_X2_MAE_ratio", "X2_std_ratio"], "Centered X2 fluctuation ratio", 1),
        "X2_mean_shift": (["mean_X2_shift_vs_baseline"], "X2 mean shift (percentage points)", 0),
        "control_TV": (["P100_TV_ratio", "F200_TV_ratio"], "Actual-input TV ratio", 1),
        "control_RMS_du": (["P100_RMS_du_ratio", "F200_RMS_du_ratio"], "Actual-input RMS move ratio", 1),
        "reward_components": (["economic_reward_component", "smoothness_regularization_component", "lagrangian_constraint_component", "total_training_reward"], "Fixed paired evaluation reward sums", 0),
        "smoothness_fraction": (["regularizer_fraction"], "Regularizer / sum |stepwise economics|", .05),
    }
    for name, (keys, title, reference) in spec.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        draw(ax, keys, title, reference)
        savefig(fig, "learning_curve_"+name)
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for ax, key in zip(axs, ("minimum_X2_margin", "physical_state_violation_steps", "QP_infeasible_count", "Omega_exit_count")):
        draw(ax, [key], key, 0)
    savefig(fig, "learning_curve_safety")
    for key, name in (("episode_return", "episode_return"), ("actor_loss", "actor_loss"), ("q_loss", "critic_loss"), ("entropy", "entropy")):
        fig, ax = plt.subplots(figsize=(10, 4))
        drawn = False
        for seed, rows in training.items():
            good = [r for r in rows if r.get(key) not in (None, "") and np.isfinite(float(r[key]))]
            if good:
                drawn = True
                ax.plot([int(r["episode"]) for r in good], [float(r[key]) for r in good], label=f"seed {seed}")
        if drawn:
            ax.set(xlabel="Training episode", ylabel=key, title="Diagnostic training log, not main learning evidence")
            ax.legend(); savefig(fig, "learning_curve_"+name)
        else: plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    for ax, name in zip(axs.flat, ("economic_improvement", "X2_IAE_ratio", "X2_centered_fluctuation", "control_TV")):
        draw(ax, *spec[name])
    savefig(fig, "paper_learning_curves_combined")
    fig, axs = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for ax, key in zip(axs, spec["reward_components"][0]): draw(ax, [key], key, 0)
    savefig(fig, "paper_reward_decomposition")
    for name, key in (("X2_IAE", "X2_IAE_ratio"), ("X2_std", "X2_std_ratio"),
        ("centered_X2_MAE", "centered_X2_MAE_ratio"), ("min_X2_margin", "minimum_X2_margin"),
        ("P100_TV", "P100_TV_ratio"), ("F200_TV", "F200_TV_ratio"), ("total_control_activity", "total_control_activity")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for seed, rows in curves.items():
            ax.scatter([r["mean_economic_improvement_pct"] for r in rows], [r[key] for r in rows],
                       label=f"seed {seed}", s=20, alpha=.65)
        ax.axvline(0, ls="--", color="grey")
        if key.endswith("ratio"): ax.axhline(1, ls="--", color="grey")
        ax.set(xlabel="Paired economic improvement (%)", ylabel=key)
        ax.legend(); savefig(fig, "economic_vs_"+name)

    for seed in curves:
        trajectory = root/f"seed_{seed}/evaluation_trajectories/episode_0100/seed_420000.csv"
        if not trajectory.exists(): continue
        data = figure_data(v1.read_csv(trajectory))
        base, records = archived_pair(trajectory)
        metric, _ = paired_metrics(records, base, env, weight, np.zeros(5))
        # Include the last physical state in the response and min annotations.
        for name, rows in (("baseline", base), ("SAC", records)):
            last = rows[-1]
            next_n = design.a@model.normalized_state(last["state"])+design.b@model.normalized_input(last["control"])+design.affine+last["w_hat"]
            terminal = model.physical_state(next_n)
            for axis in range(2):
                key = f"{name}_state_{axis}"
                data[key] = np.r_[data[key], terminal[axis]]
        target = figures/f"seed_{seed}"
        panel_keys = ("state_0", "state_1", "control_0", "control_1")
        labels = ("X2", "P2", "P100", "F200")
        boundaries = ([25.], cfg.state_lower[1:2].tolist()+cfg.state_upper[1:2].tolist(),
                      [model.physical_input(design.robust_input_lower)[0], model.physical_input(design.robust_input_upper)[0]],
                      [model.physical_input(design.robust_input_lower)[1], model.physical_input(design.robust_input_upper)[1]])

        def response(ax, i):
            key = panel_keys[i]
            times = np.arange(len(data["baseline_"+key]))
            ax.plot(times, data["baseline_"+key], label="Paired zero-residual baseline", lw=.8)
            ax.plot(times, data["SAC_"+key], label=f"Safe-SAC v2 seed {seed} ep100", lw=.8, alpha=.85)
            for bound in boundaries[i]: ax.axhline(bound, ls="--", color="grey", lw=.8)
            ax.set_ylabel(labels[i]); ax.legend(fontsize=7)
            if i == 0:
                ax.set_ylim(24.98, max(data["baseline_state_0"].max(), data["SAC_state_0"].max())+.05)
                ax.set_title(f"min baseline={min(data['baseline_state_0']):.5f}, SAC={min(data['SAC_state_0']):.5f}; constraint X2>=25")
        for i, label in enumerate(labels):
            fig, ax = plt.subplots(figsize=(11, 4)); response(ax, i); ax.set_xlabel("Time step")
            savefig(fig, "fig2_style_"+label+"_response", target)
        fig, ax = plt.subplots(figsize=(11, 4)); response(ax, 0)
        ax.set_xlim(0, min(200, len(data["step"])-1)); ax.set_xlabel("Time step; fixed first 200 steps")
        # Same y-range as full plot: no artificial magnification.
        savefig(fig, "fig2_style_X2_zoomed", target)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(data["step"], data["delta"], lw=.8); ax.axhline(0, ls="--", color="grey")
        ax.set(xlabel="Time step", ylabel="l_SAC - l_baseline (negative favors SAC)")
        savefig(fig, "fig2_style_instantaneous_economic_difference", target)
        fig, ax = plt.subplots(figsize=(11, 4))
        for name in ("baseline", "SAC"):
            ax.plot(data["step"], np.cumsum(data[name+"_economic_cost"]), label=name)
        ax.set(xlabel="Time step", ylabel="Cumulative economic cost", title=f"Paired gain={metric['economic_improvement_pct']:.6f}%")
        ax.legend(); savefig(fig, "fig2_style_cumulative_economic_cost", target)
        fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
        for i in range(4): response(axs[i], i)
        axs[4].plot(data["step"], data["delta"], lw=.8); axs[4].axhline(0, color="grey", ls="--")
        axs[4].set(ylabel="l_SAC - l_baseline", xlabel="Time step")
        fig.suptitle("Fig.2-type presentation, not ECC2019 Fig.2 reproduction; validation seed420000")
        savefig(fig, "paper_fig2_style_combined", target)
        v1.save(target/"fig2_summary.json", dict(training_seed=seed, episode=100, validation_seed=420000,
            source=str(trajectory), source_sha256=v1.digest(trajectory), **metric,
            SAC_instantaneous_better_fraction=float(np.mean(data["delta"] < 0)),
            final_cumulative_cost_gap_SAC_minus_baseline=float(data["delta"].sum())))
        if seed == 42:
            # Fixed seed42 presentation at root, other seeds also kept separately.
            import shutil
            for p in target.glob("*.png"): shutil.copyfile(p, figures/p.name)
