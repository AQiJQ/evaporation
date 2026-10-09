"""Development reports with missing runs exposed, not invented or pooled as seeds."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import numpy as np
import torch

from . import zanon2019_paired_experiment as v1
from . import zanon2019_paired_v2 as v2
from .zanon2019_benchmark import make_agent, load_actor, sample_disturbance_path, PAPER_VARIANCES_F1_X1_T1_T200, write_csv
from .zanon2019_economic_recovery import (SEEDS, DEV, PAIRS, run_dir, read, save, immutable,
    locked, environment, full_metrics, trace, weight, protocol_key)
from .zanon2019_economic_recovery_metrics import (pair_economics, control_activity, recovery, aggregate, headroom_utilization)
from .train import ZeroResidualPolicy


def numeric(row):
    result = {}
    for k, v in row.items():
        if v in (None, ""): result[k] = None
        elif v in ("True", "False"): result[k] = v == "True"
        else:
            try: result[k] = float(v)
            except (ValueError, TypeError): result[k] = v
    return result


def freeze_figure(root, pair, device):
    protocol = locked(root)
    choice = protocol["presentation_checkpoint"]
    seed, episode, dev = choice["training_seed"], choice["episode"], choice["validation_seed"]
    source = run_dir(root, pair, seed)
    if not (source/"run_summary.json").exists() or read(source/"run_summary.json")["run_state"] != "completed":
        raise RuntimeError("Complete training before development presentation freezing")
    checkpoint = source/"models"/f"episode_{episode:04d}_actor.pth"
    statuses = read(source/"checkpoints.json")
    status = next(r for r in statuses if r["episode"] == episode)
    if not status["post_warmup"] or not status["empirical_safe"]:
        raise RuntimeError("Presentation requires post-warmup empirical-safe checkpoint")
    target = root/pair/"frozen_presentation"
    v1.fresh(target)
    actor = target/"frozen_actor.pth"; shutil.copy2(checkpoint, actor)
    selection = dict(**choice, pair=PAIRS[pair], checkpoint_source=str(checkpoint),
        source_sha256=v1.digest(checkpoint), frozen_actor_sha256=v1.digest(actor),
        checkpoint_protocol="post-training episode100 development presentation only; NOT economic_first_final_v2",
        trajectory_source="new paired rollout after policy freeze, never a training trajectory",
        protocol_sha256=v1.digest(root/"protocol.json"), final_test_performed=False)
    immutable(target/"selection.json", selection)
    env = environment(root, pair, seed)
    agent = make_agent(env[0], device); load_actor(agent, actor)
    before = {k: v.detach().clone() for k, v in agent.actor.state_dict().items()}
    path, info = sample_disturbance_path(env[0], "zanon2019_stochastic", dev, 1000,
                                       PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
    with torch.no_grad():
        base, _, _ = v1.rollout(env, path, dev, ZeroResidualPolicy())
        records, _, _ = v1.rollout(env, path, dev, agent)
    if v1.digest(actor) != selection["frozen_actor_sha256"] or any(
            not torch.equal(before[k], v) for k, v in agent.actor.state_dict().items()):
        raise RuntimeError("Policy changed during frozen evaluation")
    metric, _, series = full_metrics(base, records, env, weight(root, pair), np.zeros(5))
    write_csv(target/"paired_rollout.csv", trace(base, records, series))
    immutable(target/"summary.json", dict(**metric, disturbance=info, selection=selection,
        trajectory_sha256=v1.digest(target/"paired_rollout.csv"), certification_status="empirical_only_under_ECC2019_stochastic_disturbance"))


def fixed_evidence(root, pair):
    per, curves, sources = [], {}, []
    reference = read(root/"protocol.json")["strong_reference"] if pair == "strong" else \
        read(root/"conservative/conservative_baseline_selection.json")["selected_reference"]
    env = environment(root, pair)
    lower, upper = env[1].physical_input(env[2].robust_input_lower), env[1].physical_input(env[2].robust_input_upper)
    for seed in SEEDS:
        directory = run_dir(root, pair, seed)
        if not (directory/"fixed_evaluation.csv").exists(): continue
        if pair == "strong" and directory.parent == v2.ROOT:
            manifest = read(directory/"experiment_manifest.json")
            if manifest["smoothness_config_sha256"] != read(root/"protocol.json")["strong_config_sha256"]:
                raise RuntimeError("Imported Strong run has wrong frozen objective")
        grouped = defaultdict(list)
        source_csv = directory/"fixed_evaluation.csv"
        sources.append(dict(training_seed=seed, source=str(source_csv), sha256=v1.digest(source_csv), read_only=True))
        for row in v1.read_csv(source_csv):
            metric = numeric(row)
            ep, dev = int(metric["episode"]), int(metric["validation_seed"])
            if dev not in DEV: raise ValueError("Non-development evidence in learning curve")
            path = directory/"evaluation_trajectories"/f"episode_{ep:04d}"/f"seed_{dev}.csv"
            base, records = v2.archived_pair(path)
            economic, _ = pair_economics(base, records, reference)
            if not np.isclose(economic["economic_improvement_pct"], metric["economic_improvement_pct"], atol=1e-9):
                raise RuntimeError("Archived fixed evaluation economics inconsistent")
            metric.update(economic, training_seed=seed, pair=PAIRS[pair], metric_source="fixed_paired_validation")
            metric.update(control_activity(records, lower, upper))
            metric.update({"baseline_"+k: v for k, v in control_activity(base, lower, upper).items()})
            per.append(metric); grouped[ep].append(metric)
        curves[seed] = []
        for ep, rows in sorted(grouped.items()):
            state = v2.assessment(rows, ep, ep*1000, 5000)
            state.update(training_seed=seed, pair=PAIRS[pair], metric_source="fixed_paired_validation")
            curves[seed].append(state)
    return per, curves, sources


def figure_source(root, pair):
    target = root/pair/"frozen_presentation"
    if not (target/"summary.json").exists(): return None
    summary = read(target/"summary.json")
    selection = read(target/"selection.json")
    if selection["trajectory_source"] != "new paired rollout after policy freeze, never a training trajectory" or \
        v1.digest(target/"frozen_actor.pth") != selection["frozen_actor_sha256"] or \
        v1.digest(target/"paired_rollout.csv") != summary["trajectory_sha256"]:
        raise RuntimeError("Frozen presentation provenance changed")
    return [numeric(r) for r in v1.read_csv(target/"paired_rollout.csv")]


def report(root):
    locked(root)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})
    outputs, pending, summaries, evidence, finalpaths = [], [], {}, {}, {}
    figure_folder = root/"figures"

    def figsave(fig, filename, folder=figure_folder):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/filename; fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
        outputs.append(str(path))

    for pair in PAIRS:
        available = pair == "strong" or (root/"conservative/conservative_baseline_selection.json").exists() and \
            read(root/"conservative/conservative_baseline_selection.json")["selected_reference"] is not None
        per, curves, sources = fixed_evidence(root, pair) if available else ([], {}, [])
        evidence[pair] = (per, curves, sources)
        write_csv(root/pair/"paired_economics_per_realization.csv", per)
        summary = []
        for seed in SEEDS:
            directory = run_dir(root, pair, seed)
            row = dict(training_seed=seed, pair=PAIRS[pair], run_state="not_run", presentation_episode=None)
            if (directory/"run_summary.json").exists(): row.update(read(directory/"run_summary.json"))
            if seed in curves:
                latest = next((r for r in curves[seed] if r["episode"] == 100), None)
                if latest:
                    row.update(latest, presentation_episode=100)
                    finalpaths[pair, seed] = [r for r in per if r["training_seed"] == seed and r["episode"] == 100]
            summary.append(row)
        summaries[pair] = summary
        write_csv(root/"comparison"/f"three_seed_{pair}_summary.csv", summary)
        write_csv(root/pair/"paired_economics_summary.csv", summary)
        save(root/pair/"sources.json", dict(fixed_learning_sources=sources,
             closed_loop_sources="frozen_presentation only; NOT fixed learning or training log trajectories"))
        metrics = ("mean_economic_improvement_pct", "economic_win_fraction", "mean_X2_shift_vs_baseline",
                   "std_X2", "centered_X2_MAE", "minimum_X2_margin", "P100_TV", "F200_TV",
                   "P100_delta_RMS", "F200_delta_RMS")
        finished = [r for r in summary if r["run_state"] == "completed" and r["presentation_episode"] == 100]
        write_csv(root/"comparison"/f"{pair}_training_seed_statistics.csv", aggregate(finished, metrics))
        specs = {
            "economic_improvement": (["mean_economic_improvement_pct", "worst_seed_economic_improvement_pct"], "Paired economic improvement (%)", 0.),
            "economic_win_fraction": (["economic_win_fraction"], "Fraction of steps with lower SAC cost", .5),
            "X2_mean_shift": (["mean_X2_shift_vs_baseline"], "X2 mean shift vs own baseline (pp)", 0.),
            "X2_centered_fluctuation": (["X2_std_ratio", "centered_X2_MAE_ratio"], "Centered X2 fluctuation ratio", 1.),
            "control_TV": (["P100_TV_ratio", "F200_TV_ratio"], "Actual-input TV ratio", 1.),
            "control_RMS": (["P100_RMS_du_ratio", "F200_RMS_du_ratio"], "Actual-input RMS move ratio", 1.),
            "fixed_reward": (["economic_reward_component", "smoothness_regularization_component", "total_training_reward"], "Fixed paired evaluation reward sum", 0.),
        }
        for name, (keys, label, reference) in specs.items():
            filename = f"learning_{pair}_{name}.png"
            if not curves:
                pending.append(str(root/pair/"learning_curves"/filename)); continue
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for seed, rows in curves.items():
                for key in keys:
                    ax.plot([r["episode"] for r in rows], [r[key] for r in rows],
                            marker=".", ms=3, label=f"seed {seed}: {key}")
            ax.axhline(reference, ls="--", color="grey", lw=.7)
            ax.set(xlabel="Training episode", ylabel=label,
                   title=f"{pair.capitalize()} pair: fixed development evaluation; {len(curves)}/3 training seeds")
            ax.legend(fontsize=7); figsave(fig, filename, root/pair/"learning_curves")
        # Supplementary raw return, clearly NOT the main economic learning curve.
        training = {s: v1.read_csv(run_dir(root, pair, s)/"training_log.csv") for s in curves
                    if (run_dir(root, pair, s)/"training_log.csv").exists()}
        if training:
            fig, ax = plt.subplots(figsize=(10, 4))
            for seed, rows in training.items():
                ax.plot([float(r["episode"]) for r in rows], [float(r["episode_return"]) for r in rows], label=f"seed {seed}")
            ax.set(xlabel="Episode", ylabel="Paired episode return", title="Supplementary training return, not fixed-validation learning evidence")
            ax.legend(); figsave(fig, f"learning_{pair}_training_return_diagnostic.png", root/pair/"learning_curves")
        data = figure_source(root, pair)
        if data:
            fig, axs = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
            t = np.arange(len(data))
            for i, (key, label) in enumerate((("state_0", "X2 (pp)"), ("state_1", "P2 (kPa)"),
                                               ("control_0", "P100 (kPa)"), ("control_1", "F200"))):
                axs[i].plot(t, [r["baseline_"+key] for r in data], label=PAIRS[pair], lw=.8)
                axs[i].plot(t, [r["SAC_"+key] for r in data], label=f"{pair} + frozen Safe-SAC", lw=.8)
                axs[i].set_ylabel(label); axs[i].legend(fontsize=8)
            axs[0].axhline(25, ls="--", color="black", lw=.7, label="X2>=25")
            axs[0].set_ylim(24.98, max(max(r["baseline_state_0"], r["SAC_state_0"]) for r in data)+.03)
            for ax, key, label in ((axs[4], "instantaneous_economic_advantage_pct", "Instantaneous advantage (%)"),
                                    (axs[5], "G_cum", "Cumulative advantage (%)")):
                ax.plot(t, [r[key] for r in data], lw=.8); ax.axhline(0, ls="--", color="grey"); ax.set_ylabel(label)
            axs[5].set_xlabel("Sampling step (1 s)")
            fig.suptitle(f"{pair.capitalize()} pair, frozen episode100 actor, development disturbance 420000\nFig.2-style presentation, not ECC2019 controller reproduction")
            figsave(fig, f"{pair}_fig2_style_combined.png")
        else: pending.append(str(figure_folder/f"{pair}_fig2_style_combined.png"))

    # Four method comparison: unavailable columns stay blank, never synthetic zero.
    table = []
    keys = ("J_econ", "mean_stage_cost", "minimum_X2_margin", "mean_X2", "std_X2", "centered_X2_MAE",
            "X2_IAE", "X2_ISE", "P2_IAE", "P2_ISE", "P100_TV", "F200_TV", "P100_delta_RMS", "F200_delta_RMS",
            "physical_state_violation_steps", "physical_input_violation_count", "QP_infeasible_count", "Omega_exit_count")
    for key in keys:
        row = dict(metric=key, aggregation="development seed42 episode100, mean of 10 disturbance paths; not final or cross-reference ranking")
        for pair in PAIRS:
            rows = finalpaths.get((pair, 42), [])
            for sac in (False, True):
                label = PAIRS[pair]+("_plus_SAC" if sac else "")
                source = key if sac else "baseline_"+key
                if key == "mean_stage_cost":
                    values = [r["J_econ" if sac else "baseline_J_econ"]/1000. for r in rows]
                elif not sac and key in v1.SAFETY:
                    # Baseline was audited before training/evaluation; do not infer from SAC counters.
                    values = [0. for r in rows] if rows and all(r["baseline_J_econ"] > 0 for r in rows) else []
                else: values = [r[source] for r in rows if r.get(source) is not None]
                row[label] = float(np.mean(values)) if len(values) == 10 else None
        table.append(row)
    write_csv(root/"comparison/four_method_economic_comparison.csv", table)
    recoveries = []
    strongpaths = {r["validation_seed"]: r for r in finalpaths.get(("strong", 42), [])}
    for seed in SEEDS:
        for row in finalpaths.get(("conservative", seed), []):
            dev = row["validation_seed"]
            if dev not in strongpaths: continue
            r = recovery(strongpaths[dev]["baseline_J_econ"], row["baseline_J_econ"], row["J_econ"])
            recoveries.append(dict(training_seed=seed, validation_seed=dev, **r,
                                   aggregation="per disturbance recovery ratio; cross-reference initial states differ"))
    recovery_summary = []
    for seed in SEEDS:
        rows = [r for r in recoveries if r["training_seed"] == seed]
        if not rows: continue
        penalty = float(np.mean([r["conservatism_cost"] for r in rows]))
        recovered_cost = float(np.mean([r["recovered_cost"] for r in rows]))
        ratios = [r["economic_recovery_ratio_pct"] for r in rows if r["economic_recovery_ratio_pct"] is not None]
        recovery_summary.append(dict(training_seed=seed, n_development_paths=len(rows), conservatism_cost=penalty,
            recovered_cost=recovered_cost, economic_recovery_ratio_pct=100.*recovered_cost/penalty if penalty > 0 else None,
            mean_per_path_recovery_ratio_pct=float(np.mean(ratios)) if ratios else None,
            ratio_definition="ratio of mean costs; mean of per-path ratios is separately labelled"))
    write_csv(root/"comparison/economic_recovery_per_realization.csv", recoveries)
    write_csv(root/"comparison/economic_recovery_summary.csv", recovery_summary or [dict(status="pending_conservative_results",
        conservatism_cost=None, recovered_cost=None, economic_recovery_ratio_pct=None)])
    write_csv(root/"comparison/recovery_training_seed_statistics.csv", aggregate(recovery_summary,
        ["conservatism_cost", "recovered_cost", "economic_recovery_ratio_pct"]))
    headroom = []
    for pair in PAIRS:
        path = root/pair/f"oracle_safe_economic_upper_bound_{pair}.json"
        result = read(path) if path.exists() else None
        for row in summaries[pair]:
            gain = row.get("mean_economic_improvement_pct")
            oracle_gain = result["best_found_improvement_pct"] if result else None
            env = environment(root, pair) if result else None
            util = headroom_utilization(gain, oracle_gain, protocol_key(env, pair), result["protocol"]) \
                if gain is not None and result else None
            headroom.append(dict(pair=PAIRS[pair], training_seed=row["training_seed"], SAC_gain_pct=gain,
                oracle_best_found_gain_pct=oracle_gain, utilization_pct=util,
                oracle_status="best_found_not_certified_upper_bound" if result else "pending_denser_search"))
    write_csv(root/"comparison/oracle_headroom_comparison.csv", headroom)
    # Only draw genuinely four-method views when both frozen pairs exist.
    strongdata, cdata = figure_source(root, "strong"), figure_source(root, "conservative")
    four_names = ("four_method_X2_comparison.png", "four_method_cumulative_economic_cost.png", "economic_recovery_from_conservatism.png")
    if strongdata and cdata:
        for filename, state in zip(four_names[:2], (True, False)):
            fig, ax = plt.subplots(figsize=(11, 5))
            for pair, data in (("strong", strongdata), ("conservative", cdata)):
                for prefix in ("baseline", "SAC"):
                    values = [r[prefix+("_state_0" if state else "_economic_cost")] for r in data]
                    ax.plot(np.arange(len(values)), values if state else np.cumsum(values), lw=.8, label=PAIRS[pair]+(" + SAC" if prefix == "SAC" else ""))
            if state: ax.axhline(25., ls="--", color="black")
            ax.set(xlabel="Sampling step (1 s)", ylabel="X2 (pp)" if state else "Cumulative economic cost",
                   title="Reference sensitivity: interpret each pair, not a cross-reference global ranking")
            ax.legend(fontsize=8); figsave(fig, filename)
        js = sum(r["baseline_economic_cost"] for r in strongdata)
        jc = sum(r["baseline_economic_cost"] for r in cdata); jcs = sum(r["SAC_economic_cost"] for r in cdata)
        rec = recovery(js, jc, jcs)
        fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(["Strong baseline", "Conservative baseline", "Conservative + SAC"], [js, jc, jcs])
        ax.set(ylabel="Total economic cost", title=f"Development 420000, recovery ratio={rec['economic_recovery_ratio_pct']}%")
        figsave(fig, four_names[2])
    else: pending.extend(str(figure_folder/n) for n in four_names)
    sweep = root/"conservative/baseline_conservatism_sweep.csv"
    if sweep.exists():
        grouped = defaultdict(list)
        for r in v1.read_csv(sweep):
            if r.get("J_econ"): grouped[float(r["delta_X2"])].append(numeric(r))
        if grouped:
            zero_cost = np.mean([r["J_econ"] for r in grouped.get(0., [])]) if 0. in grouped else None
            if zero_cost is not None:
                fig, ax = plt.subplots(figsize=(8, 5))
                for delta, rows in grouped.items():
                    margin = min(r["minimum_X2_margin"] for r in rows)
                    cost = np.mean([r["J_econ"] for r in rows])-zero_cost
                    ax.scatter(margin, cost); ax.annotate(f"delta={delta:.2f}", (margin, cost))
                ax.set(xlabel="Worst development minimum X2 margin (pp)", ylabel="Mean J_econ minus Strong baseline")
                figsave(fig, "baseline_safety_economics_tradeoff.png")
    else: pending.append(str(figure_folder/"baseline_safety_economics_tradeoff.png"))
    if recoveries:
        fig, ax = plt.subplots(figsize=(8, 5))
        for pair in PAIRS:
            for seed in SEEDS:
                for row in finalpaths.get((pair, seed), []):
                    ax.scatter(row["minimum_X2_margin"], row["J_econ"], label=f"{pair} seed{seed}" if row["validation_seed"] == 420000 else None, s=20)
        ax.set(xlabel="Remaining minimum X2 margin", ylabel="J_econ", title="Development pairwise safety-economic sensitivity")
        ax.legend(fontsize=7); figsave(fig, "safe_sac_economic_recovery_tradeoff.png")
    else: pending.append(str(figure_folder/"safe_sac_economic_recovery_tradeoff.png"))
    save(root/"comparison/development_summary.json", dict(training_runs=summaries, headroom=headroom,
        recovered_per_realization=recoveries, completed_figures=outputs, pending_figures=pending,
        three_seed_CI="Student t across training seeds; n=1 CI unavailable, never ten validation seeds as ten training replications",
        independent_test_performed=False, final_selection_rule="not locked",
        no_claim_of_pointwise_dominance=True, no_claim_of_certified_nonlinear_Gaussian_safety=True))
    print(f"Report: {len(outputs)} actual figures; {len(pending)} pending, no missing-result fabrication.", flush=True)
