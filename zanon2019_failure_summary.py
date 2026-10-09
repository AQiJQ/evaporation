"""Summarize saved offline diagnosis evidence without any plant/policy updates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .zanon2019_benchmark import REPO_DIR, write_csv, make_setup
from .zanon2019_failure_diagnosis import PENALTIES
from .zanon2019_train import save_json


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def vector(rows, key):
    return np.array([float(r[key]) for r in rows])


def distribution(v):
    return {"mean": float(np.mean(v)), "std": float(np.std(v)),
            "min": float(np.min(v)), "max": float(np.max(v)),
            "p1": float(np.percentile(v,1)), "p5": float(np.percentile(v,5)),
            "fraction_abs_gt_0p9": float(np.mean(np.abs(v)>.9)),
            "negative_fraction": float(np.mean(v < -1e-8)),
            "positive_fraction": float(np.mean(v > 1e-8))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "evaporation_safe_sac/outputs_zanon2019_failure_diagnosis")
    args = parser.parse_args()
    out = args.output_dir
    results = json.loads((out / "diagnosis_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "diagnosis_manifest.json").read_text(encoding="utf-8"))
    cascade = json.loads((out / "cascade_analysis.json").read_text(encoding="utf-8"))
    raw_reward = read_csv(out / "reward_components_per_step.csv")
    aggregates, margins, actor_stats, shape_ratios = [], [], [], {}
    for label, alpha in results["selected_for_audit_only"].items():
        rewards = [r for r in raw_reward if r["label"] == label]
        econ_abs = float(np.mean(np.abs(vector(rewards,"economic_reward"))))
        components = ["economic_reward", *PENALTIES,
                      "actual_training_replay_reward_same_transition", "actual_evaluation_reward"]
        shape_ratios[label] = {}
        for key in components:
            v = vector(rewards,key)
            ratio = float(np.mean(np.abs(v))) / max(econ_abs,1e-12)
            shape_ratios[label][key] = ratio
            aggregates.append({"label": label, "alpha": alpha, "component":key,
                "mean_signed":float(np.mean(v)), "median_signed":float(np.median(v)),
                "mean_abs":float(np.mean(np.abs(v))), "p95_abs":float(np.percentile(np.abs(v),95)),
                "cumulative_all_seeds":float(np.sum(v)),
                "mean_episode_cumulative":float(np.sum(v)/len(manifest["seeds"])),
                "mean_abs_over_economic":ratio})
        trajectories = [r for seed in manifest["seeds"] for r in read_csv(out / "trajectories" / f"alpha_{alpha:.2f}" / f"seed_{seed}.csv")]
        for scope, key in (("pre_state", "state_0"), ("post_state", "actual_next_0")):
            v = vector(trajectories,key)-25.
            margins.append({"label":label,"alpha":alpha,"scope":scope,
                "mean_margin":float(np.mean(v)), "min_margin":float(np.min(v)),
                "p1_margin":float(np.percentile(v,1)), "p5_margin":float(np.percentile(v,5)),
                **{f"fraction_lt_{limit:.2f}":float(np.mean(v<limit)) for limit in (.05,.1,.2)}})
        if alpha == 1.:
            for axis, name in enumerate(("P100","F200")):
                raw = vector(trajectories,f"actor_unscaled_{axis}")
                residual = vector(trajectories,f"residual_{axis}")
                for kind, v in (("raw_actor_action",raw), ("requested_residual_normalized",residual)):
                    actor_stats.append({"dimension":name,"kind":kind,**distribution(v),
                        "raw_nonzero_sign_disagreement_with_requested_residual":float(np.mean(
                            (np.sign(raw)!=np.sign(residual)) & (np.abs(raw)>1e-8) & (np.abs(residual)>1e-8)))})
    write_csv(out / "reward_component_aggregate.csv",aggregates)
    write_csv(out / "X2_margin_aggregate.csv",margins)
    write_csv(out / "actor_action_aggregate.csv",actor_stats)
    increment = []
    for label in results["selected_for_audit_only"]:
        for key in ["economic_reward",*PENALTIES,"actual_training_replay_reward_same_transition"]:
            b = next(r for r in aggregates if r["label"]=="baseline" and r["component"]==key)
            p = next(r for r in aggregates if r["label"]==label and r["component"]==key)
            increment.append({"label":label,"component":key,
                "delta_mean_contribution_vs_baseline":p["mean_signed"]-b["mean_signed"],
                "delta_mean_episode_contribution_vs_baseline":p["mean_episode_cumulative"]-b["mean_episode_cumulative"]})
    write_csv(out / "reward_increment_vs_baseline.csv",increment)
    details = []
    for c in cascade["seeds"]:
        trace = read_csv(out / "trajectories/alpha_1.00" / f"seed_{c['seed']}.csv")
        failed = [r for r in trace if r["QP_feasible"]=="False"]
        details.append({"seed":c["seed"],"QP_infeasible_total":len(failed),
            "QP_infeasible_with_Omega_before_false":sum(r["Omega_before"]=="False" for r in failed),
            "QP_infeasible_while_Omega_before_true":sum(r["Omega_before"]=="True" for r in failed),
            "fallback_mode_counts":{mode:sum(r["safety_mode"]==mode for r in failed) for mode in sorted({r["safety_mode"] for r in failed})}})
    joint = results["joint_alpha"]
    safe_positive = results["safe_positive_alpha"]
    replay_delta = next(r["delta_mean_contribution_vs_baseline"] for r in increment
                       if r["label"]=="episode5_actor" and r["component"]=="actual_training_replay_reward_same_transition")
    # Cases are diagnostic evidence patterns, not statistically identified
    # causes. Never automatically apply a successful diagnostic alpha.
    primary = "A" if joint else "B" if safe_positive else "B/E requires inspection of frontier"
    findings = {"primary_frontier_case":primary,"safe_positive_alpha":safe_positive,"joint_alpha":joint,
        "audit_representative_alpha":results["selected_for_audit_only"]["frontier_representative"],
        "actor_replay_reward_mean_change_vs_baseline":replay_delta,
        "reward_magnitude_ratios":shape_ratios,"margins":margins,"actor_action_statistics":actor_stats,
        "QP_context":details,
        "priority": "residual authority diagnostic/retraining discussion" if joint else "policy objective/optimization diagnosis before any retraining",
        "limits": "Only 3 fixed realizations, one early actor, no causal identification or Gaussian guarantee. Small absolute economic increments and worst-seed signs must be considered."}
    first = results.get("first_failure")
    if first:
        f = first["first_failure_evidence"]
        original = json.loads((Path(manifest["run_dir"]) / "experiment_manifest.json").read_text(encoding="utf-8"))
        _, model, _, _, _, _ = make_setup(manifest["steps"], original["seed"])
        x = np.array([f["state_0"],f["state_1"]])
        noise = np.array([f[f"disturbance_{i}"] for i in range(4)])
        base = np.array([f[f"baseline_input_at_same_state_{i}"] for i in range(2)])
        findings["first_failure_one_step_counterfactual"] = {
            "seed": first["seed"], "step": first["first_physical_transition_step"],
            "same_state_same_disturbance": True,
            "zero_residual_safe_input_at_this_controller_state": base.tolist(),
            "zero_residual_counterfactual_next_state": model.step(x,base,noise).tolist(),
            "actor_actual_next_state": [f["actual_next_0"],f["actual_next_1"]],
            "scope": "one-step causal action comparison only, separate from autonomous scale sweep; not a certificate"}
    save_json(out / "diagnosis_findings.json",findings)
    report = out / "zanon2019_failure_diagnosis_report.md"
    text = report.read_text(encoding="utf-8")
    text = text.split("\n## Aggregated evidence and interpretation\n")[0]
    lines = ["", "## Aggregated evidence and interpretation", "",
        f"Frontier classification: {primary}. Positive-economic, empirically safe scales: {safe_positive}; Level-C scales: {joint}.",
        f"Episode-5 replay reward mean change versus baseline: {replay_delta:.8f}. A negative value means the frozen actual replay reward does not rank this unsafe policy above baseline on these paths. Economic component magnitude alone must not be used to claim why SAC accepted it.",
        "", "### Reward components (three-seed average per step, signed)", "",
        "|component|baseline|alpha=1|frontier representative|", "|---|---|---|---|"]
    for key in ["economic_reward",*PENALTIES,"actual_training_replay_reward_same_transition"]:
        vals = [next(r["mean_signed"] for r in aggregates if r["label"]==label and r["component"]==key)
                for label in ("baseline","episode5_actor","frontier_representative")]
        lines.append(f"|{key}|{vals[0]:.8g}|{vals[1]:.8g}|{vals[2]:.8g}|")
    lines += ["", "### X2 physical margins (post-state, includes final transition)", "",
        "|label|alpha|mean|min|p1|p5|fraction<0.05|fraction<0.10|fraction<0.20|", "|---|---|---|---|---|---|---|---|---|"]
    for r in margins:
        if r["scope"]=="post_state":
            lines.append(f"|{r['label']}|{r['alpha']:g}|{r['mean_margin']:.6g}|{r['min_margin']:.6g}|{r['p1_margin']:.6g}|{r['p5_margin']:.6g}|{r['fraction_lt_0.05']:.4%}|{r['fraction_lt_0.10']:.4%}|{r['fraction_lt_0.20']:.4%}|")
    lines += ["", "### Actor signs and saturation", "", "```json",json.dumps(actor_stats,indent=2),"```",
        "", "### QP infeasibility context", "", "```json",json.dumps(details,indent=2),"```",
        "", "Temporal ordering of later failures is recorded per seed. A later event after the first physical failure is not automatically attributed causally to that one event; repeated fresh Gaussian exceedances and re-entry can create separate excursions.",
        "", "### Diagnostic decision", "",f"Priority: {findings['priority']}. Do not change alpha automatically. Reward magnitude imbalance can coexist with excessive authority or early actor saturation; this sweep does not isolate all causal mechanisms. A single early checkpoint cannot establish SAC convergence or policy optimality.",
        "", "No positive-X2-buffer term exists in the frozen reward. State recovery uses fixed balanced B, move penalties use final actual inputs, and physical-violation event penalties act after violating the bound. RPI remains audited but is not a stochastic replay penalty."]
    base_shape = shape_ratios["baseline"]
    actor_shape = shape_ratios["episode5_actor"]
    lines += ["", "### Same-state first-failure counterfactual", "",
        "```json",json.dumps(findings.get("first_failure_one_step_counterfactual"),indent=2),"```",
        "", "### Reward imbalance versus optimization", "",
        f"Recovery penalty is {100*base_shape['state_recovery_penalty']:.5f}% of mean absolute economic reward for baseline and {100*actor_shape['state_recovery_penalty']:.5f}% for alpha=1. P100 move penalty is not uniformly weak ({100*base_shape['p100_move_penalty']:.2f}% / {100*actor_shape['p100_move_penalty']:.2f}%). The full actor replay reward is worse than baseline, so these data do not show that the full weighted objective prefers the unsafe trajectory. There is a recovery-scale imbalance plus an early policy suboptimal under its own recorded objective; this alone does not identify a SAC implementation bug or convergence failure.",
        "", "### Action mapping semantics", "",
        "The interior-anchor command is a fraction of the current feasible polytope, not a fixed physical residual bound. Raw action signs need not equal physical residual signs because the boundary ray originates at the interior anchor before blending with the zero-action baseline. This is not evidence of a numerical mapper bug.",
        "", "### Conclusion", "",
        f"Case {primary} is a diagnostic frontier classification only. Joint scales {joint} pass the defined mean-economic and per-seed 10%-IAE/TV gates on these fixed realizations, not all Gaussian inputs. Inspect their worst-seed economics before describing a gain as stable. No alpha is applied to training and the original alpha=1 actor is not promoted as a paper candidate. Priority: {findings['priority']}. Weak recovery shaping remains a secondary audit concern, not an automatic weight change."]
    report.write_text(text+"\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:findings[k] for k in ("primary_frontier_case","safe_positive_alpha","joint_alpha","audit_representative_alpha","actor_replay_reward_mean_change_vs_baseline","priority")},indent=2))


if __name__ == "__main__":
    main()
