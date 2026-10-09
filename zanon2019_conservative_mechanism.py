"""Baseline-only Conservative mechanism stage, with fail-closed oracle gates.

Latest user authorization prioritizes this stage before Strong seeds 2027 and
314159. The earlier protocol, Strong runs and all controller sources remain
read-only. No training is invoked by any phase in this module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from . import zanon2019_paired_experiment as v1
from . import zanon2019_paired_v2 as v2
from . import zanon2019_economic_recovery as recovery_exp
from .zanon2019_economic_recovery_reference import reference_environment
from .zanon2019_economic_recovery_metrics import pair_economics, control_activity
from .zanon2019_paired_v2_objective import paired_metrics, audited_metrics
from .zanon2019_benchmark import sample_disturbance_path, write_csv, PAPER_VARIANCES_F1_X1_T1_T200
from .zanon2019_training_report import CERTIFICATION_STATUS
from .train import ZeroResidualPolicy, run_episode

ROOT = recovery_exp.ROOT/"conservative_mechanism_stage2"
DEV, DELTAS = v2.DEV, recovery_exp.DELTAS
SAFETY = v1.SAFETY


def source_hashes():
    return {p.name: v1.digest(p) for p in v1.REPO.glob("zanon2019_conservative_mechanism*.py")}


def save(path, value):
    recovery_exp.save(path, value)


def load(path):
    return v2.load(path)


def prepare(root):
    if root.resolve() != ROOT.resolve():
        raise ValueError("Use isolated conservative_mechanism_stage2; preserve historical evidence")
    old = recovery_exp.locked(recovery_exp.ROOT)
    protocol = dict(schema="conservative_mechanism_stage2", original_protocol_sha256=v1.digest(recovery_exp.ROOT/"protocol.json"),
        authorization="latest human request explicitly prioritizes Conservative mechanism; Strong three-seed completion is deferred, not fabricated",
        phase_order=["baseline sweep", "baseline-only selection freeze", "penalty gate", "finite actor-action oracle", "training eligibility report"],
        deltas_X2=list(DELTAS), P2_reference_fixed=old["conservative_P2_fixed"],
        common_initial_state=old["strong_reference"], initial_condition_choice="all sweep candidates and own paired oracle use the same Strong B initial state",
        reference_choice="inherit preregistered fixed P2; exact nonlinear steady_input; no P2 optimization, K redesign or tightening relaxation",
        validation_seeds=list(DEV), steps=1000, episodes_if_later_authorized=100,
        dt_min=old["dt_min"], stochastic_residual_scale=old["stochastic_residual_scale"],
        disturbance_variances=old["disturbance_variances"], disturbance_choice=old["disturbance_choice"],
        margin_increase_tolerance=old["margin_increment_guard"],
        selection_rule="first positive delta passing all reference/safety/numerical gates and increasing each of ten minimum X2 margins by >1e-4; aggregate minimum must also increase; no SAC or economic ranking",
        penalty_numerical_guard_abs=2*1000*1e-7,
        penalty_guard_source="two horizon sums of the existing audited_metrics absolute stage-cost check tolerance 1e-7; not a desired economic percent target",
        oracle_budget=old["oracle_budget"], oracle_search_seed=old["oracle_search_seed"], oracle_segments=old["oracle_segments"],
        oracle_label="best economic improvement found by current finite search; never mathematical upper bound",
        economic_scale=200., smoothness_scales=[100., 100.], target_regularizer_fraction=.05,
        economic_definition=old["instantaneous"], cumulative_definition=old["cumulative"], rolling20=old["rolling20"],
        physical_cost_unchanged=True, old_protocol_unchanged=True, final_test_performed=False,
        training_performed=False, source_hashes=source_hashes())
    recovery_exp.immutable(root/"protocol.json", protocol)
    print("Stage2 preregistered: no safety/reward/controller changes, no training.", flush=True)


def locked(root):
    if root.resolve() != ROOT.resolve(): raise ValueError("Wrong evidence root")
    protocol = load(root/"protocol.json")
    recovery_exp.locked(recovery_exp.ROOT)
    if protocol["original_protocol_sha256"] != v1.digest(recovery_exp.ROOT/"protocol.json") or \
        protocol["source_hashes"] != source_hashes():
        raise RuntimeError("Locked original protocol or stage implementation changed")
    if protocol["deltas_X2"] != list(DELTAS) or protocol["validation_seeds"] != list(DEV) \
        or protocol["margin_increase_tolerance"] != 1e-4:
        raise RuntimeError("Predefined grid, validation set or margin rule changed")
    recovery_exp.require_dev(protocol["validation_seeds"])
    return protocol


def rollout(env, path, seed, policy, initial):
    """Identical existing online controller; explicit common physical reset."""
    cfg, model, d, omega, domain = env[:5]
    ctrl = v1.PairedEvidenceController(cfg, model, d, omega, domain, path)
    try:
        with torch.no_grad():
            _, records, _ = run_episode(cfg, model, ctrl, policy, None,
                np.random.default_rng(seed+900000), training=False, global_step=0,
                disturbance_trajectory=path, initial_state_override=np.asarray(initial, float))
        metric = audited_metrics(records, env)
        if len(records) != len(path) or any(metric[k] for k in SAFETY):
            raise RuntimeError("Incomplete/unsafe conservative mechanism rollout")
        return records, metric
    except Exception as exc:
        exc.evidence = ctrl.evidence
        raise


def per_path_metrics(env, records, initial):
    metric, _ = paired_metrics(records, records, env, 0., np.zeros(5))
    economic, _ = pair_economics(records, records, initial)
    metric.update(economic)
    lower, upper = [env[1].physical_input(v) for v in (env[2].robust_input_lower, env[2].robust_input_upper)]
    metric.update(control_activity(records, lower, upper))
    return metric


def summarize(rows):
    if len(rows) != 10 or sorted(r["validation_seed"] for r in rows) != list(DEV):
        raise ValueError("Exactly ten complete predefined validation paths required")
    metric_keys = ("J_econ", "mean_X2", "std_X2", "centered_X2_MAE", "X2_IAE", "X2_ISE", "P2_IAE", "P2_ISE",
                   "P100_TV", "P100_delta_RMS", "F200_TV", "F200_delta_RMS", "W_exceedance_rate")
    result = {"mean_"+k: float(np.mean([r[k] for r in rows])) for k in metric_keys}
    result.update(minimum_X2=min(r["minimum_X2"] for r in rows),
                  minimum_X2_margin=min(r["minimum_X2_margin"] for r in rows),
                  validation_path_count=len(rows))
    for key in SAFETY: result[key] = int(sum(r[key] for r in rows))
    result["empirical_safety_passed"] = all(result[k] == 0 for k in SAFETY)
    return result


def margin_gate(candidate, strong, tolerance):
    if len(candidate) != 10 or len(strong) != 10: return False
    s = {r["validation_seed"]: r for r in strong}
    if sorted(s) != list(DEV) or sorted(r["validation_seed"] for r in candidate) != list(DEV): return False
    if any(any(r[k] != 0 for k in SAFETY) for r in candidate): return False
    differences = [r["minimum_X2_margin"]-s[r["validation_seed"]]["minimum_X2_margin"] for r in candidate]
    return bool(min(differences) > tolerance and
                min(r["minimum_X2_margin"] for r in candidate) > min(r["minimum_X2_margin"] for r in strong)+tolerance)


def baseline_sweep(root):
    protocol = locked(root)
    if (root/"conservative_baseline_selection.json").exists():
        raise RuntimeError("Preserve frozen selection, never overwrite or reselect after SAC")
    strong = v2.guard(recovery_exp.args_for())
    initial = np.asarray(protocol["common_initial_state"])
    trajectories, summaries, per_seed, checks = {}, [], [], []
    selected = None
    for delta in protocol["deltas_X2"]:
        env, check = reference_environment(strong, delta)
        checks.append(dict(delta_X2=delta, **check))
        save(root/"reference_checks"/f"delta_{delta:.2f}.json", check)
        xref, uref = check["reference"], check["steady_input"]
        lo, hi = [env[1].physical_input(v) for v in (env[2].u_lower_tight, env[2].u_upper_tight)]
        candidate = dict(delta_X2=delta, X2_ref=float(xref[0]), P2_ref=float(xref[1]),
            P100_ref=float(uref[0]), F200_ref=float(uref[1]),
            initial_X2=float(initial[0]), initial_P2=float(initial[1]), reference_checks_passed=check["passed"],
            tightened_F200_lower=float(lo[1]), tightened_F200_upper=float(hi[1]),
            F200_tightened_upper_excess=float(max(0., uref[1]-hi[1])),
            steady_stage_cost=float(env[1].economic_cost(xref, uref, env[0].disturbance_nominal)),
            baseline_evaluation_status="not_run_reference_gate_failed", validation_path_count=0,
            mean_J_econ=None, conservatism_penalty_abs=None, conservatism_penalty_pct=None,
            minimum_X2_margin=None, minimum_X2=None, safety_margin_selection_passed=False,
            limiting_gates=";".join(check["limiting_gates"]))
        # Never simulate an invalid reference and report its projected steady
        # controller as a valid baseline for the requested reference.
        if not check["passed"]:
            summaries.append(candidate)
            print(f"delta={delta:.2f} rejected: {check['limiting_gates']}; u_ref={uref}", flush=True)
            continue
        current = []
        try:
            for seed in DEV:
                if delta == 0.:
                    path, _ = sample_disturbance_path(env[0], "zanon2019_stochastic", seed, 1000,
                        PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
                else: path = trajectories[0.][seed][0]
                print(f"baseline delta={delta:.2f} seed={seed}", flush=True)
                records, _ = rollout(env, path, seed, ZeroResidualPolicy(), initial)
                metric = per_path_metrics(env, records, initial)
                current.append(dict(delta_X2=delta, validation_seed=seed, X2_ref=float(xref[0]), P2_ref=float(xref[1]),
                                    P100_ref=float(uref[0]), F200_ref=float(uref[1]), **metric))
                trajectories.setdefault(delta, {})[seed] = (path, records)
                _, series = pair_economics(records, records, initial)
                output = root/"baseline_trajectories"/f"delta_{delta:.2f}"/f"seed_{seed}.csv"
                write_csv(output, recovery_exp.trace(records, records, series))
        except Exception as exc:
            if getattr(exc, "evidence", None): write_csv(root/"baseline_failures"/f"delta_{delta:.2f}_seed_{seed}.csv", exc.evidence)
            candidate.update(baseline_evaluation_status="aborted_safety_or_numerical_failure", failure=str(exc))
            summaries.append(candidate)
            if delta == 0.: raise
            continue
        stats = summarize(current)
        candidate.update(stats, baseline_evaluation_status="completed", empirical_safety_passed=True)
        if delta == 0.:
            baseline = current
            baseline_cost = stats["mean_J_econ"]
        else:
            candidate["safety_margin_selection_passed"] = margin_gate(current, baseline, protocol["margin_increase_tolerance"])
        candidate["conservatism_penalty_abs"] = stats["mean_J_econ"]-baseline_cost
        candidate["conservatism_penalty_pct"] = 100.*candidate["conservatism_penalty_abs"]/baseline_cost
        for r in current:
            paired = next(s for s in baseline if s["validation_seed"] == r["validation_seed"])
            r.update(economic_cost_increase_vs_Strong_pct=100.*(r["J_econ"]-paired["J_econ"])/paired["J_econ"],
                     minimum_X2_margin_increase_vs_Strong=r["minimum_X2_margin"]-paired["minimum_X2_margin"])
        per_seed.extend(current); summaries.append(candidate)
        if selected is None and delta > 0 and candidate["safety_margin_selection_passed"]:
            selected = candidate
        # Scan the full fixed grid for transparent diagnostics, but selection
        # remains the FIRST passing level. No replacement by farther levels.
    write_csv(root/"baseline_conservatism_sweep.csv", summaries)
    write_csv(root/"baseline_conservatism_sweep_per_seed.csv", per_seed)
    save(root/"baseline_conservatism_sweep.json", dict(protocol_sha256=v1.digest(root/"protocol.json"),
        candidates=summaries, per_seed=per_seed, reference_checks=checks, initial_state=initial,
        aggregation="mean economic costs; minimum margin = worst across all ten complete paths including terminal state",
        rejected_candidates_retained=True, certification_status=CERTIFICATION_STATUS, training_performed=False))
    selection = dict(candidate_set=protocol["deltas_X2"], ordered_selection_rule=protocol["selection_rule"],
        strong_reference=protocol["common_initial_state"], strong_baseline_metrics=summaries[0],
        all_candidate_metrics=summaries, selected_delta_X2=selected["delta_X2"] if selected else None,
        selected_x_ref=[selected["X2_ref"], selected["P2_ref"]] if selected else None,
        selected_u_ref=[selected["P100_ref"], selected["F200_ref"]] if selected else None,
        selected_baseline_metrics=selected, reason_selected="first passing safety-margin candidate, no economic/SAC ranking" if selected else
            "no positive increment passed existing reference/tightened-input gates; do not relax bounds or change P2/grid",
        selected_before_SAC_training=True, selection_used_SAC_results=False,
        status="selected" if selected else "no_eligible_conservative_reference",
        protocol_sha256=v1.digest(root/"protocol.json"), sweep_sha256=v1.digest(root/"baseline_conservatism_sweep.json"))
    recovery_exp.immutable(root/"conservative_baseline_selection.json", selection)
    if selected:
        absolute = selected["conservatism_penalty_abs"]
        penalty = dict(conservatism_penalty_abs=absolute, conservatism_penalty_pct=selected["conservatism_penalty_pct"],
            J_strong=baseline_cost, J_conservative=selected["mean_J_econ"],
            meaningful=absolute > protocol["penalty_numerical_guard_abs"],
            numerical_guard_abs=protocol["penalty_numerical_guard_abs"],
            status="meaningful" if absolute > protocol["penalty_numerical_guard_abs"] else "no_meaningful_economic_headroom")
    else:
        penalty = dict(J_strong=baseline_cost, J_conservative=None, conservatism_penalty_abs=None,
            conservatism_penalty_pct=None, meaningful=False, status="not_applicable_no_eligible_reference")
    recovery_exp.immutable(root/"conservatism_economic_penalty.json", penalty)
    tradeoff_plot(root, summaries, selected)
    print(json.dumps(dict(selection=selection["status"], penalty=penalty)), flush=True)


def tradeoff_plot(root, rows, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, note) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    good = [r for r in rows if r["baseline_evaluation_status"] == "completed"]
    for row in good:
        label = "Strong baseline" if row["delta_X2"] == 0 else f"delta={row['delta_X2']:.2f}"
        if selected and row["delta_X2"] == selected["delta_X2"]: label += " (selected)"
        x, y = row["minimum_X2_margin"], row["conservatism_penalty_pct"]
        ax.scatter(x, y, s=65); ax.annotate(label, (x, y), xytext=(5, 7), textcoords="offset points", fontsize=9)
    if len(good) == 1:
        ax.set_xlim(good[0]["minimum_X2_margin"]-.03, good[0]["minimum_X2_margin"]+.03)
        ax.set_ylim(-.1, .1)
    ax.set(xlabel="Worst-path minimum X2 safety margin (pp)", ylabel="Economic cost increase vs Strong (%)")
    ax.grid(alpha=.2); note.axis("off")
    lines = ["All predefined candidates retained", "", "Rejected points have no simulated J/margin:"]
    for row in rows:
        if row["baseline_evaluation_status"] != "completed":
            lines.append(f"delta={row['delta_X2']:.2f}: {row['limiting_gates']}")
            lines.append(f"  F200_ref={row['F200_ref']:.4f}; tight upper={row['tightened_F200_upper']:.4f}")
    lines += ["", "No rejected point is plotted at fabricated coordinates."]
    note.text(0., 1., "\n".join(lines), va="top", fontsize=9)
    fig.suptitle("Baseline-only safety-economic trade-off" if selected else "No eligible Conservative reference under frozen tightening")
    fig.tight_layout(); fig.savefig(root/"baseline_safety_economics_tradeoff.png", dpi=180); plt.close(fig)


def oracle_precondition(selection, penalty):
    if selection["selected_delta_X2"] is None:
        return False, "no_eligible_conservative_reference"
    if not selection["selected_before_SAC_training"] or selection["selection_used_SAC_results"]:
        return False, "selection_not_frozen_baseline_only"
    if not penalty["meaningful"] or penalty["J_conservative"] <= penalty["J_strong"]:
        return False, "conservative_reference_did_not_create_meaningful_economic_penalty"
    return True, "passed"


def oracle(root):
    """No candidate control is simulated until a valid baseline is frozen."""
    protocol = locked(root)
    selection = load(root/"conservative_baseline_selection.json")
    penalty = load(root/"conservatism_economic_penalty.json")
    allowed, reason = oracle_precondition(selection, penalty)
    path = root/"oracle_conservative_summary.json"
    if path.exists(): raise RuntimeError("Preserve existing finite-search evidence")
    if not allowed:
        recovery_exp.immutable(path, dict(status="blocked_before_search", reason=reason, total_candidates=0,
            safety_passing_candidates=0, positive_economic_candidates=0,
            best_economic_improvement_found_pct=None, best_all_seed_positive_candidate=None,
            worst_validation_improvement_pct=None, economic_win_fraction=None, minimum_X2_margin=None,
            P100_TV_ratio=None, F200_TV_ratio=None, P100_RMS_ratio=None, F200_RMS_ratio=None,
            finite_search_result_not_mathematical_upper_bound=True, recovery_headroom_demonstrated=False,
            conservative_seed42_training_allowed=False, training_performed=False, final_test_performed=False,
            selection_sha256=v1.digest(root/"conservative_baseline_selection.json")))
        print(f"Conservative oracle NOT RUN: {reason}; training gate closed", flush=True)
        return
    # This branch is a finite, privileged offline action search, never training
    # or MPC. All raw actor actions pass the real interior-anchor/QP pipeline.
    env, check = reference_environment(v2.guard(recovery_exp.args_for()), selection["selected_delta_X2"])
    if not check["passed"]: raise RuntimeError("Frozen reference no longer passes")
    initial = np.asarray(protocol["common_initial_state"])
    cache = {}
    for seed in DEV:
        path_d, _ = sample_disturbance_path(env[0], "zanon2019_stochastic", seed, 1000,
                                          PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
        base, _ = rollout(env, path_d, seed, ZeroResidualPolicy(), initial)
        cache[seed] = path_d, base
    rng = np.random.default_rng(protocol["oracle_search_seed"])
    summaries, per, elites = [], [], []
    for index in range(protocol["oracle_budget"]):
        segment = protocol["oracle_segments"][0 if index < protocol["oracle_budget"]//2 else 1]
        count = 1000//segment
        if index < 25:
            grid = np.linspace(-1., 1., 5)
            actions = np.tile([grid[index//5], grid[index % 5]], (count, 1))
        elif elites and index % 3:
            previous = elites[int(rng.integers(len(elites)))]["actions"]
            at = np.minimum(np.arange(count)*len(previous)//count, len(previous)-1)
            actions = np.clip(previous[at]+rng.normal(0, .10, (count, 2)), -1, 1)
        else:
            actions = np.clip(rng.uniform(-.6, .6, (1, 2))+rng.normal(0, .2, (count, 2)), -1, 1)
        current, traces, failure = [], {}, None
        for seed, (disturbance, base) in cache.items():
            try:
                records, _ = rollout(env, disturbance, seed, v1.SegmentPolicy(actions, segment), initial)
                metric, _ = paired_metrics(records, base, env, 0., np.zeros(5))
                economic, series = pair_economics(base, records, initial); metric.update(economic)
                metric.update(instantaneous_advantage_mean=float(np.mean(series["instantaneous_economic_advantage_pct"])),
                              instantaneous_advantage_median=float(np.median(series["instantaneous_economic_advantage_pct"])))
                current.append(dict(candidate=index, validation_seed=seed, **metric))
                traces[seed] = recovery_exp.trace(base, records, series)
            except Exception as exc:
                failure = str(exc)
                if getattr(exc, "evidence", None): write_csv(root/"oracle/failures"/f"candidate_{index:04d}_seed_{seed}.csv", exc.evidence)
                break
        safe = len(current) == 10 and failure is None
        row = dict(candidate=index, segment_steps=segment, safety_passed=safe, failure=failure)
        keys = ("economic_improvement_pct", "economic_win_fraction", "mean_X2_shift_vs_baseline", "X2_std_ratio",
                "centered_X2_MAE_ratio", "P100_TV_ratio", "F200_TV_ratio", "P100_RMS_du_ratio", "F200_RMS_du_ratio", "W_exceedance_rate")
        keys += ("instantaneous_advantage_mean", "instantaneous_advantage_median")
        row.update({k: float(np.mean([r[k] for r in current])) if safe else None for k in keys})
        row.update(worst_validation_improvement_pct=min(r["economic_improvement_pct"] for r in current) if safe else None,
            minimum_X2_margin=min(r["minimum_X2_margin"] for r in current) if safe else None,
            all_seed_positive=bool(safe and min(r["economic_improvement_pct"] for r in current) > 0),
            control_joint_witness=bool(safe and all(r["economic_improvement_pct"] > 0 and r["P100_TV_ratio"] <= 1 and r["F200_TV_ratio"] <= 1 for r in current)))
        summaries.append(row); per.extend(current)
        if safe:
            value = lambda r: np.array([-r["economic_improvement_pct"], r["P100_TV_ratio"], r["F200_TV_ratio"], r["X2_std_ratio"]])
            trial = dict(**row, actions=actions); front = [*elites, trial]
            elites = [r for r in front if not any(np.all(value(t) <= value(r)) and np.any(value(t) < value(r)) for t in front)]
            if any(r["candidate"] == index for r in elites):
                for seed, rows in traces.items(): write_csv(root/"oracle/witness_trajectories"/f"candidate_{index:04d}_seed_{seed}.csv", rows)
                np.save(root/"oracle"/f"candidate_{index:04d}_actions.npy", actions)
        write_csv(root/"oracle/all_candidates.csv", summaries); write_csv(root/"oracle/per_realization.csv", per)
        print(f"oracle {index+1}/{protocol['oracle_budget']} safe={safe} gain={row['economic_improvement_pct']}", flush=True)
    safe = [r for r in summaries if r["safety_passed"]]
    positive = [r for r in safe if r["economic_improvement_pct"] > 0]
    all_positive = [r for r in positive if r["all_seed_positive"]]
    best = max(positive, key=lambda r: r["economic_improvement_pct"]) if positive else None
    best_all = max(all_positive, key=lambda r: r["economic_improvement_pct"]) if all_positive else None
    write_csv(root/"oracle/pareto_front.csv", [{k: v for k, v in r.items() if k != "actions"} for r in elites])
    report = dict(status="completed_finite_search", total_candidates=len(summaries),
        safety_passing_candidates=len(safe), positive_economic_candidates=len(positive),
        best_economic_improvement_found_pct=best["economic_improvement_pct"] if best else None,
        best_candidate=best, best_all_seed_positive_candidate=best_all,
        control_joint_witness_count=sum(r["control_joint_witness"] for r in safe),
        recovery_headroom_demonstrated=bool(positive), conservative_seed42_training_allowed=bool(positive),
        finite_search_result_not_mathematical_upper_bound=True, privileged_offline_search=True,
        training_performed=False, final_test_performed=False, selection_sha256=v1.digest(root/"conservative_baseline_selection.json"))
    for key in ("worst_validation_improvement_pct", "economic_win_fraction", "mean_X2_shift_vs_baseline", "X2_std_ratio",
                "centered_X2_MAE_ratio", "minimum_X2_margin", "P100_TV_ratio", "F200_TV_ratio",
                "P100_RMS_du_ratio", "F200_RMS_du_ratio"):
        report[key] = best[key] if best else None
    recovery_exp.immutable(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "sweep", "oracle", "status"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.phase == "prepare": prepare(args.root)
    elif args.phase == "sweep": baseline_sweep(args.root)
    elif args.phase == "oracle": oracle(args.root)
    else:
        locked(args.root)
        for name in ("conservative_baseline_selection.json", "conservatism_economic_penalty.json", "oracle_conservative_summary.json"):
            print(name, json.dumps(load(args.root/name)) if (args.root/name).exists() else "not_run", flush=True)


if __name__ == "__main__": main()
