"""Staged development-only safe-SAC economic recovery experiment.

Run from the repository parent. Training is opt-in via `train`, never performed
by prepare/tests/report/oracle/sweep. No independent-test phase exists.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import unittest
import numpy as np
import torch

from . import zanon2019_paired_experiment as v1
from . import zanon2019_paired_v2 as v2
from .zanon2019_paired_v2_objective import (SafetyDual, EconomicReplay, paired_metrics, components, SAFETY_KEYS)
from .zanon2019_benchmark import (make_agent, sample_disturbance_path, write_csv,
    PAPER_VARIANCES_F1_X1_T1_T200, DEFAULT_DESIGN, DEFAULT_OMEGA)
from .zanon2019_training_report import trace_rows, CERTIFICATION_STATUS
from .zanon2019_economic_recovery_metrics import pair_economics, control_activity
from .zanon2019_economic_recovery_reference import reference_environment
from .train import ZeroResidualPolicy, run_episode
from .sac import set_seed

ROOT = v1.REPO/"evaporation_safe_sac/economic_recovery_v1"
PAIRS = {"strong": "primary_strong_baseline", "conservative": "sensitivity_conservative_baseline"}
SEEDS, DEV = v2.TRAIN_SEEDS, v2.DEV
DELTAS = (0., .05, .10, .15, .20)


def args_for(seed=42):
    return argparse.Namespace(root=v2.ROOT, seed=seed, steps=1000, episodes=100,
        economic_scale=200., design=DEFAULT_DESIGN, omega_vertices=DEFAULT_OMEGA)


def read(path):
    return v2.load(path)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    v1.save(path, value)


def immutable(path, value):
    if path.exists():
        raise RuntimeError(f"Immutable evidence already exists: {path}")
    save(path, value)


def code_hashes():
    return {p.name: v1.digest(p) for p in v1.REPO.glob("zanon2019_economic_recovery*.py")}


def require_dev(seeds):
    if not seeds or any(seed not in DEV for seed in seeds):
        raise ValueError("Development disturbance seeds 420000..420009 only; no independent test")


def prepare(root):
    if root.resolve() != ROOT.resolve():
        raise ValueError("Use isolated economic_recovery_v1 root; old runs cannot be overwritten")
    cfg, _, d = v2.guard(args_for())[:3]
    config = v2.locked_config(v2.ROOT)
    protocol = dict(schema="safe_sac_economic_recovery_v1", pairs=PAIRS,
        training_seeds=SEEDS, development_seeds=DEV, representative_seed=420000,
        episodes=100, steps=1000, dt_min=cfg.dt_min, economic_scale=200.,
        stochastic_residual_scale=.10, disturbance_variances=PAPER_VARIANCES_F1_X1_T1_T200,
        disturbance_choice="implementation choice for reproduction: iid Gaussian literal variance, negative F1/X1 clip",
        strong_reference=cfg.robust_economic_reference_state,
        conservative_deltas_X2=DELTAS, conservative_P2_fixed=float(cfg.robust_economic_reference_state[1]),
        conservative_selection_rule="first positive delta with all ten paired baseline minimum X2 margins increased by >1e-4; safety/reference gates pass; no economic ranking",
        margin_increment_guard=1e-4, margin_guard_meaning="observable numerical guard, not an engineering margin target",
        smoothness_calibration="same 5% lambda=0.05*mean(abs(paired economic reward))/mean(excess applied squared move); same 69 archived excitation profiles, equal 1000-step weighting",
        conservative_excitation_choice="replay the identical raw actor-action profiles at the new reference through the unchanged controller; not direct physical-q actions",
        strong_lambda_smooth=config["lambda_smooth"], smoothness_physical_scales=[100., 100.],
        instantaneous="100*(ell_base-ell_SAC)/ell_base", raw_difference="ell_SAC-ell_base",
        rolling20="trailing full 20 steps; first 19 unavailable", cumulative="100*cumsum(base-SAC)/cumsum(base)",
        win_fraction="strict count(ell_SAC<ell_base)/1000", recovery="100*(JC-JCSAC)/(JC-JS), defined only JC>JS",
        oracle_budget=512, oracle_segments=[10, 5], oracle_search_seed=190019,
        oracle_status="finite best-found estimate, NOT certified economic upper bound",
        presentation_checkpoint=dict(episode=100, training_seed=42, validation_seed=420000,
                                     status="predeclared development presentation; not final selection"),
        final_selection_rule="not locked; register after all six development training runs, before independent testing",
        phase_order=["strong metrics/tests/oracle/pilot/3seeds", "conservative baseline sweep/freeze/oracle",
                     "conservative calibration/3seeds", "development comparison"],
        frozen_geometry_sha256={str(p): v1.digest(p) for p in (DEFAULT_DESIGN, DEFAULT_OMEGA)},
        strong_config_sha256=v1.digest(v2.ROOT/"smoothness_regularization_config.json"),
        frozen_v2_sources=v2.source_hashes(), source_hashes=code_hashes(), final_test_performed=False)
    immutable(root/"protocol.json", protocol)
    for folder in ("strong", "conservative", "comparison", "figures"):
        (root/folder).mkdir(exist_ok=True)
    audit = read(v2.ROOT/"reward_scale_audit.json")
    audit.update(source=str(v2.ROOT/"reward_scale_audit.json"), source_sha256=v1.digest(v2.ROOT/"reward_scale_audit.json"),
                 pair=PAIRS["strong"], copied_read_only=True)
    immutable(root/"strong/strong_reward_scale_audit.json", audit)
    print("Prepared protocol. No training or conservative selection performed.", flush=True)


def locked(root):
    if root.resolve() != ROOT.resolve():
        raise ValueError("Preserve all existing experiments; use economic_recovery_v1")
    p = read(root/"protocol.json")
    if p["source_hashes"] != code_hashes() or p["frozen_v2_sources"] != v2.source_hashes():
        raise RuntimeError("Experiment source changed after preregistration")
    v2.locked_config(v2.ROOT)
    if p["strong_config_sha256"] != v1.digest(v2.ROOT/"smoothness_regularization_config.json"):
        raise RuntimeError("Strong calibration changed")
    for path, sha in p["frozen_geometry_sha256"].items():
        if v1.digest(path) != sha:
            raise RuntimeError("Frozen safety geometry changed")
    require_dev(p["development_seeds"])
    return p


def run_dir(root, pair, seed):
    directory = root/pair/f"seed_{seed}"
    if pair == "strong" and not (directory/"run_summary.json").exists():
        return v2.ROOT/f"seed_{seed}"
    return directory


def completed(root, pair, seed):
    directory = run_dir(root, pair, seed)
    if not (directory/"run_summary.json").exists():
        return False
    return v2.pilot_gate(read(directory/"run_summary.json"), read(directory/"checkpoints.json"))


def require_strong_complete(root):
    missing = [s for s in SEEDS if not completed(root, "strong", s)]
    if missing:
        raise RuntimeError(f"Conservative phase blocked: Strong seeds not complete/passed: {missing}")


def environment(root, pair, seed=42):
    strong = v2.guard(args_for(seed))
    if pair == "strong":
        return strong
    selection = read(root/"conservative/conservative_baseline_selection.json")
    if not selection["selected_before_SAC_training"] or selection["selected_delta_X2"] is None:
        raise RuntimeError("No baseline-only frozen Conservative reference")
    env, check = reference_environment(strong, selection["selected_delta_X2"])
    if not check["passed"]:
        raise RuntimeError(f"Frozen Conservative design no longer passes: {check['limiting_gates']}")
    return env


def full_metrics(base, records, env, weight, dual):
    metric, raw = paired_metrics(records, base, env, weight, dual)
    economic, series = pair_economics(base, records, env[0].robust_economic_reference_state)
    if not np.isclose(economic["economic_improvement_pct"], metric["economic_improvement_pct"], atol=1e-10):
        raise RuntimeError("Economic cumulative endpoint inconsistent with paired cost")
    metric.update(economic)
    lower, upper = env[1].physical_input(env[2].robust_input_lower), env[1].physical_input(env[2].robust_input_upper)
    metric.update(control_activity(records, lower, upper))
    metric.update({"baseline_"+k: v for k, v in control_activity(base, lower, upper).items()})
    return metric, raw, series


def trace(base, records, series):
    rows = trace_rows(base, records, .10)
    for k, row in enumerate(rows):
        for name in ("delta_ell", "instantaneous_economic_advantage_pct", "G_roll20", "G_cum"):
            value = series[name][k]
            row[name] = None if value is None else float(value)
    return rows


def baseline_sweep(root):
    p = locked(root)
    require_strong_complete(root)
    target = root/"conservative"
    if (target/"conservative_baseline_selection.json").exists():
        raise RuntimeError("Conservative baseline selection is already immutable")
    strong = environment(root, "strong")
    original = v1.paths_and_baselines(strong, DEV, target/"strong_baseline_reference")
    rows, checks, selected = [], [], None
    for delta in p["conservative_deltas_X2"]:
        env, check = reference_environment(strong, delta)
        checks.append(dict(delta_X2=delta, **check))
        save(target/"reference_checks"/f"delta_{delta:.2f}.json", check)
        if not check["passed"]:
            rows.append(dict(delta_X2=delta, reference_X2=float(env[0].robust_economic_reference_state[0]),
                             reference_passed=False, failure=";".join(check["limiting_gates"])))
            continue
        current, margins = [], []
        try:
            for seed in DEV:
                path, base, _, _ = original[seed]
                records, _, _ = v1.rollout(env, path, seed, ZeroResidualPolicy())
                metric, _, series = full_metrics(records, records, env, 0., np.zeros(5))
                strong_min = min(r["state"][0] for r in base)
                # Include the terminal transition through audited metrics, for both pairs.
                strong_metric, _, _ = full_metrics(base, base, strong, 0., np.zeros(5))
                margin_delta = metric["minimum_X2_margin"]-strong_metric["minimum_X2_margin"]
                margins.append(margin_delta)
                current.append(dict(delta_X2=delta, reference_passed=True, validation_seed=seed,
                                    margin_increase=margin_delta, **metric))
                write_csv(target/"sweep_trajectories"/f"delta_{delta:.2f}"/f"seed_{seed}.csv", trace(records, records, series))
        except Exception as exc:
            if getattr(exc, "evidence", None):
                write_csv(target/"sweep_failures"/f"delta_{delta:.2f}_seed_{seed}.csv", exc.evidence)
            rows.append(dict(delta_X2=delta, reference_passed=True, failure=str(exc), empirical_safe=False))
            continue
        rows.extend(current)
        if selected is None and delta > 0 and min(margins) > p["margin_increment_guard"]:
            selected = dict(delta=delta, check=check, baseline_only_metrics=current)
    write_csv(target/"baseline_conservatism_sweep.csv", rows)
    immutable(target/"conservative_baseline_selection.json", dict(
        selected_before_SAC_training=True, candidates=p["conservative_deltas_X2"],
        selection_rule=p["conservative_selection_rule"], selected_delta_X2=selected["delta"] if selected else None,
        selected_reference=selected["check"]["reference"] if selected else None,
        baseline_only_metrics=selected["baseline_only_metrics"] if selected else [],
        reference_checks=checks, protocol_sha256=v1.digest(root/"protocol.json"),
        sweep_sha256=v1.digest(target/"baseline_conservatism_sweep.csv"),
        status="selected" if selected else "no_candidate_passed_do_not_expand_grid"))
    print("Conservative selection:", selected["check"]["reference"] if selected else "none", flush=True)


def conservative_audit(root):
    locked(root); require_strong_complete(root)
    env = environment(root, "conservative")
    target = root/"conservative/conservative_reward_scale_audit.json"
    if target.exists():
        raise RuntimeError("Calibration is immutable")
    sources, arrays, metrics = [], [], []
    for role, trainseed, seed, path in v2.calibration_sources():
        require_dev([seed])
        _, archived = v2.archived_pair(path)
        disturbance = np.array([r["disturbance"] for r in archived])
        base, _, _ = v1.rollout(env, disturbance, seed, ZeroResidualPolicy())
        policy = v1.SegmentPolicy([r["raw_action"] for r in archived], 1)
        records, _, _ = v1.rollout(env, disturbance, seed, policy)
        metric, raw, series = full_metrics(base, records, env, 0., np.zeros(5))
        arrays.append(raw)
        name = f"{role}_{trainseed}_seed_{seed}.csv"
        output = root/"conservative/calibration_trajectories"/name
        write_csv(output, trace(base, records, series))
        sources.append(dict(role=role, validation_seed=seed, raw_action_source=str(path),
                            source_sha256=v1.digest(path), calibrated_trajectory=str(output), sha256=v1.digest(output)))
        metrics.append(metric)
    raw = np.concatenate(arrays)
    stats = v2.calibrate(raw[:, 0], raw[:, 1])
    immutable(target, dict(**stats, target_fraction=.05, economic_scale=200., physical_scales=[100., 100.],
        frozen=True, pair=PAIRS["conservative"], calibration_files=sources,
        selection_sha256=v1.digest(root/"conservative/conservative_baseline_selection.json"),
        rule="same 69 excitation profiles, equal step pooling and deterministic 5% rule as Strong",
        implementation_choice="same archived raw-action excitation replayed through own reference-dependent controller",
        certification_status=CERTIFICATION_STATUS, final_test_performed=False))


def weight(root, pair):
    audit = read(root/pair/f"{pair}_reward_scale_audit.json")
    if pair == "conservative":
        if audit["selection_sha256"] != v1.digest(root/"conservative/conservative_baseline_selection.json"):
            raise RuntimeError("Conservative reference changed after calibration")
        for source in audit["calibration_files"]:
            if v1.digest(source["raw_action_source"]) != source["source_sha256"] or \
               v1.digest(source["calibrated_trajectory"]) != source["sha256"]:
                raise RuntimeError("Calibration evidence changed")
    return float(audit["lambda_smooth"])


def protocol_key(env, pair):
    return dict(pair=PAIRS[pair], reference=env[0].robust_economic_reference_state.tolist(),
                seeds=list(DEV), steps=1000, alpha=.10, economic_scale=200.)


def oracle(root, pair):
    p = locked(root)
    if pair == "conservative":
        require_strong_complete(root)
    env = environment(root, pair)
    folder = root/pair/"oracle"
    v1.fresh(folder)
    cache = v1.paths_and_baselines(env, DEV, folder)
    rng = np.random.default_rng(p["oracle_search_seed"])
    summaries, allrows, elites = [], [], []
    # Frozen development policy trajectories seed the search; a new search must
    # never claim less available headroom than a feasible known learned policy.
    known = []
    for training_seed in SEEDS:
        directory = run_dir(root, "strong", training_seed)
        for episode in (40, 100):
            if (directory/"models"/f"episode_{episode:04d}_actor.pth").exists():
                known.append((training_seed, episode, directory))
    budget = p["oracle_budget"]
    for index in range(budget):
        actions = None
        if index < len(known):
            seed, episode, directory = known[index]
            policy = make_agent(env[0], "cpu")
            from .zanon2019_benchmark import load_actor
            load_actor(policy, directory/"models"/f"episode_{episode:04d}_actor.pth")
            role = f"frozen_strong_seed{seed}_ep{episode}"
            segment = 0
        else:
            segment = p["oracle_segments"][0 if index < budget//2 else 1]
            count = 1000//segment
            if index < len(known)+25:
                grid = np.linspace(-1, 1, 5)
                number = index-len(known)
                actions = np.tile([grid[number//5], grid[number % 5]], (count, 1))
                role = "constant_grid"
            elif elites and index % 3 != 0:
                elite = elites[int(rng.integers(len(elites)))]["actions"]
                if elite is None:
                    elite = np.zeros((count, 2))
                at = np.minimum(np.arange(count)*len(elite)//count, len(elite)-1)
                actions = np.clip(elite[at]+rng.normal(0, .12 if index % 2 else .04, (count, 2)), -1, 1)
                role = "Pareto_local_mutation"
            else:
                # Coherent profiles plus innovations; not 300 independent online inputs.
                actions = np.clip(rng.uniform(-.6, .6, (1, 2))+rng.normal(0, .2, (count, 2)), -1, 1)
                role = "piecewise_exploration"
        current, failure = [], None
        for seed, (path, base, _, _) in cache.items():
            try:
                candidate = policy if actions is None else v1.SegmentPolicy(actions, segment)
                with torch.no_grad():
                    records, _, _ = v1.rollout(env, path, seed, candidate)
                metric, _, series = full_metrics(base, records, env, 0., np.zeros(5))
                current.append(dict(candidate=index, validation_seed=seed, **metric))
                # Save every safe witness, not only attractive cost endpoints.
                write_csv(folder/"witness_trajectories"/f"candidate_{index:04d}_seed_{seed}.csv", trace(base, records, series))
            except Exception as exc:
                failure = str(exc)
                if getattr(exc, "evidence", None):
                    write_csv(folder/"failures"/f"candidate_{index:04d}_seed_{seed}.csv", exc.evidence)
                break
        safe = len(current) == len(DEV) and failure is None
        row = dict(candidate=index, role=role, segment_steps=segment, safety_passed=safe, failure=failure,
            economic_improvement_pct=float(np.mean([r["economic_improvement_pct"] for r in current])) if safe else None,
            worst_economic_improvement_pct=min(r["economic_improvement_pct"] for r in current) if safe else None,
            total_control_activity=float(np.mean([r["total_control_activity"] for r in current])) if safe else None,
            control_joint_witness=bool(safe and all(r["economic_improvement_pct"] > 0 and
                r["P100_TV_ratio"] <= 1 and r["F200_TV_ratio"] <= 1 for r in current)),
            X2_std_ratio=float(np.mean([r["X2_std_ratio"] for r in current])) if safe else None,
            minimum_X2_margin=min(r["minimum_X2_margin"] for r in current) if safe else None)
        summaries.append(row); allrows.extend(current)
        write_csv(folder/"all_candidates.csv", summaries)
        write_csv(folder/"per_realization.csv", allrows)
        if actions is not None:
            np.save(folder/f"candidate_{index:04d}_actions.npy", actions)
        if safe:
            elites.append(dict(**row, actions=actions))
            def vector(x):
                return np.array([-x["economic_improvement_pct"], x["total_control_activity"], x["X2_std_ratio"]])
            elites = [r for r in elites if not any(np.all(vector(t) <= vector(r)) and
                      np.any(vector(t) < vector(r)) for t in elites)]
        print(f"{pair} oracle {index+1}/{budget} safe={safe} gain={row['economic_improvement_pct']}", flush=True)
    safe = [r for r in summaries if r["safety_passed"]]
    best = max(safe, key=lambda r: r["economic_improvement_pct"]) if safe else None
    write_csv(folder/"pareto_front.csv", [{k: v for k, v in r.items() if k != "actions"} for r in elites])
    immutable(root/pair/f"oracle_safe_economic_upper_bound_{pair}.json", dict(
        status="best_found_not_certified_upper_bound", label="finite safe actor-action economic headroom search",
        best_candidate=best, best_found_improvement_pct=best["economic_improvement_pct"] if best else None,
        completed_candidates=budget, safe_candidates=len(safe), protocol=protocol_key(env, pair),
        control_joint_witness_count=sum(r["control_joint_witness"] for r in summaries),
        privileged_offline_search=True, not_deployable_controller=True, final_test_performed=False,
        candidates_sha256=v1.digest(folder/"all_candidates.csv"), pareto_sha256=v1.digest(folder/"pareto_front.csv")))


def require_training(root, pair, seed):
    locked(root)
    receipt = read(root/"tests_receipt.json")
    if not receipt["passed"] or receipt["source_hashes"] != code_hashes():
        raise RuntimeError("Run all recovery tests before training")
    if pair == "conservative":
        require_strong_complete(root)
    for prior in SEEDS[:SEEDS.index(seed)]:
        if not completed(root, pair, prior):
            raise RuntimeError(f"Prior {pair} seed {prior} has not passed")
    if completed(root, pair, seed):
        raise RuntimeError("Completed seed preserved; do not retrain/overwrite")
    oracle_path = root/pair/f"oracle_safe_economic_upper_bound_{pair}.json"
    data = read(oracle_path)
    if data["completed_candidates"] != read(root/"protocol.json")["oracle_budget"] or not data["best_candidate"] \
            or data["best_found_improvement_pct"] <= 0:
        raise RuntimeError("Complete comparable safe positive-headroom oracle required")
    if data["candidates_sha256"] != v1.digest(root/pair/"oracle/all_candidates.csv"):
        raise RuntimeError("Oracle evidence changed")
    if data["protocol"] != protocol_key(environment(root, pair, seed), pair):
        raise RuntimeError("Oracle protocol does not match training validation")


def train(root, pair, seed, device):
    """Same SAC/run_episode/component replay as v2, only baseline reference differs."""
    require_training(root, pair, seed)
    env = environment(root, pair, seed)
    cfg, model, design, omega, domain = env[:5]
    cfg.episodes = 100
    output = root/pair/f"seed_{seed}"
    v1.fresh(output)
    set_seed(seed)
    agent = make_agent(cfg, device); agent.zero_initialize_residual_mean()
    dual = SafetyDual.create()
    replay = EconomicReplay(23, 2, cfg.replay_capacity, agent.device, dual, weight(root, pair))
    model_dir = output/"models"; model_dir.mkdir()
    selection = root/"conservative/conservative_baseline_selection.json"
    selection_hash = v1.digest(selection) if pair == "conservative" else None
    calibration_hash = v1.digest(root/pair/f"{pair}_reward_scale_audit.json")
    save(output/"experiment_manifest.json", dict(pair=PAIRS[pair], training_seed=seed,
        steps=1000, episodes=100, eval_every=5, protocol_sha256=v1.digest(root/"protocol.json"),
        calibration_sha256=calibration_hash, selection_sha256=selection_hash,
        reference=cfg.robust_economic_reference_state, frozen_SAC_config=vars(agent.cfg),
        frozen_experiment_config=vars(cfg), validation_seeds=DEV, final_selection_rule="not locked",
        disturbance_seed_rule="seed*1000000+episode; local action rng=path_seed+77", final_test_performed=False))
    training, fixed, checkpoints, cache = [], [], [], {}
    global_step, max_collapse = 0, 0.

    def persist(state, failure=None):
        write_csv(output/"training_log.csv", training)
        write_csv(output/"fixed_evaluation.csv", fixed)
        write_csv(output/"checkpoint_performance.csv", checkpoints)
        save(output/"checkpoints.json", checkpoints)
        summary = dict(run_state=state, completed_episodes=len(training), global_step=global_step,
            max_action_collapse_fraction=max_collapse, replay_decomposition_checked=True,
            no_nonfinite=failure is None, failure=failure, training_performed=True, final_test_performed=False,
            certification_status=CERTIFICATION_STATUS)
        summary["continue_to_next_seed"] = v2.pilot_gate(summary, checkpoints)
        save(output/"run_summary.json", summary)

    def evaluate(episode):
        nonlocal max_collapse
        actor_path = model_dir/f"episode_{episode:04d}_actor.pth"
        agent.save_actor(actor_path)
        latest = []
        with torch.no_grad():
            for validation_seed, (path, base, _, _) in cache.items():
                print(f"{pair} fixed_eval episode={episode} seed={validation_seed}", flush=True)
                records, _, _ = v1.rollout(env, path, validation_seed, agent)
                metric, raw, series = full_metrics(base, records, env, replay.weight, dual.values)
                latest.append(dict(episode=episode, global_step=global_step, validation_seed=validation_seed, **metric))
                rows = trace(base, records, series)
                for r, c in zip(rows, raw):
                    r.update(economic_reward_raw=float(c[0]), smoothness_penalty_raw=float(c[1]),
                             smoothness_contribution=-replay.weight*float(c[1]))
                write_csv(output/"evaluation_trajectories"/f"episode_{episode:04d}"/f"seed_{validation_seed}.csv", rows)
        fixed.extend(latest)
        status = v2.assessment(latest, episode, global_step, cfg.warmup_steps)
        status.update(checkpoint=str(actor_path), metric_source="fixed_paired_validation", pair=PAIRS[pair])
        checkpoints.append(status)
        max_collapse = max(max_collapse, *(r["action_collapse_fraction"] for r in latest))
        print("fixed_eval", json.dumps(status, allow_nan=False), flush=True); persist("running")
        late = [s for s in checkpoints if s["post_warmup"]][-5:]
        if len(late) == 5 and all(s.get("regularizer_fraction") is not None and
                s["regularizer_fraction"] > .10 and s["mean_economic_improvement_pct"] <= 0 for s in late):
            raise RuntimeError("Regularizer scale mismatch; stop, never retune mid-run")

    try:
        cache = v1.paths_and_baselines(env, DEV, output)
        evaluate(0)
        for episode in range(1, 101):
            locked(root)
            if v1.digest(root/pair/f"{pair}_reward_scale_audit.json") != calibration_hash or \
                (selection_hash is not None and v1.digest(selection) != selection_hash):
                raise RuntimeError("Frozen reference/reward changed during training")
            path_seed = seed*1000000+episode
            path, _ = sample_disturbance_path(cfg, "zanon2019_stochastic", path_seed, 1000,
                                             PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
            base, _, _ = v1.rollout(env, path, path_seed, ZeroResidualPolicy())
            ctrl = v1.PairedEvidenceController(cfg, model, design, omega, domain, path)
            replay.bind(ctrl, base); dual_used = dual.values.copy()
            stat, records, global_step = run_episode(cfg, model, ctrl, agent, replay,
                np.random.default_rng(path_seed+77), training=True, global_step=global_step,
                disturbance_trajectory=path, initial_state_override=cfg.robust_economic_reference_state)
            metric, raw, _ = full_metrics(base, records, env, replay.weight, dual_used)
            if not np.allclose(replay.episode_components, raw, rtol=0, atol=1e-8):
                raise RuntimeError("Replay decomposition mismatch")
            if not all(torch.isfinite(p).all().item() for net in (agent.actor, agent.q1, agent.q2) for p in net.parameters()) \
                    or not np.isfinite(agent.alpha) or any(not np.isfinite(stat[k]) for k in
                        ("actor_loss", "q_loss", "entropy") if replay.size >= cfg.batch_size):
                raise RuntimeError("Nonfinite SAC parameter/loss/entropy")
            dual.update(raw[:, 2:].sum(0)/1000.)
            row = dict(episode=episode, global_step=global_step, disturbance_seed=path_seed, **metric,
                       episode_return=metric["total_training_reward"], legacy_return_diagnostic=float(stat["return"]),
                       **{f"lambda_{k}": float(v) for k, v in zip(SAFETY_KEYS, dual.values)})
            for key in ("actor_loss", "q_loss", "entropy"):
                if key in stat and np.isfinite(stat[key]): row[key] = float(stat[key])
            training.append(row); max_collapse = max(max_collapse, metric["action_collapse_fraction"])
            persist("running"); print(f"{pair} episode {episode}/100 gain={metric['economic_improvement_pct']:.7f}%", flush=True)
            if episode % 5 == 0: evaluate(episode)
            agent.save_checkpoint(model_dir/"last_checkpoint.pth")
        persist("completed")
    except Exception as exc:
        if "ctrl" in locals(): write_csv(output/"failure_trajectory.csv", ctrl.evidence)
        if getattr(exc, "evidence", None): write_csv(output/"evaluation_failure_trajectory.csv", exc.evidence)
        persist("aborted", dict(reason=str(exc), attempted_global_step=global_step)); raise


def tests(root):
    # Reuse complete v2 suite without modifying its receipt or source hashes.
    suite = unittest.TestSuite(); mains = []
    for path in sorted(v1.REPO.glob("*tests.py")):
        module = importlib.import_module("evaporation."+path.stem)
        loaded = unittest.defaultTestLoader.loadTestsFromModule(module); suite.addTests(loaded)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and function.__module__ == module.__name__ and not inspect.signature(function).parameters:
                suite.addTest(unittest.FunctionTestCase(function))
        if not loaded.countTestCases() and hasattr(module, "main"): mains.append(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    fallback = None
    if result.wasSuccessful() and len(result.skipped) == 1 and "CasADi" in result.skipped[0][1]:
        exe = v1.REPO/".venv-ecc2019/Scripts/python.exe"
        if exe.exists():
            process = subprocess.run([str(exe), "-m", "unittest", "evaporation.ecc2019_reproduction_tests.SolverTests", "-v"],
                                     cwd=v1.REPO.parent, capture_output=True, text=True)
            print(process.stdout, process.stderr, flush=True); fallback = process.returncode == 0
    passed = result.wasSuccessful() and (not result.skipped or fallback is True)
    integrations = []
    if passed:
        for module in [importlib.import_module("evaporation.test"), *mains]:
            module.main(); integrations.append(module.__name__)
    save(root/"tests_receipt.json", dict(passed=passed, unit_tests=result.testsRun, optional_CasADi_passed=fallback,
        integration_suites=integrations, source_hashes=code_hashes(), training_performed=False))
    if not passed: raise RuntimeError("Test failure blocks training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "tests", "report", "freeze-figure", "oracle", "sweep", "audit", "train", "status"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--pair", choices=tuple(PAIRS), default="strong")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.root.resolve() != ROOT.resolve(): raise ValueError("Isolated economic_recovery_v1 root required")
    if args.phase == "prepare": prepare(args.root)
    elif args.phase == "tests": tests(args.root)
    elif args.phase in ("report", "freeze-figure"):
        from .zanon2019_economic_recovery_report import report, freeze_figure
        locked(args.root)
        if args.phase == "report": report(args.root)
        else: freeze_figure(args.root, args.pair, args.device)
    elif args.phase == "oracle": oracle(args.root, args.pair)
    elif args.phase == "sweep": baseline_sweep(args.root)
    elif args.phase == "audit": conservative_audit(args.root)
    elif args.phase == "train": train(args.root, args.pair, args.seed, args.device)
    else:
        locked(args.root)
        print(json.dumps({pair: {s: completed(args.root, pair, s) for s in SEEDS} for pair in PAIRS}, indent=2))


if __name__ == "__main__": main()
