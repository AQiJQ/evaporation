"""Isolated Candidate C: tests -> audit -> reclassify -> LOCAL train -> report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import unittest
import importlib
import inspect
import subprocess
import re
import numpy as np
import torch

from . import zanon2019_paired_experiment as v1
from .zanon2019_paired_v2_objective import (CRITERION, SAFETY_KEYS, SafetyDual, EconomicReplay,
    components, decomposition, paired_metrics, joint)
from .zanon2019_benchmark import (DEFAULT_DESIGN, DEFAULT_OMEGA, sample_disturbance_path,
    PAPER_VARIANCES_F1_X1_T1_T200, make_agent, write_csv)
from .zanon2019_training_report import trace_rows, CERTIFICATION_STATUS
from .train import ZeroResidualPolicy, run_episode
from .sac import set_seed

ROOT = v1.REPO / "evaporation_safe_sac/paired_economic_v2"
DEV = tuple(range(420000, 420010))
TRAIN_SEEDS = (42, 2027, 314159)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    return {str(p): v1.digest(p) for p in v1.REPO.glob("zanon2019_paired_v2*.py")}


def guard(args):
    if args.root.resolve() != ROOT.resolve():
        raise ValueError("Use the new paired_economic_v2 root only; preserve old artifacts")
    env = v1.setup(args)
    if not np.array_equal(env[0].input_scale, [100., 100.]):
        raise ValueError("Frozen mapping physical input normalization changed")
    return env


def require_tests(root):
    data = load(root / "tests_receipt.json")
    if not data["passed"] or data["source_hashes"] != source_hashes():
        raise RuntimeError("Run v2 tests after the last source edit, before audit/training")
    return data


def tests(args):
    """Run all existing top-level *_tests.py and the legacy integration main."""
    args.root.mkdir(parents=True, exist_ok=True)
    suite = unittest.TestSuite()
    mains = []
    for path in sorted(v1.REPO.glob("*tests.py")):
        module = importlib.import_module("evaporation."+path.stem)
        loaded = unittest.defaultTestLoader.loadTestsFromModule(module)
        suite.addTests(loaded)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and function.__module__ == module.__name__ \
                    and not inspect.signature(function).parameters:
                suite.addTest(unittest.FunctionTestCase(function))
        if loaded.countTestCases() == 0 and hasattr(module, "main"):
            mains.append(module)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    fallback = None
    if result.wasSuccessful() and result.skipped:
        optional_python = v1.REPO/".venv-ecc2019/Scripts/python.exe"
        if len(result.skipped) == 1 and "CasADi" in result.skipped[0][1] and optional_python.exists():
            command = [str(optional_python), "-m", "unittest",
                       "evaporation.ecc2019_reproduction_tests.SolverTests", "-v"]
            process = subprocess.run(command, cwd=v1.REPO.parent, capture_output=True, text=True)
            print(process.stdout, process.stderr, flush=True)
            count = re.search(r"Ran (\d+) tests?", process.stderr)
            fallback = dict(command=command, passed=process.returncode == 0 and count is not None,
                            tests=int(count.group(1)) if count else 0)
    integrations = []
    if result.wasSuccessful():
        for module in [importlib.import_module("evaporation.test"), *mains]:
            print("integration suite:", module.__name__, flush=True)
            module.main()
            integrations.append(module.__name__)
    passed = result.wasSuccessful() and (not result.skipped or (fallback and fallback["passed"]))
    v1.save(args.root / "tests_receipt.json", dict(passed=passed, unit_tests=result.testsRun,
        skipped=[dict(test=str(test), reason=reason) for test, reason in result.skipped],
        optional_environment_checks=fallback, total_test_methods=result.testsRun+(fallback["tests"] if fallback else 0),
        integration_suites=integrations, source_hashes=source_hashes(), training_performed=False))
    if not passed:
        raise RuntimeError("Tests failed; calibration and training are blocked")


def archived_pair(path):
    rows = v1.read_csv(path)
    if [int(r["step"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Non-contiguous archived trajectory")
    result = []
    for prefix in ("baseline", "SAC"):
        records = []
        for r in rows:
            record = {key: np.array([float(r[f"{prefix}_{key}_{i}"]) for i in range(n)])
                      for key, n in (("state", 2), ("control", 2), ("raw_action", 2), ("w_hat", 2),
                                     ("residual", 2), ("applied_residual", 2))}
            record["disturbance"] = np.array([float(r[key]) for key in ("F1", "X1", "T1", "T200")])
            record["economic_cost"] = float(r[f"{prefix}_economic_cost"])
            for key in ("in_Z_before", "in_Omega_before", "qp_feasible", "collapsed_action",
                        "physical_state_violation", "physical_input_violation", "robust_region_violation"):
                record[key] = r.get(f"{prefix}_{key}", "False" if "violation" in key else "True") == "True"
            for key in ("final_verification_gap", "applied_displacement_from_baseline", "interior_chebyshev_radius"):
                record[key] = float(r[f"{prefix}_{key}"]) if r.get(f"{prefix}_{key}") else 0.
            record["safety_mode"] = r[f"{prefix}_safety_mode"]
            records.append(record)
        result.append(records)
    return tuple(result)


def calibration_sources():
    lock = load(v1.LOCK_ROOT / "locked_final_policies.json")
    files = []
    for policy in lock["policies"]:
        seed = policy["training_seed"]
        run = Path(policy["checkpoint_path"]).parents[2]
        for ep, role in ((policy["selected_episode"], "Candidate_A"), (100, "old_late_high_F200")):
            for dev in DEV:
                files.append((role, seed, dev, run/"evaluation_trajectories"/f"episode_{ep:04d}"/f"seed_{dev}.csv"))
    for candidate in (71, 76, 133):
        for dev in DEV[:3]:
            files.append((f"oracle_{candidate}", None, dev, v1.DEFAULT_OUT/"oracle10/witness_trajectories"/
                          f"candidate_{candidate:04d}_seed_{dev}.csv"))
    return files


def calibrate(econ, smooth):
    econ, smooth = np.asarray(econ), np.asarray(smooth)
    magnitude, excitation = float(np.abs(econ).mean()), float(smooth.mean())
    if not np.isfinite(econ).all() or not np.isfinite(smooth).all() or magnitude <= 1e-10 \
            or excitation <= 1e-10 or np.count_nonzero(smooth > 1e-12) < 100:
        raise RuntimeError("Unstable/unexcited calibration; do not invent lambda")
    return dict(mean_abs_r_econ=magnitude, median_abs_r_econ=float(np.median(np.abs(econ))),
        mean_smoothness_penalty=excitation,
        median_nonzero_smoothness_penalty=float(np.median(smooth[smooth > 1e-12])),
        p95_smoothness_penalty=float(np.percentile(smooth, 95)),
        lambda_smooth=.05*magnitude/excitation)


def audit(args):
    require_tests(args.root)
    env = guard(args)
    cfg, model, design = env[:3]
    for name in ("reward_scale_audit.json", "smoothness_regularization_config.json"):
        if (args.root/name).exists():
            raise RuntimeError("Preserve immutable calibration; re-audit requires a new experiment version")
    arrays, sources, family = [], [], []
    for role, seed, dev, path in calibration_sources():
        base, records = archived_pair(path)
        paired_metrics(records, base, env, 0., np.zeros(5))  # timing, complete safety, physical inputs
        raw = components(records, base, cfg.input_scale, model.physical_input(design.v_ref))
        baseline_identity = components(base, base, cfg.input_scale, model.physical_input(design.v_ref))
        if np.any(baseline_identity):
            raise RuntimeError("Zero-residual paired identity audit failed")
        if len(raw) != 1000:
            raise ValueError("Expected complete development 1000-step evidence")
        arrays.append(raw)
        sources.append(dict(role=role, training_seed=seed, validation_seed=dev, path=str(path), sha256=v1.digest(path)))
        family.append(dict(role=role, training_seed=seed, validation_seed=dev,
            mean_abs_r_econ=float(np.abs(raw[:, 0]).mean()), mean_smoothness_penalty=float(raw[:, 1].mean())))
    all_raw = np.concatenate(arrays)
    stats = calibrate(all_raw[:, 0], all_raw[:, 1])
    weight = stats["lambda_smooth"]
    report = dict(**stats, calibration_files=sources, per_trajectory=family,
        pooling="69 complete paired trajectories, 1000 equally weighted steps each; 30 A, 30 late, 9 oracle",
        paired_baseline_zero_identity="baseline columns independently give zero economics and zero excess",
        implementation_choice="fixed 5% pooled development scale, not an ECC2019 paper-defined coefficient",
        sensitivity_10pct_lambda=2*weight, training_performed=False, final_test_performed=False,
        certification_status=CERTIFICATION_STATUS)
    v1.save(args.root/"reward_scale_audit.json", report)
    config = dict(lambda_smooth=weight, target_fraction=.05, scale_P100=float(cfg.input_scale[0]),
        scale_F200=float(cfg.input_scale[1]), normalization_source="unchanged cfg.input_scale used by frozen actual-input move and mapping; physical [100,100]",
        first_step="both previous actual inputs=physical(design.v_ref), identical to run_episode reset",
        calibration_files=sources, mean_abs_r_econ=stats["mean_abs_r_econ"],
        mean_raw_smoothness_penalty=stats["mean_smoothness_penalty"], estimated_regularizer_fraction=.05,
        regularizer_fraction_definition="lambda*sum(raw_smooth)/sum(abs(stepwise_paired_econ)); undefined denominator -> null",
        audit_sha256=v1.digest(args.root/"reward_scale_audit.json"), source_hashes=source_hashes(), criterion=CRITERION,
        economic_scale=200., stochastic_residual_scale=.10, frozen=True)
    v1.save(args.root/"smoothness_regularization_config.json", config)
    print(json.dumps(stats), flush=True)


def locked_config(root):
    config = load(root/"smoothness_regularization_config.json")
    if config["source_hashes"] != source_hashes() or config["audit_sha256"] != v1.digest(root/"reward_scale_audit.json"):
        raise RuntimeError("Source or calibration changed; immutable experiment is blocked")
    audit = load(root/"reward_scale_audit.json")
    if config["lambda_smooth"] != audit["lambda_smooth"] or config["target_fraction"] != .05 \
            or [config["scale_P100"], config["scale_F200"]] != [100., 100.] \
            or config["economic_scale"] != 200. or config["stochastic_residual_scale"] != .10:
        raise RuntimeError("Frozen reward scale/normalization was modified")
    for src in config["calibration_files"]:
        if v1.digest(src["path"]) != src["sha256"]:
            raise RuntimeError("Calibration evidence changed")
    return config


def reclassify(args):
    require_tests(args.root)
    config = locked_config(args.root)
    env = guard(args)
    oracle = v1.DEFAULT_OUT/"oracle10"
    summary = load(oracle/"oracle_summary.json")
    if summary["joint_witness_count"] != 0:
        raise ValueError("Unexpected v1 result; never relabel or overwrite v1")
    for name in ("oracle_v2_summary.json", "oracle_v2_reclassification.csv"):
        if (args.root/name).exists():
            raise RuntimeError("Reclassification already archived; preserve evidence")
    classified, evidence = [], []
    old = v1.read_csv(oracle/"all_candidates.csv")
    for row in old:
        candidate = int(row["candidate"])
        path = oracle/"candidate_metrics"/f"candidate_{candidate:04d}.csv"
        per = v1.read_csv(path)
        if sorted(int(r["seed"]) for r in per) != list(DEV[:3]):
            raise ValueError("Oracle disturbance membership changed")
        numerical = [{k: (v == "True" if k == "safety_passed" else float(v))
                      for k, v in r.items() if k in ("safety_passed", "economic_improvement_pct", "P100_TV_ratio", "F200_TV_ratio")} for r in per]
        classified.append(dict(candidate=candidate, criterion=CRITERION, joint_v1=row["joint_candidate"] == "True",
            joint_v2=joint(numerical), mean_economic_improvement_pct=float(row["economic_improvement_pct"]),
            worst_economic_improvement_pct=min(r["economic_improvement_pct"] for r in numerical),
            worst_P100_TV_ratio=max(r["P100_TV_ratio"] for r in numerical),
            worst_F200_TV_ratio=max(r["F200_TV_ratio"] for r in numerical),
            X2_IAE_ratio=float(row["X2_IAE_ratio"]), P2_IAE_ratio=float(row["P2_IAE_ratio"]),
            X2_ISE_ratio=float(row["X2_ISE_ratio"]), P2_ISE_ratio=float(row["P2_ISE_ratio"])))
        evidence.append(dict(path=str(path), sha256=v1.digest(path), per_realization=per))
    if len(classified) != 145:
        raise ValueError("Expected all 145 original candidates")
    op = []
    for candidate in (71, 76, 133):
        for dev in DEV[:3]:
            path = oracle/"witness_trajectories"/f"candidate_{candidate:04d}_seed_{dev}.csv"
            base, records = archived_pair(path)
            metric, _ = paired_metrics(records, base, env, config["lambda_smooth"], np.zeros(5))
            op.append(dict(candidate=candidate, validation_seed=dev, **metric))
            evidence.append(dict(path=str(path), sha256=v1.digest(path)))
    write_csv(args.root/"oracle_v2_reclassification.csv", classified)
    write_csv(args.root/"oracle_v2_operating_point_metrics.csv", op)
    result = dict(criterion=CRITERION, joint_witness_count_v1=0, candidates=145,
        v2_joint_witness_count=sum(r["joint_v2"] for r in classified),
        joint_v2_candidates=[r["candidate"] for r in classified if r["joint_v2"]],
        reclassification_sha256=v1.digest(args.root/"oracle_v2_reclassification.csv"),
        v1_original_summary_sha256=v1.digest(oracle/"oracle_summary.json"), evidence=evidence,
        smoothness_config_sha256=v1.digest(args.root/"smoothness_regularization_config.json"),
        no_tracking_gate=True, training_performed=False, final_test_performed=False)
    v1.save(args.root/"oracle_v2_summary.json", result)
    print("v2 witnesses:", result["joint_v2_candidates"], flush=True)
    if not result["v2_joint_witness_count"]:
        raise RuntimeError("No joint_v2 witnesses: training blocked")


def assessment(rows, episode, step, warmup):
    if len(rows) != 10 or sorted(r["validation_seed"] for r in rows) != list(DEV):
        raise ValueError("Fixed validation must contain all ten prespecified paths")
    numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float, np.number)) and k != "validation_seed"]
    result = {k: float(np.mean([r[k] for r in rows])) for k in numeric
              if all(r[k] is not None and np.isfinite(r[k]) for r in rows)}
    result.update(episode=episode, global_step=step, post_warmup=step >= warmup,
        mean_economic_improvement_pct=float(np.mean([r["economic_improvement_pct"] for r in rows])),
        worst_seed_economic_improvement_pct=min(r["economic_improvement_pct"] for r in rows),
        positive_seed_fraction=float(np.mean([r["economic_improvement_pct"] > 0 for r in rows])),
        minimum_X2_margin=min(r["minimum_X2_margin"] for r in rows),
        empirical_safe=all(all(r[k] == 0 for k in v1.SAFETY) for r in rows),
        joint_v2=joint(rows), criterion=CRITERION, certification_status=CERTIFICATION_STATUS,
        regularizer_fraction=(sum(-r["smoothness_regularization_component"] for r in rows)
            /sum(r["sum_abs_economic_reward"] for r in rows)) if sum(r["sum_abs_economic_reward"] for r in rows) > 1e-12 else None)
    for key in v1.SAFETY:
        result[key] = int(sum(r[key] for r in rows))
    result["positive_economic_safe_learned"] = bool(result["post_warmup"] and result["empirical_safe"]
                                                   and result["mean_economic_improvement_pct"] > 0)
    return result


def pilot_gate(summary, checkpoints):
    return bool(summary["run_state"] == "completed" and summary["completed_episodes"] == 100
        and summary["global_step"] == 100000 and summary["max_action_collapse_fraction"] == 0.
        and summary["replay_decomposition_checked"] and summary["no_nonfinite"]
        and any(s["positive_economic_safe_learned"] for s in checkpoints))


def train(args):
    require_tests(args.root)
    env = guard(args)
    config = locked_config(args.root)
    oracle = load(args.root/"oracle_v2_summary.json")
    config_hash = v1.digest(args.root/"smoothness_regularization_config.json")
    if oracle["v2_joint_witness_count"] <= 0 or oracle["smoothness_config_sha256"] != config_hash:
        raise RuntimeError("Verified joint_v2 evidence required")
    if oracle["reclassification_sha256"] != v1.digest(args.root/"oracle_v2_reclassification.csv"):
        raise RuntimeError("Oracle classification changed")
    for src in oracle["evidence"]:
        if v1.digest(src["path"]) != src["sha256"]:
            raise RuntimeError("Original oracle evidence changed")
    for prior in TRAIN_SEEDS[:TRAIN_SEEDS.index(args.seed)]:
        directory = args.root/f"seed_{prior}"
        if not pilot_gate(load(directory/"run_summary.json"), load(directory/"checkpoints.json")):
            raise RuntimeError(f"Prior seed {prior} has not passed continuation gate")
    cfg, model, design, omega, domain = env[:5]
    cfg.episodes = 100
    output = args.root/f"seed_{args.seed}"
    v1.fresh(output)
    set_seed(args.seed)
    agent = make_agent(cfg, args.device)
    agent.zero_initialize_residual_mean()
    dual = SafetyDual.create()
    replay = EconomicReplay(23, 2, cfg.replay_capacity, agent.device, dual, config["lambda_smooth"])
    model_dir = output/"models"; model_dir.mkdir()
    manifest = dict(criterion=CRITERION, protocol=v1.protocol(args, cfg),
        training_seed=args.seed, steps=1000, episodes=100, eval_every=5,
        frozen_SAC_config=vars(agent.cfg), frozen_experiment_config=vars(cfg),
        smoothness_config_sha256=config_hash, source_hashes=source_hashes(),
        validation_seeds=DEV, representative_seed=420000,
        objective="paired economics /200 minus fixed excess-square smoothness and safety-only dual terms",
        tracking_hard_gate=False, final_selection_rule="not locked; all checkpoints retained",
        disturbance_seed_rule="seed*1000000+episode; local action rng=path_seed+77",
        gaussian_iid="implementation choice for reproduction", final_test_performed=False)
    manifest["protocol"].update(schema="paired_economic_v2", objective=manifest["objective"],
        IAE_TV_tolerance="v1 diagnostic only, not v2 objective or success gate", boundary_allowance="diagnostic only")
    v1.save(output/"experiment_manifest.json", manifest)
    training, fixed, checkpoints = [], [], []
    global_step = 0
    max_collapse = 0.
    cache = {}

    def persist(state, failure=None):
        write_csv(output/"training_log.csv", training)
        write_csv(output/"fixed_evaluation.csv", fixed)
        write_csv(output/"checkpoint_performance.csv", checkpoints)
        v1.save(output/"checkpoints.json", checkpoints)
        summary = dict(run_state=state, completed_episodes=len(training), global_step=global_step,
            max_action_collapse_fraction=max_collapse, replay_decomposition_checked=True, no_nonfinite=failure is None,
            failure=failure, training_performed=True, final_test_performed=False,
            smoothness_config_sha256=config_hash, certification_status=CERTIFICATION_STATUS)
        summary["continue_to_next_seed"] = pilot_gate(summary, checkpoints)
        v1.save(output/"run_summary.json", summary)

    def evaluate(episode):
        nonlocal max_collapse
        actor_path = model_dir/f"episode_{episode:04d}_actor.pth"
        agent.save_actor(actor_path)
        latest = []
        with torch.no_grad():
            for seed, (path, base, _, _) in cache.items():
                print(f"fixed_eval episode={episode}, validation_seed={seed}", flush=True)
                records, _, _ = v1.rollout(env, path, seed, agent)
                metric, raw = paired_metrics(records, base, env, replay.weight, dual.values)
                latest.append(dict(episode=episode, global_step=global_step, validation_seed=seed, **metric))
                trace = trace_rows(base, records, .10)
                for r, c in zip(trace, raw):
                    r.update(economic_reward_raw=float(c[0]), smoothness_penalty_raw=float(c[1]),
                             smoothness_contribution=-replay.weight*float(c[1]))
                write_csv(output/"evaluation_trajectories"/f"episode_{episode:04d}"/f"seed_{seed}.csv", trace)
        fixed.extend(latest)
        status = assessment(latest, episode, global_step, cfg.warmup_steps)
        status["checkpoint"] = str(actor_path)
        checkpoints.append(status)
        max_collapse = max(max_collapse, *(r["action_collapse_fraction"] for r in latest))
        print("fixed_eval", json.dumps(status, allow_nan=False), flush=True)
        persist("running")
        # Five consecutive non-positive evaluations with >10% regularization:
        # explicit sustained-scale-mismatch diagnostic, never retune mid-run.
        late = [s for s in checkpoints if s["post_warmup"]][-5:]
        if len(late) == 5 and all(s.get("regularizer_fraction", 0) is not None
                and s["regularizer_fraction"] > .10 and s["mean_economic_improvement_pct"] <= 0 for s in late):
            raise RuntimeError("regularizer scale mismatch; new calibration needs a new experiment version")

    try:
        cache = v1.paths_and_baselines(env, DEV, output)
        evaluate(0)
        for episode in range(1, 101):
            if v1.digest(args.root/"smoothness_regularization_config.json") != config_hash:
                raise RuntimeError("Immutable lambda/config modified during training")
            seed = args.seed*1000000+episode
            path, _ = sample_disturbance_path(cfg, "zanon2019_stochastic", seed, 1000,
                                             PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
            base, _, _ = v1.rollout(env, path, seed, ZeroResidualPolicy())
            ctrl = v1.PairedEvidenceController(cfg, model, design, omega, domain, path)
            replay.bind(ctrl, base)
            dual_used = dual.values.copy()
            stat, records, global_step = run_episode(cfg, model, ctrl, agent, replay,
                np.random.default_rng(seed+77), training=True, global_step=global_step,
                disturbance_trajectory=path, initial_state_override=cfg.robust_economic_reference_state)
            metric, raw = paired_metrics(records, base, env, replay.weight, dual_used)
            if not np.allclose(replay.episode_components, raw, rtol=0, atol=1e-8):
                raise RuntimeError("Replay decomposition differs from audited actual-input metrics")
            if not all(torch.isfinite(p).all().item() for net in (agent.actor, agent.q1, agent.q2) for p in net.parameters()):
                raise RuntimeError("Nonfinite SAC parameter")
            if not np.isfinite(agent.alpha) or any(not np.isfinite(stat[key])
                    for key in ("actor_loss", "q_loss", "entropy") if replay.size >= cfg.batch_size):
                raise RuntimeError("Nonfinite SAC loss/entropy/alpha")
            dual.update(raw[:, 2:].sum(0)/1000.)
            row = dict(episode=episode, global_step=global_step, disturbance_seed=seed, **metric,
                episode_return=metric["total_training_reward"], legacy_return_diagnostic=float(stat["return"]),
                **{f"lambda_{k}": float(v) for k, v in zip(SAFETY_KEYS, dual.values)})
            for key in ("actor_loss", "q_loss", "entropy"):
                if key in stat and np.isfinite(stat[key]): row[key] = float(stat[key])
            training.append(row)
            max_collapse = max(max_collapse, metric["action_collapse_fraction"])
            persist("running")
            print(f"episode {episode}/100 gain={metric['economic_improvement_pct']:.7f}%", flush=True)
            if episode % 5 == 0: evaluate(episode)
            agent.save_checkpoint(model_dir/"last_checkpoint.pth")
        agent.save_actor(model_dir/"episode100_development_actor.pth")
        persist("completed")
        from .zanon2019_paired_v2_report import report
        report(args.root, env)
    except Exception as exc:
        if "ctrl" in locals(): write_csv(output/"failure_trajectory.csv", ctrl.evidence)
        if getattr(exc, "evidence", None): write_csv(output/"evaluation_failure_trajectory.csv", exc.evidence)
        persist("aborted", dict(reason=str(exc), attempted_global_step=global_step))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("tests", "audit", "reclassify", "train", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--seed", type=int, choices=TRAIN_SEEDS, default=42)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    if args.root.resolve() != ROOT.resolve():
        raise ValueError("All v2 outputs must stay in paired_economic_v2; never overwrite v1")
    args.steps, args.episodes, args.economic_scale = 1000, 100, 200.
    args.design, args.omega_vertices = DEFAULT_DESIGN, DEFAULT_OMEGA
    if args.phase == "report":
        from .zanon2019_paired_v2_report import report
        report(args.root, guard(args))
    else:
        {"tests": tests, "audit": audit, "reclassify": reclassify, "train": train}[args.phase](args)


if __name__ == "__main__":
    main()
