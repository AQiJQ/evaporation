"""Isolated paired-economic experiment: audit, actor-action oracle, local train.

Never changes the locked Experiment I or any controller/safety design. Gaussian
closed-loop validation remains empirical, not a continuous-domain certificate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .sac import set_seed
from .train import OBS_DIM, ZeroResidualPolicy, run_episode
from .omega_constrained_replay import DualState
from .zanon2019_benchmark import (make_setup, make_agent, sample_disturbance_path,
    PAPER_VARIANCES_F1_X1_T1_T200, DEFAULT_DESIGN, DEFAULT_OMEGA, write_csv)
from .zanon2019_training_report import (EvidenceController, metrics, relative,
    trace_rows, CERTIFICATION_STATUS)
from .zanon2019_paired_objective import (KEYS, PRECONDITION_FLOORS, Limits,
    limits_from_metrics, violations, component_series, PairedReplay)

REPO = Path(__file__).resolve().parent
LOCK_ROOT = REPO / "evaporation_safe_sac/final_stochastic_candidate_A"
DEFAULT_OUT = REPO / "evaporation_safe_sac/paired_economic_v1"
RESERVED = set(range(430000, 430050))
SAFETY = ("physical_state_violation_steps", "physical_input_violation_count",
          "QP_infeasible_count", "Omega_exit_count", "robust_region_violation_count")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    def encode(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=encode,
                                   allow_nan=False), encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fresh(path):
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Preserve existing evidence; select a fresh --output-dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


def setup(args):
    from .zanon2019_final_experiment import verify_policy_lock
    verify_policy_lock(LOCK_ROOT)  # read-only; old Experiment I must remain intact
    if digest(args.design) != digest(DEFAULT_DESIGN) or digest(args.omega_vertices) != digest(DEFAULT_OMEGA):
        raise ValueError("This experiment requires the unchanged balanced-B geometry")
    cfg, model, design, omega, domain, old_weights = make_setup(
        args.steps, args.seed, args.design, args.omega_vertices)
    cfg.stochastic_residual_scale = .10
    cfg.abort_on_first_uncertified_step = True
    ranges = np.asarray(cfg.input_scale) * (design.robust_input_upper - design.robust_input_lower)
    return cfg, model, design, omega, domain, ranges, old_weights


def protocol(args, cfg):
    frozen = json.loads((LOCK_ROOT / "protocol_lock.json").read_text(encoding="utf-8"))
    return dict(schema="zanon2019_paired_economics_v1", alpha=.10,
        steps=args.steps, dt_min=cfg.dt_min, disturbance_mode="zanon2019_stochastic",
        variances=PAPER_VARIANCES_F1_X1_T1_T200,
        initial_state=cfg.robust_economic_reference_state,
        geometry_sha256=[digest(args.design), digest(args.omega_vertices)],
        frozen_source_sha256=frozen["source_sha256"],
        safety_constraint_policy="unchanged; abort physical/input/QP/Omega/robust-region anomalies; W audit only",
        baseline="unchanged paired zero-residual balanced-B controller; not ECC2019 NMPC",
        objective="paired baseline minus SAC economic stage cost; independent normalized constraints",
        economic_scale=args.economic_scale, IAE_TV_tolerance=.10,
        boundary_allowance="10% additional baseline outer-10% occupancy; new implementation choice, not paper-defined",
        nonlinear_safety_scope=CERTIFICATION_STATUS,
        observation_dim=23, architecture="original interior anchor Z/Omega, inactive no-jump Gm/Bj",
        extra_baseline="never used for main reward or checkpoint selection")


class ConstantPolicy:
    def __init__(self, action=(0., 0.)): self.action = np.asarray(action, dtype=float)
    def select_action(self, _obs, deterministic=True): return self.action.copy()


class SegmentPolicy:
    def __init__(self, actions, segment_steps):
        self.actions, self.segment_steps, self.step = np.asarray(actions), segment_steps, 0
    def select_action(self, _obs, deterministic=True):
        a = self.actions[min(self.step // self.segment_steps, len(self.actions)-1)]
        self.step += 1
        return a.copy()


class PairedEvidenceController(EvidenceController):
    """Extra immediate robust-region audit; never changes action or QP."""
    def audit_next_state(self, next_state):
        super().audit_next_state(next_state)
        xn = self.model.normalized_state(next_state)
        un = self.model.normalized_input(self.last_control)
        bad = bool(np.any(xn < self.d.robust_state_lower-1e-8)
                   or np.any(xn > self.d.robust_state_upper+1e-8)
                   or np.any(un < self.d.robust_input_lower-1e-8)
                   or np.any(un > self.d.robust_input_upper+1e-8))
        self.evidence[-1]["robust_region_violation"] = int(bad)
        if bad:
            raise RuntimeError(f"Paired experiment robust-region anomaly: {self.evidence[-1]}")


def rollout(env, path, path_seed, policy):
    cfg, model, design, omega, domain = env[:5]
    ctrl = PairedEvidenceController(cfg, model, design, omega, domain, path)
    try:
        stat, records, _ = run_episode(cfg, model, ctrl, policy, None,
            np.random.default_rng(path_seed + 900000), training=False, global_step=0,
            disturbance_trajectory=path, initial_state_override=cfg.robust_economic_reference_state)
        metric = metrics(records, cfg, model, design, omega)
        if len(records) != len(path) or any(metric[k] != 0 for k in SAFETY):
            raise RuntimeError("Incomplete/unsafe rollout")
        return records, metric, ctrl
    except Exception as exc:
        # Caller persists failures; never score a fallback/partial rollout as safe.
        exc.evidence = ctrl.evidence
        raise


def paths_and_baselines(env, seeds, output):
    cfg = env[0]
    cache = {}
    for seed in seeds:
        if seed in RESERVED: raise ValueError("Reserved independent test seed is forbidden")
        print(f"paired baseline seed={seed}", flush=True)
        path, _ = sample_disturbance_path(cfg, "zanon2019_stochastic", seed,
            cfg.steps_per_episode, PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
        records, metric, _ = rollout(env, path, seed, ZeroResidualPolicy())
        limits = limits_from_metrics(metric, len(path), env[5])
        cache[seed] = (path, records, metric, limits)
        write_csv(output / "paired_baselines" / f"seed_{seed}.csv", trace_rows(records, records, .10))
    return cache


def paired_result(metric, baseline, limits):
    g = violations(metric, limits)
    gain = 100 * (baseline["J_econ"] - metric["J_econ"]) / baseline["J_econ"]
    safe = all(metric[k] == 0 for k in SAFETY)
    return dict(economic_improvement_pct=float(gain),
        **{f"g_{k}": float(v) for k, v in zip(KEYS, g)},
        **{k+"_ratio": relative(metric[k], baseline[k])
           for k in ("X2_IAE", "P2_IAE", "X2_ISE", "P2_ISE", "P100_TV", "F200_TV")},
        five_constraints_feasible=bool(np.all(g <= 1e-9)), safety_passed=safe,
        joint_candidate=bool(safe and gain > 0 and np.all(g <= 1e-9)))


def audit(args):
    env = setup(args)
    lock = json.loads((args.lock_root / "locked_final_policies.json").read_text(encoding="utf-8"))
    fresh(args.output_dir)
    save(args.output_dir / "protocol.json", protocol(args, env[0]))
    rows, sources = [], {}
    for p in lock["policies"]:
        manifest_path = Path(p["config_manifest"])
        if digest(manifest_path) != p["config_manifest_sha256"]:
            raise ValueError("Locked manifest changed")
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if m["steps"] != args.steps or m["stochastic_residual_scale"] != .10:
            raise ValueError("Audit must match locked alpha=.10 and horizon")
        source = manifest_path.parent / "fixed_evaluation.csv"
        sources[str(source)] = digest(source)
        for r in read_csv(source):
            bm = {k.removeprefix("baseline_"): float(v) for k, v in r.items()
                  if k.startswith("baseline_") and v not in ("", "True", "False")}
            pm = {k.removeprefix("SAC_"): float(v) for k, v in r.items()
                  if k.startswith("SAC_") and v not in ("", "True", "False")}
            limits = limits_from_metrics(bm, args.steps, env[5])
            result = paired_result(pm, bm, limits)
            rows.append(dict(training_seed=p["training_seed"], episode=int(r["episode"]),
                disturbance_seed=int(r["seed"]), **result))
    write_csv(args.output_dir / "checkpoint_constraint_audit.csv", rows)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["training_seed"], row["episode"]), []).append(row)
    checkpoint_rows = [dict(training_seed=seed, episode=ep,
        mean_gain_pct=float(np.mean([r["economic_improvement_pct"] for r in group])),
        worst_gain_pct=min(r["economic_improvement_pct"] for r in group),
        all_constraints_feasible=all(r["five_constraints_feasible"] for r in group),
        all_safety_passed=all(r["safety_passed"] for r in group),
        joint_checkpoint=bool(len(group) == 10 and ep*args.steps >= env[0].warmup_steps
            and all(r["five_constraints_feasible"] and r["safety_passed"] for r in group)
            and np.mean([r["economic_improvement_pct"] for r in group]) > 0
            and min(r["economic_improvement_pct"] for r in group) >= 0),
        **{f"worst_g_{k}": max(r[f"g_{k}"] for r in group) for k in KEYS})
        for (seed, ep), group in grouped.items()]
    write_csv(args.output_dir / "checkpoint_constraint_summary.csv", checkpoint_rows)
    positive = np.array([[max(0., r[f"g_{k}"]) for k in KEYS] for r in rows])
    scales = np.maximum(np.quantile(positive, .90, axis=0), PRECONDITION_FLOORS)
    dominant = {k: 0 for k in KEYS}
    for r in rows:
        normalized = np.array([r[f"g_{k}"] for k in KEYS]) / scales
        if normalized.max() > 0: dominant[KEYS[int(np.argmax(normalized))]] += 1
    summary = dict(schema="paired_constraint_audit_v1", protocol=protocol(args, env[0]),
        row_count=len(rows), sources=sources, component_scales=scales,
        preconditioning="fixed positive-violation p90, numerical floors; inequality thresholds unchanged",
        dominant_constraint_counts=dominant,
        feasible_positive_rows=sum(r["joint_candidate"] for r in rows),
        joint_checkpoint_count=sum(r["joint_checkpoint"] for r in checkpoint_rows),
        late_infeasible_rows=sum(r["episode"] >= 80 and not r["five_constraints_feasible"] for r in rows),
        warning="reused development validation; no independent test; no proof of global feasibility/infeasibility")
    save(args.output_dir / "constraint_audit.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k in
                     ("row_count", "dominant_constraint_counts", "feasible_positive_rows", "joint_checkpoint_count", "late_infeasible_rows")}, indent=2))


def nondominated(rows):
    safe = [r for r in rows if r["safety_passed"]]
    def v(r): return np.r_[-r["economic_improvement_pct"],
        [r[k+"_ratio"] for k in ("X2_IAE", "P2_IAE", "X2_ISE", "P2_ISE", "P100_TV", "F200_TV")],
        r["boundary_fraction"]]
    return [r for r in safe if not any(np.all(v(t) <= v(r)+1e-10)
            and np.any(v(t) < v(r)-1e-10) for t in safe if t is not r)]


def oracle(args):
    env = setup(args)
    cfg, model, design = env[:3]
    fresh(args.output_dir)
    proto = protocol(args, cfg)
    save(args.output_dir / "protocol.json", proto)
    cache = paths_and_baselines(env, args.oracle_seeds, args.output_dir)
    rng = np.random.default_rng(args.search_seed)
    count = int(np.ceil(args.steps / args.segment_steps))
    rows, profiles, failure_rows = [], {}, []

    def test_profile(profile):
        index = len(rows)
        per_seed = []
        try:
            for seed, (path, baseline, bm, limits) in cache.items():
                print(f"oracle candidate={index}: rollout seed={seed}, steps={args.steps}", flush=True)
                records, metric, _ = rollout(env, path, seed, SegmentPolicy(profile, args.segment_steps))
                per_seed.append(dict(seed=seed, **paired_result(metric, bm, limits),
                                     boundary_fraction=metric["any_outer_10pct_steps"] / args.steps))
            row = dict(candidate=index, safety_passed=True,
                economic_improvement_pct=float(np.mean([r["economic_improvement_pct"] for r in per_seed])),
                worst_gain_pct=min(r["economic_improvement_pct"] for r in per_seed),
                boundary_fraction=float(np.mean([r["boundary_fraction"] for r in per_seed])),
                **{k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]
                   if k.endswith("_ratio")},
                **{f"g_{k}": max(r[f"g_{k}"] for r in per_seed) for k in KEYS})
            row["joint_candidate"] = bool(row["economic_improvement_pct"] > 0
                                          and row["worst_gain_pct"] >= 0
                                          and all(row[f"g_{k}"] <= 1e-9 for k in KEYS))
            profiles[index] = profile.copy()
            write_csv(args.output_dir / "candidate_metrics" / f"candidate_{index:04d}.csv", per_seed)
        except Exception as exc:
            row = dict(candidate=index, safety_passed=False, joint_candidate=False,
                       failure_reason=str(exc))
            failure_rows.append(row)
            write_csv(args.output_dir / "failed_candidates" / f"candidate_{index:04d}.csv", getattr(exc, "evidence", []))
        rows.append(row)
        print(f"oracle candidate={index} safe={row['safety_passed']} gain={row.get('economic_improvement_pct')} joint={row['joint_candidate']}", flush=True)
        write_csv(args.output_dir / "all_candidates.csv", rows)

    test_profile(np.zeros((count, 2)))
    for _ in range(args.generations):
        front = nondominated(rows)
        for _ in range(args.population):
            if front and rng.random() < .8:
                parent = profiles[front[int(rng.integers(len(front)))]["candidate"]]
                profile = parent.copy()
                # Block mutations preserve segmented control, no 1000 free actions.
                selected = rng.choice(count, size=max(1, count//8), replace=False)
                profile[selected] += rng.normal(0, .25, (len(selected), 2))
                if rng.random() < .3: profile += rng.normal(0, .08, (1, 2))
                profile = np.clip(profile, -1, 1)
            else:
                profile = np.tile(rng.uniform(-1, 1, (1, 2)), (count, 1))
            test_profile(profile)
    front = nondominated(rows)
    write_csv(args.output_dir / "pareto_front.csv", front)
    witnesses = [r for r in front if r["joint_candidate"]]
    # Representatives, not an assertion that economic max is universally best.
    selected = witnesses if witnesses else sorted(front, key=lambda r: -r["economic_improvement_pct"])[:3]
    witness_paths = []
    for row in selected:
        index = row["candidate"]
        save(args.output_dir / f"candidate_{index:04d}_actions.json",
             dict(raw_actor_actions=profiles[index], segment_steps=args.segment_steps))
        for seed, (path, baseline, _, _) in cache.items():
            records, _, _ = rollout(env, path, seed, SegmentPolicy(profiles[index], args.segment_steps))
            trace = args.output_dir / "witness_trajectories" / f"candidate_{index:04d}_seed_{seed}.csv"
            write_csv(trace, trace_rows(baseline, records, .10))
            witness_paths.append(dict(candidate=index, seed=seed, path=str(trace), sha256=digest(trace)))
    summary = dict(schema="paired_actor_oracle_v1", protocol=proto,
        search_seed=args.search_seed, disturbance_seeds=args.oracle_seeds,
        segment_steps=args.segment_steps, candidates=len(rows), failed_candidates=len(failure_rows),
        fully_safe_candidates=sum(r["safety_passed"] for r in rows),
        joint_witness_count=len(witnesses), witnesses=witnesses, trajectory_evidence=witness_paths,
        interpretation=("Joint trajectory witnesses found; no guarantee SAC learns them"
                        if witnesses else "No joint witness found by this finite search; not a global impossibility proof"),
        privileged_oracle="offline direct shooting knows the evaluated paths; not a deployable policy or fair controller comparator",
        training_performed=False)
    save(args.output_dir / "oracle_summary.json", summary)
    print(summary["interpretation"], flush=True)
    plot_oracle(front, args.output_dir)


def plot_oracle(front, output):
    if not front: return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in front:
        ax.scatter(r["economic_improvement_pct"], .5*(r["X2_IAE_ratio"]+r["P2_IAE_ratio"]),
                   s=20+30*.5*(r["P100_TV_ratio"]+r["F200_TV_ratio"]),
                   color="#009E73" if r["joint_candidate"] else "#D55E00")
    ax.axhline(1.1, ls="--", color="gray"); ax.axvline(0, color="gray", lw=.6)
    ax.set(xlabel="Mean paired economic improvement (%)", ylabel="Mean normalized IAE ratio",
           title="Offline Pareto candidates; green = joint witness, size = TV activity")
    fig.tight_layout(); fig.savefig(output / "oracle_pareto.png", dpi=180); plt.close(fig)


def verified_inputs(args, cfg):
    audit_path = args.constraint_audit
    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
    oracle_data = json.loads(args.oracle_summary.read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(protocol(args, cfg), default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)))
    for data in (audit_data, oracle_data):
        if data["protocol"] != expected: raise ValueError("Audit/oracle protocol differs from training")
    if oracle_data.get("joint_witness_count", 0) < 1:
        raise RuntimeError("No joint trajectory witness: stop and report trade-off, do not blindly train")
    for source, hash_value in audit_data["sources"].items():
        if digest(source) != hash_value: raise ValueError("Audit source changed")
    if not oracle_data.get("trajectory_evidence"):
        raise ValueError("Oracle claim lacks trajectory evidence")
    for evidence in oracle_data["trajectory_evidence"]:
        if digest(evidence["path"]) != evidence["sha256"]:
            raise ValueError("Oracle trajectory changed")
    return np.asarray(audit_data["component_scales"], dtype=float)


def train(args):
    env = setup(args)
    cfg, model, design, omega, domain, ranges, old_weights = env
    cfg.episodes = args.episodes
    scales = verified_inputs(args, cfg)
    fresh(args.output_dir)
    # Preserve original RNG rule and network/hyperparameters. Paired baseline
    # rollouts use local generators and do not draw Torch/global NumPy actions.
    set_seed(args.seed)
    agent = make_agent(cfg, args.device)
    agent.zero_initialize_residual_mean()
    dual = DualState.create(np.full(5, args.dual_lr), args.lambda_max)
    replay = PairedReplay(OBS_DIM+4, 2, cfg.replay_capacity, agent.device, dual,
                          scales=scales, economic_scale=args.economic_scale)
    model_dir = args.output_dir / "models"; model_dir.mkdir()
    manifest = dict(protocol=protocol(args, cfg), training_seed=args.seed,
        episodes=args.episodes, device=str(agent.device), frozen_SAC_config=vars(agent.cfg),
        frozen_experiment_config=vars(cfg), old_absolute_shaping_weights_diagnostic_only=old_weights,
        constraint_audit=str(args.constraint_audit), constraint_audit_sha256=digest(args.constraint_audit),
        oracle_summary=str(args.oracle_summary), oracle_summary_sha256=digest(args.oracle_summary),
        component_scales=scales, dual_lr=args.dual_lr, lambda_max=args.lambda_max,
        training_disturbance_seed_rule="seed*1000000+episode; local action RNG=path_seed+77",
        validation_seeds=args.eval_seeds, final_test_performed=False,
        learning_curve_target="fixed paired economic reward improves to a positive plateau; zero is baseline, not forced optimum")
    save(args.output_dir / "experiment_manifest.json", manifest)
    training_rows, fixed_rows, statuses = [], [], []
    global_step, best_joint = 0, None
    cache = {}

    def persist(state, failure=None):
        write_csv(args.output_dir / "training_log.csv", training_rows)
        write_csv(args.output_dir / "fixed_evaluation.csv", fixed_rows)
        write_csv(args.output_dir / "checkpoint_performance.csv", statuses)
        save(args.output_dir / "run_summary.json", dict(run_state=state,
            completed_episodes=len(training_rows), global_step=global_step,
            best_joint_checkpoint=best_joint, failure=failure,
            training_performed=True, certification_status=CERTIFICATION_STATUS,
            frozen_geometry_sha256=manifest["protocol"]["geometry_sha256"]))

    def evaluate(episode):
        nonlocal best_joint
        actor_path = model_dir / f"episode_{episode:04d}_actor.pth"
        agent.save_actor(actor_path)
        latest = []
        for seed, (path, base, bm, limits) in cache.items():
            print(f"fixed_eval episode={episode}: seed={seed}", flush=True)
            records, metric, _ = rollout(env, path, seed, agent)
            components = component_series(records, base, limits, cfg, model, design, args.economic_scale)
            result = paired_result(metric, bm, limits)
            row = dict(episode=episode, global_step=global_step, disturbance_seed=seed,
                       **result, **{f"baseline_{k}": v for k, v in bm.items()},
                       **{f"SAC_{k}": v for k, v in metric.items()},
                       paired_economic_reward_total=float(components[:, 0].sum()),
                       effective_objective_return=float(components[:, 0].sum()
                           - np.dot(dual.values, components[:, 1:].sum(axis=0)/scales)))
            latest.append(row)
            write_csv(args.output_dir / "evaluation_trajectories" /
                      f"episode_{episode:04d}" / f"seed_{seed}.csv", trace_rows(base, records, .10))
        fixed_rows.extend(latest)
        status = dict(episode=episode, global_step=global_step, post_warmup=global_step >= cfg.warmup_steps,
            economic_improvement_pct=float(np.mean([r["economic_improvement_pct"] for r in latest])),
            worst_gain_pct=min(r["economic_improvement_pct"] for r in latest),
            paired_economic_reward_total=float(np.mean([r["paired_economic_reward_total"] for r in latest])),
            **{f"worst_g_{k}": max(r[f"g_{k}"] for r in latest) for k in KEYS},
            all_safety_passed=all(r["safety_passed"] for r in latest),
            all_constraints_feasible=all(r["five_constraints_feasible"] for r in latest),
            mean_IAE_ratio=float(np.mean([.5*(r["X2_IAE_ratio"]+r["P2_IAE_ratio"]) for r in latest])),
            mean_TV_ratio=float(np.mean([.5*(r["P100_TV_ratio"]+r["F200_TV_ratio"]) for r in latest])),
            checkpoint=str(actor_path))
        status["eligible_joint_checkpoint"] = bool(status["post_warmup"] and status["all_safety_passed"]
            and status["all_constraints_feasible"] and status["economic_improvement_pct"] > 0
            and status["worst_gain_pct"] >= 0)
        statuses.append(status)
        if status["eligible_joint_checkpoint"]:
            score = (status["mean_IAE_ratio"] + status["mean_TV_ratio"], -status["economic_improvement_pct"])
            if best_joint is None or score < tuple(best_joint["selection_score"]):
                shutil.copyfile(actor_path, model_dir / "best_joint_actor.pth")
                best_joint = dict(status, selection_score=list(score),
                                  checkpoint=str(model_dir / "best_joint_actor.pth"))
        print("fixed_eval", json.dumps(status), flush=True)
        persist("running")
        plot_learning(training_rows, fixed_rows, args.output_dir)

    try:
        cache = paths_and_baselines(env, args.eval_seeds, args.output_dir)
        evaluate(0)
        for episode in range(1, args.episodes+1):
            seed = args.seed*1000000+episode
            if seed in RESERVED: raise ValueError("Training path collides with reserved test")
            path, _ = sample_disturbance_path(cfg, "zanon2019_stochastic", seed, args.steps,
                                             PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
            print(f"episode={episode}: paired zero-residual baseline seed={seed}", flush=True)
            baseline, bm, _ = rollout(env, path, seed, ZeroResidualPolicy())
            limits = limits_from_metrics(bm, args.steps, ranges)
            ctrl = PairedEvidenceController(cfg, model, design, omega, domain, path)
            replay.bind(ctrl, baseline, limits)
            lambda_used = dual.values.copy()
            stat, records, global_step = run_episode(cfg, model, ctrl, agent, replay,
                np.random.default_rng(seed+77), training=True, global_step=global_step,
                disturbance_trajectory=path, initial_state_override=cfg.robust_economic_reference_state)
            metric = metrics(records, cfg, model, design, omega)
            result = paired_result(metric, bm, limits)
            raw_sum = np.asarray(replay.episode_components).sum(axis=0)
            g = np.array([result[f"g_{k}"] for k in KEYS])
            if not np.allclose(raw_sum[1:], g, rtol=0, atol=1e-8):
                raise RuntimeError("Replay component sums do not match episode metrics")
            if not result["safety_passed"]: raise RuntimeError("Training robust/physical safety anomaly")
            dual.update(g/scales)
            effective = float(raw_sum[0] - np.dot(lambda_used, raw_sum[1:]/scales))
            row = dict(episode=episode, global_step=global_step, disturbance_seed=seed,
                reward_return=effective, paired_economic_reward_total=float(raw_sum[0]),
                legacy_weighted_return_diagnostic=stat["return"], **result,
                **{f"lambda_used_{k}": float(v) for k, v in zip(KEYS, lambda_used)},
                **{f"lambda_next_{k}": float(v) for k, v in zip(KEYS, dual.values)},
                **{f"normalized_g_{k}": float(v) for k, v in zip(KEYS, g/scales)},
                **{f"metric_{k}": v for k, v in metric.items()})
            training_rows.append(row)
            print(f"episode={episode}/{args.episodes} paired_gain={result['economic_improvement_pct']:.6f}% g={g.tolist()} lambda={dual.values.tolist()}", flush=True)
            persist("running")
            if episode % args.eval_every == 0 or episode == args.episodes: evaluate(episode)
            agent.save_checkpoint(model_dir / "last_checkpoint.pth")
            # Finite cap is a stop/report gate, not permission to grow penalties indefinitely.
            if np.any((dual.values >= args.lambda_max-1e-9) & (g/scales > 0)):
                raise RuntimeError("Dual cap reached with violated constraints; report Pareto trade-off, do not increase cap")
        agent.save_actor(model_dir / ("final_actor.pth" if global_step >= cfg.warmup_steps else "final_diagnostic_actor.pth"))
        persist("completed"); plot_learning(training_rows, fixed_rows, args.output_dir)
    except Exception as exc:
        failure = dict(reason=str(exc), attempted_global_step=global_step)
        if "ctrl" in locals(): write_csv(args.output_dir / "failure_trajectory.csv", ctrl.evidence)
        if getattr(exc, "evidence", None): write_csv(args.output_dir / "evaluation_failure_trajectory.csv", exc.evidence)
        persist("aborted", failure)
        plot_learning(training_rows, fixed_rows, args.output_dir)
        raise


def plot_learning(training, fixed, output):
    if not fixed: return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    eps = sorted({r["episode"] for r in fixed})
    for ax, key, label in zip(axes.ravel()[:2], ("economic_improvement_pct", "paired_economic_reward_total"),
                             ("Fixed paired economic improvement (%)", "Fixed paired economic reward sum")):
        for seed in sorted({r["disturbance_seed"] for r in fixed}):
            rows = [r for r in fixed if r["disturbance_seed"] == seed]
            ax.plot([r["episode"] for r in rows], [r[key] for r in rows], alpha=.2, lw=.6)
        ax.plot(eps, [np.mean([r[key] for r in fixed if r["episode"] == e]) for e in eps], color="black")
        ax.axhline(0, color="gray", ls="--"); ax.set_ylabel(label)
    for key in KEYS:
        axes[1, 0].plot([r["episode"] for r in training], [r[f"lambda_next_{key}"] for r in training], label=key)
        axes[1, 1].plot(eps, [max(r[f"g_{key}"] for r in fixed if r["episode"] == e) for e in eps], label=key)
    axes[1, 0].set_ylabel("Adaptive multipliers"); axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_ylabel("Worst validation constraint g (<=0 feasible)")
    axes[1, 1].axhline(0, color="gray", ls="--"); axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel(): ax.set_xlabel("Episode"); ax.grid(alpha=.2)
    fig.suptitle("Paired objective; zero is baseline, not a forced convergence target")
    fig.tight_layout(); fig.savefig(output / "paired_learning_curves.png", dpi=180); plt.close(fig)


def baseline_sensitivity(args):
    env = setup(args)
    fresh(args.output_dir)
    cache = paths_and_baselines(env, args.oracle_seeds, args.output_dir)
    rows = []
    for seed, (path, _, bm, limits) in cache.items():
        records, metric, _ = rollout(env, path, seed, ConstantPolicy(args.supplemental_action))
        rows.append(dict(seed=seed, baseline_identity="supplemental fixed performance-bias action; NOT primary baseline or paper NMPC",
                         raw_action=args.supplemental_action, **paired_result(metric, bm, limits),
                         **{f"primary_{k}": v for k, v in bm.items()},
                         **{f"supplemental_{k}": v for k, v in metric.items()}))
    write_csv(args.output_dir / "baseline_sensitivity.csv", rows)
    save(args.output_dir / "summary.json", dict(protocol=protocol(args, env[0]), rows=rows,
        warning="Predeclared supplementary baseline only; cannot replace primary reward/selection baseline or inflate the main reported gain",
        certification_status=CERTIFICATION_STATUS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("audit", "oracle", "train", "baseline-sensitivity"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--lock-root", type=Path, default=LOCK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--economic-scale", type=float, default=200.)
    parser.add_argument("--oracle-seeds", type=int, nargs="+", default=[420000, 420001, 420002])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(420000, 420010)))
    parser.add_argument("--segment-steps", type=int, choices=(5, 10), default=10)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--search-seed", type=int, default=2019001)
    parser.add_argument("--dual-lr", type=float, default=.1)
    parser.add_argument("--lambda-max", type=float, default=100.)
    parser.add_argument("--constraint-audit", type=Path, default=DEFAULT_OUT / "audit/constraint_audit.json")
    parser.add_argument("--oracle-summary", type=Path, default=DEFAULT_OUT / "oracle10/oracle_summary.json")
    parser.add_argument("--supplemental-action", type=float, nargs=2, default=[.25, 0.])
    args = parser.parse_args()
    args.output_dir = args.output_dir or DEFAULT_OUT / {"oracle": "oracle10", "train": "seed42_100x1000"}.get(args.phase, args.phase)
    if min(args.steps, args.episodes, args.population, args.generations) <= 0:
        raise ValueError("Positive horizons and search budgets required")
    if not np.isfinite(args.economic_scale) or args.economic_scale <= 0:
        raise ValueError("Positive finite economic scale required")
    if not np.isfinite(args.dual_lr) or args.dual_lr <= 0 or not np.isfinite(args.lambda_max) or args.lambda_max <= 0:
        raise ValueError("Positive finite dual parameters required")
    for seeds in (args.oracle_seeds, args.eval_seeds):
        if len(set(seeds)) != len(seeds) or not seeds or any(seed not in range(420000, 420010) for seed in seeds):
            raise ValueError("Use unique development validation seeds 420000..420009; reserved final seeds forbidden")
    if not np.isfinite(args.supplemental_action).all() or np.max(np.abs(args.supplemental_action)) > 1:
        raise ValueError("Supplemental raw action must be in [-1,1]^2")
    {"audit": audit, "oracle": oracle, "train": train, "baseline-sensitivity": baseline_sensitivity}[args.phase](args)


if __name__ == "__main__":
    main()
