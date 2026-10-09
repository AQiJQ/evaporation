"""Read-only frozen-policy/critic diagnosis. No replay additions or SAC updates.

All sampling budgets, diagnostic thresholds and MC protocol are implementation
choices for reproduction, not specified in the paper. Missing replay/critics
are unavailable, never replaced by evaluation proxies presented as replay.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .controlled_invariant_error_set import contains
from .train import observation, run_episode
from .zanon2019_authority import AuthorityController
from .zanon2019_benchmark import make_setup, make_agent, load_actor, sample_disturbance_path, write_csv
from .zanon2019_reward_audit import apply_calibration, DEFAULT_OUTPUT as CALIBRATION_DIR

REPO = Path(__file__).parent
SOURCE = REPO / "evaporation_safe_sac/outputs_zanon2019_alpha020_rewardcal_seed42_100x1000"
OUTPUT = REPO / "evaporation_safe_sac/outputs_zanon2019_sac_optimization_audit"


def save_json(path, data):
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(data, indent=2, default=encode, allow_nan=False), encoding="utf-8")


def distribution(values):
    x = np.asarray(values, dtype=float)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def correlation(x, y, rank=False):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12: return None
    def ranks(v):
        v = np.asarray(v)
        order = np.argsort(v, kind='stable')
        out = np.empty(len(v), dtype=float)
        start = 0
        while start < len(v):
            end = start + 1
            while end < len(v) and v[order[end]] == v[order[start]]:
                end += 1
            out[order[start:end]] = (start + end - 1) / 2
            start = end
        return out
    if rank: x, y = ranks(x), ranks(y)
    return float(np.corrcoef(x, y)[0, 1])


def paired_policy_summary(rows):
    """Average action realizations within each disturbance seed first."""
    output = []
    for ep in sorted({r['episode'] for r in rows}):
        seed_rows = []
        for seed in sorted({r['seed'] for r in rows if r['episode'] == ep}):
            selected = [r for r in rows if r['episode'] == ep and r['seed'] == seed]
            kinds = {k: [r for r in selected if r['policy'] == k] for k in ('zero','deterministic','stochastic')}
            if not all(kinds.values()) or not all(r['completed'] for r in selected):
                continue
            seed_rows.append({k: {metric: float(np.mean([r[metric] for r in values]))
                                  for metric in ('environment_return','discounted_entropy_objective','J_econ','X2_IAE','P2_IAE','P100_TV','F200_TV')}
                              for k, values in kinds.items()})
        if not seed_rows: continue
        item = {'episode':ep,'paired_disturbance_seed_count':len(seed_rows)}
        for metric in seed_rows[0]['zero']:
            # Deterministic/zero actions are Dirac policies, not Gaussian policies
            # with a defined comparable differential entropy bonus.
            if metric == 'discounted_entropy_objective': continue
            for left, right in (('stochastic','deterministic'),('deterministic','zero'),('stochastic','zero')):
                delta = np.array([s[left][metric]-s[right][metric] for s in seed_rows])
                item[f'{metric}_{left}_minus_{right}'] = {
                    'mean':float(delta.mean()), 'sample_std':float(delta.std(ddof=1)) if len(delta)>1 else None,
                    'normal_approx_95pct_CI': [float(delta.mean()-1.96*delta.std(ddof=1)/np.sqrt(len(delta))),
                                               float(delta.mean()+1.96*delta.std(ddof=1)/np.sqrt(len(delta)))] if len(delta)>1 else None,
                    'min':float(delta.min()), 'max':float(delta.max())}
        output.append(item)
    return output


class AuditEnv:
    """Diagnostic-only frozen dynamics; step convention validated vs run_episode."""
    def __init__(self, cfg, model, design, omega, domain):
        self.cfg, self.model, self.design, self.omega = cfg, model, design, omega
        self.ctrl = AuthorityController(cfg, model, design, omega, domain)
        self.x = cfg.robust_economic_reference_state.copy()
        self.ctrl.reset(self.x)
        self.previous_u = design.v_ref.copy()
        self.previous_residual = np.zeros(2)
        self.w_est = np.zeros(2)
        self.reference_cost = model.economic_cost(cfg.linearization_state,
            cfg.linearization_input, cfg.disturbance_nominal)

    def snapshot(self):
        return {"x": self.x.copy(), "z": self.ctrl.inner.z.copy(),
                "previous_u": self.previous_u.copy(), "previous_residual": self.previous_residual.copy(),
                "w_est": self.w_est.copy(), "was_in_omega": self.ctrl.was_in_omega}

    def restore(self, s):
        for k in ("x", "previous_u", "previous_residual", "w_est"):
            setattr(self, k, s[k].copy())
        self.ctrl.inner.z = s["z"].copy()
        self.ctrl.was_in_omega = s["was_in_omega"]

    def obs(self):
        return observation(self.model, self.ctrl, self.x, self.previous_u, self.w_est)

    def step(self, action, disturbance):
        c, m, d = self.cfg, self.model, self.design
        before = self.x.copy()
        u, info = self.ctrl.act(self.x, action)
        xn = m.step(self.x, u, disturbance)
        nn = m.normalized_state(xn)
        w = nn - (d.a @ m.normalized_state(self.x) + d.b @ m.normalized_input(u) + d.affine)
        beta = c.disturbance_estimate_ema
        self.w_est = beta * self.w_est + (1 - beta) * w
        applied = np.asarray(info["applied_residual"])
        cost = m.economic_cost(xn, u, disturbance)
        loss_x = ((xn - c.robust_economic_reference_state) / c.state_scale) ** 2
        du = (u - m.physical_input(self.previous_u)) / c.input_scale
        lo, hi = [m.physical_input(v) for v in (d.robust_input_lower, d.robust_input_upper)]
        eta = np.abs(u - (lo + hi) / 2) / ((hi - lo) / 2)
        sx = (np.maximum(c.state_lower - xn, 0) + np.maximum(xn - c.state_upper, 0)) / c.state_scale
        su = (np.maximum(c.input_lower - u, 0) + np.maximum(u - c.input_upper, 0)) / c.input_scale
        sb, ub = bool(np.any(sx > 1e-8 / c.state_scale)), bool(np.any(su > 1e-8 / c.input_scale))
        qp = not info["qp_feasible"]
        e = nn - self.ctrl.z
        rb = not contains(d.rpi_boundary, e, tol=1e-8)
        rr = np.where(e >= 0, e / np.maximum(d.rpi_support_upper, 1e-12),
                      -e / np.maximum(d.rpi_support_lower, 1e-12))
        components = {
            "economic_reward": (self.reference_cost - cost) / c.reward_cost_scale,
            "state_recovery_penalty": c.paper2016_state_recovery_penalty_weight * float(loss_x.sum()),
            "p100_move_penalty": c.paper2016_p100_move_penalty_weight * float(du[0] ** 2),
            "f200_move_penalty": c.paper2016_f200_move_penalty_weight * float(du[1] ** 2),
            "saturation_penalty": c.paper2016_saturation_penalty_weight * float(np.maximum(eta - .9, 0).dot(np.maximum(eta - .9, 0))),
            "projection_penalty": c.projection_penalty_weight * float(info["projection_gap"]) ** 2,
            "mapping_penalty": c.feasible_action_mapping_penalty_weight * float(info["feasible_action_mapping_gap"]) ** 2,
            "move_penalty": c.input_move_penalty_weight * float(np.sum((applied - self.previous_residual) ** 2)),
            "state_violation_penalty": c.state_violation_event_penalty * sb,
            "input_violation_penalty": c.input_violation_event_penalty * ub,
            "qp_infeasible_penalty": c.qp_infeasible_event_penalty * qp,
            "state_excess_penalty": c.state_excess_square_weight * float(sx @ sx),
            "input_excess_penalty": c.input_excess_square_weight * float(su @ su),
        }
        reward = components["economic_reward"] - sum(v for k, v in components.items() if k != "economic_reward")
        rpi_penalty = c.rpi_violation_event_penalty * rb + c.rpi_excess_square_weight * float(np.sum(np.maximum(rr - 1, 0) ** 2))
        ob = not contains(self.omega, nn - d.z_ref, tol=1e-8)
        robust = bool(np.any(nn < d.robust_state_lower - 1e-8) or np.any(nn > d.robust_state_upper + 1e-8)
                      or np.any(info["actual_norm"] < d.robust_input_lower - 1e-8)
                      or np.any(info["actual_norm"] > d.robust_input_upper + 1e-8))
        self.x, self.previous_u, self.previous_residual = xn, m.normalized_input(u), applied.copy()
        return {"state": before, "next_state": xn, "input": np.array(u), "reward": float(reward),
                "raw_eval_reward": float(reward - rpi_penalty), "cost": float(cost),
                "X2_margin": float(xn[0] - c.state_lower[0]), "physical_state": int(sb),
                "physical_input": int(ub), "QP_infeasible": int(qp), "Omega_exit": int(ob),
                "robust_region": int(robust), **components}


class Policy:
    def __init__(self, agent, kind, seed):
        self.agent, self.kind = agent, kind
        self.generator = torch.Generator(device=agent.device).manual_seed(seed)
    def action(self, obs):
        if self.kind == "zero": return np.zeros(2), 0.
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.agent.device)[None]
            mean, logstd = self.agent.actor(t)
            if self.kind == "deterministic": return torch.tanh(mean)[0].cpu().numpy(), 0.
            eps = torch.randn(mean.shape, generator=self.generator, device=self.agent.device)
            raw = mean + logstd.exp() * eps
            a = torch.tanh(raw)
            normal = torch.distributions.Normal(mean, logstd.exp())
            logp = (normal.log_prob(raw) - torch.log(1 - a * a + 1e-6)).sum()
            return a[0].cpu().numpy(), float(logp)


def frozen_rollout(env, policy, path, gamma, alpha, *, snapshot=None, first_action=None, capture=False):
    if snapshot is not None: env.restore(snapshot)
    rows, snapshots, actions = [], [], []
    hard, soft = 0., 0.
    for k, disturbance in enumerate(path):
        obs = env.obs()
        if capture: snapshots.append((env.snapshot(), obs.copy()))
        if k == 0 and first_action is not None: action, logp = first_action, 0.
        else: action, logp = policy.action(obs)
        result = env.step(action, disturbance)
        actions.append(action)
        hard += gamma ** k * result["reward"]
        # Conditional Q(s,a): first action fixed, no first entropy bonus.
        soft += gamma ** k * (result["reward"] - (alpha * logp if k > 0 or first_action is None else 0))
        rows.append(result)
        if result["physical_state"] or result["physical_input"] or result["QP_infeasible"] or result["Omega_exit"]:
            break  # retain anomaly; never call this aborted rollout a full-horizon return
    states = np.array([r["state"] for r in rows]); u = np.array([r["input"] for r in rows])
    error = np.abs(states - env.cfg.robust_economic_reference_state)
    metrics = {"steps": len(rows), "requested_steps": len(path), "completed": len(rows) == len(path),
        "environment_return": float(sum(r["reward"] for r in rows)),
        "raw_evaluation_return_including_RPI": float(sum(r["raw_eval_reward"] for r in rows)),
        "discounted_environment_return": hard, "discounted_entropy_objective": soft,
        "J_econ": float(sum(r["cost"] for r in rows)),
        "X2_IAE": float(error[:, 0].sum()), "P2_IAE": float(error[:, 1].sum()),
        "P100_TV": float(np.abs(np.diff(u[:, 0])).sum()), "F200_TV": float(np.abs(np.diff(u[:, 1])).sum()),
        **{key: sum(r[key] for r in rows) for key in ("physical_state", "physical_input", "QP_infeasible", "Omega_exit", "robust_region")}}
    metrics['entropy_objective_valid'] = policy.kind == 'stochastic' if hasattr(policy,'kind') else False
    return metrics, snapshots, rows, actions


def q_values(agent, obs, actions):
    actions = np.asarray(actions, dtype=np.float32)
    o = torch.as_tensor(np.repeat(np.asarray(obs)[None], len(actions), axis=0), device=agent.device)
    a = torch.as_tensor(actions, device=agent.device)
    with torch.no_grad():
        q1, q2 = agent.q1(o, a), agent.q2(o, a)
        t1, t2 = agent.q1_target(o, a), agent.q2_target(o, a)
    return np.c_[q1.cpu().numpy(), q2.cpu().numpy(), torch.minimum(q1, q2).cpu().numpy(),
                 torch.minimum(t1, t2).cpu().numpy()]


def validate_step(env_factory, cfg, model, design, omega, domain):
    """Official evaluation path is the independent reference, not our own reward."""
    cfg_small = copy.copy(cfg); cfg_small.steps_per_episode = 12
    path, _ = sample_disturbance_path(cfg, "zanon2019_stochastic", 420000, 12, [2, 1, 8, 5], 20, 50)
    class Constant:
        def select_action(self, obs, deterministic=True): return np.array([-.3, .2], dtype=np.float32)
    _, official, _ = run_episode(cfg_small, model,
        AuthorityController(cfg, model, design, omega, domain), Constant(), None, np.random.default_rng(1),
        training=False, global_step=0, disturbance_trajectory=path,
        initial_state_override=cfg.robust_economic_reference_state)
    env = env_factory(); output = []
    for k, reference in enumerate(official):
        row = env.step(np.array([-.3, .2], dtype=np.float32), path[k])
        for key in row.keys() & reference.keys():
            if np.isscalar(row[key]) and key.endswith(("penalty", "reward")):
                error = abs(row[key] - reference[key])
                # Official eval reward intentionally contains the RPI terms.
                if key == "reward": continue
                if error > 1e-8: raise RuntimeError(f"Diagnostic reward mismatch: {key} {error}")
                output.append({"source": "official_eval_trace_not_replay", "step": k, "component": key,
                               "stored": reference[key], "recomputed": row[key], "abs_error": error})
        expected = reference["total_reward"] + reference["rpi_violation_event_penalty"] + reference["rpi_excess_penalty"]
        if abs(row["reward"] - expected) > 1e-8 or np.max(np.abs(row["input"] - reference["control"])) > 1e-8:
            raise RuntimeError("Diagnostic replay-equivalent reward/action mismatch")
        output.append({"source": "official_eval_trace_not_replay", "step": k, "component": "replay_equivalent",
                       "stored": expected, "recomputed": row["reward"], "abs_error": abs(row["reward"] - expected)})
    return output


def plot_reports(out, policy_rows, mc_rows, pairs, disagreements, surfaces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def finish(name, title):
        plt.title(title); plt.tight_layout(); plt.savefig(out / name, dpi=150); plt.close()
    plt.figure()
    kinds = ["zero", "deterministic", "stochastic"]
    values = [[r["environment_return"] for r in policy_rows if r["policy"] == k and r["completed"]] for k in kinds]
    if all(values): plt.boxplot(values, tick_labels=kinds)
    plt.ylabel("Environment return (NO entropy, RPI excluded)")
    finish("deterministic_vs_stochastic_return.png", "Frozen policy evaluation; checkpoints shown in CSV")
    plt.figure()
    good = [r for r in mc_rows if r["all_MC_completed"]]
    if good: plt.scatter([r["mc_soft_return"] for r in good], [r["Qmin"] for r in good], s=12)
    plt.xlabel("Finite-horizon conditional soft MC return"); plt.ylabel("Predicted soft Q")
    finish("critic_predicted_vs_actual_return.png", "Q vs MC; horizon/tail limitations in summary")
    plt.figure()
    if pairs: plt.scatter([r["delta_Q_pred"] for r in pairs], [r["delta_return_soft"] for r in pairs])
    plt.axhline(0, color="gray"); plt.axvline(0, color="gray")
    plt.xlabel("Actor minus zero: soft Q"); plt.ylabel("Actor minus zero: conditional soft MC")
    finish("actor_vs_zero_predicted_actual.png", "Same stochastic continuation / common random numbers")
    plt.figure()
    for name in ("zero", "actor", "sample"):
        v = [r["disagreement"] for r in disagreements if r["action"] == name]
        if v: plt.hist(v, bins=25, alpha=.4, label=name)
    plt.legend(); finish("Q1_Q2_disagreement.png", "Frozen episode100 critics")
    plt.figure(); plt.text(.5, .5, "Replay snapshot unavailable\nEvaluation actions are NOT replay coverage", ha="center")
    plt.axis("off"); finish("replay_action_distribution.png", "Unavailable replay evidence")
    for name, surface in surfaces.items():
        plt.figure(); plt.contourf(surface["axis"], surface["axis"], surface["Q"], levels=30); plt.colorbar(label="min(Q1,Q2)")
        for label, point in (("zero", [0, 0]), ("actor", surface["actor"]), ("max-Q", surface["max_action"])):
            plt.scatter(*point, label=label)
        plt.legend(); plt.xlabel("Raw P100 actor action"); plt.ylabel("Raw F200 actor action")
        finish(f"critic_Q_surface_{name}.png", "Raw action Q surface; alpha=.2 original mapping")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", nargs="+", type=int, default=[100, 90, 80])
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=list(range(420000, 420010)))
    parser.add_argument("--stochastic-repeats", type=int, default=20)
    parser.add_argument("--prediction-states", type=int, default=512)
    parser.add_argument("--mc-states", type=int, default=12)
    parser.add_argument("--mc-repeats", type=int, default=20)
    parser.add_argument("--mc-horizon", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=1000, help="Evaluation horizon; shorter is explicitly diagnostic-only")
    parser.add_argument("--surface-grid", type=int, default=41, help="41 for formal audit; small grid only for smoke tests")
    args = parser.parse_args()
    if min(args.steps, args.stochastic_repeats, args.mc_states, args.mc_repeats, args.prediction_states, args.mc_horizon) < 1:
        raise ValueError("Positive diagnostic budgets required")
    if set(args.validation_seeds) - set(range(420000, 420010)):
        raise ValueError("Only original validation seeds allowed; held-out test remains untouched")
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise RuntimeError("Choose fresh diagnostic output")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)  # diagnostic host runtime only, not a training parameter
    manifest = json.loads((args.source_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    geometry = list(manifest["frozen_geometry_sources"])
    cfg, model, design, omega, domain, _ = make_setup(args.steps, 42, *geometry)
    cfg.stochastic_residual_scale = .2
    calibration = Path(manifest["reward_calibration_source"])
    if not calibration.is_absolute(): calibration = REPO.parent / calibration
    apply_calibration(cfg, calibration, geometry_sources=geometry)
    factory = lambda: AuditEnv(cfg, model, design, omega, domain)
    rows_consistency = validate_step(factory, cfg, model, design, omega, domain)
    write_csv(args.output_dir / "reward_consistency.csv", rows_consistency + [{
        "source": "replay", "status": "unavailable", "reason": "No saved replay snapshot; cannot validate stored training rewards"}])
    write_csv(args.output_dir / "done_mask_audit.csv", [{"status": "source_verified_snapshot_unavailable",
        "horizon_done": "done=True at step999 in original 1000-step episodes", "bootstrap_mask": "1-done",
        "terminated_vs_truncated": "not separately represented", "reward_normalization": "none beyond economic_scale=200 and fixed shaping",
        "target": "r+gamma*(1-done)*(min(target_Q1,target_Q2)-alpha*logp)",
        "interpretation": "consistent finite episodic terminal convention; if intended continuing-task truncation, no-bootstrap needs clarification, not an established bug"}])
    for name in ("replay_action_coverage.csv", "ood_action_audit.csv"):
        write_csv(args.output_dir / name, [{"status": "unavailable", "reason": "Replay not serialized by save_checkpoint; no training transition trace; evaluation is not a replay proxy"}])
    agent = make_agent(cfg, args.device)
    checkpoint = args.source_dir / "models/final_checkpoint.pth"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key, value in saved['cfg'].items():
        if hasattr(agent.cfg, key) and getattr(agent.cfg, key) != value:
            raise RuntimeError(f"Diagnostic SAC config differs from saved model: {key}")
    agent.load_checkpoint(checkpoint, load_optimizers=False)
    if agent.policy_output_scale != 1.:
        raise RuntimeError('This audit requires the original unit output scale; do not silently change policy semantics')
    gamma, entropy_alpha = agent.cfg.gamma, agent.alpha
    protected = [checkpoint, *map(Path, geometry), calibration,
                 *(REPO/name for name in ('sac.py','train.py','zanon2019_train.py','zanon2019_authority.py')),
                 *(args.source_dir/f'models/evaluation/episode_{ep:04d}_actor.pth' for ep in args.episodes)]
    fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    started = time.perf_counter()
    policy_rows, pool = [], []
    for ep in args.episodes:
        path_actor = args.source_dir / f"models/evaluation/episode_{ep:04d}_actor.pth"
        load_actor(agent, path_actor)
        actor_data = torch.load(path_actor, map_location="cpu", weights_only=False)
        alpha = float(actor_data["alpha"])
        for seed in args.validation_seeds:
            path, _ = sample_disturbance_path(cfg, "zanon2019_stochastic", seed, args.steps, [2, 1, 8, 5], 20, 50)
            for kind in ("zero", "deterministic", "stochastic"):
                for repeat in range(args.stochastic_repeats if kind == "stochastic" else 1):
                    result, snaps, _, _ = frozen_rollout(factory(), Policy(agent, kind, seed*100+repeat), path,
                        gamma, alpha, capture=(ep == 100 and kind in ("zero", "deterministic")))
                    policy_rows.append({"episode": ep, "seed": seed, "policy": kind,
                        "action_realization": repeat, "entropy_alpha": alpha, **result})
                    if snaps:
                        for step in np.linspace(0, len(snaps)-1, min(args.prediction_states, len(snaps)), dtype=int):
                            snap, obs = snaps[step]
                            pool.append({"snapshot": snap, "obs": obs, "source_policy": kind,
                                "source_seed": seed, "source_step": int(step), "X2_margin": snap["x"][0]-25})
                    if repeat % 5 == 0:
                        write_csv(args.output_dir / "deterministic_vs_stochastic_eval.csv", policy_rows)
                        print(f"eval episode={ep} seed={seed} policy={kind} repeat={repeat} completed={result['completed']}", flush=True)
    write_csv(args.output_dir / "deterministic_vs_stochastic_eval.csv", policy_rows)
    # Only matching episode100 critics are used; never attach these to 80/90.
    agent.load_checkpoint(checkpoint, load_optimizers=False)
    if not pool: raise ValueError("Include episode100 for matching critic audit")
    indices = np.linspace(0, len(pool)-1, min(args.prediction_states, len(pool)), dtype=int)
    states = [pool[i] for i in indices]
    save_json(args.output_dir/'evaluation_state_contexts.json',states)
    disagreements, prediction_rows = [], []
    actions_by_state = []
    for i, state in enumerate(states):
        actor, _ = Policy(agent, "deterministic", i).action(state["obs"])
        sample, _ = Policy(agent, "stochastic", i+80000).action(state["obs"])
        actions = [(f"lambda_{v:g}", v*actor) for v in (0, .1, .25, .5, .75, 1)] + [("sample", sample)]
        actions_by_state.append(actions)
        q = q_values(agent, state["obs"], [a for _, a in actions])
        for (name, a), values in zip(actions, q):
            row = {"state_id": i, "state_source": "fixed evaluation regenerated context NOT replay",
                "source_policy": state["source_policy"], "source_seed": state["source_seed"], "source_step": state["source_step"],
                "action": name, "raw_action_0": a[0], "raw_action_1": a[1],
                "Q1": values[0], "Q2": values[1], "Qmin": values[2], "target_Qmin": values[3]}
            prediction_rows.append(row)
            if name in ("lambda_0", "lambda_1", "sample"):
                disagreements.append({**row, "action": {"lambda_0":"zero", "lambda_1":"actor"}.get(name,name),
                                      "disagreement": abs(values[0]-values[1])})
    write_csv(args.output_dir / "critic_predictions.csv", prediction_rows)
    write_csv(args.output_dir / "critic_q_disagreement.csv", disagreements)
    mc_rows, pairs, gradient_rows = [], [], []
    chosen = np.linspace(0, len(states)-1, min(args.mc_states, len(states)), dtype=int)
    for i in chosen:
        state = states[i]; evaluated = {}; samples = {}
        # Branch from full controller state, previous FINAL input, EMA mismatch.
        # Common random numbers for all first actions, stochastic continuation.
        for name, action in actions_by_state[i]:
            estimates, one_steps, completed = [], [], []
            remaining = 1000 - state['source_step']
            branch_horizon = min(args.mc_horizon, remaining)
            for repeat in range(args.mc_repeats):
                ds = 9000000 + int(i)*100 + repeat
                path, _ = sample_disturbance_path(cfg, "zanon2019_stochastic", ds, args.mc_horizon, [2,1,8,5],20,50)
                result, _, trace, _ = frozen_rollout(factory(), Policy(agent,"stochastic",ds+1234), path[:branch_horizon],
                    gamma, entropy_alpha, snapshot=state["snapshot"], first_action=action)
                estimates.append(result); one_steps.append(trace[0]); completed.append(result["completed"])
            pred = next(r for r in prediction_rows if r["state_id"] == i and r["action"] == name)
            row = {**pred, "MC_repeats": args.mc_repeats, "MC_horizon": args.mc_horizon,
                "actual_branch_horizon":branch_horizon,"remaining_training_episode_steps":remaining,
                "reaches_original_terminal":branch_horizon==remaining,
                "gamma_to_horizon": gamma**branch_horizon,
                "all_MC_completed": all(completed), "safety_aborted_MC_count": completed.count(False),
                "mc_environment_return": float(np.mean([r['discounted_environment_return'] for r in estimates])),
                "mc_soft_return": float(np.mean([r['discounted_entropy_objective'] for r in estimates])),
                "one_step_reward": float(np.mean([r['reward'] for r in one_steps])),
                "one_step_X2_margin": float(np.mean([r['X2_margin'] for r in one_steps])),
                "one_step_cost": float(np.mean([r['cost'] for r in one_steps])),
                "one_step_move_penalty": float(np.mean([r['p100_move_penalty']+r['f200_move_penalty'] for r in one_steps]))}
            row['soft_prediction_error'] = row['Qmin'] - row['mc_soft_return']
            evaluated[name] = row; mc_rows.append(row)
            samples[name] = np.array([r['discounted_entropy_objective'] for r in estimates])
        zero, actor = evaluated['lambda_0'], evaluated['lambda_1']
        pairs.append({"state_id": int(i), "delta_Q_pred": actor['Qmin']-zero['Qmin'],
            "delta_return_actual": actor['mc_environment_return']-zero['mc_environment_return'],
            "delta_return_soft": actor['mc_soft_return']-zero['mc_soft_return'],
            "delta_one_step_reward": actor['one_step_reward']-zero['one_step_reward'],
            "delta_one_step_X2_margin":actor['one_step_X2_margin']-zero['one_step_X2_margin'],
            "delta_one_step_cost":actor['one_step_cost']-zero['one_step_cost'],
            "delta_one_step_move_penalty":actor['one_step_move_penalty']-zero['one_step_move_penalty'],
            "paired_soft_difference_SE":float((samples['lambda_1']-samples['lambda_0']).std(ddof=1)/np.sqrt(args.mc_repeats)) if args.mc_repeats>1 else None,
            "comparable_completed_MC": zero['all_MC_completed'] and actor['all_MC_completed']})
        ot = torch.as_tensor(state['obs'][None],device=agent.device)
        at = torch.zeros((1,2),device=agent.device,requires_grad=True)
        q = torch.minimum(agent.q1(ot,at),agent.q2(ot,at))
        gradient = torch.autograd.grad(q.sum(), at)[0].detach().cpu().numpy()[0]
        # Actual finite-difference first-action directions, same continuation.
        local = []
        for axis in range(2):
            a = np.zeros(2); a[axis]=np.sign(gradient[axis])*.05
            path,_=sample_disturbance_path(cfg,'zanon2019_stochastic',9000000+int(i)*100,min(args.mc_horizon,1000-state['source_step']),[2,1,8,5],20,50)
            result,_,_,_=frozen_rollout(factory(),Policy(agent,'stochastic',9000000+int(i)*100+1234),path,
                gamma,entropy_alpha,snapshot=state['snapshot'],first_action=a)
            env0,_,_,_=frozen_rollout(factory(),Policy(agent,'stochastic',9000000+int(i)*100+1234),path,
                gamma,entropy_alpha,snapshot=state['snapshot'],first_action=np.zeros(2))
            local.append(result['discounted_environment_return']-env0['discounted_environment_return'])
        gradient_rows.append({'state_id':int(i),'state_X2':state['snapshot']['x'][0], 'state_P2':state['snapshot']['x'][1],
            'dQ_da0_at_zero':gradient[0],'dQ_da1_at_zero':gradient[1],
            'actual_return_direction_0':local[0],'actual_return_direction_1':local[1],
            'scope':'single-CRN local-direction diagnostic, not precise gradient or actor-error proof'})
        write_csv(args.output_dir/'critic_calibration.csv',mc_rows)
        write_csv(args.output_dir/'actor_vs_zero_action.csv',pairs)
        write_csv(args.output_dir/'actor_gradient_audit.csv',gradient_rows)
        print(f"MC state={i} completed; horizon={args.mc_horizon} tail={gamma**args.mc_horizon:.3g}",flush=True)
    representatives = {'nominal':min(states,key=lambda s:np.linalg.norm(s['snapshot']['x']-cfg.robust_economic_reference_state)),
        'near_boundary':min(states,key=lambda s:s['X2_margin'])}
    representative_availability = {}
    for label, lower, upper in (('margin_above_0p3',.3,np.inf),('margin_approx_0p2',.175,.225),('margin_approx_0p1',.075,.125)):
        candidates = [s for s in states if lower <= s['X2_margin'] <= upper]
        representative_availability[label] = bool(candidates)
        if candidates: representatives[label] = candidates[len(candidates)//2]
    lowers = []
    for s in states:
        env=factory();env.restore(s['snapshot']);base,_=env.ctrl.act(env.x,np.zeros(2))
        env.restore(s['snapshot']);a,_=Policy(agent,'deterministic',0).action(s['obs']);actual,_=env.ctrl.act(env.x,a)
        lowers.append((actual[0]-base[0],s))
    representatives['most_negative_P100'] = min(lowers,key=lambda v:v[0])[1]
    surfaces={}; grid_rows=[]
    axis=np.linspace(-1,1,args.surface_grid); xx,yy=np.meshgrid(axis,axis); grid=np.c_[xx.ravel(),yy.ravel()]
    for name,s in representatives.items():
        values=q_values(agent,s['obs'],grid)[:,2]
        actor,_=Policy(agent,'deterministic',0).action(s['obs'])
        surfaces[name]={'axis':axis,'Q':values.reshape(args.surface_grid,args.surface_grid),'actor':actor,'max_action':grid[values.argmax()]}
        path,_=sample_disturbance_path(cfg,'zanon2019_stochastic',420000,1,[2,1,8,5],20,50)
        for a,v in zip(grid,values):
            env=factory();env.restore(s['snapshot']);one=env.step(a,path[0]);u=one['input']
            grid_rows.append({'representative':name,'X2_margin':s['X2_margin'],'raw_a0':a[0],'raw_a1':a[1],
                'Qmin':v,'applied_P100':u[0],'applied_F200':u[1],'qp_feasible':not one['QP_infeasible'],
                'one_step_reward':one['reward'],'one_step_cost':one['cost'],'one_step_X2_margin':one['X2_margin']})
        print(f'action surface {name} completed',flush=True)
    write_csv(args.output_dir/'critic_action_grid.csv',grid_rows)
    plot_reports(args.output_dir,policy_rows,mc_rows,pairs,disagreements,surfaces)
    valid=[r for r in mc_rows if r['all_MC_completed']]
    pred=[r['Qmin'] for r in valid];actual=[r['mc_soft_return'] for r in valid]
    errors=np.array(pred)-np.array(actual)
    summary={'training_performed':False,'source':str(args.source_dir),'budgets':vars(args),
        'matching_critics':{'100':'available','90':'unavailable','80':'unavailable'},
        'replay_snapshot':'unavailable','reward_snapshot_validation':'unavailable; official eval-path check passed',
        'gamma':gamma,'entropy_alpha_episode100':entropy_alpha,'entropy_excluded_from_performance':True,
        'MC_Q_convention':'first action fixed; entropy only on stochastic continuation from step1; stop at original episode terminal; no critic tail bootstrap',
        'MC_tail_factor':gamma**args.mc_horizon,'MC_scope':'finite-episode conditional soft Q; observation does not encode remaining time, so time aliasing can also affect calibration; shortened nonterminal branches are truncated estimates',
        'full_requested_budget':args.steps==1000 and args.mc_horizon==1000 and len(args.validation_seeds)==10 and args.stochastic_repeats>=20 and args.mc_repeats>=20 and args.surface_grid==41 and set(args.episodes)=={80,90,100},
        'paired_policy_comparison':paired_policy_summary(policy_rows),
        'representative_availability':representative_availability,
        'elapsed_seconds':time.perf_counter()-started,
        'critic_calibration':{'MAE':float(np.abs(errors).mean()),'RMSE':float(np.sqrt(np.mean(errors**2))),
            'bias':float(errors.mean()),'Pearson':correlation(pred,actual),'Spearman':correlation(pred,actual,True)} if valid else None,
        'quadrants':{f'pred_{p}_actual_{a}':int(sum((r['delta_Q_pred']>0)==p and (r['delta_return_soft']>0)==a
                       for r in pairs if r['comparable_completed_MC'])) for p in (True,False) for a in (True,False)},
        'hard_environment_quadrants':{f'pred_{p}_actual_{a}':int(sum((r['delta_Q_pred']>0)==p and (r['delta_return_actual']>0)==a
                       for r in pairs if r['comparable_completed_MC'])) for p in (True,False) for a in (True,False)},
        'disagreement':{name:distribution([r['disagreement'] for r in disagreements if r['action']==name]) for name in ('zero','actor','sample')},
        'replay_coverage_case_C':'unavailable','OOD_case_D':'unavailable without replay',
        'Case_A':'requires soft-Q/MC ranking review; do not compare soft Q to hard return alone',
        'Case_B':'requires within-disturbance-seed stochastic averages, not pooling action repeats as independent disturbance seeds',
        'Case_E':'no reward convention discrepancy in validated diagnostic path; stored replay unavailable; terminal vs truncation intent unspecified',
        'Case_F':'not established before calibration/coverage evidence',
        'low_margin_representative':representatives['near_boundary']['X2_margin'],
        'source_hashes_unchanged':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in fingerprints.items()),
        'certification_status':'empirical_only_under_ECC2019_stochastic_disturbance'}
    save_json(args.output_dir/'summary.json',summary)
    text=['# SAC optimization audit','', 'No training, no changes to reward/SAC/safety/replay.','',
        '## Availability', 'Episode100 critics/targets available; episode80/90 critics and all replay snapshots unavailable.',
        'Evaluation states are not replay coverage. Case C / OOD conclusions cannot be inferred.', '',
        '## Monte Carlo scope', f'Horizon={args.mc_horizon}, gamma^H={gamma**args.mc_horizon:.6g}.',
        'Soft Q is compared with stochastic-continuation soft MC. Environment performance contains NO entropy.',
        'The deterministic/zero rows contain zero bonus placeholders, NOT comparable differential-entropy objectives.',
        'Safety-aborted MC is not treated as a complete-horizon Q calibration sample.', '',
        '## Results', json.dumps(summary['critic_calibration'],ensure_ascii=False),
        json.dumps(summary['quadrants'],ensure_ascii=False), '',
        'Review deterministic_vs_stochastic_eval.csv by episode and disturbance seed before concluding Case B.',
        'No automatic recommendation to tune SAC or add replay data.']
    text += ['', '## Paired deterministic / stochastic comparison',
             'Action repetitions are averaged within each fixed disturbance seed before cross-seed comparison. CI is a normal approximation (implementation choice), not a training-seed confidence interval.',
             json.dumps(summary['paired_policy_comparison'],ensure_ascii=False,indent=2),
             '', '## Completion status',
             f"Full requested budget completed: {summary['full_requested_budget']}",
             'A smoke/short-horizon run is an implementation test, not evidence of Q calibration or any Case A-F.',
             '', '## Twelve diagnostic questions',
             '1. Deterministic vs stochastic: see the paired environment-return differences above; entropy is separate.',
             '2. Actor vs zero ranking: see four quadrants; incomplete MC excluded and sampling uncertainty remains.',
             f"3. Q/MC correlations: {summary['critic_calibration']}",
             '4. Overestimation: bias is reported; a truncated-MC positive bias alone is not a systematic-overestimation proof.',
             '5. Actor exploiting error: compare gradient and actual local direction; single-CRN probes are preliminary only.',
             f"6. Q disagreement: {summary['disagreement']}; sampled actor action is NOT random replay action.",
             '7. Zero-action replay coverage: unavailable (no snapshot).',
             '8. Stored/recomputed replay reward: unavailable; independent official evaluation-path formula test passed.',
             '9. Done mask: finite-terminal convention matches source; continuing-task truncation intent not specified.',
             '10. OOD replay support: unavailable (no snapshot).',
             '11. Cases: A/B require the numerical evidence above; C/D unsupported without replay, E not established, F not established.',
             '12. Next modification: do not choose critic/extraction/replay/initialization/hyperparameters before the full audit.',
             '', '## Frozen source hashes', json.dumps(fingerprints,indent=2),
             '', 'All diagnostic budgets, state bands and finite differences are implementation choices for reproduction.']
    (args.output_dir/'zanon2019_sac_optimization_audit.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='budgets'},default=str),flush=True)


if __name__=='__main__':
    main()
