"""Opt-in local 300x1000 extension of the unchanged Strong paired v2 recipe.

No old run, reward calibration, production source or safety geometry is edited.
This is a fresh seed42 run, not an actor-only resume from episode100. Gaussian
validation is empirical only. `preflight` and `plot` never perform training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from . import zanon2019_paired_experiment as v1
from . import zanon2019_paired_v2 as v2
from . import zanon2019_economic_recovery as recovery
from .zanon2019_benchmark import make_agent, sample_disturbance_path, write_csv, PAPER_VARIANCES_F1_X1_T1_T200
from .zanon2019_paired_v2_objective import EconomicReplay, SafetyDual, SAFETY_KEYS
from .zanon2019_training_report import CERTIFICATION_STATUS
from .sac import set_seed
from .train import ZeroResidualPolicy, run_episode

ROOT = v1.REPO / "evaporation_safe_sac/paired_economic_v2_300"
DEFAULT_OUTPUT = ROOT / "seed42_300x1000"
EPISODES, STEPS, SEED, EVAL_EVERY = 300, 1000, 42, 5


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("phase", choices=("preflight", "train", "plot"))
    result.add_argument("--seed", type=int, choices=(SEED,), default=SEED)
    result.add_argument("--episodes", type=int, choices=(EPISODES,), default=EPISODES)
    result.add_argument("--steps", type=int, choices=(STEPS,), default=STEPS)
    result.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return result


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path == ROOT.resolve():
        raise ValueError(f"Use a new run directory below {ROOT}; preserve historical experiments")
    return path


def episode_seed(episode):
    # Identical first 100 shock/disturbance paths and action RNGs to old v2.
    return SEED * 1000000 + episode


def frozen_hashes():
    paths = [p for p in v1.REPO.glob("*.py")]
    paths += [v2.ROOT / "smoothness_regularization_config.json",
              v2.ROOT / "reward_scale_audit.json",
              recovery.DEFAULT_DESIGN, recovery.DEFAULT_OMEGA]
    return {str(path): v1.digest(path) for path in sorted(paths)}


def preflight(args):
    output = output_path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Preserve the existing run; select a fresh --output-dir: {output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use .venv-cuda Python; no silent CPU fallback")
    v2.require_tests(v2.ROOT)
    config = v2.locked_config(v2.ROOT)
    oracle = v2.load(v2.ROOT / "oracle_v2_summary.json")
    if oracle["v2_joint_witness_count"] <= 0 or oracle["smoothness_config_sha256"] != v1.digest(
            v2.ROOT / "smoothness_regularization_config.json"):
        raise RuntimeError("Unchanged v2 oracle/reward evidence required")
    if oracle["reclassification_sha256"] != v1.digest(v2.ROOT / "oracle_v2_reclassification.csv"):
        raise RuntimeError("Oracle reclassification evidence changed")
    for src in oracle["evidence"]:
        if v1.digest(src["path"]) != src["sha256"]:
            raise RuntimeError("Original oracle evidence changed")
    old_run = v2.ROOT / "seed_42"
    if not v2.pilot_gate(v2.load(old_run / "run_summary.json"), v2.load(old_run / "checkpoints.json")):
        raise RuntimeError("Existing seed42 100x1000 pilot must have passed its continuation gate")
    setup_args = recovery.args_for(SEED)
    setup_args.episodes, setup_args.steps = EPISODES, STEPS
    env = v2.guard(setup_args)
    env[0].episodes = EPISODES
    if env[0].steps_per_episode != STEPS or env[0].stochastic_residual_scale != .10:
        raise RuntimeError("Frozen horizon/residual authority mismatch")
    free_gib = shutil.disk_usage(v1.REPO).free / 2**30
    if free_gib < 2:
        raise RuntimeError(f"Only {free_gib:.2f} GiB free; preserve evidence and free space before training")
    print(json.dumps(dict(ready=True, training_performed=False, seed=SEED, episodes=EPISODES,
        steps=STEPS, device=args.device, input_domain="original production U_R; not C2_8",
        alpha=.10, lambda_smooth=config["lambda_smooth"], output_dir=str(output),
        free_disk_GiB=free_gib, warning="Allow space for 610 complete validation trajectories and checkpoints",
        fixed_validation_seeds=list(v2.DEV), independent_final_test=False), allow_nan=False), flush=True)
    return env, config, output, setup_args


def trailing_mean(values, window=10):
    """Trailing full windows only. No future points, padding or invented data."""
    values = np.asarray(values, dtype=float)
    if window < 1 or not np.isfinite(values).all():
        raise ValueError("Positive window and finite recorded returns required")
    result = np.full(len(values), np.nan)
    if len(values) >= window:
        result[window-1:] = np.convolve(values, np.ones(window)/window, mode="valid")
    return result


def learning_data(output):
    training_path, fixed_path = output / "training_log.csv", output / "checkpoints.json"
    training = v1.read_csv(training_path) if training_path.exists() else []
    fixed = v2.load(fixed_path) if fixed_path.exists() else []
    train_rows, eval_rows = [], []
    for row in training:
        total = float(row["episode_return"])
        components = sum(float(row[key]) for key in (
            "economic_reward_component", "smoothness_regularization_component", "lagrangian_constraint_component"))
        if not np.isfinite(total) or not np.isclose(total, components, rtol=0, atol=1e-8):
            raise ValueError("Logged return does not equal the actual paired replay objective")
        train_rows.append((int(row["episode"]), total))
    for row in fixed:
        total = float(row["total_training_reward"])
        if not np.isfinite(total):
            raise ValueError("Nonfinite recorded fixed evaluation return")
        eval_rows.append((int(row["episode"]), total, float(row["mean_economic_improvement_pct"])))
    for rows in (train_rows, eval_rows):
        episodes = [row[0] for row in rows]
        if episodes != sorted(set(episodes)):
            raise ValueError("Duplicate or unordered episodes; do not silently sort away evidence")
    return train_rows, eval_rows


def plot_returns(output):
    """Plot completed recorded episodes only, including partial/aborted runs."""
    output = output_path(output)
    train_rows, eval_rows = learning_data(output)
    if not train_rows and not eval_rows:
        raise ValueError("No recorded return data yet. Start local training first; no synthetic curve")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    status_path = output / "run_summary.json"
    status = v2.load(status_path)["run_state"] if status_path.exists() else "partial"
    specs = [("training_return_learning_curve.png", train_rows, "Actual paired training return", True),
             ("fixed_eval_return_learning_curve.png", eval_rows, "Fixed deterministic mean return (10 development paths)", False)]
    for filename, rows, label, smooth in specs:
        if not rows:
            continue
        x, y = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(x, y, lw=.9, alpha=.5 if smooth else 1, marker=None if smooth else "o", ms=3,
                label="Recorded episode return" if smooth else "Recorded fixed evaluation mean")
        if smooth and len(y) >= 10:
            ax.plot(x, trailing_mean(y), lw=1.8, label="Trailing 10-episode mean")
        ax.axhline(0, color="grey", lw=.8, ls="--", label="Zero-residual paired objective = 0")
        ax.set(xlabel="Training episode", ylabel="Undiscounted sum of paired objective rewards",
               title=f"Seed42 / original geometry / {status}: {label}")
        ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figures / filename, dpi=160); plt.close(fig)
    if eval_rows:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot([r[0] for r in eval_rows], [r[2] for r in eval_rows], marker="o", ms=3)
        ax.axhline(0, color="grey", lw=.8, ls="--")
        ax.set(xlabel="Training episode", ylabel="Mean paired economic improvement (%)",
               title=f"Seed42 / fixed deterministic development evaluation / {status}")
        ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(figures / "fixed_eval_economic_learning_curve.png", dpi=160); plt.close(fig)
    return figures


def train(args):
    env, config, output, setup_args = preflight(args)
    cfg, model, design, omega, domain = env[:5]
    v1.fresh(output)
    set_seed(SEED)
    agent = make_agent(cfg, args.device)
    agent.zero_initialize_residual_mean()
    dual = SafetyDual.create()
    replay = EconomicReplay(23, 2, cfg.replay_capacity, agent.device, dual, config["lambda_smooth"])
    hashes = frozen_hashes()
    model_dir = output / "models"
    model_dir.mkdir()
    manifest = dict(schema="strong_paired_v2_seed42_300", training_seed=SEED,
        episodes=EPISODES, steps=STEPS, eval_every=EVAL_EVERY, device=str(agent.device),
        fresh_initialization=True, resume=False, original_production_UR=True, C2_8_enabled=False,
        protocol=v1.protocol(setup_args, cfg), frozen_SAC_config=vars(agent.cfg),
        frozen_experiment_config=vars(cfg), source_and_evidence_hashes=hashes,
        lambda_smooth=config["lambda_smooth"], fixed_validation_seeds=list(v2.DEV),
        disturbance_seed_rule="42*1000000+episode; local action rng=path_seed+77",
        objective="unchanged v2 paired economics /200 minus fixed excess-move smoothness and safety dual",
        return_definition="undiscounted sum of actual replay-objective components; not legacy diagnostic reward",
        presentation="all fixed evaluations; no economic-only automatic final selection",
        final_test_performed=False, certification_status=CERTIFICATION_STATUS)
    v1.save(output / "experiment_manifest.json", manifest)
    training, fixed, checkpoints, cache = [], [], [], {}
    global_step, max_collapse = 0, 0.

    def persist(state, failure=None):
        write_csv(output / "training_log.csv", training)
        write_csv(output / "fixed_evaluation.csv", fixed)
        write_csv(output / "checkpoint_performance.csv", checkpoints)
        v1.save(output / "checkpoints.json", checkpoints)
        v1.save(output / "run_summary.json", dict(run_state=state, completed_episodes=len(training),
            planned_episodes=EPISODES, global_step=global_step, max_action_collapse_fraction=max_collapse,
            replay_decomposition_checked=True, failure=failure, training_performed=True,
            final_test_performed=False, automatic_next_seed=False, certification_status=CERTIFICATION_STATUS))

    def evaluate(episode):
        nonlocal max_collapse
        actor_path = model_dir / f"episode_{episode:04d}_actor.pth"
        agent.save_actor(actor_path)
        latest = []
        with torch.no_grad():
            for dev, (path, base, _, _) in cache.items():
                print(f"fixed_eval episode={episode}, validation_seed={dev}", flush=True)
                records, _, _ = v1.rollout(env, path, dev, agent)
                metric, raw, series = recovery.full_metrics(base, records, env, replay.weight, dual.values)
                latest.append(dict(episode=episode, global_step=global_step, validation_seed=dev, **metric))
                rows = recovery.trace(base, records, series)
                for row, component in zip(rows, raw):
                    row.update(economic_reward_raw=float(component[0]), smoothness_penalty_raw=float(component[1]),
                               smoothness_contribution=-replay.weight*float(component[1]))
                write_csv(output / "evaluation_trajectories" / f"episode_{episode:04d}" / f"seed_{dev}.csv", rows)
        fixed.extend(latest)
        status = v2.assessment(latest, episode, global_step, cfg.warmup_steps)
        status.update(checkpoint=str(actor_path), metric_source="fixed_paired_validation")
        checkpoints.append(status)
        max_collapse = max(max_collapse, *(r["action_collapse_fraction"] for r in latest))
        persist("running")
        plot_returns(output)
        print("fixed_eval", json.dumps(status, allow_nan=False), flush=True)
        late = [s for s in checkpoints if s["post_warmup"]][-5:]
        if len(late) == 5 and all(s.get("regularizer_fraction") is not None and
                s["regularizer_fraction"] > .10 and s["mean_economic_improvement_pct"] <= 0 for s in late):
            raise RuntimeError("Regularizer scale mismatch; stop, never retune mid-run")

    try:
        cache = v1.paths_and_baselines(env, v2.DEV, output)
        evaluate(0)
        for episode in range(1, EPISODES + 1):
            if hashes != frozen_hashes():
                raise RuntimeError("Frozen source, reward calibration or geometry changed during training")
            path_seed = episode_seed(episode)
            path, _ = sample_disturbance_path(cfg, "zanon2019_stochastic", path_seed, STEPS,
                                             PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
            base, _, _ = v1.rollout(env, path, path_seed, ZeroResidualPolicy())
            ctrl = v1.PairedEvidenceController(cfg, model, design, omega, domain, path)
            replay.bind(ctrl, base)
            dual_used = dual.values.copy()
            stat, records, global_step = run_episode(cfg, model, ctrl, agent, replay,
                np.random.default_rng(path_seed + 77), training=True, global_step=global_step,
                disturbance_trajectory=path, initial_state_override=cfg.robust_economic_reference_state)
            metric, raw, _ = recovery.full_metrics(base, records, env, replay.weight, dual_used)
            if not np.allclose(replay.episode_components, raw, rtol=0, atol=1e-8):
                raise RuntimeError("Replay decomposition differs from the recorded return objective")
            if not all(torch.isfinite(p).all().item() for net in (agent.actor, agent.q1, agent.q2) for p in net.parameters()) \
                    or not np.isfinite(agent.alpha) or any(not np.isfinite(stat[key]) for key in
                        ("actor_loss", "q_loss", "entropy") if replay.size >= cfg.batch_size):
                raise RuntimeError("Nonfinite SAC parameter/loss/entropy")
            dual.update(raw[:, 2:].sum(0) / STEPS)
            row = dict(episode=episode, global_step=global_step, disturbance_seed=path_seed, **metric,
                episode_return=metric["total_training_reward"], legacy_return_diagnostic=float(stat["return"]),
                **{f"lambda_{key}": float(value) for key, value in zip(SAFETY_KEYS, dual.values)})
            for key in ("actor_loss", "q_loss", "entropy"):
                if key in stat and np.isfinite(stat[key]):
                    row[key] = float(stat[key])
            training.append(row)
            max_collapse = max(max_collapse, metric["action_collapse_fraction"])
            persist("running")
            print(f"episode {episode}/{EPISODES} step={global_step} return={row['episode_return']:.6f} "
                  f"gain={metric['economic_improvement_pct']:.7f}% device={agent.device}", flush=True)
            agent.save_checkpoint(model_dir / "last_checkpoint.pth")
            if episode % EVAL_EVERY == 0:
                evaluate(episode)
        persist("completed")
        plot_returns(output)
    except (Exception, KeyboardInterrupt) as exc:
        if "ctrl" in locals():
            write_csv(output / "failure_trajectory.csv", ctrl.evidence)
        if getattr(exc, "evidence", None):
            write_csv(output / "evaluation_failure_trajectory.csv", exc.evidence)
        persist("interrupted" if isinstance(exc, KeyboardInterrupt) else "aborted",
                dict(reason=str(exc) or type(exc).__name__, attempted_global_step=global_step))
        if training or checkpoints:
            plot_returns(output)
        raise


def main():
    args = parser().parse_args()
    if args.phase == "preflight":
        preflight(args)
    elif args.phase == "plot":
        print(plot_returns(args.output_dir))
    else:
        train(args)


if __name__ == "__main__":
    main()
