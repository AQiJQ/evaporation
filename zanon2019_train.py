"""Frozen balanced-B stochastic SAC; paired empirical reporting, not Gaussian certification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .sac import ReplayBuffer, set_seed
from .train import OBS_DIM, run_episode
from .zanon2019_benchmark import (
    DEFAULT_DESIGN, DEFAULT_OMEGA, DEFAULT_OUTPUT,
    PAPER_VARIANCES_F1_X1_T1_T200, make_agent, make_setup, load_actor,
    sample_disturbance_path, paper_specification, write_csv,
)
from .zanon2019_training_report import (
    CERTIFICATION_STATUS, SELECTION_RULES, CheckpointSelection,
    EvidenceController, FixedPairedEvaluator, classify, metrics,
    plot_learning, update_pareto, evaluation_safety_anomalies,
)


class EvaluationSafetyAbort(RuntimeError):
    """Evaluation evidence has already been persisted; do not relabel as training failure."""


def save_json(path, data):
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False,
                                   default=encode), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disturbance-mode", choices=(
        "zanon2019_stochastic", "piecewise_stochastic"), default="zanon2019_stochastic")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seeds", type=int, default=3)
    parser.add_argument("--eval-seed-start", type=int, default=420000)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--variances", type=float, nargs=4,
                        default=PAPER_VARIANCES_F1_X1_T1_T200.tolist(),
                        metavar=("F1", "X1", "T1", "T200"))
    parser.add_argument("--hold-min", type=int, default=20)
    parser.add_argument("--hold-max", type=int, default=50)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT.with_name(
        "outputs_zanon2019_stochastic_seed42_300x1000"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if min(args.episodes, args.steps, args.eval_seeds, args.eval_every) < 1:
        raise ValueError("Episodes, steps, eval seeds and eval interval must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("Output contains evidence; choose a fresh --output-dir")
    set_seed(args.seed)
    cfg, model, design, omega, domain, weights = make_setup(
        args.steps, args.seed, args.design, args.omega_vertices)
    cfg.disturbance_mode, cfg.episodes = args.disturbance_mode, args.episodes
    cfg.abort_on_first_uncertified_step = True
    agent = make_agent(cfg, args.device)
    agent.zero_initialize_residual_mean()
    replay = ReplayBuffer(OBS_DIM + 4, 2, cfg.replay_capacity, agent.device)
    out, model_dir = args.output_dir, args.output_dir / "models"
    model_dir.mkdir(parents=True)
    eval_seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_seeds))
    manifest = {
        "episodes": args.episodes, "steps": args.steps, "seed": args.seed,
        "mode": args.disturbance_mode, "observation_dim": 23,
        "device": str(agent.device), "certification_status": CERTIFICATION_STATUS,
        "formal_safety_claim": False,
        "architecture": "frozen balanced B interior-anchor Z/Omega; unchanged Gm/Bj inactive in no-jump Experiment I",
        "reward": "unchanged weighted reward", "weights": weights,
        "reward_scope": "out-of-W RPI event/excess remains excluded from replay reward and fully audited",
        "variance_F1_X1_T1_T200": args.variances,
        "evaluation_seeds": eval_seeds, "evaluation_every": args.eval_every,
        "baseline": "same architecture zero residual; NOT ECC2019 RL-NMPC",
        "warmup_steps": cfg.warmup_steps, "selection_rules": SELECTION_RULES,
        "evaluation_stop_rule": "after each fixed paired evaluation, any baseline/SAC physical/input/QP/Omega anomaly stops before further training; W alone does not stop",
        "frozen_experiment_config": vars(cfg), "frozen_SAC_config": vars(agent.cfg),
        "frozen_geometry_sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (args.design, args.omega_vertices)},
        "training_disturbance_seed_rule": "seed*1000000 + episode; action RNG=path_seed+77",
        "empirical_G_definition": "L2 norm of fixed-B state error / L2 norm of disturbance standardized by literal standard deviations; not Hinf gain",
        "paper_specification": paper_specification(args.disturbance_mode, cfg, args.steps,
            np.asarray(args.variances), args.hold_min, args.hold_max),
    }
    save_json(out / "experiment_manifest.json", manifest)
    agent.save_actor(model_dir / "initial_diagnostic_actor.pth")
    training_rows, fixed_rows, statuses = [], [], []
    global_step = 0
    selector = CheckpointSelection(model_dir)

    def persist(state, failure=None, comparison=None):
        update_pareto(statuses)
        front = [s for s in statuses if s["pareto_nondominated"]]
        if front:
            closest = min(front, key=lambda s: (s["distance_to_joint"],
                          -s["economic_improvement_pct"], s["mean_IAE_ratio"], s["mean_TV_ratio"]))
            destination = model_dir / "closest_pareto_actor.pth"
            shutil.copyfile(closest["checkpoint"], destination)
            selector.best["closest_pareto_actor"] = {
                **closest, "checkpoint": str(destination),
                "selection_score": [closest["distance_to_joint"], -closest["economic_improvement_pct"]]}
        write_csv(out / "training_log.csv", training_rows)
        write_csv(out / "fixed_evaluation.csv", fixed_rows)
        write_csv(out / "checkpoint_pareto.csv", statuses)
        save_json(out / "checkpoint_selection.json", selector.best)
        counters = {key: int(sum(r[key] for r in training_rows)) for key in (
            "physical_state_violation_steps", "physical_input_violation_count",
            "QP_infeasible_count", "Omega_exit_count", "W_exceedance_count",
            "robust_region_violation_count")}
        save_json(out / "run_summary.json", {
            **manifest, "run_state": state, "completed_episodes": len(training_rows),
            "global_step": global_step, "training_safety_counters_completed_episodes": counters,
            "failure_evidence": failure,
            "selected_checkpoints": selector.best, "comparison": comparison,
            "case": "D" if failure else classify(statuses),
            "multi_seed_evaluation_recommendation": (
                "consider best_empirical_joint_actor after reviewing Pareto trade-off; empirical evaluation only"
                if state == "completed" and classify(statuses) == "A" else
                "none automatically recommended; inspect Pareto/evidence first"),
        })

    try:
        evaluator = FixedPairedEvaluator(cfg, model, design, omega, domain,
            args.disturbance_mode, args.variances, args.hold_min, args.hold_max,
            eval_seeds, out)
    except Exception as exc:
        persist("evaluation_setup_failed", {"reason": str(exc)})
        raise

    def stop_on_evaluation_anomaly(episode, actor_path, latest, status,
                                   checked_agent, stage="fixed_evaluation"):
        anomalies = evaluation_safety_anomalies(latest)
        if not anomalies:
            return
        # The entire paired evaluation has already been written, including
        # unsafe actors. Finish this diagnostic batch, never another episode.
        checked_agent.save_actor(model_dir / "evaluation_failure_actor.pth")
        checked_agent.save_checkpoint(model_dir / "evaluation_failure_checkpoint.pth")
        folder = stage if stage != "fixed_evaluation" else f"episode_{episode:04d}"
        failure = {"phase": stage, "episode": episode, "global_step": status["global_step"],
                   "source_checkpoint": str(actor_path), "anomalies": anomalies,
                   "assessment": status, "certification_status": CERTIFICATION_STATUS,
                   "paired_metrics": str(out / "evaluation_failure_metrics.csv"),
                   "trajectories": str(out / "evaluation_trajectories" / folder),
                   "reason": "physical/input/QP/Omega anomaly in paired evaluation; no further training permitted"}
        write_csv(out / "evaluation_failure_metrics.csv", latest)
        save_json(out / "evaluation_abort.json", failure)
        persist("evaluation_aborted", failure)
        plot_learning(fixed_rows, out)
        raise EvaluationSafetyAbort(
            f"ECC2019 evaluation safety anomaly at episode {episode}: {anomalies}. "
            f"Stopped before further training; evidence: {out / 'evaluation_abort.json'}")

    def evaluate(episode):
        actor_path = model_dir / "evaluation" / f"episode_{episode:04d}_actor.pth"
        actor_path.parent.mkdir(exist_ok=True)
        agent.save_actor(actor_path)
        latest, status = evaluator.evaluate(agent, episode, global_step)
        status["checkpoint"] = str(actor_path)
        fixed_rows.extend(latest)
        statuses.append(status)
        print("fixed_eval", json.dumps(status), flush=True)
        # Check before checkpoint selection and before starting the next
        # training episode, also at diagnostic episode 0 / before warmup.
        stop_on_evaluation_anomaly(episode, actor_path, latest, status, agent)
        selector.consider(agent, status)
        persist("running")

    evaluate(0)
    print(f"device={agent.device}; paired seeds={eval_seeds}; scope={CERTIFICATION_STATUS}", flush=True)
    for episode in range(1, args.episodes + 1):
        path_seed = args.seed * 1000000 + episode
        disturbance, metadata = sample_disturbance_path(cfg, args.disturbance_mode,
            path_seed, args.steps, args.variances, args.hold_min, args.hold_max)
        ctrl = EvidenceController(cfg, model, design, omega, domain, disturbance)
        episode_start = global_step
        try:
            stat, records, global_step = run_episode(cfg, model, ctrl, agent, replay,
                np.random.default_rng(path_seed + 77), training=True, global_step=global_step,
                disturbance_trajectory=disturbance,
                initial_state_override=cfg.robust_economic_reference_state)
            metric = metrics(records, cfg, model, design, omega)
            training_rows.append({"episode": episode, "global_step": global_step,
                "disturbance_seed": path_seed, "mode": args.disturbance_mode,
                "disturbance_clip_count_F1": metadata["physical_clip_counts_F1_X1"][0],
                "disturbance_clip_count_X1": metadata["physical_clip_counts_F1_X1"][1],
                "reward_return": stat["return"], **metric})
            persist("running")
            print(f"episode={episode}/{args.episodes} step={global_step} "
                  f"J={metric['J_econ']:.4f} QP={metric['QP_infeasible_count']} "
                  f"Omega={metric['Omega_exit_count']} W={metric['W_exceedance_count']}", flush=True)
            if episode % args.eval_every == 0 or episode == args.episodes:
                evaluate(episode)
            agent.save_checkpoint(model_dir / "last_checkpoint.pth")
        except EvaluationSafetyAbort:
            # Do not overwrite evaluation failure with the preceding safe
            # training episode's trace or falsely label that episode unsafe.
            raise
        except Exception as exc:
            agent.save_actor(model_dir / "failure_actor.pth")
            agent.save_checkpoint(model_dir / "failure_checkpoint.pth")
            write_csv(out / "failure_trajectory.csv", ctrl.evidence)
            write_csv(out / "failure_disturbance.csv", [
                {"step": i, **{name: float(v) for name, v in zip(("F1", "X1", "T1", "T200"), d)}}
                for i, d in enumerate(disturbance)])
            failure = {"episode": episode, "episode_start_global_step": episode_start,
                "reason": str(exc), "attempted_steps": len(ctrl.evidence),
                "observed_next_steps": sum(r["next_state_observed"] for r in ctrl.evidence),
                "partial_episode_safety_counters": {
                    key: sum(int(r.get(key, 0)) for r in ctrl.evidence)
                    for key in ("physical_state_violation", "physical_input_violation",
                                "QP_infeasible", "Omega_exit_before", "Omega_exit_after",
                                "W_exceedance")},
                "episode_already_in_completed_counters": any(
                    r["episode"] == episode for r in training_rows),
                "last_step_evidence": ctrl.evidence[-1] if ctrl.evidence else None,
                "certification_status": CERTIFICATION_STATUS,
                "trajectory": str(out / "failure_trajectory.csv")}
            save_json(out / "training_abort.json", failure)
            persist("aborted", failure)
            plot_learning(fixed_rows, out)
            raise

    final_name = "final_actor" if global_step >= cfg.warmup_steps else "final_diagnostic_actor"
    agent.save_actor(model_dir / f"{final_name}.pth")
    agent.save_checkpoint(model_dir / "final_checkpoint.pth")
    comparison_rows, comparison_statuses = [], []
    eval_agent = make_agent(cfg, args.device)
    selected = ("best_empirical_joint_actor", "best_economic_safe_actor",
                "best_recovery_safe_actor", "best_low_activity_safe_actor", "final_actor")
    for name in selected:
        source = selector.best.get(name)
        if name == "final_actor" and final_name == "final_actor":
            source = {"checkpoint": str(model_dir / "final_actor.pth"),
                      "episode": args.episodes, "global_step": global_step}
        if source is None:
            comparison_statuses.append({"name": name, "available": False,
                "reason": "no eligible post-warmup checkpoint; not fabricated"})
            continue
        load_actor(eval_agent, Path(source["checkpoint"]))
        latest, status = evaluator.evaluate(eval_agent, source["episode"],
            source["global_step"], label=name)
        comparison_rows.extend(latest)
        comparison_statuses.append({"name": name, "available": True, **status})
        write_csv(out / "selected_checkpoint_comparison.csv", comparison_statuses)
        write_csv(out / "selected_checkpoint_per_seed.csv", comparison_rows)
        stop_on_evaluation_anomaly(source["episode"], source["checkpoint"], latest,
                                   status, eval_agent, stage=name)
    write_csv(out / "selected_checkpoint_comparison.csv", comparison_statuses)
    write_csv(out / "selected_checkpoint_per_seed.csv", comparison_rows)
    plot_learning(fixed_rows, out)
    persist("completed", comparison=comparison_statuses)


if __name__ == "__main__":
    main()
