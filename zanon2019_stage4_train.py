"""Opt-in LOCAL seed42 pilot, only after the complete stage4 oracle gate.

No training is invoked by the stage4 audit/oracle. No other seed entry exists.
The replay objective/network/optimizer are the frozen v2 implementation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from . import zanon2019_stage4 as stage
from . import zanon2019_stage4_metrics as m
from .zanon2019_benchmark import make_agent, sample_disturbance_path, PAPER_VARIANCES_F1_X1_T1_T200
from .zanon2019_paired_v2_objective import EconomicReplay, SafetyDual, SAFETY_KEYS, paired_metrics
from .zanon2019_economic_recovery_metrics import pair_economics, control_activity
from .sac import set_seed
from .train import run_episode, ZeroResidualPolicy

RUN = stage.ROOT/'seed42'


def require_training():
    p = stage.locked()
    boundary = stage.old.load(stage.ROOT/'conservative_reference_boundary_audit.json')
    summary = stage.old.load(stage.ROOT/'oracle_conservative_512_summary.json')
    if not m.training_gate(boundary, summary): raise RuntimeError('Conservative-SAC training gate failed')
    for file, key in (('oracle_conservative_512_candidates.csv', 'candidates_sha256'),
                      ('conservative_reference_boundary_audit.json', 'boundary_audit_sha256')):
        if stage.old.v1.digest(stage.ROOT/file) != summary[key]: raise RuntimeError('Oracle/audit evidence changed')
    if summary['selection_sha256'] != p['selection_sha256']: raise RuntimeError('Primary reference changed')
    if not stage.old.load(stage.ROOT/'tests_receipt.json')['passed']: raise RuntimeError('Tests required')
    return p, summary


def evaluate_metrics(env, records, base, strong, diagnostic, weight, dual, initial):
    metric, raw = paired_metrics(records, base, env, weight, dual)
    additional, series = pair_economics(base, records, initial)
    economy, series3 = m.economic_triplet([r['economic_cost'] for r in strong],
        [r['economic_cost'] for r in base], [r['economic_cost'] for r in records])
    metric.update(additional); metric.update(economy); metric.update(m.f200_summary(diagnostic))
    lower, upper = [env[1].physical_input(v) for v in (env[2].robust_input_lower, env[2].robust_input_upper)]
    metric.update(control_activity(records, lower, upper))
    series.update(series3)
    return metric, raw, series


def qualified(rows, global_step, warmup):
    return bool(global_step >= warmup and len(rows) == 10
        and all(r[k] == 0 for r in rows for k in stage.old.SAFETY)
        and np.mean([r['economic_improvement_pct'] for r in rows]) > 0
        and min(r['economic_improvement_pct'] for r in rows) >= 0
        and m.reasonable_activity(rows)
        and all(r['minimum_F200_tightened_margin'] is None or r['minimum_F200_tightened_margin'] >= -1e-7 for r in rows))


def train(device='cuda'):
    p, oracle = require_training()
    if RUN.exists(): raise RuntimeError('Preserve existing/partial seed42 run; no overwrite')
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    env, check = stage.environment()
    if not check['passed']: raise RuntimeError('Primary reference check failed')
    strong_env = stage.v2.guard(stage.exp.args_for())
    cfg, model, design, omega, domain = env[:5]
    cfg.episodes = 100
    initial = np.array(p['common_initial_state'])
    seed = 42
    set_seed(seed)
    agent = make_agent(cfg, device); agent.zero_initialize_residual_mean()
    dual = SafetyDual.create()
    replay = EconomicReplay(23, 2, cfg.replay_capacity, agent.device, dual, p['smoothness_lambda'])
    (RUN/'models').mkdir(parents=True)
    stage.exp.immutable(RUN/'experiment_manifest.json', dict(training_seed=42, episodes=100, steps=1000,
        device=device, eval_every=5, common_initial_state=initial, selected_x_ref=p['selected_x_ref'],
        selected_u_ref=p['selected_u_ref'], protocol_sha256=stage.old.v1.digest(stage.ROOT/'protocol_stage4.json'),
        oracle_sha256=stage.old.v1.digest(stage.ROOT/'oracle_conservative_512_summary.json'),
        objective='paired Primary Conservative economics /200 minus unchanged fixed baseline-relative applied-input smoothness and original safety duals',
        smoothness_lambda=replay.weight, frozen_SAC_config=vars(agent.cfg), frozen_experiment_config=vars(cfg),
        development_checkpoint_rule='predeclared terminal episode100; all other checkpoints retained, no highest-economic automatic best-policy label',
        disturbance_seed_rule='42*1000000+episode; action rng=path_seed+77',
        validation_seeds=list(stage.DEV), final_test_performed=False))
    training, fixed, checkpoints = [], [], []
    global_step = 0
    cache = {}

    def persist(state, error=None):
        stage.previous.write_csv(RUN/'training_log.csv', training)
        stage.previous.write_csv(RUN/'fixed_evaluation.csv', fixed)
        stage.previous.write_csv(RUN/'checkpoint_performance.csv', checkpoints)
        stage.exp.save(RUN/'checkpoints.json', checkpoints)
        terminal = checkpoints[-1] if checkpoints else {}
        stage.exp.save(RUN/'run_summary.json', dict(run_state=state, completed_episodes=len(training),
            global_step=global_step, no_nonfinite=error is None, failure=error,
            continue_to_2027_recommended=bool(state == 'completed' and terminal.get('eligible_local_pilot_checkpoint')),
            continue_to_314159_recommended=False,
            reason_314159='no automatic continuation; first complete and validate independently authorized seed2027',
            training_performed=True, final_test_performed=False, replay_decomposition_checked=True))

    def evaluate(episode):
        agent.save_actor(RUN/'models'/f'episode_{episode:04d}_actor.pth')
        latest = []
        for validation_seed, (path, base, strong) in cache.items():
            print(f'Primary fixed_eval episode={episode} seed={validation_seed}', flush=True)
            records, diagnostic = stage.rollout(env, path, validation_seed, agent, initial)
            metric, raw, series = evaluate_metrics(env, records, base, strong, diagnostic, replay.weight, dual.values, initial)
            latest.append(dict(episode=episode, global_step=global_step, validation_seed=validation_seed, **metric))
            rows = stage.exp.trace(base, records, series)
            for k, (r, c) in enumerate(zip(rows, raw)):
                r.update(diagnostic[k], economic_reward_raw=float(c[0]), smoothness_penalty_raw=float(c[1]),
                         smoothness_contribution=-replay.weight*float(c[1]),
                         strong_cumulative_cost=float(series['strong_cumulative_cost'][k]),
                         conservative_cumulative_cost=float(series['conservative_cumulative_cost'][k]),
                         candidate_cumulative_cost=float(series['candidate_cumulative_cost'][k]))
            stage.previous.write_csv(RUN/'evaluation_trajectories'/f'episode_{episode:04d}'/f'seed_{validation_seed}.csv', rows)
        fixed.extend(latest)
        status = stage.v2.assessment(latest, episode, global_step, cfg.warmup_steps)
        for key in ('economic_improvement_vs_strong_pct', 'economic_recovery_ratio_pct', 'economic_difference_vs_strong_abs'):
            status['mean_per_path_'+key] = float(np.mean([r[key] for r in latest]))
        js, jc, jp = [float(np.mean([r[k] for r in latest])) for k in ('J_strong', 'J_cons', 'J_candidate')]
        status.update(J_strong=js, J_cons=jc, J_candidate=jp, economic_improvement_pct=100*(jc-jp)/jc,
            economic_improvement_vs_strong_pct=100*(js-jp)/js, **m.recovery(js, jc, jp),
            eligible_local_pilot_checkpoint=qualified(latest, global_step, cfg.warmup_steps),
            checkpoint=str(RUN/'models'/f'episode_{episode:04d}_actor.pth'))
        checkpoints.append(status)
        print('fixed_eval', json.dumps(status, allow_nan=False), flush=True)
        persist('running')
        late = [s for s in checkpoints if s['post_warmup']][-5:]
        if len(late) == 5 and all(s.get('regularizer_fraction') is not None and
                s['regularizer_fraction'] > .10 and s['mean_economic_improvement_pct'] <= 0 for s in late):
            raise RuntimeError('Regularizer scale mismatch; stop, no mid-run retuning')

    try:
        for validation_seed in stage.DEV:
            path, _ = sample_disturbance_path(cfg, 'zanon2019_stochastic', validation_seed, 1000, PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
            base, _ = stage.rollout(env, path, validation_seed, ZeroResidualPolicy(), initial)
            strong, _ = stage.old.rollout(strong_env, path, validation_seed, ZeroResidualPolicy(), initial)
            cache[validation_seed] = path, base, strong
        evaluate(0)
        for episode in range(1, 101):
            require_training()
            path_seed = 42*1000000+episode
            path, _ = sample_disturbance_path(cfg, 'zanon2019_stochastic', path_seed, 1000, PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
            base, _ = stage.rollout(env, path, path_seed, ZeroResidualPolicy(), initial)
            strong, _ = stage.old.rollout(strong_env, path, path_seed, ZeroResidualPolicy(), initial)
            ctrl = stage.BoundaryEvidenceController(cfg, model, design, omega, domain, path)
            replay.bind(ctrl, base); dual_used = dual.values.copy()
            stat, records, global_step = run_episode(cfg, model, ctrl, agent, replay,
                np.random.default_rng(path_seed+77), training=True, global_step=global_step,
                disturbance_trajectory=path, initial_state_override=initial)
            diagnostic = [r['F200_diagnostic'] for r in ctrl.evidence]
            metric, raw, _ = evaluate_metrics(env, records, base, strong, diagnostic, replay.weight, dual_used, initial)
            if not np.allclose(replay.episode_components, raw, rtol=0, atol=1e-8): raise RuntimeError('Replay decomposition mismatch')
            if not all(torch.isfinite(p).all().item() for net in (agent.actor, agent.q1, agent.q2) for p in net.parameters()) or not np.isfinite(agent.alpha):
                raise RuntimeError('Nonfinite SAC parameter/alpha')
            for key in ('actor_loss', 'q_loss', 'entropy'):
                if replay.size >= cfg.batch_size and not np.isfinite(stat[key]): raise RuntimeError('Nonfinite SAC loss')
            dual.update(raw[:, 2:].sum(0)/1000.)
            row = dict(episode=episode, global_step=global_step, disturbance_seed=path_seed, **metric,
                episode_return=metric['total_training_reward'], legacy_return_diagnostic=float(stat['return']),
                **{f'lambda_{k}':float(v) for k,v in zip(SAFETY_KEYS, dual.values)})
            for key in ('actor_loss', 'q_loss', 'entropy'):
                if key in stat and np.isfinite(stat[key]): row[key] = float(stat[key])
            training.append(row); persist('running')
            print(f'Primary Conservative episode {episode}/100 gain={metric["economic_improvement_pct"]:.8f}%', flush=True)
            if episode % 5 == 0: evaluate(episode)
            agent.save_checkpoint(RUN/'models/last_checkpoint.pth')
        persist('completed')
        cp = RUN/'models/episode_0100_actor.pth'
        stage.exp.immutable(RUN/'selected_development_checkpoint.json', dict(episode=100, training_seed=42,
            checkpoint=str(cp), sha256=stage.old.v1.digest(cp), representative_seed=420000,
            global_step=global_step, post_warmup=global_step >= cfg.warmup_steps,
            eligible=checkpoints[-1]['eligible_local_pilot_checkpoint'],
            selection_rule='predeclared terminal episode100; not highest economic benefit', final_test_performed=False))
        report()
    except Exception as exc:
        if 'ctrl' in locals(): stage.exp.save(RUN/'failure_evidence.json', ctrl.evidence)
        if getattr(exc, 'evidence', None): stage.exp.save(RUN/'evaluation_failure_evidence.json', exc.evidence)
        persist('aborted', str(exc)); raise


def report():
    stage.locked()
    run = stage.old.load(RUN/'run_summary.json')
    selection = stage.old.load(RUN/'selected_development_checkpoint.json')
    if run['run_state'] != 'completed' or not selection['post_warmup']:
        raise RuntimeError('Completed frozen post-warmup policy required for formal figures')
    if stage.old.v1.digest(Path(selection['checkpoint'])) != selection['sha256']: raise RuntimeError('Frozen actor changed')
    # Fresh paired representative rollout: never use training disturbance for Fig2.
    env, _ = stage.environment(); strong_env = stage.v2.guard(stage.exp.args_for())
    path, _ = sample_disturbance_path(env[0], 'zanon2019_stochastic', 420000, 1000, PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
    initial = np.array(stage.locked()['common_initial_state'])
    agent = make_agent(env[0], 'cpu')
    from .zanon2019_benchmark import load_actor
    load_actor(agent, Path(selection['checkpoint']))
    base, _ = stage.rollout(env, path, 420000, ZeroResidualPolicy(), initial)
    strong, _ = stage.old.rollout(strong_env, path, 420000, ZeroResidualPolicy(), initial)
    records, diagnostic = stage.rollout(env, path, 420000, agent, initial)
    metric, _, series = evaluate_metrics(env, records, base, strong, diagnostic, stage.locked()['smoothness_lambda'], np.zeros(5), initial)
    figures = stage.ROOT/'figures'; figures.mkdir(exist_ok=True)
    stage.previous.write_csv(figures/'frozen_seed42_development420000.csv', stage.exp.trace(base, records, series))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.arange(1000)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for ax, channel, index, label in ((axes[0,0], 'state', 0, 'X2 (%)'), (axes[0,1], 'state', 1, 'P2 (kPa)'),
        (axes[1,0], 'control', 0, 'P100 (kPa)'), (axes[1,1], 'control', 1, 'F200')):
        ax.plot(t, [r[channel][index] for r in base], label='Primary Conservative zero residual')
        ax.plot(t, [r[channel][index] for r in records], label='Frozen Conservative-SAC')
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0,0].axhline(25, color='black', linestyle='--', label='Physical X2 lower bound')
    axes[0,0].legend(fontsize=8)
    axes[2,0].plot(t, series['instantaneous_economic_advantage_pct']); axes[2,0].axhline(0, color='black', lw=.7)
    axes[2,0].set_ylabel('Instantaneous advantage vs Conservative (%)')
    axes[2,1].plot(t, series['G_cum']); axes[2,1].axhline(0, color='black', lw=.7)
    axes[2,1].set_ylabel('Cumulative advantage vs Conservative (%)')
    for ax in axes.flat: ax.set_xlabel('Time (s)')
    fig.suptitle('Frozen seed42 episode100; development path420000; empirical stochastic safety')
    fig.tight_layout(); fig.savefig(figures/'conservative_fig2_style_combined.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, key in (('Strong zero residual', 'strong_cumulative_cost'), ('Primary Conservative zero residual', 'conservative_cumulative_cost'),
                       ('Frozen Conservative-SAC', 'candidate_cumulative_cost')):
        ax.plot(t+1, series[key], label=label)
    ax.set(xlabel='Completed simulation steps (1 s)', ylabel='Cumulative economic cost')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures/'cumulative_cost_vs_strong.png', dpi=160); plt.close(fig)
    training = stage.old.v1.read_csv(RUN/'training_log.csv')
    checkpoints = stage.old.load(RUN/'checkpoints.json')
    curves = stage.ROOT/'learning_curves'; curves.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot([int(r['episode']) for r in training], [float(r['episode_return']) for r in training], label='True replay-objective episode return')
    axes[0].set(ylabel='Paired economics - frozen regularizer return'); axes[0].legend()
    for label, key in (('vs Conservative', 'economic_improvement_pct'), ('vs Strong', 'economic_improvement_vs_strong_pct')):
        axes[1].plot([r['episode'] for r in checkpoints], [r[key] for r in checkpoints], label=label)
    axes[1].axhline(0, color='black', lw=.7); axes[1].set(ylabel='Fixed deterministic economic improvement (%)'); axes[1].legend()
    for ax in axes: ax.set_xlabel('Episode'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(curves/'seed42_learning_curves.png', dpi=160); plt.close(fig)
    stage.exp.save(figures/'frozen_seed42_development420000_summary.json', dict(metric=metric, checkpoint=selection,
        continue_to_2027_recommended=run['continue_to_2027_recommended'], continue_to_314159_recommended=False,
        all_figures_from_frozen_development_policy=True, final_test_performed=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'report'))
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.phase == 'train': train(args.device)
    else: report()


if __name__ == '__main__': main()
