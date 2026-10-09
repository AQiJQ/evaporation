"""Read-only boundary audit and FULL preregistered Primary Conservative oracle.

Separate stage4 evidence: historical stage2/stage3 and all controller sources
remain read-only. Parallelism is across the ten independent paths of ONE
candidate. Candidate RNG, elite updates, order, count and raw-action profiles
are those of the frozen stage3 search. No training is automatically started.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import inspect
import json
import platform
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

from . import zanon2019_reference2d as previous
from . import zanon2019_stage4_metrics as metrics4
from .control import project_qp_2d
from .train import run_episode
from .zanon2019_paired_v2_objective import paired_metrics, audited_metrics
from .zanon2019_economic_recovery_metrics import pair_economics, control_activity

old, exp, v2 = previous.old, previous.exp, previous.v2
ROOT = exp.ROOT/'conservative_mechanism_stage4_oracle_training'
DEV = old.DEV
UNSTABLE = 'Primary Conservative reference lies too close to the tightened input boundary for numerically robust downstream evaluation.'


def source_hashes():
    return {p.name: old.v1.digest(p) for p in old.v1.REPO.glob('zanon2019_stage4*.py')}


def selection():
    previous.locked()
    value = old.load(previous.ROOT/'conservative_2d_selection.json')
    if not previous.oracle_gate(value) or value['selected_delta_X2'] != .01:
        raise RuntimeError('Primary selection must be the already frozen FIRST +0.01 candidate')
    return value


def prepare():
    selected = selection()
    p3 = previous.locked()
    protocol = dict(schema='primary_conservative_stage4',
        selected_x_ref=selected['selected_x_ref'], selected_u_ref=selected['selected_u_ref'],
        selection_sha256=old.v1.digest(previous.ROOT/'conservative_2d_selection.json'),
        stage3_snapshot=previous.snapshot(previous.ROOT), stage2_snapshot=previous.snapshot(old.ROOT),
        frozen_geometry_sha256=p3['frozen_geometry_sha256'], frozen_v2_sources=v2.source_hashes(),
        frozen_controller_sources={p.name: old.v1.digest(p) for p in old.v1.REPO.glob('*.py')
            if not p.name.startswith('zanon2019_stage4')},
        source_hashes=source_hashes(), validation_seeds=list(DEV), common_initial_state=p3['common_initial_state'],
        steps=1000, dt_min=p3['dt_min'], oracle_budget=512,
        oracle_segments=p3['oracle_segments'], oracle_search_seed=p3['oracle_search_seed'],
        search_generation='exact stage3 constants/Pareto mutations; sequential elite updates, mean per-path economic gain used ONLY for original search generation',
        economic_summary='ratios of mean paired absolute costs; mean of per-path percentages additionally recorded',
        boundary_repeat_in_process=20, boundary_fresh_processes=3, boundary_fresh_process_repeats=3,
        solver_backend='unchanged NumPy float64 exact 2D active-set enumeration',
        solver_tolerance_normalized=inspect.signature(project_qp_2d).parameters['tol'].default,
        reference_feasibility_tolerance_normalized=1e-8,
        diagnostic_near_F200_physical=1e-6,
        diagnostic_near_source='existing physical bound occupancy absolute tolerance; diagnostic only, never a changed constraint',
        control_activity_ratio_limit=metrics4.ACTIVITY_LIMIT,
        control_rule='all ten paths TV and RMS ratios <= existing 1.10 stochastic quality band; no new IAE gate or reward tuning',
        economic_scale=200., smoothness_lambda=p3['smoothness_lambda'],
        break_even_definition='first completed transition whose cumulative Conservative-minus-candidate saving >= FULL horizon (J_cons-J_strong); null if penalty<=0 or not reached',
        strong_outperformance_definition='first completed transition with cumulative candidate cost strictly < cumulative Strong cost',
        reference_description='a state-conservative reference selected according to the preregistered minimum-increment feasibility rule; NOT globally more robust in all margins',
        statistical_scope='empirical stochastic safety; NOT formal Gaussian safety',
        full_trajectory_storage='lossless float64 NPZ for all safe paths, all candidate/path metrics CSV+JSON; preserve historical files on low-space C drive',
        finite_search_result_not_mathematical_upper_bound=True,
        training_performed=False, final_test_performed=False)
    exp.immutable(ROOT/'protocol_stage4.json', protocol)
    print('Prepared immutable stage4; Primary reference and frozen architecture unchanged.', flush=True)


def locked():
    p = old.load(ROOT/'protocol_stage4.json')
    selection()
    if p['source_hashes'] != source_hashes(): raise RuntimeError('Stage4 source changed after preregistration')
    if p['stage3_snapshot'] != previous.snapshot(previous.ROOT) or p['stage2_snapshot'] != previous.snapshot(old.ROOT):
        raise RuntimeError('Historical stage2/stage3 evidence changed')
    for name, digest in p['frozen_controller_sources'].items():
        if old.v1.digest(old.v1.REPO/name) != digest: raise RuntimeError(f'Frozen source changed: {name}')
    for path, digest in p['frozen_geometry_sha256'].items():
        if old.v1.digest(path) != digest: raise RuntimeError('Geometry hash changed')
    if p['oracle_budget'] != 512 or p['validation_seeds'] != list(DEV):
        raise RuntimeError('No changed budget or new development/final-test seeds')
    exp.require_dev(p['validation_seeds'])
    return p


def environment():
    selected = selection()
    return previous.full_check(v2.guard(exp.args_for()), selected['selected_x_ref'],
                               previous.locked()['common_initial_state'])


def boundary_trial():
    env, check = environment()
    cfg, model, design = env[:3]
    u = model.steady_input(cfg.robust_economic_reference_state)
    upper = float(model.physical_input(design.u_upper_tight)[1])
    return dict(F200_ref=float(u[1]), raw_margin=float(upper-u[1]),
        normalized_upper_residual=float(model.normalized_input(u)[1]-design.u_upper_tight[1]),
        steady_derivative_inf=float(np.linalg.norm(model.derivative(cfg.robust_economic_reference_state, u, cfg.disturbance_nominal), np.inf)),
        feasible=bool(check['passed']), QP_init_feasible=bool(check['gates']['online_QP_initialization']),
        reference_input_tightened=bool(check['gates']['reference_input_tightened']),
        limiting_gates=check['limiting_gates'], F200_hex=float(u[1]).hex(),
        state_hex=[float(v).hex() for v in cfg.robust_economic_reference_state])


def boundary_audit():
    p = locked()
    target = ROOT/'conservative_reference_boundary_audit.json'
    if target.exists(): raise RuntimeError('Preserve boundary audit; no tolerance/reference rescue')
    trials = [dict(origin='same Windows process float64 repeat', **boundary_trial())
              for _ in range(p['boundary_repeat_in_process'])]
    for _ in range(p['boundary_fresh_processes']):
        result = subprocess.run([sys.executable, '-m', 'evaporation.zanon2019_stage4', 'boundary-child'],
            cwd=old.v1.REPO.parent, check=True, capture_output=True, text=True)
        trials.extend(dict(origin='fresh Windows Python process', **r) for r in json.loads(result.stdout))
    selected = selection()
    fixture = ROOT/'boundary_serialization_roundtrip.json'
    exp.immutable(fixture, dict(state=selected['selected_x_ref'], input=selected['selected_u_ref']))
    loaded = old.load(fixture)
    env, _ = environment()
    recomputed = env[1].steady_input(np.asarray(loaded['state'], dtype=np.float64))
    serialized = bool(np.array_equal(np.asarray(loaded['state']), selected['selected_x_ref'])
                      and np.array_equal(recomputed, selected['selected_u_ref']))
    margins = np.array([t['raw_margin'] for t in trials])
    # Stable repeatability is NOT a claim that the margin exceeds solver tol.
    stable = bool(all(t['feasible'] and t['QP_init_feasible'] for t in trials)
        and np.isfinite(margins).all() and np.min(margins) >= 0 and serialized
        and len(set(t['F200_hex'] for t in trials)) == 1)
    report = dict(F200_ref=selected['selected_u_ref'][1],
        tightened_F200_upper=float(env[1].physical_input(env[2].u_upper_tight)[1]),
        raw_margin=float(margins[0]), solver_tolerance=p['solver_tolerance_normalized'],
        feasibility_tolerance=p['reference_feasibility_tolerance_normalized'],
        tolerance_units='normalized input; input_scale_F200=100 physical units per normalized unit',
        solver_tolerance_F200_physical=p['solver_tolerance_normalized']*env[0].input_scale[1],
        feasibility_tolerance_F200_physical=p['reference_feasibility_tolerance_normalized']*env[0].input_scale[1],
        feasibility_semantics='all(v_ref >= u_lower_tight - 1e-8) and all(v_ref <= u_upper_tight + 1e-8); raw physical margin separately reported',
        repeat_count=len(trials), min_recomputed_margin=float(margins.min()), max_recomputed_margin=float(margins.max()),
        mean_recomputed_margin=float(margins.mean()), feasible_count=sum(t['feasible'] for t in trials),
        infeasible_count=sum(not t['feasible'] for t in trials),
        QP_init_feasible_count=sum(t['QP_init_feasible'] for t in trials),
        serialization_reload_consistent=serialized,
        steady_input_max_repeat_error=float(np.ptp([t['F200_ref'] for t in trials])),
        repeated_feasible_infeasible_flip=bool(any(t['feasible'] for t in trials) and any(not t['feasible'] for t in trials)),
        numerically_stable=stable, margin_exceeds_solver_tolerance_physical=bool(margins.min() > p['solver_tolerance_normalized']*env[0].input_scale[1]),
        backend=p['solver_backend'], python=sys.version, executable=sys.executable,
        platform=platform.platform(), numpy_version=np.__version__, torch_version=torch.__version__,
        trials=trials, constraints_changed=False, reference_changed=False, input_clamped=False,
        interpretation='deterministic repeat/load/fresh-process stability only; near-zero input margin remains, no global-margin robustness claim',
        status='stable_repeatability' if stable else UNSTABLE)
    exp.immutable(target, report)
    print(f"Boundary numerically_stable={stable}; raw/min/max margin={report['raw_margin']}/{report['min_recomputed_margin']}/{report['max_recomputed_margin']}", flush=True)
    if not stable: raise RuntimeError(UNSTABLE)


class BoundaryEvidenceController(old.v1.PairedEvidenceController):
    """Only read returned action info; do not alter control, QP, state or obs."""
    def act(self, state, actor_action, *, action_is_normalized=True):
        control, info = super().act(state, actor_action, action_is_normalized=action_is_normalized)
        self.evidence[-1]['F200_diagnostic'] = metrics4.f200_step(self.model, self.d, control, info)
        return control, info


def rollout(env, path, seed, policy, initial):
    cfg, model, design, omega, domain = env[:5]
    ctrl = BoundaryEvidenceController(cfg, model, design, omega, domain, path)
    try:
        with torch.no_grad():
            _, records, _ = run_episode(cfg, model, ctrl, policy, None,
                np.random.default_rng(seed+900000), training=False, global_step=0,
                disturbance_trajectory=path, initial_state_override=np.asarray(initial))
        for key in ('state', 'control', 'raw_action', 'w_hat', 'economic_cost'):
            if not np.isfinite(np.array([r[key] for r in records])).all(): raise ValueError(f'Nonfinite {key}')
        audited = audited_metrics(records, env)
        if len(records) != len(path) or any(audited[k] for k in old.SAFETY):
            raise RuntimeError('Incomplete or unsafe Primary Conservative rollout')
        return records, [r['F200_diagnostic'] for r in ctrl.evidence]
    except Exception as exc:
        exc.evidence = ctrl.evidence
        raise


_worker_env = None
_worker_cache = None


def worker_init():
    global _worker_env, _worker_cache
    torch.set_num_threads(1)
    _worker_env, report = environment()
    if not report['passed']: raise RuntimeError('Selected reference no longer passes')
    selected = selection()
    cache = {}
    for seed in DEV:
        base_file = previous.ROOT/'baseline_trajectories'/f"delta_{selected['selected_delta_X2']:.2f}"/f'seed_{seed}.csv'
        strong_file = old.ROOT/'baseline_trajectories/delta_0.00'/f'seed_{seed}.csv'
        base, _ = v2.archived_pair(base_file)
        strong, _ = v2.archived_pair(strong_file)
        path, _ = previous.sample_disturbance_path(_worker_env[0], 'zanon2019_stochastic', seed, 1000,
            previous.PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
        for records in (base, strong):
            if len(records) != 1000: raise RuntimeError('Incomplete historical pair')
            np.testing.assert_array_equal(np.array([r['disturbance'] for r in records]), path)
            np.testing.assert_array_equal(records[0]['state'], previous.locked()['common_initial_state'])
        if any(audited_metrics(base, _worker_env)[k] for k in old.SAFETY): raise RuntimeError('Unsafe baseline archive')
        cache[seed] = path, base, strong
    _worker_cache = cache


def worker_trial(task):
    index, segment, actions, seed = task
    folder = ROOT/'oracle'/'candidates'/f'candidate_{index:04d}'
    saved = folder/f'seed_{seed}_metrics.json'
    action_sha = old.v1.digest(ROOT/'oracle'/'actions'/f'candidate_{index:04d}.npy')
    if saved.exists():
        value = old.load(saved)
        if value['action_sha256'] != action_sha: raise RuntimeError('Cached action profile changed')
        if value.get('trace') and old.v1.digest(Path(value['trace'])) != value['trace_sha256']:
            raise RuntimeError('Cached trace changed')
        return value
    env = _worker_env
    path, base, strong = _worker_cache[seed]
    try:
        records, diagnostics = rollout(env, path, seed, old.v1.SegmentPolicy(actions, segment),
                                       previous.locked()['common_initial_state'])
        metric, _ = paired_metrics(records, base, env, 0., np.zeros(5))
        economic, series = metrics4.economic_triplet([r['economic_cost'] for r in strong],
            [r['economic_cost'] for r in base], [r['economic_cost'] for r in records])
        detail, _ = pair_economics(base, records, previous.locked()['common_initial_state'])
        metric.update(detail); metric.update(economic); metric.update(metrics4.f200_summary(diagnostics))
        lo, hi = [env[1].physical_input(v) for v in (env[2].robust_input_lower, env[2].robust_input_upper)]
        metric.update(control_activity(records, lo, hi))
        metric.update(candidate=index, validation_seed=seed)
        # Lossless compression, not reduced precision or downsampled evidence.
        # Undefined Omega nominal margins use indexed Z-only arrays, not fake
        # zeros or a blanket NaN that might hide actual nonfinite plant values.
        folder.mkdir(parents=True, exist_ok=True)
        trace = folder/f'seed_{seed}_trajectory.npz'
        z_indices = np.array([k for k,r in enumerate(diagnostics) if r['Z_nominal_F200_tightened_margin'] is not None], dtype=np.int64)
        arrays = {f'{label}_{key}':np.array([r[key] for r in data])
            for label,data in (('Strong', strong), ('Conservative', base), ('candidate', records))
            for key in ('state', 'control', 'economic_cost', 'raw_action', 'w_hat')}
        arrays.update(disturbance=path, Z_nominal_F200_margin_steps=z_indices,
            Z_nominal_F200_tightened_margin=np.array([diagnostics[k]['Z_nominal_F200_tightened_margin'] for k in z_indices]),
            actual_F200_comparison_to_nominal_tightened_upper_margin=np.array([r['actual_F200_comparison_to_nominal_tightened_upper_margin'] for r in diagnostics]),
            qp_F200_input_upper_active=np.array([r['qp_F200_input_upper_active'] for r in diagnostics]),
            qp_Z_F200_tightened_upper_active=np.array([r['qp_Z_F200_tightened_upper_active'] for r in diagnostics]),
            mapped_residual_rejected_or_clipped_due_to_F200=np.array([r['mapped_residual_rejected_or_clipped_due_to_F200'] for r in diagnostics]),
            instantaneous_advantage_pct=series['instantaneous_economic_advantage_pct'], G_cum=series['G_cum'])
        arrays.update({key:series[key] for key in ('strong_cumulative_cost', 'conservative_cumulative_cost',
            'candidate_cumulative_cost', 'cumulative_improvement_vs_strong_pct')})
        np.savez_compressed(trace, **arrays)
        result = dict(metric=metric, failure=None, trace=str(trace), trace_sha256=old.v1.digest(trace),
                      trace_format='lossless float64 NPZ; indexed Z-only nominal input margins', action_sha256=action_sha)
    except Exception as exc:
        evidence = getattr(exc, 'evidence', None)
        if evidence: previous.write_csv(folder/f'seed_{seed}_failure.csv', evidence)
        result = dict(metric=None, failure=str(exc), trace=None, action_sha256=action_sha)
    exp.immutable(saved, result)
    return result


def actions_for(index, protocol, rng, elites):
    segment = protocol['oracle_segments'][0 if index < 256 else 1]
    count = 1000//segment
    if index < 25:
        grid = np.linspace(-1., 1., 5)
        actions = np.tile([grid[index//5], grid[index % 5]], (count, 1))
    elif elites and index % 3:
        previous_actions = elites[int(rng.integers(len(elites)))]['actions']
        at = np.minimum(np.arange(count)*len(previous_actions)//count, len(previous_actions)-1)
        actions = np.clip(previous_actions[at]+rng.normal(0, .10, (count, 2)), -1, 1)
    else:
        actions = np.clip(rng.uniform(-.6, .6, (1, 2))+rng.normal(0, .2, (count, 2)), -1, 1)
    return segment, actions


def update_elites(elites, row, actions):
    # Preserve the old search generation objective/ordering, separately from
    # reporting ratio-of-means economics required by the new audit.
    def value(r):
        return np.array([-r['mean_per_path_economic_improvement_pct'], r['P100_TV_ratio'],
                         r['F200_TV_ratio'], r['X2_std_ratio']])
    front = [*elites, dict(**row, actions=actions)]
    return [r for r in front if not any(np.all(value(t) <= value(r)) and np.any(value(t) < value(r)) for t in front)]


def finish_summary(rows, boundary):
    if len(rows) != 512 or [r['candidate'] for r in rows] != list(range(512)):
        raise RuntimeError('Full 512 candidates required; no early witness stop')
    safe = [r for r in rows if r['safety_passed']]
    positive = [r for r in safe if r['economic_improvement_pct'] > 0]
    eligible = [r for r in positive if r['all_validation_nonnegative'] and r['reasonable_control_activity']]
    best = max(positive, key=lambda r: r['economic_improvement_pct']) if positive else None
    # Report maximum economics separately from the control-qualified witness.
    witness = max(eligible, key=lambda r: r['economic_improvement_pct']) if eligible else None
    summary = dict(total_candidate_count=len(rows), completed_all_512=True,
        safety_passing_candidate_count=len(safe), positive_vs_Conservative_count=len(positive),
        all_validation_positive_count=sum(r['all_validation_positive'] for r in positive),
        positive_vs_Strong_count=sum(r['economic_improvement_vs_strong_pct'] > 0 for r in safe),
        maximum_improvement_vs_Conservative_pct=best['economic_improvement_pct'] if best else None,
        maximum_improvement_vs_Strong_pct=max((r['economic_improvement_vs_strong_pct'] for r in safe), default=None),
        best_recovery_ratio_pct=max((r['economic_recovery_ratio_pct'] for r in safe
                                     if r['economic_recovery_ratio_pct'] is not None), default=None),
        best_economic_candidate=best, control_qualified_positive_candidate_count=len(eligible), training_witness=witness,
        candidate_recovered_full_penalty_and_surpassed_Strong=any(r['economic_improvement_vs_strong_pct'] > 0 for r in safe),
        finite_search_result_not_mathematical_upper_bound=True,
        label='best economic performance found by the preregistered finite search',
        certification_status='empirical stochastic safety', training_performed=False, final_test_performed=False)
    for k in ('worst_validation_improvement_pct', 'economic_win_fraction', 'mean_X2_shift_vs_baseline',
        'X2_std_ratio', 'centered_X2_MAE_ratio', 'minimum_X2_margin', 'minimum_F200_tightened_margin',
        'qp_F200_upper_bound_active_fraction', *metrics4.ACTIVITY_KEYS):
        summary[k] = best[k] if best else None
    summary['conservative_seed42_training_allowed'] = metrics4.training_gate(boundary, summary)
    return summary


def oracle(workers=8):
    p = locked()
    boundary = old.load(ROOT/'conservative_reference_boundary_audit.json')
    if not boundary['numerically_stable']: raise RuntimeError(UNSTABLE)
    receipt = old.load(ROOT/'tests_receipt.json')
    if not receipt['passed']: raise RuntimeError('Complete tests must pass before oracle')
    summary_path = ROOT/'oracle_conservative_512_summary.json'
    if summary_path.exists(): raise RuntimeError('Completed oracle preserved')
    rng = np.random.default_rng(p['oracle_search_seed'])
    rows, per, elites = [], [], []
    (ROOT/'oracle/actions').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        for index in range(512):
            segment, actions = actions_for(index, p, rng, elites)
            action_path = ROOT/'oracle/actions'/f'candidate_{index:04d}.npy'
            if action_path.exists(): np.testing.assert_array_equal(np.load(action_path), actions)
            else: np.save(action_path, actions)
            results = list(pool.map(worker_trial, [(index, segment, actions, seed) for seed in DEV]))
            current = [r['metric'] for r in results if r['metric'] is not None]
            failures = [r['failure'] for r in results if r['failure']]
            row = metrics4.summarize_candidate(index, segment, current, '; '.join(failures) if failures else None, old.SAFETY)
            rows.append(row); per.extend(current)
            if row['safety_passed']: elites = update_elites(elites, row, actions)
            previous.write_csv(ROOT/'oracle_conservative_512_candidates.csv', rows)
            previous.write_csv(ROOT/'oracle_conservative_512_per_path.csv', per)
            exp.save(ROOT/'oracle_progress.json', dict(completed_candidates=len(rows), budget=512,
                safety_passing=sum(r['safety_passed'] for r in rows),
                positive_vs_Conservative=sum(r['safety_passed'] and r['economic_improvement_pct'] > 0 for r in rows),
                running=len(rows) < 512, training_allowed=False, workers=workers))
            print(f"Primary oracle {index+1}/512: safe={row['safety_passed']}, gainC={row['economic_improvement_pct']}, gainStrong={row['economic_improvement_vs_strong_pct']}", flush=True)
    summary = finish_summary(rows, boundary)
    summary.update(protocol_sha256=old.v1.digest(ROOT/'protocol_stage4.json'),
        candidates_sha256=old.v1.digest(ROOT/'oracle_conservative_512_candidates.csv'),
        selection_sha256=p['selection_sha256'], boundary_audit_sha256=old.v1.digest(ROOT/'conservative_reference_boundary_audit.json'))
    previous.write_csv(ROOT/'oracle_conservative_512_pareto.csv', [{k: v for k, v in r.items() if k != 'actions'} for r in elites])
    exp.immutable(summary_path, summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


def sensitivity():
    """Read-only secondary intensity analysis; NEVER replace Primary +0.01."""
    locked()
    rows = old.load(previous.ROOT/'conservative_2d_baseline_results.json')['candidates']
    prev_margin, prev_penalty = selection()['Strong_metrics']['minimum_X2_margin'], 0.
    output = []
    for r in rows:
        dm, dp = r['minimum_X2_margin']-prev_margin, r['conservatism_penalty_pct']-prev_penalty
        output.append(dict(delta_X2=r['delta_X2'], minimum_X2_margin=r['minimum_X2_margin'],
            economic_penalty_pct=r['conservatism_penalty_pct'], incremental_margin=dm,
            incremental_penalty_percentage_points=dp, marginal_margin_per_penalty_percentage_point=dm/dp if dp > 0 else None,
            primary=r['delta_X2'] == .01, role='primary frozen' if r['delta_X2'] == .01 else 'secondary intensity only'))
        prev_margin, prev_penalty = r['minimum_X2_margin'], r['conservatism_penalty_pct']
    previous.write_csv(ROOT/'secondary_conservatism_intensity.csv', output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'tests', 'boundary', 'boundary-child', 'oracle', 'sensitivity', 'status'))
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.phase == 'prepare': prepare()
    elif args.phase == 'tests': exp.tests(ROOT)
    elif args.phase == 'boundary-child':
        locked()
        print(json.dumps([boundary_trial() for _ in range(3)]))
    elif args.phase == 'boundary': boundary_audit()
    elif args.phase == 'oracle':
        if not 1 <= args.workers <= 10: raise ValueError('One to ten independent path workers only')
        oracle(args.workers)
    elif args.phase == 'sensitivity': print(sensitivity())
    else:
        locked()
        for name in ('conservative_reference_boundary_audit.json', 'oracle_progress.json', 'oracle_conservative_512_summary.json'):
            print(name, old.load(ROOT/name) if (ROOT/name).exists() else 'not_run')


if __name__ == '__main__': main()
