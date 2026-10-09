"""Frozen Candidate-A reproducibility + independent paired test orchestration.

No training starts without --phase train/run. prepare is read-only with respect
to algorithms and all existing runs. Final tests never select a checkpoint.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
import numpy as np
from .ecc2019_reproduction import REPO, SOURCES, digest
from .zanon2019_authority import AUTHORITY_RULES, VALIDATION_SEEDS, RESERVED_TEST_SEEDS, AuthorityController, hierarchy_key
from .zanon2019_benchmark import make_setup, make_agent, load_actor, sample_disturbance_path, write_csv, DEFAULT_DESIGN, DEFAULT_OMEGA
from .zanon2019_training_report import CERTIFICATION_STATUS, metrics, RATIO_METRICS, relative
from .train import run_episode, ZeroResidualPolicy
from .zanon2019_final_selection import VERSION as FINAL_VERSION, rule_document, scan as final_selection_scan
from .zanon2019_final_validation_report import write_report as write_validation_report

SEEDS = (42, 2027, 314159)
ROOT = REPO/'evaporation_safe_sac/final_stochastic_candidate_A'
SOURCE42 = SOURCES['proposed_alpha010_ep100']
REPRESENTATIVE = 430000


def plain(value):
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if hasattr(v,'tolist') else str(v)))


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def protected_files():
    # Freeze the existing production implementation, not its last committed
    # version: pre-existing user edits are preserved and hashed as they are.
    return sorted(p for p in REPO.glob('*.py') if not p.name.startswith(
        ('paper_', 'ecc2019_', 'zanon2019_final_')))


def fingerprint(manifest):
    """Compare every frozen config field; only seed is allowed to vary."""
    cfg=dict(manifest['frozen_experiment_config']); cfg.pop('seed',None)
    keys=('episodes','steps','stochastic_residual_scale','authority_pilot','validation_seeds',
          'variance_F1_X1_T1_T200','mode','observation_dim','reward','weights','reward_scope',
          'warmup_steps','evaluation_every','selection_rules','frozen_SAC_config','frozen_geometry_sources')
    return dict(config=cfg,**{k:manifest[k] for k in keys})


def assert_frozen(root):
    p=read_json(root/'protocol_lock.json')
    for path, sha in p['source_sha256'].items():
        if digest(path)!=sha: raise RuntimeError(f'Frozen source/geometry changed: {path}')
    if p.get('selection_rule_version')==FINAL_VERSION:
        rule=root/'final_checkpoint_selection_rule.json'
        if digest(rule)!=p['selection_rule_sha256'] or read_json(rule)!=rule_document():
            raise RuntimeError('Permanent final_v1 selection rule/source changed')
    return p


def seed_usage_audit(root):
    """Scan executed seed fields, not mentions in reserved/plan declarations.

    This is an evidence audit of the local artifacts, not a proof of unseen
    human use. Retain paths and hashes so the audited scope is inspectable.
    """
    results=REPO/'evaporation_safe_sac'; forbidden=set(RESERVED_TEST_SEEDS)
    rows=[]; scans=[]; collisions=[]
    active_keys={'seed','test_seed','disturbance_seed','evaluation_seed','training_seed'}
    def inspect_json(value,path):
        if isinstance(value,dict):
            for key,item in value.items():
                if key in active_keys and isinstance(item,int) and item in forbidden:
                    collisions.append(dict(path=str(path),seed=item,evidence='JSON '+key))
                if key in ('validation_seeds','evaluation_seeds','diagnostic_seeds') and isinstance(item,list):
                    for seed in item:
                        if isinstance(seed,int):
                            rows.append(dict(seed=seed,role='validation_or_diagnostic_declaration',path=str(path),field=key))
                            if seed in forbidden:collisions.append(dict(path=str(path),seed=seed,evidence=key))
                if key not in ('reserved_final_test_seeds','final_test_seeds','test_seeds','reserved_test_seeds'):
                    inspect_json(item,path)
        elif isinstance(value,list):
            for item in value:inspect_json(item,path)
    for path in sorted(results.rglob('*')):
        if not path.is_file(): continue
        if path.parent==root and path.name in (
            'protocol_lock.json','seed_usage_audit.csv','seed_usage_audit.json',
            'local_training_commands.json','seed42_validation_lock.json','locked_final_policies.json'):
            continue  # Our declarations/audit output are not prior execution.
        if path.suffix.lower() not in ('.csv','.json'): continue
        if any(path.name==f'seed_{s}.csv' for s in forbidden):
            collisions.append(dict(path=str(path),evidence='reserved test trajectory filename'))
        if path.suffix=='.csv':
            with path.open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f)
                keys=set(reader.fieldnames or ()) & active_keys
                if not keys: continue
                scans.append(dict(path=str(path),sha256=digest(path),kind='executed CSV seed fields'))
                seen=set()
                for row in reader:
                    # A previous audit's unused reserve entries are declarations,
                    # not executed final-test evidence.
                    if row.get('role') in ('unused_test_reserved','reserved_unused'): continue
                    for key in keys:
                        try: seed=int(row[key])
                        except (ValueError,TypeError): continue
                        if (key,seed) in seen: continue
                        seen.add((key,seed))
                        role=('training' if key=='disturbance_seed' and path.name=='training_log.csv' else
                              'validation' if path.name=='fixed_evaluation.csv' else 'diagnostic_or_evaluation')
                        rows.append(dict(seed=seed,role=role,path=str(path),field=key))
                        if seed in forbidden: collisions.append(dict(path=str(path),seed=seed,evidence=key))
        else:
            data=read_json(path)
            scans.append(dict(path=str(path),sha256=digest(path),kind='JSON execution metadata and nested seed fields'))
            if not isinstance(data,dict): continue
            inspect_json(data,path)
            if data.get('final_test_performed') is True:
                collisions.append(dict(path=str(path),evidence='final_test_performed=True; manual provenance review required'))
    rows.extend(dict(seed=s,role='reserved_unused',path='local artifact audit',field='no execution evidence found') for s in RESERVED_TEST_SEEDS)
    root.mkdir(parents=True,exist_ok=True)
    write_csv(root/'seed_usage_audit.csv',rows)
    report=dict(passed=not collisions,collisions=collisions,scanned_files=scans,
                scope=str(results),limitation='no evidence of prior use in local artifacts; unseen external use cannot be ruled out',
                final_test_seeds=RESERVED_TEST_SEEDS,training_seeds=list(SEEDS))
    save(root/'seed_usage_audit.json',report)
    if collisions: raise RuntimeError('Reserved final-test seed prior-use evidence found. Stop; do not replace seeds.')
    return report


def prepare(root):
    if (root/'test_started.json').exists(): raise RuntimeError('Test already started; do not recreate protocol or select policies')
    if (root/'locked_final_policies.json').exists():raise RuntimeError('Policies already locked; cannot change selection')
    old=assert_frozen(root) if (root/'protocol_lock.json').exists() else None
    audit=seed_usage_audit(root)
    template=read_json(SOURCE42/'experiment_manifest.json')
    for path,sha in template['frozen_geometry_sources'].items():
        if digest(path)!=sha:raise RuntimeError(f'Candidate-A frozen geometry differs: {path}')
    cfg,model,design,omega,domain,weights=make_setup(1000,42)
    cfg.episodes=100; cfg.stochastic_residual_scale=.1
    cfg.authority_pilot=True; cfg.abort_on_first_uncertified_step=True
    current=plain(vars(cfg)); expected=template['frozen_experiment_config']
    if current!=expected: raise RuntimeError('Current config differs from Candidate A; no silent seed42 reuse')
    agent=make_agent(cfg,'cpu')  # inspect configuration only; never optimizes
    if plain(vars(agent.cfg))!=template['frozen_SAC_config']: raise RuntimeError('SAC hyperparameter mismatch')
    if weights!=template['weights'] or template['stochastic_residual_scale']!=.1:
        raise RuntimeError('Candidate A reward/authority mismatch')
    summary=read_json(SOURCE42/'run_summary.json')
    if summary['run_state']!='completed' or summary['completed_episodes']!=100 or summary['global_step']!=100000:
        raise RuntimeError('Existing seed42 run incomplete; cannot label valid reuse')
    if template['selection_rules']!=AUTHORITY_RULES: raise RuntimeError('Existing selection rule changed')
    rule_path=root/'final_checkpoint_selection_rule.json'
    rule=rule_document()
    if rule_path.exists() and read_json(rule_path)!=rule:raise RuntimeError('final_v1 already permanently locked; no modification allowed')
    if not rule_path.exists():save(rule_path,rule)
    source_files=protected_files()+[Path(p) for p in template['frozen_geometry_sources']]
    protocol=dict(training_seeds=list(SEEDS),validation_seeds=VALIDATION_SEEDS,test_seeds=RESERVED_TEST_SEEDS,
                  episodes=100,steps=1000,authority=.1,representative_test_seed=REPRESENTATIVE,
                  selection='final_v1: safe + mean>0 + worst>=0; X2 performance then margin/P2/activity/economic tie-break',
                  selection_warning='validation selection only; interim AuthoritySelection aliases are diagnostic, not final policies',
                  selection_rule_version=FINAL_VERSION,selection_rule_sha256=digest(rule_path),
                  template_fingerprint=fingerprint(template),template_manifest=str(SOURCE42/'experiment_manifest.json'),
                  template_manifest_sha256=digest(SOURCE42/'experiment_manifest.json'),
                  source_sha256={str(p):digest(p) for p in source_files},
                  reporting_source_sha256=digest(Path(__file__)),certification_status=CERTIFICATION_STATUS,
                  Gaussian_iid='implementation choice for reproduction',reuse_seed42=True,
                  seed_audit_sha256=digest(root/'seed_usage_audit.json'),
                  run_dirs={str(s):str(SOURCE42 if s==42 else root/f'training_seed{s}') for s in SEEDS})
    if (root/'protocol_lock.json').exists():
        old=read_json(root/'protocol_lock.json')
        # During initial preparation only, the reporting implementation can be
        # finalized without touching any frozen experimental choice. A final
        # three-policy lock freezes this runner too.
        comparable=lambda p:{k:v for k,v in p.items() if k not in ('reporting_source_sha256','seed_audit_sha256')}
        if old.get('selection_rule_version')!=FINAL_VERSION:
            # Explicit user-authorized, pre-test migration. Preserve the old
            # episode5 draft, and verify that NO algorithm/config choice differs.
            ignore={'selection','selection_warning','selection_rule_version','selection_rule_sha256'}
            if {k:v for k,v in comparable(old).items() if k not in ignore}!={k:v for k,v in comparable(protocol).items() if k not in ignore}:
                raise RuntimeError('Migration would change non-selection frozen protocol')
            history=root/'history';history.mkdir(exist_ok=True)
            for name in ('protocol_lock.json','seed42_validation_lock.json'):
                src=root/name;dest=history/('pre_final_v1_'+name)
                if src.exists():
                    if dest.exists() and digest(dest)!=digest(src):raise RuntimeError('Historical selection evidence conflict')
                    if not dest.exists():shutil.copyfile(src,dest)
        elif comparable(old)!=comparable(protocol):
            raise RuntimeError('Protocol lock differs or policies already locked; no overwrite allowed')
    save(root/'protocol_lock.json',protocol)
    commands=[]
    for s in SEEDS:
        if s==42: continue
        commands.append(dict(training_seed=s,argv=[sys.executable,'-m','evaporation.zanon2019_train',
            '--disturbance-mode','zanon2019_stochastic','--stochastic-residual-scale','0.10',
            '--authority-pilot','--episodes','100','--steps','1000','--seed',str(s),
            '--eval-every','5','--eval-seeds','10','--eval-seed-start','420000','--device','cuda',
            '--output-dir',protocol['run_dirs'][str(s)]]))
    save(root/'local_training_commands.json',commands)
    # Lock seed42's pre-existing selector result now, before further training or
    # test. This is not a final three-policy lock yet.
    selected=(load_locked_seed42(root,protocol) if old and old.get('selection_rule_version')==FINAL_VERSION
              and (root/'seed42_validation_lock.json').exists() else select_run(SOURCE42,42,protocol))
    if old and old.get('selection_rule_version')==FINAL_VERSION and (root/'seed42_validation_lock.json').exists():
        prior=read_json(root/'seed42_validation_lock.json')
        if any(selected.get(k)!=v for k,v in prior.items()):
            raise RuntimeError('Prelocked seed42 checkpoint/config/validation evidence changed')
    save(root/'seed42_validation_lock.json',selected)
    validation_reproducibility(root)
    print(f'Prepared final_v1. seed42 reused; automatically selected episode {selected["checkpoint_episode"]}. No training/test run.',flush=True)


def validate_run(path,seed,protocol):
    m=read_json(path/'experiment_manifest.json')
    if m['seed']!=seed or fingerprint(m)!=protocol['template_fingerprint']:
        raise RuntimeError(f'Frozen config mismatch in training seed {seed}')
    return m


def load_locked_seed42(root,protocol):
    """Read the already locked episode; do not rescan/reselect seed42."""
    selected=read_json(root/'seed42_validation_lock.json')
    validate_run(SOURCE42,42,protocol)
    for path,sha in ((selected['checkpoint_path'],selected['checkpoint_sha256']),
                     (selected['config_manifest'],selected['config_manifest_sha256']),
                     (selected['validation_source'],selected['validation_source_sha256'])):
        if digest(path)!=sha:raise RuntimeError('Locked seed42 evidence changed; never reselect it')
    if selected['selection_rule_version']!=FINAL_VERSION:raise RuntimeError('Locked seed42 rule mismatch')
    return selected


def training_metadata(selected,root):
    # Enrich a reporting copy only, preserving seed42_validation_lock bytes.
    p=dict(selected)
    path=p.get('run_summary')
    summary=read_json(path) if path and Path(path).exists() else None
    p['completed_episodes']=summary.get('completed_episodes') if summary else None
    p['training_completed_100x1000']=bool(summary and summary['run_state']=='completed'
        and summary['completed_episodes']==100 and summary['global_step']==100000)
    if summary:
        failure=summary.get('failure_evidence') or {}
        counters=failure.get('partial_episode_safety_counters',{})
        p['training_safety_abort']=(summary['run_state']=='evaluation_aborted' or
            any(v>0 for k,v in counters.items() if k!='W_exceedance'))
        p['failure_evidence']=failure or None
    else:p['training_safety_abort']=None
    audit=root/f'seed{p["training_seed"]}_final_selection_audit.csv'
    p['eligible_checkpoint_count']=sum(r['eligible_final_checkpoint']=='True' for r in csv_rows(audit)) if audit.exists() else None
    return p


def select_run(path,seed,protocol):
    m=validate_run(path,seed,protocol)
    summary=read_json(path/'run_summary.json'); selection=read_json(path/'checkpoint_selection.json')
    if protocol.get('selection_rule_version')!=FINAL_VERSION:raise RuntimeError('Prepare final_v1 before selecting/training')
    audits,chosen=final_selection_scan(path,m)
    root=Path(protocol['run_dirs']['2027']).parent
    write_csv(root/f'seed{seed}_final_selection_audit.csv',audits)
    completed=(summary['run_state']=='completed' and summary['completed_episodes']==100 and summary['global_step']==100000)
    if not completed:
        status='training_in_progress' if summary['run_state'] in ('running','training','initialized') else 'training_run_unsuccessful'
    else:status='successful' if chosen else 'no_stable_positive_economic_safe_checkpoint'
    chosen=chosen if completed else None
    checkpoint=Path(chosen['checkpoint_path']) if chosen else None
    return dict(training_seed=seed,checkpoint_path=str(checkpoint) if checkpoint else None,
                selected_checkpoint=str(checkpoint) if checkpoint else None,
                checkpoint_sha256=digest(checkpoint) if checkpoint else None,
                checkpoint_episode=int(chosen['episode']) if chosen else None,
                selected_episode=int(chosen['episode']) if chosen else None,
                global_step=int(chosen['global_step']) if chosen else None,
                selection_rule_version=FINAL_VERSION,eligible_final_checkpoint=bool(chosen),
                selection_key=FINAL_VERSION,status=status,run_status=status,
                mean_validation_economic_improvement=chosen['mean_economic_improvement_pct'] if chosen else None,
                worst_validation_economic_improvement=chosen['worst_seed_economic_improvement_pct'] if chosen else None,
                X2_IAE_ratio=chosen['X2_IAE_ratio'] if chosen else None,
                minimum_X2_margin=chosen['minimum_X2_margin'] if chosen else None,
                control_activity={k:chosen[k] for k in ('P100_TV_ratio','F200_TV_ratio','mean_TV_ratio','mean_RMS_du_ratio','boundary_occupancy_fraction')} if chosen else None,
                config_hash=digest(path/'experiment_manifest.json'),
                validation_metrics=chosen,alpha=.1,reward_version=m['reward'],reward_weights=m['weights'],
                config_manifest=str(path/'experiment_manifest.json'),config_manifest_sha256=digest(path/'experiment_manifest.json'),
                fingerprint=fingerprint(m),validation_source=str(path/'fixed_evaluation.csv'),
                validation_source_sha256=digest(path/'fixed_evaluation.csv'),
                best_empirical_safe_checkpoint=selection.get('best_economic_actor'),
                final_checkpoint=selection.get('final_actor'),run_summary=str(path/'run_summary.json'))


def validation_reproducibility(root):
    protocol=assert_frozen(root);policies=[];rows=[]
    for seed in SEEDS:
        path=Path(protocol['run_dirs'][str(seed)])
        if seed==42:selected=load_locked_seed42(root,protocol)
        elif (path/'run_summary.json').exists():selected=select_run(path,seed,protocol)
        else:
            attempted=(root/f'training_process_seed{seed}.json').exists()
            selected=dict(training_seed=seed,status='training_run_unsuccessful' if attempted else 'training_pending',
                run_status='training_run_unsuccessful' if attempted else 'training_pending',
                checkpoint_path=None,selected_checkpoint=None,checkpoint_sha256=None,checkpoint_episode=None,
                selected_episode=None,global_step=None,validation_metrics=None,eligible_final_checkpoint=False,
                selection_rule_version=FINAL_VERSION,best_empirical_safe_checkpoint=None,final_checkpoint=None)
        selected=training_metadata(selected,root)
        policies.append(selected)
        row=dict(training_seed=seed,run_status=selected['status'],selected_episode=selected['checkpoint_episode'],
                 eligible_final_checkpoint=selected['eligible_final_checkpoint'],selection_rule_version=FINAL_VERSION)
        row.update(selected['validation_metrics'] or {})
        rows.append(row)
    write_csv(root/'training_seed_reproducibility_validation.csv',rows)
    pending=any(p['status'] in ('training_pending','training_in_progress') for p in policies)
    count=sum(p['eligible_final_checkpoint'] for p in policies)
    verdict='pending_training_completion' if pending else 'Case A' if count==3 else 'Case B' if count==2 else 'Case C'
    save(root/'training_seed_reproducibility_validation.json',dict(successful_training_seeds=count,
        completed_training_screening=not pending,case=verdict,validation_only=True,independent_final_test_run=False,
        final_test_recommendation='not yet; training pending' if pending else 'eligible to request independent test' if count>=2 else 'pause final test; reproducibility insufficient',
        policies=policies))
    write_validation_report(root,policies,save,write_csv)
    return policies


def _is_float(value):
    try: float(value); return True
    except (ValueError,TypeError): return False


def train_pending(root):
    protocol=assert_frozen(root)
    if protocol.get('selection_rule_version')!=FINAL_VERSION:raise RuntimeError('final_v1 must be locked before training')
    if (root/'test_started.json').exists() or (root/'locked_final_policies.json').exists():
        raise RuntimeError('Policies/test locked; further training prohibited')
    for item in read_json(root/'local_training_commands.json'):
        path=Path(protocol['run_dirs'][str(item['training_seed'])])
        if (path/'run_summary.json').exists():
            validate_run(path,item['training_seed'],protocol)
            print(f'Preserving existing training evidence seed={item["training_seed"]}; never extend/retrain',flush=True)
            continue
        if path.exists() and any(path.iterdir()): raise RuntimeError(f'Partial run exists: {path}. Preserve and review; no overwrite/retrain.')
        assert_frozen(root)
        result=subprocess.run(item['argv'],cwd=REPO.parent,check=False)
        save(root/f'training_process_seed{item["training_seed"]}.json',dict(returncode=result.returncode,argv=item['argv']))
        # Training safety aborts remain unchanged. Failed seed is preserved;
        # proceed to the next seed, never alter parameters to force success.
        print(f'training seed={item["training_seed"]} process exit={result.returncode}',flush=True)
    validation_reproducibility(root)


def lock_policies(root):
    protocol=assert_frozen(root)
    lock=root/'locked_final_policies.json'
    if lock.exists(): verify_policy_lock(root); print('Existing immutable three-policy lock verified'); return
    if (root/'test_started.json').exists(): raise RuntimeError('Cannot choose checkpoints after test starts')
    policies=validation_reproducibility(root)
    if any(p['status'] in ('training_pending','training_in_progress') for p in policies):
        raise RuntimeError('Training pending; no three-policy lock fabricated')
    for selected in policies:
        seed=selected['training_seed']
        if seed==42 and any(selected.get(k)!=v for k,v in read_json(root/'seed42_validation_lock.json').items()):
            raise RuntimeError('Prelocked seed42 validation selection changed')
    # All validation-based selections are decided before test access. Copy
    # actors to a new locked artifact directory, preserving originals.
    for selected in policies:
        selected['validation_mean_economic_improvement_pct']=selected.get('mean_validation_economic_improvement')
        selected['validation_worst_seed_economic_improvement_pct']=selected.get('worst_validation_economic_improvement')
        selected['manifest_path']=selected.get('config_manifest')
        if not selected['eligible_final_checkpoint']:
            selected['locked_checkpoint_path']=None
            continue  # No fallback actor is ever masqueraded as final eligible.
        target=root/'locked_models'/f'seed{selected["training_seed"]}_episode{selected["checkpoint_episode"]:04d}.pth'
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and digest(target)!=selected['checkpoint_sha256']: raise RuntimeError('Locked actor conflict')
        if not target.exists(): shutil.copyfile(selected['checkpoint_path'],target)
        selected['locked_checkpoint_path']=str(target)
    save(lock,dict(protocol_sha256=digest(root/'protocol_lock.json'),policies=policies,
                   test_runner_source_sha256=digest(Path(__file__)),
                   selection_rule_version=FINAL_VERSION,selection_rule_sha256=digest(root/'final_checkpoint_selection_rule.json'),
                   selection_uses_test=False,test_seeds=RESERVED_TEST_SEEDS,representative=REPRESENTATIVE))
    print('Three validation-selected policies locked before final-test access',flush=True)


def verify_policy_lock(root):
    assert_frozen(root); lock=read_json(root/'locked_final_policies.json')
    if lock['protocol_sha256']!=digest(root/'protocol_lock.json'): raise RuntimeError('Protocol changed after selection')
    if lock.get('test_runner_source_sha256')!=digest(Path(__file__)):raise RuntimeError('Test/report implementation changed after policy lock')
    if [p['training_seed'] for p in lock['policies']]!=list(SEEDS): raise RuntimeError('Missing/reordered training seed')
    for p in lock['policies']:
        artifacts=[]
        if p.get('locked_checkpoint_path'):artifacts.append((p['locked_checkpoint_path'],p['checkpoint_sha256']))
        if p.get('config_manifest'):artifacts.append((p['config_manifest'],p['config_manifest_sha256']))
        if p.get('validation_source'):artifacts.append((p['validation_source'],p['validation_source_sha256']))
        for file,sha in artifacts:
            if digest(file)!=sha: raise RuntimeError(f'Locked artifact changed: {file}')
    return lock


class FinalEvidenceController(AuthorityController):
    """Passive final-test evidence only: never aborts/reselects on a violation."""
    def __init__(self,*args): super().__init__(*args); self.evidence=[]
    def act(self,state,action,**kwargs):
        if len(self.evidence)%250==0 and hasattr(self,'progress_label'):
            print(f'{self.progress_label}: step {len(self.evidence)}/1000',flush=True)
        u,info=super().act(state,action,**kwargs)
        self.evidence.append(dict(step=len(self.evidence),state=np.asarray(state).tolist(),control=np.asarray(u).tolist(),
                                  raw_action=np.asarray(action).tolist(),qp_feasible=bool(info['qp_feasible']),
                                  in_Omega=bool(info['in_Omega']),mode=info.get('mode')))
        return u,info
    def audit_next_state(self,x): self.evidence[-1]['next_state']=np.asarray(x).tolist()


def rollout(cfg,model,design,omega,domain,policy,path,seed,progress_label=None):
    ctrl=FinalEvidenceController(cfg,model,design,omega,domain)
    if progress_label is not None:ctrl.progress_label=progress_label
    try:
        _,records,_=run_episode(cfg,model,ctrl,policy,None,np.random.default_rng(seed+900000),
            training=False,global_step=0,disturbance_trajectory=path,
            initial_state_override=np.asarray(cfg.robust_economic_reference_state))
        if len(records)!=1000: raise RuntimeError('Incomplete final-test rollout')
        if not np.array_equal(np.asarray([r['disturbance'] for r in records]),path): raise RuntimeError('Paired disturbance path mismatch')
        if any(np.any(r['paper_state_shock']) for r in records): raise RuntimeError('State shock leaked into stochastic test')
        result=metrics(records,cfg,model,design,omega)
        # Include initial and terminal state for safety without changing the
        # historical pre-state IAE/ISE/G/TV definitions.
        xs=np.vstack([np.asarray([r['state'] for r in records]),ctrl.evidence[-1]['next_state']])
        for i,name in enumerate(('X2','P2')):
            result[name+'_mean']=float(xs[:-1,i].mean())
            result[name+'_min']=float(xs[:,i].min());result[name+'_max']=float(xs[:,i].max())
        margin=xs[:,0]-25
        result.update(X2_margin_mean=float(margin.mean()),X2_margin_min=float(margin.min()),
            **{f'X2_margin_p{p}':float(np.percentile(margin,p)) for p in (1,5)},
            **{f'X2_margin_fraction_lt_{s}':float((margin<v).mean()) for s,v in (('0',0),('0p05',.05),('0p10',.1),('0p20',.2))})
        result['X2_violation_count']=int((xs[:,0]<25-1e-8).sum())
        result['P2_physical_violation_count']=int(((xs[:,1]<40-1e-8)|(xs[:,1]>80+1e-8)).sum())
        result['physical_state_violation_count']=int(np.any((xs<cfg.state_lower-1e-8)|(xs>cfg.state_upper+1e-8),axis=1).sum())
        result['X2_violation_rate']=result['X2_violation_count']/1001
        result['X2_max_violation_magnitude']=float(np.maximum(-margin,0).max())
        result['X2_cumulative_violation_magnitude']=float(np.maximum(-margin,0).sum())
        result['state_safety_samples']=1001
        return dict(status='completed',metrics=result,records=records,evidence=ctrl.evidence)
    except Exception:
        return dict(status='program_failure_incomplete',metrics=None,records=[],evidence=ctrl.evidence,error=traceback.format_exc())


def archive_rollout(folder,result,path,meta):
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'summary.json',dict(status=result['status'],metrics=result['metrics'],error=result.get('error'),disturbance_meta=meta))
    save(folder/'execution_evidence.json',result['evidence'])
    if not result['records']: return
    rows=[]
    for k,r in enumerate(result['records']):
        row=dict(step=k,time_seconds=k,stage_cost=float(r['economic_cost']),
                 **{n:float(path[k,i]) for i,n in enumerate(('F1','X1','T1','T200'))})
        for key,names in (('state',('X2','P2')),('control',('P100','F200')),('w_hat',('w0','w1'))):
            row.update({n:float(r[key][i]) for i,n in enumerate(names)})
        rows.append(row)
    write_csv(folder/'trajectory.csv',rows)


def paired_row(training_seed,test_seed,base,policy,selected):
    row=dict(training_seed=training_seed,test_seed=test_seed,selected_checkpoint_episode=selected['checkpoint_episode'],
             training_run_status=selected['status'],status=policy['status'],baseline_status=base['status'],certification_status=CERTIFICATION_STATUS)
    if base['metrics'] is None or policy['metrics'] is None: return row
    b,p=base['metrics'],policy['metrics']
    row.update(baseline_J=b['J_econ'],policy_J=p['J_econ'],improvement_pct=(b['J_econ']-p['J_econ'])/b['J_econ']*100)
    for k,v in p.items():
        if isinstance(v,(int,float,np.number)): row[k]=float(v)
    for key,metric in RATIO_METRICS.items(): row[key]=relative(p[metric],b[metric])
    row['physical_violation']=p['physical_state_violation_count']+p['physical_input_violation_count']
    row['QP_infeasible']=p['QP_infeasible_count'];row['Omega_exit']=p['Omega_exit_count']
    row['minimum_X2_margin']=p['X2_margin_min']
    return row


def final_test(root,device):
    authorization=root/'independent_test_authorization.json'
    if not authorization.exists() or read_json(authorization).get('approved_for_independent_final_test') is not True:
        raise RuntimeError('Current stage is validation reproducibility only; independent final test has not been authorized. No final disturbance sampled.')
    if read_json(authorization).get('policy_lock_sha256')!=digest(root/'locked_final_policies.json'):
        raise RuntimeError('Independent-test authorization does not match locked policies')
    lock=verify_policy_lock(root); started=root/'test_started.json'
    if started.exists(): raise RuntimeError('Final test already started. No rerun/reselection; use --phase report on saved evidence.')
    # Reaudit before any final disturbance is generated. No policy test data
    # have been consulted by validation selection.
    seed_usage_audit(root)
    save(started,dict(policy_lock_sha256=digest(root/'locked_final_policies.json'),test_seeds=RESERVED_TEST_SEEDS))
    cfg,model,design,omega,domain,weights=make_setup(1000,42)
    cfg.stochastic_residual_scale=.1
    agents={}
    for selected in lock['policies']:
        if not selected.get('locked_checkpoint_path'):continue
        agent=make_agent(cfg,device);load_actor(agent,Path(selected['locked_checkpoint_path']))
        agents[selected['training_seed']]=agent
    rows=[]; baselines=[]
    for seed in RESERVED_TEST_SEEDS:
        verify_policy_lock(root)
        path,meta=sample_disturbance_path(cfg,'zanon2019_stochastic',seed,1000,[2,1,8,5],20,50)
        base=rollout(cfg,model,design,omega,domain,ZeroResidualPolicy(),path,seed,f'test {seed} baseline')
        archive_rollout(root/f'test_trajectories/seed_{seed}/baseline',base,path,meta)
        baselines.append(dict(test_seed=seed,status=base['status'],**(base['metrics'] or {})))
        for selected in lock['policies']:
            ts=selected['training_seed']
            if ts in agents:
                proposed=rollout(cfg,model,design,omega,domain,agents[ts],path,seed,f'test {seed} policy {ts}')
                archive_rollout(root/f'test_trajectories/seed_{seed}/policy_seed{ts}',proposed,path,meta)
            else:
                proposed=dict(status='no_eligible_final_policy',metrics=None,records=[],evidence=[])
            rows.append(paired_row(ts,seed,base,proposed,selected))
        write_csv(root/'final_test_per_policy.csv',rows)
        write_csv(root/'final_test_baseline.csv',baselines)
        print(f'final test {seed}: baseline once + 3 policies; progress {seed-429999}/50',flush=True)
    save(root/'test_completed.json',dict(policy_lock_sha256=digest(root/'locked_final_policies.json'),
        baseline_rollouts=50,policy_rollouts=50*len(agents),policy_unavailable_rows=50*(3-len(agents)),complete=len(rows)==150,
        program_failures=sum(r['status']!='completed' or r['baseline_status']!='completed' for r in rows),
        certification_status=CERTIFICATION_STATUS))
    report(root)


def summarize(values):
    v=np.asarray(values,dtype=float);n=len(v)
    if not n:return dict(n=0,mean=None,median=None,std=None,CI95_lower=None,CI95_upper=None,min=None,max=None,p5=None,p95=None)
    sd=float(v.std(ddof=1)) if n>1 else None
    critical={3:4.302652729911275,50:2.0095752371292397}.get(n)
    # Partial test coverage has no final CI; never quietly replace the planned
    # n=50/n=3 design with a convenient normal approximation.
    half=critical*sd/np.sqrt(n) if critical is not None and sd is not None else None
    return dict(n=n,mean=float(v.mean()),median=float(np.median(v)),std=sd,
        CI95_lower=float(v.mean()-half) if half is not None else None,
        CI95_upper=float(v.mean()+half) if half is not None else None,min=float(v.min()),max=float(v.max()),
        p5=float(np.percentile(v,5)),p95=float(np.percentile(v,95)))


def summaries(rows,policies):
    per=[]
    counters=('physical_violation','physical_state_violation_count','physical_input_violation_count','X2_violation_count',
              'P2_physical_violation_count','QP_infeasible','Omega_exit','robust_region_violation_count')
    metrics_list=('improvement_pct','X2_IAE_ratio','X2_ISE_ratio','X2_std_ratio','P2_IAE_ratio','P2_ISE_ratio','P2_std_ratio',
                  'P100_TV_ratio','F200_TV_ratio','P100_RMS_du_ratio','F200_RMS_du_ratio','G_X2_ratio','G_P2_ratio',
                  'minimum_X2_margin','W_exceedance_rate')
    detailed=[]
    for selected in policies:
        ts=selected['training_seed'];all_rows=[r for r in rows if int(r['training_seed'])==ts]
        valid=[r for r in all_rows if r['status']=='completed' and r['baseline_status']=='completed']
        s=dict(training_seed=ts,selected_checkpoint_episode=selected['checkpoint_episode'],training_run_status=selected['status'],
               test_seed_count=len(all_rows),completed_test_seed_count=len(valid),expected_test_seed_count=50,
               complete_coverage=len(all_rows)==50 and len(valid)==50)
        for k in metrics_list:
            for key,value in summarize([float(r[k]) for r in valid]).items():s[f'{k}_{key}']=value
        numeric=set.intersection(*(set(k for k,v in r.items() if _is_float(v)) for r in valid)) if valid else set()
        for k in sorted(numeric-{'training_seed','test_seed','selected_checkpoint_episode'}):
            detailed.append(dict(training_seed=ts,metric=k,level='within-training-seed independent disturbance realizations',
                                 **summarize([float(r[k]) for r in valid])))
        for k in counters:s['total_'+k]=sum(float(r[k]) for r in valid) if valid else None
        gains=[float(r['improvement_pct']) for r in valid]
        s.update(positive_improvement_seed_count=sum(v>0 for v in gains),
                 positive_improvement_fraction=sum(v>0 for v in gains)/len(gains) if gains else None,
                 positive_improvement_fraction_of_planned_50=sum(v>0 for v in gains)/50,
                 negative_improvement_worst_case=min(gains+[0.]) if gains else None,
                 number_of_test_seeds_with_any_X2_violation=sum(float(r['X2_violation_count'])>0 for r in valid),
                 number_of_test_seeds_with_any_physical_violation=sum(float(r['physical_violation'])>0 for r in valid),
                 global_minimum_X2_margin=min(float(r['minimum_X2_margin']) for r in valid) if valid else None)
        per.append(s)
    cross=[]
    for metric in metrics_list:
        values=[s[metric+'_mean'] for s in per if s['complete_coverage']]
        cross.append(dict(metric=metric,level='between-training-seed means, NOT 150 iid samples',
                          **summarize(values),coverage='all three policies' if len(values)==3 else 'incomplete; no final reproducibility claim'))
    for counter in counters:
        values=[s['total_'+counter] for s in per if s['complete_coverage']]
        cross.append(dict(metric='total_'+counter,level='sum observed counters, not independent sample inference',
                          n=len(values),total=sum(values) if len(values)==3 else None))
    return per,cross,detailed


def final_plots(root,per,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    plots=root/'figures';plots.mkdir(exist_ok=True)
    for name,keys,ylabel in (
        ('final_economic_improvement_by_training_seed',('improvement_pct',),'Economic improvement (%)'),
        ('final_X2_IAE_by_training_seed',('X2_IAE_ratio',),'X2 IAE ratio vs baseline'),
        ('final_X2_std_by_training_seed',('X2_std_ratio',),'X2 within-trajectory std ratio'),
        ('final_X2_margin_by_training_seed',('minimum_X2_margin',),'Minimum X2 margin per realization'),
        ('final_control_TV_by_training_seed',('P100_TV_ratio','F200_TV_ratio'),'Input TV ratio vs baseline')):
        fig,ax=plt.subplots(figsize=(7,4))
        for i,k in enumerate(keys):
            x=np.arange(3)+(i-.5*(len(keys)-1))*.16
            means=[s[k+'_mean'] for s in per]
            errors=[(s[k+'_CI95_upper']-s[k+'_mean']) if s[k+'_CI95_upper'] is not None else 0 for s in per]
            ax.errorbar(x,means,yerr=errors,fmt='o',capsize=4,label=k)
        ax.set(xticks=range(3),xticklabels=[str(s) for s in SEEDS],xlabel='Independent SAC training seed',ylabel=ylabel)
        ax.set_title('50 paired independent disturbance seeds per frozen policy');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(plots/(name+'.png'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for s in per:
        ax.scatter(s['improvement_pct_mean'],s['global_minimum_X2_margin'],label=f'training {s["training_seed"]}')
    ax.set(xlabel='Mean paired economic improvement (%)',ylabel='Global minimum X2 margin');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(plots/'final_economic_safety_tradeoff.png');plt.close(fig)
    base_path=root/f'test_trajectories/seed_{REPRESENTATIVE}'
    data={'baseline':csv_rows(base_path/'baseline/trajectory.csv')}
    data.update({f'policy seed{s}':csv_rows(base_path/f'policy_seed{s}/trajectory.csv') for s in SEEDS})
    for key in ('X2','P2','P100','F200','economic_cost'):
        fig,ax=plt.subplots(figsize=(8,4))
        for label,rr in data.items():
            y=[float(r['stage_cost' if key=='economic_cost' else key]) for r in rr]
            if key=='economic_cost':y=np.cumsum(y)
            ax.plot(range(len(y)),y,lw=.8,label=label)
        if key=='X2':ax.axhline(25,color='black',ls='--',lw=1.5,label='X2 quality bound')
        ax.set(xlabel='Time (s)',ylabel='Cumulative economic cost' if key=='economic_cost' else key,
               title=f'Predeclared representative independent seed {REPRESENTATIVE}')
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(plots/f'final_representative_{key}.png');plt.close(fig)


def report(root):
    lock=verify_policy_lock(root)
    if not (root/'test_completed.json').exists(): raise RuntimeError('Final test incomplete; do not generate final claims from partial results')
    rows=csv_rows(root/'final_test_per_policy.csv')
    pairs={(int(r['training_seed']),int(r['test_seed'])) for r in rows}
    if len(rows)!=150 or pairs!={(s,t) for s in SEEDS for t in RESERVED_TEST_SEEDS}: raise RuntimeError('Missing/duplicated final test seed; never omit bad results')
    per,cross,detailed=summaries(rows,lock['policies'])
    baselines=csv_rows(root/'final_test_baseline.csv')
    if len(baselines)!=50 or {int(r['test_seed']) for r in baselines}!=set(RESERVED_TEST_SEEDS):
        raise RuntimeError('Missing/duplicated baseline test seed')
    write_csv(root/'final_test_training_seed_summary.csv',per)
    write_csv(root/'final_test_cross_training_seed_summary.csv',cross)
    write_csv(root/'final_test_metric_summary_long.csv',detailed)
    full=all(s['complete_coverage'] for s in per) and all(r['status']=='completed' for r in baselines)
    if full:final_plots(root,per,rows)
    positives=sum(s['improvement_pct_mean'] is not None and s['improvement_pct_mean']>0 for s in per)
    anomalies=sum(s['total_physical_violation'] or 0 for s in per)
    baseline_anomalies=sum(float(r['physical_state_violation_count'])+float(r['physical_input_violation_count'])
                           for r in baselines if r['status']=='completed')
    train_success=all(s['training_run_status']=='successful' for s in per)
    lines=['|Training seed|Selected episode|Test coverage|Mean gain %|95% CI|Worst gain %|Positive fraction|X2 IAE ratio|F200 TV ratio|Min margin|',
           '|---|---:|---|---:|---|---:|---:|---:|---:|---:|']
    for s in per:
        def f(key):return 'unavailable' if s.get(key) is None else f'{s[key]:.6f}'
        lines.append(f'|{s["training_seed"]}|{s["selected_checkpoint_episode"]}|{s["completed_test_seed_count"]}/50|{f("improvement_pct_mean")}|'
            f'{f("improvement_pct_CI95_lower")} to {f("improvement_pct_CI95_upper")}|{f("improvement_pct_min")}|'
            f'{f("positive_improvement_fraction")}|{f("X2_IAE_ratio_mean")}|{f("F200_TV_ratio_mean")}|{f("global_minimum_X2_margin")}|')
    table='\n'.join(lines)
    headings_en=('Experimental protocol','Three independent training seeds','Independent 50-seed test','Economic performance',
                 'Disturbance rejection','Safety margin','Control activity','Training-seed reproducibility','Qualitative ECC2019 comparison','Claim limitations')
    headings_cn=('实验协议','三个独立训练种子','独立50-seed测试','经济性能','扰动抑制','安全余量','控制活动','训练种子可复现性','ECC2019定性比较','结论边界')
    common=f'''{table}

Frozen alpha=.10, 100x1000 training, validation 420000..420009, final test
430000..430049. Each disturbance path is generated once and shared by baseline
and all policies. Baseline is simulated only once per test seed. Reference for
IAE/ISE/G is fixed B, not the final trajectory mean. Input TV uses actual input.
Gaussian iid / 1 s / initial B are implementation choices for reproduction.
Safety includes the terminal state; counters are observed samples/steps, not
distinct failure episodes. W exceedance remains diagnostic, not a test rejection.
Ratios are computed within each test seed before averaging. Margin p1/p5 are
per-trajectory quantiles; global minimum is reported separately.
Within-policy t95 CI uses n=50 disturbance realizations. Between-policy t95
uses n=3 policy means (df=2), not 150 independently trained samples. Ratios>1
remain performance trade-offs, not automatic training failures.
The unchanged margin-first selector may choose economically comparable negative
gain checkpoints; test outcomes never change that choice.
'''
    conclusions=f'''Coverage complete={full}; three training runs successful={train_success}.
Positive mean gain policies={positives}/3. Observed physical violation sample
total={anomalies if full else 'unavailable/incomplete'}. Economic reproducibility
must be described using this count, not by selecting only seed42. Zero observed
physical violations across independent tests is supported only if full coverage
and the total is zero. Even then no formal Gaussian robust guarantee follows.
All per-policy metrics and activity/margins/safety counters are in the CSVs;
neither bad policies nor bad disturbance seeds are excluded.
'''
    en='# Final stochastic Experiment I\n\n'
    cn='# 最终随机工况实验 I\n\n'
    for i,(he,hc) in enumerate(zip(headings_en,headings_cn),1):
        text=common if i==1 else conclusions if i in (4,8,10) else (
            'ECC2019 remains qualitative external literature comparison only. No author std/IAE/violation rate or comparable economic gain is inferred from Fig.2.' if i==9 else
            'See the complete per-policy and hierarchical summary tables. Safety architecture, reward, observation, mapping and SAC parameters remain frozen.')
        en+=f'## {i}. {he}\n\n{text}\n\n'
        translated=(f'{table}\n\n训练100×1000，α=0.10；validation 与 final test 严格分离。每个 test seed 的扰动路径只生成一次，baseline只运行一次，三policy严格配对。'
            'Gaussian/iid、1s和初态B为 implementation choice for reproduction。IAE/ISE/G固定参考B；实际输入TV；安全统计含terminal。'
            '每个policy的50seed t95统计与三个policy均值的跨训练seed t95统计分开，不把150轨迹当作独立训练样本。' if i==1 else
            f'完整覆盖={full}；训练run全部成功={train_success}；经济均值为正policy={positives}/3；物理违约观测样本总计={anomalies if full else "缺失/不完整"}。'
            '保留负收益及坏seed，不根据test重选checkpoint。只有完整覆盖且计数为零才能写独立测试零观测违约；不能升级为formal Gaussian保证。' if i in (4,8,10) else
            'ECC2019只做外部定性比较，不提取原图std/IAE/违约率或与本文可比的经济收益。' if i==9 else
            '详见完整per-policy与分层统计CSV。安全架构、reward、observation、mapping和SAC参数全部冻结。现有安全余量优先selection可选中小幅负收益但经济可比的checkpoint。')
        cn+=f'## {i}. {hc}\n\n{translated}\n\n'
    (root/'final_stochastic_results_en.md').write_text(en,encoding='utf-8')
    (root/'final_stochastic_results_cn.md').write_text(cn,encoding='utf-8')
    save(root/'final_claims.json',dict(complete_coverage=full,training_runs_successful=train_success,
        positive_mean_economic_policies=positives,stable_positive_economic_across_three=full and train_success and positives==3,
        zero_observed_physical_violations=full and anomalies==0,
        baseline_zero_observed_physical_violations=full and baseline_anomalies==0,
        baseline_observed_physical_violation_samples=baseline_anomalies if full else None,
        policy_observed_physical_violation_samples=anomalies if full else None,
        formal_Gaussian_guarantee=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','train','validation','lock','test','report','run'),default='prepare')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();root=args.root.resolve()
    if args.phase=='prepare':prepare(root)
    elif args.phase=='train':train_pending(root)
    elif args.phase=='validation':validation_reproducibility(root)
    elif args.phase=='lock':lock_policies(root)
    elif args.phase=='test':final_test(root,args.device)
    elif args.phase=='report':report(root)
    else:
        if not (root/'protocol_lock.json').exists():prepare(root)
        train_pending(root);lock_policies(root)
        print('Stopped after final_v1 validation screening and policy lock. NO independent final test started.',flush=True)


if __name__=='__main__':main()
