"""final_v1: validation-only final policy selection. No actor training/rollout.

The X2 near-tie rule uses bands anchored at the smallest remaining ratio.
It is order-independent, unlike a pairwise tolerance comparator (which can
be non-transitive). Economics is a strict eligibility gate and last tie-break.
"""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path
import numpy as np

VERSION='final_v1'
VALIDATION=list(range(420000,420010))
RULE={
    'version':VERSION,'created_before_final_test':True,
    'validation_seed_list':VALIDATION,
    'learned_gate':'global_step >= manifest warmup_steps; all ten paired 1000-step validation realizations present',
    'safety_gate':{'physical_state_violation':0,'physical_input_violation':0,'QP_infeasible':0,'Omega_exit':0},
    'economic_gate':{'mean_economic_improvement_pct':'> 0','worst_seed_economic_improvement_pct':'>= 0'},
    'economic_formula':'100*(J_baseline-J_policy)/J_baseline; mean of paired per-seed gains',
    'W_exceedance':'recorded diagnostic; never rejection condition',
    'ranking_order':['ascending X2_IAE_ratio bands','descending minimum_X2_margin',
        'descending p1_X2_margin','descending p5_X2_margin','ascending fraction_margin_lt_0p05',
        'ascending fraction_margin_lt_0p10','ascending P2_IAE_ratio','ascending P2_ISE_ratio',
        'ascending P2_std_ratio','ascending mean_TV_ratio','ascending mean_RMS_du_ratio',
        'ascending boundary_occupancy_fraction','descending mean_economic_improvement_pct','ascending episode'],
    'tie_tolerances':{'X2_IAE_ratio_absolute_band':0.01,'other_ranking_fields':0.0,'economic_gate':0.0},
    'band_definition':'Repeatedly take smallest remaining X2 IAE ratio r; band contains ratios <= r+0.01; sort band by remaining lexicographic fields. No pairwise chaining.',
    'no_X2_ratio_hard_gate':True,
    'no_eligible_status':'no_stable_positive_economic_safe_checkpoint',
    'final_test_selection_allowed':False,
    'certification_status':'empirical_only_under_ECC2019_stochastic_disturbance',
}


def rule_document():
    return {**RULE,'selector_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def finite(value):
    x=float(value)
    if not np.isfinite(x):raise ValueError('Non-finite validation metric')
    return x


def aggregate(episode,rows,global_step,warmup):
    rows=sorted(rows,key=lambda r:int(r['seed']))
    complete=(len(rows)==10 and [int(r['seed']) for r in rows]==VALIDATION)
    avg=lambda key:float(np.mean([finite(r[key]) for r in rows]))
    total=lambda key:int(sum(finite(r[key]) for r in rows))
    gains=[]
    for r in rows:
        base=finite(r['baseline_J_econ']);cost=finite(r['SAC_J_econ'])
        if base<=0:raise ValueError('Expected positive baseline cost')
        gains.append((base-cost)/base*100)
        if int(r['global_step'])!=global_step:raise ValueError('Validation global_step mismatch')
        if finite(r['paired_disturbance_identity_max_error'])!=0:complete=False
        if int(r['baseline_steps'])!=1000 or int(r['SAC_steps'])!=1000:complete=False
    counters={'physical_state_violation':total('SAC_physical_state_violation_steps'),
              'physical_input_violation':total('SAC_physical_input_violation_count'),
              'QP_infeasible':total('SAC_QP_infeasible_count'),'Omega_exit':total('SAC_Omega_exit_count')}
    safety=all(v==0 for v in counters.values())
    mean=float(np.mean(gains));worst=float(min(gains))
    positive=mean>0 and worst>=0
    result=dict(episode=episode,global_step=global_step,post_warmup=global_step>=warmup,
        validation_complete=complete,validation_seed_count=len(rows),**counters,
        physical_violation=counters['physical_state_violation']+counters['physical_input_violation'],
        input_violation=counters['physical_input_violation'],empirical_safety_gate=safety,
        mean_economic_improvement_pct=mean,worst_seed_economic_improvement_pct=worst,
        positive_validation_seed_fraction=sum(v>0 for v in gains)/len(gains),
        nonnegative_validation_seed_fraction=sum(v>=0 for v in gains)/len(gains),
        stable_positive_economic_validation=positive,
        minimum_X2_margin=min(finite(r['SAC_X2_margin_min']) for r in rows),
        p1_X2_margin=min(finite(r['SAC_X2_margin_p1']) for r in rows),
        p5_X2_margin=min(finite(r['SAC_X2_margin_p5']) for r in rows),
        fraction_margin_lt_0p05=avg('SAC_X2_margin_fraction_lt_0p05'),
        fraction_margin_lt_0p10=avg('SAC_X2_margin_fraction_lt_0p10'),
        boundary_occupancy_fraction=avg('SAC_any_outer_10pct_fraction'),
        W_exceedance_rate=avg('SAC_W_exceedance_rate'),
        robust_region_violation=total('SAC_robust_region_violation_count'),
        eligible_final_checkpoint=complete and global_step>=warmup and safety and positive,
        final_rank=None,selection_rule_version=VERSION)
    for key in ('X2_IAE_ratio','X2_ISE_ratio','P2_IAE_ratio','P2_ISE_ratio','P2_std_ratio',
                'P100_TV_ratio','F200_TV_ratio','P100_RMS_du_ratio','F200_RMS_du_ratio'):
        # Recompute paired ratios from absolute columns, not rounded summaries.
        metric={'P100_RMS_du_ratio':'P100_delta_RMS','F200_RMS_du_ratio':'F200_delta_RMS'}.get(key,key.removesuffix('_ratio'))
        ratios=[]
        for r in rows:
            b=finite(r['baseline_'+metric]);p=finite(r['SAC_'+metric])
            ratio=1.0 if abs(b)<=1e-12 and abs(p)<=1e-12 else p/max(abs(b),1e-12)
            ratios.append(ratio)
        result[key]=float(np.mean(ratios))
    result['mean_TV_ratio']=.5*(result['P100_TV_ratio']+result['F200_TV_ratio'])
    result['mean_RMS_du_ratio']=.5*(result['P100_RMS_du_ratio']+result['F200_RMS_du_ratio'])
    reasons=[]
    if not complete:reasons.append('incomplete_or_unpaired_validation')
    if global_step<warmup:reasons.append('pre_warmup_diagnostic')
    if not safety:reasons.append('empirical_safety_gate_failed')
    if mean<=0:reasons.append('nonpositive_mean_economics')
    if worst<0:reasons.append('negative_worst_validation_seed')
    result['rejection_reason']=';'.join(reasons)
    return result


def secondary_key(r):
    return (-r['minimum_X2_margin'],-r['p1_X2_margin'],-r['p5_X2_margin'],
        r['fraction_margin_lt_0p05'],r['fraction_margin_lt_0p10'],r['P2_IAE_ratio'],
        r['P2_ISE_ratio'],r['P2_std_ratio'],r['mean_TV_ratio'],r['mean_RMS_du_ratio'],
        r['boundary_occupancy_fraction'],-r['mean_economic_improvement_pct'],r['episode'])


def rank(rows):
    remaining=[r for r in rows if r['eligible_final_checkpoint']]
    ordered=[];band=0
    while remaining:
        anchor=min(r['X2_IAE_ratio'] for r in remaining)
        group=[r for r in remaining if r['X2_IAE_ratio']<=anchor+0.01]
        for r in sorted(group,key=secondary_key):
            r['final_rank']=len(ordered)+1;r['X2_ratio_band']=band;r['X2_ratio_band_anchor']=anchor
            ordered.append(r)
        episodes={r['episode'] for r in group}
        remaining=[r for r in remaining if r['episode'] not in episodes];band+=1
    return ordered


def scan(path,manifest):
    path=Path(path)
    with (path/'fixed_evaluation.csv').open(encoding='utf-8-sig',newline='') as f: fixed=list(csv.DictReader(f))
    with (path/'checkpoint_pareto.csv').open(encoding='utf-8-sig',newline='') as f: statuses=list(csv.DictReader(f))
    episodes=[int(s['episode']) for s in statuses]
    if len(episodes)!=len(set(episodes)):raise ValueError('Duplicate validation checkpoint episode')
    if set(episodes)!={int(r['episode']) for r in fixed}:raise ValueError('Validation status/per-seed episodes disagree')
    audits=[]
    for s in statuses:
        ep=int(s['episode'])
        data=[r for r in fixed if int(r['episode'])==ep]
        row=aggregate(ep,data,int(s['global_step']),manifest['warmup_steps'])
        actor=path/'models/evaluation'/f'episode_{ep:04d}_actor.pth'
        row['checkpoint_path']=str(actor.resolve());row['artifact_available']=actor.is_file()
        if not actor.is_file():
            row['eligible_final_checkpoint']=False;row['rejection_reason']+=';missing_checkpoint_artifact'
        audits.append(row)
    ordered=rank(audits)
    return sorted(audits,key=lambda r:r['episode']),ordered[0] if ordered else None
