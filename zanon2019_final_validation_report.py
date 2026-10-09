"""Validation reporting only. Never selects a checkpoint or samples disturbance.

final_v1 and locked actors are read-only. Statistics distinguish conditional
within-policy validation variation and variation between eligible policies.
"""
import csv
import json
from pathlib import Path
import numpy as np


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def statistic(values):
    v=np.asarray(values,dtype=float)
    return dict(n=len(v),mean=float(v.mean()) if len(v) else None,
        sample_std=float(v.std(ddof=1)) if len(v)>1 else None,
        min=float(v.min()) if len(v) else None,max=float(v.max()) if len(v) else None)


def enrich(selected):
    """Add reporting fields to a COPY, never to seed42's immutable lock."""
    if not selected.get('eligible_final_checkpoint'):return {},[]
    rows=[r for r in read_rows(selected['validation_source']) if int(r['episode'])==selected['checkpoint_episode']]
    rows.sort(key=lambda r:int(r['seed']))
    if [int(r['seed']) for r in rows]!=list(range(420000,420010)):
        raise RuntimeError('Selected policy requires ten unique validation seeds')
    data=dict(selected['validation_metrics']);within=[]
    paired={
        'X2_IAE_ratio':'X2_IAE','X2_ISE_ratio':'X2_ISE','X2_std_ratio':'X2_std',
        'P2_IAE_ratio':'P2_IAE','P2_ISE_ratio':'P2_ISE','P2_std_ratio':'P2_std',
        'P100_TV_ratio':'P100_TV','F200_TV_ratio':'F200_TV',
        'P100_RMS_du_ratio':'P100_delta_RMS','F200_RMS_du_ratio':'F200_delta_RMS',
        'G_X2_ratio':'G_X2_emp','G_P2_ratio':'G_P2_emp'}
    distributions={}
    for name,key in paired.items():
        values=[]
        for r in rows:
            base=float(r['baseline_'+key]);policy=float(r['SAC_'+key])
            values.append(1. if abs(base)<=1e-12 and abs(policy)<=1e-12 else policy/max(abs(base),1e-12))
        distributions[name]=values
    distributions['mean_economic_improvement_pct']=[100*(float(r['baseline_J_econ'])-float(r['SAC_J_econ']))/float(r['baseline_J_econ']) for r in rows]
    for key,values in distributions.items():
        s=statistic(values)
        if key in data and not np.isclose(data[key],s['mean'],atol=1e-12,rtol=1e-12):
            raise RuntimeError(f'Reporting disagrees with immutable selection: {key}')
        data[key]=s['mean']
        half=2.2621571627409915*s['sample_std']/np.sqrt(10)
        within.append(dict(training_seed=selected['training_seed'],metric=key,
            **s,CI95_lower=s['mean']-half,CI95_upper=s['mean']+half,
            scope='conditional descriptive validation interval; reused development seeds, not final test'))
    return data,within


def build_summary(policies,rows,history):
    eligible=[r for r in rows if r['eligible_final_checkpoint']]
    statuses={p['training_seed']:p['status'] for p in policies}
    pending=sum(s in ('training_pending','training_in_progress') for s in statuses.values())
    completed=sum(p.get('training_completed_100x1000',p['status'] in ('successful','no_stable_positive_economic_safe_checkpoint')) for p in policies)
    finished=3-pending
    n=len(eligible)
    case='pending_training_completion' if pending else 'Case A' if n==3 else 'Case B' if n==2 else 'Case C'
    summary=dict(n_training_seeds=3,n_completed_runs=completed,n_finished_runs=finished,n_pending_runs=pending,
        n_runs_with_eligible_checkpoint=n,eligible_policy_count=n,success_fraction=n/3,
        case=case,policy_aggregate_population='eligible selected policies ONLY; null policies excluded, never filled with zero',
        validation_only=True,independent_final_test_run=False,certification_status='empirical_only_under_ECC2019_stochastic_disturbance',
        status_by_training_seed=statuses,
        case_definition=('pending' if pending else 'strong_validation_training_seed_reproducibility' if n==3 else
                         'partial_training_seed_reproducibility' if n==2 else 'insufficient_training_seed_reproducibility'),
        final_test_recommendation=('wait for remaining training runs' if pending else
            'may request final independent test; do not auto-run' if n==3 else
            'review with failed training seed explicitly retained; do not auto-run' if n==2 else 'pause final test'),
        stable_positive_economic_validation_across_all_three=not pending and n==3,
        validation_counter_scope='all available SAC validation checkpoint executions, including rejected checkpoints; not just selected policies; occurrence counts, not iid events',
        validation_history_runs_available=len(history))
    mappings={
        'mean_economic_improvement_pct':('mean_of_selected_policy_economic_improvements','std_between_training_seed_economic_improvements'),
        'X2_IAE_ratio':('mean_X2_IAE_ratio','std_between_training_seed_X2_IAE_ratio'),
        'X2_std_ratio':('mean_X2_std_ratio','std_between_training_seed_X2_std_ratio'),
        'minimum_X2_margin':('mean_minimum_X2_margin','std_between_training_seed_minimum_X2_margin'),
        'P100_TV_ratio':('mean_P100_TV_ratio','std_between_training_seed_P100_TV_ratio'),
        'F200_TV_ratio':('mean_F200_TV_ratio','std_between_training_seed_F200_TV_ratio')}
    for key,(mean_key,std_key) in mappings.items():
        stat=statistic([r[key] for r in eligible]);summary[mean_key]=stat['mean'];summary[std_key]=stat['sample_std']
        if key=='mean_economic_improvement_pct':
            summary['min_training_seed_mean_economic_improvement']=stat['min']
            summary['max_training_seed_mean_economic_improvement']=stat['max']
    summary['between_training_seed_statistics_provisional']=pending>0
    for label,fields in (
        ('physical_violations',('SAC_physical_state_violation_steps','SAC_physical_input_violation_count')),
        ('QP_infeasible',('SAC_QP_infeasible_count',)),('Omega_exit',('SAC_Omega_exit_count',))):
        summary['total_validation_'+label]=sum(int(float(r[k])) for rr in history for r in rr for k in fields) if history else None
    summary['F200_TV_increase_policy_count']=sum(r['F200_TV_ratio']>1 for r in eligible)
    summary['F200_TV_increase_repeated_across_all_three']=all(r['F200_TV_ratio']>1 for r in eligible) if n==3 and not pending else None
    summary['F200_TV_ratios_by_training_seed']={r['training_seed']:r.get('F200_TV_ratio') for r in rows}
    return summary


def plots(root,rows,within,qa_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'savefig.dpi':180,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    seeds=[42,2027,314159];by_seed={r['training_seed']:r for r in rows}
    lookup={(r['training_seed'],r['metric']):r for r in within}
    for filename,metrics,ylabel in (
        ('training_seed_validation_economic_improvement.png',('mean_economic_improvement_pct',),'Paired economic improvement (%)'),
        ('training_seed_validation_X2_IAE.png',('X2_IAE_ratio',),'X2 IAE ratio vs zero residual'),
        ('training_seed_validation_X2_margin.png',('minimum_X2_margin',),'Minimum X2 margin (percentage points)'),
        ('training_seed_validation_control_activity.png',('P100_TV_ratio','F200_TV_ratio'),'Actual-input TV ratio vs zero residual')):
        fig,ax=plt.subplots(figsize=(8,4));points=0
        for i,key in enumerate(metrics):
            for j,seed in enumerate(seeds):
                r=by_seed[seed]
                if not r['eligible_final_checkpoint']:continue
                value=r[key];stats=lookup.get((seed,key));error=stats['CI95_upper']-stats['mean'] if stats else 0
                display={'mean_economic_improvement_pct':'Economic improvement','X2_IAE_ratio':'X2 IAE ratio',
                         'minimum_X2_margin':'Minimum X2 margin','P100_TV_ratio':'P100 TV ratio','F200_TV_ratio':'F200 TV ratio'}[key]
                ax.errorbar(j+(i-.5*(len(metrics)-1))*.15,value,yerr=error,fmt='o',capsize=4,
                            color=('#0072B2','#D55E00')[i],label=display if j==next(k for k,s in enumerate(seeds) if by_seed[s]['eligible_final_checkpoint']) else None)
                points+=1
        for j,seed in enumerate(seeds):
            if not by_seed[seed]['eligible_final_checkpoint']:
                ax.text(j,.5,'no eligible policy',transform=ax.get_xaxis_transform(),ha='center',fontsize=9)
        ax.set(xticks=range(3),xticklabels=[str(s) for s in seeds],xlabel='SAC training seed',ylabel=ylabel,xlim=(-.5,2.5))
        ax.set_title('SYNTHETIC UNIT-TEST FIXTURE (not experiment data)' if qa_only else
                     'Validation only: reused development disturbance seeds')
        if points:ax.legend(fontsize=8)
        else:ax.set_ylim(0,1)
        ax.grid(alpha=.2)
        fig.text(.02,.01,'Error bars: within-policy descriptive t95 (n=10); no pooled 30-sample IID CI. Margin has no CI.',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,1));fig.savefig(Path(root)/filename);plt.close(fig)


def write_report(root,policies,save_json,write_csv):
    rows=[];within=[];history=[]
    for p in policies:
        metrics,stats=enrich(p);within.extend(stats)
        row=dict(training_seed=p['training_seed'],run_status=p['status'],selected_episode=p['checkpoint_episode'],
            eligible_final_checkpoint=p['eligible_final_checkpoint'],selection_rule_version='final_v1',
            eligible_checkpoint_count=p.get('eligible_checkpoint_count'),training_completed_100x1000=p.get('training_completed_100x1000',False),
            completed_episodes=p.get('completed_episodes'),training_safety_abort=p.get('training_safety_abort'))
        for key in ('mean_economic_improvement_pct','worst_seed_economic_improvement_pct','positive_validation_seed_fraction',
            'X2_IAE_ratio','X2_ISE_ratio','X2_std_ratio','P2_IAE_ratio','P2_ISE_ratio','P2_std_ratio','minimum_X2_margin',
            'p1_X2_margin','p5_X2_margin','P100_TV_ratio','F200_TV_ratio','P100_RMS_du_ratio','F200_RMS_du_ratio',
            'G_X2_ratio','G_P2_ratio','physical_violation','input_violation','QP_infeasible','Omega_exit','W_exceedance_rate'):
            row[key]=None
        row.update(metrics);rows.append(row)
        path=p.get('validation_source')
        if path and Path(path).exists():history.append(read_rows(path))
    write_csv(Path(root)/'training_seed_reproducibility_validation.csv',rows)
    if within:write_csv(Path(root)/'training_seed_validation_within_seed_metrics.csv',within)
    summary=build_summary(policies,rows,history)
    save_json(Path(root)/'training_seed_reproducibility_summary.json',summary)
    if not summary['n_pending_runs']:plots(root,rows,within)
    return summary
