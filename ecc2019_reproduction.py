"""Exploratory reproduction, not used as the primary quantitative benchmark.

Historical ECC2019 comparison pipeline; no SAC/safety changes. Retained for
provenance, not a recommendation to continue numerical reproduction/training.

review/import-development are read-only with respect to existing experiments.
train-ecc ONLY learns isolated NMPC parameters and must be explicitly requested.
development requires a trained ECC artifact; never compares naive as RL-tuned.
Final held-out seeds are prohibited here. See the feasibility report first.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import time
import sys
import numpy as np

# The isolated solver environment reads existing benchmark dependencies only.
# It does not install into, overwrite, or train the existing CUDA/SAC environment.
_shared_site=Path(__file__).parent/'.venv-cuda/Lib/site-packages'
if Path(sys.prefix).name=='.venv-ecc2019' and _shared_site.exists():
    sys.path.append(str(_shared_site))

from .config import ExperimentConfig
from .model import EvaporatorModel
from .ecc2019_nmpc import ECCNMPC, ECCSettings, LABEL, initial_theta, theta_certificate, fit_td
from .zanon2019_benchmark import sample_disturbance_path, PAPER_VARIANCES_F1_X1_T1_T200, write_csv

REPO=Path(__file__).parent
OUTPUT=REPO/'evaporation_safe_sac/outputs_ecc2019_reproduction'
PDF=Path('C:/Users/cushy/Desktop/Practical_Reinforcement_Learning_of_Stabilizing_Economic_MPC.pdf')
SOURCES={
    'proposed_alpha010_ep100':REPO/'evaporation_safe_sac/authority_cuda_seed42_100x1000/outputs_zanon2019_alpha010_seed42_100x1000',
    'proposed_alpha020_rewardcal_ep100':REPO/'evaporation_safe_sac/outputs_zanon2019_alpha020_rewardcal_seed42_100x1000'}
DEV_SEEDS=list(range(420000,420010));REFERENCE=np.array([25.,49.743]);B=np.array([25.39,50.125])


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,data):
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,default=encode,allow_nan=False),encoding='utf-8')


def common_model():
    return EvaporatorModel(ExperimentConfig(benchmark_profile='zanon2016',experiment_mode='proposed'))


def protocol(model):
    c=model.cfg
    return {'model_source_sha256':digest(REPO/'model.py'),'config_source_sha256':digest(REPO/'config.py'),'dt_min':c.dt_min,
            'initial_state':B.tolist(),'common_reference':REFERENCE.tolist(),'paired_diagnostic_reference':B.tolist(),
            'nominal':c.disturbance_nominal.tolist(),'variances':PAPER_VARIANCES_F1_X1_T1_T200.tolist(),
            'disturbance_mode':'zanon2019_stochastic','steps':1000,
            'economic_cost_convention':'existing next_state/final_actual_input/current_exogenous cost',
            'source_origin':'implementation_choice_for_reproduction','cost_excludes_slack_and_SAC_shaping':True}


def assumptions(settings=ECCSettings()):
    entries=[]
    def item(name,value,category,where,note=''):
        entries.append(dict(name=name,value=value,category=category,source=where,note=note))
    A='specified_in_paper';D='derived_from_cited_model/current_common_evaporation_model';C='implementation_choice_for_reproduction'
    for name,value in [('horizon',10),('state_lower',[25,40]),('state_upper',[100,80]),('input_lower',[100,100]),('input_upper',[400,400]),
                       ('variances_F1_X1_T1_T200',[2,1,8,5]),('slack_linear_weight',[1]*4),('slack_quadratic_matrix','I4'),
                       ('damping_alpha',.01),('policy_update_steps',500),('exploration_probability',.1)]:
        item(name,value,A,'Numerical Example pp.2261-2262')
    item('NLP_structure','non-condensed quadratic arrival/stage/terminal; nominal nonlinear step+cf; learned state bounds',A,'Eq3 / Numerical Example')
    item('RL_update','full constrained TD squared-error fit Eq8; frozen target V; damped theta update',A,'Section IV Eq8')
    item('reported_gains',{'vs_naive':14,'vs_nominal_economic':12},A,'p2262','Not robust-baseline gains; no explicit algebraic formula.')
    item('plant/economic_expression','existing model.py;10.09*(F2+F3)+600*F100+.6*F200',D,'ECC cites26/27; current common model')
    item('nominal_F1_X1_T1_T200',[10,5,40,25],D,'current common benchmark')
    for name,value in asdict(settings).items():
        if name not in {'horizon','damping_alpha','policy_update_steps','exploration_probability'}:
            item(name,value,C,'not specified in the paper' if name!='fit_every' else 'Section IV says full fit at each step',
                 'Non-unit fit_every is a separately labeled algorithm approximation, never exact schedule.' if name=='fit_every' else '')
    for name,value in [('distribution','iid zero-mean Gaussian each1s step; negative F1/X1 clipped/logged'),
                       ('initial_physical_state',B.tolist()),('sampling_seconds',1),('comparison_steps',1000),
                       ('common_reference',REFERENCE.tolist()),('quadratic_convention','0.5*v.T H v+h.T v+c; physical-unit deviations from nominal x=[25,49.743], corresponding nonlinear steady u'),
                       ('terminal_constraint','same learned soft state bounds at k=N; no invented hard terminal set'),
                       ('initial_terminal_H','1e-6 I instead of zero to meet strict PD; logged inconsistency'),
                       ('soft_bound_sign','[xl-x,x-xu]<=slack; paper prints reversed signs'),
                       ('exploration_literal','a=clip(N(0,sqrt10),100,400); sqrt10 chosen std, independent components; near-always100'),
                       ('true_TD_stage_cost','common economic cost + sum physical-bound deficit + sum deficit^2'),
                       ('constant_gauge','c_lambda=0 in fitting; uniform objective scale10000 for NLP'),
                       ('gain_formula','(J_reference-J_RL)/J_reference*100'),('Fig2_sign','cost_RL-cost_naive; paper sign not explicit'),
                       ('development_seeds',DEV_SEEDS),('economic_gap_tolerances_pct',[.5,1.])]:
        item(name,value,C,'not specified in the paper; implementation choice for reproduction')
    return {'label':LABEL,'exact_reproduction':False,'paper_sha256':digest(PDF),
            'entries':entries,'protocol':protocol(common_model()),'warning':'No original learned theta or authors solver/discretization is supplied; Fig2 is never digitized.'}


def metrics(states,controls,costs,disturbances,model,slack=None):
    states,controls,costs=map(np.asarray,(states,controls,costs));c=model.cfg
    if states.shape!=(len(costs)+1,2) or controls.shape!=(len(costs),2):raise ValueError('Expected full terminal state for safety audit')
    # Performance integrates left-endpoint T samples; safety includes terminal T+1 states.
    x=states[:-1];error=x-REFERENCE;paired=x-B;noise=(disturbances-c.disturbance_nominal)/np.sqrt(PAPER_VARIANCES_F1_X1_T1_T200)
    denominator=float(np.linalg.norm(noise));deficit=np.maximum(25-states[:,0],0);margin=states[:,0]-25
    out={'steps':len(costs),'state_safety_samples':len(states),'J_econ':float(costs.sum()),'mean_stage_cost':float(costs.mean()),
         'median_stage_cost':float(np.median(costs)),'X2_violation_count':int((deficit>1e-8).sum()),
         'X2_violation_rate':float((deficit>1e-8).mean()),'X2_max_violation_magnitude':float(deficit.max()),
         'X2_cumulative_violation_magnitude':float(deficit.sum()),
         'physical_state_violation_count':int(np.any((states<c.state_lower-1e-8)|(states>c.state_upper+1e-8),axis=1).sum()),
         'physical_input_violation_count':int(np.any((controls<c.input_lower-1e-8)|(controls>c.input_upper+1e-8),axis=1).sum()),
         'X2_margin_mean':float(margin.mean()),'X2_margin_min':float(margin.min())}
    physical_slack=np.c_[np.maximum(c.state_lower-states[1:],0),np.maximum(states[1:]-c.state_upper,0)]
    out['J_econ_plus_physical_soft_cost']=float(costs.sum()+physical_slack.sum()+np.sum(physical_slack**2))
    for p in (1,5,50):out[f'X2_margin_p{p}']=float(np.percentile(margin,p))
    for t,name in ((0,'0'),(.05,'0p05'),(.1,'0p10'),(.2,'0p20')):out['X2_margin_fraction_lt_'+name]=float((margin<t).mean())
    for i,name in enumerate(('X2','P2')):
        out.update({f'{name}_mean':float(x[:,i].mean()),f'{name}_std':float(x[:,i].std()),f'{name}_min':float(states[:,i].min()),
                    f'{name}_max':float(states[:,i].max()),f'{name}_IAE':float(np.abs(error[:,i]).sum()),f'{name}_ISE':float((error[:,i]**2).sum()),
                    f'{name}_peak_deviation':float(np.abs(error[:,i]).max()),
                    f'{name}_IAE_B_diagnostic':float(np.abs(paired[:,i]).sum()),f'{name}_ISE_B_diagnostic':float((paired[:,i]**2).sum()),
                    f'G_{name}_emp':float(np.linalg.norm(paired[:,i])/max(denominator,1e-12)),
                    f'G_{name}_centered_emp':float(np.linalg.norm(x[:,i]-x[:,i].mean())/max(denominator,1e-12))})
        for p in (1,5,50,95,99):out[f'{name}_p{p}']=float(np.percentile(x[:,i],p))
    for i,name in enumerate(('P100','F200')):
        du=np.diff(controls[:,i]);sign=np.sign(du[np.abs(du)>1e-6]);lo,hi=c.input_lower[i],c.input_upper[i]
        eta=np.abs(controls[:,i]-(lo+hi)/2)/((hi-lo)/2)
        out.update({f'{name}_mean':float(controls[:,i].mean()),f'{name}_std':float(controls[:,i].std()),
                    f'{name}_min':float(controls[:,i].min()),f'{name}_max':float(controls[:,i].max()),
                    f'{name}_TV':float(np.abs(du).sum()),f'{name}_RMS_du':float(np.sqrt(np.mean(du**2))) if len(du) else 0.,
                    f'{name}_sign_change_count':int((sign[1:]!=sign[:-1]).sum()),
                    f'{name}_lower_bound_steps':int(np.isclose(controls[:,i],lo,atol=1e-6,rtol=0).sum()),
                    f'{name}_upper_bound_steps':int(np.isclose(controls[:,i],hi,atol=1e-6,rtol=0).sum()),
                    f'{name}_outer_10pct_steps':int((eta>.9).sum())})
    if slack is not None:
        slack=np.asarray(slack);out.update(slack_count=int(np.any(slack>1e-6,axis=1).sum()),slack_mean=float(slack.mean()),
                                           slack_max=float(slack.max()),slack_cumulative=float(slack.sum()))
    return out


def read_ours_archive(source,seed,prefix,model):
    path=source/f'evaluation_trajectories/episode_0100/seed_{seed}.csv'
    with path.open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=1000:raise ValueError(f'Incomplete archived trajectory: {path}')
    x=np.array([[float(r[f'{prefix}_state_{i}']) for i in range(2)] for r in rows])
    u=np.array([[float(r[f'{prefix}_control_{i}']) for i in range(2)] for r in rows])
    d=np.array([[float(r[k]) for k in ('F1','X1','T1','T200')] for r in rows]);cost=np.array([float(r[f'{prefix}_economic_cost']) for r in rows])
    generated,generator_meta=sample_disturbance_path(model.cfg,'zanon2019_stochastic',seed,1000,PAPER_VARIANCES_F1_X1_T1_T200,20,50)
    if not np.array_equal(d,generated) or not np.array_equal(x[0],B):raise ValueError('Archive differs from paired disturbance/initial-state protocol')
    next_states=np.array([model.step(a,b,w) for a,b,w in zip(x,u,d)])
    if np.max(np.abs(next_states[:-1]-x[1:]))>1e-8:raise ValueError('Archive does not match current nonlinear plant')
    calculated=np.array([model.economic_cost(a,b,w) for a,b,w in zip(next_states,u,d)])
    if np.max(np.abs(cost-calculated))>1e-7:raise ValueError('Archive economic convention mismatch')
    states=np.vstack([x,next_states[-1]])
    internal={'QP_infeasible_count':sum(r[f'{prefix}_qp_feasible']=='False' for r in rows),
              'Omega_exit_count':sum(r[f'{prefix}_in_Omega_before']=='False' for r in rows),
              'robust_region_violation_count':sum(r[f'{prefix}_robust_region_violation']=='True' for r in rows)}
    with (source/'fixed_evaluation.csv').open(encoding='utf-8') as stream:
        fixed=[r for r in csv.DictReader(stream) if int(r['episode'])==100 and int(r['seed'])==seed]
    if len(fixed)!=1:raise ValueError('Missing unique episode100 archived evaluation provenance')
    for key in ('W_exceedance_count','W_exceedance_rate','W_max_facet_excess','P100_lower_bound_steps','P100_upper_bound_steps',
                'F200_lower_bound_steps','F200_upper_bound_steps','P100_outer_10pct_steps','F200_outer_10pct_steps'):
        internal[key]=float(fixed[0][f'{prefix}_{key}'])
    return {'states':states,'controls':u,'costs':cost,'disturbances':d,'disturbance_meta':generator_meta,'source':str(path),'source_sha256':digest(path),
            'metrics':metrics(states,u,cost,d,model),'internal_safety':internal}


def validate_sources():
    manifests={name:json.loads((path/'experiment_manifest.json').read_text(encoding='utf-8')) for name,path in SOURCES.items()}
    geometry=None
    for name,m in manifests.items():
        if m['mode']!='zanon2019_stochastic' or m['steps']!=1000 or m['validation_seeds']!=DEV_SEEDS:raise ValueError('Frozen source protocol mismatch')
        if not np.array_equal(m['variance_F1_X1_T1_T200'],PAPER_VARIANCES_F1_X1_T1_T200):raise ValueError('Frozen variance mismatch')
        g=m['frozen_geometry_sources']
        for p,h in g.items():
            if digest(p)!=h:raise ValueError('Frozen geometry changed')
        if geometry is not None and g!=geometry:raise ValueError('Candidates have different safety geometry')
        geometry=g
    return manifests


def aggregate(rows):
    result=[]
    for method in sorted({r['method'] for r in rows if r['status']=='available'}):
        subset=[r for r in rows if r['method']==method and r['status']=='available']
        keys=[k for k,v in subset[0].items() if isinstance(v,(float,int,np.floating,np.integer)) and not isinstance(v,bool) and k!='seed']
        for metric in keys:
            v=np.array([r[metric] for r in subset]);std=float(v.std(ddof=1)) if len(v)>1 else None
            radius=2.262*std/np.sqrt(len(v)) if len(v)==10 else (1.96*std/np.sqrt(len(v)) if std is not None else None)
            result.append(dict(method=method,label=LABEL if method=='ecc2019_reproduction' else method,metric=metric,n_seeds=len(v),mean=float(v.mean()),sample_std=std,
                               CI95_lower=float(v.mean()-radius) if radius is not None else None,CI95_upper=float(v.mean()+radius) if radius is not None else None,
                               min=float(v.min()),max=float(v.max()),CI_definition='paired development realization statistics; NOT training-seed CI'))
    return result


def comparison(out,ecc_artifact=None,plot=False):
    model=common_model();manifests=validate_sources();rows=[];internal=[];paths={};records={};ecc=None
    if ecc_artifact:
        saved=json.loads(Path(ecc_artifact).read_text(encoding='utf-8'))
        if saved.get('label')!=LABEL or saved.get('status')!='trained' or saved.get('successful_full_fits',0)<1:raise ValueError('Not a successfully trained ECC artifact; naive is NOT RL-tuned')
        if saved['protocol']!=protocol(model):raise ValueError('ECC artifact plant/protocol mismatch')
        if saved.get('controller_source_sha256')!=digest(REPO/'ecc2019_nmpc.py'):raise ValueError('ECC controller implementation differs from fitted artifact')
        ecc=ECCNMPC(model,ECCSettings(**saved['settings']));ecc.theta=np.asarray(saved['theta'])
    locked={name:{'checkpoint':str(source/'models/evaluation/episode_0100_actor.pth'),
                  'sha256':digest(source/'models/evaluation/episode_0100_actor.pth'),
                  'selection':'predeclared episode100, existing validation only; no held-out access'} for name,source in SOURCES.items()}
    if (out/'development_candidate_registry.json').exists():
        if json.loads((out/'development_candidate_registry.json').read_text(encoding='utf-8'))!=locked:raise ValueError('Locked development candidate registry changed')
    save_json(out/'development_candidate_registry.json',locked)
    for seed in DEV_SEEDS:
        current={}
        for name,source in SOURCES.items():
            current[name]=read_ours_archive(source,seed,'SAC',model)
        current['robust_zero_residual']=read_ours_archive(next(iter(SOURCES.values())),seed,'baseline',model)
        other=read_ours_archive(SOURCES['proposed_alpha020_rewardcal_ep100'],seed,'baseline',model)
        if not np.array_equal(other['states'],current['robust_zero_residual']['states']) or not np.array_equal(other['controls'],current['robust_zero_residual']['controls']):raise ValueError('Paired robust baseline changed across experiments')
        d=current['robust_zero_residual']['disturbances'];paths[seed]=digest(current['robust_zero_residual']['source'])
        if ecc is not None:
            ecc.warm.clear();x=B.copy();xs=[x.copy()];us=[];cs=[];slacks=[];statuses=[]
            for k,w in enumerate(d):
                solved=ecc.solve(x);u=solved['action'];xn=model.step(x,u,w)
                if not np.isfinite(xn).all():raise RuntimeError('Nonfinite ECC nonlinear plant')
                cs.append(model.economic_cost(xn,u,w));us.append(u);xs.append(xn.copy());slacks.append(solved['first_slack']);statuses.append(solved['solver_status']);x=xn
                if k%100==0:print(f'development ECC seed={seed} step={k}/1000',flush=True)
            current['ecc2019_reproduction']={'states':np.array(xs),'controls':np.array(us),'costs':np.array(cs),'disturbances':d,
                                            'metrics':metrics(xs,us,cs,d,model,slack=slacks)}
        else:
            rows.append(dict(method='ecc2019_reproduction',seed=seed,status='unavailable',label=LABEL,
                             reason='No trained ECC parameter artifact; naive controller is not an RL-tuned comparison'))
        for name,data in current.items():
            rows.append(dict(method=name,seed=seed,status='available',label=LABEL if name=='ecc2019_reproduction' else name,**data['metrics']))
            if 'internal_safety' in data:internal.append(dict(method=name,seed=seed,certification_status='empirical_only_under_ECC2019_stochastic_disturbance',**data['internal_safety']))
            trajectory=[{'step':k,'time_seconds':k,'X2':data['states'][k,0],'P2':data['states'][k,1],
                         'next_X2':data['states'][k+1,0],'next_P2':data['states'][k+1,1],
                         'P100':data['controls'][k,0],'F200':data['controls'][k,1],'stage_cost':data['costs'][k],
                         **{key:float(d[k,i]) for i,key in enumerate(('F1','X1','T1','T200'))}} for k in range(1000)]
            write_csv(out/f'trajectories/{name}/seed_{seed}.csv',trajectory)
        records[seed]=current
        print(f'development seed={seed}: archived ours verified; ECC={"available" if ecc else "unavailable"}',flush=True)
    write_csv(out/'ecc2019_development_comparison.csv',rows);write_csv(out/'per_seed_comparison.csv',rows)
    write_csv(out/'aggregate_method_comparison.csv',aggregate(rows));write_csv(out/'ours_internal_safety_diagnostics.csv',internal)
    matched=[]
    for r in rows:
        if r['status']!='available' or r['method']=='ecc2019_reproduction':continue
        peers=[p for p in rows if p['seed']==r['seed'] and p['method']=='ecc2019_reproduction' and p['status']=='available']
        if peers:
            e=peers[0];gap=(r['J_econ']-e['J_econ'])/e['J_econ']*100
            matched.append(dict(method=r['method'],ecc_label=LABEL,seed=r['seed'],economic_gap_vs_ecc2019_pct=gap,
                                primary_comparable=abs(gap)<=.5,sensitivity_comparable=abs(gap)<=1,
                                X2_std_difference=r['X2_std']-e['X2_std'],X2_min_margin_difference=r['X2_margin_min']-e['X2_margin_min'],
                                X2_violation_count_difference=r['X2_violation_count']-e['X2_violation_count']))
        else:matched.append(dict(method=r['method'],ecc_label=LABEL,seed=r['seed'],status='unavailable',reason='ECC trained reproduction not yet available'))
    write_csv(out/'matched_economic_comparison.csv',matched)
    save_json(out/'comparison_protocol.json',dict(protocol=protocol(model),source_manifests={n:digest(p/'experiment_manifest.json') for n,p in SOURCES.items()},
              frozen_design_hashes=next(iter(manifests.values()))['frozen_geometry_sources'],frozen_SAC_candidates=locked,
              paired_disturbance_identity_verified=True,archived_nonlinear_transition_verified=True,archived_economic_cost_verified=True,
              disturbance_metadata={seed:records[seed]['robust_zero_residual']['disturbance_meta'] for seed in DEV_SEEDS},
              final_test_performed=False,ECC_label=LABEL,ECC_available=ecc is not None,
              performance_state_samples='1000 left endpoints',safety_state_samples='1001 including final endpoint',
              G_emp_definition='||state-B||2 / ||exogenous disturbance standardized by sqrt(literal variance)||2; NOT Hinf norm',
              boundary_definition='COMMON physical100..400 inputs; frozen ours robust-bound occupancy is separate internal diagnostic',
              qualitative_Fig2_status='pending review of trained ECC traces, no numerical curve fitting',
              economics_comparison_excludes_slack_and_shaping=True))
    if plot:plots(out,records,ecc is not None)
    text=['# ECC2019 development comparison','',f'ECC controller label: {LABEL}',
          f'Trained ECC available: {ecc is not None}. Final50seed test: NOT RUN.',
          'This is development only, not final paper statistics. No SAC retraining or test-based selection.',
          'Archives independently checked against identical disturbance arrays, initial B, existing plant and stage cost.',
          'Main IAE/ISE reference=[25,49.743]; B-relative diagnostics separate. Safety includes terminal state.',
          'If ECC is unavailable, economic gaps and better-safety conclusions are unavailable, NOT zero.',
          'No inference from missing ECC about comparable economics or rare violations. No forced14%/12% gains.',
          'Fig2 qualitative consistency requires the learned ECC trajectory; figures alone do not certify reproduction.',
          'Do not proceed to50final seeds until learned ECC solver/parameter/qualitative review and one prelocked SAC candidate are approved.']
    text += ['', '| Method | J_econ mean | X2 mean | X2 std (mean within-seed) | X2 global range | X2 violation count |',
             '|---|---:|---:|---:|---|---:|']
    for name in ('ecc2019_reproduction','robust_zero_residual',*SOURCES):
        selected=[r for r in rows if r['method']==name and r['status']=='available']
        if not selected:text.append(f'| {name} | unavailable | unavailable | unavailable | unavailable | unavailable |');continue
        text.append(f"| {name} | {np.mean([r['J_econ'] for r in selected]):.6f} | {np.mean([r['X2_mean'] for r in selected]):.6f} | {np.mean([r['X2_std'] for r in selected]):.6f} | {min(r['X2_min'] for r in selected):.6f} to {max(r['X2_max'] for r in selected):.6f} | {sum(r['X2_violation_count'] for r in selected)} |")
    text += ['', 'SAC vs robust baseline is a performance-margin/economics trade-off, not a requirement to dominate every metric.',
             'External success requires comparable J and empirical safety/rejection advantages over a validated TRAINED ECC controller.',
             'Solver-smoke is NAIVE, not RL-tuned; it is excluded from every comparison above.']
    (out/'ecc2019_reproduction_report.md').write_text('\n'.join(text),encoding='utf-8')
    return rows


def plots(out,records,ecc_available):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    first=records[DEV_SEEDS[0]]
    for variable in ('X2','P2','P100','F200','cumulative_economic_cost','X2_safety_margin'):
        fig,ax=plt.subplots(figsize=(9,4))
        for name,data in first.items():
            values={'X2':data['states'][:,0],'P2':data['states'][:,1],'P100':data['controls'][:,0],
                    'F200':data['controls'][:,1],'cumulative_economic_cost':np.r_[0,np.cumsum(data['costs'])],
                    'X2_safety_margin':data['states'][:,0]-25}[variable]
            ax.plot(np.arange(len(values)),values,label=LABEL if name=='ecc2019_reproduction' else name,lw=1)
        if variable=='X2':ax.axhline(25,color='black',ls='--',label='X2>=25')
        if variable=='X2_safety_margin':ax.axhline(0,color='black',ls='--')
        ax.set_xlabel('Time (s), shared1s implementation');ax.set_ylabel(variable);ax.legend(fontsize=7)
        ax.set_title('Development; '+('ECC assumptions reproduction included' if ecc_available else 'ECC UNAVAILABLE; ours only'))
        fig.tight_layout();fig.savefig(out/f'comparison_{variable}.png',dpi=150);plt.close(fig)
    methods=list(first);fig,ax=plt.subplots(figsize=(9,4))
    values=[np.concatenate([records[s][name]['states'][:,0] for s in DEV_SEEDS]) for name in methods]
    ax.boxplot(values,tick_labels=methods);ax.axhline(25,color='black',ls='--');ax.tick_params(axis='x',labelrotation=12)
    ax.set_title('Development X2 distribution; pooled visualization, NOT independent statistical samples')
    fig.tight_layout();fig.savefig(out/'comparison_X2_distribution.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    if ecc_available:
        for name in methods:
            if name=='ecc2019_reproduction':continue
            gaps=[(records[s][name]['metrics']['J_econ']/records[s]['ecc2019_reproduction']['metrics']['J_econ']-1)*100 for s in DEV_SEEDS]
            ax.scatter(gaps,[records[s][name]['metrics']['X2_margin_min'] for s in DEV_SEEDS],label=name)
        ax.axvspan(-.5,.5,color='gray',alpha=.15);ax.legend(fontsize=7)
    else:ax.text(.5,.5,'ECC trained parameters unavailable\nMatched economics NOT evaluated',ha='center',va='center',transform=ax.transAxes)
    ax.set_xlabel('Economic cost gap vs ECC (%)');ax.set_ylabel('Minimum X2 margin');fig.tight_layout()
    fig.savefig(out/'comparison_economics_safety.png',dpi=150);plt.close(fig)


def train_ecc(args):
    # This function is NEVER called by review, smoke or development.
    settings=ECCSettings(gamma=args.gamma,fit_every=args.fit_every,fit_window=args.fit_window,fit_max_iter=args.fit_max_iter)
    model=common_model();solver=ECCNMPC(model,settings);theta=initial_theta(settings);policy_theta=theta.copy();target_theta=theta.copy()
    x=B.copy();rng=np.random.default_rng(args.training_seed+77);history=[];logs=[];successful=0
    path,_=sample_disturbance_path(model.cfg,'zanon2019_stochastic',args.training_seed,args.training_steps,PAPER_VARIANCES_F1_X1_T1_T200,20,50)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True);artifact=out/'ecc2019_learned_theta.json'
    if artifact.exists():raise RuntimeError('Preserve existing ECC training evidence; choose fresh --output-dir')
    def persist(status,error=None):
        save_json(artifact,dict(label=LABEL,status=status,theta=policy_theta,latest_theta=theta,
                  settings=asdict(settings),successful_full_fits=successful,completed_steps=len(history),
                  controller_source_sha256=digest(REPO/'ecc2019_nmpc.py'),
                  policy_refresh_steps=settings.policy_update_steps,protocol=protocol(model),
                  assumptions=assumptions(settings),training_seed=args.training_seed,error=error,
                  exact_paper_update_schedule=settings.fit_every==1,SAC_training_performed=False))
        write_csv(out/'ecc2019_training_log.csv',logs)
    persist('training_incomplete')
    try:
        for k,w in enumerate(path):
            explored=rng.random()<settings.exploration_probability
            u=np.clip(rng.normal(0,np.sqrt(10),size=2),model.cfg.input_lower,model.cfg.input_upper) if explored else solver.solve(x,policy_theta)['action']
            xn=model.step(x,u,w)
            if not np.isfinite(xn).all():raise RuntimeError('ECC training plant became nonfinite')
            deficit=np.r_[np.maximum(model.cfg.state_lower-xn,0),np.maximum(xn-model.cfg.state_upper,0)]
            ell=model.economic_cost(xn,u,w)+float(deficit.sum()+deficit@deficit)
            history.append((x.copy(),u.copy(),xn.copy(),ell));x=xn
            fit={}
            if (k+1)%settings.fit_every==0:
                theta,fit=fit_td(solver,history[-settings.fit_window:],theta,target_theta);successful+=1
            if (k+1)%settings.policy_update_steps==0:policy_theta=theta.copy();target_theta=theta.copy()
            logs.append(dict(step=k+1,explored=explored,X2=x[0],P2=x[1],P100=u[0],F200=u[1],true_stage_cost=ell,**fit))
            if (k+1)%500==0:
                persist('training_incomplete');print(f'ECC fit step={k+1} count={successful} explored={explored}',flush=True)
        # Last parameter deployment is explicit, not an unlogged early policy refresh.
        if args.training_steps%settings.policy_update_steps:raise ValueError('Training length must end on a500-step refresh for a trained deployment artifact')
        if successful<1:raise RuntimeError('No successful ECC fit; cannot label artifact trained')
        persist('trained')
    except Exception as exc:persist('training_failed',str(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('review','solver-smoke','train-ecc','import-development','development'))
    p.add_argument('--output-dir',type=Path,default=OUTPUT);p.add_argument('--ecc-parameters',type=Path)
    p.add_argument('--training-seed',type=int,default=419001);p.add_argument('--training-steps',type=int,default=100000)
    p.add_argument('--gamma',type=float,default=.99);p.add_argument('--fit-every',type=int,default=1)
    p.add_argument('--fit-window',type=int,default=32);p.add_argument('--fit-max-iter',type=int,default=100)
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if args.training_seed in range(420000,420010) or args.training_seed in range(430000,430050):raise ValueError('Training cannot use development/final comparison seeds')
    if not 0<args.gamma<=1 or min(args.fit_every,args.fit_window,args.fit_max_iter,args.training_steps)<1:raise ValueError('Invalid ECC settings')
    if args.mode=='train-ecc':train_ecc(args);return
    save_json(out/'ecc2019_assumptions.json',assumptions())
    if args.mode=='review':print('Feasibility reviewed; exact reproduction=false; trained ECC parameters required. No training/test.');return
    if args.mode=='solver-smoke':
        model=common_model();s=ECCNMPC(model);v=s.solve(B);q=s.solve(B,first_action=v['action'],gradient=True)
        result=dict(label=LABEL,status='naive_solver_smoke_NOT_RL_tuned',value=v['value'],Q_at_V=q['value'],
                    action=v['action'],slack=v['first_slack'],PD=theta_certificate(s.theta),
                    Q_V_difference=abs(q['value']-v['value']),solver_status=v['solver_status'],training_performed=False)
        save_json(out/'solver_smoke.json',result);print(json.dumps(result,default=lambda x:x.tolist()));return
    if args.mode=='development' and args.ecc_parameters is None:raise ValueError('Development three-method comparison requires trained --ecc-parameters; use import-development only for explicitly incomplete ours archive report')
    for filename in ('per_seed_comparison.csv','ecc2019_development_comparison.csv'):
        if (out/filename).exists():raise RuntimeError('Preserve existing comparison; choose a fresh --output-dir')
    comparison(out,args.ecc_parameters,plot=True)


if __name__=='__main__':main()
