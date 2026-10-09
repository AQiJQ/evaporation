"""Independent U_R provenance/authority audit; never redesigns or simulates.

Counterfactuals change standalone inequality arrays only, not cfg/design/sets.
Archived x/actions are replayed through the unchanged controller to recover z;
the nonlinear plant is NEVER advanced and no disturbance path is generated.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import csv
import inspect
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from . import zanon2019_input_tightening_audit as prior
from . import zanon2019_paired_v2 as v2
from . import zanon2019_reference2d as stage3
from . import zanon2019_economic_recovery as exp
from .theta_learning import OnlineThetaLearner
from .control import build_safety_design
from .controlled_invariant_error_set import facets, hull
from .omega_safe_action_filter_diagnosis import qp_rows
from .residual_action_space_diagnosis import verification_rows
from .zanon2019_paired_experiment import PairedEvidenceController
from .zanon2019_benchmark import write_csv, DEFAULT_DESIGN, DEFAULT_OMEGA

REPO=Path(__file__).resolve().parent
ROOT=exp.ROOT/'ur_region_audit'
CHANNELS=('P100','F200')
EPSILON_PHYSICAL=1e-6  # Diagnostic perturbation, NOT any solver/gate tolerance.


def protected_hashes():
    files=[p for p in REPO.glob('*.py') if not p.name.startswith('zanon2019_ur_region_audit')]
    files += [DEFAULT_DESIGN,DEFAULT_OMEGA,stage3.ROOT/'conservative_2d_selection.json']
    # Old audit and finite-jump certificates remain read-only too.
    for folder in (prior.ROOT, REPO/'evaporation_safe_sac/outputs_omega_dwell_shock_B'):
        files += [p for p in folder.rglob('*') if p.is_file()]
    return {str(p):prior.sha(p) for p in files}


def reconstruct(env):
    cfg,model,d=env[:3]
    saved=v2.load(prior.ROOT/'audit_summary.json')['support']
    scale=float(saved['robust_region_scale'])
    # Invoke only the actual pure region-bounds method. No learner init,
    # optimize_static_safety_design, new W samples or gain search is called.
    reader=OnlineThetaLearner.__new__(OnlineThetaLearner)
    reader.cfg,reader.model=cfg,model
    xl,xh,ul,uh=reader._robust_region_bounds(d,scale)
    lower,upper=model.physical_input(ul),model.physical_input(uh)
    np.testing.assert_allclose(lower,model.physical_input(d.robust_input_lower),rtol=0,atol=1e-10)
    np.testing.assert_allclose(upper,model.physical_input(d.robust_input_upper),rtol=0,atol=1e-10)
    np.testing.assert_allclose(lower,saved['robust_lower'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(upper,saved['robust_upper'],rtol=0,atol=1e-10)
    center=model.physical_input(d.v_ref)
    half=cfg.robust_region_initial_input_half_range.copy()
    h=np.vstack([np.eye(2),-np.eye(2),np.eye(2),-np.eye(2)])
    b=np.r_[cfg.input_upper,-cfg.input_lower,center+scale*half,-center+scale*half]
    labels=['physical_P100_upper','physical_F200_upper','physical_P100_lower','physical_F200_lower',
            'local_window_P100_upper','local_window_F200_upper','local_window_P100_lower','local_window_F200_lower']
    origins=[prior.source(OnlineThetaLearner._robust_region_bounds,'self.cfg.input_upper')]*4+[
        prior.source(OnlineThetaLearner._robust_region_bounds,'+ float(scale) * self.cfg.robust_region_initial_input_half_range')]*4
    active=[]
    for channel in range(2):
        point=center.copy();point[channel]=upper[channel]
        slacks=b-h@point
        dual=np.zeros(8);dual[4+channel]=1.
        # This is an AUDIT support LP of the constructed box, not the builder.
        np.testing.assert_allclose(h.T@dual,np.eye(2)[channel],atol=1e-12)
        for i,label in enumerate(labels):
            active.append(dict(channel=CHANNELS[channel],constraint_id=i,constraint_source=label,
                formula=f'{h[i,0]:g} P100 + {h[i,1]:g} F200 <= {b[i]:.12f}',
                slack_physical=float(slacks[i]),active=bool(abs(slacks[i])<=1e-7),
                nearly_active=bool(abs(slacks[i])<=EPSILON_PHYSICAL),
                audit_LP_multiplier=float(dual[i]),original_builder_multiplier=None,
                interpretation='configured reference-centered local input window' if i>=4 else 'physical input hard bound',
                point_P100=float(point[0]),point_F200=float(point[1]),source_file=origins[i]['file'],
                source_function=origins[i]['function'],source_line=origins[i]['line']))
    with np.load(DEFAULT_DESIGN) as archive: saved_keys=list(archive.files)
    return dict(scale=scale,original_reference_state=cfg.robust_economic_reference_state,
        original_reference_input=center,input_half_width=half,physical_lower=cfg.input_lower,
        physical_upper=cfg.input_upper,UR_lower=lower,UR_upper=upper,
        XR_lower=model.physical_state(xl),XR_upper=model.physical_state(xh),
        H_physical=h,h_physical=b,row_labels=labels,
        builder=prior.source(OnlineThetaLearner._robust_region_bounds),
        first_physical_F200_upper_expression=prior.source(OnlineThetaLearner._robust_region_bounds,'+ float(scale) * self.cfg.robust_region_initial_input_half_range'),
        downstream_intersection=prior.source(build_safety_design,'physical_u_hi = np.minimum'),
        downstream_tightening=prior.source(build_safety_design,'u_hi_t = u_hi - input_support_upper'),
        definition='U_R(rho,u_c) = U intersect product_i [u_c_i-rho*h_i, u_c_i+rho*h_i]; u_c=physical_input(seed_design.v_ref), h=[35,30]',
        actual_input=True,state_independent=True,reference_independent=False,
        frozen_runtime_reference_independent=True,mode_independent=True,
        derived_from_predecessor=False,all_u_in_UR_guarantee_next_state_invariant=False,
        purpose='declared nonlinear/affine mismatch sampling domain and hard actual-input region; joint certificate then selects an admissible scale',
        original_scale_selection=prior.source(OnlineThetaLearner.build_certified_robust_operating_design),
        indirect_gate_source=prior.source(OnlineThetaLearner._candidate_region_diagnostics),
        mismatch_domain_source=prior.source(OnlineThetaLearner._sample_robust_region_residuals),
        original_selected_scale_failure_log_status='not archived for this exact balanced-B candidate; neighboring-reference logs are not substituted',
        evidence_for_missing_log='certified_reference_search.evaluate_candidate keeps learner.robust_region_search_diagnostics in memory but saves only final row and NPZ; the NPZ keys below contain no failed-scale trace',
        saved_design_keys=saved_keys,
        original_limiting_scale_gate=None,
        caution='largest certified scale found by finite gain/angle/scale search is NOT a globally maximal robust admissible input set'),active


def window_sensitivity(env,region):
    cfg,model,d=env[:3];h,b=region['H_physical'],region['h_physical']
    all_rows=[]
    cases=[('physical_only',list(range(4)),'physical-only audit; not certified controller'),
           ('physical_plus_local_P100_window',[0,1,2,3,4,6],'explicit P100 window only'),
           ('physical_plus_local_F200_window',[0,1,2,3,5,7],'explicit F200 window only'),
           ('full_UR',list(range(8)),'frozen scale, center and window'),
           ('remove_local_P100_upper',[i for i in range(8) if i!=4],'isolated box facet removal'),
           ('remove_local_F200_upper',[i for i in range(8) if i!=5],'isolated box facet removal')]
    # These families are absent from the direct U_R formula; a missing term
    # must not be fabricated. Scale-selection effects remain unidentified.
    for family in ('X2_facets','P2_facets','W_direct_term','Z_direct_term','S','Omega','predecessor','Gm_Bj'):
        cases.append(('remove_'+family,list(range(8)),'not present directly; rho held fixed; indirect scale-selection effect not quantified'))
    for label,indices,note in cases:
        _,upper,_=prior.polytope_extrema(h[indices],b[indices])
        all_rows.append(dict(audit_configuration=label,P100_upper=float(upper[0]),F200_upper=float(upper[1]),
            P100_increase_vs_full=float(upper[0]-region['UR_upper'][0]),F200_increase_vs_full=float(upper[1]-region['UR_upper'][1]),
            rho_fixed=region['scale'],note=note,certified_for_production=False))
    for channel in range(2):
        adjusted=b.copy(); adjusted[4+channel]+=EPSILON_PHYSICAL
        _,upper,_=prior.polytope_extrema(h,adjusted)
        all_rows.append(dict(audit_configuration='epsilon_local_'+CHANNELS[channel]+'_upper',
            P100_upper=float(upper[0]),F200_upper=float(upper[1]),
            P100_increase_vs_full=float(upper[0]-region['UR_upper'][0]),F200_increase_vs_full=float(upper[1]-region['UR_upper'][1]),
            rho_fixed=region['scale'],note=f'only standalone RHS perturbed by {EPSILON_PHYSICAL:g} physical units; no tolerance change',certified_for_production=False))
    return all_rows


def plant_sensitivity(env):
    cfg,model,d=env[:3];x,u=cfg.robust_economic_reference_state,cfg.robust_economic_reference_input
    eps=1e-4;derivatives=[]
    for i in range(2):
        delta=np.eye(2)[i]*eps
        minus,plus=model.algebraic(x,u-delta,cfg.disturbance_nominal),model.algebraic(x,u+delta,cfg.disturbance_nominal)
        dx=(model.derivative(x,u+delta,cfg.disturbance_nominal)-model.derivative(x,u-delta,cfg.disturbance_nominal))/(2*eps)
        derivatives.append(dict(input=CHANNELS[i],dF4_du=(plus.f4-minus.f4)/(2*eps),dF5_du=(plus.f5-minus.f5)/(2*eps),
            dX2dot_du_per_min=float(dx[0]),dP2dot_du_per_min=float(dx[1])))
    return dict(local_physical_one_second_B=np.diag(cfg.state_scale)@d.b@np.diag(1/cfg.input_scale),
        fixed_state_algebraic_derivatives=derivatives,
        F200_path='F200 -> Q200 = UA2*(T3-T200)/(1+UA2/(2 Cp F200)) -> F5 -> (F4-F5)/pressure_capacitance; at fixed x P100, F4 independent of F200',
        F200_direction='F200 increase directly increases F5 and lowers P2; small positive next-X2 effect is indirect via pressure/temperature coupling; this does not establish a P2-active U_R facet',
        P100_path='P100 -> T100 -> Q100 -> F4; directly raises X2 and P2 rates at this reference',
        source=prior.source(model.algebraic))


def archive_sources(envs):
    sources=[]
    for role,path in [('Strong_baseline',stage3.old.ROOT/'baseline_trajectories/delta_0.00/seed_420000.csv'),
                      ('Conservative_baseline',stage3.ROOT/'baseline_trajectories/delta_0.01/seed_420000.csv')]:
        records,_=v2.archived_pair(path)
        sources.append(dict(role=role,reference='Strong' if role.startswith('Strong') else 'Primary_Conservative',
            path=path,states=np.array([r['state'] for r in records]),controls=np.array([r['control'] for r in records]),
            actions=np.array([r['raw_action'] for r in records]),disturbances=np.array([r['disturbance'] for r in records])))
    path=exp.ROOT/'conservative_mechanism_stage4_oracle_training/oracle/candidates/candidate_0002/seed_420000_trajectory.npz'
    if path.exists():
        receipt=v2.load(path.with_name('seed_420000_metrics.json'))
        if prior.sha(path)!=receipt['trace_sha256']: raise RuntimeError('Archived oracle trace changed')
        with np.load(path) as z:
            sources.append(dict(role='archived_oracle_candidate_0002',reference='Primary_Conservative',path=path,
                states=z['candidate_state'].copy(),controls=z['candidate_control'].copy(),
                actions=z['candidate_raw_action'].copy(),disturbances=z['disturbance'].copy()))
    for s in sources:
        if len(s['states'])!=1000 or len(s['actions'])!=1000: raise RuntimeError('Incomplete historical path')
        s['sha256']=prior.sha(s['path'])
    return sources


def representative_indices(states,reference):
    roles={0:['initial'],20:['early'],499:['mid_horizon'],999:['late']}
    median=np.median(states,axis=0)
    choices={'near_minimum_X2':int(np.argmin(states[:,0])),'near_maximum_P2':int(np.argmax(states[:,1])),
        'near_median_state':int(np.argmin(np.sum(((states-median)/[15,20])**2,axis=1))),
        'near_reference':int(np.argmin(np.sum(((states-reference)/[15,20])**2,axis=1)))}
    for role,index in choices.items():roles.setdefault(index,[]).append(role)
    for index in (249,749,99,899):
        if len(roles)>=8:break
        roles.setdefault(index,['additional_coverage'])
    return roles


def static_contexts(envs):
    contexts=[]
    for name,env in envs.items():
        cfg,model,d=env[:3]
        contexts.append(dict(reference=name,role='reference_reset',step=0,state=cfg.robust_economic_reference_state.copy(),
            z=d.z_ref.copy(),source=None,archived_action=None,archived_control=None))
    sources=archive_sources(envs)
    for saved in sources:
        env=envs[saved['reference']];cfg,model,d,omega,domain=env[:5]
        controller=PairedEvidenceController(cfg,model,d,omega,domain,saved['disturbances'])
        controller.reset(saved['states'][0])
        chosen=representative_indices(saved['states'],cfg.robust_economic_reference_state)
        max_error=0.
        print('read-only controller replay:',saved['role'],'seed420000; no nonlinear plant propagation',flush=True)
        for step in range(1000):
            z_before=controller.inner.z.copy()
            control,info=controller.act(saved['states'][step],saved['actions'][step])
            error=float(np.linalg.norm(control-saved['controls'][step]))
            max_error=max(max_error,error)
            if error>1e-6 or not info['qp_feasible']:
                raise RuntimeError(f'Archive controller replay differs: {saved["role"]}/{step}: {error}')
            if step in chosen:
                contexts.append(dict(reference=saved['reference'],role=saved['role']+':'+','.join(chosen[step]),step=step,
                    state=saved['states'][step].copy(),z=z_before,source=str(saved['path']),
                    archived_action=saved['actions'][step].copy(),archived_control=saved['controls'][step].copy()))
        saved['max_replay_control_error']=max_error
    return contexts,[{key:s[key] for key in ('role','reference','path','sha256','max_replay_control_error')} for s in sources]


def point_control(env,context,action):
    cfg,model,d,omega,domain=env[:5]
    ctrl=PairedEvidenceController(cfg,model,d,omega,domain,np.array([cfg.disturbance_nominal]))
    ctrl.reset(context['state']);ctrl.inner.z=context['z'].copy()
    # Only a one-state call. The returned nominal z is discarded.
    return ctrl.act(context['state'],np.asarray(action,float))


def row_family(mode,index):
    if index<4:
        channel=CHANNELS[index%2];side='upper' if index<2 else 'lower'
        return ('UR_minus_KZ_nominal_' if mode.startswith('Z_') else 'UR_actual_')+channel+'_'+side
    if mode.startswith('Z_'):
        return ('S_next_X2_upper','S_next_P2_upper','S_next_X2_lower','S_next_P2_lower')[index-4]
    return 'Omega_robust_next_facet_'+str(index-4)


def attribution(envs,contexts):
    action_rng=np.random.default_rng(190019)  # Static probes, not a disturbance seed/path.
    actions=[np.zeros(2),*[np.array(a) for a in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))],
        *action_rng.uniform(-1,1,(8,2))]
    rows=[];authorities=[];family_counts=Counter();isolated=[];snapshots=[]
    for index,context in enumerate(contexts):
        env=envs[context['reference']];cfg,model,d,omega,domain=env[:5]
        _,base_info=point_control(env,context,[0,0])
        if not base_info['qp_feasible']:raise RuntimeError('Archived point not QP feasible')
        h,b=np.asarray(base_info['action_safe_rows']),np.asarray(base_info['action_safe_bounds'])
        center=np.asarray(base_info['action_center_coordinate'])
        lo,hi,_=prior.polytope_extrema(h,b)
        mode=base_info['mode'];alpha=cfg.stochastic_residual_scale
        offset=base_info['ancillary'] if mode.startswith('Z_') else d.v_ref
        physical_range=lambda l,u:(model.physical_input(l+offset),model.physical_input(u+offset))
        app_lo,app_hi=physical_range((1-alpha)*center+alpha*lo,(1-alpha)*center+alpha*hi)
        authorities.append(dict(context=index,reference=context['reference'],role=context['role'],step=context['step'],mode=mode,
            state_X2=float(context['state'][0]),state_P2=float(context['state'][1]),
            P100_residual_min=float(alpha*(lo[0]-center[0])*cfg.input_scale[0]),
            P100_residual_max=float(alpha*(hi[0]-center[0])*cfg.input_scale[0]),
            F200_residual_min=float(alpha*(lo[1]-center[1])*cfg.input_scale[1]),
            F200_residual_max=float(alpha*(hi[1]-center[1])*cfg.input_scale[1]),
            applied_P100_min=float(app_lo[0]),applied_P100_max=float(app_hi[0]),
            applied_F200_min=float(app_lo[1]),applied_F200_max=float(app_hi[1])))
        # State-specific removals remain copies of inequalities only. No new
        # returned control is applied and no certificate is claimed.
        variants={'full':(h.copy(),b.copy())}
        if mode.startswith('Z_'):
            variants['without_S_next_rows']=(h[:4].copy(),b[:4].copy())
            bb=b.copy();bb[:4]=np.r_[d.robust_input_upper,-d.robust_input_lower]
            variants['without_KZ_term']=(h.copy(),bb)
            bb=b.copy();s=v2.load(prior.ROOT/'audit_summary.json')['support']
            pure_lo=model.normalized_input(s['pure_physical_U_minus_KZ_design_lower']);pure_hi=model.normalized_input(s['pure_physical_U_minus_KZ_design_upper'])
            bb[:4]=np.r_[pure_hi,-pure_lo];variants['physical_U_KZ_without_local_UR']=(h.copy(),bb)
        else:
            variants['without_Omega_next_rows']=(h[:4].copy(),b[:4].copy())
            oh,_=facets(omega);ws=np.max(d.w_vertices@oh.T,axis=0)
            bb=b.copy();bb[4:]+=ws;variants['without_W_next_support_only']=(h.copy(),bb)
            bb=b.copy();pl=model.normalized_input(cfg.input_lower)-d.v_ref;pu=model.normalized_input(cfg.input_upper)-d.v_ref
            bb[:4]=np.r_[pu,-pl];variants['physical_actual_U_without_local_UR']=(h.copy(),bb)
        for label,(hh,bb) in variants.items():
            vl,vh,_=prior.polytope_extrema(hh,bb);physical_lo,physical_hi=physical_range(vl,vh)
            isolated.append(dict(context=index,reference=context['reference'],role=context['role'],mode=mode,audit_configuration=label,
                full_feasible_P100_min=float(physical_lo[0]),full_feasible_P100_max=float(physical_hi[0]),
                full_feasible_F200_min=float(physical_lo[1]),full_feasible_F200_max=float(physical_hi[1]),
                certified_for_production=False,note='fixed existing state/set/dynamics; inequality-only isolation, NOT domain redesign'))
        snapshots.append(dict(context=context,mode=mode,rows=h,rhs=b,row_meanings=[row_family(mode,i) for i in range(len(b))],
            center=center,omega_next_facet_normals=facets(omega)[0] if not mode.startswith('Z_') else None))
        for k,action in enumerate(actions):
            applied,info=point_control(env,context,action)
            hh,bb=info['action_safe_rows'],info['action_safe_bounds']
            rho=float(np.max(np.abs(action)));final=np.asarray(info['action_final_coordinate'])
            boundary=np.asarray(info['interior_boundary_coordinate'])
            target_active=[] if rho==0 else np.flatnonzero(np.abs(bb-hh@boundary)<=1e-8).tolist()
            final_active=np.flatnonzero(np.abs(bb-hh@final)<=1e-9).tolist()
            meanings=[row_family(mode,i) for i in target_active]
            for family in meanings:family_counts[(context['reference'],mode,family)]+=1
            requested=np.asarray(info['requested_residual'])*cfg.input_scale
            residual=np.asarray(info['applied_residual'])*cfg.input_scale
            rows.append(dict(context=index,reference=context['reference'],role=context['role'],step=context['step'],probe=k,mode=mode,
                raw_actor_0=float(action[0]),raw_actor_1=float(action[1]),raw_rho=rho,scaled_rho=alpha*rho,
                mapping_authority_contracted=bool(rho>0 and alpha<1),
                target_active_rows=';'.join(map(str,target_active)),target_active_families=';'.join(meanings),
                final_active_rows=';'.join(map(str,final_active)),final_active_families=';'.join(row_family(mode,i) for i in final_active),
                target_min_slack=None if rho==0 else float(np.min(bb-hh@boundary)),
                final_min_slack=float(np.min(bb-hh@final)),
                QP_modified=bool(info['final_verification_gap']>1e-8),QP_feasible=bool(info['qp_feasible']),
                QP_gap_normalized=float(info['final_verification_gap']),
                final_clip_modified=bool(info.get('final_physical_clip_gap',0)>1e-9),
                physical_input_facet_active=bool(np.any(np.minimum(applied-cfg.input_lower,cfg.input_upper-applied)<=1e-6)),
                robust_actual_input_facet_active=bool(np.any(np.minimum(applied-model.physical_input(d.robust_input_lower),model.physical_input(d.robust_input_upper)-applied)<=1e-6)),
                requested_P100=float(requested[0]),requested_F200=float(requested[1]),applied_residual_P100=float(residual[0]),applied_residual_F200=float(residual[1]),
                applied_P100=float(applied[0]),applied_F200=float(applied[1]),
                residual_mapping_reconstruction_error_physical=float(np.linalg.norm(requested-residual))))
    counts=[dict(reference=k[0],mode=k[1],family=k[2],target_boundary_facet_hits=v) for k,v in sorted(family_counts.items())]
    summary=dict(context_count=len(contexts),trajectory_context_count=sum(c['source'] is not None for c in contexts),
        actions_per_context=len(actions),probe_count=len(rows),nonzero_probes=sum(r['raw_rho']>0 for r in rows),
        QP_modification_count=sum(r['QP_modified'] for r in rows),QP_infeasible_count=sum(not r['QP_feasible'] for r in rows),
        actual_clip_count=sum(r['final_clip_modified'] for r in rows),
        mapping_contraction_count=sum(r['mapping_authority_contracted'] for r in rows),
        physical_input_active_count=sum(r['physical_input_facet_active'] for r in rows),
        robust_actual_input_active_count=sum(r['robust_actual_input_facet_active'] for r in rows),
        mode_counts=dict(Counter(r['mode'] for r in rows)),target_facet_counts=counts,
        interpretation='boundary-target active facet limits FULL feasible authority; alpha contraction is mapping, not a rejected request; final-QP modifications counted separately; static probe frequencies are NOT whole-path clipping rates')
    return rows,authorities,isolated,snapshots,summary


def dependencies():
    return dict(direct_UR_inputs=['physical U','reference input u_c','input half-window [35,30]','selected scalar rho'],
        direct_UR_absent=['current x','A/B','K','W','Z','S','Omega','Gm','Bj','one-step predecessor'],
        indirect_scale_gates=['nonlinear mismatch samples on X_R x U_R x bounded external domain','W hull/inflation','Hinf','RPI','X_R minus Z','U_R minus KZ','reference feasibility','S','actual verification QP','residual authority floor'],
        finite_jump_independent=True,
        downstream=['U_R -> U_R minus KZ -> Z-mode nominal verification QP -> v+Ke -> actual robust clip',
                    'U_R -> q bounds -> Omega construction/predecessor -> Omega online QP',
                    'U_R/A/B/W/Omega -> later finite-jump Gm/Bj'],
        scope='direct construction independent of finite-jump machinery; W/Z/S influence original scale acceptance indirectly, not additive hard facets in U_R')


def run():
    if (ROOT/'ur_audit_summary.json').exists():raise RuntimeError('Preserve completed independent audit')
    before=protected_hashes();envs=prior.environments()
    region,active=reconstruct(envs['Strong'])
    sensitivity=window_sensitivity(envs['Strong'],region)
    supports=v2.load(prior.ROOT/'audit_summary.json')['support']
    physical=region['physical_upper'];upper=region['UR_upper'];nominal=np.asarray(supports['runtime_nominal_upper'])
    decomposition=[]
    for label,values,old,source in (
        ('Physical U',physical,physical,'physical upper bound'),
        ('+ local P100 window',np.array([upper[0],physical[1]]),physical,'rho*[35,30] around original Strong reference'),
        ('+ local F200 window = U_R',upper,np.array([upper[0],physical[1]]),'local_window_F200_upper; no state facet in direct formula'),
        ('+ directional KZ = nominal U_R minus KZ',nominal,upper,'control._rpi_support and build_safety_design')):
        decomposition.append(dict(Stage=label,P100_upper=float(values[0]),F200_upper=float(values[1]),
            P100_reduction=float(old[0]-values[0]),F200_reduction=float(old[1]-values[1]),active_source=source))
    margins=[]
    for name,detail in v2.load(prior.ROOT/'audit_summary.json')['references'].items():
        for i,channel in enumerate(CHANNELS):
            margins.append(dict(reference=name,channel=channel,u_ref=detail['reference_input'][i],
                physical_upper_margin=detail['physical_actual_margin_upper'][i],UR_upper_margin=detail['robust_actual_margin_upper'][i],
                nominal_tight_upper_margin=detail['nominal_reference_margin_upper'][i],
                actual_applied_residual_min=detail['actual_raw_actor_applied_residual_lower'][i],
                actual_applied_residual_max=detail['actual_raw_actor_applied_residual_upper'][i],
                zero_residual_applied=detail['zero_residual_applied'][i]))
    contexts,sources=static_contexts(envs)
    rows,authorities,isolated,poly_snapshots,stats=attribution(envs,contexts)
    after=protected_hashes()
    if before!=after:raise RuntimeError('Production/historical evidence changed')
    potential=[dict(potential_reducible_conservatism='reference-centered rectangular mismatch domain with shared scale and fixed aspect ratio',
        source=region['builder'],mathematical_reason='not a globally maximal robust admissible domain; requires local bounded mismatch certificate',
        physical_minus_UR_F200= float(physical[1]-upper[1]),
        estimated_certified_extra_F200_headroom=None,
        effect_note='150.471393369777 is removed-box geometric cap, NOT certified recoverable headroom',
        would_alter_formal_proof=True,requires_redesign_recertification=True),
        dict(potential_reducible_conservatism='finite joint scale/gain/angle search and box S inner approximation',
            source=region['original_scale_selection'],mathematical_reason='coupled X_R/U_R growth and finite bisection are implementation choices; failed-scale trace for exact candidate not saved',
            estimated_certified_extra_F200_headroom=None,would_alter_formal_proof=True,requires_redesign_recertification=True),
        dict(potential_reducible_conservatism='directional support tail enclosure and saved outer Z polygon',
            source=prior.source(__import__('evaporation.control',fromlist=['_rpi_support'])._rpi_support),
            estimated_F200_polygon_vs_directional_gap=float(supports['polygon_vs_design_support_max_difference'][1]),
            UR_upper_direct_effect=0,estimated_tail_bound_economic_headroom=None,
            would_alter_formal_proof=True,requires_redesign_recertification=True),
        dict(potential_reducible_conservatism='frozen raw-actor alpha=0.1 homothety and safe projected baseline',
            source=prior.source(__import__('evaporation.zanon2019_authority',fromlist=['AuthorityController']).AuthorityController.act),
            mathematical_reason='rho restricted before interior-anchor boundary interpolation; reduces image without changing hard polytope',
            full_vs_used_image_width_ratio=10.,UR_upper_direct_effect=0,estimated_certified_extra_F200_headroom=None,
            would_alter_formal_proof='hard geometry need not change, but action mapping/performance protocol and empirical validation would change',
            requires_redesign_recertification='not authorized; requires separate action-authority review and closed-loop validation')]
    result=dict(schema='read_only_UR_audit',region=region,dependency=dependencies(),plant=plant_sensitivity(envs['Strong']),
        window_sensitivity=sensitivity,reference_margins=margins,authority_summary=stats,archived_sources=sources,
        direct_W_Z_S_Omega_Gm_Bj_contribution_to_UR_upper=0,
        indirect_original_scale_selection_contributions=None,
        unknown_original_failure_gate_not_fabricated=True,
        no_new_plant_rollouts=True,no_new_disturbance_paths=True,no_training=True,
        scope='static affine inequality diagnostics and controller replay of existing seed420000; no continuous nonlinear proof',
        protected_hashes_before=before,protected_hashes_after=after,unchanged=True)
    ROOT.mkdir(parents=True,exist_ok=True)
    for file,data in [('ur_tightening_decomposition.csv',decomposition),('ur_active_constraints.csv',active),
                      ('ur_constraint_sensitivity.csv',sensitivity),('reference_input_margins.csv',margins),
                      ('residual_authority_attribution.csv',rows),('residual_authority_ranges.csv',authorities),
                      ('point_constraint_removal.csv',isolated),('authority_target_facets.csv',stats['target_facet_counts'])]:
        write_csv(ROOT/file,data)
    exp.save(ROOT/'potential_ur_conservatism.json',potential)
    exp.save(ROOT/'authority_polytope_snapshots.json',poly_snapshots)
    exp.save(ROOT/'ur_audit_summary.json',result)
    graph(ROOT);plots(ROOT,region,nominal);report(ROOT,result)
    print('U_R audit complete:',stats,flush=True)
    return result


def graph(root):
    text='''# U_R dependency graph

```text
physical U + seed reference u_c + configured input half-window [35,30]
                          + scalar rho
                              |
                    direct rectangular U_R(rho)
                              |
 X_R(rho) + nonlinear mismatch samples over X_R x U_R x external domain
                              |
                        W hull/inflation
                              |
                  A/B, Hinf K -> RPI Z/support
                              |
           X_R minus Z + U_R minus KZ -> box invariant S
                              |
               verification QP + residual authority floor
                              |
        joint candidate gate -> scale scan/growth/bisection -> rho_selected

FROZEN RUNTIME
 U_R minus KZ + S -> Z nominal mapping/QP -> v + Ke -> actual u in U_R
 U_R -> q bounds + A/B/W -> Omega predecessor construction -> Omega QP
 Omega/A/B/W/q bounds -> later finite-jump Gm and Bj -> event supervisor
```

U_R is independent of finite-jump machinery. The code dependency is downstream
from U_R to Omega/Gm/Bj, not upstream. W/Z/S indirectly affect the original
accepted scale; they are not separate additive facets in the direct U_R box.
No failure gate for the exact original balanced-B scale boundary is inferred
from neighboring-reference logs. New region/gain/set certification is not run.
'''
    (root/'ur_dependency_graph.md').write_text(text,encoding='utf-8')


def plots(root,region,nominal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for i,name in enumerate(CHANNELS):
        values=[float(region['physical_upper'][i]),float(region['UR_upper'][i]),float(nominal[i])]
        fig,ax=plt.subplots(figsize=(8,4.8))
        x=np.arange(3);ax.bar(x,values,color=['#697586','#356C96','#6C8D63'],width=.58)
        for j,value in enumerate(values):ax.text(j,value+6,f'{value:.6f}',ha='center',fontsize=11)
        for j in (0,1):
            reduction=values[j]-values[j+1]
            ax.annotate(f'-{reduction:.6f}',xy=(j+.5,(values[j]+values[j+1])/2),ha='center',fontsize=10)
        ax.set_xticks(x,['Physical U\nupper','Local U_R\nupper','Nominal U_R minus KZ\nupper'])
        ax.set_ylim(0,440);ax.set_ylabel(name+' upper (physical input units)')
        ax.set_title(name+' input-bound provenance')
        ax.text(.5,-.20,'Local window is a declared domain, not a state-facet LP result.',transform=ax.transAxes,ha='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        fig.tight_layout();fig.savefig(root/(name.lower()+'_tightening_waterfall.png'),dpi=160);plt.close(fig)


def report(root,result):
    r=result['region'];st=result['authority_summary'];fmt=lambda x:np.array2string(np.asarray(x),precision=12,separator=', ')
    p,rup,nom=r['physical_upper'],r['UR_upper'],np.asarray(v2.load(prior.ROOT/'audit_summary.json')['support']['runtime_nominal_upper'])
    lines=['# U_R region provenance and conservatism audit','',
        '## Definition and direct active source','',
        'U_R(rho,u_c) = U intersect product_i [u_c_i - rho h_i, u_c_i + rho h_i], h=[35,30].',
        f"rho={r['scale']}; u_c={fmt(r['original_reference_input'])}.",
        f"First builder: {r['builder']}; downstream pre-KZ intersection: {r['downstream_intersection']}.",
        f"Actual-input U_R lower/upper: {fmt(r['UR_lower'])} / {fmt(r['UR_upper'])}.",
        f"F200: min(400, {r['original_reference_input'][1]:.12f} + {r['scale']} * 30) = {rup[1]:.12f}.",
        f"P100: min(400, {r['original_reference_input'][0]:.12f} + {r['scale']} * 35) = {rup[0]:.12f}.",
        'This is NOT an LP-generated maximal input set, U intersect Pre(S), or an all-input next-state invariance certificate. The box defines the declared nonlinear-mismatch/actual-input domain. Admissible online actions must additionally pass the corresponding state-dependent QP.',
        'The active support-LP facet for F200 is local_window_F200_upper; for P100 it is local_window_P100_upper. The constructed static LP has multiplier 1 for that facet. The original builder has no LP dual. Other input can be interior, so incidental active corner facets need not be mistaken for the limiting F200 facet.',
        'These facets mean reference-centered local operating-input limits, NOT a P2/X2/S/Omega/Gm/Bj state facet. They are state-independent and mode-independent. The construction is reference-dependent; both current references inherit the frozen Strong box, so runtime physical U_R does not shift with Conservative reference.',
        '', '## Original scale selection and evidence limit','',
        f"Original search: {r['original_scale_selection']}. The grow/bisect process resamples the nonlinear mismatch domain and can refine gains/W at each candidate scale, then checks Hinf/RPI/tightening/reference/S/QP/authority gates.",
        'The exact balanced-B candidate CSV saves selected scale 1.1083984375 and its accepted gates. certified_reference_search saves final NPZ/row, not the failed-scale diagnostic trace. Therefore the specific gate that rejected the next larger scale cannot be recovered from these artifacts. Logs for the different [25.3825,50.1125] candidate are NOT substituted. No fresh redesign is performed to manufacture the missing history.',
        'W/Z/S matter INDIRECTLY to scale acceptance. At frozen rho, removing a nonexistent direct W/P2/S/Omega term from the U_R formula changes its upper by exactly zero. This is NOT a claim that removing W/S in a new certification search would leave the selected rho unchanged.',
        '', '## Addition/removal decomposition','',
        '| Stage | P100 upper | F200 upper |','|---|---:|---:|',
        f'| Physical U | {p[0]:.9f} | {p[1]:.9f} |',
        f'| Add local P100 window | {rup[0]:.9f} | {p[1]:.9f} |',
        f'| Add local F200 window: U_R | {rup[0]:.9f} | {rup[1]:.9f} |',
        f'| Subtract directional KZ: nominal interval | {nom[0]:.9f} | {nom[1]:.9f} |','',
        f"Direct local-window upper reduction: P100={p[0]-rup[0]:.12f}; F200={p[1]-rup[1]:.12f}.",
        'Removing only F200 local upper restores max F200=400 in the standalone box audit. Removing P100 upper does not change F200 max. Perturbing only F200 upper RHS by diagnostic epsilon=1e-6 increases max F200 by approximately 1e-6 (unit sensitivity). Solver/gate tolerances and production inequalities remain unchanged.',
        'There is no evidence of multiple state facets forming the direct U_R upper. W-off direct-box audit stays 249.528606630223 at frozen rho. The extra KZ subtraction is 33.225325642612; it is a separate downstream tube requirement, not part of the 400 -> U_R reduction.',
        '', '## P100 versus F200','',
        'P100 is actually MORE reduced at the U_R layer (400 ->233.655388), compared with F200 (400 ->249.528607). The important asymmetry occurs after KZ tightening: P100 max support ~7.71946, F200 ~33.22533. F200 support nearly uses all its original upper half-window 33.251953125; P100 retains large steady-reference headroom. P2 dominates KZ, but does NOT identify the direct U_R active constraint.',
        f"Physical one-second B: {fmt(result['plant']['local_physical_one_second_B'])}.",
        result['plant']['F200_path'],result['plant']['F200_direction'],
        '', '## Reference margins','',
        '| Reference/channel | To physical upper | To U_R upper | To nominal tightened upper | Applied residual min/max at reset |',
        '|---|---:|---:|---:|---|']
    for m in result['reference_margins']:
        lines.append(f"| {m['reference']}/{m['channel']} | {m['physical_upper_margin']:.12g} | {m['UR_upper_margin']:.12g} | {m['nominal_tight_upper_margin']:.12g} | {m['actual_applied_residual_min']:.9f}, {m['actual_applied_residual_max']:.9f} |")
    lines += ['', 'Neither Strong nor Conservative steady F200 is near the actual U_R upper: both retain about33.23 physical units. Conservative is near only the NOMINAL STEADY REFERENCE upper. Safe base projection gives reset F200≈212.25512 and retains bidirectional residual freedom.',
        '', '## Z/Omega and finite-jump dependencies','',
        'Z-mode: actor -> interior-anchor nominal candidate -> nominal QP (U_R minus KZ and next-z in S) -> u_actual=v+Ke -> robust actual clip. Omega-mode: actor -> actual-q polytope candidate -> QP (u_ref+q in U_R, robust next-xi in Omega) -> applied actual u. There is no second KZ subtraction from Omega actual q.',
        'U_R is independent of finite-jump machinery. U_R/q bounds feed Omega and later Gm/Bj; these sets do not construct or shrink the saved U_R. ECC2019 stochastic use has no active finite-jump automaton.',
        '', '## Archived-state authority attribution','',
        f"{st['trajectory_context_count']} archived-state contexts, plus two reference reset contexts; {st['probe_count']} static action probes. Historical x/action replay recovers moving z without plant integration or sampling any new disturbance path. Replay source SHA and exact controls are checked.",
        f"Mode probe counts: {st['mode_counts']}. QP infeasible={st['QP_infeasible_count']}; QP modifications={st['QP_modification_count']}; actual clips={st['actual_clip_count']}; mapping alpha contractions={st['mapping_contraction_count']}.",
        'See authority_target_facets.csv for FULL boundary-target active families, residual_authority_attribution.csv for per-probe actual clipping/projection, residual_authority_ranges.csv for exact ranges, and point_constraint_removal.csv for isolated S/Omega/W/local-window sensitivity.',
        'The boundary target uses full safe-polytope authority. The frozen alpha=0.1 contracts the raw-action image toward the safe baseline before final QP. A limiting boundary target is not evidence that final applied action was clipped. Static frequencies are not 1000-step learned-policy clipping rates.',
        '', '## Conservatism classification and conclusions','',
        'A Necessary under current construction: hard physical U, declared X_R/U_R mismatch domain, bounded W conditional tube certificate, directional KZ subtraction for v+Ke, and state-dependent S/Omega QP. Enlarging the declared domain invalidates relying on the old coverage/certificate without re-audit.',
        'B Approximation: axis-aligned domain windows, box controlled-invariant S, finite W hull/inflation and eigenbasis-tail support enclosure. No globally maximal-domain comparison is available, so the certified recoverable F200 headroom is NOT quantified. The saved polygon/directional-support gap has essentially zero F200 magnitude; its P100 certificate-contract concern remains in the previous audit.',
        'C Implementation choices: h=[35,30], shared rho for X_R/U_R channels, finite grow/bisect/gain search, baseline reserve and alpha=0.1 mapping. They are not universal necessities of Hinf/RPI theory. The physical-minus-window gap is a geometric cap, NOT evidence all of it can safely be recovered.',
        'D No obvious duplicate mathematical subtraction was found. U_R and nominal U_R minus KZ apply to different variables; final input rows/clips enforce actual bounds. Do not confuse declared local domain and online feasible action set.',
        'Economic headroom is geometrically restricted by U_R and downstream nominal tightening, but actual economics limitation cannot be causally assigned mainly to U_R from this static audit. Strong/Conservative difference is largely steady nominal admissibility; current effective authority is also strongly contracted by the frozen mapping scale. No safe gain from relaxing U_R is established.',
        'If authorized in a later task, study domain/aspect-ratio and certificate-aware geometry feasibility before more blind SAC training. Do not automatically enlarge U_R, change W/sets, or alter mapping based on these counterfactuals. A new domain requires coverage validation and complete relevant re-certification, including downstream Omega/Gm/Bj when affected.',
        '', '## Preservation and tests','',
        'All production sources, K/W/Z/S/Omega, references, previous audit outputs and finite-jump saved files were hash-checked unchanged. No optimization result was connected to the controller. No SAC training, Oracle continuation, new final seeds, commit or push. Test results are in tests_receipt.json.']
    (root/'UR_REGION_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def tests():
    before=protected_hashes()
    v2.tests(SimpleNamespace(root=ROOT))  # Existing full test runner; writes only this independent root.
    if before!=protected_hashes():raise RuntimeError('Tests mutated production or prior evidence')
    receipt=v2.load(ROOT/'tests_receipt.json');receipt.update(UR_production_hashes_unchanged=True,
        audit_source_sha256={p.name:prior.sha(p) for p in REPO.glob('zanon2019_ur_region_audit*.py')})
    exp.save(ROOT/'tests_receipt.json',receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tests',action='store_true')
    args=parser.parse_args();tests() if args.tests else run()
