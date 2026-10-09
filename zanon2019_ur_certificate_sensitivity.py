"""Preregistered, isolated fixed-certificate U_R study. NEVER trains SAC.

CERTIFIED_PASS means the existing bounded-W *numerical acceptance protocol*
passes, including finite nonlinear coverage. It is not a continuous-domain
nonlinear proof or a Gaussian-disturbance safety certificate. K/W/Z/S/Omega,
the affine predictor, actor, alpha, reward and controller code are frozen.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import csv
import hashlib
import shutil
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from . import zanon2019_input_tightening_audit as prior
from . import zanon2019_ur_region_audit as old
from . import zanon2019_economic_recovery as exp
from . import zanon2019_paired_v2 as v2
from . import zanon2019_paired_experiment as io
from .model import EvaporatorModel
from .control import (_rpi_support, estimate_hinf_norm, safe_projected_base,
                      project_qp_2d, point_in_convex_polygon)
from .controlled_invariant_error_set import bounds_and_domains, certificate, facets
from .residual_action_space_diagnosis import verification_rows
from .omega_interior_anchor import action_for_interior_point
from .zanon2019_benchmark import DEFAULT_DESIGN, DEFAULT_OMEGA, write_csv, make_agent, load_actor

ROOT = exp.ROOT / 'ur_certificate_sensitivity'
RHO = 1.1083984375
ALPHA = .1
DEV = tuple(range(420000, 420010))
FROZEN = ('a','b','affine','k','w_vertices','rpi_boundary','rpi_support_lower',
          'rpi_support_upper','robust_state_lower','robust_state_upper',
          'invariant_lower','invariant_upper','z_ref','v_ref',
          'nominal_policy_gain','nominal_policy_offset','theta_h','theta_p')

def require_counterfactual_disk_space(root):
    # Resource preflight only. Never delete historical results automatically.
    available=shutil.disk_usage(root).free
    if available<2*1024**3:
        raise OSError(28,f'Counterfactual requires >=2 GiB free disk space; available={available}. Preserve results and free space before resuming.')

def env():
    return v2.guard(exp.args_for())

def array_hash(x):
    x = np.asarray(x)
    return hashlib.sha256(str(x.shape).encode()+str(x.dtype).encode()+x.tobytes()).hexdigest()

def geometry_hashes(e):
    return {**{k: array_hash(getattr(e[2],k)) for k in FROZEN}, 'Omega': array_hash(e[3])}

def protected_hashes():
    files = [p for p in io.REPO.glob('*.py') if not p.name.startswith('zanon2019_ur_certificate_')]
    files += [DEFAULT_DESIGN, DEFAULT_OMEGA, io.REPO/'evaporation_safe_sac/paired_economic_v2/smoothness_regularization_config.json']
    # Preserve the exact frozen actor and archived development evidence.
    run = exp.run_dir(exp.ROOT,'strong',42)
    files += [p for p in run.rglob('*') if p.is_file()]
    for folder in (prior.ROOT, old.ROOT, io.REPO/'evaporation_safe_sac/outputs_omega_dwell_shock_B'):
        files += [p for p in folder.rglob('*') if p.is_file()]
    return {str(p): prior.sha(p) for p in files}

def candidates():
    out = [dict(candidate='baseline', family='baseline', hP=35., hF=30., rhoP=RHO, rhoF=RHO)]
    for i, factor in enumerate((.90,1.,1.05,1.10,1.15,1.20)):
        out.append(dict(candidate=f'A{i}',family='shared_rho',hP=35.,hF=30.,rhoP=RHO*factor,rhoF=RHO*factor))
    for i, hf in enumerate((30.,32.5,35.,37.5,40.,45.,50.)):
        out.append(dict(candidate=f'B{i}',family='F200_half_width',hP=35.,hF=hf,rhoP=RHO,rhoF=RHO))
    for i, factor in enumerate((1.,1.05,1.10,1.15,1.20)):
        out.append(dict(candidate=f'C{i}',family='independent_rho',hP=35.,hF=30.,rhoP=RHO,rhoF=RHO*factor))
    for i, (fp,ff) in enumerate(product((.95,1.,1.05),(1.,1.05,1.10,1.15))):
        out.append(dict(candidate=f'C2_{i}',family='independent_2D',hP=35.,hF=30.,rhoP=RHO*fp,rhoF=RHO*ff))
    return out

def independent(e, c):
    cfg = deepcopy(e[0]); model = EvaporatorModel(cfg); d = deepcopy(e[2])
    center = cfg.robust_economic_reference_input
    half = np.array([c['hP']*c['rhoP'],c['hF']*c['rhoF']])
    lower, upper = np.maximum(cfg.input_lower,center-half), np.minimum(cfg.input_upper,center+half)
    d.robust_input_lower, d.robust_input_upper = model.normalized_input(lower), model.normalized_input(upper)
    acl = d.a+d.b@d.k
    # Exactly the directional infinite-sum KZ support used by the builder.
    kl = _rpi_support(acl,d.w_vertices,-d.k,max_terms=cfg.rpi_series_max_terms)
    ku = _rpi_support(acl,d.w_vertices,d.k,max_terms=cfg.rpi_series_max_terms)
    d.input_rpi_support_lower, d.input_rpi_support_upper = kl, ku
    d.u_lower_tight, d.u_upper_tight = d.robust_input_lower+kl, d.robust_input_upper-ku
    d.x_lower_tight = d.robust_state_lower+d.rpi_support_lower
    d.x_upper_tight = d.robust_state_upper-d.rpi_support_upper
    domain,_ = bounds_and_domains(cfg,model,d)
    result=(cfg,model,d,np.array(e[3],copy=True),domain,upper-lower,deepcopy(e[6]))
    if geometry_hashes(result)!=geometry_hashes(e):
        raise RuntimeError('Frozen geometry mutated')
    return result

def fixed_gates(e):
    cfg,m,d=e[:3]; acl=d.a+d.b@d.k
    radius=float(max(abs(np.linalg.eigvals(acl))))
    norm=float(estimate_hinf_norm(d.a,d.b,d.k,cfg.hinf_q,cfg.hinf_r,points=4096))
    lo=_rpi_support(acl,d.w_vertices,-np.eye(2),max_terms=cfg.rpi_series_max_terms)
    hi=_rpi_support(acl,d.w_vertices,np.eye(2),max_terms=cfg.rpi_series_max_terms)
    hz,bz=facets(d.rpi_boundary)
    polygon_excess=float(max(np.max((d.rpi_boundary@acl.T+w)@hz.T-bz) for w in d.w_vertices))
    return dict(spectral_radius=radius,sampled_Hinf_norm=norm,gamma=cfg.hinf_gamma,
        Hinf=bool(radius<1 and norm<cfg.hinf_gamma),
        RPI=bool(radius<1 and np.all(lo<=d.rpi_support_lower+1e-10) and np.all(hi<=d.rpi_support_upper+1e-10)),
        RPI_outer_polygon_direct_excess=polygon_excess,
        RPI_gate='unchanged infinite Minkowski-sum support and eigenbasis tail bound; plotted outer polygon is an enclosure')

def coverage(e, attempt_index=1):
    """Original 256 corners + 10000 joint samples; plus normalized facets.

    Common uniforms/attempt index across candidates isolate geometry changes.
    D is the original bounded nominal external set, not Gaussian validation.
    State jump is never merged into W. All samples retain input/state evidence.
    """
    cfg,m,d=e[:3]
    xl,xh=d.robust_state_lower,d.robust_state_upper
    ul,uh=d.robust_input_lower,d.robust_input_upper
    dl,dh=cfg.disturbance_nominal-cfg.disturbance_half_range,cfg.disturbance_nominal+cfg.disturbance_half_range
    samples=[(np.array(x),np.array(u),np.array(w)) for x,u,w in product(
        list(product(*zip(xl,xh))),list(product(*zip(ul,uh))),list(product(*zip(dl,dh))))]
    rng=np.random.default_rng(cfg.seed+81000+attempt_index)
    samples += [(rng.uniform(xl,xh),rng.uniform(ul,uh),rng.uniform(dl,dh))
                for _ in range(cfg.robust_region_random_samples)]
    h,b=facets(d.w_vertices); ws=[]; worst=None; max_excess=-np.inf; legacy_bad=0
    for i,(x,u,w) in enumerate(samples):
        actual=m.normalized_state(m.step(m.physical_state(x),m.physical_input(u),w))
        error=actual-(d.a@x+d.b@u+d.affine)
        excess=h@error-b; value=float(np.max(excess)); ws.append(error)
        legacy_bad += int(not point_in_convex_polygon(error,d.w_vertices,tol=cfg.robust_region_membership_tolerance))
        if value>max_excess:
            max_excess=value
            worst=dict(sample=i,state=m.physical_state(x),input=m.physical_input(u),
                       disturbance=w,mismatch=error,facet=int(np.argmax(excess)),facet_excess=value)
    ws=np.asarray(ws); utilization=ws@h.T/np.where(abs(b)>1e-18,b,np.nan)
    facet_bad=int(np.sum(np.max(ws@h.T-b,axis=1)>cfg.robust_region_membership_tolerance))
    return dict(count=len(ws),legacy_membership_violation_count=legacy_bad,
        normalized_facet_violation_count=facet_bad,max_normalized_facet_excess=max_excess,
        min_mismatch=ws.min(axis=0),max_mismatch=ws.max(axis=0),
        max_facet_utilization=np.max(utilization,axis=0),worst=worst,
        tolerance=cfg.robust_region_membership_tolerance,
        passed=legacy_bad==0 and facet_bad==0,
        sampling='original full corner cross product + 10000 joint uniform samples; same uniforms for all candidates',
        claim='finite sampled coverage only; not continuous-domain nonlinear proof')

def evaluate(e, c, common):
    cfg,m,d=e[:3]; t=cfg.robust_region_membership_tolerance; failures=[]; gates={}
    def gate(name, passed, equation, value, threshold, evidence=None):
        gates[name]=bool(passed)
        if not passed:
            failures.append(dict(candidate=c['candidate'],failed_gate=name,check=equation,
                value=value,threshold=threshold,associated_state_input=evidence if evidence is not None else
                    dict(reference_state=cfg.robust_economic_reference_state,reference_input=cfg.robust_economic_reference_input),
                failure_type='structural',numerical=False))
    gate('frozen_geometry',True,'A/B/c/K/W/Z/S/Omega hash equality',True,True)
    gate('Hinf',common['Hinf'],'rho(A+BK)<1 AND sampled norm<gamma',common['sampled_Hinf_norm'],common['gamma'])
    gate('RPI',common['RPI'],common['RPI_gate'],common['RPI_outer_polygon_direct_excess'],'original support gate, NOT polygon-facet invariance')
    derivative=float(np.max(np.abs(m.derivative(cfg.robust_economic_reference_state,cfg.robust_economic_reference_input,cfg.disturbance_nominal))))
    gate('nonlinear_steady',derivative<1e-9,'||nonlinear derivative||inf < 1e-9',derivative,1e-9)
    gate('physical_input_domain',np.all(d.robust_input_lower>=m.normalized_input(cfg.input_lower)-t) and np.all(d.robust_input_upper<=m.normalized_input(cfg.input_upper)+t), 'U_R subset physical U',True,True)
    gate('x_tightening',np.all(d.x_upper_tight>d.x_lower_tight),'X_R minus Z nonempty',float(np.min(d.x_upper_tight-d.x_lower_tight)),0)
    width=float(np.min(d.u_upper_tight-d.u_lower_tight))
    gate('u_tightening',width>0,'U_R minus KZ nonempty',width,0)
    sm=float(np.min(np.r_[d.invariant_lower-d.x_lower_tight,d.x_upper_tight-d.invariant_upper]))
    gate('S_subset_X_minus_Z',sm>=-t,'S subset X_R minus Z',sm,-t)
    tube=float(np.min(np.r_[d.invariant_lower-d.rpi_support_lower-d.robust_state_lower,d.robust_state_upper-d.invariant_upper-d.rpi_support_upper]))
    gate('S_plus_Z',tube>=-t,'S+Z subset X_R',tube,-t)
    input_tube=float(np.min(np.r_[d.u_lower_tight-d.input_rpi_support_lower-d.robust_input_lower,d.robust_input_upper-d.u_upper_tight-d.input_rpi_support_upper]))
    gate('input_tube',input_tube>=-t,'(U_R minus KZ)+KZ subset U_R',input_tube,-t)
    reference_margin=float(np.min(np.r_[d.z_ref-d.invariant_lower,d.invariant_upper-d.z_ref]))
    gate('reference_in_S',reference_margin>=-t,'z_ref in S',reference_margin,-t)
    ref_input=float(np.min(np.r_[d.v_ref-d.u_lower_tight,d.u_upper_tight-d.v_ref]))
    gate('reference_input',ref_input>=-t,'v_ref in U_R minus KZ',ref_input,-t)
    interior=float(np.min(np.r_[d.z_ref-d.robust_state_lower,d.robust_state_upper-d.z_ref]))
    gate('safe_center',interior>t,'z_ref strictly inside X_R',interior,t)
    # For rectangular S the affine robust nominal predecessor inequalities are
    # convex. Feasible controls at every vertex imply controlled invariance.
    probes=[d.z_ref,*[np.array(x) for x in product(*zip(d.invariant_lower,d.invariant_upper))]]
    checks=[]; minimum=np.inf; qp=True; corners_ok=True
    for z in probes:
        h,b=verification_rows(d,z)
        for mode in ('floor',cfg.residual_reserve_mode):
            base,ok,authority=safe_projected_base(d.v_ref,h,b,cfg.residual_action_scale,
                cfg.qp_min_residual_authority,cfg.invariant_set_margin,
                reserve_mode=mode,reserve_fraction=cfg.residual_reserve_fraction)
            minimum=min(minimum,authority); qp &= ok
            max_gap=float(np.max(h@base-b)); modified=0
            for signs in product((-1.,1.),repeat=2):
                action=base+cfg.qp_min_residual_authority*cfg.residual_action_scale*np.array(signs)
                applied,feasible=project_qp_2d(action,h,b)
                gap=float(np.max(np.abs(applied-action)))
                corners_ok &= feasible and gap<=1e-8
                modified += int(gap>1e-8)
            checks.append(dict(z=z,reserve_mode=mode,base=base,feasible=ok,authority=authority,
                zero_residual_gap=max_gap,corner_modification_count=modified,rows=h,rhs=b))
    gate('verification_QP',qp,'reference + all S vertices, floor AND production reserve',qp,True)
    gate('residual_corners',corners_ok,'all +/- 5% corners preserved by actual verification QP',corners_ok,True)
    gate('minimum_authority',minimum>=cfg.qp_min_residual_authority-t,'minimum residual authority >= floor',minimum,cfg.qp_min_residual_authority-t)
    oc=certificate(e[3],e[4],d)
    gate('Omega_controlled_invariance',oc['passed'],'original Omega vertex-feedback certificate',oc,True)
    cov=coverage(e)
    gate('nonlinear_W_coverage',cov['passed'],'original cross membership AND retained normalized facet audit',cov['max_normalized_facet_excess'],t,cov['worst'])
    report=dict(candidate=c,gates=gates,coverage=cov,QP_probes=checks,Omega_certificate=oc,
                geometry=dict(robust_input_normalized=[d.robust_input_lower,d.robust_input_upper],
                    tightened_input_normalized=[d.u_lower_tight,d.u_upper_tight],
                    actual_input_physical=[m.physical_input(d.robust_input_lower),m.physical_input(d.robust_input_upper)],
                    nominal_input_physical=[m.physical_input(d.u_lower_tight),m.physical_input(d.u_upper_tight)],
                    q_bounds=e[4],XR=[d.robust_state_lower,d.robust_state_upper],
                    X_minus_Z=[d.x_lower_tight,d.x_upper_tight],S=[d.invariant_lower,d.invariant_upper],
                    input_KZ_support_lower=d.input_rpi_support_lower,input_KZ_support_upper=d.input_rpi_support_upper),
                minimum_residual_authority=minimum,common=common,geometry_hashes=geometry_hashes(e),
                certificate='CERTIFIED_PASS' if all(gates.values()) else 'CERTIFIED_FAIL')
    return report,failures

def geometry_row(e,c):
    cfg,m,d=e[:3]; lo=m.physical_input(d.robust_input_lower);hi=m.physical_input(d.robust_input_upper)
    tl=m.physical_input(d.u_lower_tight);tu=m.physical_input(d.u_upper_tight)
    baseline=cfg.robust_economic_reference_input+RHO*np.array([35.,30.])
    with np.load(DEFAULT_DESIGN) as z: base_nom=m.physical_input(z['u_upper_tight'])
    r=dict(c,reclaimed_F200_UR_upper=float(hi[1]-baseline[1]),
           reclaimed_F200_nominal_upper=float(tu[1]-base_nom[1]))
    for i,name in enumerate(('P100','F200')):
        r.update({f'UR_{name}_lower':float(lo[i]),f'UR_{name}_upper':float(hi[i]),
            f'nominal_tight_{name}_lower':float(tl[i]),f'nominal_tight_{name}_upper':float(tu[i]),
            f'delta_upper_{name}_vs_current':float(hi[i]-baseline[i]),
            f'{name}_physical_reference_min_margin':float(min(cfg.robust_economic_reference_input[i]-cfg.input_lower[i],cfg.input_upper[i]-cfg.robust_economic_reference_input[i])),
            f'{name}_UR_reference_min_margin':float(min(cfg.robust_economic_reference_input[i]-lo[i],hi[i]-cfg.robust_economic_reference_input[i])),
            f'{name}_nominal_reference_lower_margin':float(cfg.robust_economic_reference_input[i]-tl[i]),
            f'{name}_nominal_reference_upper_margin':float(tu[i]-cfg.robust_economic_reference_input[i])})
    return r

def authority(e,contexts,candidate):
    result=[]
    for index,context in enumerate(contexts):
        control,info=old.point_control(e,context,[0,0])
        if not info['qp_feasible']: raise RuntimeError('Certified context QP infeasible')
        h,b=np.asarray(info['action_safe_rows']),np.asarray(info['action_safe_bounds'])
        center=np.asarray(info['action_center_coordinate']);anchor=info['interior_anchor_coordinate']
        lower,upper,points=prior.polytope_extrema(h,b)
        maximum_error=0.;modifications=0;active=set()
        for point in points:
            target=(1-ALPHA)*center+ALPHA*point
            scaled,_=action_for_interior_point(center,anchor,target,h,b,e[0].input_scale)
            applied,actual=old.point_control(e,context,scaled/ALPHA)
            expected=control+(target-center)*e[0].input_scale
            maximum_error=max(maximum_error,float(np.max(np.abs(applied-expected))))
            modifications+=int(actual['final_verification_gap']>1e-8)
            active.update(old.row_family(info['mode'],i) for i in np.flatnonzero(abs(h@point-b)<=1e-8))
        if maximum_error>1e-6 or modifications: raise RuntimeError('Actual mapping authority verification failed')
        result.append(dict(candidate=candidate,context=index,role=context['role'],step=context['step'],mode=info['mode'],
            state_X2=float(context['state'][0]),state_P2=float(context['state'][1]),
            P100_applied_residual_min=float(ALPHA*(lower[0]-center[0])*e[0].input_scale[0]),
            P100_applied_residual_max=float(ALPHA*(upper[0]-center[0])*e[0].input_scale[0]),
            F200_applied_residual_min=float(ALPHA*(lower[1]-center[1])*e[0].input_scale[1]),
            F200_applied_residual_max=float(ALPHA*(upper[1]-center[1])*e[0].input_scale[1]),
            authority_reconstruction_error=maximum_error,QP_modification_count=modifications,
            active_constraints=';'.join(sorted(active)),alpha=ALPHA))
    return result

def snapshots(e):
    # Reuse only the same old Strong development path; never generate paths.
    path=exp.run_dir(exp.ROOT,'strong',42)/'evaluation_trajectories/episode_0100/seed_420000.csv'
    base,sac=v2.archived_pair(path)
    states=np.array([r['state'] for r in sac]); controls=np.array([r['control'] for r in sac])
    chosen=old.representative_indices(states,e[0].robust_economic_reference_state)
    chosen.setdefault(int(np.argmin(states[:,1])),[]).append('low_P2')
    lo=e[1].physical_input(e[2].robust_input_lower);hi=e[1].physical_input(e[2].robust_input_upper)
    chosen.setdefault(int(np.argmin(np.min(np.minimum(controls-lo,hi-controls),axis=1))),[]).append('near_current_input_boundary')
    ctrl=io.PairedEvidenceController(*e[:5],np.array([r['disturbance'] for r in sac]))
    ctrl.reset(states[0]);result=[dict(role='reference_reset',step=0,state=e[0].robust_economic_reference_state.copy(),z=e[2].z_ref.copy())]
    for k,r in enumerate(sac):
        z=ctrl.inner.z.copy();u,info=ctrl.act(r['state'],r['raw_action'])
        if np.max(abs(u-r['control']))>1e-6:raise RuntimeError('Frozen archive replay changed')
        if k in chosen: result.append(dict(role=','.join(chosen[k]),step=k,state=r['state'],z=z))
    return result,dict(path=path,sha256=prior.sha(path))

def development_paths():
    run=exp.run_dir(exp.ROOT,'strong',42);data={};sources={}
    for seed in DEV:
        path=run/f'evaluation_trajectories/episode_0100/seed_{seed}.csv'
        base,sac=v2.archived_pair(path)
        # The legacy audit parser deliberately omits reward/export metadata.
        # Recover those *recorded* fields instead of inventing zeros or
        # recomputing a baseline with a different geometry.
        for r,raw in zip(base,io.read_csv(path)):
            for name in ('interior_anchor','boundary_target'):
                r[name]=np.array([float(raw[f'baseline_{name}_{i}']) for i in range(2)])
            for key,value in raw.items():
                if not key.startswith('baseline_'):continue
                name=key[len('baseline_'):]
                if name.endswith(('_0','_1')):continue
                if value in ('True','False'):r[name]=value=='True'
                else:
                    try:r[name]=float(value)
                    except ValueError:r[name]=value
        disturbance=np.array([r['disturbance'] for r in sac])
        if len(base)!=1000 or len(sac)!=1000:raise RuntimeError('Incomplete frozen development archive')
        np.testing.assert_array_equal(disturbance,np.array([r['disturbance'] for r in base]))
        data[seed]=(disturbance,base,sac)
        sources[seed]=dict(path=path,sha256=prior.sha(path),disturbance_sha256=array_hash(disturbance))
    return data,sources

def economic_metrics(base,records,terminal_state=None):
    bc=np.array([r['economic_cost'] for r in base]);sc=np.array([r['economic_cost'] for r in records])
    x=np.array([r['state'] for r in records]);u=np.array([r['control'] for r in records]);du=np.diff(u,axis=0)
    return dict(economic_improvement_pct=float(100*(bc.sum()-sc.sum())/bc.sum()),
        economic_win_fraction=float(np.mean(sc<bc)),J_econ=float(sc.sum()),
        X2_min_margin=float(np.min(np.r_[x[:,0],terminal_state[0] if terminal_state is not None else []])-25),
        X2_std=float(np.std(x[:,0])),centered_X2_MAE=float(np.mean(abs(x[:,0]-np.mean(x[:,0])))),
        P100_TV=float(abs(du[:,0]).sum()),F200_TV=float(abs(du[:,1]).sum()),
        P100_RMS_du=float(np.sqrt(np.mean(du[:,0]**2))),F200_RMS_du=float(np.sqrt(np.mean(du[:,1]**2))))

def counterfactual(e,candidate,paths,root,actor):
    rows=[]
    for seed,(disturbance,base,archived) in paths.items():
        print('frozen_policy_geometry_sensitivity',candidate,seed,flush=True)
        try:
            records,metrics,ctrl=io.rollout(e,disturbance,seed,actor)
            row=dict(candidate=candidate,seed=seed,status='COMPLETE',
                     scope='frozen_policy_geometry_sensitivity',**economic_metrics(base,records,np.array([ctrl.evidence[-1]['next_state_0'],ctrl.evidence[-1]['next_state_1']])),
                     **{k:metrics[k] for k in io.SAFETY},
                     QP_modification_count=sum(int(float(r.get('final_verification_gap') or 0)>1e-8) for r in ctrl.evidence),
                     W_exceedance_rate=float(np.mean([r['W_exceedance'] for r in ctrl.evidence])),
                     nonlinear_claim='Gaussian empirical only')
            # Production baseline is always the paired denominator; also
            # retain the candidate's own zero-residual counterfactual.
            own=base if candidate=='baseline' else io.rollout(e,disturbance,seed,exp.ZeroResidualPolicy())[0]
            row['vs_candidate_zero_residual_improvement_pct']=economic_metrics(own,records)['economic_improvement_pct']
            row['geometry_only_zero_residual_improvement_pct']=economic_metrics(base,own)['economic_improvement_pct']
            if candidate=='baseline':
                gap=float(np.max(abs(np.array([r['control'] for r in records])-np.array([r['control'] for r in archived]))))
                row['archived_policy_control_reproduction_max_gap']=gap
                # Diagnostic only: CPU/CUDA actor arithmetic may differ.
                # This does NOT change any safety membership/QP tolerance.
                if gap>1e-4:raise RuntimeError(f'Frozen policy archive reproduction differs: {gap}')
            write_csv(root/'counterfactual'/candidate/f'seed_{seed}.csv',exp.trace(base,records,exp.full_metrics(base,records,e,0.,np.zeros(5))[2]))
        except Exception as error:
            row=dict(candidate=candidate,seed=seed,status='NUMERIC_FAIL',error=repr(error),scope='frozen_policy_geometry_sensitivity')
            io.save(root/'counterfactual'/f'{candidate}_seed_{seed}_failure.json',dict(row,evidence=getattr(error,'evidence',None)))
        rows.append(row)
    return rows

def _counterfactual_worker(payload):
    # OS processes are independent numerical workers, not agents/training.
    # Each task has its own controller and same frozen actor/archived path.
    c,seed,path,root=payload
    v2.torch.set_num_threads(1)
    original=env();actor=make_agent(original[0],'cpu')
    load_actor(actor,exp.run_dir(exp.ROOT,'strong',42)/'models/episode_0100_actor.pth')
    return counterfactual(independent(original,c),c['candidate'],{seed:path},root,actor)[0]

def export(root,rows,failures,authority_rows,economic_rows):
    write_csv(root/'ur_certificate_sensitivity.csv',rows)
    # Always emit an explicit header even if every candidate passes.
    failure_fields=['candidate','failed_gate','check','value','threshold','associated_state_input','failure_type','numerical']
    with (root/'ur_certificate_failure_reasons.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=failure_fields);w.writeheader();w.writerows(failures)
    write_csv(root/'trajectory_state_authority.csv',authority_rows)
    write_csv(root/'shared_vs_independent_ur_scaling.csv',[r for r in rows if r['family'] in ('shared_rho','independent_rho','independent_2D')])
    if economic_rows:write_csv(root/'frozen_policy_geometry_sensitivity.csv',economic_rows)

def plots(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5))
    for status,color in (('CERTIFIED_PASS','#237a57'),('CERTIFIED_FAIL','#b74343'),('NUMERIC_FAIL','#a88125')):
        subset=[r for r in rows if r['certificate']==status]
        if subset:ax.scatter([r['UR_F200_upper'] for r in subset],[r.get('W_max_facet_excess',np.nan) for r in subset],color=color,label=status,s=36)
    ax.axhline(1e-8,color='black',ls='--',lw=1,label='Existing normalized coverage tolerance')
    ax.set(xlabel='F200 U_R actual upper bound',ylabel='Maximum normalized mismatch facet excess',title='Frozen-certificate U_R sensitivity')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'ur_certificate_vs_authority.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    available=[r for r in rows if r['certificate']=='CERTIFIED_PASS' and r.get('economic_improvement_pct') is not None]
    if available:
        seen=set()
        for r in available:
            key=(r['hP']*r['rhoP'],r['hF']*r['rhoF'])
            if key in seen:continue
            seen.add(key)
            ax.scatter(r['F200_applied_residual_max'],r['economic_improvement_pct'],s=35)
            ax.annotate(r['candidate'],(r['F200_applied_residual_max'],r['economic_improvement_pct']),fontsize=8,xytext=(4,4),textcoords='offset points')
        if len(seen)==1:
            ax.text(.04,.93,'Production baseline reproduced.\nExpanded-geometry counterfactuals incomplete;\nno economic attribution conclusion yet.',
                    transform=ax.transAxes,ha='left',va='top',fontsize=10)
    else:ax.text(.5,.5,'Counterfactual NOT_EVALUATED\nNo missing result is plotted as zero',ha='center',va='center',transform=ax.transAxes)
    ax.set(xlabel='Applied positive F200 residual authority (alpha=0.1)',ylabel='Frozen-policy economic improvement (%)',title='frozen_policy_geometry_sensitivity — development paths only')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'ur_authority_vs_economic_improvement.png',dpi=170);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counterfactual',action='store_true',help='Optional frozen actor; NEVER training')
    p.add_argument('--tests',action='store_true')
    p.add_argument('--resume',action='store_true',help='Resume same immutable preregistration; keep partial evidence')
    p.add_argument('--workers',type=int,default=1,help='Independent counterfactual path processes; runtime only')
    args=p.parse_args();root=ROOT
    root.mkdir(parents=True,exist_ok=True)
    (root/'candidates').mkdir(exist_ok=True)
    (root/'counterfactual').mkdir(exist_ok=True)
    if args.tests:
        v2.tests(SimpleNamespace(root=root));return
    if args.counterfactual:require_counterfactual_disk_space(root)
    if (root/'protocol.json').exists() and not args.resume:raise FileExistsError('Preserve prior study; do not overwrite')
    reconstruction=v2.load(root/'rho_search_reconstruction.json')
    if reconstruction['status']!='COMPLETE':raise RuntimeError('Complete original rho reconstruction before the sensitivity study')
    original=env();before=protected_hashes(); frozen=geometry_hashes(original)
    protocol=dict(candidates=candidates(),alpha=ALPHA,training=False,production_changed=False,
        fixed_geometry=frozen,original_XR_fixed=True,regular_external_disturbance=original[0].disturbance_nominal,
        external_half_range=original[0].disturbance_half_range,coverage_random_samples=10000,
        original_coverage_rng='42+81000+1; common samples across candidates',
        membership_checks='original cross product tolerance AND existing normalized-facet audit; neither relaxed',
        authority_meaningful_threshold='>= 10% larger positive F200 applied authority (report only, not safety gate)',
        counterfactual_enabled=args.counterfactual,counterfactual_seeds=DEV if args.counterfactual else [],
        counterfactual_actor=str(exp.run_dir(exp.ROOT,'strong',42)/'models/episode_0100_actor.pth'),
        counterfactual_denominator='same archived production zero-residual baseline; additionally own-geometry baseline',
        selection='all certified unique geometries evaluated; duplicate family labels reuse identical result',
        nonlinear_claim='existing finite bounded-W acceptance protocol only; Gaussian safety empirical',
        protected_hashes=before)
    if args.resume:
        recorded=v2.load(root/'protocol.json')
        if recorded['candidates']!=protocol['candidates'] or recorded['alpha']!=ALPHA or recorded['protected_hashes']!=before:
            raise RuntimeError('Preregistration or production evidence changed; cannot resume')
        if recorded['counterfactual_enabled']!=args.counterfactual:raise RuntimeError('Counterfactual scope cannot change on resume')
        protocol=recorded
    else:
        io.save(root/'protocol.json',protocol)  # Written BEFORE any candidate outcome.
    common=fixed_gates(original);rows=[];failures=[];authorities=[];economics=[];cache={}
    contexts,source=snapshots(original);io.save(root/'state_snapshots.json',dict(contexts=contexts,source=source))
    paths,sources=development_paths() if args.counterfactual else ({},{})
    if args.counterfactual:
        # Runtime throughput choice only; no SAC/actor hyperparameter changes.
        # Tiny deterministic inference is faster without eight-thread overhead.
        v2.torch.set_num_threads(1)
        actor=make_agent(original[0],'cpu');actor_path=Path(protocol['counterfactual_actor']);load_actor(actor,actor_path)
        actor_sha=prior.sha(actor_path);io.save(root/'counterfactual_sources.json',dict(actor=actor_path,actor_sha256=actor_sha,paths=sources))
    for c in protocol['candidates']:
        print('certificate candidate',c['candidate'],c['family'],c['rhoP'],c['rhoF'],c['hF'],flush=True)
        key=(c['hP']*c['rhoP'],c['hF']*c['rhoF'])
        ce=independent(original,c);row=geometry_row(ce,c)
        try:
            if key in cache:
                report,errs,ars,ers=deepcopy(cache[key]);report['candidate']=c
                for r in errs+ars+ers:r['candidate']=c['candidate']
                row['duplicate_geometry_of']=report.get('evaluated_geometry_id')
            else:
                report,errs=evaluate(ce,c,common);report['evaluated_geometry_id']=c['candidate'];ars=[];ers=[]
                if report['certificate']=='CERTIFIED_PASS':
                    ars=authority(ce,contexts,c['candidate'])
                cache[key]=deepcopy((report,errs,ars,ers))
            row.update(certificate=report['certificate'],failed_gates=';'.join(k for k,v in report['gates'].items() if not v),
                minimum_residual_authority=report['minimum_residual_authority'],
                W_max_facet_excess=report['coverage']['max_normalized_facet_excess'],
                W_facet_violation_count=report['coverage']['normalized_facet_violation_count'],
                W_legacy_violation_count=report['coverage']['legacy_membership_violation_count'],
                economic_counterfactual_status='NOT_EVALUATED',economic_improvement_pct=None)
            if ars:
                row.update({k:ars[0][k] for k in ('P100_applied_residual_min','P100_applied_residual_max','F200_applied_residual_min','F200_applied_residual_max')})
            if ers and all(r['status']=='COMPLETE' for r in ers):
                row.update(economic_counterfactual_status='COMPLETE',economic_improvement_pct=float(np.mean([r['economic_improvement_pct'] for r in ers])),
                    worst_economic_improvement_pct=min(r['economic_improvement_pct'] for r in ers),
                    economic_win_fraction=float(np.mean([r['economic_win_fraction'] for r in ers])),
                    minimum_X2_margin=min(r['X2_min_margin'] for r in ers))
                for k in ('P100_TV','F200_TV','P100_RMS_du','F200_RMS_du','X2_std','centered_X2_MAE','QP_modification_count'):
                    row[k]=float(np.mean([r[k] for r in ers]))
                row['vs_candidate_zero_residual_improvement_pct']=float(np.mean([r['vs_candidate_zero_residual_improvement_pct'] for r in ers]))
                row['geometry_only_zero_residual_improvement_pct']=float(np.mean([r['geometry_only_zero_residual_improvement_pct'] for r in ers]))
            elif ers:row['economic_counterfactual_status']='NUMERIC_FAIL'
            failures.extend(errs);authorities.extend(ars);economics.extend(ers)
            io.save(root/'candidates'/f"{c['candidate']}.json",report)
        except Exception as error:
            row.update(certificate='NUMERIC_FAIL',error=repr(error),economic_counterfactual_status='NOT_EVALUATED')
            failures.append(dict(candidate=c['candidate'],failed_gate='numeric_exception',check='evaluation completed without exception',
                value=repr(error),threshold='none',associated_state_input=None,failure_type='numerical',numerical=True))
        rows.append(row);export(root,rows,failures,authorities,economics)
        print('result',c['candidate'],row['certificate'],row.get('failed_gates',''),flush=True)
    # Follow the requested order: finish all certificate/authority candidates
    # before any optional closed-loop policy counterfactual.
    if args.counterfactual:
        economic_cache={}
        if args.workers>1:
            from concurrent.futures import ProcessPoolExecutor,as_completed
            unique={}
            for row in rows:
                if row['certificate']=='CERTIFIED_PASS':
                    c=next(c for c in protocol['candidates'] if c['candidate']==row['candidate'])
                    unique.setdefault((c['hP']*c['rhoP'],c['hF']*c['rhoF']),c)
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(_counterfactual_worker,(c,seed,path,root)):(key,seed)
                    for key,c in unique.items() for seed,path in paths.items()}
                for future in as_completed(futures):
                    key,seed=futures[future]
                    economic_cache.setdefault(key,[]).append(future.result())
                    print('counterfactual complete',unique[key]['candidate'],seed,flush=True)
                for results in economic_cache.values():results.sort(key=lambda r:r['seed'])
        for row in rows:
            if row['certificate']!='CERTIFIED_PASS':continue
            c=next(c for c in protocol['candidates'] if c['candidate']==row['candidate'])
            key=(c['hP']*c['rhoP'],c['hF']*c['rhoF'])
            if key not in economic_cache:
                economic_cache[key]=counterfactual(independent(original,c),c['candidate'],paths,root,actor)
            ers=deepcopy(economic_cache[key])
            for r in ers:r['candidate']=c['candidate']
            economics.extend(ers)
            if all(r['status']=='COMPLETE' for r in ers):
                row['economic_counterfactual_status']='COMPLETE'
                row['economic_improvement_pct']=float(np.mean([r['economic_improvement_pct'] for r in ers]))
                row['worst_economic_improvement_pct']=min(r['economic_improvement_pct'] for r in ers)
                row['minimum_X2_margin']=min(r['X2_min_margin'] for r in ers)
                for k in ('economic_win_fraction','P100_TV','F200_TV','P100_RMS_du','F200_RMS_du','X2_std',
                          'centered_X2_MAE','QP_modification_count','W_exceedance_rate',
                          'vs_candidate_zero_residual_improvement_pct','geometry_only_zero_residual_improvement_pct'):
                    row[k]=float(np.mean([r[k] for r in ers]))
            else:row['economic_counterfactual_status']='NUMERIC_FAIL'
            export(root,rows,failures,authorities,economics)
    after=protected_hashes()
    if before!=after or frozen!=geometry_hashes(original):raise RuntimeError('Production evidence changed')
    if args.counterfactual and actor_sha!=prior.sha(actor_path):raise RuntimeError('Frozen actor changed')
    passed=[r for r in rows if r['certificate']=='CERTIFIED_PASS']
    base=rows[0];meaningful=[r for r in passed if r['F200_applied_residual_max']>=base.get('F200_applied_residual_max',np.inf)*1.1]
    summary=dict(status='COMPLETE',production_hashes_unchanged=True,geometry_hashes_unchanged=True,
        training=False,final_seeds_used=False,commit=False,push=False,alpha=ALPHA,
        candidates=len(rows),certified_pass=len(passed),certified_fail=sum(r['certificate']=='CERTIFIED_FAIL' for r in rows),
        maximum_certified_F200_UR_upper=max((r['UR_F200_upper'] for r in passed),default=None),
        maximum_certified_F200_nominal_upper=max((r['nominal_tight_F200_upper'] for r in passed),default=None),
        meaningful_authority_candidates=[r['candidate'] for r in meaningful],
        original_rho_reconstruction=root/'rho_search_reconstruction.json',
        certificate_scope='unchanged bounded-W numerical protocol + finite sampled coverage, not global nonlinear proof',
        gaussian_scope='frozen_policy_geometry_sensitivity; empirical only',rows=rows,
        common=common,protected_hashes_after=after)
    io.save(root/'summary.json',summary);plots(root,rows)
    print('study complete',summary['certified_pass'],summary['certified_fail'],flush=True)

if __name__=='__main__':main()
