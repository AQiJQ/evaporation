"""Finalize completed certificates and salvage ONLY complete saved paths.

No simulation/training. Missing or resource-aborted counterfactuals remain
NOT_EVALUATED; never score partial paths or substitute zero improvements.
"""
import numpy as np
from . import zanon2019_ur_certificate_sensitivity as a
from . import zanon2019_paired_v2 as io
from . import zanon2019_ur_certificate_report as report
from .zanon2019_benchmark import write_csv

def main():
    root=a.ROOT;protocol=io.load(root/'protocol.json');original=a.env()
    if a.protected_hashes()!=protocol['protected_hashes']:raise RuntimeError('Production evidence changed')
    rows=[];errors=[];authorities=[];economic=[]
    contexts=io.load(root/'state_snapshots.json')['contexts']
    for context in contexts:
        context['state']=np.array(context['state']);context['z']=np.array(context['z'])
    for c in protocol['candidates']:
        d=io.load(root/'candidates'/f"{c['candidate']}.json")
        ce=a.independent(original,c);row=a.geometry_row(ce,c)
        row.update(certificate=d['certificate'],failed_gates=';'.join(k for k,v in d['gates'].items() if not v),
            minimum_residual_authority=d['minimum_residual_authority'],
            W_max_facet_excess=d['coverage']['max_normalized_facet_excess'],
            W_facet_violation_count=d['coverage']['normalized_facet_violation_count'],
            W_legacy_violation_count=d['coverage']['legacy_membership_violation_count'],
            economic_counterfactual_status='NOT_EVALUATED',economic_improvement_pct=None)
        if row['certificate']=='CERTIFIED_PASS':
            ar=a.authority(ce,contexts,c['candidate']);authorities+=ar
            row.update({k:ar[0][k] for k in ('P100_applied_residual_min','P100_applied_residual_max','F200_applied_residual_min','F200_applied_residual_max')})
        for gate,passed in d['gates'].items():
            if not passed:
                source=d['coverage'] if gate=='nonlinear_W_coverage' else d['geometry']
                errors.append(dict(candidate=c['candidate'],failed_gate=gate,
                    check='w in W: original cross membership AND normalized facet tolerance' if gate=='nonlinear_W_coverage' else 'v_ref in U_R minus KZ',
                    value=d['coverage']['max_normalized_facet_excess'] if gate=='nonlinear_W_coverage' else min(row['P100_nominal_reference_lower_margin'],row['P100_nominal_reference_upper_margin'],row['F200_nominal_reference_lower_margin'],row['F200_nominal_reference_upper_margin'])/100,
                    threshold=1e-8 if gate=='nonlinear_W_coverage' else -1e-8,
                    associated_state_input=source.get('worst',source),failure_type='structural',numerical=False))
        # Duplicate labels reuse the first physical geometry's saved paths.
        key=(c['hP']*c['rhoP'],c['hF']*c['rhoF'])
        primary=next(x for x in protocol['candidates'] if (x['hP']*x['rhoP'],x['hF']*x['rhoF'])==key)
        current=[]
        for seed in a.DEV:
            path=root/'counterfactual'/primary['candidate']/f'seed_{seed}.csv'
            if not path.exists():continue
            try:
                base,sac=io.archived_pair(path)
                if len(base)!=1000 or len(sac)!=1000:raise ValueError('Incomplete path')
                # One terminal-state reconstruction from the recorded final
                # state/input/disturbance, not another trajectory simulation.
                terminal=ce[1].step(sac[-1]['state'],sac[-1]['control'],sac[-1]['disturbance'])
                h,b=a.facets(ce[3]);xn=ce[1].normalized_state(np.vstack([r['state'] for r in sac]+[terminal]))
                u=np.array([r['control'] for r in sac]);un=ce[1].normalized_input(u)
                safety=dict(physical_state_violation_steps=int(np.sum(np.any((ce[1].physical_state(xn)<ce[0].state_lower-1e-8)|(ce[1].physical_state(xn)>ce[0].state_upper+1e-8),axis=1))),
                    physical_input_violation_count=int(np.sum(np.any((u<ce[0].input_lower-1e-8)|(u>ce[0].input_upper+1e-8),axis=1))),
                    QP_infeasible_count=sum(not r['qp_feasible'] for r in sac),
                    Omega_exit_count=int(np.sum(np.max((xn-ce[2].z_ref)@h.T-b,axis=1)>1e-8)),
                    robust_region_violation_count=int(np.sum(np.any((xn<ce[2].robust_state_lower-1e-8)|(xn>ce[2].robust_state_upper+1e-8),axis=1)))+int(np.sum(np.any((un<ce[2].robust_input_lower-1e-8)|(un>ce[2].robust_input_upper+1e-8),axis=1))))
                if any(safety.values()):raise ValueError(f'Unsafe saved path: {safety}')
                wh,wb=a.facets(ce[2].w_vertices)
                actual_w=xn[1:]-(xn[:-1]@ce[2].a.T+un@ce[2].b.T+ce[2].affine)
                r=dict(candidate=c['candidate'],seed=seed,status='COMPLETE_SAVED_PATH',
                    scope='frozen_policy_geometry_sensitivity',**a.economic_metrics(base,sac,terminal),**safety,
                    QP_modification_count=sum(r['final_verification_gap']>1e-8 for r in sac),source=str(path),sha256=a.prior.sha(path))
                r['W_exceedance_rate']=float(np.mean(np.max(actual_w@wh.T-wb,axis=1)>1e-8))
                current.append(r);economic.append(r)
            except Exception as error:
                economic.append(dict(candidate=c['candidate'],seed=seed,status='INCOMPLETE_SAVED_PATH',error=repr(error),source=str(path)))
        row['completed_counterfactual_seeds']=len(current)
        if len(current)==10:
            row.update(economic_counterfactual_status='COMPLETE_SAVED_PATHS',
                economic_improvement_pct=float(np.mean([r['economic_improvement_pct'] for r in current])),
                worst_economic_improvement_pct=min(r['economic_improvement_pct'] for r in current),
                minimum_X2_margin=min(r['X2_min_margin'] for r in current))
            for key in ('economic_win_fraction','P100_TV','F200_TV','P100_RMS_du','F200_RMS_du','X2_std','centered_X2_MAE','QP_modification_count','W_exceedance_rate'):
                row[key]=float(np.mean([r[key] for r in current]))
        elif current:row['economic_counterfactual_status']='INCOMPLETE_DUE_RESOURCE'
        rows.append(row)
    a.export(root,rows,errors,authorities,economic)
    passed=[r for r in rows if r['certificate']=='CERTIFIED_PASS']
    summary=dict(status='CERTIFICATE_COMPLETE_COUNTERFACTUAL_INCOMPLETE',candidates=len(rows),
        certified_pass=len(passed),certified_fail=len(rows)-len(passed),rows=rows,
        blocker="Counterfactual writes and latest full-suite rerun raised OSError(28, 'No space left on device'). No partial-path economic average reported.",
        production_hashes_unchanged=True,geometry_hashes_unchanged=a.geometry_hashes(original)==protocol['fixed_geometry'],
        training=False,final_seeds_used=False,commit=False,push=False,alpha=.1,
        maximum_certified_F200_UR_upper=max(r['UR_F200_upper'] for r in passed),
        maximum_certified_F200_nominal_upper=max(r['nominal_tight_F200_upper'] for r in passed),
        certificate_scope='frozen bounded-W numerical gates + finite coverage, not continuous nonlinear/Gaussian certification')
    a.io.save(root/'summary.json',summary);a.plots(root,rows);report.main()
    print(summary['status'],summary['certified_pass'],summary['certified_fail'])

if __name__=='__main__':main()
