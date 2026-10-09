"""Read-only input tightening provenance audit. No plant rollout or training.

All control calls below are independent single-state probes, reset before each
action. The frozen controller, references, tolerances and saved geometry are
never edited. CSV/JSON outputs belong exclusively to this independent audit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import inspect
import json
from pathlib import Path
import unittest
import importlib
import subprocess
import sys
import re
from itertools import combinations

import numpy as np
try:
    from scipy.optimize import linprog
except ImportError:
    linprog = None  # CUDA env has NumPy only; no new optimizer dependency.

from . import zanon2019_paired_v2 as v2
from . import zanon2019_economic_recovery as exp
from . import zanon2019_reference2d as previous
from .control import _rpi_support, project_qp_2d, build_safety_design, SafeController
from .controlled_invariant_error_set import facets, hull
from .omega_safe_action_filter_diagnosis import qp_rows
from .omega_interior_anchor import action_for_interior_point
from .zanon2019_paired_experiment import PairedEvidenceController
from .zanon2019_benchmark import DEFAULT_DESIGN, DEFAULT_OMEGA, DEFAULT_CALIBRATION, write_csv
from .model import EvaporatorModel
from .config import ExperimentConfig

REPO = Path(__file__).resolve().parent
ROOT = exp.ROOT / 'input_tightening_audit'
CHANNELS = ('P100', 'F200')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_hashes():
    # Existing dirty user changes are the baseline, not reverted or overwritten.
    files = [p for p in REPO.glob('*.py') if not p.name.startswith('zanon2019_input_tightening_audit')]
    files += [DEFAULT_DESIGN, DEFAULT_OMEGA, DEFAULT_CALIBRATION,
              previous.ROOT/'conservative_2d_selection.json']
    return {str(p): sha(p) for p in files}


def source(function, needle=None):
    lines, start = inspect.getsourcelines(function)
    offset = next((i for i, line in enumerate(lines) if needle and needle in line), 0)
    return dict(file=inspect.getsourcefile(function), function=function.__qualname__,
                line=start+offset, expression=needle)


def environments():
    strong = v2.guard(exp.args_for())
    previous.locked()  # Read-only verification of the previously frozen selection.
    selected = v2.load(previous.ROOT/'conservative_2d_selection.json')
    conservative = previous.environment(strong, selected['selected_x_ref'])
    np.testing.assert_array_equal(conservative[0].robust_economic_reference_input, selected['selected_u_ref'])
    return {'Strong': strong, 'Primary_Conservative': conservative}


def support_audit(env):
    cfg, model, d = env[:3]
    z = np.asarray(d.rpi_boundary)
    hz, bz = facets(hull(z))
    kz_vertices = z @ d.k.T * cfg.input_scale
    lo, hi = kz_vertices.min(axis=0), kz_vertices.max(axis=0)
    # Optional independent LP of the saved polygon. Vertex enumeration is
    # already an exact support evaluation for this stored V-representation.
    physical_rows = d.k * cfg.input_scale[:, None]
    lp_lo, lp_hi = [], []
    for row in physical_rows:
        if linprog is None:
            lp_lo.append(float(np.min(z@row))); lp_hi.append(float(np.max(z@row)))
        else:
            minimum = linprog(row, A_ub=hz, b_ub=bz, bounds=[(None, None)]*2, method='highs')
            maximum = linprog(-row, A_ub=hz, b_ub=bz, bounds=[(None, None)]*2, method='highs')
            if not minimum.success or not maximum.success:
                raise RuntimeError('Saved RPI polygon support LP failed')
            lp_lo.append(minimum.fun); lp_hi.append(-maximum.fun)
    acl = d.a+d.b@d.k
    series_lower = _rpi_support(acl, d.w_vertices, -d.k, max_terms=cfg.rpi_series_max_terms)
    series_upper = _rpi_support(acl, d.w_vertices, d.k, max_terms=cfg.rpi_series_max_terms)
    used_lo, used_hi = -series_lower*cfg.input_scale, series_upper*cfg.input_scale
    robust_lo = np.maximum(cfg.input_lower, model.physical_input(d.robust_input_lower))
    robust_hi = np.minimum(cfg.input_upper, model.physical_input(d.robust_input_upper))
    tight_lo, tight_hi = model.physical_input(d.u_lower_tight), model.physical_input(d.u_upper_tight)
    with np.load(DEFAULT_DESIGN) as archive:
        saved_lo, saved_hi = archive['input_rpi_support_lower'], archive['input_rpi_support_upper']
    scale_csv=DEFAULT_DESIGN.parent.parent/'reference_search_candidates.csv'
    with scale_csv.open(encoding='utf-8-sig',newline='') as stream:
        row=next(row for row in csv.DictReader(stream) if row['design_file']==DEFAULT_DESIGN.name)
    scale=float(row['robust_region_scale'])
    worst = z[int(np.argmax(kz_vertices[:, 1]))]
    contributions = physical_rows[1]*worst
    tol = inspect.signature(project_qp_2d).parameters['tol'].default
    return dict(K_runtime=d.k, K_shape=list(d.k.shape),
        K_source=source(__import__('evaporation.residual_action_space_diagnosis', fromlist=['load_fixed_b']).load_fixed_b, 'k=values'),
        K_physical=np.diag(cfg.input_scale)@d.k@np.diag(1/cfg.state_scale),
        physical_lower=cfg.input_lower, physical_upper=cfg.input_upper,
        robust_lower=robust_lo, robust_upper=robust_hi, robust_region_scale=scale,
        robust_region_original_center=(robust_lo+robust_hi)/2,
        robust_region_original_half_width=(robust_hi-robust_lo)/2,
        robust_region_half_width_formula_error=(robust_hi-robust_lo)/2-scale*cfg.robust_region_initial_input_half_range,
        Z_representation='720-vertex normalized error outer polygon; formal design uses directional infinite Minkowski-sum support with eigenbasis tail enclosure',
        Z_dimension=2, Z_vertices=z, Z_H=hz, Z_h=bz, Z_vertices_physical_error=z*cfg.state_scale,
        Z_extent_physical_error=[np.min(z*cfg.state_scale, axis=0), np.max(z*cfg.state_scale, axis=0)],
        KZ_polygon_min=lo, KZ_polygon_max=hi, KZ_polygon_LP_min=lp_lo, KZ_polygon_LP_max=lp_hi,
        support_verification_method='V-representation exhaustive support + independent scipy HiGHS LP' if linprog else 'exact exhaustive vertex support; optional HiGHS unavailable in this environment',
        robust_region_scale_source=str(scale_csv),
        KZ_runtime_support_min=used_lo, KZ_runtime_support_max=used_hi,
        saved_input_support_min=-saved_lo*cfg.input_scale,
        saved_input_support_max=saved_hi*cfg.input_scale,
        polygon_vs_design_support_min_difference=lo-used_lo,
        polygon_vs_design_support_max_difference=hi-used_hi,
        pure_physical_U_minus_KZ_polygon_lower=cfg.input_lower-lo,
        pure_physical_U_minus_KZ_polygon_upper=cfg.input_upper-hi,
        pure_physical_U_minus_KZ_design_lower=cfg.input_lower-used_lo,
        pure_physical_U_minus_KZ_design_upper=cfg.input_upper-used_hi,
        runtime_nominal_lower=tight_lo, runtime_nominal_upper=tight_hi,
        recomputed_runtime_nominal_lower=robust_lo-used_lo,
        recomputed_runtime_nominal_upper=robust_hi-used_hi,
        comparison_to_printed_216303281=dict(
            pure_upper_absolute_difference=float(abs(cfg.input_upper[1]-used_hi[1]-216.303281)),
            matches_pure_U_minus_KZ=bool(abs(cfg.input_upper[1]-used_hi[1]-216.303281) <= tol*cfg.input_scale[1]),
            runtime_difference_from_rounded_label=float(tight_hi[1]-216.303281),
            rounding_note='216.303281 is a six-decimal display label; exact stored upper is audited separately'),
        F200_support_maximizing_error_physical=worst*cfg.state_scale,
        F200_X2_contribution=float(contributions[0]), F200_P2_contribution=float(contributions[1]),
        F200_dominant_contribution='X2 error' if abs(contributions[0])>abs(contributions[1]) else 'P2 error',
        normalization_state_origin=cfg.linearization_state, normalization_input_origin=cfg.linearization_input,
        state_scale=cfg.state_scale, input_scale=cfg.input_scale,
        control_sign='actual normalized u = nominal v + K @ (normalized x - normalized z); K already includes its signs',
        solver_tolerance_normalized=tol, rpi_series_max_terms=cfg.rpi_series_max_terms)


def polytope_extrema(rows, rhs):
    # Exact 2D bounded-polytope LP via feasible pairwise active intersections.
    # The production normalized tolerance is unchanged; no random sampling.
    rows,rhs=np.asarray(rows),np.asarray(rhs)
    vertices=[]
    for i,j in combinations(range(len(rhs)),2):
        pair=rows[[i,j]]
        if abs(np.linalg.det(pair))<=1e-12: continue
        vertex=np.linalg.solve(pair,rhs[[i,j]])
        if np.max(rows@vertex-rhs)<=1e-9: vertices.append(vertex)
    if not vertices: raise RuntimeError('Action polytope is empty or degenerate')
    vertices=np.asarray(vertices)
    points=np.asarray([vertices[np.argmax(vertices@row)] for row in (*np.eye(2),*(-np.eye(2)))])
    if linprog is not None:
        for row,point in zip((*np.eye(2),*(-np.eye(2))),points):
            result=linprog(-row,A_ub=rows,b_ub=rhs,bounds=[(None,None)]*2,method='highs')
            if not result.success or abs(row@(result.x-point))>1e-7: raise RuntimeError('Independent action LP disagreement')
    return points.min(axis=0),points.max(axis=0),points


def controller_at(env, action):
    cfg, model, d, omega, domain = env[:5]
    ctrl = PairedEvidenceController(cfg, model, d, omega, domain, np.array([cfg.disturbance_nominal]))
    state = cfg.robust_economic_reference_state.copy()
    ctrl.reset(state)
    control, info = ctrl.act(state, np.asarray(action, float))
    return control, info


def reference_audit(env):
    cfg, model, d, omega, domain = env[:5]
    before = deepcopy(vars(cfg)), deepcopy(vars(d))
    control, info = controller_at(env, [0, 0])
    h, b = info['action_safe_rows'], info['action_safe_bounds']
    base, anchor = info['action_center_coordinate'], info['interior_anchor_coordinate']
    lower, upper, points = polytope_extrema(h, b)
    alpha = cfg.stochastic_residual_scale
    # The frozen AuthorityController restricts rho to alpha. Interior map image
    # is exactly (1-alpha)*base + alpha*Q, a homothety of the convex polytope.
    effective_points = (1-alpha)*base+alpha*points
    checks = []
    for target in effective_points:
        scaled, projected = action_for_interior_point(base, anchor, target, h, b, cfg.input_scale)
        actual, actual_info = controller_at(env, scaled/alpha)
        expected = model.physical_input(target+info['ancillary'])
        checks.append(dict(raw_action=scaled/alpha, mapped_target=target,
            expected_applied=expected, actual_applied=actual,
            reconstruction_error=float(np.linalg.norm(actual-expected)),
            QP_feasible=bool(actual_info['qp_feasible']),
            final_QP_gap=float(actual_info['final_verification_gap'])))
    oqh, oqb = qp_rows(np.zeros(2), omega, d, domain)
    omega_lo, omega_hi, _ = polytope_extrema(oqh, oqb)
    uref = cfg.robust_economic_reference_input
    result = dict(reference_state=cfg.robust_economic_reference_state, reference_input=uref,
        selected_mode=info['mode'], rpi_error=info['rpi_error'], omega_error=info['omega_error'],
        nominal_reference_margin_lower=uref-model.physical_input(d.u_lower_tight),
        nominal_reference_margin_upper=model.physical_input(d.u_upper_tight)-uref,
        physical_actual_margin_lower=uref-cfg.input_lower, physical_actual_margin_upper=cfg.input_upper-uref,
        robust_actual_margin_lower=uref-model.physical_input(d.robust_input_lower),
        robust_actual_margin_upper=model.physical_input(d.robust_input_upper)-uref,
        zero_residual_applied=control, zero_residual_nominal=model.physical_input(info['nominal']),
        zero_residual_base_physical=model.physical_input(base),
        nominal_reference_is_mapping_center=bool(np.allclose(base, d.v_ref, atol=1e-12, rtol=0)),
        residual_parameterization=cfg.residual_parameterization, stochastic_residual_scale=alpha,
        residual_action_scale=cfg.residual_action_scale, full_safe_mapping_nominal_lower=model.physical_input(lower),
        full_safe_mapping_nominal_upper=model.physical_input(upper),
        full_safe_residual_lower=(lower-base)*cfg.input_scale,
        full_safe_residual_upper=(upper-base)*cfg.input_scale,
        actual_raw_actor_image_lower=model.physical_input((1-alpha)*base+alpha*lower+info['ancillary']),
        actual_raw_actor_image_upper=model.physical_input((1-alpha)*base+alpha*upper+info['ancillary']),
        actual_raw_actor_applied_residual_lower=alpha*(lower-base)*cfg.input_scale,
        actual_raw_actor_applied_residual_upper=alpha*(upper-base)*cfg.input_scale,
        actual_raw_actor_extreme_probes=checks,
        Omega_at_reference_q_lower=omega_lo, Omega_at_reference_q_upper=omega_hi,
        Omega_at_reference_applied_lower=model.physical_input(d.v_ref+omega_lo),
        Omega_at_reference_applied_upper=model.physical_input(d.v_ref+omega_hi),
        Omega_point_note='constraint-only counterfactual at xi=0; reset reference selects Z-mode, not Omega-mode',
        QP_nominal_rows=h, QP_nominal_rhs=b,
        QP_nominal_row_labels=['P100 upper', 'F200 upper', 'P100 lower', 'F200 lower',
            'next nominal X2 upper in S', 'next nominal P2 upper in S', 'next nominal X2 lower in S', 'next nominal P2 lower in S'],
        QP_applied_rows_in_normalized_absolute_u=h,
        QP_applied_rhs_in_normalized_absolute_u=b+h@info['ancillary'],
        Z_QP_absolute_physical_rows=h/cfg.input_scale,
        Z_QP_absolute_physical_rhs=b+h@info['ancillary']+h@(cfg.linearization_input/cfg.input_scale),
        Omega_QP_q_rows=oqh, Omega_QP_q_rhs=oqb,
        Omega_QP_absolute_physical_rows=oqh/cfg.input_scale,
        Omega_QP_absolute_physical_rhs=oqb+oqh@d.v_ref+oqh@(cfg.linearization_input/cfg.input_scale),
        Omega_QP_input_row_labels=['P100 actual upper', 'F200 actual upper', 'P100 actual lower', 'F200 actual lower'],
        actual_hard_lower=np.maximum(cfg.input_lower,model.physical_input(d.robust_input_lower)),
        actual_hard_upper=np.minimum(cfg.input_upper,model.physical_input(d.robust_input_upper)),
        S_physical_lower=model.physical_state(d.invariant_lower),
        S_physical_upper=model.physical_state(d.invariant_upper),
        nominal_actual_relation='Z: u=physical_input(v+K e), e=0 here; Omega: u=u_ref+input_scale*q',
        physical_constraint_enforcement='Z nominal tightening plus actual robust clip; Omega first four QP rows use physical/robust intersection; EvidenceController audits physical applied u directly')
    for prior, current in ((before[0],vars(cfg)), (before[1],vars(d))):
        for key, old in prior.items():
            new = current[key]
            if isinstance(old,np.ndarray): np.testing.assert_array_equal(old,new)
            else: assert old==new, key
    return result


def sanity(env, name, support, detail):
    cfg, model, d = env[:3]
    tol = inspect.signature(project_qp_2d).parameters['tol'].default
    values = [100, 160, 162.30, 194.86, 200, 216.20, 216.27, 216.30,
              216.30327, 216.303281, 216.31, 220, 225.94, 250, 300, 399, 400]
    rows = []
    for channel in range(2):
        for value in values:
            u=cfg.robust_economic_reference_input.copy(); u[channel]=value
            un=model.normalized_input(u)
            h=np.asarray(detail['QP_nominal_rows']); b=np.asarray(detail['QP_nominal_rhs'])
            projected, qp_exists=project_qp_2d(un,h,b)
            direct_excess=float(np.max(h@un-b))
            # Fix the OTHER input to its reference; Omega uses xi=0, no rollout.
            q=un-d.v_ref; oh=np.asarray(detail['Omega_QP_q_rows']); ob=np.asarray(detail['Omega_QP_q_rhs'])
            rows.append(dict(reference=name, channel=CHANNELS[channel], tested_physical_value=value,
                physical_feasible=bool(np.all(u>=cfg.input_lower) and np.all(u<=cfg.input_upper)),
                robust_actual_feasible=bool(np.all(un>=d.robust_input_lower-tol) and np.all(un<=d.robust_input_upper+tol)),
                pure_physical_tightened_feasible=bool(np.all(u>=support['pure_physical_U_minus_KZ_design_lower']-tol*cfg.input_scale)
                    and np.all(u<=support['pure_physical_U_minus_KZ_design_upper']+tol*cfg.input_scale)),
                reference_input_gate_only=bool(np.all(un>=d.u_lower_tight-1e-8) and np.all(un<=d.u_upper_tight+1e-8)),
                reference_gate_note='input gate only; no claim tested value is nonlinear steady input',
                full_safe_polytope_candidate_feasible=bool(direct_excess<=tol),
                raw_actor_mapping_image_feasible=bool(np.max(h@((un-(1-cfg.stochastic_residual_scale)*np.asarray(detail['QP_nominal_rows_center']))/cfg.stochastic_residual_scale)-b)<=tol),
                QP_original_candidate_feasible=bool(direct_excess<=tol), QP_projection_problem_feasible=bool(qp_exists),
                QP_projected_P100=float(model.physical_input(projected)[0]), QP_projected_F200=float(model.physical_input(projected)[1]),
                QP_projection_distance_physical=float(np.linalg.norm(model.physical_input(projected)-u)),
                Omega_QP_candidate_feasible_at_xi0=bool(np.max(oh@q-ob)<=tol)))
    return rows


def audit(root=ROOT):
    if root.resolve()!=ROOT.resolve(): raise ValueError('Use independent input_tightening_audit root')
    if (root/'audit_summary.json').exists(): raise RuntimeError('Preserve completed audit evidence')
    before=protected_hashes()
    envs=environments()
    s=support_audit(envs['Strong'])
    refs={key:reference_audit(env) for key,env in envs.items()}
    sanity_rows=[]; provenance=[]
    def layer(label, lo, hi, variable, src, context='both references'):
        provenance.append(dict(Layer=label, P100_lower=float(lo[0]),P100_upper=float(hi[0]),
            F200_lower=float(lo[1]), F200_upper=float(hi[1]),Variable_Type=variable,
            Source_Function=src, Context=context))
    layer('A Physical U',s['physical_lower'],s['physical_upper'],'absolute applied physical u','config.ExperimentConfig.input_lower/input_upper')
    layer('B KZ saved outer polygon LP',s['KZ_polygon_min'],s['KZ_polygon_max'],'physical feedback correction','min/max diag(input_scale) K e over saved polygon')
    layer('B KZ design support',s['KZ_runtime_support_min'],s['KZ_runtime_support_max'],'physical feedback correction','control._rpi_support(A+BK,W,+/-K)')
    layer('B Pure physical U minus KZ (design support)',s['pure_physical_U_minus_KZ_design_lower'],s['pure_physical_U_minus_KZ_design_upper'],'absolute nominal physical v','physical U minus directional RPI supports')
    layer('B Pure physical U minus saved polygon KZ',s['pure_physical_U_minus_KZ_polygon_lower'],s['pure_physical_U_minus_KZ_polygon_upper'],'absolute nominal physical v','vertex/LP support of saved outer polygon')
    layer('C Frozen robust actual-input region U_R',s['robust_lower'],s['robust_upper'],'absolute actual physical u','theta_learning.OnlineThetaLearner._robust_region_bounds')
    layer('C/D Actual runtime U_R minus KZ',s['runtime_nominal_lower'],s['runtime_nominal_upper'],'absolute nominal/reference physical v','control.build_safety_design: u_lo+support_lower, u_hi-support_upper')
    for name, detail in refs.items():
        cfg,model,d=envs[name][:3]
        detail['QP_nominal_rows_center']=model.normalized_input(detail['zero_residual_base_physical'])
        layer('E Full interior-anchor feasible mapping',detail['full_safe_mapping_nominal_lower'],detail['full_safe_mapping_nominal_upper'],'absolute nominal v at reference e=0','omega_interior_anchor.InteriorAnchorController.act',name)
        layer('E Raw actor image (frozen alpha)',detail['actual_raw_actor_image_lower'],detail['actual_raw_actor_image_upper'],'absolute applied u at reference e=0','zanon2019_authority.AuthorityController.act',name)
        layer('E Raw actor applied residual',detail['actual_raw_actor_applied_residual_lower'],detail['actual_raw_actor_applied_residual_upper'],'physical residual relative to safe zero-residual baseline','interior-anchor homothety; verified real act calls',name)
        layer('F Z QP nominal input rows',model.physical_input(d.u_lower_tight),model.physical_input(d.u_upper_tight),'absolute nominal v; not final u','control.SafeController.act rows 0..3',name)
        layer('G Omega QP actual input rows',detail['actual_hard_lower'],detail['actual_hard_upper'],'absolute final applied u; q=u_norm-v_ref','omega_safe_action_filter_diagnosis.qp_rows rows 0..3',name)
        layer('G Omega QP full feasible extrema at xi=0',detail['Omega_at_reference_applied_lower'],detail['Omega_at_reference_applied_upper'],'absolute final applied u; diagnostic unselected mode','Omega input rows plus robust-next-Omega facets',name)
        sanity_rows+=sanity(envs[name],name,s,detail)
    issues=[]
    gap=np.maximum(s['KZ_polygon_max']-s['KZ_runtime_support_max'],s['KZ_runtime_support_min']-s['KZ_polygon_min'])
    if np.any(gap>s['solver_tolerance_normalized']*s['input_scale']):
        issues.append(dict(suspected_issue='Saved outer Z polygon membership vs infinite-sum support contract gap',
            **source(__import__('evaporation.omega_feasible_normalized',fromlist=['FeasibleSetNormalizedController']).FeasibleSetNormalizedController.act,'in_z = contains'),
            expected_behavior='If ALL saved polygon vertices are certified errors, tightening must cover their supports; otherwise clarify polygon is membership approximation/enclosure, not identical infinite RPI set.',
            actual_behavior='Membership uses the saved outer polygon; tightening uses separate infinite-series/tail directional supports.',
            impact='At a nominal input boundary plus the polygon extreme, P100 pre-clip robust bound may be exceeded by a small enclosure gap. Actual robust clip still enforces input bounds; this is not an observed closed-loop violation or a unit bug.',
            excess_physical=gap, confidence='high numerical evidence; safety-certificate interpretation requires review; no core repair authorized'))
    issues.append(dict(suspected_issue='Ambiguous U_minus_KZ labels actually denote U_R minus KZ',
        **source(build_safety_design,'diagnostics["u_minus_kz_upper_physical"]'),
        expected_behavior='Distinguish global physical U=[100,400]^2 from local certified actual-input region U_R.',
        actual_behavior='Saved U_minus_KZ names refer to the intersection with robust_input region before tightening.',
        impact='Reporting/provenance ambiguity only; explains 216.303281 without any changed physical constraint.', confidence='confirmed label ambiguity, not control-math bug'))
    after=protected_hashes()
    if before!=after: raise RuntimeError('Audit mutated production source/geometry/reference')
    result=dict(schema='read_only_input_tightening_audit', support=s, references=refs,
        upper_first_computation=source(build_safety_design,'u_hi_t = u_hi - input_support_upper'),
        physical_bounds_source=source(ExperimentConfig), normalization_source=source(EvaporatorModel.normalized_input),
        robust_bounds_source=source(__import__('evaporation.theta_learning',fromlist=['OnlineThetaLearner']).OnlineThetaLearner._robust_region_bounds),
        reference_reload_source=source(previous.environment,'d.z_ref, d.v_ref'),
        upper_type='absolute physical nominal input bound; also gates steady reference; neither physical applied upper nor residual delta',
        upper_classification=['runtime-computed U_R minus KZ at original design','saved NPZ loaded at runtime','reference-only stages inherit frozen physical geometry'],
        Strong_ref_change_full_design='Original independent reference/geometry design can change U_R, K, W, Z and hence upper; this is not a universal global physical-U bound.',
        Conservative_ref_change_frozen_stage='No recomputation: physical U_R/K/W/Z/S and normalized origins frozen, v_ref/affine/offset updated, Omega coordinate origin translated; q-domain recalculated relative to new v_ref.',
        model_input_bounds_note='EvaporatorModel accepts supplied inputs; controller clips/constrains them and EvidenceController audits applied physical bounds. Plant equations alone do not clamp to [100,400].',
        no_unit_or_scale_or_double_tightening_bug_found=True, numerical_enclosure_issue_requires_review=bool(np.any(gap>1e-7)),
        production_hashes_before=before, production_hashes_after=after, unchanged=True,
        no_rollouts=True, no_disturbance_paths_sampled=True, no_training=True, no_commit=True,no_push=True)
    root.mkdir(parents=True,exist_ok=True)
    exp.save(root/'rpi_Z_snapshot.json',{k:s[k] for k in ('Z_representation','Z_dimension','Z_vertices','Z_H','Z_h','Z_vertices_physical_error','Z_extent_physical_error')})
    exp.save(root/'qp_final_input_constraints.json',refs)
    exp.save(root/'suspected_tightening_issues.json',issues)
    write_csv(root/'input_tightening_provenance.csv',provenance)
    write_csv(root/'single_point_feasibility.csv',sanity_rows)
    exp.save(root/'audit_summary.json',result)
    report(root,result,issues)
    print(json.dumps(dict(physical_bounds=[s['physical_lower'].tolist(),s['physical_upper'].tolist()],
        pure_upper=s['pure_physical_U_minus_KZ_design_upper'].tolist(),runtime_upper=s['runtime_nominal_upper'].tolist(),
        report=str(root/'INPUT_TIGHTENING_AUDIT.md'),unchanged=True),indent=2))
    return result


def report(root,r,issues):
    s=r['support']; fmt=lambda v:np.array2string(np.asarray(v),precision=12,separator=', ')
    lines=['# P100 / F200 tightening provenance audit','',
        '## Main finding','',
        '**216.303280987611 is the frozen local nominal `(physical U ∩ U_R) ⊖ KZ` F200 upper. It is NOT the pure global `[100,400] ⊖ KZ` upper, and NOT the final applied F200 upper.**','',
        f"Physical lower/upper: {fmt(s['physical_lower'])} / {fmt(s['physical_upper'])}.",
        f"Runtime normalized K: `{fmt(s['K_runtime'])}`.",
        f"Physical-units K = diag(input_scale) K diag(1/state_scale): `{fmt(s['K_physical'])}`.",
        f"Z: {s['Z_representation']}; physical error extents `{fmt(s['Z_extent_physical_error'])}`.",
        f"Saved polygon KZ exact vertex min/max: `{fmt(s['KZ_polygon_min'])}` / `{fmt(s['KZ_polygon_max'])}`. Optional independent HiGHS check is also included in tests when available.",
        f"Runtime infinite-support KZ min/max: `{fmt(s['KZ_runtime_support_min'])}` / `{fmt(s['KZ_runtime_support_max'])}`.",
        f"Pure physical-U tightening (design support): `{fmt(s['pure_physical_U_minus_KZ_design_lower'])}` / `{fmt(s['pure_physical_U_minus_KZ_design_upper'])}`.",
        f"Pure physical-U tightening (saved outer polygon): `{fmt(s['pure_physical_U_minus_KZ_polygon_lower'])}` / `{fmt(s['pure_physical_U_minus_KZ_polygon_upper'])}`.",
        f"Frozen actual robust input U_R: `{fmt(s['robust_lower'])}` / `{fmt(s['robust_upper'])}`.",
        f"Actual runtime nominal tightening U_R minus KZ: `{fmt(s['runtime_nominal_lower'])}` / `{fmt(s['runtime_nominal_upper'])}`.",
        f"First computation: `{r['upper_first_computation']}`.",
        'Runtime load: `residual_action_space_diagnosis.load_fixed_b -> values[u_upper_tight]` from the frozen balanced-B NPZ. It is not hard-coded, an anchor constant, a residual width, or generated by Bj/Gm.',
        '', '## Why F200 is strongly tightened','',
        f"U_R was created around `{fmt(s['robust_region_original_center'])}` with scale {s['robust_region_scale']} and half widths `{fmt(s['robust_region_original_half_width'])}`.",
        f"F200 runtime upper = {s['robust_upper'][1]:.12f} - {s['KZ_runtime_support_max'][1]:.12f} = {s['runtime_nominal_upper'][1]:.12f}.",
        'The 183.696719 gap from 400 includes the pre-existing local U_R restriction, not a KZ support of 183.696719.',
        f"At the saved F200 support extreme, X2/P2 contributions are {s['F200_X2_contribution']:.12f} / {s['F200_P2_contribution']:.12f}; P2 dominates.",
        'X2 uses percentage points (25 means 25%, not 0.25); P2 is kPa. Z is normalized ERROR around zero, not an absolute state set or Omega. Input correction is diag(input_scale) K e; no reference offset is added to Ke.',
        '', '## Reference and mapping audit','',
        '| Reference | F200 nominal ref upper margin | F200 robust actual upper margin | P100 applied residual min/max | F200 applied residual min/max |',
        '|---|---:|---:|---|---|']
    for name,t in r['references'].items():
        lo,hi=t['actual_raw_actor_applied_residual_lower'],t['actual_raw_actor_applied_residual_upper']
        lines.append(f"| {name} | {t['nominal_reference_margin_upper'][1]:.12g} | {t['robust_actual_margin_upper'][1]:.12g} | {lo[0]:.9f}, {hi[0]:.9f} | {lo[1]:.9f}, {hi[1]:.9f} |")
    for name,t in r['references'].items():
        lines += ['',f"{name} x_ref/u_ref: `{fmt(t['reference_state'])}` / `{fmt(t['reference_input'])}`.",
            f"Safe zero-residual applied input: `{fmt(t['zero_residual_applied'])}`; it need NOT equal u_ref: projection reserves authority.",
            f"P100 nominal reference lower/upper headroom: `{fmt([t['nominal_reference_margin_lower'][0],t['nominal_reference_margin_upper'][0]])}`. F200 physical actual upper headroom: {t['physical_actual_margin_upper'][1]:.12f}.",
            f"Full feasible nominal mapping lower/upper: `{fmt(t['full_safe_mapping_nominal_lower'])}` / `{fmt(t['full_safe_mapping_nominal_upper'])}`.",
            f"Raw actor applied-input image at frozen alpha={t['stochastic_residual_scale']}: `{fmt(t['actual_raw_actor_image_lower'])}` / `{fmt(t['actual_raw_actor_image_upper'])}`."]
    lines += ['',r['Strong_ref_change_full_design'],r['Conservative_ref_change_frozen_stage'],
        'Strong and Conservative have identical frozen nominal upper. Their reference headroom differs, but their projected safe baseline centers retain nonzero bidirectional actor authority. Exhausting steady-reference F200 headroom is not exhausting physical/robust actual-input authority.',
        'P100 does not have the same upper-headroom bottleneck: its directional KZ support is much smaller and its steady reference is near the center of the local tightened interval. F200 KZ asymmetry almost consumes the original U_R upper half-width.',
        '', '## Coordinates and final QP','',
        '| Variable | Meaning | Constraint |','|---|---|---|',
        '| u / u_applied | absolute physical input | physical U and tighter actual U_R |',
        '| v / u_nom / nominal | normalized nominal input (physical_input converts to absolute) | U_R minus KZ plus next-z in S |',
        '| v_ref / u_ref | normalized / physical steady reference | reference membership in frozen tightened nominal interval |',
        '| e / rpi_error | normalized x minus moving z | saved Z membership |',
        '| xi / omega_error | normalized x minus fixed reference | translated Omega membership |',
        '| ancillary / Ke | normalized feedback correction | K already includes sign; scale once |',
        '| q | normalized total actual-input deviation from v_ref | Omega QP bounds q_lower/u_ref to q_upper/u_ref |',
        '| residual / delta_u | normalized mapped action minus safe baseline | state-dependent action polytope |',
        '| actor_action | dimensionless raw [-1,1]^2 | frozen alpha scales rho before interior-anchor map |','',
        'Z QP variables are nominal normalized v. Rows 0..3 are +P100,+F200,-P100,-F200; rows 4..7 impose next nominal state in S. Actual input is v+Ke, followed by the unchanged robust actual-input clip. At the audited reference e=0, final applied Z ranges equal the nominal polytope extrema. Away from e=0 they translate by Ke, not a universal applied upper 216.303281.',
        'Omega QP variables are q=(u-u_ref)/input_scale. First four rows impose actual physical/robust input intersection; other rows impose robust next Omega. Actual F200 may legally exceed the nominal 216.303281 upper. Full row matrices, RHS, shifted physical-coordinate inequalities and LP extrema are in qp_final_input_constraints.json.',
        'In stochastic operation Gm/Bj jump automaton is inactive (no state jumps). No finite-jump-set constraint creates this upper.',
        r['model_input_bounds_note'],
        '', '## Single-point tests','',
        'single_point_feasibility.csv fixes the other input to its reference. `QP_original_candidate_feasible` checks the exact candidate, whereas `QP_projection_problem_feasible` indicates an admissible projected input exists. These must not be conflated. Reference input gate is not a nonlinear equilibrium claim. No plant rollout or new disturbance sample is used.',
        'Mapping-image feasibility checks BOTH inputs jointly. At these reset references, fixed F200_ref is outside the raw-actor alpha=0.1 image around the projected safe baseline. Consequently a P100-only probe can have mapping_image_feasible=false because the fixed OTHER input is outside that image. This does not imply zero P100 authority; exact reachable extrema are separately inverted and verified by real controller calls.',
        '', '## Implementation issues / limits','',
        'No absolute/deviation confusion, percent/fraction unit bug, repeated normalization, double K multiplication, or double mathematical tightening was found in the traced production path. Local U_R followed by KZ subtraction and one-step S/Omega constraints are different requirements, not duplicate tightening.',
        'A reporting ambiguity is confirmed: U_minus_KZ labels omit the prior local U_R intersection. Separately, saved outer polygon membership and infinite-series directional tightening are not numerically identical. The latter is a suspected certificate-contract issue, not an explanation for the 183.7 gap and not a demonstrated nonlinear closed-loop failure. No core fix is made.',
        f"Polygon vs directional support maximum gap (physical): `{fmt(s['polygon_vs_design_support_max_difference'])}`.",
        'Frozen Gaussian stochastic experiments remain empirical-only: this static audit does not certify unbounded disturbances or a continuous nonlinear domain.',
        '', '## Preservation','',
        'Production source/config/geometry/reference hashes and in-memory cfg/design fields were verified unchanged. Only independent audit script/tests and this output directory are added. No training, oracle continuation, seed430000..430049 rollout, reference selection, commit, or push. Tests are recorded separately in tests_receipt.json.']
    (root/'INPUT_TIGHTENING_AUDIT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def all_tests():
    before=protected_hashes()
    suite=unittest.TestSuite(); mains=[]
    for path in sorted(REPO.glob('*tests.py')):
        module=importlib.import_module('evaporation.'+path.stem)
        loaded=unittest.defaultTestLoader.loadTestsFromModule(module); suite.addTests(loaded)
        for name,function in inspect.getmembers(module,inspect.isfunction):
            if name.startswith('test_') and function.__module__==module.__name__ and not inspect.signature(function).parameters:
                suite.addTest(unittest.FunctionTestCase(function))
        if loaded.countTestCases()==0 and hasattr(module,'main'): mains.append(module)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    optional=REPO/'.venv-ecc2019/Scripts/python.exe'
    proc=subprocess.run([str(optional),'-m','unittest','evaporation.ecc2019_reproduction_tests.SolverTests','-v'],
        cwd=REPO.parent,capture_output=True,text=True) if optional.exists() else None
    if proc: print(proc.stdout,proc.stderr,flush=True)
    integrations=[]; integration_error=None
    if result.wasSuccessful():
        try:
            for module in [importlib.import_module('evaporation.test'),*mains]:
                module.main(); integrations.append(module.__name__)
        except Exception as error: integration_error=repr(error)
    skipped_ok=not result.skipped or (len(result.skipped)==1 and 'CasADi' in result.skipped[0][1] and proc and proc.returncode==0)
    passed=bool(result.wasSuccessful() and skipped_ok and proc and proc.returncode==0 and integration_error is None)
    same=before==protected_hashes()
    count=re.search(r'Ran (\d+) tests?',proc.stderr) if proc else None
    ROOT.mkdir(parents=True,exist_ok=True)
    exp.save(ROOT/'tests_receipt.json',dict(passed=passed and same,main_tests=result.testsRun,
        skipped=[dict(test=str(t),reason=r) for t,r in result.skipped],optional_CasADi_passed=bool(proc and proc.returncode==0),
        optional_CasADi_tests=int(count.group(1)) if count else 0,integration_suites=integrations,
        integration_error=integration_error,production_hashes_unchanged=same,training_performed=False))
    if not passed or not same: raise RuntimeError('Audit tests failed; see receipt')


def refresh_report():
    """Refresh this audit's presentation/added coordinate fields only.

    Existing numeric evidence is checked for exact reproduction first. No
    historical experiment/protocol or production file is rewritten.
    """
    before=protected_hashes()
    result=v2.load(ROOT/'audit_summary.json')
    refs={key:reference_audit(env) for key,env in environments().items()}
    def encode(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        raise TypeError(type(v).__name__)
    for name,old in result['references'].items():
        refs[name]['QP_nominal_rows_center']=old['QP_nominal_rows_center']
        for key,value in old.items():
            if json.dumps(value,sort_keys=True)!=json.dumps(refs[name][key],sort_keys=True,default=encode):
                raise RuntimeError(f'Numeric audit evidence changed: {name}/{key}')
    if before!=protected_hashes(): raise RuntimeError('Production changed during report refresh')
    result['references']=refs
    exp.save(ROOT/'audit_summary.json',result)
    exp.save(ROOT/'qp_final_input_constraints.json',refs)
    report(ROOT,result,v2.load(ROOT/'suspected_tightening_issues.json'))


def verify_independent_lp():
    if linprog is None: raise RuntimeError('Use existing .venv-ecc2019 for optional independent HiGHS LP')
    before=protected_hashes()
    support=support_audit(environments()['Strong'])
    saved=v2.load(ROOT/'audit_summary.json')['support']
    keys=('KZ_polygon_min','KZ_polygon_max','KZ_runtime_support_min','KZ_runtime_support_max')
    for key in keys: np.testing.assert_allclose(support[key],saved[key],rtol=0,atol=1e-7)
    np.testing.assert_allclose(support['KZ_polygon_min'],support['KZ_polygon_LP_min'],rtol=0,atol=1e-7)
    np.testing.assert_allclose(support['KZ_polygon_max'],support['KZ_polygon_LP_max'],rtol=0,atol=1e-7)
    if before!=protected_hashes(): raise RuntimeError('Production changed during LP check')
    exp.save(ROOT/'independent_lp_verification.json',dict(passed=True,method=support['support_verification_method'],
        numpy_version=np.__version__,KZ_polygon_LP_min=support['KZ_polygon_LP_min'],KZ_polygon_LP_max=support['KZ_polygon_LP_max'],
        production_unchanged=True,no_rollouts=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests',action='store_true',help='all root tests + optional CasADi + evaporation.test; no training')
    parser.add_argument('--refresh-report',action='store_true',help='refresh independent audit presentation; require identical existing numeric evidence')
    parser.add_argument('--verify-lp',action='store_true',help='optional HiGHS LP cross-check under existing .venv-ecc2019')
    args=parser.parse_args()
    if sum((args.tests,args.refresh_report,args.verify_lp))>1: parser.error('Choose one audit phase')
    if args.tests: all_tests()
    elif args.refresh_report: refresh_report()
    elif args.verify_lp: verify_independent_lp()
    else: audit()
