"""Primary Conservative economics and read-only F200 diagnostics.

Nominal U-minus-KZ applies to the Z nominal input, not to the total actual
input in Omega mode. All actual-input comparisons are separately labelled.
Break-even steps are 1-based completed transitions, not zero-based CSV rows.
"""
from __future__ import annotations
import numpy as np
from .zanon2019_economic_recovery_metrics import economic_series, recovery

ACTIVITY_KEYS = ('P100_TV_ratio', 'F200_TV_ratio', 'P100_RMS_du_ratio', 'F200_RMS_du_ratio')
ACTIVITY_LIMIT = 1.10  # Reuse existing stochastic validation quality band.


def first_step(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]+1) if len(indices) else None


def economic_triplet(strong, conservative, candidate):
    s, c, p = [np.asarray(v, float) for v in (strong, conservative, candidate)]
    if s.ndim != 1 or not len(s) or s.shape != c.shape or s.shape != p.shape:
        raise ValueError('Three complete equal-length paired cost paths required')
    if not np.isfinite([s, c, p]).all() or np.any(s <= 0) or np.any(c <= 0):
        raise ValueError('Nonfinite or nonpositive economic denominator')
    series = economic_series(c, p)
    js, jc, jp = map(float, (s.sum(), c.sum(), p.sum()))
    scalar = dict(J_strong=js, J_cons=jc, J_candidate=jp,
        economic_improvement_pct=series['economic_improvement_pct'],
        economic_difference_vs_strong_abs=jp-js,
        economic_improvement_vs_strong_pct=100*(js-jp)/js,
        economic_win_fraction=series['economic_win_fraction'], **recovery(js, jc, jp))
    penalty = jc-js
    scalar['recovery_break_even_step'] = first_step(np.cumsum(c-p) >= penalty) if penalty > 0 else None
    scalar['strong_outperformance_step'] = first_step(np.cumsum(p) < np.cumsum(s))
    scalar['cumulative_advantage_final_pct'] = float(series['G_cum'][-1])
    series.update(strong_cumulative_cost=np.cumsum(s), conservative_cumulative_cost=np.cumsum(c),
                  candidate_cumulative_cost=np.cumsum(p),
                  cumulative_improvement_vs_strong_pct=100*np.cumsum(s-p)/np.cumsum(s))
    return scalar, series


def f200_step(model, design, control, info, near_physical=1e-6):
    """Instrument the already returned controller info without another act()."""
    mode = info['mode']
    in_z = mode == 'Z_mode_existing_controller'
    upper = float(model.physical_input(design.u_upper_tight)[1])
    final = np.asarray(info['action_final_coordinate'])
    requested = np.asarray(info.get('interior_target_coordinate', info['action_candidate_coordinate']))
    rows, bounds = np.asarray(info['action_safe_rows']), np.asarray(info['action_safe_bounds'])
    matches = np.flatnonzero(np.all(np.isclose(rows, [0., 1.], rtol=0, atol=1e-12), axis=1))
    facet = int(matches[0]) if len(matches) else None
    physical_margin = upper-float(control[1])
    nominal_margin = upper-float(model.physical_input(final)[1]) if in_z else None
    active = bool(facet is not None and abs(bounds[facet]-rows[facet]@final)*model.cfg.input_scale[1] <= near_physical)
    rejected = bool(facet is not None and rows[facet]@requested-bounds[facet] > 1e-9
                    and abs(final[1]-requested[1]) > 1e-9)
    return dict(mode=mode, Z_nominal_F200_tightened_margin=nominal_margin,
        actual_F200_comparison_to_nominal_tightened_upper_margin=physical_margin,
        actual_physical_F200=float(control[1]), qp_F200_input_upper_active=active,
        qp_Z_F200_tightened_upper_active=bool(in_z and active),
        mapped_residual_rejected_or_clipped_due_to_F200=rejected,
        final_QP_modification=bool(float(info.get('final_verification_gap') or 0) > 1e-8))


def f200_summary(rows, near_physical=1e-6):
    if not rows: raise ValueError('Missing action-boundary diagnostics')
    nominal = [r['Z_nominal_F200_tightened_margin'] for r in rows if r['Z_nominal_F200_tightened_margin'] is not None]
    actual = [r['actual_F200_comparison_to_nominal_tightened_upper_margin'] for r in rows]
    return dict(minimum_F200_tightened_margin=float(min(nominal)) if nominal else None,
        mean_F200_tightened_margin=float(np.mean(nominal)) if nominal else None,
        fraction_near_tightened_F200_boundary=float(np.mean(np.abs(nominal) <= near_physical)) if nominal else None,
        Z_nominal_F200_diagnostic_steps=len(nominal),
        qp_F200_upper_bound_active_fraction=float(np.mean([r['qp_F200_input_upper_active'] for r in rows])),
        qp_Z_F200_tightened_upper_active_fraction=float(np.mean([r['qp_Z_F200_tightened_upper_active'] for r in rows])),
        actual_physical_F200_max=float(max(r['actual_physical_F200'] for r in rows)),
        actual_F200_comparison_to_nominal_tightened_upper_minimum_margin=float(min(actual)),
        actual_F200_comparison_to_nominal_tightened_upper_mean_margin=float(np.mean(actual)),
        residual_attempts_rejected_or_clipped_due_to_F200=int(sum(r['mapped_residual_rejected_or_clipped_due_to_F200'] for r in rows)),
        final_QP_modification_count=int(sum(r['final_QP_modification'] for r in rows)))


def reasonable_activity(rows):
    return bool(rows and all(r.get(k) is not None and np.isfinite(r[k]) and r[k] <= ACTIVITY_LIMIT
                             for r in rows for k in ACTIVITY_KEYS))


def training_gate(boundary, summary):
    witness = summary.get('training_witness')
    return bool(boundary.get('numerically_stable') and summary.get('completed_all_512')
        and summary.get('total_candidate_count') == 512 and witness
        and witness.get('safety_passed') and witness.get('no_nonfinite')
        and witness['economic_improvement_pct'] > 0
        and witness['worst_validation_improvement_pct'] >= 0
        and witness.get('reasonable_control_activity'))


def summarize_candidate(index, segment, paths, failure, safety_keys):
    safe = len(paths) == 10 and failure is None and all(r[k] == 0 for r in paths for k in safety_keys)
    row = dict(candidate=index, segment_steps=segment, safety_passed=safe, no_nonfinite=safe,
               failure=failure, validation_path_count=len(paths))
    metrics = ('economic_win_fraction', 'mean_X2_shift_vs_baseline', 'X2_std_ratio', 'centered_X2_MAE_ratio',
        'P100_TV_ratio', 'F200_TV_ratio', 'P100_RMS_du_ratio', 'F200_RMS_du_ratio', 'W_exceedance_rate',
        'qp_F200_upper_bound_active_fraction', 'qp_Z_F200_tightened_upper_active_fraction',
        'mean_F200_tightened_margin', 'fraction_near_tightened_F200_boundary',
        'P100_TV', 'F200_TV', 'P100_delta_RMS', 'F200_delta_RMS',
        'P100_input_sign_changes', 'F200_input_sign_changes', 'any_outer_10pct_fraction')
    for k in metrics:
        values = [r.get(k) for r in paths]
        row[k] = float(np.mean(values)) if safe and all(v is not None for v in values) else None
    for k in ('J_strong', 'J_cons', 'J_candidate'):
        row[k] = float(np.mean([r[k] for r in paths])) if safe else None
    if safe:
        js, jc, jp = [row[k] for k in ('J_strong', 'J_cons', 'J_candidate')]
        row.update(economic_improvement_pct=100*(jc-jp)/jc,
            mean_per_path_economic_improvement_pct=float(np.mean([r['economic_improvement_pct'] for r in paths])),
            economic_difference_vs_strong_abs=jp-js, economic_improvement_vs_strong_pct=100*(js-jp)/js,
            **recovery(js, jc, jp))
    else:
        row.update({k: None for k in ('economic_improvement_pct', 'mean_per_path_economic_improvement_pct',
            'economic_difference_vs_strong_abs', 'economic_improvement_vs_strong_pct',
            'conservatism_cost', 'recovered_cost', 'economic_recovery_ratio_pct')})
    row.update(worst_validation_improvement_pct=min(r['economic_improvement_pct'] for r in paths) if safe else None,
        all_validation_positive=bool(safe and all(r['economic_improvement_pct'] > 0 for r in paths)),
        all_validation_nonnegative=bool(safe and all(r['economic_improvement_pct'] >= 0 for r in paths)),
        minimum_X2_margin=min(r['minimum_X2_margin'] for r in paths) if safe else None,
        minimum_F200_tightened_margin=min((r['minimum_F200_tightened_margin'] for r in paths
            if r['minimum_F200_tightened_margin'] is not None), default=None) if safe else None,
        actual_physical_F200_max=max(r['actual_physical_F200_max'] for r in paths) if safe else None,
        residual_attempts_rejected_or_clipped_due_to_F200=sum(r['residual_attempts_rejected_or_clipped_due_to_F200'] for r in paths) if safe else None,
        reasonable_control_activity=bool(safe and reasonable_activity(paths)))
    for k in safety_keys: row[k] = sum(r[k] for r in paths) if safe else None
    return row
