"""Locked economic definitions for development economic recovery v1.

Never modify physical stage costs, clip negative benefits, or identify a finite
actor-action search's best result as a mathematical upper bound.
"""
from __future__ import annotations

import numpy as np
from .zanon2019_paired_v2_objective import components, operating_metrics


def economic_series(baseline_cost, policy_cost):
    base, sac = np.asarray(baseline_cost, float), np.asarray(policy_cost, float)
    if base.ndim != 1 or not len(base) or base.shape != sac.shape:
        raise ValueError("Complete equal-length paired cost vectors required")
    if not np.isfinite(base).all() or not np.isfinite(sac).all() or np.any(base <= 0):
        raise ValueError("Nonfinite/nonpositive economic denominator: benefit unavailable")
    saving = base-sac
    cumulative = 100.*np.cumsum(saving)/np.cumsum(base)
    rolling = [None]*len(base)
    # Trailing full windows only: steps 0..18 are unavailable, not zeros.
    for k in range(19, len(base)):
        rolling[k] = float(100.*saving[k-19:k+1].sum()/base[k-19:k+1].sum())
    improvement = float(cumulative[-1])
    return dict(delta_ell=sac-base, instantaneous_economic_advantage_pct=100.*saving/base,
                G_roll20=rolling, G_cum=cumulative, economic_improvement_pct=improvement,
                economic_win_fraction=float(np.mean(sac < base)),
                baseline_J_econ=float(base.sum()), SAC_J_econ=float(sac.sum()))


def recovery(strong_cost, conservative_cost, conservative_sac_cost):
    values = np.array([strong_cost, conservative_cost, conservative_sac_cost], float)
    if not np.isfinite(values).all():
        raise ValueError("Missing or nonfinite recovery costs")
    penalty, recovered = float(values[1]-values[0]), float(values[1]-values[2])
    return dict(conservatism_cost=penalty, recovered_cost=recovered,
                economic_recovery_ratio_pct=100.*recovered/penalty if penalty > 0 else None,
                recovery_ratio_status="defined" if penalty > 0 else "nonpositive_conservatism_cost")


def headroom_utilization(gain, oracle_gain, policy_protocol, oracle_protocol):
    if policy_protocol != oracle_protocol or oracle_gain is None or oracle_gain <= 0:
        return None
    # May exceed 100%: best-found oracle is not a certified upper bound.
    return float(100.*gain/oracle_gain)


def pair_identity(base, records, reference):
    if not base or len(base) != len(records):
        raise ValueError("Incomplete pair")
    if not np.allclose(base[0]["state"], reference, rtol=0, atol=1e-10):
        raise ValueError("Wrong reference/baseline for this pair")
    # Uses the existing component validator for initial state and every disturbance.
    components(records, base, [100., 100.], [0., 0.])


def pair_economics(base, records, reference):
    pair_identity(base, records, reference)
    series = economic_series([r["economic_cost"] for r in base],
                             [r["economic_cost"] for r in records])
    scalar = {k: v for k, v in series.items() if np.isscalar(v)}
    instant = series["instantaneous_economic_advantage_pct"]
    scalar.update(instantaneous_advantage_min=float(instant.min()),
                  instantaneous_advantage_p5=float(np.percentile(instant, 5)),
                  instantaneous_advantage_p50=float(np.median(instant)),
                  instantaneous_advantage_p95=float(np.percentile(instant, 95)),
                  instantaneous_advantage_max=float(instant.max()))
    for key, value in operating_metrics(base).items():
        # Complete next/terminal-state minimum is owned by audited_metrics.
        if key not in ("minimum_X2", "minimum_X2_margin"):
            scalar["baseline_"+key] = value
    return scalar, series


def control_activity(records, lower, upper):
    u = np.array([r["control"] for r in records])
    lower, upper = np.asarray(lower), np.asarray(upper)
    eta = abs(u-(lower+upper)/2)/((upper-lower)/2)
    result = {}
    for i, name in enumerate(("P100", "F200")):
        result[name+"_lower_bound_steps"] = int(np.sum(np.isclose(u[:, i], lower[i], atol=1e-6, rtol=0)))
        result[name+"_upper_bound_steps"] = int(np.sum(np.isclose(u[:, i], upper[i], atol=1e-6, rtol=0)))
        result[name+"_outer_10pct_steps"] = int(np.sum(eta[:, i] > .9))
        du = np.diff(u[:, i]); nonzero = np.sign(du[np.abs(du) > 1e-8])
        result[name+"_input_sign_changes"] = int(np.sum(nonzero[1:] != nonzero[:-1]))
    return result


def aggregate(rows, keys):
    """Training seed, not disturbance realization, is the replication unit."""
    result = []
    for key in keys:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        if not values:
            result.append(dict(metric=key, n=0, mean=None, sample_std=None,
                               ci95_half_width=None, minimum=None, maximum=None))
            continue
        x = np.asarray(values)
        std = float(x.std(ddof=1)) if len(x) > 1 else None
        # Student t for small training-seed samples, not 1.96 for n=3.
        if std is not None:
            from scipy.stats import t
            ci = float(t.ppf(.975, len(x)-1)*std/np.sqrt(len(x)))
        else:
            ci = None
        result.append(dict(metric=key, n=len(x), mean=float(x.mean()), sample_std=std,
                           ci95_half_width=ci, minimum=float(x.min()), maximum=float(x.max())))
    return result
