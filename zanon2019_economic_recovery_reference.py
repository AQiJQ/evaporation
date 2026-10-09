"""Reference-only sensitivity: fixed B Jacobians, K/W/Z, physical sets.

Implementation choice for reproduction: retain the original normalization and
Jacobian A/B. Re-anchor the affine predictor at the exact new nonlinear steady
pair. This is NOT Jacobian relinearization or gain/geometry optimization.
S/tightening stay physically identical; Omega vertices change coordinates only.
Every candidate is rejected if these unchanged physical sets fail the existing
affine/W certificates. Gaussian validation remains empirical only.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import numpy as np

from .model import EvaporatorModel
from .control import estimate_hinf_norm, safe_projected_base, _rpi_support
from .controlled_invariant_error_set import bounds_and_domains, certificate, facets, hull
from .residual_action_space_diagnosis import verification_rows


def reference_environment(strong_env, delta_x2):
    cfg = deepcopy(strong_env[0])
    model = EvaporatorModel(cfg)
    design = deepcopy(strong_env[2])
    original = strong_env[2]
    reference = cfg.robust_economic_reference_state.copy()+[float(delta_x2), 0.]
    steady = model.steady_input(reference)
    if not np.isfinite(steady).all():
        raise ValueError("Nonfinite nonlinear steady input")
    cfg.robust_economic_reference_state = reference.copy()
    cfg.robust_economic_reference_input = steady.copy()
    cfg.safe_center_state = reference.copy()
    design.z_ref = model.normalized_state(reference)
    design.v_ref = model.normalized_input(steady)
    # Keep delta=0 bitwise identical, including the saved predictor intercept.
    if delta_x2 != 0:
        design.affine = design.z_ref-design.a@design.z_ref-design.b@design.v_ref
        design.nominal_policy_offset = design.v_ref-design.nominal_policy_gain@design.z_ref
    omega = np.asarray(strong_env[3]).copy()+original.z_ref-design.z_ref
    domain, _ = bounds_and_domains(cfg, model, design)
    env = (cfg, model, design, omega, domain, strong_env[5].copy(), deepcopy(strong_env[6]))
    report = validate(env, strong_env)
    return env, report


def validate(env, strong):
    cfg, model, d, omega, domain = env[:5]
    old = strong[2]
    derivative = model.derivative(cfg.robust_economic_reference_state,
                                  cfg.robust_economic_reference_input, cfg.disturbance_nominal)
    gate = dict(nonlinear_steady=bool(np.linalg.norm(derivative, np.inf) < 1e-9),
                affine_steady=bool(np.max(np.abs(d.a@d.z_ref+d.b@d.v_ref+d.affine-d.z_ref)) < 1e-8),
                reference_in_S=bool(np.all(d.z_ref >= d.invariant_lower-1e-8)
                                    and np.all(d.z_ref <= d.invariant_upper+1e-8)),
                reference_input_tightened=bool(np.all(d.v_ref >= d.u_lower_tight-1e-8)
                                               and np.all(d.v_ref <= d.u_upper_tight+1e-8)))
    frozen = ("a", "b", "k", "w_vertices", "rpi_boundary", "robust_state_lower", "robust_state_upper",
              "robust_input_lower", "robust_input_upper", "u_lower_tight", "u_upper_tight",
              "invariant_lower", "invariant_upper", "nominal_policy_gain")
    gate["frozen_arrays_identical"] = all(np.array_equal(getattr(d, k), getattr(old, k)) for k in frozen)
    gate["physical_Omega_identical"] = bool(np.allclose(omega+d.z_ref, strong[3]+old.z_ref, atol=1e-12))
    rho = float(max(abs(np.linalg.eigvals(d.a+d.b@d.k))))
    norm = estimate_hinf_norm(d.a, d.b, d.k, cfg.hinf_q, cfg.hinf_r, points=4096)
    gate["Hinf"] = bool(rho < 1 and norm < cfg.hinf_gamma)
    hz, bz = facets(hull(d.rpi_boundary))
    predicted = np.asarray(d.rpi_boundary)@(d.a+d.b@d.k).T
    excess = float(max(np.max((predicted+w)@hz.T-bz) for w in d.w_vertices))
    # Preserve the existing design's convergent infinite support/tail-bound gate.
    # Its plotted outer polygon is an enclosure, not itself guaranteed RCI.
    # A direct polygon-facet excess is reported separately; do NOT silently
    # reinterpret the old gate as a new arbitrary facet tolerance.
    support_upper = _rpi_support(d.a+d.b@d.k, d.w_vertices, np.eye(2),
                                 max_terms=cfg.rpi_series_max_terms)
    support_lower = _rpi_support(d.a+d.b@d.k, d.w_vertices, -np.eye(2),
                                 max_terms=cfg.rpi_series_max_terms)
    gate["RPI"] = bool(rho < 1 and np.all(support_upper <= d.rpi_support_upper+1e-10)
                        and np.all(support_lower <= d.rpi_support_lower+1e-10))
    gate["Omega"] = bool(certificate(omega, domain, d)["passed"])
    minimum = float("inf")
    probes = [d.z_ref, *[np.array(v) for v in product(*zip(d.invariant_lower, d.invariant_upper))]]
    gate["verification_QP"] = True
    for z in probes:
        a, b = verification_rows(d, z)
        _, feasible, authority = safe_projected_base(d.v_ref, a, b, cfg.residual_action_scale,
            cfg.qp_min_residual_authority, cfg.invariant_set_margin,
            reserve_mode=cfg.residual_reserve_mode, reserve_fraction=cfg.residual_reserve_fraction)
        gate["verification_QP"] &= feasible
        minimum = min(minimum, authority)
    gate["residual_authority"] = bool(minimum >= cfg.qp_min_residual_authority-1e-8)
    # Nominal mismatch coverage audit in exactly the retained physical domain.
    # Finite probes are NOT a continuous nonlinear-domain proof.
    hx, bx = facets(hull(d.w_vertices))
    xlower = np.maximum(model.normalized_state(cfg.state_lower), d.robust_state_lower)
    xupper = np.minimum(model.normalized_state(cfg.state_upper), d.robust_state_upper)
    ulower = np.maximum(model.normalized_input(cfg.input_lower), d.robust_input_lower)
    uupper = np.minimum(model.normalized_input(cfg.input_upper), d.robust_input_upper)
    rng = np.random.default_rng(190019)
    xs = np.vstack([list(product(*zip(xlower, xupper))), rng.uniform(xlower, xupper, (256, 2))])
    us = np.vstack([list(product(*zip(ulower, uupper))), rng.uniform(ulower, uupper, (256, 2))])
    max_w_excess = -float("inf")
    for x, u in zip(xs, us):
        w = model.nominal_normalized_step(x, u)-(d.a@x+d.b@u+d.affine)
        max_w_excess = max(max_w_excess, float(np.max(hx@w-bx)))
    # Also all vertex x input corner combinations.
    for x in xs[:4]:
        for u in us[:4]:
            w = model.nominal_normalized_step(x, u)-(d.a@x+d.b@u+d.affine)
            max_w_excess = max(max_w_excess, float(np.max(hx@w-bx)))
    gate["nominal_W_sample_coverage"] = bool(max_w_excess <= 1e-8)
    return dict(reference=cfg.robust_economic_reference_state, steady_input=cfg.robust_economic_reference_input,
        derivative=derivative, gates=gate, passed=all(gate.values()), limiting_gates=[k for k, v in gate.items() if not v],
        minimum_residual_authority=minimum, spectral_radius=rho, sampled_Hinf_norm=norm,
        gamma=cfg.hinf_gamma, RPI_max_facet_excess=excess, nominal_W_max_facet_excess=max_w_excess,
        RPI_gate_definition="unchanged stable infinite Minkowski-sum support with existing eigenbasis tail bound",
        saved_RPI_outer_polygon_direct_containment_passed=bool(excess <= 1e-10),
        changed_because_reference_changed=["nonlinear steady state/input", "z_ref/v_ref", "nominal policy offset",
            "affine predictor equilibrium intercept", "Omega error coordinates (same physical set)", "recovery reward reference"],
        kept_identical=[*frozen, "physical/robust constraints", "normalization origins at B", "QP formulation",
                       "interior-anchor mapping", "SAC/observation", "stochastic distribution"],
        predictor_choice="fixed B Jacobians with exact new equilibrium intercept; not new Jacobian linearization",
        nonlinear_certification="finite nominal mismatch probes only; stochastic safety is empirical")
