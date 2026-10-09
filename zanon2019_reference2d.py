"""Independent, preregistered reference-only 2D Conservative feasibility stage.

Implementation choice for reproduction: retain the Strong normalization,
Jacobians and physical safety sets; re-anchor only the affine equilibrium and
reference/nominal-policy offset, as in the existing reference-only audit.
No economic objective is used to select P2. No training entry is exposed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np

from . import zanon2019_conservative_mechanism as old
from . import zanon2019_economic_recovery as exp
from . import zanon2019_paired_v2 as v2
from .model import EvaporatorModel
from .controlled_invariant_error_set import bounds_and_domains
from .zanon2019_economic_recovery_reference import validate
from .zanon2019_economic_recovery_metrics import pair_economics
from .zanon2019_paired_v2_objective import audited_metrics
from .zanon2019_paired_v2_objective import paired_metrics
from .zanon2019_benchmark import sample_disturbance_path, write_csv, PAPER_VARIANCES_F1_X1_T1_T200
from .train import ZeroResidualPolicy

ROOT = exp.ROOT / "conservative_mechanism_stage3_2d"
DELTAS = (.01, .02, .03, .05, .10, .15, .20)
STEP = np.array([1e-4, 1e-4])
NO_CANDIDATE = ("No valid reference-based Conservative baseline exists under the frozen "
                "current safety geometry within the preregistered 2D search.")


def snapshot(folder):
    return {str(p.relative_to(folder)): old.v1.digest(p)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def code_hashes():
    return {p.name: old.v1.digest(p) for p in old.v1.REPO.glob("zanon2019_reference2d*.py")}


def jacobian(model, reference, step=STEP):
    reference, step = np.asarray(reference, float), np.asarray(step, float)
    if reference.shape != (2,) or step.shape != (2,) or np.any(step <= 0):
        raise ValueError("Two finite positive difference steps required")
    if not np.isfinite(reference).all() or not np.isfinite(step).all():
        raise ValueError("Nonfinite finite-difference arguments")
    return np.column_stack([(model.steady_input(reference + np.eye(2)[i]*step[i]) -
                             model.steady_input(reference - np.eye(2)[i]*step[i])) / (2*step[i])
                            for i in range(2)])


def physical_bounds(env):
    cfg, model, d = env[:3]
    return dict(input_lower=model.physical_input(d.u_lower_tight),
                input_upper=model.physical_input(d.u_upper_tight),
                state_lower=np.maximum(cfg.state_lower, model.physical_state(d.invariant_lower)),
                state_upper=np.minimum(cfg.state_upper, model.physical_state(d.invariant_upper)))


def make_protocol(strong):
    cfg, model = strong[:2]
    ref = cfg.robust_economic_reference_state.copy()
    j = jacobian(model, ref)
    if abs(j[1, 1]) < 1e-9:
        raise RuntimeError("No stable local F200 compensation derivative; do not invent a search range")
    slope = -float(j[1, 0]/j[1, 1])
    bounds = physical_bounds(strong)
    # Fixed geometric rule, independent of any closed-loop economics/SAC result.
    radius = max(.05, 2*abs(slope)*max(DELTAS))
    p2_range = [max(float(bounds["state_lower"][1]), ref[1]-radius),
                min(float(bounds["state_upper"][1]), ref[1]+radius)]
    return dict(schema="reference2d_stage3", deltas_X2=list(DELTAS),
        strong_reference=ref, common_initial_state=ref,
        difference_step_sizes=STEP, sensitivity_step_multipliers=[.5, 2.],
        derivative_rtol=1e-6, derivative_atol=1e-7,
        P2_range=p2_range, P2_range_rule="intersection of physical/S bounds and Strong P2 +/- max(0.05, 2*abs(local compensation slope)*0.20)",
        P2_compensation_slope=slope, P2_fallback_grid_points=401,
        P2_root_xtol=1e-10, P2_gate_boundary_tolerance=1e-8,
        tightened_input_numerical_inset_physical=1e-8,
        search_rule="nearest P2 to Strong: monotone exact steady-input bracket roots, then complete gates; if root fails, fixed 401-point full-gate grid and nearest pass/fail boundary refinement",
        nearest_claim="exact input-bound root if complete gates pass; otherwise nearest detected full-gate interval to 1e-8, not a proof excluding sub-grid disconnected islands",
        validation_seeds=list(old.DEV), steps=1000, dt_min=cfg.dt_min,
        disturbance_variances=PAPER_VARIANCES_F1_X1_T1_T200,
        margin_increase_tolerance=1e-4, penalty_numerical_guard_abs=2e-4,
        selection_rule="first ascending X2 target with full reference gates, ten safe rollouts, each minimum margin increased >1e-4 and mean J increased >2e-4; never replace after selection",
        reference_only_changes=["exact nonlinear reference/steady input", "affine equilibrium intercept", "nominal policy offset", "Omega coordinate origin only (same physical set)"],
        frozen_geometry_sha256=exp.locked(exp.ROOT)["frozen_geometry_sha256"],
        frozen_v2_sources=v2.source_hashes(), stage2_snapshot=snapshot(old.ROOT),
        source_hashes=code_hashes(), smoothness_lambda=v2.locked_config(v2.ROOT)["lambda_smooth"],
        economic_scale=200., oracle_budget=512, oracle_segments=[10, 5], oracle_search_seed=190019,
        training_performed=False, final_test_performed=False,
        statistical_scope="finite nominal coverage and stochastic numerical validation; not continuous-domain nonlinear certification")


def prepare(root=ROOT):
    if root.resolve() != ROOT.resolve():
        raise ValueError("Use independent stage3_2d output root")
    if (root/"protocol_2d.json").exists():
        raise RuntimeError("Keep immutable preregistration; never expand after inspecting results")
    strong = v2.guard(exp.args_for())
    protocol = make_protocol(strong)
    exp.immutable(root/"protocol_2d.json", protocol)
    ref, model = strong[0].robust_economic_reference_state, strong[1]
    main = jacobian(model, ref)
    alternatives = {str(scale): jacobian(model, ref, STEP*scale) for scale in (.5, 2.)}
    stable = all(np.allclose(main, j, rtol=protocol["derivative_rtol"], atol=protocol["derivative_atol"])
                 for j in alternatives.values())
    bounds = physical_bounds(strong)
    inputs = model.steady_input(ref)
    report = dict(base_reference=ref, base_steady_input=inputs, jacobian=main,
        dP100_dX2=float(main[0, 0]), dP100_dP2=float(main[0, 1]),
        dF200_dX2=float(main[1, 0]), dF200_dP2=float(main[1, 1]),
        numerical_step_sizes=STEP, sensitivity_jacobians=alternatives, sensitivity_passed=stable,
        tightened_bounds=bounds, interpretation="positive X2 increases F200; P2 compensation is determined from the exact steady map, not economics")
    exp.immutable(root/"steady_input_local_jacobian.json", report)
    if not stable:
        raise RuntimeError("Jacobian sensitivity check failed; search blocked")
    margins = np.minimum(inputs-bounds["input_lower"], bounds["input_upper"]-inputs)
    slope = protocol["P2_compensation_slope"]
    exp.immutable(root/"feasible_direction_analysis.json", dict(
        local_compensation_dP2_dX2=slope,
        compensated_input_derivative_per_X2=main@np.array([1., slope]),
        compensation_exists=bool(np.isfinite(slope)),
        closest_tightened_input_facet=("P100" if margins[0] < margins[1] else "F200") +
            (" upper" if inputs[np.argmin(margins)] > np.mean([bounds["input_lower"], bounds["input_upper"]], axis=0)[np.argmin(margins)] else " lower"),
        input_nearest_margins=margins,
        fixed_P2_X2_increment_linear_upper_limit=float((bounds["input_upper"][1]-inputs[1])/main[1, 0]),
        local_P2_search_range=protocol["P2_range"],
        narrow_channel_interpretation="fixed-P2 headroom is tiny; raising P2 offsets F200 growth. The full P2 corridor width is measured per target, not assumed narrow.",
        selected_using_economic_results=False, selected_using_SAC_results=False))
    print(f"Preregistered P2 range={protocol['P2_range']}; Ju={main.tolist()}; slope={slope}", flush=True)


def locked(root=ROOT):
    if root.resolve() != ROOT.resolve(): raise ValueError("Wrong evidence root")
    p = old.load(root/"protocol_2d.json")
    exp.locked(exp.ROOT); old.locked(old.ROOT)
    if p["source_hashes"] != code_hashes() or p["stage2_snapshot"] != snapshot(old.ROOT):
        raise RuntimeError("Stage3 source or historical stage2 evidence changed")
    if p["frozen_v2_sources"] != v2.source_hashes(): raise RuntimeError("Strong source changed")
    for path, sha in p["frozen_geometry_sha256"].items():
        if old.v1.digest(path) != sha: raise RuntimeError("Frozen geometry changed")
    if p["deltas_X2"] != list(DELTAS): raise RuntimeError("Predefined targets changed")
    exp.require_dev(p["validation_seeds"])
    return p


def environment(strong, reference):
    cfg, d = deepcopy(strong[0]), deepcopy(strong[2])
    model = EvaporatorModel(cfg)
    reference = np.asarray(reference, float)
    steady = model.steady_input(reference)
    cfg.robust_economic_reference_state = reference.copy()
    cfg.robust_economic_reference_input = steady.copy()
    cfg.safe_center_state = reference.copy()
    d.z_ref, d.v_ref = model.normalized_state(reference), model.normalized_input(steady)
    if not np.array_equal(reference, strong[0].robust_economic_reference_state):
        d.affine = d.z_ref-d.a@d.z_ref-d.b@d.v_ref
        d.nominal_policy_offset = d.v_ref-d.nominal_policy_gain@d.z_ref
    omega = strong[3].copy()+strong[2].z_ref-d.z_ref
    domain, _ = bounds_and_domains(cfg, model, d)
    return cfg, model, d, omega, domain, strong[5].copy(), deepcopy(strong[6])


def full_check(strong, reference, common_initial):
    env = environment(strong, reference)
    report = validate(env, strong)
    cfg, model, d = env[:3]
    x, u = np.asarray(reference), cfg.robust_economic_reference_input
    # The frozen runtime namespace intentionally stores S, not X-minus-Z.
    # Recover the unchanged tightening from the saved robust bounds and exact
    # saved directional RPI supports, without adding fields to that controller.
    tight_lower = d.robust_state_lower+d.rpi_support_lower
    tight_upper = d.robust_state_upper-d.rpi_support_upper
    gates = report["gates"]
    gates.update(numerical_valid=bool(np.isfinite(x).all() and np.isfinite(u).all()),
        physical_state=bool(np.all(x >= cfg.state_lower) and np.all(x <= cfg.state_upper)),
        physical_input=bool(np.all(u >= cfg.input_lower) and np.all(u <= cfg.input_upper)),
        tightened_state=bool(np.all(d.z_ref >= tight_lower-1e-8) and np.all(d.z_ref <= tight_upper+1e-8)))
    # Actual unchanged online QP initialization at BOTH reference and common reset.
    gates["online_QP_initialization"] = True
    initialization = []
    for reset in (x, np.asarray(common_initial)):
        controller = old.v1.PairedEvidenceController(*env[:5], np.tile(cfg.disturbance_nominal, (1, 1)))
        try:
            controller.reset(reset)
            control, info = controller.act(reset, np.zeros(2))
            ok = bool(info["qp_feasible"] and np.isfinite(control).all())
            initialization.append(dict(state=reset, control=control, passed=ok))
            gates["online_QP_initialization"] &= ok
        except Exception as exc:
            gates["online_QP_initialization"] = False
            initialization.append(dict(state=reset, passed=False, error=str(exc)))
    report.update(passed=all(gates.values()), limiting_gates=[k for k, v in gates.items() if not v],
                  initialization_probes=initialization)
    return env, report


def input_interval(model, x2, p2_range, lower, upper, xtol, inset):
    """Exact nonlinear monotone steady input constraints; no economic objective.

    P100 is affine increasing in P2; F200 is decreasing on its positive branch.
    We verify this branch and monotonicity before using bracketed roots.
    """
    start, end = map(float, p2_range)
    grid = np.linspace(start, end, 33)
    values = np.array([model.steady_input([x2, p]) for p in grid])
    if not np.isfinite(values).all() or np.any(values[:, 1] <= 0):
        return None, "invalid_or_disconnected_steady_input_branch"
    if not (np.all(np.diff(values[:, 0]) > 0) and np.all(np.diff(values[:, 1]) < 0)):
        return None, "steady_input_monotonicity_not_verified"
    lo, hi = start, end
    for channel in range(2):
        for target, is_lower in ((lower[channel]+inset, True), (upper[channel]-inset, False)):
            f = lambda p: float(model.steady_input([x2, p])[channel]-target)
            fs, fe = f(start), f(end)
            good_start = fs >= 0 if is_lower else fs <= 0
            good_end = fe >= 0 if is_lower else fe <= 0
            if not good_start and not good_end: return None, f"channel_{channel}_tightened_bound_unreachable"
            if good_start and good_end: continue
            root = bracket_root(f, start, end, xtol)
            if good_start: hi = min(hi, root)
            else: lo = max(lo, root)
    return ([lo, hi], None) if lo <= hi else (None, "empty_tightened_input_interval")


def bracket_root(function, lower, upper, tolerance):
    """Deterministic bisection, no additional numerical dependency."""
    fl, fu = function(lower), function(upper)
    if not np.isfinite([fl, fu]).all() or fl*fu > 0 or tolerance <= 0:
        raise ValueError("Finite sign-changing bracket and positive tolerance required")
    if fl == 0: return lower
    if fu == 0: return upper
    while upper-lower > tolerance:
        middle = .5*(lower+upper)
        fm = function(middle)
        if not np.isfinite(fm): raise ValueError("Nonfinite bracket interior")
        if fm == 0: return middle
        if fl*fm <= 0: upper = middle
        else: lower, fl = middle, fm
    return .5*(lower+upper)


def nearest_reference(strong, delta, protocol):
    bounds = physical_bounds(strong)
    x2 = float(protocol["strong_reference"][0]+delta)
    p0 = float(protocol["strong_reference"][1])
    probes = []
    interval, reason = input_interval(strong[1], x2, protocol["P2_range"], bounds["input_lower"],
        bounds["input_upper"], protocol["P2_root_xtol"], protocol["tightened_input_numerical_inset_physical"])
    if not bounds["state_lower"][0] <= x2 <= bounds["state_upper"][0]:
        return None, dict(reference_feasible=False, reject_reason="X2_outside_frozen_reference_S", probes=probes)
    if interval is None:
        return None, dict(reference_feasible=False, reject_reason=reason, probes=probes)
    candidate = float(np.clip(p0, *interval))
    def check(p):
        env, report = full_check(strong, [x2, p], protocol["common_initial_state"])
        probes.append(dict(P2=p, passed=report["passed"], limiting_gates=report["limiting_gates"]))
        return env, report
    env, report = check(candidate)
    method = "exact_nearest_tightened_input_boundary_full_gates_passed"
    if not report["passed"]:
        good = []
        for p in np.linspace(*interval, protocol["P2_fallback_grid_points"]):
            e, r = check(float(p))
            if r["passed"]: good.append((abs(p-p0), float(p), e, r))
        if not good:
            return None, dict(reference_feasible=False, reject_reason="no_full_gate_feasible_point_in_fixed_interval",
                              input_feasible_P2_interval=interval, nearest_input_point=candidate,
                              nearest_input_limiting_gates=report["limiting_gates"], probes=probes)
        _, passed_p, env, report = min(good, key=lambda g: (g[0], g[1]))
        failed_p = candidate
        while abs(passed_p-failed_p) > protocol["P2_gate_boundary_tolerance"]:
            midpoint = .5*(passed_p+failed_p)
            e, r = check(midpoint)
            if r["passed"]: passed_p, env, report = midpoint, e, r
            else: failed_p = midpoint
        candidate = passed_p
        method = "nearest_detected_full_gate_interval_boundary_refined_not_subgrid_global_proof"
    u = env[0].robust_economic_reference_input
    report.update(reference_feasible=True, reject_reason=None, input_feasible_P2_interval=interval,
        input_feasible_P2_corridor_width=float(interval[1]-interval[0]), P2_search_method=method,
        delta_X2=delta, X2_ref=x2, P2_ref=candidate, delta_P2=candidate-p0,
        P100_ref=float(u[0]), F200_ref=float(u[1]),
        P100_tightened_margin=float(min(u[0]-bounds["input_lower"][0], bounds["input_upper"][0]-u[0])),
        F200_tightened_margin=float(min(u[1]-bounds["input_lower"][1], bounds["input_upper"][1]-u[1])),
        steady_state_residual=float(np.linalg.norm(report["derivative"], np.inf)), probes=probes)
    return env, report


def search(root=ROOT):
    p = locked(root)
    if (root/"conservative_2d_reference_candidates.json").exists(): raise RuntimeError("Preserve reference search evidence")
    strong = v2.guard(exp.args_for())
    rows = []
    for delta in p["deltas_X2"]:
        _, report = nearest_reference(strong, delta, p)
        exp.save(root/"reference_checks"/f"delta_{delta:.2f}.json", report)
        row = {k: report.get(k) for k in ("delta_X2", "X2_ref", "P2_ref", "delta_P2", "P100_ref", "F200_ref",
            "P100_tightened_margin", "F200_tightened_margin", "steady_state_residual", "reference_feasible", "reject_reason",
            "input_feasible_P2_corridor_width", "P2_search_method")}
        row.update(delta_X2=delta, X2_ref=p["strong_reference"][0]+delta)
        rows.append(row)
        print(f"reference delta={delta:.2f}: feasible={row['reference_feasible']}; P2={row['P2_ref']}; reason={row['reject_reason']}", flush=True)
    write_csv(root/"conservative_2d_reference_candidates.csv", rows)
    exp.immutable(root/"conservative_2d_reference_candidates.json", rows)


def eligible(row, per_seed, strong_rows, p):
    return bool(row["reference_feasible"] and row.get("baseline_evaluation_status") == "completed"
        and old.margin_gate(per_seed, strong_rows, p["margin_increase_tolerance"])
        and row["conservatism_penalty_abs"] > p["penalty_numerical_guard_abs"])


def baselines(root=ROOT):
    p = locked(root)
    if (root/"conservative_2d_selection.json").exists(): raise RuntimeError("Preserve frozen selection/results")
    candidates = old.load(root/"conservative_2d_reference_candidates.json")
    strong = v2.guard(exp.args_for())
    initial = np.asarray(p["common_initial_state"])
    cache, strong_rows, strong_sources = {}, [], {}
    # Read-only reuse of already audited stage2 Strong paths, with independent
    # plant/cost/terminal-safety and exact regenerated-disturbance verification.
    for seed in old.DEV:
        source = old.ROOT/"baseline_trajectories/delta_0.00"/f"seed_{seed}.csv"
        records, _ = v2.archived_pair(source)
        path, _ = sample_disturbance_path(strong[0], "zanon2019_stochastic", seed, p["steps"], PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
        if len(records) != p["steps"] or not np.array_equal(records[0]["state"], initial):
            raise RuntimeError("Historical Strong reset/horizon differs")
        np.testing.assert_array_equal(np.array([r["disturbance"] for r in records]), path)
        metric = audited_metrics(records, strong)
        if any(metric[k] for k in old.SAFETY): raise RuntimeError("Historical Strong baseline no longer passes")
        strong_rows.append(dict(validation_seed=seed, **old.per_path_metrics(strong, records, initial)))
        cache[seed] = path
        strong_sources[str(source)] = old.v1.digest(source)
    strong_stats = old.summarize(strong_rows)
    rows, per_seed, selected = [], [], None
    for candidate in candidates:
        row = dict(candidate, baseline_evaluation_status="not_run_reference_gate_failed", mean_J_econ=None,
            conservatism_penalty_abs=None, conservatism_penalty_pct=None, minimum_X2_margin=None,
            safety_margin_increase=None, valid_Conservative_candidate=False, validation_path_count=0)
        current = []
        if candidate["reference_feasible"]:
            env, check = full_check(strong, [candidate["X2_ref"], candidate["P2_ref"]], initial)
            if not check["passed"]: raise RuntimeError("Frozen reference check changed")
            try:
                for seed in old.DEV:
                    print(f"baseline 2D delta={row['delta_X2']:.2f} seed={seed}", flush=True)
                    records, _ = old.rollout(env, cache[seed], seed, ZeroResidualPolicy(), initial)
                    metric = old.per_path_metrics(env, records, initial)
                    s = next(r for r in strong_rows if r["validation_seed"] == seed)
                    metric.update(economic_cost_increase_vs_Strong_pct=100*(metric["J_econ"]-s["J_econ"])/s["J_econ"],
                                  minimum_X2_margin_increase_vs_Strong=metric["minimum_X2_margin"]-s["minimum_X2_margin"])
                    current.append(dict(delta_X2=row["delta_X2"], validation_seed=seed, **metric))
                    _, series = pair_economics(records, records, initial)
                    write_csv(root/"baseline_trajectories"/f"delta_{row['delta_X2']:.2f}"/f"seed_{seed}.csv", exp.trace(records, records, series))
            except Exception as exc:
                row.update(baseline_evaluation_status="aborted_safety_or_numerical_failure", failure=str(exc))
                if getattr(exc, "evidence", None):
                    write_csv(root/"baseline_failures"/f"delta_{row['delta_X2']:.2f}_seed_{seed}.csv", exc.evidence)
            else:
                row.update(old.summarize(current), baseline_evaluation_status="completed")
                penalty = row["mean_J_econ"]-strong_stats["mean_J_econ"]
                row.update(conservatism_penalty_abs=penalty, conservatism_penalty_pct=100*penalty/strong_stats["mean_J_econ"],
                           safety_margin_increase=row["minimum_X2_margin"]-strong_stats["minimum_X2_margin"])
                row["valid_Conservative_candidate"] = eligible(row, current, strong_rows, p)
                if selected is None and row["valid_Conservative_candidate"]: selected = row.copy()
        rows.append(row); per_seed.extend(current)
        # Persist complete/partial evidence throughout, but do not freeze
        # selection until the preregistered grid is completely accounted for.
        write_csv(root/"conservative_2d_baseline_results.csv", rows)
        write_csv(root/"conservative_2d_baseline_results_per_seed.csv", per_seed)
    exp.immutable(root/"conservative_2d_baseline_results.json", dict(candidates=rows, per_seed=per_seed,
        strong_metrics=strong_stats, strong_per_seed=strong_rows, Strong_readonly_sources=strong_sources,
        common_initial_state=initial, certification_status=old.CERTIFICATION_STATUS,
        aggregation="mean J across ten paths; minimum margin worst including terminal state; IAE is relative to each controller's own frozen reference"))
    selection = dict(preregistered_X2_targets=p["deltas_X2"], P2_search_protocol=p,
        local_jacobian=old.load(root/"steady_input_local_jacobian.json"), all_candidate_results=rows,
        selected_x_ref=[selected["X2_ref"], selected["P2_ref"]] if selected else None,
        selected_u_ref=[selected["P100_ref"], selected["F200_ref"]] if selected else None,
        selected_delta_X2=selected["delta_X2"] if selected else None,
        Strong_metrics=strong_stats, Conservative_metrics=selected,
        safety_margin_increase=selected["safety_margin_increase"] if selected else None,
        economic_penalty_pct=selected["conservatism_penalty_pct"] if selected else None,
        selected_before_oracle=True, selected_before_SAC_training=True, selection_used_SAC_results=False,
        status="selected" if selected else "no_valid_conservative_reference",
        reason=p["selection_rule"] if selected else NO_CANDIDATE,
        protocol_sha256=old.v1.digest(root/"protocol_2d.json"))
    exp.immutable(root/"conservative_2d_selection.json", selection)
    plot(root, rows, selected, strong_stats)
    if not selected:
        exp.immutable(root/"oracle_conservative_summary.json", dict(status="not_run", reason=NO_CANDIDATE,
            total_candidates=0, safety_passing_candidates=0, positive_economic_candidates=0,
            best_economic_improvement_found_pct=None, conservative_seed42_training_allowed=False,
            finite_search_result_not_mathematical_upper_bound=True, training_performed=False, final_test_performed=False))
    print(f"Selection: {selection['status']}; {selection['reason']}", flush=True)


def plot(root, rows, selected, strong):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, note) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    ax.scatter(strong["minimum_X2_margin"], 0, marker="s", color="black", label="Strong")
    for r in rows:
        if r["baseline_evaluation_status"] != "completed": continue
        chosen = selected is not None and r["delta_X2"] == selected["delta_X2"]
        ax.scatter(r["minimum_X2_margin"], r["conservatism_penalty_pct"], s=95 if chosen else 40,
                   color="tab:green" if r["valid_Conservative_candidate"] else "tab:orange")
        ax.annotate(f"+{r['delta_X2']:.2f}"+(" selected" if chosen else ""),
                    (r["minimum_X2_margin"], r["conservatism_penalty_pct"]), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Worst-path minimum X2 margin (pp)", ylabel="Economic penalty vs Strong (%)")
    ax.grid(alpha=.2); ax.legend(); note.axis("off")
    lines = ["Preregistered 2D reference-only search", "Green: valid Conservative candidate", "Orange: simulated, not jointly eligible", ""]
    for r in rows:
        if r["baseline_evaluation_status"] != "completed":
            lines += [f"+{r['delta_X2']:.2f}: no scored closed-loop point", str(r.get("reject_reason") or r.get("failure"))[:75]]
    lines += ["", "No economic/SAC ranking used to choose P2.", "No failed candidate has fabricated J or margin."]
    note.text(0, 1, "\n".join(lines), va="top", fontsize=9)
    fig.tight_layout(); fig.savefig(root/"conservative_2d_safety_economics_tradeoff.png", dpi=180); plt.close(fig)


def oracle_gate(selection):
    selected = selection.get("Conservative_metrics")
    return bool(selected and selected.get("valid_Conservative_candidate")
        and selected["conservatism_penalty_abs"] > 2e-4
        and selection.get("selected_before_oracle") and selection.get("selected_before_SAC_training")
        and not selection.get("selection_used_SAC_results"))


def oracle(root=ROOT):
    """Finite privileged raw actor-action search, not MPC or a global bound."""
    p = locked(root)
    selection = old.load(root/"conservative_2d_selection.json")
    if not oracle_gate(selection):
        print("Conservative oracle NOT RUN: no jointly eligible frozen baseline; training blocked", flush=True)
        return
    output = root/"oracle_conservative_summary.json"
    if output.exists(): raise RuntimeError("Preserve finite-search evidence")
    if not old.load(root/"tests_receipt.json")["passed"]: raise RuntimeError("Tests must pass before oracle")
    env, check = full_check(v2.guard(exp.args_for()), selection["selected_x_ref"], p["common_initial_state"])
    if not check["passed"]: raise RuntimeError("Selected reference gate changed")
    initial = np.asarray(p["common_initial_state"])
    cache = {}
    for seed in old.DEV:
        base, _ = v2.archived_pair(root/"baseline_trajectories"/f"delta_{selection['selected_delta_X2']:.2f}"/f"seed_{seed}.csv")
        disturbance, _ = sample_disturbance_path(env[0], "zanon2019_stochastic", seed, p["steps"], PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
        np.testing.assert_array_equal(np.array([r["disturbance"] for r in base]), disturbance)
        if len(base) != p["steps"] or any(audited_metrics(base, env)[k] for k in old.SAFETY):
            raise RuntimeError("Paired Conservative baseline incomplete/unsafe")
        cache[seed] = disturbance, base
    rng = np.random.default_rng(p["oracle_search_seed"])
    summaries, per, elites = [], [], []
    for index in range(p["oracle_budget"]):
        segment = p["oracle_segments"][0 if index < p["oracle_budget"]//2 else 1]
        count = p["steps"]//segment
        if index < 25:
            grid = np.linspace(-1., 1., 5)
            actions = np.tile([grid[index//5], grid[index % 5]], (count, 1))
        elif elites and index % 3:
            previous = elites[int(rng.integers(len(elites)))]["actions"]
            at = np.minimum(np.arange(count)*len(previous)//count, len(previous)-1)
            actions = np.clip(previous[at]+rng.normal(0, .10, (count, 2)), -1, 1)
        else:
            actions = np.clip(rng.uniform(-.6, .6, (1, 2))+rng.normal(0, .2, (count, 2)), -1, 1)
        current, traces, failure = [], {}, None
        for seed, (disturbance, base) in cache.items():
            try:
                records, _ = old.rollout(env, disturbance, seed, old.v1.SegmentPolicy(actions, segment), initial)
                metric, _ = paired_metrics(records, base, env, 0., np.zeros(5))
                economic, series = pair_economics(base, records, initial); metric.update(economic)
                current.append(dict(candidate=index, validation_seed=seed, **metric))
                traces[seed] = exp.trace(base, records, series)
            except Exception as exc:
                failure = str(exc)
                if getattr(exc, "evidence", None):
                    write_csv(root/"oracle/failures"/f"candidate_{index:04d}_seed_{seed}.csv", exc.evidence)
                break
        safe = len(current) == 10 and failure is None
        row = dict(candidate=index, segment_steps=segment, safety_passed=safe, failure=failure)
        keys = ("economic_improvement_pct", "economic_win_fraction", "X2_std_ratio", "centered_X2_MAE_ratio",
                "P100_TV_ratio", "F200_TV_ratio", "P100_RMS_du_ratio", "F200_RMS_du_ratio", "W_exceedance_rate")
        row.update({k: float(np.mean([r[k] for r in current])) if safe else None for k in keys})
        row.update(worst_validation_improvement_pct=min(r["economic_improvement_pct"] for r in current) if safe else None,
            minimum_X2_margin=min(r["minimum_X2_margin"] for r in current) if safe else None,
            all_seed_positive=bool(safe and all(r["economic_improvement_pct"] > 0 for r in current)))
        for k in old.SAFETY: row[k] = sum(r[k] for r in current) if safe else None
        summaries.append(row); per.extend(current)
        if safe:
            value = lambda r: np.array([-r["economic_improvement_pct"], r["P100_TV_ratio"], r["F200_TV_ratio"], r["X2_std_ratio"]])
            trial = dict(**row, actions=actions); front = [*elites, trial]
            elites = [r for r in front if not any(np.all(value(t) <= value(r)) and np.any(value(t) < value(r)) for t in front)]
            if any(r["candidate"] == index for r in elites):
                for seed, rows in traces.items():
                    write_csv(root/"oracle/witness_trajectories"/f"candidate_{index:04d}_seed_{seed}.csv", rows)
                np.save(root/"oracle"/f"candidate_{index:04d}_actions.npy", actions)
        write_csv(root/"oracle/all_candidates.csv", summaries)
        write_csv(root/"oracle/per_realization.csv", per)
        print(f"2D oracle {index+1}/{p['oracle_budget']} safe={safe} gain={row['economic_improvement_pct']}", flush=True)
    safe = [r for r in summaries if r["safety_passed"]]
    positive = [r for r in safe if r["economic_improvement_pct"] > 0]
    best = max(positive, key=lambda r: r["economic_improvement_pct"]) if positive else None
    all_positive = [r for r in positive if r["all_seed_positive"]]
    write_csv(root/"oracle/pareto_front.csv", [{k: v for k, v in r.items() if k != "actions"} for r in elites])
    exp.immutable(output, dict(status="completed_finite_search", total_candidates=len(summaries),
        safety_passing_candidates=len(safe), positive_economic_candidates=len(positive),
        best_economic_improvement_found_pct=best["economic_improvement_pct"] if best else None,
        best_candidate=best, all_path_positive_count=len(all_positive),
        best_all_seed_positive_candidate=max(all_positive, key=lambda r: r["economic_improvement_pct"]) if all_positive else None,
        finite_search_result_not_mathematical_upper_bound=True, privileged_offline_search=True,
        conservative_seed42_training_allowed=bool(positive),
        training_performed=False, final_test_performed=False,
        selection_sha256=old.v1.digest(root/"conservative_2d_selection.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "tests", "search", "baselines", "oracle", "status"))
    args = parser.parse_args()
    if args.phase == "prepare": prepare()
    elif args.phase == "tests": exp.tests(ROOT)
    elif args.phase == "search": search()
    elif args.phase == "baselines": baselines()
    elif args.phase == "oracle": oracle()
    else:
        locked()
        print(old.load(ROOT/"conservative_2d_selection.json"))


if __name__ == "__main__": main()
