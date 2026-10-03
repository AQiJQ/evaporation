"""Stochastic evaporation comparison motivated by Zanon et al. (ECC 2019).

This is NOT a reproduction of their RL-NMPC controller. The paper states
disturbance variances, but does not specify a distribution, resampling cadence,
initial state, sampling time, random seed, or an algebraic Fig. 2 cost-difference
sign. Choices below are explicitly tagged "implementation choice for
reproduction" in summary.json. Fixed B/K/W/Z/Omega and the original safety
controllers are loaded unchanged. The finite-jump Gm/Bj automaton belongs to
the separate finite-jump benchmark, and is not invoked without state jumps.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .controlled_invariant_error_set import bounds_and_domains, certificate, facets, hull
from .model import EvaporatorModel
from .omega_dwell_event_supervisor import feasible_q_audit
from .omega_interior_anchor import InteriorAnchorController
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b
from .sac import SACAgent, SACConfig
from .train import OBS_DIM, ZeroResidualPolicy, run_episode


DEFAULT_OMEGA = REPO_DIR / (
    "evaporation_safe_sac/outputs_controlled_invariant_error_B/"
    "Omega_anchor_B_vertices.csv"
)
DEFAULT_CALIBRATION = REPO_DIR / (
    "evaporation_safe_sac/outputs_omega_safe_B_interior_anchor_seed42_300x300/"
    "baseline_reward_calibration.json"
)
DEFAULT_OUTPUT = REPO_DIR / "evaporation_safe_sac/outputs_zanon2019_stochastic"
PAPER_VARIANCES_F1_X1_T1_T200 = np.array([2.0, 1.0, 8.0, 5.0])
VARIABLE_NAMES = ("F1", "X1", "T1", "T200")
MODES = (
    "zanon2019_stochastic", "piecewise_stochastic",
    "zanon2019_sigma_as_std_audit",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_disturbance_path(cfg, mode: str, seed: int, steps: int,
                            variances, hold_min: int, hold_max: int):
    """Implementation choice for reproduction: zero-mean Gaussian noise.

    The paper calls the four reported sigma values *variances*, so standard
    deviations are their square roots. It does not specify Gaussianity or
    stepwise independence. `zanon2019_stochastic` uses iid steps by choice;
    `piecewise_stochastic` is an additional non-paper experiment. The
    sigma-as-std mode squares the same printed numbers and is sensitivity
    analysis only, never the main benchmark or training protocol.
    """
    if mode not in MODES or steps < 1:
        raise ValueError("Unknown stochastic mode or nonpositive horizon")
    supplied = np.asarray(variances, dtype=float)
    if supplied.shape != (4,) or np.any(supplied < 0) or not np.all(np.isfinite(supplied)):
        raise ValueError("Expected four finite, nonnegative paper scale values in F1,X1,T1,T200 order")
    # Sensitivity only: reinterpret the same four printed sigma numbers as
    # standard deviations. This is not the literal-variance main benchmark.
    variance = supplied ** 2 if mode == "zanon2019_sigma_as_std_audit" else supplied
    if hold_min < 1 or hold_max < hold_min:
        raise ValueError("Invalid piecewise hold interval")
    rng = np.random.default_rng(int(seed))
    nominal = np.asarray(cfg.disturbance_nominal, dtype=float)
    disturbance = np.empty((steps, 4), dtype=float)
    holds = []
    if mode != "piecewise_stochastic":
        disturbance[:] = nominal + rng.normal(size=(steps, 4)) * np.sqrt(variance)
        holds = [1] * steps
    else:
        at = 0
        while at < steps:
            hold = int(rng.integers(hold_min, hold_max + 1))
            sample = nominal + rng.normal(size=4) * np.sqrt(variance)
            disturbance[at:min(steps, at + hold)] = sample
            holds.append(min(steps - at, hold))
            at += hold
    # Implementation choice for reproduction: only clip nonphysical negative
    # feed flow/concentration; log exact counts and preserve other Gaussians.
    clipped = np.sum(disturbance[:, :2] < 0.0, axis=0).astype(int)
    disturbance[:, 0] = np.maximum(disturbance[:, 0], 0.0)
    disturbance[:, 1] = np.maximum(disturbance[:, 1], 0.0)
    return disturbance, {
        "seed": int(seed), "mode": mode, "steps": steps,
        "reported_scale_interpretation": (
            "sigma_as_standard_deviation_sensitivity_only"
            if mode == "zanon2019_sigma_as_std_audit" else "literal_variance"
        ),
        "variances_F1_X1_T1_T200": variance.tolist(),
        "standard_deviations_F1_X1_T1_T200": np.sqrt(variance).tolist(),
        "physical_clip_counts_F1_X1": clipped.tolist(),
        "hold_lengths": holds if mode == "piecewise_stochastic" else None,
    }


class StochasticInterior23(InteriorAnchorController):
    """Unchanged interior-anchor action path with 23D no-jump observation.

    Implementation choice for reproduction: the three finite-jump supervisor
    fields are zero (no jumps pending, no recovery, rank zero). The fourth
    remains the normalized current Omega admissible-q radius. No future clock
    or disturbance is exposed. This is not an active Gm/Bj certificate.
    """

    def actor_observation_extra(self, state):
        xi = self.model.normalized_state(state) - self.d.z_ref
        audit = feasible_q_audit(xi, self.omega, self.d, self.domain)
        q_half_width = 0.5 * float(np.min(
            self.domain["q_upper"] - self.domain["q_lower"]
        ))
        radius = 0.0 if audit is None else float(audit["radius"])
        return np.array([0.0, 0.0, 0.0,
                         np.clip(radius / max(q_half_width, 1e-12), 0, 1)],
                        dtype=np.float32)


def make_setup(steps: int, seed: int, design_path=DEFAULT_DESIGN,
               omega_path=DEFAULT_OMEGA, calibration_path=DEFAULT_CALIBRATION):
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed",
                           seed=seed, episodes=1, steps_per_episode=steps,
                           residual_parameterization="state_dependent_polytope")
    cfg.main_experiment_protocol = "zanon2019_stochastic_operating_uncertainty"
    cfg.disturbance_mode = "zanon2019_stochastic"
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    cfg.omega_performance_shaping_enabled = True
    cfg.training_segment_steps = 0
    # Frozen previous reward, not a new objective or a retuning step.
    frozen = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    weights = frozen["calibrated_weights"]
    cfg.paper2016_state_recovery_penalty_weight = float(weights["state_recovery"])
    cfg.paper2016_p100_move_penalty_weight = float(weights["P100_move"])
    cfg.paper2016_f200_move_penalty_weight = float(weights["F200_move"])
    cfg.paper2016_saturation_penalty_weight = float(weights["saturation"])
    model = EvaporatorModel(cfg)
    design = load_fixed_b(Path(design_path), cfg, model)
    omega = hull(np.loadtxt(omega_path, delimiter=",", skiprows=1))
    domain, _ = bounds_and_domains(cfg, model, design)
    if not certificate(omega, domain, design)["passed"]:
        raise RuntimeError("Frozen balanced-B Omega certificate failed")
    return cfg, model, design, omega, domain, weights


def make_agent(cfg, device="auto"):
    sac_cfg = SACConfig(
        gamma=cfg.gamma_rl, tau=cfg.tau,
        actor_lr=cfg.actor_learning_rate, critic_lr=cfg.critic_learning_rate,
        alpha_lr=cfg.entropy_learning_rate, hidden_dim=cfg.hidden_dim,
        alpha=cfg.alpha_initial, alpha_min=cfg.alpha_min,
        entropy_tuning_warmup_updates=cfg.entropy_tuning_warmup_updates,
        auto_entropy_tuning=cfg.auto_entropy_tuning,
        target_entropy=cfg.target_entropy,
        actor_ema_decay=cfg.actor_ema_decay,
    )
    return SACAgent(OBS_DIM + 4, 2, sac_cfg, device=device)


def load_actor(agent, path: Path):
    import torch
    try:
        saved = torch.load(path, map_location=agent.device, weights_only=False)
    except TypeError:
        saved = torch.load(path, map_location=agent.device)
    if int(saved.get("obs_dim", 23)) != 23 or int(saved.get("action_dim", 2)) != 2:
        raise ValueError("Actor checkpoint must have 23D observation and 2D action")
    agent.actor.load_state_dict(saved["actor"])
    agent.actor_ema.load_state_dict(saved["actor"])
    agent.set_policy_output_scale(float(saved.get("policy_output_scale", 1.0)))


def paired_rollouts(cfg, model, design, omega, domain, actor, path, seed):
    initial = np.asarray(cfg.robust_economic_reference_state, dtype=float).copy()
    def one(policy):
        ctrl = StochasticInterior23(cfg, model, design, omega, domain)
        return run_episode(
            cfg, model, ctrl, policy, None, np.random.default_rng(seed + 900000),
            training=False, global_step=0,
            disturbance_trajectory=path, initial_state_override=initial,
        )[:2]
    baseline = one(ZeroResidualPolicy())
    proposed = one(actor) if actor is not None else None
    for stat, records in (baseline, proposed) if proposed is not None else (baseline,):
        if len(records) != len(path) or any(np.max(np.abs(
            np.asarray(r["disturbance"]) - path[i]
        )) > 0 for i, r in enumerate(records)):
            raise RuntimeError("Paired stochastic disturbance trajectory mismatch")
        if any(np.max(np.abs(r["paper_state_shock"])) > 0 for r in records):
            raise RuntimeError("2016 state shock leaked into stochastic benchmark")
    if proposed is not None and np.max(np.abs(
        baseline[1][0]["state"] - proposed[1][0]["state"]
    )) > 0:
        raise RuntimeError("Paired initial states do not match")
    return baseline, proposed


def _changes(signal):
    delta = np.diff(signal)
    active = np.sign(delta[np.abs(delta) > 1e-6])
    return (float(np.sum(np.abs(delta))),
            float(np.sqrt(np.mean(delta ** 2))) if len(delta) else 0.0,
            int(np.sum(active[1:] != active[:-1])))


def trajectory_metrics(records, cfg, model, design):
    state = np.asarray([r["state"] for r in records], dtype=float)
    control = np.asarray([r["control"] for r in records], dtype=float)
    costs = np.asarray([r["economic_cost"] for r in records], dtype=float)
    errors = state - cfg.robust_economic_reference_state
    w = np.asarray([r["w_hat"] for r in records], dtype=float)
    # Normalized facet residuals avoid the unscaled cross-product tolerance
    # that can severely undercount violations for this very small W.
    hw, bw = facets(hull(design.w_vertices))
    w_excess = np.max(w @ hw.T - bw, axis=1)
    robust_lower = model.physical_input(design.robust_input_lower)
    robust_upper = model.physical_input(design.robust_input_upper)
    mid, half = 0.5 * (robust_lower + robust_upper), 0.5 * (
        robust_upper - robust_lower
    )
    eta = np.abs(control - mid) / half
    p_tv, p_rms, p_flips = _changes(control[:, 0])
    f_tv, f_rms, f_flips = _changes(control[:, 1])
    x2_deficit = np.maximum(25.0 - state[:, 0], 0.0)
    return {
        "steps": len(records), "J_econ": float(np.sum(costs)),
        "mean_stage_cost": float(np.mean(costs)),
        "median_stage_cost": float(np.median(costs)),
        "X2_IAE": float(np.sum(np.abs(errors[:, 0]))),
        "X2_ISE": float(np.sum(errors[:, 0] ** 2)),
        "P2_IAE": float(np.sum(np.abs(errors[:, 1]))),
        "P2_ISE": float(np.sum(errors[:, 1] ** 2)),
        "P100_TV": p_tv, "F200_TV": f_tv,
        "P100_delta_RMS": p_rms, "F200_delta_RMS": f_rms,
        "P100_input_sign_changes": p_flips,
        "F200_input_sign_changes": f_flips,
        "X2_violation_count": int(np.sum(x2_deficit > 0)),
        "X2_violation_rate": float(np.mean(x2_deficit > 0)),
        "X2_max_violation_magnitude": float(np.max(x2_deficit)),
        "X2_cumulative_violation_magnitude": float(np.sum(x2_deficit)),
        "P2_physical_violation_count": int(np.sum(
            (state[:, 1] < 40) | (state[:, 1] > 80))),
        "physical_input_violation_count": int(np.sum(np.any(
            (control < cfg.input_lower) | (control > cfg.input_upper), axis=1))),
        "physical_state_violation_steps": int(sum(bool(r["physical_state_violation"])
                                                   for r in records)),
        "Omega_exit_count": int(sum(not bool(r["in_Omega_before"])
                                    for r in records)),
        "Omega_exit_events": int(sum(bool(r["omega_exit_event"])
                                      for r in records)),
        "local_RPI_Z_exit_count": int(sum(not bool(r["in_Z_before"])
                                          for r in records)),
        "QP_infeasible_count": int(sum(not bool(r["qp_feasible"])
                                       for r in records)),
        "W_exceedance_count": int(np.sum(w_excess > 1e-8)),
        "W_exceedance_rate": float(np.mean(w_excess > 1e-8)),
        "W_max_facet_excess": float(np.max(w_excess)),
        "robust_region_violation_count": int(sum(bool(r["robust_region_violation"])
                                                  for r in records)),
        "P100_lower_bound_steps": int(np.sum(np.isclose(control[:, 0], robust_lower[0], atol=1e-6, rtol=0))),
        "P100_upper_bound_steps": int(np.sum(np.isclose(control[:, 0], robust_upper[0], atol=1e-6, rtol=0))),
        "F200_lower_bound_steps": int(np.sum(np.isclose(control[:, 1], robust_lower[1], atol=1e-6, rtol=0))),
        "F200_upper_bound_steps": int(np.sum(np.isclose(control[:, 1], robust_upper[1], atol=1e-6, rtol=0))),
        "P100_outer_10pct_steps": int(np.sum(eta[:, 0] > 0.9)),
        "F200_outer_10pct_steps": int(np.sum(eta[:, 1] > 0.9)),
        "any_outer_10pct_steps": int(np.sum(np.any(eta > 0.9, axis=1))),
    }


def save_trajectory(path: Path, baseline_records, policy_records, disturbance):
    rows = []
    for i, d in enumerate(disturbance):
        base = baseline_records[i]
        pol = policy_records[i] if policy_records is not None else None
        row = {"step": i, "time_seconds": i,
               **{name: float(d[j]) for j, name in enumerate(VARIABLE_NAMES)}}
        for prefix, record in (("baseline", base), ("SAC", pol)):
            if record is None:
                continue
            row.update({
                f"{prefix}_X2": float(record["state"][0]),
                f"{prefix}_P2": float(record["state"][1]),
                f"{prefix}_P100": float(record["control"][0]),
                f"{prefix}_F200": float(record["control"][1]),
                f"{prefix}_stage_cost": float(record["economic_cost"]),
                f"{prefix}_in_Z": bool(record["in_Z_before"]),
                f"{prefix}_in_Omega": bool(record["in_Omega_before"]),
                f"{prefix}_QP_feasible": bool(record["qp_feasible"]),
            })
        if pol is not None:
            row["delta_l_SAC_minus_baseline"] = (
                pol["economic_cost"] - base["economic_cost"]
            )
        rows.append(row)
    write_csv(path, rows)


def plot_six(path: Path, baseline_records, policy_records, cfg):
    import matplotlib.pyplot as plt
    path.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(baseline_records))
    signals = (
        ("Figure_A_X2.png", "X2", 0, 25.0, None),
        ("Figure_B_P2.png", "P2", 1, 40.0, 80.0),
        ("Figure_C_P100.png", "P100", 0, 100.0, 400.0),
        ("Figure_D_F200.png", "F200", 1, 100.0, 400.0),
    )
    for filename, label, axis, lower, upper in signals:
        key = "state" if label in ("X2", "P2") else "control"
        b = np.array([r[key][axis] for r in baseline_records])
        p = np.array([r[key][axis] for r in policy_records])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, b, label="paired zero-residual robust baseline", lw=1)
        ax.plot(t, p, label="residual SAC", lw=1)
        ax.axhline(lower, color="black", ls="--", label=f"physical lower {lower:g}")
        if upper is not None:
            ax.axhline(upper, color="gray", ls="--", label=f"physical upper {upper:g}")
        if label == "X2":
            ymin = min(float(np.min(b)), float(np.min(p)), lower) - 0.2
            ax.axhspan(ymin, lower, color="red", alpha=0.08)
            ax.scatter(t[p < lower], p[p < lower], color="red", s=9,
                       label="SAC X2 violation")
        ax.set(xlabel="time (sampling steps; 1 s/step implementation choice)",
               ylabel=label)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
    b_cost = np.array([r["economic_cost"] for r in baseline_records])
    p_cost = np.array([r["economic_cost"] for r in policy_records])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, p_cost - b_cost, lw=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="sampling step", ylabel="stage cost difference",
           title="Figure E: delta_l = l_SAC - l_baseline; negative favors SAC")
    fig.tight_layout(); fig.savefig(path / "Figure_E_instantaneous_economic_difference.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, np.cumsum(b_cost), label="paired zero-residual baseline")
    ax.plot(t, np.cumsum(p_cost), label="residual SAC")
    ax.set(xlabel="sampling step", ylabel="cumulative economic cost",
           title="Figure F: cumulative J(t) = sum stage costs")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(path / "Figure_F_cumulative_economic_cost.png", dpi=150)
    plt.close(fig)


def aggregate(rows):
    numeric = [key for key, value in rows[0].items()
               if isinstance(value, (int, float, np.integer, np.floating))
               and key != "seed"]
    result = []
    for key in numeric:
        values = np.asarray([r[key] for r in rows], dtype=float)
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
        half = 1.96 * sd / np.sqrt(len(values)) if sd is not None else None
        result.append({"metric": key, "mean": float(np.mean(values)),
                       "sample_std": sd, "median": float(np.median(values)),
                       "min": float(np.min(values)), "max": float(np.max(values)),
                       "ci95_low": float(np.mean(values) - half) if half is not None else None,
                       "ci95_high": float(np.mean(values) + half) if half is not None else None})
    return result


def paper_specification(mode, cfg, steps, variances, hold_min, hold_max):
    effective_variances = (
        np.asarray(variances, dtype=float) ** 2
        if mode == "zanon2019_sigma_as_std_audit"
        else np.asarray(variances, dtype=float)
    )
    return {
        "source": "Zanon, Gros, Bemporad, ECC 2019, Numerical Example pp. 2261-2262 and Fig. 2 p. 2263",
        "paper_reported_disturbance_variances": {
            "X1": 1, "F1": 2, "T1": 8, "T200": 5,
        },
        "paper_feed_flow_notation_note": "prose says flow F2, variance subscript says F1; project plant uses exogenous F1",
        "paper_model_equations_and_economic_cost": "omitted in ECC2019; referred to earlier publications",
        "paper_random_distribution": "not specified in the paper",
        "paper_per_step_independence": "not specified in the paper",
        "paper_initial_state": "not specified in the paper",
        "paper_sampling_time": "not specified in the paper",
        "paper_evaluation_seed_or_realization": "not specified in the paper",
        "paper_exact_gain_formula": "not specified in the paper",
        "paper_gain_baselines": "14% vs naive-initial-guess NMPC; 12% vs nominal-economic-tuned NMPC",
        "paper_state_relaxation": "all state bounds softened by s_k in Eq. (3e); W_s=I and w_s=1 in Numerical Example; exact solver handling not specified in the paper",
        "paper_fig2_horizon": "horizontal axis shown 0-1000; textual simulation horizon not specified in the paper",
        "paper_fig2_cost_difference_sign": "not algebraically specified in the paper; plotted values are mostly negative",
        "implementation_choice_for_reproduction": {
            "mode": mode,
            "distribution": "independent zero-mean Gaussian per step" if mode != "piecewise_stochastic"
                            else "Gaussian draw held for uniformly sampled 20-50 steps",
            "reported_scale_interpretation": (
                "sigma-as-std sensitivity only; not main paper benchmark"
                if mode == "zanon2019_sigma_as_std_audit"
                else "literal variance"
            ),
            "variances_F1_X1_T1_T200": effective_variances.tolist(),
            "standard_deviations_F1_X1_T1_T200": np.sqrt(effective_variances).tolist(),
            "nominal_F1_X1_T1_T200": cfg.disturbance_nominal.tolist(),
            "initial_state": cfg.robust_economic_reference_state.tolist(),
            "horizon_samples": steps, "sampling_seconds": cfg.dt_min * 60,
            "piecewise_hold_min_max": [hold_min, hold_max] if mode == "piecewise_stochastic" else None,
            "negative_F1_X1_clipping": "clip to zero, count each clip",
            "Fig2_difference": "l_SAC - l_paired_zero_residual; negative favors SAC",
            "training_observation": "23D: original 19D plus [0,0,0,normalized Omega q radius]; no future information",
            "finite_jump_supervisor": "not used in no-state-jump Experiment I; unchanged in Experiment III",
            "state_constraints": "unchanged hard physical constraints, unlike the paper's softened NMPC state bounds",
        },
    }


def report_markdown(path, spec, rows, agg):
    mean = next((r["mean"] for r in agg
                 if r["metric"] == "economic_improvement_percent"), None)
    out = ["# ECC 2019-motivated stochastic evaporation comparison", "",
           "## Benchmark definition", "",
           "Project simplified nonlinear evaporation plant, physical states X2/P2 and inputs P100/F200; stochastic exogenous F1/X1/T1/T200. ECC2019 omits model and cost equations and refers to earlier publications, so exact plant identity cannot be established from this PDF alone.",
           f"Implementation choice for reproduction: {spec['implementation_choice_for_reproduction']['distribution']}; reported paper variances are converted to standard deviations with square roots.",
           "The paper does not specify the distribution, per-step independence, initial state, sampling time, exact seed, or exact gain formula.",
           "", "## Our paired zero-residual and residual SAC results", "",
           f"Evaluated {len(rows)} paired seeds. Mean economic improvement vs *our own* zero-residual robust baseline: {mean if mean is not None else 'unavailable'}%.",
           "Both controllers use identical initial state and saved disturbance sequence for each seed.",
           "", "## Constraint/safety result", "",
           "See per_seed_metrics.csv for X2/P2/input, Omega/Z/QP and strict normalized-facet W exceedance counts. An out-of-W rollout is not formally certified by the fixed W/Omega/Gm assumptions.",
           "", "## Economic and control-quality result", "",
           "See aggregate_summary.csv for J_econ, stage costs, X2/P2 IAE/ISE, P100/F200 TV, RMS moves, sign changes and robust-bound occupancy.",
           "", "## Comparison scope with ECC 2019", "",
           "Under a stochastic operating-uncertainty benchmark motivated by Zanon et al. (ECC 2019), the proposed controller is evaluated with explicit physical and robust-safety auditing.",
           "This is not an implementation of their RL-NMPC, whose state constraints are softened. The paper's 14%/12% gains are against naive and nominal-economic NMPC, not our zero-residual baseline. No direct economic or formal-safety superiority claim is made.",
           "", "## Other experiments", "",
           "Experiment II (single finite jump) and Experiment III (unknown-time repeated jumps with dwell20 Gm/Bj) remain separate and unchanged.", ""]
    path.write_text("\n".join(out), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disturbance-mode", choices=MODES,
                        default="zanon2019_stochastic")
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=420000)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--variances", type=float, nargs=4,
                        default=PAPER_VARIANCES_F1_X1_T1_T200.tolist(),
                        metavar=("F1", "X1", "T1", "T200"))
    parser.add_argument("--hold-min", type=int, default=20)
    parser.add_argument("--hold-max", type=int, default=50)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--baseline-only", action="store_true",
                        help="Explicit diagnostic without paired SAC comparison")
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.eval_seeds < 1:
        raise ValueError("--eval-seeds must be positive")
    if (args.actor_checkpoint is None) != args.baseline_only:
        raise ValueError("Supply exactly one of --actor-checkpoint or --baseline-only")
    if (args.output_dir / "summary.json").exists():
        raise RuntimeError("Output directory already contains a completed evaluation")
    cfg, model, design, omega, domain, weights = make_setup(
        args.steps, args.seed_start, args.design, args.omega_vertices
    )
    cfg.disturbance_mode = args.disturbance_mode
    actor = None
    if args.actor_checkpoint is not None:
        actor = make_agent(cfg, args.device)
        load_actor(actor, args.actor_checkpoint)
    spec = paper_specification(args.disturbance_mode, cfg, args.steps,
                               np.asarray(args.variances), args.hold_min, args.hold_max)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, draws = [], []
    for index in range(args.eval_seeds):
        seed = args.seed_start + index
        disturbance, metadata = sample_disturbance_path(
            cfg, args.disturbance_mode, seed, args.steps,
            args.variances, args.hold_min, args.hold_max,
        )
        base, proposed = paired_rollouts(
            cfg, model, design, omega, domain, actor, disturbance, seed,
        )
        base_metrics = trajectory_metrics(base[1], cfg, model, design)
        row = {"seed": seed, "disturbance_clip_count_F1": metadata["physical_clip_counts_F1_X1"][0],
               "disturbance_clip_count_X1": metadata["physical_clip_counts_F1_X1"][1],
               **{f"baseline_{k}": v for k, v in base_metrics.items()}}
        if proposed is not None:
            policy_metrics = trajectory_metrics(proposed[1], cfg, model, design)
            row.update({f"SAC_{k}": v for k, v in policy_metrics.items()})
            row["economic_improvement_percent"] = 100 * (
                base_metrics["J_econ"] - policy_metrics["J_econ"]
            ) / max(abs(base_metrics["J_econ"]), 1e-12)
            row["X2_IAE_ratio"] = policy_metrics["X2_IAE"] / max(base_metrics["X2_IAE"], 1e-12)
            row["P2_IAE_ratio"] = policy_metrics["P2_IAE"] / max(base_metrics["P2_IAE"], 1e-12)
            row["paired_disturbance_identity_max_error"] = 0.0
        rows.append(row)
        draws.append(metadata)
        save_trajectory(args.output_dir / "trajectories" / f"seed_{seed}.csv",
                        base[1], proposed[1] if proposed is not None else None,
                        disturbance)
        if index == 0 and proposed is not None:
            plot_six(args.output_dir / "figures", base[1], proposed[1], cfg)
        write_csv(args.output_dir / "per_seed_metrics.csv", rows)
        print(f"seed={seed} baseline_W_exceed={base_metrics['W_exceedance_count']}"
              + (f" SAC_W_exceed={policy_metrics['W_exceedance_count']}"
                 f" improvement={row['economic_improvement_percent']:.4f}%"
                 if proposed is not None else ""), flush=True)
    agg = aggregate(rows)
    write_csv(args.output_dir / "aggregate_summary.csv", agg)
    (args.output_dir / "aggregate_summary.json").write_text(
        json.dumps(agg, indent=2), encoding="utf-8")
    w_outside = any(r["baseline_W_exceedance_count"] > 0
                    or r.get("SAC_W_exceedance_count", 0) > 0 for r in rows)
    result = {"benchmark": "ECC2019-motivated; not exact paper reproduction",
              "formal_safety_status": (
                  "not_certified_W_exceeded_in_evaluation"
                  if w_outside else "conditional_affine_W_certificate_only"
              ),
              "controller": "fixed balanced B interior-anchor + Z/Omega one-step QP; Gm/Bj reserved for finite-jump test",
              "actor_checkpoint": str(args.actor_checkpoint) if args.actor_checkpoint else None,
              "paper_specification": spec,
              "frozen_reward_weights": weights,
              "paired_random_seed_start": args.seed_start,
              "eval_seeds": args.eval_seeds,
              "ci95_method": "mean +/- 1.96 * sample_std / sqrt(number_of_seeds); null for one seed",
              "per_seed_W_exceedance": [
                  {"seed": r["seed"], "baseline": r["baseline_W_exceedance_count"],
                   "SAC": r.get("SAC_W_exceedance_count")} for r in rows],
              "clipping_by_seed": draws,
              "gain_14_percent_baseline": "paper naive-initial-guess NMPC, not implemented here",
              "gain_12_percent_baseline": "paper nominal-economic NMPC, not implemented here"}
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    report_markdown(args.output_dir / "zanon2019_comparison_summary.md", spec, rows, agg)


if __name__ == "__main__":
    main()
