"""Read-only observation/Markov audit for the fixed balanced-B controller."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .controlled_invariant_error_set import bounds_and_domains, contains, hull
from .model import EvaporatorModel
from .omega_online_closed_loop import OmegaSafeOnlineController
from .residual_action_space_diagnosis import DEFAULT_DESIGN, REPO_DIR, load_fixed_b
from .train import OBS_DIM, observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_controlled_invariant_error_B/"
        "Omega_anchor_B_vertices.csv"
    ))
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_omega_observation_audit_B"
    ))
    args = parser.parse_args()
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed")
    cfg.residual_parameterization = "state_dependent_polytope"
    cfg.residual_reserve_mode = "fraction_of_max"
    cfg.residual_reserve_fraction = 0.8
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    omega = hull(np.loadtxt(args.omega_vertices, delimiter=",", skiprows=1))
    domain, _ = bounds_and_domains(cfg, model, design)

    labels = (
        "xi_X2", "xi_P2", "e_X2", "e_P2",
        "previous_final_P100_minus_vref", "previous_final_F200_minus_vref",
        "K_00", "K_01", "K_10", "K_11",
        "theta_h_0", "theta_h_1", "theta_h_2", "theta_h_3",
        "theta_p_0", "theta_p_1", "theta_m_angle", "w_est_X2", "w_est_P2",
    )
    if len(labels) != OBS_DIM:
        raise RuntimeError(f"Observation labels {len(labels)} != OBS_DIM {OBS_DIM}")
    reference = cfg.robust_economic_reference_state.copy()
    controller = OmegaSafeOnlineController(cfg, model, design, omega, domain)
    controller.reset(reference)
    previous_u = design.v_ref + np.array([0.12, -0.09])
    obs = observation(model, controller, reference, previous_u, np.zeros(2))
    previous_error = float(np.max(np.abs(
        obs[4:6] - (previous_u - design.v_ref)
    )))
    if previous_error > 1e-7:
        raise RuntimeError("Observation did not preserve previous applied input")

    # Choose a single actual state that is in S and Omega but outside Z when
    # the nominal state remains at the reference.  Resetting the nominal state
    # to that same actual state gives a certified Z-mode comparison point.
    probe_state = None
    for delta in (0.01, 0.02, 0.05, 0.10, 0.20, 0.40):
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                candidate = reference.copy()
                candidate[axis] += sign * delta
                x = model.normalized_state(candidate)
                xi = x - design.z_ref
                if (contains(omega, xi) and not contains(design.rpi_boundary, xi)
                        and np.all(x >= design.invariant_lower + 1e-8)
                        and np.all(x <= design.invariant_upper - 1e-8)):
                    probe_state = candidate
                    break
            if probe_state is not None:
                break
        if probe_state is not None:
            break
    if probe_state is None:
        raise RuntimeError("No common physical state for the two-mode probe")

    rows = []
    for action in (
        np.array([1.0, 1.0]), np.array([1.0, -1.0]),
        np.array([-1.0, 1.0]), np.array([-1.0, -1.0]),
        np.array([0.5, 0.5]), np.array([0.0, 0.0]),
    ):
        for nominal_start, expected_mode in (
            (probe_state, "Z_mode_existing_controller"),
            (reference, "Omega_safe_one_step_QP"),
        ):
            ctrl = OmegaSafeOnlineController(cfg, model, design, omega, domain)
            ctrl.reset(nominal_start)
            probe_obs = observation(model, ctrl, probe_state, design.v_ref, np.zeros(2))
            control, info = ctrl.act(probe_state, action)
            if info["mode"] != expected_mode or not info["qp_feasible"]:
                raise RuntimeError(f"Two-mode probe failed: {info['mode']}")
            requested = np.asarray(info["requested_residual"])
            applied = np.asarray(info["applied_residual"])
            rows.append({
                "actual_X2": probe_state[0], "actual_P2": probe_state[1],
                "nominal_start_X2": nominal_start[0],
                "nominal_start_P2": nominal_start[1],
                "mode": info["mode"],
                "actor_a_P100": action[0], "actor_a_F200": action[1],
                "observation_xi_X2": probe_obs[0],
                "observation_xi_P2": probe_obs[1],
                "observation_e_X2": probe_obs[2],
                "observation_e_P2": probe_obs[3],
                "requested_residual_P100_normalized": requested[0],
                "requested_residual_F200_normalized": requested[1],
                "applied_residual_P100_normalized": applied[0],
                "applied_residual_F200_normalized": applied[1],
                "applied_over_requested": (
                    float(np.linalg.norm(applied) / np.linalg.norm(requested))
                    if np.linalg.norm(requested) > 1e-12 else None
                ),
                "final_P100": control[0], "final_F200": control[1],
            })
    paired_differences = []
    for index in range(0, len(rows), 2):
        z, om = rows[index:index + 2]
        paired_differences.append({
            "actor_action": [z["actor_a_P100"], z["actor_a_F200"]],
            "final_input_difference_Omega_minus_Z": [
                om["final_P100"] - z["final_P100"],
                om["final_F200"] - z["final_F200"],
            ],
            "applied_residual_difference_Omega_minus_Z": [
                om["applied_residual_P100_normalized"]
                - z["applied_residual_P100_normalized"],
                om["applied_residual_F200_normalized"]
                - z["applied_residual_F200_normalized"],
            ],
        })
    report = {
        "observation_dimension": OBS_DIM,
        "observation_index_labels": list(labels),
        "normalized_actual_state_explicit": False,
        "normalized_actual_state_recoverable_from_xi_plus_fixed_zref": True,
        "rpi_error_explicit": True,
        "nominal_z_recoverable_from_x_and_e": True,
        "previous_final_applied_input_explicit_as_normalized_offset": True,
        "previous_applied_input_reconstruction_max_abs_error": previous_error,
        "previous_applied_input_offset_robust_range": [
            (design.robust_input_lower - design.v_ref).tolist(),
            (design.robust_input_upper - design.v_ref).tolist(),
        ],
        "previous_applied_input_offset_clipped_within_robust_range": bool(
            np.any(np.abs(np.concatenate([
                design.robust_input_lower - design.v_ref,
                design.robust_input_upper - design.v_ref,
            ])) > 4.0)
        ),
        "mode_indicator_explicit": False,
        "mode_inferable_from_e_and_fixed_Z_in_exact_arithmetic": True,
        "feasible_action_authority_explicit": False,
        "move_penalty_hidden_previous_input_history": False,
        "paper_shock_episode_clock_explicit": False,
        "paper_shock_clock_markov_caveat": (
            "The fixed 0/20/40-second state-shock schedule is not an observation "
            "channel.  If future shocks are treated as part of the transition "
            "law, an episode-clock state is missing, independent of move penalties."
        ),
        "clipping_note": "Each observation component is clipped to [-4,4]; verify information loss if operating ranges change.",
        "same_actual_state_two_mode_probe": probe_state.tolist(),
        "paired_projection_differences": paired_differences,
        "interpretation": (
            "The previous final applied input is present, so move penalties do not "
            "hide that one-step history.  Safety mode and state-dependent action "
            "authority are not explicit observation channels, although with "
            "fixed known geometry they can be inferred from x and e; the same "
            "actor output can map to different applied controls."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "observation_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "two_mode_action_probe.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
