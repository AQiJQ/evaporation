"""High-resolution certification and zero-residual audit for robust reference.

This entry point rebuilds the complete proposed design from configuration.  It
does not train SAC.  The generated calibration recommendation is diagnostic;
source configuration is never changed by this script.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .config import ExperimentConfig
from .control import (
    SafeController,
    build_safety_design,
    estimate_hinf_norm,
    point_in_convex_polygon,
    project_qp_2d,
    safe_projected_base,
    spectral_radius,
)
from .model import EvaporatorModel
from .theta_learning import OnlineThetaLearner
from .train import (
    PAPER2016_SCENARIOS,
    ZeroResidualPolicy,
    records_to_arrays,
    run_episode,
    save_rollout_csv,
    save_theta_design,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hinf-points", type=int, default=262144)
    parser.add_argument("--qp-grid-points", type=int, default=41)
    parser.add_argument("--target-penalty-fraction", type=float, default=0.10)
    parser.add_argument(
        "--reuse-design", type=Path, default=None,
        help="Resume post-build auditing from a previously saved design NPZ.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "evaporation_safe_sac/outputs_robust_reference_margin_audit"
        ),
    )
    return parser.parse_args()


def _json_ready(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _physical_bounds(model, lower, upper, *, state: bool):
    transform = model.physical_state if state else model.physical_input
    values = transform(np.vstack([lower, upper]))
    return {"lower": values[0], "upper": values[1]}


def _load_design(path: Path, cfg, model):
    """Restore every field used by SafeController and the dense audit."""
    with np.load(path) as data:
        values = {key: np.asarray(data[key]).copy() for key in data.files}
    k = values["K"]
    boundary = values["rpi_boundary"]
    input_boundary = boundary @ k.T
    rpi_lower = values["rpi_support_lower"]
    rpi_upper = values["rpi_support_upper"]
    input_lower = np.max(-input_boundary, axis=0)
    input_upper = np.max(input_boundary, axis=0)
    return SimpleNamespace(
        a=values["A"], b=values["B"], affine=values["affine"], k=k,
        gamma_design=float(cfg.hinf_gamma), gamma_sampled=float("nan"),
        w_bound=np.max(np.abs(values["W_vertices"]), axis=0),
        w_vertices=values["W_vertices"],
        w_data_hull=values["W_data_hull"],
        theta_m_matrix=values["M"], theta_m_bound=values["m"],
        theta_m_angle=float(values["M_angle"]),
        rpi_boundary=boundary,
        rpi_support=np.maximum(rpi_lower, rpi_upper),
        rpi_support_lower=rpi_lower, rpi_support_upper=rpi_upper,
        input_rpi_support=np.maximum(input_lower, input_upper),
        input_rpi_support_lower=input_lower,
        input_rpi_support_upper=input_upper,
        robust_state_lower=values["robust_state_lower"],
        robust_state_upper=values["robust_state_upper"],
        robust_input_lower=values["robust_input_lower"],
        robust_input_upper=values["robust_input_upper"],
        robust_region_scale=float(values["robust_region_scale"]),
        x_lower_tight=values["x_lower_tight"],
        x_upper_tight=values["x_upper_tight"],
        u_lower_tight=values["u_lower_tight"],
        u_upper_tight=values["u_upper_tight"],
        z_ref=model.normalized_state(cfg.robust_economic_reference_state),
        v_ref=model.normalized_input(cfg.robust_economic_reference_input),
        invariant_lower=values["invariant_lower"],
        invariant_upper=values["invariant_upper"],
        invariant_scale=float("nan"),
        gain_state_weight_scale=float("nan"),
        gain_input_weight_scale=float("nan"),
        gain_candidates_feasible=0,
        reference_rpi_width_physical=(rpi_lower + rpi_upper) * cfg.state_scale,
        reference_rpi_area_physical=float("nan"),
        reference_x_minus_z_width_physical=(
            values["x_upper_tight"] - values["x_lower_tight"]
        ) * cfg.state_scale,
        reference_x_minus_z_area_physical=float("nan"),
        box_reference_rpi_width_physical=(rpi_lower + rpi_upper) * cfg.state_scale,
        box_reference_rpi_area_physical=float("nan"),
        theta_h=values["h"], theta_p=values["p"],
        nominal_policy_gain=values["nominal_policy_gain"],
        nominal_policy_offset=values["nominal_policy_offset"],
        minimum_residual_authority=float(values["minimum_residual_authority"]),
    )


def _qp_grid_audit(cfg, design, points: int) -> dict[str, object]:
    axes = [
        np.linspace(lo, hi, int(points))
        for lo, hi in zip(design.invariant_lower, design.invariant_upper)
    ]
    minimum_authority = 1.0
    base_failures = zero_failures = corner_failures = 0
    probes = 0
    for coordinates in product(*axes):
        probes += 1
        z = np.asarray(coordinates, dtype=float)
        center_next = design.a @ z + design.affine
        aq = np.vstack([np.eye(2), -np.eye(2), design.b, -design.b])
        bq = np.concatenate([
            design.u_upper_tight,
            -design.u_lower_tight,
            design.invariant_upper - center_next,
            -design.invariant_lower + center_next,
        ])
        theta = design.nominal_policy_gain @ z + design.nominal_policy_offset
        base, feasible, authority = safe_projected_base(
            theta, aq, bq, cfg.residual_action_scale,
            cfg.qp_min_residual_authority, cfg.invariant_set_margin,
        )
        minimum_authority = min(minimum_authority, float(authority))
        base_failures += int(not feasible)
        zero, zero_feasible = project_qp_2d(base, aq, bq)
        zero_failures += int(
            (not zero_feasible) or np.linalg.norm(zero - base) > 1e-8
        )
        corner_ok = True
        for signs in product((-1.0, 1.0), repeat=2):
            corner = (
                base + cfg.qp_min_residual_authority
                * cfg.residual_action_scale * np.asarray(signs)
            )
            projected, corner_feasible = project_qp_2d(corner, aq, bq)
            if (
                not corner_feasible
                or np.linalg.norm(projected - corner) > 1e-8
            ):
                corner_ok = False
                break
        corner_failures += int(not corner_ok)
    return {
        "grid_points_per_axis": int(points),
        "probe_count": probes,
        "base_failures": base_failures,
        "zero_residual_failures": zero_failures,
        "minimum_authority_corner_failures": corner_failures,
        "minimum_residual_authority": minimum_authority,
        "authority_requirement": float(cfg.qp_min_residual_authority),
        "authority_surplus": (
            minimum_authority - float(cfg.qp_min_residual_authority)
        ),
        "passed": bool(
            base_failures == zero_failures == corner_failures == 0
            and minimum_authority
            >= float(cfg.qp_min_residual_authority) - 1e-8
        ),
    }


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.target_penalty_fraction <= 0.15:
        raise ValueError("target penalty fraction must be in [0.05, 0.15]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(
        benchmark_profile="zanon2016", experiment_mode="proposed",
        seed=args.seed,
    )
    model = EvaporatorModel(cfg)
    exact_input = model.steady_input(
        cfg.robust_economic_reference_state, cfg.disturbance_nominal
    )
    if not np.allclose(exact_input, cfg.robust_economic_reference_input, atol=1e-9):
        raise RuntimeError("configured robust-reference input is not exact")
    derivative = model.derivative(
        cfg.robust_economic_reference_state, exact_input,
        cfg.disturbance_nominal,
    )

    if args.reuse_design is not None:
        design = _load_design(args.reuse_design, cfg, model)
    else:
        initial = build_safety_design(
            cfg, model, np.random.default_rng(args.seed),
            theta_k=cfg.theta_k_initial_gain,
        )
        learner = OnlineThetaLearner(
            cfg, model, initial, np.random.default_rng(args.seed + 17001)
        )
        design, _ = learner.build_certified_robust_operating_design(initial)
        save_theta_design(args.output_dir / "certified_design.npz", design)

    acl = design.a + design.b @ design.k
    hinf_dense = estimate_hinf_norm(
        design.a, design.b, design.k, cfg.hinf_q, cfg.hinf_r,
        points=args.hinf_points,
    )
    rpi_width = (
        design.rpi_support_lower + design.rpi_support_upper
    ) * cfg.state_scale
    rpi_area = 0.5 * abs(float(np.sum(
        (design.rpi_boundary[:, 0] * cfg.state_scale[0])
        * np.roll(design.rpi_boundary[:, 1] * cfg.state_scale[1], -1)
        - (design.rpi_boundary[:, 1] * cfg.state_scale[1])
        * np.roll(design.rpi_boundary[:, 0] * cfg.state_scale[0], -1)
    )))
    rpi_failures = sum(
        not point_in_convex_polygon(acl @ vertex + disturbance,
                                    design.rpi_boundary, tol=2e-8)
        for vertex in design.rpi_boundary
        for disturbance in design.w_vertices
    )
    s_bounds = _physical_bounds(
        model, design.invariant_lower, design.invariant_upper, state=True
    )
    x_tight = _physical_bounds(
        model, design.x_lower_tight, design.x_upper_tight, state=True
    )
    u_tight = _physical_bounds(
        model, design.u_lower_tight, design.u_upper_tight, state=False
    )
    reference = np.asarray(cfg.robust_economic_reference_state)
    steady_input = np.asarray(cfg.robust_economic_reference_input)
    state_lower_margin = reference - s_bounds["lower"]
    state_upper_margin = s_bounds["upper"] - reference
    input_lower_margin = steady_input - u_tight["lower"]
    input_upper_margin = u_tight["upper"] - steady_input
    qp_audit = _qp_grid_audit(cfg, design, args.qp_grid_points)

    design_audit = {
        "reference_state": reference,
        "steady_input": steady_input,
        "steady_derivative": derivative,
        "steady_derivative_inf_norm": float(np.max(np.abs(derivative))),
        "steady_economic_cost": float(model.economic_cost(
            reference, steady_input, cfg.disturbance_nominal
        )),
        "paper2016_original_benchmark_state": cfg.paper2016_steady_state,
        "paper2016_original_benchmark_input": cfg.paper2016_steady_input,
        "reference_in_S": bool(
            np.all(design.z_ref >= design.invariant_lower - 1e-10)
            and np.all(design.z_ref <= design.invariant_upper + 1e-10)
        ),
        "S_physical_bounds": s_bounds,
        "reference_to_S_lower_margin": state_lower_margin,
        "reference_to_S_upper_margin": state_upper_margin,
        "reference_to_S_minimum_margin": float(np.min(np.r_[
            state_lower_margin, state_upper_margin
        ])),
        "RPI_width_physical": rpi_width,
        "RPI_area_physical": rpi_area,
        "RPI_one_step_vertex_disturbance_failures": int(rpi_failures),
        "X_minus_Z_physical_bounds": x_tight,
        "U_minus_KZ_physical_bounds": u_tight,
        "steady_input_to_U_minus_KZ_lower_margin": input_lower_margin,
        "steady_input_to_U_minus_KZ_upper_margin": input_upper_margin,
        "steady_input_to_U_minus_KZ_minimum_margin": float(np.min(np.r_[
            input_lower_margin, input_upper_margin
        ])),
        "spectral_radius": spectral_radius(acl),
        "hinf_frequency_points": int(args.hinf_points),
        "hinf_sampled_norm": hinf_dense,
        "hinf_gamma": float(cfg.hinf_gamma),
        "hinf_gamma_minus_sampled_norm": float(cfg.hinf_gamma - hinf_dense),
        "robust_region_scale": float(design.robust_region_scale),
        "qp_grid_audit": qp_audit,
    }
    design_audit["all_dense_gates_passed"] = bool(
        design_audit["reference_in_S"]
        and rpi_failures == 0
        and spectral_radius(acl) < 1.0
        and hinf_dense < float(cfg.hinf_gamma)
        and qp_audit["passed"]
        and design_audit["steady_input_to_U_minus_KZ_minimum_margin"] >= -1e-8
        and design_audit["reference_to_S_minimum_margin"] >= -1e-8
    )

    scenario_stats: dict[str, dict[str, float]] = {}
    pooled_records: list[dict[str, object]] = []
    for index, scenario in enumerate(PAPER2016_SCENARIOS):
        stat, records, _ = run_episode(
            cfg, model, SafeController(cfg, model, design),
            ZeroResidualPolicy(), None,
            np.random.default_rng(args.seed + 31000 + index),
            training=False, global_step=0, paper_scenario=scenario,
        )
        scenario_stats[scenario] = stat
        pooled_records.extend(records)
        save_rollout_csv(
            args.output_dir / f"zero_residual_{scenario}.csv",
            records_to_arrays(records),
        )

    safety_fields = (
        "violation_rate", "robust_operating_region_violation_rate",
        "qp_infeasible_rate", "disturbance_bound_exceedance_rate",
    )
    rollout_safety_passed = all(
        abs(float(stat[key])) <= 1e-15
        for stat in scenario_stats.values() for key in safety_fields
    )
    mean_abs_economic_reward = float(np.mean([
        abs(float(row["economic_reward"])) for row in pooled_records
    ]))
    mean_state_loss = float(np.mean([
        float(row["state_recovery_loss"]) for row in pooled_records
    ]))
    mean_f200_loss = float(np.mean([
        float(row["f200_move_loss"]) for row in pooled_records
    ]))
    target = float(args.target_penalty_fraction)
    lambda_x = target * mean_abs_economic_reward / max(mean_state_loss, 1e-15)
    lambda_f = target * mean_abs_economic_reward / max(mean_f200_loss, 1e-15)
    calibration = {
        "target_fraction": target,
        "pooled_steps": len(pooled_records),
        "mean_absolute_economic_reward": mean_abs_economic_reward,
        "mean_state_recovery_loss": mean_state_loss,
        "mean_f200_move_loss": mean_f200_loss,
        "recommended_lambda_x": lambda_x,
        "recommended_lambda_F": lambda_f,
        "state_penalty_mean_at_recommended_weight": lambda_x * mean_state_loss,
        "f200_penalty_mean_at_recommended_weight": lambda_f * mean_f200_loss,
    }
    result = {
        "design_audit": design_audit,
        "zero_residual_scenario_stats": scenario_stats,
        "zero_residual_required_safety_passed": rollout_safety_passed,
        "reward_calibration": calibration,
        "ready_for_short_training": bool(
            design_audit["all_dense_gates_passed"] and rollout_safety_passed
        ),
    }
    with (args.output_dir / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_json_ready(result), stream, indent=2, ensure_ascii=False)
    print(json.dumps(_json_ready(result), indent=2, ensure_ascii=False))
    if not result["ready_for_short_training"]:
        raise RuntimeError("robust-reference audit failed; do not train SAC")


if __name__ == "__main__":
    main()
