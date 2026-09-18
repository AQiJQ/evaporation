"""Diagnose executable SAC residual authority for a fixed safety design.

This scan changes only the declared residual half range.  The saved K/W/Z,
tightened sets and controlled-invariant set are loaded from a formal training
artifact and remain unchanged for every multiplier.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig
from .control import (
    SafetyDesign,
    SafeController,
    _rpi_support,
    estimate_hinf_norm,
    finite_horizon_nominal_policy,
    project_qp_2d,
)
from .model import EvaporatorModel
from .paper2016_compare import SCENARIOS
from .train import run_episode


BASE_RESIDUAL_HALF_RANGE = np.array([0.12, 0.10], dtype=float)
MULTIPLIERS = (1.0, 1.5, 2.0, 3.0, 4.0)


class _ZeroResidualPolicy:
    def select_action(self, observation, deterministic: bool = True) -> np.ndarray:
        del observation, deterministic
        return np.zeros(2, dtype=np.float32)


class _TracingSafeController(SafeController):
    """Retain the exact runtime QP diagnostics without changing the controller."""

    def __init__(self, cfg, model, design):
        super().__init__(cfg, model, design)
        self.trace: list[dict[str, Any]] = []

    def act(self, state, residual, *, action_is_normalized=False):
        control, info = super().act(
            state, residual, action_is_normalized=action_is_normalized
        )
        copied: dict[str, Any] = {"state": np.asarray(state, dtype=float).copy()}
        for key, value in info.items():
            copied[key] = value.copy() if isinstance(value, np.ndarray) else value
        self.trace.append(copied)
        return control, info


def _polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    return 0.5 * abs(float(np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - points[:, 1] * np.roll(points[:, 0], -1)
    )))


def load_fixed_safety_design(
    cfg: ExperimentConfig,
    model: EvaporatorModel,
    path: Path,
) -> SafetyDesign:
    """Load the final saved geometry without rerunning any design search."""
    with np.load(path) as saved:
        a = np.asarray(saved["A"], dtype=float)
        b = np.asarray(saved["B"], dtype=float)
        affine = np.asarray(saved["affine"], dtype=float)
        k = np.asarray(saved["K"], dtype=float)
        w_bound = np.asarray(saved["W"], dtype=float)
        w_vertices = np.asarray(saved["disturbance_vertices"], dtype=float)
        w_data_hull = np.asarray(saved["disturbance_data_hull"], dtype=float)
        theta_m_matrix = np.asarray(saved["M"], dtype=float)
        theta_m_bound = np.asarray(saved["m"], dtype=float)
        rpi_boundary = np.asarray(saved["rpi_boundary"], dtype=float)
        rpi_support = np.asarray(saved["rpi_support"], dtype=float)
        rpi_support_lower = np.asarray(saved["rpi_support_lower"], dtype=float)
        rpi_support_upper = np.asarray(saved["rpi_support_upper"], dtype=float)
        robust_state_lower = np.asarray(saved["robust_state_lower"], dtype=float)
        robust_state_upper = np.asarray(saved["robust_state_upper"], dtype=float)
        robust_input_lower = np.asarray(saved["robust_input_lower"], dtype=float)
        robust_input_upper = np.asarray(saved["robust_input_upper"], dtype=float)
        invariant_lower = np.asarray(saved["invariant_lower"], dtype=float)
        invariant_upper = np.asarray(saved["invariant_upper"], dtype=float)
        invariant_scale = float(saved["invariant_scale"])
        robust_region_scale = float(saved["robust_region_scale"])
        minimum_residual_authority = float(saved["minimum_residual_authority"])
        theta_h = np.asarray(saved["h"], dtype=float)
        theta_p = np.asarray(saved["p"], dtype=float)
        gain_state_weight_scale = float(saved["gain_state_weight_scale"])
        gain_input_weight_scale = float(saved["gain_input_weight_scale"])
        box_width = np.asarray(
            saved["box_reference_rpi_width_physical"], dtype=float
        )
        box_area = float(saved["box_reference_rpi_area_physical"])

    acl = a + b @ k
    input_support_upper = _rpi_support(
        acl, w_vertices, k, max_terms=cfg.rpi_series_max_terms
    )
    input_support_lower = _rpi_support(
        acl, w_vertices, -k, max_terms=cfg.rpi_series_max_terms
    )
    input_support = np.maximum(input_support_lower, input_support_upper)
    x_lower_tight = robust_state_lower + rpi_support_lower
    x_upper_tight = robust_state_upper - rpi_support_upper
    u_lower_tight = robust_input_lower + input_support_lower
    u_upper_tight = robust_input_upper - input_support_upper
    z_ref = model.normalized_state(cfg.safe_center_state)
    safe_u = model.linearized_steady_input(cfg.safe_center_state, a, b, affine)
    v_ref = model.normalized_input(safe_u)
    nominal_gain, _ = finite_horizon_nominal_policy(
        cfg,
        a,
        b,
        np.zeros_like(affine),
        np.zeros_like(theta_h),
        np.zeros_like(theta_p),
    )
    nominal_offset = v_ref - nominal_gain @ z_ref
    first_normal = theta_m_matrix[0]
    theta_m_angle = float(np.arctan2(first_normal[1], first_normal[0]))
    physical_rpi = rpi_boundary * cfg.state_scale
    rpi_width = (rpi_support_lower + rpi_support_upper) * cfg.state_scale
    x_minus_z_width = (x_upper_tight - x_lower_tight) * cfg.state_scale

    return SafetyDesign(
        a=a,
        b=b,
        affine=affine,
        k=k,
        gamma_design=float(cfg.hinf_gamma),
        gamma_sampled=estimate_hinf_norm(
            a, b, k, cfg.hinf_q, cfg.hinf_r
        ),
        w_bound=w_bound,
        w_vertices=w_vertices,
        w_data_hull=w_data_hull,
        theta_m_matrix=theta_m_matrix,
        theta_m_bound=theta_m_bound,
        theta_m_angle=theta_m_angle,
        rpi_boundary=rpi_boundary,
        rpi_support=rpi_support,
        rpi_support_lower=rpi_support_lower,
        rpi_support_upper=rpi_support_upper,
        input_rpi_support=input_support,
        input_rpi_support_lower=input_support_lower,
        input_rpi_support_upper=input_support_upper,
        robust_state_lower=robust_state_lower,
        robust_state_upper=robust_state_upper,
        robust_input_lower=robust_input_lower,
        robust_input_upper=robust_input_upper,
        robust_region_scale=robust_region_scale,
        x_lower_tight=x_lower_tight,
        x_upper_tight=x_upper_tight,
        u_lower_tight=u_lower_tight,
        u_upper_tight=u_upper_tight,
        z_ref=z_ref,
        v_ref=v_ref,
        invariant_lower=invariant_lower,
        invariant_upper=invariant_upper,
        invariant_scale=invariant_scale,
        gain_state_weight_scale=gain_state_weight_scale,
        gain_input_weight_scale=gain_input_weight_scale,
        gain_candidates_feasible=1,
        reference_rpi_width_physical=rpi_width,
        reference_rpi_area_physical=_polygon_area(physical_rpi),
        reference_x_minus_z_width_physical=x_minus_z_width,
        reference_x_minus_z_area_physical=float(np.prod(x_minus_z_width)),
        box_reference_rpi_width_physical=box_width,
        box_reference_rpi_area_physical=box_area,
        theta_h=theta_h,
        theta_p=theta_p,
        nominal_policy_gain=nominal_gain,
        nominal_policy_offset=nominal_offset,
        minimum_residual_authority=minimum_residual_authority,
    )


def invariant_probe_points(design: SafetyDesign) -> list[tuple[str, np.ndarray]]:
    lower = design.invariant_lower
    upper = design.invariant_upper
    center = 0.5 * (lower + upper)
    vertices = [
        np.array([lower[0], lower[1]]),
        np.array([lower[0], upper[1]]),
        np.array([upper[0], upper[1]]),
        np.array([upper[0], lower[1]]),
    ]
    points: list[tuple[str, np.ndarray]] = [("invariant_center", center)]
    points.extend((f"invariant_vertex_{i + 1}", value) for i, value in enumerate(vertices))
    points.extend(
        (f"invariant_edge_midpoint_{i + 1}", 0.5 * (vertices[i] + vertices[(i + 1) % 4]))
        for i in range(4)
    )
    return points


def _probe_geometry(
    cfg: ExperimentConfig,
    model: EvaporatorModel,
    design: SafetyDesign,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for label, z in invariant_probe_points(design):
        controller = _TracingSafeController(cfg, model, design)
        state = model.physical_state(z)
        controller.reset(state)
        controller.act(state, np.zeros(2), action_is_normalized=True)
        item = controller.trace[0]
        item["sample_source"] = label
        diagnostics.append(item)
    return diagnostics


def _paper_baselines(
    cfg: ExperimentConfig,
    model: EvaporatorModel,
    design: SafetyDesign,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    traces: list[dict[str, Any]] = []
    scenario_stats: dict[str, dict[str, float]] = {}
    policy = _ZeroResidualPolicy()
    for index, scenario in enumerate(SCENARIOS):
        controller = _TracingSafeController(cfg, model, design)
        stats, _, _ = run_episode(
            cfg,
            model,
            controller,
            policy,
            None,
            np.random.default_rng(6100 + index),
            training=False,
            global_step=0,
            paper_scenario=scenario,
        )
        scenario_stats[scenario] = {
            key: float(stats[key])
            for key in (
                "economic_cost_mean",
                "qp_infeasible_rate",
                "violation_rate",
                "robust_operating_region_violation_rate",
                "rpi_violation_rate",
            )
        }
        for second, item in enumerate(controller.trace):
            item["sample_source"] = f"trajectory:{scenario}:{second}"
            traces.append(item)
    return traces, scenario_stats


def _zero_residual_qp_feasible(info: dict[str, Any]) -> bool:
    base = np.asarray(info["base"], dtype=float)
    projected, feasible = project_qp_2d(
        base, np.asarray(info["a_q"]), np.asarray(info["b_q"])
    )
    return bool(
        info["base_qp_feasible"]
        and feasible
        and np.linalg.norm(projected - base) <= 1e-8
    )


def _qp_authority_metrics(
    diagnostics: list[dict[str, Any]],
    residual_half_range: np.ndarray,
    input_scale: np.ndarray,
) -> dict[str, float]:
    feasible_scales: list[float] = []
    normalized_authority: list[np.ndarray] = []
    zero_checks: list[bool] = []
    corner_checks: list[bool] = []
    all_corner_checks: list[bool] = []
    displacements: list[float] = []
    for info in diagnostics:
        base = np.asarray(info["base"], dtype=float)
        aq = np.asarray(info["a_q"], dtype=float)
        bq = np.asarray(info["b_q"], dtype=float)
        zero_exact = _zero_residual_qp_feasible(info)
        rho = float(info["residual_feasible_scale"]) if zero_exact else 0.0
        half_width = rho * residual_half_range
        feasible_scales.append(rho)
        normalized_authority.append(half_width)
        zero_checks.append(zero_exact)
        state_corners: list[bool] = []
        for signs in product((-1.0, 1.0), repeat=2):
            corner = base + half_width * np.asarray(signs, dtype=float)
            projected, feasible = project_qp_2d(corner, aq, bq)
            corner_exact = bool(
                zero_exact
                and feasible
                and np.linalg.norm(projected - corner) <= 1e-8
            )
            corner_checks.append(corner_exact)
            state_corners.append(corner_exact)
        all_corner_checks.append(all(state_corners))
        displacements.append(float(info["theta_projection_gap"]))

    rho = np.asarray(feasible_scales, dtype=float)
    authority_n = np.asarray(normalized_authority, dtype=float)
    authority_p = authority_n * np.asarray(input_scale, dtype=float)
    displacement = np.asarray(displacements, dtype=float)
    return {
        "minimum_feasible_scale": float(np.min(rho)),
        "mean_feasible_scale": float(np.mean(rho)),
        "maximum_feasible_scale": float(np.max(rho)),
        "minimum_absolute_P100_authority_normalized": float(np.min(authority_n[:, 0])),
        "mean_absolute_P100_authority_normalized": float(np.mean(authority_n[:, 0])),
        "maximum_absolute_P100_authority_normalized": float(np.max(authority_n[:, 0])),
        "minimum_absolute_F200_authority_normalized": float(np.min(authority_n[:, 1])),
        "mean_absolute_F200_authority_normalized": float(np.mean(authority_n[:, 1])),
        "maximum_absolute_F200_authority_normalized": float(np.max(authority_n[:, 1])),
        "minimum_absolute_P100_authority_kPa": float(np.min(authority_p[:, 0])),
        "mean_absolute_P100_authority_kPa": float(np.mean(authority_p[:, 0])),
        "maximum_absolute_P100_authority_kPa": float(np.max(authority_p[:, 0])),
        "minimum_absolute_F200_authority_kg_min": float(np.min(authority_p[:, 1])),
        "mean_absolute_F200_authority_kg_min": float(np.mean(authority_p[:, 1])),
        "maximum_absolute_F200_authority_kg_min": float(np.max(authority_p[:, 1])),
        "zero_residual_qp_feasible_rate": float(np.mean(zero_checks)),
        "residual_corner_qp_feasible_rate": float(np.mean(corner_checks)),
        "states_all_residual_corners_feasible_rate": float(np.mean(all_corner_checks)),
        "base_projection_displacement_mean": float(np.mean(displacement)),
        "base_projection_displacement_max": float(np.max(displacement)),
        "base_projection_displacement_physical_mean": float(
            np.mean(displacement) * float(input_scale[0])
        ),
        "base_projection_displacement_physical_max": float(
            np.max(displacement) * float(input_scale[0])
        ),
    }


def scan_residual_authority(
    cfg: ExperimentConfig,
    model: EvaporatorModel,
    design: SafetyDesign,
    multipliers: tuple[float, ...] = MULTIPLIERS,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    rows: list[dict[str, float]] = []
    details: dict[str, Any] = {}
    for multiplier in multipliers:
        declared = BASE_RESIDUAL_HALF_RANGE * float(multiplier)
        scan_cfg = copy.deepcopy(cfg)
        scan_cfg.residual_action_scale = declared.copy()
        geometry = _probe_geometry(scan_cfg, model, design)
        trajectories, scenario_stats = _paper_baselines(scan_cfg, model, design)
        all_diagnostics = geometry + trajectories
        authority = _qp_authority_metrics(
            all_diagnostics, declared, scan_cfg.input_scale
        )
        trajectory_authority = _qp_authority_metrics(
            trajectories, declared, scan_cfg.input_scale
        )
        geometry_authority = _qp_authority_metrics(
            geometry, declared, scan_cfg.input_scale
        )
        scenario_values = list(scenario_stats.values())
        row: dict[str, float] = {
            "residual_range_multiplier": float(multiplier),
            "declared_residual_P100_normalized": float(declared[0]),
            "declared_residual_F200_normalized": float(declared[1]),
            "declared_residual_P100_physical": float(declared[0] * scan_cfg.input_scale[0]),
            "declared_residual_F200_physical": float(declared[1] * scan_cfg.input_scale[1]),
            **authority,
            "qp_infeasible_rate": float(np.mean([
                item["qp_infeasible_rate"] for item in scenario_values
            ])),
            "physical_constraint_violation_rate": float(np.mean([
                item["violation_rate"] for item in scenario_values
            ])),
            "robust_operating_region_violation_rate": float(np.mean([
                item["robust_operating_region_violation_rate"]
                for item in scenario_values
            ])),
            "zero_residual_baseline_economic_cost": float(np.mean([
                item["economic_cost_mean"] for item in scenario_values
            ])),
            "rpi_violation_rate": float(np.mean([
                item["rpi_violation_rate"] for item in scenario_values
            ])),
            "trajectory_base_projection_displacement_mean": float(
                trajectory_authority["base_projection_displacement_mean"]
            ),
            "trajectory_base_projection_displacement_max": float(
                trajectory_authority["base_projection_displacement_max"]
            ),
            "geometry_base_projection_displacement_mean": float(
                geometry_authority["base_projection_displacement_mean"]
            ),
            "geometry_base_projection_displacement_max": float(
                geometry_authority["base_projection_displacement_max"]
            ),
        }
        rows.append(row)
        failed_samples = [
            str(item["sample_source"])
            for item in all_diagnostics
            if not _zero_residual_qp_feasible(item)
        ]
        details[f"{multiplier:g}x"] = {
            "sample_count": len(all_diagnostics),
            "geometry_probe_count": len(geometry),
            "trajectory_probe_count": len(trajectories),
            "zero_residual_qp_infeasible_samples": failed_samples,
            "geometry_only_authority": geometry_authority,
            "trajectory_only_authority": trajectory_authority,
            "scenario_metrics": scenario_stats,
            "aggregate": row,
        }
    return rows, details


def _design_digest(design: SafetyDesign) -> str:
    digest = hashlib.sha256()
    for value in (
        design.k,
        design.theta_m_matrix,
        design.w_vertices,
        design.rpi_boundary,
        design.invariant_lower,
        design.invariant_upper,
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def save_results(
    output_dir: Path,
    rows: list[dict[str, float]],
    details: dict[str, Any],
    design_path: Path,
    design: SafetyDesign,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "residual_authority_scan.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "purpose": "fixed-geometry residual declaration authority scan",
        "safety_design_path": str(design_path.resolve()),
        "fixed_design_sha256": _design_digest(design),
        "paper2016_protocol": {
            "scenarios": list(SCENARIOS),
            "simulation_seconds_per_scenario": 300,
            "external_conditions": "nominal",
            "state_shock_scaling": 1.0,
            "uses_rho_d_scaling": False,
        },
        "authority_population": (
            "invariant center, four vertices, four edge midpoints, and all "
            "900 zero-residual Paper2016 baseline trajectory states"
        ),
        "fixed_geometry": {
            "K": design.k.tolist(),
            "M": design.theta_m_matrix.tolist(),
            "W_vertices": design.w_vertices.tolist(),
            "rpi_boundary_vertex_count": int(len(design.rpi_boundary)),
            "invariant_lower": design.invariant_lower.tolist(),
            "invariant_upper": design.invariant_upper.tolist(),
            "robust_region_scale": float(design.robust_region_scale),
        },
        "scan": details,
    }
    with (output_dir / "residual_authority_scan.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)

    x = np.array([row["residual_range_multiplier"] for row in rows])
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True)
    for axis, prefix, unit in (
        (axes[0], "P100", "kPa"),
        (axes[1], "F200", "kg/min"),
    ):
        minimum = np.array([
            row[f"minimum_absolute_{prefix}_authority_{'kPa' if prefix == 'P100' else 'kg_min'}"]
            for row in rows
        ])
        mean = np.array([
            row[f"mean_absolute_{prefix}_authority_{'kPa' if prefix == 'P100' else 'kg_min'}"]
            for row in rows
        ])
        axis.plot(x, minimum, "o-", label="minimum actual authority")
        axis.plot(x, mean, "s-", label="mean actual authority")
        axis.set_ylabel(f"{prefix} authority [{unit}]")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[2].plot(
        x,
        [row["zero_residual_baseline_economic_cost"] for row in rows],
        "o-",
        color="tab:red",
    )
    axes[2].set_xlabel("Residual range multiplier")
    axes[2].set_ylabel("Zero-residual baseline economic cost")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("Fixed-geometry residual authority scan")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_authority_scan.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    package_dir = Path(__file__).resolve().parent
    formal_seed = (
        package_dir
        / "evaporation_safe_sac"
        / "outputs_paper2016_main_500x300"
        / "seed_42"
    )
    parser = argparse.ArgumentParser(
        description="Scan executable residual authority with fixed safety geometry."
    )
    parser.add_argument(
        "--safety-design",
        type=Path,
        default=formal_seed / "safety_design.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=package_dir / "evaporation_safe_sac" / "residual_authority_scan_e6112b1",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = ExperimentConfig(benchmark_profile="zanon2016")
    model = EvaporatorModel(cfg)
    design = load_fixed_safety_design(cfg, model, args.safety_design)
    rows, details = scan_residual_authority(cfg, model, design)
    save_results(args.output_dir, rows, details, args.safety_design, design)
    print(f"Fixed safety design: {args.safety_design.resolve()}")
    print(f"Results: {args.output_dir.resolve()}")
    for row in rows:
        print(
            f"{row['residual_range_multiplier']:>3g}x  "
            f"P100 min/mean={row['minimum_absolute_P100_authority_kPa']:.6g}/"
            f"{row['mean_absolute_P100_authority_kPa']:.6g} kPa  "
            f"F200 min/mean={row['minimum_absolute_F200_authority_kg_min']:.6g}/"
            f"{row['mean_absolute_F200_authority_kg_min']:.6g} kg/min  "
            f"QP infeasible={row['qp_infeasible_rate']:.6g}  "
            f"cost={row['zero_residual_baseline_economic_cost']:.6g}"
        )


if __name__ == "__main__":
    main()
