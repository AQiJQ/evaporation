"""Independent disturbance certification and out-of-envelope stress tests."""
from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .control import SafeController, build_safety_design, point_in_convex_polygon
from .model import EvaporatorModel
from .theta_learning import OnlineThetaLearner


SCAN_FIELDS = (
    "phase", "rho_d", "disturbance_half_F1", "disturbance_half_X1",
    "disturbance_half_T1", "disturbance_half_T200", "hinf_feasible",
    "rpi_feasible", "x_tight_feasible", "u_tight_feasible",
    "invariant_feasible", "v_ref_feasible", "formal_certification_passed",
    "rpi_area", "x_minus_z_area", "invariant_area",
    "robust_region_scale", "max_residual_norm",
    "K_00", "K_01", "K_10", "K_11",
    "independent_K_design_passes", "failure_reason",
)


@dataclass
class DisturbanceScaleScanResult:
    rho_d_max_certified: float
    full_disturbance_formal_certified: bool
    rows: list[dict[str, Any]]
    selected_cfg: ExperimentConfig
    selected_model: EvaporatorModel
    selected_initial_design: Any
    selected_design: Any
    selected_learner: OnlineThetaLearner
    corner_summary: dict[str, Any]


def _polygon_area(vertices: np.ndarray, scale: np.ndarray) -> float:
    points = np.asarray(vertices, dtype=float) * np.asarray(scale, dtype=float)
    return 0.5 * abs(float(np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - points[:, 1] * np.roll(points[:, 0], -1)
    )))


def save_safety_design(path: Path, design: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        h=design.theta_h, p=design.theta_p,
        M=design.theta_m_matrix, m=design.theta_m_bound, K=design.k,
        W_vertices=design.w_vertices, W_data_hull=design.w_data_hull,
        M_angle=design.theta_m_angle, A=design.a, B=design.b,
        affine=design.affine, rpi_boundary=design.rpi_boundary,
        rpi_support_lower=design.rpi_support_lower,
        rpi_support_upper=design.rpi_support_upper,
        x_lower_tight=design.x_lower_tight,
        x_upper_tight=design.x_upper_tight,
        u_lower_tight=design.u_lower_tight,
        u_upper_tight=design.u_upper_tight,
        robust_state_lower=design.robust_state_lower,
        robust_state_upper=design.robust_state_upper,
        robust_input_lower=design.robust_input_lower,
        robust_input_upper=design.robust_input_upper,
        robust_region_scale=design.robust_region_scale,
        invariant_lower=design.invariant_lower,
        invariant_upper=design.invariant_upper,
        nominal_policy_gain=design.nominal_policy_gain,
        nominal_policy_offset=design.nominal_policy_offset,
    )


def disturbance_corner_feasibility(
    cfg: ExperimentConfig,
    model: EvaporatorModel,
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate required nonlinear steady inputs for all 16 full-box corners."""
    full = np.asarray(cfg.disturbance_full_half_range, dtype=float)
    lower = cfg.disturbance_nominal - full
    upper = cfg.disturbance_nominal + full
    rows: list[dict[str, Any]] = []
    for f1 in (lower[0], upper[0]):
        for x1 in (lower[1], upper[1]):
            for t1 in (lower[2], upper[2]):
                for t200 in (lower[3], upper[3]):
                    disturbance = np.array([f1, x1, t1, t200], dtype=float)
                    row: dict[str, Any] = {
                        "F1": f1, "X1": x1, "T1": t1, "T200": t200,
                        "required_P100": float("nan"),
                        "required_F200": float("nan"),
                        "P100_lower_ok": False, "P100_upper_ok": False,
                        "F200_lower_ok": False, "F200_upper_ok": False,
                        "steady_input_feasible": False,
                        "derivative_norm_at_required_input": float("nan"),
                        "error": "",
                    }
                    try:
                        required = model.steady_input(
                            cfg.safe_center_state, disturbance
                        )
                        derivative_norm = float(np.max(np.abs(model.derivative(
                            cfg.safe_center_state, required, disturbance
                        ))))
                        row.update({
                            "required_P100": float(required[0]),
                            "required_F200": float(required[1]),
                            "P100_lower_ok": bool(required[0] >= cfg.input_lower[0]),
                            "P100_upper_ok": bool(required[0] <= cfg.input_upper[0]),
                            "F200_lower_ok": bool(required[1] >= cfg.input_lower[1]),
                            "F200_upper_ok": bool(required[1] <= cfg.input_upper[1]),
                            "derivative_norm_at_required_input": derivative_norm,
                        })
                        row["steady_input_feasible"] = bool(
                            row["P100_lower_ok"] and row["P100_upper_ok"]
                            and row["F200_lower_ok"] and row["F200_upper_ok"]
                            and derivative_norm <= 1e-7
                        )
                    except Exception as exc:  # keep every failed corner visible
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    p100 = np.asarray([row["required_P100"] for row in rows], dtype=float)
    f200 = np.asarray([row["required_F200"] for row in rows], dtype=float)
    summary = {
        "full_disturbance_all_corners_steady_feasible": bool(all(
            row["steady_input_feasible"] for row in rows
        )),
        "steady_feasible_corner_count": int(sum(
            row["steady_input_feasible"] for row in rows
        )),
        "corner_count": len(rows),
        "max_required_P100": float(np.nanmax(p100)),
        "max_required_F200": float(np.nanmax(f200)),
        "min_required_P100": float(np.nanmin(p100)),
        "min_required_F200": float(np.nanmin(f200)),
    }
    return summary


def _attempt_alpha(
    base_cfg: ExperimentConfig,
    alpha: float,
    phase: str,
) -> tuple[dict[str, Any], tuple[Any, ...] | None]:
    full = np.asarray(base_cfg.disturbance_full_half_range, dtype=float)
    design_cfg = replace(
        base_cfg,
        seed=int(base_cfg.disturbance_scale_scan_seed),
        disturbance_half_range=float(alpha) * full,
        disturbance_scale_scan_enabled=False,
        robust_region_random_samples=int(
            base_cfg.disturbance_scale_random_samples
        ),
        theta_static_outer_iterations=int(
            base_cfg.disturbance_scale_static_outer_iterations
        ),
        theta_static_k_max_iterations=int(
            base_cfg.disturbance_scale_k_max_iterations
        ),
        theta_m_max_angle_degrees=float(
            base_cfg.disturbance_scale_m_angle_max_degrees
        ),
        theta_m_angle_step_degrees=float(
            base_cfg.disturbance_scale_m_angle_step_degrees
        ),
    )
    model = EvaporatorModel(design_cfg)
    initial_design = None
    independently_designed = None
    learner = None
    try:
        initial_design = build_safety_design(
            design_cfg, model, np.random.default_rng(design_cfg.seed)
        )
        # This optimizer is deliberately reconstructed at every alpha.  It is
        # seeded identically for fair comparison, but receives no K/M/design
        # from any neighbouring scale or cached scan result.
        design_learner = OnlineThetaLearner(
            design_cfg,
            model,
            initial_design,
            np.random.default_rng(design_cfg.seed + 16001),
        )
        independently_designed, _ = design_learner.optimize_static_safety_design(
            initial_design
        )

        # Static synthesis has already been completed exactly once for this
        # alpha.  Disable nested redesign while the robust-region scale is
        # expanded/bisected, so that each candidate certifies the same local
        # controller rather than silently changing K during region search.
        cfg = replace(design_cfg, theta_static_outer_iterations=0)
        model = EvaporatorModel(cfg)
        learner = OnlineThetaLearner(
            cfg,
            model,
            independently_designed,
            np.random.default_rng(cfg.seed + 17001),
        )
        design, _ = learner.build_certified_robust_operating_design(
            independently_designed
        )
        diagnostics = learner.robust_region_search_diagnostics
        final_diagnostic = next(
            row for row in reversed(diagnostics)
            if bool(row.get("feasible", False))
        )
        row = {
            "phase": phase, "rho_d": float(alpha),
            "disturbance_half_F1": float(alpha * full[0]),
            "disturbance_half_X1": float(alpha * full[1]),
            "disturbance_half_T1": float(alpha * full[2]),
            "disturbance_half_T200": float(alpha * full[3]),
            "hinf_feasible": True, "rpi_feasible": True,
            "x_tight_feasible": True, "u_tight_feasible": True,
            "invariant_feasible": True, "v_ref_feasible": True,
            "formal_certification_passed": True,
            "rpi_area": _polygon_area(design.rpi_boundary, cfg.state_scale),
            "x_minus_z_area": float(np.prod(
                (design.x_upper_tight - design.x_lower_tight) * cfg.state_scale
            )),
            "invariant_area": float(np.prod(
                (design.invariant_upper - design.invariant_lower) * cfg.state_scale
            )),
            "robust_region_scale": float(design.robust_region_scale),
            "max_residual_norm": float(final_diagnostic["max_residual_norm"]),
            "K_00": float(design.k[0, 0]), "K_01": float(design.k[0, 1]),
            "K_10": float(design.k[1, 0]), "K_11": float(design.k[1, 1]),
            "independent_K_design_passes": int(
                design_cfg.theta_static_outer_iterations
            ),
            "failure_reason": "",
        }
        return row, (cfg, model, initial_design, design, learner)
    except Exception as exc:
        attempted_k = (
            np.full((2, 2), np.nan, dtype=float)
            if independently_designed is None
            else np.asarray(independently_designed.k, dtype=float)
        )
        diagnostics = (
            [] if learner is None else learner.robust_region_search_diagnostics
        )
        row = {
            "phase": phase, "rho_d": float(alpha),
            "disturbance_half_F1": float(alpha * full[0]),
            "disturbance_half_X1": float(alpha * full[1]),
            "disturbance_half_T1": float(alpha * full[2]),
            "disturbance_half_T200": float(alpha * full[3]),
            "hinf_feasible": bool(any(d.get("hinf_feasible", False) for d in diagnostics)),
            "rpi_feasible": bool(any(d.get("rpi_feasible", False) for d in diagnostics)),
            "x_tight_feasible": bool(any(d.get("x_tightening_feasible", False) for d in diagnostics)),
            "u_tight_feasible": bool(any(d.get("u_tightening_feasible", False) for d in diagnostics)),
            "invariant_feasible": bool(any(d.get("invariant_feasible", False) for d in diagnostics)),
            "v_ref_feasible": bool(any(d.get("v_ref_feasible", False) for d in diagnostics)),
            "formal_certification_passed": False,
            "rpi_area": float("nan"), "x_minus_z_area": float("nan"),
            "invariant_area": float("nan"), "robust_region_scale": float("nan"),
            "K_00": float(attempted_k[0, 0]),
            "K_01": float(attempted_k[0, 1]),
            "K_10": float(attempted_k[1, 0]),
            "K_11": float(attempted_k[1, 1]),
            "independent_K_design_passes": int(
                design_cfg.theta_static_outer_iterations
            ),
            "max_residual_norm": float(max(
                (float(d.get("max_residual_norm", 0.0)) for d in diagnostics),
                default=float("nan"),
            )),
            "failure_reason": f"{type(exc).__name__}: {exc}".replace("\n", " | "),
        }
        return row, None


def _write_scan_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: float(row["rho_d"]))
    rho_d = np.asarray([row["rho_d"] for row in ordered], dtype=float)
    rpi = np.asarray([row["rpi_area"] for row in ordered], dtype=float)
    invariant = np.asarray([row["invariant_area"] for row in ordered], dtype=float)
    feasible = np.asarray([
        float(bool(row["formal_certification_passed"])) for row in ordered
    ])
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.2), sharex=True)
    axes[0].plot(rho_d, rpi, "o-", color="#2f6fb3")
    axes[0].set_ylabel("RPI area")
    axes[1].plot(rho_d, invariant, "o-", color="#3b8f5a")
    axes[1].set_ylabel("Invariant area")
    axes[2].step(rho_d, feasible, where="mid", color="#a84b3c")
    axes[2].scatter(rho_d, feasible, color="#a84b3c")
    axes[2].set_yticks([0, 1], ["infeasible", "certified"])
    axes[2].set_xlabel("Disturbance scaling factor ρ_d")
    for axis in axes:
        axis.grid(True, color="#dddddd", linewidth=0.6)
    fig.suptitle("Certified external-disturbance envelope versus ρ_d")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_disturbance_scale_scan(
    cfg: ExperimentConfig,
    output_dir: Path,
) -> DisturbanceScaleScanResult:
    """Independently rebuild and certify every disturbance scale."""
    output_dir.mkdir(parents=True, exist_ok=True)
    corner_model = EvaporatorModel(cfg)
    corner_summary = disturbance_corner_feasibility(
        cfg, corner_model, output_dir / "disturbance_corner_feasibility.csv"
    )
    rows: list[dict[str, Any]] = []
    feasible_designs: dict[float, tuple[Any, ...]] = {}
    grid = sorted(set(float(value) for value in cfg.disturbance_scale_scan_grid))
    if not grid or grid[0] < 0.0 or grid[-1] > 1.0:
        raise ValueError("disturbance_scale_scan_grid must lie in [0,1]")
    for alpha in grid:
        row, objects = _attempt_alpha(cfg, alpha, "grid")
        rows.append(row)
        if objects is not None:
            feasible_designs[alpha] = objects
        print(
            f"external-disturbance rho_d={alpha:.6g}: "
            f"certified={bool(row['formal_certification_passed'])}",
            flush=True,
        )
    if not feasible_designs:
        raise RuntimeError("No external-disturbance scale, including rho_d=0, was certified")

    lower = max(feasible_designs)
    upper_candidates = [
        float(row["rho_d"]) for row in rows
        if not bool(row["formal_certification_passed"])
        and float(row["rho_d"]) > lower
    ]
    if upper_candidates:
        upper = min(upper_candidates)
        for _ in range(int(cfg.disturbance_scale_bisection_iterations)):
            middle = 0.5 * (lower + upper)
            row, objects = _attempt_alpha(cfg, middle, "bisection")
            rows.append(row)
            print(
                f"external-disturbance rho_d={middle:.9g}: "
                f"certified={bool(row['formal_certification_passed'])}",
                flush=True,
            )
            if objects is None:
                upper = middle
            else:
                lower = middle
                feasible_designs[middle] = objects

    rho_d_max = max(feasible_designs)
    selected = feasible_designs[rho_d_max]
    full_row = min(rows, key=lambda row: abs(float(row["rho_d"]) - 1.0))
    full_certified = bool(
        abs(float(full_row["rho_d"]) - 1.0) <= 1e-12
        and full_row["formal_certification_passed"]
    )
    with (output_dir / "disturbance_scale_scan.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SCAN_FIELDS))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: float(row["rho_d"])))
    _write_scan_plot(output_dir / "disturbance_scale_scan.png", rows)
    save_safety_design(output_dir / "rho_d_max_certified_design.npz", selected[3])
    summary = {
        "experiment_type": "external_uncertainty_applicability_analysis",
        "disturbance_variables": ["F1", "X1", "T1", "T200"],
        "full_half_range": np.asarray(
            cfg.disturbance_full_half_range, dtype=float
        ).tolist(),
        "rho_d_max_certified": float(rho_d_max),
        "alpha_max_certified_legacy": float(rho_d_max),
        "related_to_paper_state_shock_scaling": False,
        "full_disturbance_formal_certified": full_certified,
        **corner_summary,
    }
    with (output_dir / "disturbance_scale_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return DisturbanceScaleScanResult(
        rho_d_max_certified=float(rho_d_max),
        full_disturbance_formal_certified=full_certified,
        rows=rows,
        selected_cfg=selected[0], selected_model=selected[1],
        selected_initial_design=selected[2], selected_design=selected[3],
        selected_learner=selected[4], corner_summary=corner_summary,
    )


def run_full_disturbance_stress_test(
    cfg: ExperimentConfig,
    model: EvaporatorModel,
    design: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate the certified controller at rho_d=1 without claiming a guarantee."""
    controller = SafeController(cfg, model, design)
    state = cfg.safe_center_state.copy()
    controller.reset(state)
    rng = np.random.default_rng(cfg.seed + 93001)
    disturbance = cfg.disturbance_nominal.copy()
    records: list[dict[str, Any]] = []
    for step in range(int(cfg.full_disturbance_stress_steps)):
        if step % int(cfg.disturbance_hold_steps) == 0:
            full = np.asarray(cfg.disturbance_full_half_range, dtype=float)
            disturbance = rng.uniform(
                cfg.disturbance_nominal - full,
                cfg.disturbance_nominal + full,
            )
        control, info = controller.act(
            state, np.zeros(2), action_is_normalized=True
        )
        next_state = model.step(state, control, disturbance)
        x_n = model.normalized_state(state)
        next_n = model.normalized_state(next_state)
        u_n = model.normalized_input(control)
        effective_w = next_n - (
            design.a @ x_n + design.b @ u_n + design.affine
        )
        error_next = next_n - np.asarray(info["z_next"], dtype=float)
        records.append({
            "step": step, "time_min": step * cfg.dt_min,
            "X2": state[0], "P2": state[1],
            "P100": control[0], "F200": control[1],
            "F1": disturbance[0], "X1": disturbance[1],
            "T1": disturbance[2], "T200": disturbance[3],
            "economic_cost": model.economic_cost(state, control, disturbance),
            "state_violation": bool(np.any(state < cfg.state_lower) or np.any(state > cfg.state_upper)),
            "input_violation": bool(np.any(control < cfg.input_lower) or np.any(control > cfg.input_upper)),
            "rpi_violation": not point_in_convex_polygon(error_next, design.rpi_boundary, tol=1e-8),
            "qp_infeasible": not bool(info["qp_feasible"]),
            "W_exceedance": not point_in_convex_polygon(effective_w, design.w_vertices, tol=1e-8),
        })
        state = next_state
    path = output_dir / "full_disturbance_stress_test.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6), sharex=True)
    time_values = np.asarray([row["time_min"] for row in records])
    for axis, field, label in zip(
        axes.ravel(), ("X2", "P2", "P100", "F200"),
        ("X2 [%]", "P2 [kPa]", "P100 [kPa]", "F200 [kg/min]"),
    ):
        axis.plot(time_values, [row[field] for row in records], linewidth=1.0)
        axis.set_ylabel(label)
        axis.grid(True, color="#dddddd", linewidth=0.6)
    axes[-1, 0].set_xlabel("Time [min]")
    axes[-1, 1].set_xlabel("Time [min]")
    fig.suptitle("Out-of-certified-envelope stress test (full ρ_d=1)")
    fig.tight_layout()
    fig.savefig(output_dir / "full_disturbance_stress_test.png", dpi=180)
    plt.close(fig)
    summary = {
        "experiment_label": "out-of-certified-envelope stress test",
        "formal_certification_passed": False,
        "state_violation_rate": float(np.mean([r["state_violation"] for r in records])),
        "input_violation_rate": float(np.mean([r["input_violation"] for r in records])),
        "rpi_violation_rate": float(np.mean([r["rpi_violation"] for r in records])),
        "qp_infeasible_rate": float(np.mean([r["qp_infeasible"] for r in records])),
        "W_exceedance_rate": float(np.mean([r["W_exceedance"] for r in records])),
        "economic_cost_mean": float(np.mean([r["economic_cost"] for r in records])),
    }
    return summary
