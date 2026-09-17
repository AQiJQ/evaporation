"""Zanon-Gros-Diehl (2016) evaporation benchmark and TuneMPC validation.

Formal results use a one-second sample, N=200 and 300 seconds.  ``--smoke-test``
uses N=20 and is always labelled non-formal.  TuneMPC remains an external,
read-only author-team reference implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from .config import ExperimentConfig
from .control import SafeController, build_safety_design, point_in_convex_polygon
from .model import EvaporatorModel
from .paper2016_adapter import (
    DEFAULT_TUNEMPC_PATH,
    build_tunempc_baselines,
    dependency_diagnostics,
    reference_economic_cost,
)


SCENARIOS = ("pressure_positive", "pressure_negative", "concentration_positive")
SHOCK_SECONDS = (0, 20, 40)
PAPER2016_DISTURBANCE_PROTOCOL = {
    "external_conditions": "nominal",
    "state_shock_scaling": 1.0,
    "uses_rho_d_scaling": False,
    "pressure_positive_shock_kpa": 1.0,
    "pressure_negative_shock_kpa": -1.0,
    "concentration_positive_shock_percentage_point": 1.0,
    "shock_times_seconds": list(SHOCK_SECONDS),
    "evaluation_disturbance_protocol": (
        "nominal_exogenous_conditions_plus_unscaled_state_shocks"
    ),
}
PAPER_REFERENCE_G = {
    "pressure_positive": {
        "Normal_Tracking_MPC": -3.2e-4,
        "Normal_Zero_Gradient": -1.5e-3,
        "Tuned_Diagonal": -1.1e-4,
        "Tuned_Tracking_MPC": -1.2e-7,
    },
    "pressure_negative": {
        "Normal_Tracking_MPC": -5.4e-6,
        "Normal_Zero_Gradient": -1.6e-1,
        "Tuned_Diagonal": -1.1e-4,
        "Tuned_Tracking_MPC": -1.2e-7,
    },
    "concentration_positive": {
        "Normal_Tracking_MPC": -2.6e-5,
        "Normal_Zero_Gradient": -4.2e-2,
        "Tuned_Diagonal": -1.6e-5,
        "Tuned_Tracking_MPC": -7.4e-9,
    },
}


def apply_state_shock(state: np.ndarray, scenario: str, second: int) -> np.ndarray:
    """Apply the paper's instantaneous plant-state shock, never a persistent bias."""
    result = np.asarray(state, dtype=float).copy()
    if second not in SHOCK_SECONDS:
        return result
    if scenario == "pressure_positive":
        result[1] += 1.0
    elif scenario == "pressure_negative":
        result[1] -= 1.0
    elif scenario == "concentration_positive":
        result[0] += 1.0
    else:
        raise ValueError(f"unknown paper2016 scenario: {scenario}")
    return result


def economic_metric_g(
    empc_cumulative_cost: float,
    method_cumulative_cost: float,
    sample_count: int,
    steady_stage_cost: float,
) -> float:
    """Paper metric G=(P_EMPC-P_method)/(T*P_s)."""
    denominator = float(sample_count) * float(steady_stage_cost)
    if abs(denominator) <= 1e-12:
        raise ValueError("T*P_s must be nonzero")
    return float((empc_cumulative_cost - method_cumulative_cost) / denominator)


def _checkpoint_training_rho_d(path: Path | None) -> float | None:
    """Read training metadata without making it part of paper evaluation."""
    if path is None or not path.exists():
        return None
    with np.load(path) as saved:
        for key in (
            "training_rho_d", "rho_d_max_certified", "alpha_max_certified"
        ):
            if key in saved.files:
                return float(saved[key])
    for candidate in (
        path.parent / "metrics.json",
        path.parent.parent / "metrics.json",
    ):
        if not candidate.exists():
            continue
        with candidate.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        value = metadata.get(
            "training_rho_d",
            metadata.get(
                "rho_d_max_certified", metadata.get("alpha_max_certified")
            ),
        )
        if value is not None:
            return float(value)
    return None


def _rollout(
    cfg: ExperimentConfig,
    model: EvaporatorModel,
    scenario: str,
    action: Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]],
    reset: Callable[[np.ndarray], None] | None = None,
) -> dict[str, Any]:
    state = cfg.paper2016_steady_state.copy()
    external_conditions = np.asarray(
        cfg.disturbance_nominal, dtype=float
    ).copy()
    if reset is not None:
        reset(state)
    rows: list[dict[str, Any]] = []
    solve_times: list[float] = []
    for second in range(cfg.paper2016_simulation_seconds):
        state = apply_state_shock(state, scenario, second)
        start = time.perf_counter()
        control, info = action(state.copy())
        solve_times.append(1000.0 * (time.perf_counter() - start))
        control = np.asarray(control, dtype=float).reshape(-1)
        cost = model.economic_cost(state, control, external_conditions)
        state_violation = bool(
            np.any(state < cfg.state_lower - 1e-9)
            or np.any(state > cfg.state_upper + 1e-9)
        )
        input_violation = bool(
            np.any(control < cfg.input_lower - 1e-9)
            or np.any(control > cfg.input_upper + 1e-9)
        )
        rows.append({
            "second": second,
            "X2": float(state[0]), "P2": float(state[1]),
            "P100": float(control[0]), "F200": float(control[1]),
            "economic_cost": float(cost),
            "state_violation": state_violation,
            "input_violation": input_violation,
            **info,
        })
        state = model.step(state, control, external_conditions)
    costs = np.asarray([row["economic_cost"] for row in rows])
    times = np.asarray(solve_times)
    return {
        "rows": rows,
        "cumulative_cost": float(np.sum(costs)),
        "cumulative_cost_series": np.cumsum(costs),
        "state_violation_rate": float(np.mean([row["state_violation"] for row in rows])),
        "input_violation_rate": float(np.mean([row["input_violation"] for row in rows])),
        "robust_region_violation_rate": float(np.mean([
            bool(row.get("robust_region_violation", False)) for row in rows
        ])),
        "rpi_violation_rate": float(np.mean([
            bool(row.get("rpi_violation", False)) for row in rows
        ])),
        "qp_infeasible_rate": float(np.mean([
            bool(row.get("qp_infeasible", False)) for row in rows
        ])),
        "mean_solve_time_ms": float(np.mean(times)),
        "p95_solve_time_ms": float(np.percentile(times, 95.0)),
    }


def _tunempc_action(controller: Any) -> Callable[[np.ndarray], tuple[np.ndarray, dict[str, Any]]]:
    def action(state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        value = controller.step(np.asarray(state, dtype=float).reshape(2, 1))
        if hasattr(value, "full"):
            value = value.full()
        return np.asarray(value, dtype=float).reshape(-1), {}
    return action


def _write_rollout_csv(path: Path, result: dict[str, Any]) -> None:
    fields = ["second", "X2", "P2", "P100", "F200", "economic_cost"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["rows"])


def _plot_scenario(
    output_dir: Path,
    scenario: str,
    results: dict[str, dict[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = {
        key: value for key, value in results.items()
        if key in {
            "Economic_MPC", "Tuned_Tracking_MPC", "Hinf_RPI", "Hinf_RPI_SAC"
        }
    }
    if not selected:
        selected = results
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex=True)
    fields = (("X2", "X2 [%]"), ("P2", "P2 [kPa]"),
              ("P100", "P100 [kPa]"), ("F200", "F200 [kg/min]"))
    for axis, (field, label) in zip(axes.ravel(), fields):
        for method, result in selected.items():
            axis.plot(
                [row["second"] for row in result["rows"]],
                [row[field] for row in result["rows"]],
                label=method.replace("_", " "), linewidth=1.2,
            )
        for shock in SHOCK_SECONDS:
            axis.axvline(shock, color="#aaaaaa", linewidth=0.6, linestyle="--")
        axis.set_ylabel(label)
        axis.grid(True, color="#dddddd", linewidth=0.5)
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"2016 evaporation benchmark: {scenario}")
    fig.tight_layout()
    prefix = {
        "pressure_positive": "pressure",
        "pressure_negative": "pressure_negative_validation",
        "concentration_positive": "concentration",
    }[scenario]
    fig.savefig(output_dir / f"paper2016_{prefix}_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for method, result in selected.items():
        axis.plot(
            np.arange(len(result["cumulative_cost_series"])),
            result["cumulative_cost_series"],
            label=method.replace("_", " "), linewidth=1.3,
        )
    axis.set_xlabel("Time [s]")
    axis.set_ylabel("Cumulative economic cost")
    axis.set_title(f"Cumulative cost: {scenario}")
    axis.grid(True, color="#dddddd", linewidth=0.5)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"paper2016_{prefix}_cumulative_cost.png", dpi=180)
    plt.close(fig)


def _plot_computation_times(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique: dict[str, tuple[float, float]] = {}
    for row in summary:
        unique.setdefault(
            str(row["method"]),
            (float(row["mean_solve_time_ms"]), float(row["p95_solve_time_ms"])),
        )
    labels = list(unique)
    x = np.arange(len(labels))
    width = 0.37
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.bar(x - width / 2, [unique[k][0] for k in labels], width, label="mean")
    axis.bar(x + width / 2, [unique[k][1] for k in labels], width, label="p95")
    axis.set_xticks(x, [label.replace("_", " ") for label in labels], rotation=15)
    axis.set_ylabel("Online solve time [ms]")
    axis.set_title("Per-step online computation time")
    axis.grid(True, axis="y", color="#dddddd", linewidth=0.5)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "paper2016_online_computation_time.png", dpi=180)
    plt.close(fig)


def _validate_reproduction(summary: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(row["scenario"], row["method"]): row for row in summary}
    comparisons: list[dict[str, Any]] = []
    ranking_checks: list[bool] = []
    for scenario, references in PAPER_REFERENCE_G.items():
        available: dict[str, float] = {}
        for method, paper_value in references.items():
            row = lookup.get((scenario, method))
            ours = None if row is None else float(row["G_vs_EMPC"])
            comparisons.append({
                "scenario": scenario,
                "method": method,
                "paper_reference": paper_value,
                "our_reproduction": ours,
                "absolute_error": None if ours is None else abs(ours - paper_value),
            })
            if ours is not None:
                available[method] = ours
        if "Tuned_Tracking_MPC" in available and "Normal_Tracking_MPC" in available:
            ranking_checks.append(
                abs(available["Tuned_Tracking_MPC"])
                <= abs(available["Normal_Tracking_MPC"])
            )
    return {
        "paper2016_reproduction_validated": bool(ranking_checks and all(ranking_checks)),
        "validation_rule": (
            "tuned tracking must be at least as close to EMPC in |G| as normal "
            "tracking for every available scenario; published values are metadata, "
            "not forced targets"
        ),
        "comparisons": comparisons,
    }


def _summary_rows(
    scenario: str,
    results: dict[str, dict[str, Any]],
    steady_cost: float,
    formal: bool,
) -> list[dict[str, Any]]:
    empc = results["Economic_MPC"]["cumulative_cost"]
    hinf = results.get("Hinf_RPI", {}).get("cumulative_cost")
    sac = results.get("Hinf_RPI_SAC", {}).get("cumulative_cost")
    sac_improvement = (
        float(100.0 * (hinf - sac) / max(abs(hinf), 1e-12))
        if hinf is not None and sac is not None else float("nan")
    )
    rows = []
    for method, result in results.items():
        cumulative = float(result["cumulative_cost"])
        rows.append({
            "scenario": scenario,
            "method": method,
            "cumulative_economic_cost": cumulative,
            "G_vs_EMPC": economic_metric_g(empc, cumulative, len(result["rows"]), steady_cost),
            "economic_difference_vs_EMPC_percent": float(
                100.0 * (cumulative - empc) / max(abs(empc), 1e-12)
            ),
            "sac_improvement_vs_Hinf_RPI_percent": (
                sac_improvement if method == "Hinf_RPI_SAC" else float("nan")
            ),
            "state_violation_rate": result["state_violation_rate"],
            "input_violation_rate": result["input_violation_rate"],
            "robust_region_violation_rate": result["robust_region_violation_rate"],
            "rpi_violation_rate": result["rpi_violation_rate"],
            "qp_infeasible_rate": result["qp_infeasible_rate"],
            "mean_solve_time_ms": result["mean_solve_time_ms"],
            "p95_solve_time_ms": result["p95_solve_time_ms"],
            "formal_safety_certification_passed": (
                bool(result.get("formal_safety_certification_passed", False))
                if method.startswith("Hinf") else "not_applicable"
            ),
            "benchmark_profile": "zanon2016",
            "controller_source": (
                "Proposed_fixed_design" if method.startswith("Hinf")
                else "TuneMPC_reference"
            ),
            "formal_result": bool(formal),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baselines-only", "formal"), default="baselines-only")
    parser.add_argument("--tunempc-path", type=Path, default=DEFAULT_TUNEMPC_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("evaporation_safe_sac/paper2016"))
    parser.add_argument("--scenario", choices=SCENARIOS, nargs="+", default=list(SCENARIOS))
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--paper-native-constraints", action="store_true")
    parser.add_argument("--convexifier-solver", default="cvxopt")
    parser.add_argument("--safety-design", type=Path)
    parser.add_argument("--sac-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizon = 20 if args.smoke_test else int(args.horizon)
    if not args.smoke_test and horizon != 200:
        raise ValueError("Formal paper2016 comparison requires N=200; use --smoke-test for N=20")
    formal_result = bool(not args.smoke_test and horizon == 200)
    cfg = ExperimentConfig(benchmark_profile="zanon2016")
    model = EvaporatorModel(cfg)
    diagnostics = dependency_diagnostics(args.tunempc_path)
    training_rho_d = _checkpoint_training_rho_d(args.safety_design)
    protocol = {
        "benchmark_profile": cfg.benchmark_profile,
        "dt_seconds": cfg.dt_min * 60.0,
        "prediction_horizon": horizon,
        "simulation_seconds": cfg.paper2016_simulation_seconds,
        "formal_result": formal_result,
        "comparison_uses_common_input_lower_bounds": not args.paper_native_constraints,
        "paper_native_constraint_validation": bool(args.paper_native_constraints),
        "controller_parameters_frozen_during_evaluation": True,
        "actor_gradient_updates_during_evaluation": 0,
        "theta_updates_during_evaluation": 0,
        "K_updates_during_evaluation": 0,
        "M_updates_during_evaluation": 0,
        "W_updates_during_evaluation": 0,
        **PAPER2016_DISTURBANCE_PROTOCOL,
        "training_rho_d": training_rho_d,
        "training_disturbance_protocol": (
            "certified_piecewise_constant_exogenous_uncertainty"
            if training_rho_d is not None else "not_applicable_or_not_provided"
        ),
        "tunempc": diagnostics,
    }
    with (args.output_dir / "paper2016_protocol.json").open("w", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, ensure_ascii=False)

    if not diagnostics["direct_api_available"]:
        blocked = {
            **protocol,
            "status": "not_reproduced_dependency_unavailable",
            "claim": "No TuneMPC baseline result was fabricated or reported as reproduced.",
            "paper_reference": PAPER_REFERENCE_G,
        }
        with (args.output_dir / "paper2016_dependency_diagnostics.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(blocked, stream, indent=2, ensure_ascii=False)
        print(json.dumps(blocked, indent=2, ensure_ascii=False))
        return

    preparation_start = time.perf_counter()
    controllers, tunempc_metadata = build_tunempc_baselines(
        args.tunempc_path,
        horizon=horizon,
        common_input_lower_bounds=not args.paper_native_constraints,
        convexifier_solver=args.convexifier_solver,
    )
    offline_preparation_seconds = time.perf_counter() - preparation_start
    proposed: dict[str, tuple[Callable, Callable | None]] = {}
    if args.mode == "formal":
        if args.safety_design is None or args.sac_checkpoint is None:
            raise ValueError("formal mode requires --safety-design and --sac-checkpoint")
        from .sac import SACAgent, SACConfig
        from .train import ACTION_DIM, OBS_DIM, observation

        with np.load(args.safety_design) as saved:
            design = build_safety_design(
                cfg, model, np.random.default_rng(16001),
                w_data_hull=saved["W_data_hull"],
                theta_m_angle=float(saved["M_angle"]),
                theta_h=saved["h"], theta_p=saved["p"], theta_k=saved["K"],
                robust_state_lower=saved["robust_state_lower"],
                robust_state_upper=saved["robust_state_upper"],
                robust_input_lower=saved["robust_input_lower"],
                robust_input_upper=saved["robust_input_upper"],
                robust_region_scale=float(saved["robust_region_scale"]),
            )
        safe_controller = SafeController(cfg, model, design)
        agent = SACAgent(OBS_DIM, ACTION_DIM, SACConfig(hidden_dim=cfg.hidden_dim), device=args.device)
        agent.load_checkpoint(args.sac_checkpoint, load_optimizers=False)
        agent.actor.eval()
        agent.actor.requires_grad_(False)
        agent.actor_ema.eval()
        previous_u = {"value": design.v_ref.copy()}

        def safe_reset(state: np.ndarray) -> None:
            safe_controller.reset(state)
            previous_u["value"] = design.v_ref.copy()

        def safe_action(state: np.ndarray, use_sac: bool) -> tuple[np.ndarray, dict[str, Any]]:
            obs = observation(model, safe_controller, state, previous_u["value"], np.zeros(2))
            residual = (
                agent.select_action(obs, deterministic=True)
                if use_sac else np.zeros(2, dtype=float)
            )
            control, info = safe_controller.act(state, residual, action_is_normalized=True)
            previous_u["value"] = np.asarray(info["nominal"], dtype=float)
            x_n = model.normalized_state(state)
            robust_violation = bool(
                np.any(x_n < design.robust_state_lower - 1e-9)
                or np.any(x_n > design.robust_state_upper + 1e-9)
            )
            rpi_violation = not point_in_convex_polygon(
                x_n - np.asarray(info["z"], dtype=float), design.rpi_boundary, tol=1e-8
            )
            return control, {
                "robust_region_violation": robust_violation,
                "rpi_violation": rpi_violation,
                "qp_infeasible": not bool(info["qp_feasible"]),
            }

        proposed = {
            "Hinf_RPI": (lambda state: safe_action(state, False), safe_reset),
            "Hinf_RPI_SAC": (lambda state: safe_action(state, True), safe_reset),
        }

    summary: list[dict[str, Any]] = []
    steady_cost = reference_economic_cost(
        cfg.paper2016_steady_state, cfg.paper2016_steady_input
    )
    for scenario in args.scenario:
        results: dict[str, dict[str, Any]] = {}
        for method, controller in controllers.items():
            results[method] = _rollout(
                cfg,
                model,
                scenario,
                _tunempc_action(controller),
                lambda state, controller=controller: controller.reset(),
            )
        for method, (action, reset) in proposed.items():
            results[method] = _rollout(cfg, model, scenario, action, reset)
        for method, result in results.items():
            _write_rollout_csv(
                args.output_dir / f"paper2016_{scenario}_{method}.csv", result
            )
        summary.extend(_summary_rows(scenario, results, steady_cost, formal_result))
        _plot_scenario(args.output_dir, scenario, results)

    with (args.output_dir / "paper2016_comparison_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    validation = _validate_reproduction(summary)
    metadata = {
        **protocol,
        "status": "completed",
        "offline_preparation_seconds": offline_preparation_seconds,
        "tunempc_metadata": tunempc_metadata,
        **validation,
    }
    with (args.output_dir / "paper2016_reproduction_validation.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    _plot_computation_times(args.output_dir, summary)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
