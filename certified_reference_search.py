"""Offline search for the lowest-cost fully certified steady reference.

Every accepted candidate rebuilds the proposed H-infinity/RPI/QP geometry.
This module never constructs or trains a SAC agent.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from itertools import product
from pathlib import Path
import time

import numpy as np

from .config import ExperimentConfig
from .control import (
    build_safety_design,
    estimate_hinf_norm,
    spectral_radius,
    synthesize_hinf,
)
from .model import EvaporatorModel
from .theta_learning import OnlineThetaLearner


FIELDS = (
    "stage", "X2", "P2", "steady_P100", "steady_F200",
    "steady_cost", "cost_increase", "cost_increase_percent",
    "derivative_inf_norm", "physical_state_feasible",
    "physical_input_feasible", "hinf_feasible", "rpi_feasible",
    "x_tightening_feasible", "u_tightening_feasible",
    "reference_in_tightened_state_set", "invariant_feasible",
    "reference_in_S", "s_plus_z_contained", "input_tube_contained",
    "disturbance_bound_audit_passed", "verification_qp_feasible",
    "residual_authority_feasible", "fully_certified",
    "minimum_residual_authority", "spectral_radius",
    "hinf_sampled_norm", "hinf_gamma", "rpi_area",
    "rpi_width_X2", "rpi_width_P2", "S_X2_lower", "S_P2_lower",
    "S_X2_upper", "S_P2_upper", "X_minus_Z_X2_lower",
    "X_minus_Z_P2_lower", "X_minus_Z_X2_upper",
    "X_minus_Z_P2_upper", "U_minus_KZ_P100_lower",
    "U_minus_KZ_F200_lower", "U_minus_KZ_P100_upper",
    "U_minus_KZ_F200_upper", "robust_region_scale",
    "verification_qp_probe_count", "gain_seed", "K_00", "K_01",
    "K_10", "K_11", "elapsed_seconds", "failure_reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Coarse-to-local search for the economically best fully certified "
            "Paper2016 proposed steady reference"
        )
    )
    parser.add_argument("--x2-min", type=float, default=25.0)
    parser.add_argument("--x2-max", type=float, default=30.0)
    parser.add_argument("--p2-min", type=float, default=49.5)
    parser.add_argument("--p2-max", type=float, default=55.0)
    parser.add_argument("--coarse-x2-points", type=int, default=7)
    parser.add_argument("--coarse-p2-points", type=int, default=7)
    parser.add_argument("--refine-points", type=int, default=5)
    parser.add_argument("--refine-levels", type=int, default=2)
    parser.add_argument("--top-count", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "evaporation_safe_sac/outputs_certified_economic_reference_search"
        ),
    )
    return parser.parse_args()


def _polygon_area(boundary: np.ndarray, scale: np.ndarray) -> float:
    physical = np.asarray(boundary, dtype=float) * np.asarray(scale, dtype=float)
    return 0.5 * abs(float(np.sum(
        physical[:, 0] * np.roll(physical[:, 1], -1)
        - physical[:, 1] * np.roll(physical[:, 0], -1)
    )))


def _candidate_config(state: np.ndarray, seed: int) -> ExperimentConfig:
    cfg = ExperimentConfig(
        benchmark_profile="zanon2016",
        experiment_mode="proposed",
        seed=int(seed),
    )
    cfg.safe_center_state = np.asarray(state, dtype=float).copy()
    cfg.linearization_state = np.asarray(state, dtype=float).copy()
    model = EvaporatorModel(cfg)
    cfg.linearization_input = model.steady_input(
        cfg.linearization_state, cfg.disturbance_nominal
    )
    return cfg


def _empty_row(stage: str, state: np.ndarray) -> dict[str, object]:
    row: dict[str, object] = {key: float("nan") for key in FIELDS}
    row.update({
        "stage": stage,
        "X2": float(state[0]),
        "P2": float(state[1]),
        "physical_state_feasible": False,
        "physical_input_feasible": False,
        "hinf_feasible": False,
        "rpi_feasible": False,
        "x_tightening_feasible": False,
        "u_tightening_feasible": False,
        "reference_in_tightened_state_set": False,
        "invariant_feasible": False,
        "reference_in_S": False,
        "s_plus_z_contained": False,
        "input_tube_contained": False,
        "disturbance_bound_audit_passed": False,
        "verification_qp_feasible": False,
        "residual_authority_feasible": False,
        "fully_certified": False,
        "verification_qp_probe_count": 0,
        "gain_seed": "none",
        "failure_reason": "not_evaluated",
    })
    return row


def _write_design(path: Path, design) -> None:
    np.savez(
        path,
        A=design.a,
        B=design.b,
        affine=design.affine,
        K=design.k,
        W_vertices=design.w_vertices,
        W_data_hull=design.w_data_hull,
        rpi_boundary=design.rpi_boundary,
        rpi_support_lower=design.rpi_support_lower,
        rpi_support_upper=design.rpi_support_upper,
        input_rpi_support_lower=design.input_rpi_support_lower,
        input_rpi_support_upper=design.input_rpi_support_upper,
        x_lower_tight=design.x_lower_tight,
        x_upper_tight=design.x_upper_tight,
        u_lower_tight=design.u_lower_tight,
        u_upper_tight=design.u_upper_tight,
        invariant_lower=design.invariant_lower,
        invariant_upper=design.invariant_upper,
        robust_state_lower=design.robust_state_lower,
        robust_state_upper=design.robust_state_upper,
        robust_input_lower=design.robust_input_lower,
        robust_input_upper=design.robust_input_upper,
        z_ref=design.z_ref,
        v_ref=design.v_ref,
        gamma_design=np.asarray(design.gamma_design),
        gamma_sampled=np.asarray(design.gamma_sampled),
        minimum_residual_authority=np.asarray(
            design.minimum_residual_authority
        ),
    )


def evaluate_candidate(
    state: np.ndarray,
    stage: str,
    seed: int,
    nominal_cost: float,
    design_dir: Path,
) -> dict[str, object]:
    """Rebuild and certify one candidate; never reuse another reference's K/W."""
    started = time.time()
    state = np.asarray(state, dtype=float)
    row = _empty_row(stage, state)
    cfg = _candidate_config(state, seed)
    model = EvaporatorModel(cfg)
    steady_input = np.asarray(cfg.linearization_input, dtype=float)
    derivative = model.derivative(state, steady_input, cfg.disturbance_nominal)
    derivative_norm = float(np.max(np.abs(derivative)))
    steady_cost = float(model.economic_cost(
        state, steady_input, cfg.disturbance_nominal
    ))
    row.update({
        "steady_P100": float(steady_input[0]),
        "steady_F200": float(steady_input[1]),
        "steady_cost": steady_cost,
        "cost_increase": steady_cost - nominal_cost,
        "cost_increase_percent": (
            100.0 * (steady_cost - nominal_cost) / abs(nominal_cost)
        ),
        "derivative_inf_norm": derivative_norm,
        "physical_state_feasible": bool(
            np.all(state >= cfg.state_lower - 1e-10)
            and np.all(state <= cfg.state_upper + 1e-10)
        ),
        "physical_input_feasible": bool(
            np.all(steady_input >= cfg.input_lower - 1e-10)
            and np.all(steady_input <= cfg.input_upper + 1e-10)
        ),
    })
    if (
        derivative_norm > float(cfg.safe_reference_derivative_tolerance)
        or not row["physical_state_feasible"]
        or not row["physical_input_feasible"]
    ):
        row["failure_reason"] = "nonlinear_steady_or_physical_gate"
        row["elapsed_seconds"] = time.time() - started
        return row

    a, b, _ = model.linearize(state, steady_input)
    gain_seeds: list[tuple[str, np.ndarray]] = [
        ("configured_continuous_K", np.asarray(cfg.theta_k_initial_gain).copy())
    ]
    try:
        synthesized, _, _ = synthesize_hinf(cfg, a, b)
        if not np.allclose(synthesized, gain_seeds[0][1]):
            gain_seeds.append(("candidate_hinf_riccati_K", synthesized))
    except RuntimeError:
        pass

    failures: list[str] = []
    best_payload = None
    for gain_index, (gain_name, gain) in enumerate(gain_seeds):
        gate: dict[str, object] = {}
        try:
            initial = build_safety_design(
                cfg,
                model,
                np.random.default_rng(seed + 1000 * gain_index),
                theta_k=gain,
                diagnostics=gate,
            )
        except RuntimeError as exc:
            failures.append(f"{gain_name}:initial:{exc}")
            for key in (
                "hinf_feasible", "rpi_feasible", "x_tightening_feasible",
                "u_tightening_feasible", "reference_in_tightened_state_set",
                "invariant_feasible", "verification_qp_feasible",
                "residual_authority_feasible",
            ):
                row[key] = bool(row[key] or gate.get(key, False))
            continue

        learner = OnlineThetaLearner(
            cfg, model, initial, np.random.default_rng(seed + 17001 + gain_index)
        )
        try:
            design, _ = learner.build_certified_robust_operating_design(initial)
        except RuntimeError as exc:
            failures.append(f"{gain_name}:robust_region:{exc}")
            continue
        feasible_rows = [
            item for item in learner.robust_region_search_diagnostics
            if bool(item.get("feasible", False))
        ]
        if not feasible_rows:
            failures.append(f"{gain_name}:no_fully_certified_region")
            continue
        final_gate = min(
            feasible_rows,
            key=lambda item: abs(
                float(item["scale"]) - float(design.robust_region_scale)
            ),
        )
        z_ref = model.normalized_state(state)
        reference_in_s = bool(
            np.all(z_ref >= design.invariant_lower - 1e-10)
            and np.all(z_ref <= design.invariant_upper + 1e-10)
        )
        authority_ok = bool(
            design.minimum_residual_authority
            >= float(cfg.qp_min_residual_authority) - 1e-8
        )
        fully_certified = bool(
            reference_in_s
            and authority_ok
            and final_gate.get("s_plus_z_contained", False)
            and final_gate.get("input_tube_contained", False)
            and final_gate.get("residual_membership_passed", False)
            and final_gate.get("verification_qp_feasible", False)
        )
        if not fully_certified:
            failures.append(f"{gain_name}:final_gate_rejected")
            continue
        rpi_area = _polygon_area(design.rpi_boundary, cfg.state_scale)
        rpi_width = (
            (design.rpi_support_lower + design.rpi_support_upper)
            * cfg.state_scale
        )
        s_bounds = model.physical_state(np.vstack([
            design.invariant_lower, design.invariant_upper
        ]))
        x_tight = model.physical_state(np.vstack([
            design.x_lower_tight, design.x_upper_tight
        ]))
        u_tight = model.physical_input(np.vstack([
            design.u_lower_tight, design.u_upper_tight
        ]))
        acl = design.a + design.b @ design.k
        payload = dict(row)
        payload.update({
            "hinf_feasible": True,
            "rpi_feasible": True,
            "x_tightening_feasible": True,
            "u_tightening_feasible": True,
            "reference_in_tightened_state_set": True,
            "invariant_feasible": True,
            "reference_in_S": reference_in_s,
            "s_plus_z_contained": bool(final_gate["s_plus_z_contained"]),
            "input_tube_contained": bool(final_gate["input_tube_contained"]),
            "disturbance_bound_audit_passed": bool(
                final_gate["residual_membership_passed"]
            ),
            "verification_qp_feasible": bool(
                final_gate["verification_qp_feasible"]
            ),
            "residual_authority_feasible": authority_ok,
            "fully_certified": fully_certified,
            "minimum_residual_authority": float(
                design.minimum_residual_authority
            ),
            "spectral_radius": spectral_radius(acl),
            "hinf_sampled_norm": float(estimate_hinf_norm(
                design.a, design.b, design.k, cfg.hinf_q, cfg.hinf_r
            )),
            "hinf_gamma": float(cfg.hinf_gamma),
            "rpi_area": rpi_area,
            "rpi_width_X2": float(rpi_width[0]),
            "rpi_width_P2": float(rpi_width[1]),
            "S_X2_lower": float(s_bounds[0, 0]),
            "S_P2_lower": float(s_bounds[0, 1]),
            "S_X2_upper": float(s_bounds[1, 0]),
            "S_P2_upper": float(s_bounds[1, 1]),
            "X_minus_Z_X2_lower": float(x_tight[0, 0]),
            "X_minus_Z_P2_lower": float(x_tight[0, 1]),
            "X_minus_Z_X2_upper": float(x_tight[1, 0]),
            "X_minus_Z_P2_upper": float(x_tight[1, 1]),
            "U_minus_KZ_P100_lower": float(u_tight[0, 0]),
            "U_minus_KZ_F200_lower": float(u_tight[0, 1]),
            "U_minus_KZ_P100_upper": float(u_tight[1, 0]),
            "U_minus_KZ_F200_upper": float(u_tight[1, 1]),
            "robust_region_scale": float(design.robust_region_scale),
            "verification_qp_probe_count": 5,
            "gain_seed": gain_name,
            "K_00": float(design.k[0, 0]),
            "K_01": float(design.k[0, 1]),
            "K_10": float(design.k[1, 0]),
            "K_11": float(design.k[1, 1]),
            "failure_reason": "",
        })
        if best_payload is None or (
            float(payload["minimum_residual_authority"])
            > float(best_payload[0]["minimum_residual_authority"])
        ):
            best_payload = (payload, design)

    if best_payload is None:
        row["failure_reason"] = " | ".join(failures)
        row["elapsed_seconds"] = time.time() - started
        return row
    row, design = best_payload
    row["elapsed_seconds"] = time.time() - started
    design_path = design_dir / (
        f"{stage}_X2_{state[0]:.8f}_P2_{state[1]:.8f}.npz"
    )
    _write_design(design_path, design)
    row["design_file"] = design_path.name
    return row


def _save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(FIELDS) + ["design_file"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _load_rows(path: Path) -> list[dict[str, object]]:
    boolean_fields = {
        "physical_state_feasible", "physical_input_feasible",
        "hinf_feasible", "rpi_feasible", "x_tightening_feasible",
        "u_tightening_feasible", "reference_in_tightened_state_set",
        "invariant_feasible", "reference_in_S", "s_plus_z_contained",
        "input_tube_contained", "disturbance_bound_audit_passed",
        "verification_qp_feasible", "residual_authority_feasible",
        "fully_certified",
    }
    text_fields = {"stage", "gain_seed", "failure_reason", "design_file"}
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            row: dict[str, object] = {}
            for key, value in source.items():
                if key in text_fields:
                    row[key] = value
                elif key in boolean_fields:
                    row[key] = value.strip().lower() == "true"
                else:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        row[key] = float("nan")
            rows.append(row)
    return rows


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


def _candidate_key(state: np.ndarray) -> tuple[float, float]:
    return tuple(np.round(np.asarray(state, dtype=float), 10))


def main() -> None:
    args = parse_args()
    if args.coarse_x2_points < 2 or args.coarse_p2_points < 2:
        raise ValueError("coarse grid needs at least two points per axis")
    if args.refine_points < 3 or args.refine_points % 2 == 0:
        raise ValueError("refine-points must be an odd integer >= 3")
    if args.top_count < 1:
        raise ValueError("top-count must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design_dir = args.output_dir / "candidate_designs"
    design_dir.mkdir(parents=True, exist_ok=True)

    nominal_cfg = ExperimentConfig(
        benchmark_profile="zanon2016", experiment_mode="proposed", seed=args.seed
    )
    nominal_model = EvaporatorModel(nominal_cfg)
    nominal_input = nominal_model.steady_input(
        nominal_cfg.paper2016_steady_state, nominal_cfg.disturbance_nominal
    )
    nominal_cost = nominal_model.economic_cost(
        nominal_cfg.paper2016_steady_state,
        nominal_input,
        nominal_cfg.disturbance_nominal,
    )
    csv_path = args.output_dir / "reference_search_candidates.csv"
    rows = _load_rows(csv_path) if csv_path.exists() else []
    evaluated = {
        _candidate_key(np.array([row["X2"], row["P2"]])) for row in rows
    }
    if rows:
        print(f"resuming {len(rows)} completed candidates from {csv_path}", flush=True)

    def evaluate_grid(stage: str, x_axis: np.ndarray, p_axis: np.ndarray) -> None:
        states = [np.asarray(value, dtype=float) for value in product(x_axis, p_axis)]
        states.sort(key=lambda state: float(nominal_model.economic_cost(
            state,
            nominal_model.steady_input(state, nominal_cfg.disturbance_nominal),
            nominal_cfg.disturbance_nominal,
        )))
        remaining = [state for state in states if _candidate_key(state) not in evaluated]
        completed_in_stage = 0
        while remaining:
            certified_now = sorted(
                (row for row in rows if bool(row["fully_certified"])),
                key=lambda row: float(row["steady_cost"]),
            )
            state = remaining[0]
            candidate_cost = float(nominal_model.economic_cost(
                state,
                nominal_model.steady_input(
                    state, nominal_cfg.disturbance_nominal
                ),
                nominal_cfg.disturbance_nominal,
            ))
            # Exact cost ordering provides a safe branch-and-bound rule: once
            # the next untested state is no cheaper than the current Nth fully
            # certified point, no remaining point on this grid can change the
            # requested top-N list.
            if (
                len(certified_now) >= args.top_count
                and candidate_cost
                >= float(certified_now[args.top_count - 1]["steady_cost"])
            ):
                print(
                    f"[{stage}] economic branch bound reached after "
                    f"{completed_in_stage}/{len(states)} ordered grid points",
                    flush=True,
                )
                break
            batch = remaining[:args.workers]
            remaining = remaining[len(batch):]
            for batch_state in batch:
                print(
                    f"[{stage}] certify X2={batch_state[0]:.8f}, "
                    f"P2={batch_state[1]:.8f}",
                    flush=True,
                )
            if args.workers == 1:
                batch_rows = [evaluate_candidate(
                    batch[0], stage, args.seed, nominal_cost, design_dir
                )]
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = [executor.submit(
                        evaluate_candidate,
                        batch_state,
                        stage,
                        args.seed,
                        nominal_cost,
                        design_dir,
                    ) for batch_state in batch]
                    batch_rows = [future.result() for future in as_completed(futures)]
            for row in batch_rows:
                rows.append(row)
                evaluated.add(_candidate_key(np.array([row["X2"], row["P2"]])))
                completed_in_stage += 1
                _save_rows(csv_path, rows)
                print(
                    f"  X2={float(row['X2']):.8f}, P2={float(row['P2']):.8f}, "
                    f"certified={bool(row['fully_certified'])}, "
                    f"cost={float(row['steady_cost']):.9g}, "
                    f"gate={row['failure_reason'] or 'all_passed'}, "
                    f"elapsed={float(row['elapsed_seconds']):.1f}s",
                    flush=True,
                )

    coarse_x = np.linspace(args.x2_min, args.x2_max, args.coarse_x2_points)
    coarse_p = np.linspace(args.p2_min, args.p2_max, args.coarse_p2_points)
    evaluate_grid("coarse", coarse_x, coarse_p)
    certified = [row for row in rows if bool(row["fully_certified"])]
    if not certified:
        raise RuntimeError(
            "No fully certified reference was found on the coarse 2-D grid; "
            "local refinement cannot start. See reference_search_candidates.csv."
        )

    dx = float(coarse_x[1] - coarse_x[0])
    dp = float(coarse_p[1] - coarse_p[0])
    for level in range(1, args.refine_levels + 1):
        prior_stages = {"coarse"} | {
            f"refine_{prior}" for prior in range(1, level)
        }
        prior_certified = [
            row for row in certified
            if str(row["stage"]) in prior_stages
        ]
        best = min(
            prior_certified,
            key=lambda row: float(row["steady_cost"]),
        )
        x_center, p_center = float(best["X2"]), float(best["P2"])
        x_axis = np.linspace(
            max(args.x2_min, x_center - dx),
            min(args.x2_max, x_center + dx),
            args.refine_points,
        )
        p_axis = np.linspace(
            max(args.p2_min, p_center - dp),
            min(args.p2_max, p_center + dp),
            args.refine_points,
        )
        evaluate_grid(f"refine_{level}", x_axis, p_axis)
        certified = [row for row in rows if bool(row["fully_certified"])]
        dx = float(x_axis[1] - x_axis[0])
        dp = float(p_axis[1] - p_axis[0])

    certified.sort(key=lambda row: float(row["steady_cost"]))
    top10 = certified[:args.top_count]
    best = top10[0]
    summary = {
        "method": (
            "coarse 2-D grid followed by local grid refinement; every accepted "
            "candidate independently rebuilds and passes the complete proposed "
            "H-infinity/RPI/tightening/invariant/QP/residual-authority design"
        ),
        "benchmark_profile": "zanon2016",
        "experiment_mode": "proposed",
        "sac_training_performed": False,
        "nominal_paper2016_state": nominal_cfg.paper2016_steady_state.tolist(),
        "nominal_paper2016_input": nominal_input.tolist(),
        "nominal_paper2016_cost": float(nominal_cost),
        "search_bounds": {
            "X2": [args.x2_min, args.x2_max],
            "P2": [args.p2_min, args.p2_max],
        },
        "coarse_shape": [args.coarse_x2_points, args.coarse_p2_points],
        "refine_points": args.refine_points,
        "refine_levels": args.refine_levels,
        "requested_top_count": args.top_count,
        "candidate_count": len(rows),
        "fully_certified_candidate_count": len(certified),
        "best": best,
        "top10_fully_certified": top10,
    }
    with (args.output_dir / "reference_search_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_json_ready(summary), stream, indent=2, ensure_ascii=False)
    with (args.output_dir / "top10_certified_candidates.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_json_ready(top10), stream, indent=2, ensure_ascii=False)
    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
