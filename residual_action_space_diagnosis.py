"""Fixed-B offline residual-action geometry scan; never trains or redesigns.

Run from the parent directory with ``python -m evaporation.residual_action_space_diagnosis``.
The stored B certificate and the previous best actor are read-only inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .config import ExperimentConfig
from .control import (
    SafeController, finite_horizon_nominal_policy,
    maximum_reservable_authority, point_in_convex_polygon,
    project_qp_2d, safe_projected_base,
)
from .model import EvaporatorModel
from .paper2016_compare import SCENARIOS, apply_state_shock
from .train import observation


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_DESIGN = REPO_DIR / (
    "evaporation_safe_sac/outputs_certified_reference_margin_audit/"
    "candidate_designs/refine_1_X2_25.39000000_P2_50.12500000.npz"
)
DEFAULT_ACTOR = REPO_DIR / (
    "evaporation_safe_sac/outputs_robust_reference_seed42_200x300/"
    "models/best_post_warmup_learned_actor.pth"
)
DEFAULT_STRESS_ACTOR = REPO_DIR / (
    "evaporation_safe_sac/outputs_robust_reference_seed42_200x300/"
    "models/last_actor.pth"
)
FRACTIONS = (0.5, 0.7, 0.8, 0.9)
MODES = {
    "current_5pct_box": ("floor", "state_dependent_box"),
    "max_margin_symmetric_box": ("fraction_of_max", "state_dependent_box"),
    "max_margin_asymmetric_polytope": (
        "fraction_of_max", "state_dependent_polytope"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    parser.add_argument("--stress-actor", type=Path, default=DEFAULT_STRESS_ACTOR)
    parser.add_argument("--output-dir", type=Path, default=REPO_DIR / (
        "evaporation_safe_sac/outputs_residual_action_space_B"
    ))
    return parser.parse_args()


def load_fixed_b(path: Path, cfg: ExperimentConfig, model: EvaporatorModel):
    with np.load(path) as saved:
        values = {key: np.asarray(saved[key]).copy() for key in saved.files}
    expected_state = np.array([25.39, 50.125])
    cfg.robust_economic_reference_state = expected_state.copy()
    cfg.safe_center_state = expected_state.copy()
    cfg.linearization_state = expected_state.copy()
    cfg.linearization_input = model.steady_input(expected_state)
    if not np.allclose(model.physical_state(values["z_ref"]), expected_state, atol=1e-8):
        raise ValueError("design is not the certified balanced reference B")
    cfg.robust_economic_reference_input = model.physical_input(values["v_ref"])
    cfg.linearization_input = cfg.robust_economic_reference_input.copy()
    gain, _ = finite_horizon_nominal_policy(
        cfg, values["A"], values["B"], np.zeros(2), np.zeros(4), np.zeros(2)
    )
    return SimpleNamespace(
        a=values["A"], b=values["B"], affine=values["affine"],
        k=values["K"], w_vertices=values["W_vertices"],
        w_bound=np.max(np.abs(values["W_vertices"]), axis=0),
        rpi_boundary=values["rpi_boundary"],
        rpi_support_lower=values["rpi_support_lower"],
        rpi_support_upper=values["rpi_support_upper"],
        robust_state_lower=values["robust_state_lower"],
        robust_state_upper=values["robust_state_upper"],
        robust_input_lower=values["robust_input_lower"],
        robust_input_upper=values["robust_input_upper"],
        u_lower_tight=values["u_lower_tight"],
        u_upper_tight=values["u_upper_tight"],
        invariant_lower=values["invariant_lower"],
        invariant_upper=values["invariant_upper"],
        z_ref=values["z_ref"], v_ref=values["v_ref"],
        nominal_policy_gain=gain,
        nominal_policy_offset=values["v_ref"] - gain @ values["z_ref"],
        theta_h=np.zeros(4), theta_p=np.zeros(2), theta_m_angle=0.0,
    )


def verification_rows(design, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = design.a @ z + design.affine
    return np.vstack([np.eye(2), -np.eye(2), design.b, -design.b]), np.concatenate([
        design.u_upper_tight, -design.u_lower_tight,
        design.invariant_upper - center, -design.invariant_lower + center,
    ])


def polygon_vertices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    vertices = []
    for i, j in combinations(range(len(b)), 2):
        pair = np.vstack([a[i], a[j]])
        if abs(np.linalg.det(pair)) <= 1e-12:
            continue
        vertex = np.linalg.solve(pair, np.array([b[i], b[j]]))
        if np.all(a @ vertex <= b + 1e-8):
            vertices.append(vertex)
    if not vertices:
        return np.empty((0, 2))
    unique = np.unique(np.round(vertices, 10), axis=0)
    center = np.mean(unique, axis=0)
    angles = np.arctan2(unique[:, 1] - center[1], unique[:, 0] - center[0])
    return unique[np.argsort(angles)]


def polygon_area(vertices: np.ndarray, scale: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    physical = vertices * scale
    return 0.5 * abs(float(np.sum(
        physical[:, 0] * np.roll(physical[:, 1], -1)
        - physical[:, 1] * np.roll(physical[:, 0], -1)
    )))


class FrozenActor:
    def __init__(self, path: Path, cfg: ExperimentConfig):
        import torch
        from .sac import GaussianPolicy
        torch.set_num_threads(1)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        observation_dim = int(saved["actor"]["backbone.0.weight"].shape[1])
        self.actor = GaussianPolicy(observation_dim, 2, cfg.hidden_dim)
        self.actor.load_state_dict(saved["actor"])
        self.actor.eval()
        self.scale = float(saved.get("policy_output_scale", 1.0))

    def action(self, obs: np.ndarray) -> np.ndarray:
        return self.scale * self.actor.act(obs, True, "cpu")


def rollout(cfg, model, design, scenario, mode, actor=None, probe_actors=None):
    reserve_mode, parameterization = MODES[mode]
    cfg.residual_reserve_mode = reserve_mode
    cfg.residual_reserve_fraction = 0.8
    cfg.residual_parameterization = parameterization
    controller = SafeController(cfg, model, design)
    state = cfg.robust_economic_reference_state.copy()
    controller.reset(state)
    previous_u = design.v_ref.copy()
    w_est = np.zeros(2)
    rows = []
    for second in range(300):
        state = apply_state_shock(state, scenario, second, scale=1.0)
        obs = observation(model, controller, state, previous_u, w_est)
        raw_action = np.zeros(2) if actor is None else actor.action(obs)
        probe_actions = (
            [probe.action(obs) for probe in probe_actors]
            if probe_actors is not None else None
        )
        control, info = controller.act(state, raw_action, action_is_normalized=True)
        next_state = model.step(state, control, cfg.disturbance_nominal)
        next_x = model.normalized_state(next_state)
        e_next = next_x - controller.z
        w_hat = next_x - (
            design.a @ info["x_norm"] + design.b @ info["actual_norm"]
            + design.affine
        )
        w_est = cfg.disturbance_estimate_ema * w_est + (
            1.0 - cfg.disturbance_estimate_ema
        ) * w_hat
        physical_state_bad = bool(
            np.any(next_state < cfg.state_lower - 1e-8)
            or np.any(next_state > cfg.state_upper + 1e-8)
        )
        physical_input_bad = bool(
            np.any(control < cfg.input_lower - 1e-8)
            or np.any(control > cfg.input_upper + 1e-8)
        )
        physical_bad = physical_state_bad or physical_input_bad
        robust_state_bad = bool(
            np.any(next_x < design.robust_state_lower - 1e-8)
            or np.any(next_x > design.robust_state_upper + 1e-8)
        )
        robust_input_bad = bool(
            np.any(info["actual_norm"] < design.robust_input_lower - 1e-8)
            or np.any(info["actual_norm"] > design.robust_input_upper + 1e-8)
        )
        robust_bad = robust_state_bad or robust_input_bad
        a, b = info["a_q"], info["b_q"]
        slack = b - a @ info["base"]
        vertices = polygon_vertices(a, slack)
        if len(vertices):
            neg = -np.min(vertices, axis=0) * cfg.input_scale
            pos = np.max(vertices, axis=0) * cfg.input_scale
            area = polygon_area(vertices, cfg.input_scale)
        else:
            neg = pos = np.zeros(2)
            area = 0.0
        requested = np.asarray(info["requested_residual"])
        applied = np.asarray(info["nominal"]) - np.asarray(info["base"])
        requested_norm = float(np.linalg.norm(requested))
        rows.append({
            "scenario": scenario, "second": second, "mode": mode,
            "state_X2": float(state[0]), "state_P2": float(state[1]),
            "z_X2": float(info["z"][0]), "z_P2": float(info["z"][1]),
            "actor_a_P100": float(raw_action[0]),
            "actor_a_F200": float(raw_action[1]),
            "lambda_max": float(info["maximum_residual_authority"]),
            "lambda_reserve": float(info["residual_reserve_target"]),
            "actual_authority": float(info["reserved_residual_authority"]),
            "base_projection_displacement": float(info["theta_projection_gap"]),
            "requested_residual_norm": requested_norm,
            "applied_residual_norm": float(np.linalg.norm(applied)),
            "applied_over_requested": (
                float(np.linalg.norm(applied)) / requested_norm
                if requested_norm > 1e-12 else float("nan")
            ),
            "requested_over_applied": (
                requested_norm / float(np.linalg.norm(applied))
                if np.linalg.norm(applied) > 1e-12 else float("nan")
            ),
            "raw_request_feasible": bool(
                np.all(a @ info["requested_candidate"] <= b + 1e-8)
            ),
            "positive_P100": float(pos[0]), "negative_P100": float(neg[0]),
            "positive_F200": float(pos[1]), "negative_F200": float(neg[1]),
            "P100_residual_range_lower": float(-neg[0]),
            "P100_residual_range_upper": float(pos[0]),
            "F200_residual_range_lower": float(-neg[1]),
            "F200_residual_range_upper": float(pos[1]),
            "feasible_residual_polytope_area": area,
            "qp_violation": not bool(info["qp_feasible"]),
            "physical_violation": physical_bad,
            "physical_state_violation": physical_state_bad,
            "physical_input_violation": physical_input_bad,
            "robust_region_violation": robust_bad,
            "robust_state_violation": robust_state_bad,
            "robust_input_violation": robust_input_bad,
            "rpi_violation": not point_in_convex_polygon(
                e_next, design.rpi_boundary, tol=1e-8
            ),
            "probe_best_a_P100": (
                float(probe_actions[0][0]) if probe_actions is not None else float("nan")
            ),
            "probe_best_a_F200": (
                float(probe_actions[0][1]) if probe_actions is not None else float("nan")
            ),
            "probe_last_a_P100": (
                float(probe_actions[1][0]) if probe_actions is not None else float("nan")
            ),
            "probe_last_a_F200": (
                float(probe_actions[1][1]) if probe_actions is not None else float("nan")
            ),
        })
        state = next_state
        previous_u = np.asarray(info["actual_norm"])
    return rows


def paired_action_probes(cfg, model, design, baseline):
    """Compare mappings at identical certified-B baseline states, without stepping."""
    result = []
    for sample in baseline:
        state = np.array([sample["state_X2"], sample["state_P2"]])
        z = np.array([sample["z_X2"], sample["z_P2"]])
        for actor_name, prefix in (("old_best", "probe_best"),
                                   ("nonzero_last", "probe_last")):
            action = np.array([
                sample[f"{prefix}_a_P100"], sample[f"{prefix}_a_F200"]
            ])
            for mode, (reserve_mode, parameterization) in MODES.items():
                cfg.residual_reserve_mode = reserve_mode
                cfg.residual_parameterization = parameterization
                controller = SafeController(cfg, model, design)
                controller.z = z.copy()
                _, info = controller.act(state, action, action_is_normalized=True)
                a, b = info["a_q"], info["b_q"]
                slack = b - a @ info["base"]
                vertices = polygon_vertices(a, slack)
                negative = -np.min(vertices, axis=0) * cfg.input_scale
                positive = np.max(vertices, axis=0) * cfg.input_scale
                requested = np.linalg.norm(info["requested_residual"])
                applied = np.linalg.norm(info["nominal"] - info["base"])
                result.append({
                    "actor": actor_name, "mode": mode,
                    "scenario": sample["scenario"], "second": sample["second"],
                    "lambda_max": float(info["maximum_residual_authority"]),
                    "lambda_reserve": float(info["residual_reserve_target"]),
                    "actual_authority": float(info["reserved_residual_authority"]),
                    "base_projection_displacement": float(info["theta_projection_gap"]),
                    "requested_residual_norm": float(requested),
                    "applied_residual_norm": float(applied),
                    "applied_over_requested": (
                        float(applied / requested) if requested > 1e-12 else float("nan")
                    ),
                    "requested_over_applied": (
                        float(requested / applied) if applied > 1e-12 else float("nan")
                    ),
                    "raw_request_feasible": bool(
                        np.all(a @ info["requested_candidate"] <= b + 1e-8)
                    ),
                    "positive_P100": float(positive[0]),
                    "negative_P100": float(negative[0]),
                    "positive_F200": float(positive[1]),
                    "negative_F200": float(negative[1]),
                    "feasible_residual_polytope_area": polygon_area(
                        vertices, cfg.input_scale
                    ),
                    "qp_violation": not bool(info["qp_feasible"]),
                })
    return result


def geometry_scan(cfg, design, baseline):
    points = [("reference", design.z_ref)]
    corners = [np.asarray(v) for v in product(*zip(
        design.invariant_lower, design.invariant_upper
    ))]
    points.extend((f"S_vertex_{i}", v) for i, v in enumerate(corners))
    points.extend((f"S_edge_midpoint_{i}", 0.5 * (corners[i] + corners[j]))
                  for i, j in ((0, 1), (0, 2), (1, 3), (2, 3)))
    points.extend((f"{r['scenario']}_t{r['second']}", np.array([
        r["z_X2"], r["z_P2"]
    ])) for r in baseline)
    result = []
    for label, z in points:
        a, b = verification_rows(design, z)
        theta = design.nominal_policy_gain @ z + design.nominal_policy_offset
        maximum, feasible = maximum_reservable_authority(
            theta, a, b, cfg.residual_action_scale
        )
        if not feasible:
            raise RuntimeError(f"zero-residual QP infeasible at {label}")
        support = np.abs(a) @ cfg.residual_action_scale
        for fraction in FRACTIONS:
            reserve = max(cfg.qp_min_residual_authority, fraction * maximum)
            base, ok = project_qp_2d(theta, a, b - reserve * support)
            if not ok:
                raise RuntimeError(f"reserve QP infeasible at {label}, {fraction}")
            constrained = support > 1e-12
            actual = float(np.min((b - a @ base)[constrained] / support[constrained]))
            result.append({
                "point": label, "fraction": fraction,
                "lambda_max": maximum, "lambda_reserve": reserve,
                "actual_authority": actual,
                "base_projection_displacement": float(np.linalg.norm(base - theta)),
                "symmetric_residual_width_P100": float(
                    2.0 * actual * cfg.residual_action_scale[0] * cfg.input_scale[0]
                ),
                "symmetric_residual_width_F200": float(
                    2.0 * actual * cfg.residual_action_scale[1] * cfg.input_scale[1]
                ),
            })
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    result = {}
    for mode in MODES:
        result[mode] = {}
        for scenario in SCENARIOS:
            subset = [r for r in rows if r["mode"] == mode and r["scenario"] == scenario]
            if not subset:
                continue
            metrics = {}
            for key in (
                "applied_over_requested", "requested_over_applied",
                "positive_P100", "negative_P100",
                "positive_F200", "negative_F200", "feasible_residual_polytope_area",
                "lambda_max", "lambda_reserve", "actual_authority",
                "base_projection_displacement",
            ):
                vals = np.asarray([r[key] for r in subset], dtype=float)
                vals = vals[np.isfinite(vals)]
                metrics[key] = {
                    "min": float(np.min(vals)), "p50": float(np.percentile(vals, 50)),
                    "p90": float(np.percentile(vals, 90)),
                    "mean": float(np.mean(vals)),
                } if len(vals) else None
            for key in ("raw_request_feasible", "qp_violation",
                        "physical_violation", "physical_state_violation",
                        "physical_input_violation", "robust_region_violation",
                        "robust_state_violation", "robust_input_violation",
                        "rpi_violation"):
                metrics[key + "_rate"] = (
                    float(np.mean([r[key] for r in subset]))
                    if key in subset[0] else None
                )
            result[mode][scenario] = metrics
    return result


def summarize_reserve_scan(rows):
    categories = {
        "reference": lambda label: label == "reference",
        "S_vertices": lambda label: label.startswith("S_vertex_"),
        "S_edge_midpoints": lambda label: label.startswith("S_edge_midpoint_"),
    }
    categories.update({
        scenario: (lambda label, scenario=scenario: label.startswith(scenario + "_t"))
        for scenario in SCENARIOS
    })
    result = {}
    for name, predicate in categories.items():
        result[name] = {}
        for fraction in FRACTIONS:
            subset = [r for r in rows if r["fraction"] == fraction and predicate(r["point"])]
            result[name][str(fraction)] = {
                key: {
                    "min": float(np.min(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p90": float(np.percentile(values, 90)),
                    "max": float(np.max(values)),
                }
                for key in (
                    "lambda_max", "lambda_reserve", "actual_authority",
                    "base_projection_displacement",
                    "symmetric_residual_width_P100",
                    "symmetric_residual_width_F200",
                )
                for values in [[r[key] for r in subset]]
            }
    return result


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed")
    model = EvaporatorModel(cfg)
    design = load_fixed_b(args.design, cfg, model)
    actor = FrozenActor(args.actor, cfg)
    stress_actor = FrozenActor(args.stress_actor, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = []
    for scenario in SCENARIOS:
        print(f"baseline {scenario}", flush=True)
        baseline.extend(rollout(
            cfg, model, design, scenario, "max_margin_symmetric_box",
            probe_actors=(actor, stress_actor),
        ))
    scan = geometry_scan(cfg, design, baseline)
    write_csv(args.output_dir / "reserve_fraction_scan.csv", scan)
    write_csv(args.output_dir / "zero_residual_baseline_steps.csv", baseline)
    paired = paired_action_probes(cfg, model, design, baseline)
    write_csv(args.output_dir / "paired_baseline_action_probes.csv", paired)
    actor_rows = []
    for mode in MODES:
        for scenario in SCENARIOS:
            print(f"old actor {mode} {scenario}", flush=True)
            actor_rows.extend(rollout(cfg, model, design, scenario, mode, actor))
    write_csv(args.output_dir / "old_actor_action_space_steps.csv", actor_rows)
    stress_rows = []
    for mode in MODES:
        for scenario in SCENARIOS:
            print(f"nonzero last actor {mode} {scenario}", flush=True)
            stress_rows.extend(rollout(
                cfg, model, design, scenario, mode, stress_actor
            ))
    write_csv(args.output_dir / "nonzero_last_actor_action_space_steps.csv", stress_rows)
    summary = {
        "scope": "fixed certified B geometry; no SAC training or geometry redesign",
        "reference_state": cfg.robust_economic_reference_state.tolist(),
        "reference_input": cfg.robust_economic_reference_input.tolist(),
        "K": design.k.tolist(),
        "residual_action_scale": cfg.residual_action_scale.tolist(),
        "reserve_floor": cfg.qp_min_residual_authority,
        "reserve_fraction_scan": {
            str(fraction): {
                key: float(np.min([r[key] for r in scan if r["fraction"] == fraction]))
                for key in ("lambda_max", "lambda_reserve", "actual_authority",
                            "symmetric_residual_width_P100", "symmetric_residual_width_F200")
            } for fraction in FRACTIONS
        },
        "reserve_fraction_scan_by_point_class": summarize_reserve_scan(scan),
        "baseline": summarize(baseline)["max_margin_symmetric_box"],
        "paired_baseline_action_probes": {
            label: summarize([r for r in paired if r["actor"] == label])
            for label in ("old_best", "nonzero_last")
        },
        "old_best_actor_policy_output_scale": actor.scale,
        "old_best_actor": summarize(actor_rows),
        "nonzero_last_actor_policy_output_scale": stress_actor.scale,
        "nonzero_last_actor_stress_probe": summarize(stress_rows),
        "nonzero_last_actor_closed_loop_safety_gate": {
            mode: bool(all(
                not r["qp_violation"]
                and not r["physical_violation"]
                and not r["robust_region_violation"]
                for r in stress_rows if r["mode"] == mode
            )) for mode in MODES
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary["reserve_fraction_scan"], indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
