"""Offline alpha=.2 reward-scale audit; never trains or changes safety geometry.

Implementation choice for reproduction, not specified in the paper: calibrate
existing shaping weights on ten paired stochastic zero-residual realizations.
This is a scale calibration, not evidence that a new policy will improve.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .zanon2019_benchmark import make_setup, write_csv

REPO = Path(__file__).parent
DEFAULT_SOURCE = REPO / "evaporation_safe_sac/authority_cuda_seed42_100x1000/outputs_zanon2019_alpha020_seed42_100x1000"
DEFAULT_OUTPUT = REPO / "evaporation_safe_sac/outputs_zanon2019_alpha020_reward_scale_audit_v1"
WEIGHT_FIELDS = {
    "state_recovery": "paper2016_state_recovery_penalty_weight",
    "P100_move": "paper2016_p100_move_penalty_weight",
    "F200_move": "paper2016_f200_move_penalty_weight",
    "saturation": "paper2016_saturation_penalty_weight",
}
TARGET_FRACTIONS = {"state_recovery": .10, "P100_move": .05, "F200_move": .05}


def calibrate(means, economic_magnitude, old_weights):
    if not np.isfinite(economic_magnitude) or economic_magnitude <= 1e-12:
        raise ValueError("Cannot calibrate against zero/nonfinite economic magnitude")
    weights = dict(old_weights)
    for name, fraction in TARGET_FRACTIONS.items():
        if not np.isfinite(means[name]) or means[name] <= 1e-12:
            raise ValueError(f"Cannot calibrate unexcited/nonfinite loss: {name}")
        weights[name] = float(fraction * economic_magnitude / means[name])
    # Baseline saturation is unexcited. Keep its old weight, never divide by 0.
    return weights


def apply_calibration(cfg, path, *, geometry_sources):
    """Explicit opt-in; fixed alpha=.2, unchanged scales/reference/safety/SAC."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (cfg.disturbance_mode != "zanon2019_stochastic"
            or cfg.main_experiment_protocol != "zanon2019_stochastic_operating_uncertainty"
            or cfg.stochastic_residual_scale != .2):
        raise ValueError("This reward recalibration requires stochastic alpha=0.2")
    if data.get("schema") != "zanon2019_alpha020_reward_scale_v1":
        raise ValueError("Unknown reward calibration schema")
    context = data["context"]
    for name in ("state_scale", "input_scale", "robust_economic_reference_state",
                 "linearization_state", "linearization_input", "disturbance_nominal"):
        if not np.array_equal(np.asarray(context[name]), np.asarray(getattr(cfg, name))):
            raise ValueError(f"Calibration context differs: {name}")
    if context["reward_cost_scale"] != cfg.reward_cost_scale or context["dt_min"] != cfg.dt_min:
        raise ValueError("Calibration reward/time scale differs")
    if context["geometry_sha256"] != [hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                       for p in geometry_sources]:
        raise ValueError("Calibration geometry differs; do not silently reuse weights")
    weights = data["calibrated_weights"]
    if set(weights) != set(WEIGHT_FIELDS) or any(
            not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Expected exactly four finite nonnegative reward weights")
    for name, field in WEIGHT_FIELDS.items():
        if getattr(cfg, field) != data["old_weights"][name]:
            raise ValueError(f"Frozen old reward differs: {name}")
    for name, field in WEIGHT_FIELDS.items():
        setattr(cfg, field, float(weights[name]))
    return weights, data


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def trace_components(rows, label, cfg, model, design):
    def matrix(name):
        return np.array([[float(r[f"{label}_{name}_{i}"]) for i in range(2)] for r in rows])
    states, inputs, w = matrix("state"), matrix("control"), matrix("w_hat")
    next_x = model.physical_state(model.normalized_state(states) @ design.a.T
        + model.normalized_input(inputs) @ design.b.T + design.affine + w)
    if not np.allclose(next_x[:-1], states[1:], rtol=0, atol=1e-9):
        raise ValueError("Post-state reconstruction does not match next CSV state")
    cost = np.array([float(r[f"{label}_economic_cost"]) for r in rows])
    for i in (0, len(rows) - 1):
        disturbance = np.array([float(rows[i][k]) for k in ("F1", "X1", "T1", "T200")])
        if not np.isclose(cost[i], model.economic_cost(next_x[i], inputs[i], disturbance),
                          rtol=0, atol=1e-7):
            raise ValueError("Economic cost timing/reconstruction mismatch")
    reference_cost = model.economic_cost(cfg.linearization_state, cfg.linearization_input,
                                          cfg.disturbance_nominal)
    economic = (reference_cost - cost) / cfg.reward_cost_scale
    lx = ((next_x - cfg.robust_economic_reference_state) / cfg.state_scale) ** 2
    # Exactly run_episode: first previous FINAL input is physical(v_ref).
    du = np.diff(np.vstack([model.physical_input(design.v_ref), inputs]), axis=0) / cfg.input_scale
    lo, hi = (model.physical_input(v) for v in (design.robust_input_lower, design.robust_input_upper))
    eta = np.abs(inputs - (lo + hi) / 2) / ((hi - lo) / 2)
    result = {"economic_reward_mean": float(economic.mean()),
        "mean_absolute_economic_reward": float(np.abs(economic).mean()),
        "X2_loss": float(lx[:, 0].mean()), "P2_loss": float(lx[:, 1].mean()),
        "state_recovery": float(lx.sum(axis=1).mean()),
        "P100_move": float((du[:, 0] ** 2).mean()),
        "F200_move": float((du[:, 1] ** 2).mean()),
        "saturation": float((np.maximum(eta - .9, 0) ** 2).sum(axis=1).mean())}
    return result


def reward_metrics(records):
    """Reporting only; eval raw reward contains RPI penalties, replay does not."""
    keys = ("economic_reward", "state_recovery_loss", "state_recovery_penalty",
        "p100_move_loss", "p100_move_penalty", "f200_move_loss", "f200_move_penalty",
        "saturation_loss", "saturation_penalty", "projection_penalty", "mapping_penalty",
        "move_penalty", "safety_penalty", "reward_safety_penalty",
        "state_violation_penalty", "input_violation_penalty", "qp_infeasible_penalty",
        "state_excess_penalty", "input_excess_penalty",
        "rpi_violation_event_penalty", "rpi_excess_penalty", "total_reward")
    result = {}
    for key in keys:
        values = np.array([r[key] for r in records], dtype=float)
        result[f"reward_{key}_mean"] = float(values.mean())
        result[f"reward_{key}_total"] = float(values.sum())
    result["reward_mean_absolute_economic"] = float(np.mean([abs(r["economic_reward"]) for r in records]))
    equivalent = [r["total_reward"] + (0 if r["rpi_penalty_excluded_from_training_reward"] else
                 r["rpi_violation_event_penalty"] + r["rpi_excess_penalty"]) for r in records]
    result["reward_replay_equivalent_total"] = float(sum(equivalent))
    for name in ("state_recovery", "p100_move", "f200_move", "saturation"):
        result[f"reward_{name}_fraction_of_abs_economic"] = (
            result[f"reward_{name}_penalty_mean"] / max(result["reward_mean_absolute_economic"], 1e-12))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("Audit output exists; choose a fresh directory")
    source = args.source_dir
    manifest = json.loads((source / "experiment_manifest.json").read_text(encoding="utf-8"))
    if manifest["stochastic_residual_scale"] != .2 or manifest["mode"] != "zanon2019_stochastic":
        raise ValueError("Audit requires alpha=.2 iid stochastic evidence")
    geometry = list(manifest["frozen_geometry_sources"])
    cfg, model, design, _, _, current_weights = make_setup(manifest["steps"], manifest["seed"], *geometry)
    if current_weights != manifest["weights"]:
        raise ValueError("Source reward weights differ from current frozen setup")
    context = {name: np.asarray(getattr(cfg, name)).tolist() for name in (
        "state_scale", "input_scale", "robust_economic_reference_state", "linearization_state",
        "linearization_input", "disturbance_nominal")}
    context.update(reward_cost_scale=cfg.reward_cost_scale, dt_min=cfg.dt_min,
        geometry_sha256=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in geometry])
    frozen = manifest["frozen_experiment_config"]
    for name in context.keys() - {"geometry_sha256"}:
        if not np.array_equal(np.asarray(frozen[name]), np.asarray(context[name])):
            raise ValueError(f"Source/current calibration context mismatch: {name}")
    if context["geometry_sha256"] != list(manifest["frozen_geometry_sources"].values()):
        raise ValueError("Geometry hash mismatch")
    fixed = read_rows(source / "fixed_evaluation.csv")
    base = []
    for seed in manifest["validation_seeds"]:
        base.append(trace_components(read_rows(source / "paired_baselines" / f"seed_{seed}.csv"),
                                     "baseline", cfg, model, design))
    means = {k: float(np.mean([r[k] for r in base])) for k in base[0]}
    new = calibrate(means, means["mean_absolute_economic_reward"], current_weights)
    rows = []
    for metric in fixed:
        ep, seed = int(metric["episode"]), int(metric["seed"])
        trace = read_rows(source / "evaluation_trajectories" / f"episode_{ep:04d}" / f"seed_{seed}.csv")
        for label in ("baseline", "SAC"):
            comp = trace_components(trace, label, cfg, model, design)
            saved_j = float(metric[f"{label}_J_econ"])
            # Historical J_econ is sum of stage costs, not a time integral.
            actual_j = sum(float(r[f"{label}_economic_cost"]) for r in trace)
            if not np.isclose(saved_j, actual_j, rtol=0, atol=1e-6):
                raise ValueError("Saved J_econ reconstruction mismatch")
            row = {"episode": ep, "seed": seed, "controller": label, **comp,
                **{k: float(metric[k]) for k in ("economic_improvement_percent", "X2_IAE_ratio",
                    "P2_IAE_ratio", "X2_ISE_ratio", "P2_ISE_ratio", "P100_TV_ratio", "F200_TV_ratio")}}
            for kind, weights in (("old", current_weights), ("recalibrated", new)):
                for name in weights:
                    row[f"{kind}_{name}_penalty_mean"] = weights[name] * comp[name]
                    row[f"{kind}_{name}_fraction_of_abs_economic"] = (
                        weights[name] * comp[name] / max(comp["mean_absolute_economic_reward"], 1e-12))
                row[f"{kind}_economic_minus_four_shaping_mean"] = (
                    comp["economic_reward_mean"] - sum(weights[k] * comp[k] for k in weights))
            rows.append(row)
    summary = {"schema": "zanon2019_alpha020_reward_scale_v1", "source_directory": str(source.resolve()),
        "context": context, "stochastic_residual_scale": .2,
        "calibration_seeds": manifest["validation_seeds"], "reserved_final_test_seeds_used": False,
        "baseline_means": means, "old_weights": current_weights, "calibrated_weights": new,
        "target_fractions": TARGET_FRACTIONS,
        "old_baseline_fractions": {k: current_weights[k]*means[k]/means["mean_absolute_economic_reward"] for k in new},
        "new_baseline_fractions": {k: new[k]*means[k]/means["mean_absolute_economic_reward"] for k in new},
        "weight_multipliers": {k: new[k]/current_weights[k] for k in new},
        "saturation_status": "baseline unexcited; original saturation weight retained",
        "scope": "implementation choice for reproduction; not specified in the paper",
        "training_performed": False,
        "limitations": ["Squared shaping is a surrogate, not an IAE/TV constraint or economics guarantee",
            "Historical CSV lacks some other penalties; economic-minus-four-shaping is NOT full return",
            "Raw evaluation RPI penalties must not be confused with replay reward",
            "Calibration uses validation realizations; final held-out test remains unused"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "reward_components_per_seed.csv", rows)
    aggregates = []
    for ep in sorted({r["episode"] for r in rows}):
        for label in ("baseline", "SAC"):
            group = [r for r in rows if r["episode"] == ep and r["controller"] == label]
            aggregates.append({"episode": ep, "controller": label, **{k: float(np.mean([r[k] for r in group]))
                for k in group[0] if k not in ("episode", "seed", "controller")}})
    write_csv(args.output_dir / "reward_components_by_checkpoint.csv", aggregates)
    paired = []
    for ep in sorted({r["episode"] for r in aggregates}):
        b, s = [next(r for r in aggregates if r["episode"] == ep and r["controller"] == label)
                for label in ("baseline", "SAC")]
        pair = {"episode": ep, **{k: s[k] for k in (
            "economic_improvement_percent", "X2_IAE_ratio", "P2_IAE_ratio",
            "X2_ISE_ratio", "P2_ISE_ratio", "P100_TV_ratio", "F200_TV_ratio")}}
        for kind in ("old", "recalibrated"):
            pair[f"{kind}_paired_economic_minus_four_shaping_mean"] = (
                s[f"{kind}_economic_minus_four_shaping_mean"] - b[f"{kind}_economic_minus_four_shaping_mean"])
            for name in new:
                pair[f"{kind}_{name}_extra_penalty_mean"] = (
                    s[f"{kind}_{name}_penalty_mean"] - b[f"{kind}_{name}_penalty_mean"])
        paired.append(pair)
    write_csv(args.output_dir / "paired_reward_tradeoff_by_checkpoint.csv", paired)
    learned = [r for r in paired if r["episode"] > 0]
    summary["partial_objective_correlations_not_full_return"] = {
        kind: {axis: float(np.corrcoef([r[f"{kind}_paired_economic_minus_four_shaping_mean"] for r in learned],
                                      [r[axis] for r in learned])[0, 1])
               for axis in ("economic_improvement_percent", "X2_IAE_ratio", "F200_TV_ratio")}
        for kind in ("old", "recalibrated")}
    (args.output_dir / "reward_calibration.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
