"""Zero-residual ECC2019-motivated stochastic input and safety calibration.

This module does not train SAC or alter B/K/W/Z/S/Omega/Gm/Bj, reward,
observation, mapping, or QP. Gaussian iid sampling, 1 s sampling time,
initial balanced reference and seeds are implementation choices for
reproduction, not details specified by the ECC2019 paper. Sigma-as-std is
an explicitly separate sensitivity audit, not the main benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .controlled_invariant_error_set import facets, hull
from .control import point_in_convex_polygon
from .zanon2019_benchmark import (
    DEFAULT_DESIGN, DEFAULT_OMEGA, PAPER_VARIANCES_F1_X1_T1_T200,
    VARIABLE_NAMES, make_setup, paired_rollouts, sample_disturbance_path,
    trajectory_metrics, write_csv,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parent / (
    "evaporation_safe_sac/outputs_zanon2019_calibration"
)
LITERAL_STD = np.sqrt(PAPER_VARIANCES_F1_X1_T1_T200)
PERCENTILES = (1, 5, 50, 95, 99)


def scalar_stats(values: np.ndarray) -> dict[str, float]:
    a = np.asarray(values, dtype=float).ravel()
    if not len(a) or not np.all(np.isfinite(a)):
        raise ValueError("Statistics need finite nonempty samples")
    p = np.percentile(a, PERCENTILES)
    return {
        "mean": float(np.mean(a)), "variance": float(np.var(a)),
        "std": float(np.std(a)), "min": float(np.min(a)),
        "max": float(np.max(a)),
        **{f"p{q}": float(v) for q, v in zip(PERCENTILES, p)},
    }


def collect_one(cfg, model, design, omega, domain, mode, seed, steps,
                variances, active=None):
    scales = np.asarray(variances, dtype=float).copy()
    if active is not None:
        keep = VARIABLE_NAMES.index(active)
        scales[np.arange(4) != keep] = 0.0
    disturbance, metadata = sample_disturbance_path(
        cfg, mode, seed, steps, scales, 20, 50,
    )
    (stat, records), _ = paired_rollouts(
        cfg, model, design, omega, domain, None, disturbance, seed,
    )
    states = np.asarray([r["state"] for r in records], dtype=float)
    controls = np.asarray([r["control"] for r in records], dtype=float)
    w = np.asarray([r["w_hat"] for r in records], dtype=float)
    h_w, b_w = facets(hull(design.w_vertices))
    if np.any(b_w <= 0):
        raise RuntimeError("W facet utilization H_i w / h_i needs positive h_i")
    facet_values = w @ h_w.T
    facet_ratio = facet_values / b_w
    utilization = np.max(facet_ratio, axis=1)
    active_facet = np.argmax(facet_ratio, axis=1)
    facet_excess = np.max(facet_values - b_w, axis=1)
    # Fixed literal-variance normalization also in sensitivity runs, so the
    # empirical ratios remain comparable between the two interpretations.
    d_norm = (disturbance - cfg.disturbance_nominal) / LITERAL_STD
    d_energy = float(np.sum(d_norm ** 2))
    error = states - cfg.robust_economic_reference_state
    x2_energy = float(np.sum(error[:, 0] ** 2))
    p2_energy = float(np.sum(error[:, 1] ** 2))
    attenuation = {
        "G_X2_emp": float(np.sqrt(x2_energy / d_energy)) if d_energy else None,
        "G_P2_emp": float(np.sqrt(p2_energy / d_energy)) if d_energy else None,
        "normalized_disturbance_energy": d_energy,
        "X2_error_energy": x2_energy,
        "P2_error_energy": p2_energy,
        "normalizer": "literal-variance SD [sqrt(2),1,sqrt(8),sqrt(5)]",
    }
    metrics = trajectory_metrics(records, cfg, model, design)
    metrics.update({
        "seed": seed, "mode": mode, "active_component": active or "all_four",
        "X2_std": float(np.std(states[:, 0])),
        "P2_std": float(np.std(states[:, 1])),
        "X2_error_std": float(np.std(error[:, 0])),
        "P2_error_std": float(np.std(error[:, 1])),
        "X2_peak_deviation": float(np.max(np.abs(error[:, 0]))),
        "P2_peak_deviation": float(np.max(np.abs(error[:, 1]))),
        "P100_min": float(np.min(controls[:, 0])),
        "P100_max": float(np.max(controls[:, 0])),
        "F200_min": float(np.min(controls[:, 1])),
        "F200_max": float(np.max(controls[:, 1])),
        "W_mean_facet_utilization": float(np.mean(utilization)),
        "W_max_facet_utilization": float(np.max(utilization)),
        "W_p95_facet_utilization": float(np.percentile(utilization, 95)),
        "W_p99_facet_utilization": float(np.percentile(utilization, 99)),
        "W_max_facet_excess": float(np.max(facet_excess)),
        **{key: val for key, val in attenuation.items() if isinstance(val, (int, float))},
    })
    return {
        "seed": seed, "disturbance": disturbance, "metadata": metadata,
        "states": states, "controls": controls, "w": w,
        "facet_values": facet_values, "facet_ratio": facet_ratio,
        "facet_excess": facet_excess, "utilization": utilization,
        "active_facet": active_facet, "error": error,
        "d_norm": d_norm, "attenuation": attenuation,
        "metrics": metrics, "records": records,
    }


def seed_tables(run, group, cfg):
    seed, disturbance, states = run["seed"], run["disturbance"], run["states"]
    d_rows, x_rows = [], []
    for axis, name in enumerate(VARIABLE_NAMES):
        target_variance = float(PAPER_VARIANCES_F1_X1_T1_T200[axis])
        d_rows.append({
            "group": group, "seed": seed, "variable": name,
            "target_mean": float(cfg.disturbance_nominal[axis]),
            "target_variance_literal": target_variance,
            **scalar_stats(disturbance[:, axis]),
            "physical_clip_count": int(run["metadata"]["physical_clip_counts_F1_X1"][axis])
            if axis < 2 else 0,
        })
    for axis, name in enumerate(("X2", "P2")):
        reference = float(cfg.robust_economic_reference_state[axis])
        x_rows.append({
            "group": group, "seed": seed, "state": name,
            "reference": reference, **scalar_stats(states[:, axis]),
            "error_std": float(np.std(run["error"][:, axis])),
            "peak_deviation": float(np.max(np.abs(run["error"][:, axis]))),
            "IAE": float(np.sum(np.abs(run["error"][:, axis]))),
            "ISE": float(np.sum(run["error"][:, axis] ** 2)),
        })
    return d_rows, x_rows


def w_step_rows(run):
    result = []
    for step in range(len(run["states"])):
        row = {
            "seed": run["seed"], "step": step,
            "X2": float(run["states"][step, 0]),
            "P2": float(run["states"][step, 1]),
            "P100": float(run["controls"][step, 0]),
            "F200": float(run["controls"][step, 1]),
            **{name: float(run["disturbance"][step, i])
               for i, name in enumerate(VARIABLE_NAMES)},
            "w_normalized_X2": float(run["w"][step, 0]),
            "w_normalized_P2": float(run["w"][step, 1]),
            "active_facet_index_zero_based": int(run["active_facet"][step]),
            "maximum_facet_utilization": float(run["utilization"][step]),
            "maximum_facet_excess": float(run["facet_excess"][step]),
            "W_exceeded": bool(run["facet_excess"][step] > 1e-8),
        }
        for i in range(run["facet_values"].shape[1]):
            row[f"facet_{i}_H_w"] = float(run["facet_values"][step, i])
            row[f"facet_{i}_utilization"] = float(run["facet_ratio"][step, i])
        result.append(row)
    return result


def worst_sample(run):
    step = int(np.argmax(run["facet_excess"]))
    return {
        "seed": run["seed"], "step": step,
        "state_X2_P2": run["states"][step].tolist(),
        "input_P100_F200": run["controls"][step].tolist(),
        "disturbance_F1_X1_T1_T200": run["disturbance"][step].tolist(),
        "w_normalized": run["w"][step].tolist(),
        "active_facet_index_zero_based": int(run["active_facet"][step]),
        "facet_values": run["facet_values"][step].tolist(),
        "facet_utilizations": run["facet_ratio"][step].tolist(),
        "maximum_facet_utilization": float(run["utilization"][step]),
        "maximum_facet_excess": float(run["facet_excess"][step]),
    }


def summarize_runs(runs, name):
    m = [run["metrics"] for run in runs]
    keys = (
        "W_exceedance_count", "W_exceedance_rate", "W_max_facet_utilization",
        "X2_std", "P2_std", "X2_IAE", "P2_IAE", "G_X2_emp", "G_P2_emp",
        "X2_violation_count", "P2_physical_violation_count",
        "physical_state_violation_steps", "physical_input_violation_count",
        "Omega_exit_count", "local_RPI_Z_exit_count",
        "robust_region_violation_count", "QP_infeasible_count",
        "J_econ", "P100_TV", "F200_TV",
        "P100_delta_RMS", "F200_delta_RMS",
        "P100_upper_bound_steps", "F200_upper_bound_steps",
        "P100_outer_10pct_steps", "F200_outer_10pct_steps",
    )
    return {"group": name, "seeds": len(runs),
            **{f"{key}_mean": float(np.mean([row[key] for row in m]))
               for key in keys},
            "W_exceedance_rate_pooled": float(
                sum(row["W_exceedance_count"] for row in m)
                / sum(row["steps"] for row in m)),
            "W_max_facet_utilization_overall": float(max(
                row["W_max_facet_utilization"] for row in m)),
            "X2_violation_count_total": int(sum(
                row["X2_violation_count"] for row in m)),
            "physical_state_violation_steps_total": int(sum(
                row["physical_state_violation_steps"] for row in m)),
            "Omega_exit_count_total": int(sum(
                row["Omega_exit_count"] for row in m)),
            "QP_infeasible_count_total": int(sum(
                row["QP_infeasible_count"] for row in m))}


def plot_results(out, main_runs, cfg):
    import matplotlib.pyplot as plt

    def finish(fig, filename):
        fig.tight_layout()
        fig.savefig(out / filename, dpi=160)
        plt.close(fig)

    states = np.concatenate([r["states"] for r in main_runs])
    for axis, name in enumerate(("X2", "P2")):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(states[:, axis], bins=60, density=True, color="#3469a6", alpha=0.75)
        ax.axvline(cfg.robust_economic_reference_state[axis], color="black",
                   ls="--", label="balanced reference B")
        if axis == 0:
            ax.axvline(25, color="red", ls=":", label="physical lower bound")
        ax.set(xlabel=name, ylabel="empirical density",
               title=f"Zero-residual {name}: {len(main_runs)} stochastic seeds")
        ax.legend()
        finish(fig, f"calibration_{name}_distribution.png")
        fig, ax = plt.subplots(figsize=(10, 4))
        for run in main_runs:
            ax.plot(run["states"][:, axis], lw=0.6, alpha=0.5)
        ax.axhline(cfg.robust_economic_reference_state[axis], color="black",
                   ls="--", label="balanced reference B")
        if axis == 0:
            ax.axhline(25, color="red", ls=":", label="physical lower bound")
        ax.set(xlabel="step (1 s implementation choice)", ylabel=name,
               title=f"Zero-residual {name} overlay ({len(main_runs)} seeds)")
        ax.legend()
        finish(fig, f"calibration_{name}_20seed_overlay.png")
    util = np.concatenate([r["utilization"] for r in main_runs])
    fig, ax = plt.subplots(figsize=(8, 4))
    cutoff = float(np.percentile(util, 99.5))
    ax.hist(np.minimum(util, cutoff), bins=70, color="#a34439", alpha=0.8)
    ax.axvline(1.0, color="black", ls="--", label="W boundary")
    ax.set(xlabel=f"maximum W facet utilization (clipped at p99.5={cutoff:.2g})",
           ylabel="count", title="Fixed-W mismatch coverage")
    ax.legend()
    finish(fig, "W_facet_utilization_distribution.png")
    dnorm = np.concatenate([r["d_norm"] for r in main_runs])
    errors = states - cfg.robust_economic_reference_state
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(np.linalg.norm(dnorm, axis=1), np.abs(errors[:, 0]),
               s=3, alpha=0.12)
    ax.set(xlabel="same-step standardized disturbance magnitude",
           ylabel="absolute X2 error", title="Association only; not a causal gain")
    finish(fig, "disturbance_vs_X2_response.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    ix = np.arange(len(main_runs))
    ax.plot(ix, [r["attenuation"]["G_X2_emp"] for r in main_runs],
            "o-", ms=3, label="X2 empirical amplification")
    ax.plot(ix, [r["attenuation"]["G_P2_emp"] for r in main_runs],
            "o-", ms=3, label="P2 empirical amplification")
    ax.set(xlabel="evaluation seed index", ylabel="finite-trajectory L2 ratio",
           title="Not an H-infinity norm or induced-gain bound")
    ax.legend()
    finish(fig, "normalized_disturbance_attenuation.png")


def markdown_table(rows, keys):
    lines = ["| " + " | ".join(keys) + " |",
             "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        vals = []
        for key in keys:
            val = row.get(key)
            vals.append(f"{val:.6g}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def report(out, args, cfg, disturbance_rows, state_rows, main_runs,
           main_summary, sensitivity_summary, ablation_summaries, worst,
           w_normals, w_bounds):
    dist_pool, state_pool = [], []
    for axis, name in enumerate(VARIABLE_NAMES):
        vals = np.concatenate([r["disturbance"][:, axis] for r in main_runs])
        dist_pool.append({"variable": name,
                          "target_mean": float(cfg.disturbance_nominal[axis]),
                          "target_variance": float(PAPER_VARIANCES_F1_X1_T1_T200[axis]),
                          **scalar_stats(vals)})
    for axis, name in enumerate(("X2", "P2")):
        vals = np.concatenate([r["states"][:, axis] for r in main_runs])
        err = vals - cfg.robust_economic_reference_state[axis]
        state_pool.append({"state": name,
                           "reference": float(cfg.robust_economic_reference_state[axis]),
                           **scalar_stats(vals),
                           "error_std": float(np.std(err)),
                           "peak_deviation": float(np.max(np.abs(err))),
                           "IAE": float(np.sum(np.abs(err))),
                           "ISE": float(np.sum(err ** 2))})
    source = "C:/Users/cushy/Desktop/Practical_Reinforcement_Learning_of_Stabilizing_Economic_MPC.pdf"
    dominant = max(ablation_summaries[:4],
                   key=lambda row: row["W_exceedance_rate_pooled"])
    input_total = sum(r["attenuation"]["normalized_disturbance_energy"]
                      for r in main_runs)
    empirical_x2 = np.sqrt(sum(r["attenuation"]["X2_error_energy"]
                               for r in main_runs) / input_total)
    empirical_p2 = np.sqrt(sum(r["attenuation"]["P2_error_energy"]
                               for r in main_runs) / input_total)
    physical_total = sum(r["metrics"]["physical_state_violation_steps"]
                         for r in main_runs)
    qp_total = sum(r["metrics"]["QP_infeasible_count"] for r in main_runs)
    omega_total = sum(r["metrics"]["Omega_exit_count"] for r in main_runs)
    w_total = sum(r["metrics"]["W_exceedance_count"] for r in main_runs)
    status = "empirical_only_under_ECC2019_stochastic_disturbance" if w_total else (
        "empirical_only_under_unbounded_Gaussian_no_all_draws_certificate"
    )
    sections = [
        "# ECC2019-motivated stochastic evaporation calibration", "",
        f"Source: `{source}`, Numerical Example pp. 2261-2262; Fig. 2 p. 2263.",
        "", "## Scope and interpretation", "",
        f"The paper calls the four printed sigma values *variances*. Literal variance is the main mode. Gaussian iid sampling, balanced-B initial state, 1 s step, {args.steps}-step horizon, and seeds are **implementation choice for reproduction**; the distribution, temporal independence, initial state, sampling time and fixed evaluation realization are **not specified in the paper**. Sigma-as-std is sensitivity only.",
        "The stochastic path is passed unchanged into the existing nonlinear plant as `[F1,X1,T1,T200]`; the same path is verified in the resulting records. The plant's exogenous quantities enter `EvaporatorModel.algebraic` and `derivative`, then RK4 `step`. ECC2019 prose says flow F2 while its variance subscript says F1; the project model uses F1. No Fig. 2 curve matching or disturbance rescaling was performed.",
        "", "## Disturbance input calibration", "",
        markdown_table(dist_pool, ("variable", "target_mean", "mean", "target_variance", "variance", "std", "min", "max", "p1", "p5", "p50", "p95", "p99")),
        "", "## State response about balanced reference B", "",
        markdown_table(state_pool, ("state", "reference", "mean", "std", "error_std", "min", "max", "p1", "p5", "p50", "p95", "p99", "peak_deviation", "IAE", "ISE")),
        "", "The state and error standard deviations are numerically equal for a fixed reference, but both are reported to make the reference explicit. Smaller fluctuations than a visually inspected Fig. 2 do not establish superior disturbance rejection: controllers, realizations and initial conditions are not matched, and no exact statistics were digitized from that figure.",
        "", "## Empirical disturbance amplification", "",
        f"Pooled G_X2_emp={empirical_x2:.6g}, G_P2_emp={empirical_p2:.6g}. Input denominator uses the literal-variance standard deviations `[sqrt(2),1,sqrt(8),sqrt(5)]` for all groups. These are finite-trajectory ratios, **not** an H-infinity norm, induced L2 gain, or formal bound.",
        "", "## Control effort and economics", "",
        markdown_table([main_summary], ("group", "J_econ_mean", "P100_TV_mean", "F200_TV_mean", "P100_delta_RMS_mean", "F200_delta_RMS_mean", "P100_upper_bound_steps_mean", "F200_upper_bound_steps_mean", "P100_outer_10pct_steps_mean", "F200_outer_10pct_steps_mean")),
        "", "TV=0 is not evidence of good control without the corresponding bound occupancy. There is no matched ECC2019 RL-NMPC rollout, so neither stronger overall performance nor a direct economic gain against that paper can be claimed.",
        "", "## Fixed-W coverage and safety", "",
        f"W exceedance: {w_total}/{args.main_seeds * args.steps} steps; pooled rate={main_summary['W_exceedance_rate_pooled']:.6g}. Worst sample: seed {worst['seed']}, step {worst['step']}, facet {worst['active_facet_index_zero_based']}, utilization {worst['maximum_facet_utilization']:.6g}, excess {worst['maximum_facet_excess']:.6g} normalized state units. Full sample is in `w_worst_case_sample.json`; per-step facet values are in `w_facet_per_step.csv`.",
        f"Observed physical-state violation steps={physical_total}; QP infeasible steps={qp_total}; Omega exit steps={omega_total}. See `per_seed_baseline_metrics.csv` for X2/P2/input, robust-region, Z, and violation magnitudes.",
        "`Z exit != physical violation`; `W exceedance != physical violation`; `Omega exit != automatically physical violation`; `zero observed physical violation != formal guarantee`.",
        f"Certification status: `{status}`. Gaussian support is unbounded, and the original formal certificate applies only under its e0-in-Z / w-in-W assumptions (plus separate bounded finite-jump/minimum-dwell conditions for Experiment III). No all-Gaussian formal safety guarantee is claimed.",
        "", "## Disturbance component ablation", "",
        markdown_table(ablation_summaries, ("group", "W_exceedance_rate_pooled", "W_max_facet_utilization_overall", "X2_std_mean", "P2_std_mean", "X2_IAE_mean", "P2_IAE_mean", "G_X2_emp_mean", "G_P2_emp_mean", "physical_state_violation_steps_total", "Omega_exit_count_total", "QP_infeasible_count_total")),
        "", f"Largest *isolated* W exceedance rate: `{dominant['group']}`. This is a one-at-a-time diagnostic, not an additive causal decomposition of the all-four case.",
        "", "## Sigma interpretation sensitivity", "",
        markdown_table([main_summary, sensitivity_summary], ("group", "W_exceedance_rate_pooled", "X2_std_mean", "P2_std_mean", "G_X2_emp_mean", "G_P2_emp_mean", "physical_state_violation_steps_total", "Omega_exit_count_total")),
        "", "## Decision", "",
        "Experiment I is suitable as an **empirical stochastic training/evaluation environment** only if physical/QP/Omega abort behavior remains acceptable; it is **not suitable for claims of formal safety under paper-scale Gaussian disturbance** with frozen W. Do not start formal SAC training on the assumption that the existing W certificate covers these draws. No implementation bug was inferred solely from W exceedance; it can arise because the paper-scale exogenous uncertainty lies outside the fixed certified domain. Original controller/safety code remains unchanged.",
        "", "Experiment II (single state jump) and Experiment III (unknown-time repeated jumps with D=20) remain separate and unchanged.", "",
    ]
    (out / "zanon2019_calibration_report.md").write_text(
        "\n".join(sections), encoding="utf-8",
    )
    return status, dist_pool, state_pool


def audit_existing_w_membership(out: Path, design) -> dict:
    """Diagnose, but do not change, legacy polygon-membership tolerance."""
    path = out / "w_facet_per_step.csv"
    per_seed = {}
    worst_relative = None
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            seed = int(row["seed"])
            w = np.array([float(row["w_normalized_X2"]),
                          float(row["w_normalized_P2"])])
            strict = float(row["maximum_facet_excess"]) > 1e-8
            legacy = not point_in_convex_polygon(
                w, design.w_vertices, tol=1e-8)
            counts = per_seed.setdefault(seed, {
                "seed": seed, "steps": 0, "strict_facet_outside": 0,
                "legacy_cross_product_outside": 0,
                "strict_outside_legacy_inside": 0,
            })
            counts["steps"] += 1
            counts["strict_facet_outside"] += int(strict)
            counts["legacy_cross_product_outside"] += int(legacy)
            counts["strict_outside_legacy_inside"] += int(strict and not legacy)
            if (worst_relative is None or float(row["maximum_facet_utilization"])
                    > worst_relative["maximum_facet_utilization"]):
                worst_relative = {
                    "seed": seed, "step": int(row["step"]),
                    "state_X2_P2": [float(row["X2"]), float(row["P2"])],
                    "input_P100_F200": [float(row["P100"]), float(row["F200"])],
                    "disturbance_F1_X1_T1_T200": [float(row[name])
                                                   for name in VARIABLE_NAMES],
                    "w_normalized": w.tolist(),
                    "active_facet_index_zero_based": int(
                        row["active_facet_index_zero_based"]),
                    "maximum_facet_utilization": float(
                        row["maximum_facet_utilization"]),
                    "maximum_facet_excess": float(row["maximum_facet_excess"]),
                }
    rows = list(per_seed.values())
    write_csv(out / "legacy_vs_facet_w_audit.csv", rows)
    totals = {key: int(sum(row[key] for row in rows)) for key in (
        "steps", "strict_facet_outside", "legacy_cross_product_outside",
        "strict_outside_legacy_inside")}
    totals["diagnosis"] = (
        "Legacy point_in_convex_polygon compares raw cross products to an "
        "absolute 1e-8 tolerance; for the tiny W, this can undercount outside "
        "samples. Strict audit uses unit-normal facet excess >1e-8. "
        "Frozen controller/geometry code was not modified."
    )
    (out / "w_worst_relative_utilization_sample.json").write_text(
        json.dumps(worst_relative, indent=2), encoding="utf-8")
    summary_path = out / "calibration_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        data["legacy_vs_normalized_W_membership"] = totals
        data["worst_relative_W_utilization_sample"] = worst_relative
        summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    report_path = out / "zanon2019_calibration_report.md"
    if report_path.exists():
        body = report_path.read_text(encoding="utf-8")
        insertion = (
            f"Legacy cross-product membership flags {totals['legacy_cross_product_outside']}/"
            f"{totals['steps']} while unit-normal facet membership flags "
            f"{totals['strict_facet_outside']}/{totals['steps']}; "
            f"{totals['strict_outside_legacy_inside']} samples are outside by "
            "the strict test but inside by the legacy tolerance. This is a "
            "**diagnostic undercount**, not a change to the frozen controller "
            "or its sets. See `legacy_vs_facet_w_audit.csv`.\n\n"
            f"Largest relative W facet utilization is "
            f"{worst_relative['maximum_facet_utilization']:.6g} at seed "
            f"{worst_relative['seed']}, step {worst_relative['step']}; "
            "this differs from the worst *absolute* facet excess sample. "
            "See `w_worst_relative_utilization_sample.json`.\n\n"
        )
        marker = "## Disturbance component ablation"
        if "Legacy cross-product membership flags" not in body:
            body = body.replace(marker, insertion + marker)
            report_path.write_text(body, encoding="utf-8")
        elif "Largest relative W facet utilization" not in body:
            relative_line = (
                f"Largest relative W facet utilization is "
                f"{worst_relative['maximum_facet_utilization']:.6g} at seed "
                f"{worst_relative['seed']}, step {worst_relative['step']}; "
                "this differs from the worst *absolute* facet excess sample. "
                "See `w_worst_relative_utilization_sample.json`.\n\n"
            )
            report_path.write_text(body.replace(marker, relative_line + marker),
                                   encoding="utf-8")
    return totals


def add_matched_sensitivity(out: Path, cfg, requested_seeds: int,
                            seed_start: int, steps: int) -> None:
    """Compare the same five seeds and audit actual sigma-as-std input draws."""
    with (out / "per_seed_baseline_metrics.csv").open(
            newline="", encoding="utf-8") as stream:
        main_rows = {int(row["seed"]): row for row in csv.DictReader(stream)}
    with (out / "sigma_interpretation_sensitivity.csv").open(
            newline="", encoding="utf-8") as stream:
        sigma_rows = {int(row["seed"]): row for row in csv.DictReader(stream)
                      if row.get("mode") == "zanon2019_sigma_as_std_audit"}
    seeds = sorted(set(main_rows) & set(sigma_rows))[:requested_seeds]
    if not seeds:
        raise RuntimeError("No matching literal-variance/sigma-as-std seeds")
    steps = int(main_rows[seeds[0]]["steps"])
    metrics = (
        "W_exceedance_rate", "X2_std", "P2_std", "X2_IAE", "P2_IAE",
        "G_X2_emp", "G_P2_emp", "P100_TV", "F200_TV",
        "physical_state_violation_steps", "Omega_exit_count", "QP_infeasible_count",
    )
    matched = []
    for name, source in (("literal_variance_matched", main_rows),
                         ("sigma_as_std_sensitivity", sigma_rows)):
        matched.append({"mode": name, "matched_seed_count": len(seeds),
                        "seed_start": seeds[0],
                        **{f"{key}_mean": float(np.mean([
                            float(source[seed][key]) for seed in seeds]))
                           for key in metrics}})
    write_csv(out / "sigma_interpretation_matched_seeds.csv", matched)
    paths = []
    for seed in seeds:
        path, _ = sample_disturbance_path(
            cfg, "zanon2019_sigma_as_std_audit", seed, steps,
            PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
        paths.append(path)
    draws = np.concatenate(paths)
    input_rows = []
    for axis, name in enumerate(VARIABLE_NAMES):
        input_rows.append({"variable": name,
                           "target_mean": float(cfg.disturbance_nominal[axis]),
                           "sigma_as_std_target_variance": float(
                               PAPER_VARIANCES_F1_X1_T1_T200[axis] ** 2),
                           **scalar_stats(draws[:, axis])})
    write_csv(out / "sigma_as_std_disturbance_statistics.csv", input_rows)
    section = (
        f"Matched-seed comparison (same {len(seeds)} disturbance seeds in both "
        "interpretations; no SAC training):\n\n"
        + markdown_table(matched, (
            "mode", "W_exceedance_rate_mean", "X2_std_mean", "P2_std_mean",
            "G_X2_emp_mean", "G_P2_emp_mean", "P100_TV_mean", "F200_TV_mean",
            "physical_state_violation_steps_mean", "Omega_exit_count_mean"))
        + "\n\nThe sigma-as-std input means and variances are saved in "
          "`sigma_as_std_disturbance_statistics.csv`. The older summary "
          "table below may use different seed counts and should not be treated as a "
          "paired sensitivity estimate.\n\n"
    )
    report_path = out / "zanon2019_calibration_report.md"
    if report_path.exists():
        body = report_path.read_text(encoding="utf-8")
        marker = "## Sigma interpretation sensitivity\n\n"
        if "Matched-seed comparison" not in body:
            body = body.replace(marker, marker + section)
            report_path.write_text(body, encoding="utf-8")
    summary_path = out / "calibration_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["sigma_interpretation_matched_seeds"] = matched
        summary["sigma_as_std_pooled_disturbance_statistics"] = input_rows
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-seeds", type=int, default=20)
    parser.add_argument("--ablation-seeds", type=int, default=10)
    parser.add_argument("--sensitivity-seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=420000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--omega-vertices", type=Path, default=DEFAULT_OMEGA)
    parser.add_argument("--audit-existing-w", action="store_true",
                        help="Recheck existing per-step W rows without rerunning plant")
    args = parser.parse_args()
    if args.audit_existing_w:
        cfg, model, design, omega, domain, _ = make_setup(
            1, args.seed_start, args.design, args.omega_vertices)
        print(json.dumps(audit_existing_w_membership(args.output_dir, design),
                         indent=2), flush=True)
        add_matched_sensitivity(args.output_dir, cfg, args.sensitivity_seeds,
                                args.seed_start, args.steps)
        return
    if min(args.main_seeds, args.ablation_seeds,
           args.sensitivity_seeds, args.steps) < 1:
        raise ValueError("All seed counts and steps must be positive")
    if (args.output_dir / "zanon2019_calibration_report.md").exists():
        raise RuntimeError("Completed calibration exists; choose a new --output-dir")
    cfg, model, design, omega, domain, _ = make_setup(
        args.steps, args.seed_start, args.design, args.omega_vertices,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    h_w, b_w = facets(hull(design.w_vertices))
    (args.output_dir / "W_facet_geometry.json").write_text(json.dumps({
        "coordinate": "normalized state mismatch",
        "W_vertices": design.w_vertices.tolist(),
        "H_unit_norm_rows": h_w.tolist(), "h": b_w.tolist(),
        "membership_tolerance_absolute": 1e-8,
    }, indent=2), encoding="utf-8")
    main_runs, d_rows, x_rows, attenuation_rows = [], [], [], []
    controls, w_rows, all_metrics = [], [], []
    worst = None
    for index in range(args.main_seeds):
        seed = args.seed_start + index
        run = collect_one(cfg, model, design, omega, domain,
                          "zanon2019_stochastic", seed, args.steps,
                          PAPER_VARIANCES_F1_X1_T1_T200)
        main_runs.append(run)
        d, x = seed_tables(run, "literal_variance", cfg)
        d_rows.extend(d); x_rows.extend(x)
        attenuation_rows.append({"group": "literal_variance", "seed": seed,
                                 **run["attenuation"]})
        m = run["metrics"]
        all_metrics.append(m)
        controls.extend({
            "group": "literal_variance", "seed": seed, "input": name,
            "TV": m[f"{name}_TV"], "RMS_delta_u": m[f"{name}_delta_RMS"],
            "sign_change_count": m[f"{name}_input_sign_changes"],
            "min": m[f"{name}_min"], "max": m[f"{name}_max"],
            "lower_bound_steps": m[f"{name}_lower_bound_steps"],
            "upper_bound_steps": m[f"{name}_upper_bound_steps"],
            "outer_10pct_steps": m[f"{name}_outer_10pct_steps"],
        } for name in ("P100", "F200"))
        w_rows.append({"group": "literal_variance", "seed": seed,
                       **{key: m[key] for key in (
                           "W_exceedance_count", "W_exceedance_rate",
                           "W_mean_facet_utilization", "W_max_facet_utilization",
                           "W_p95_facet_utilization", "W_p99_facet_utilization",
                           "W_max_facet_excess")}})
        sample = worst_sample(run)
        if worst is None or sample["maximum_facet_excess"] > worst["maximum_facet_excess"]:
            worst = sample
        write_csv(args.output_dir / "w_facet_per_step.csv",
                  [row for item in main_runs for row in w_step_rows(item)])
        for filename, rows in (
            ("disturbance_statistics.csv", d_rows),
            ("state_statistics.csv", x_rows),
            ("disturbance_attenuation_metrics.csv", attenuation_rows),
            ("control_activity_statistics.csv", controls),
            ("w_exceedance_statistics.csv", w_rows),
            ("per_seed_baseline_metrics.csv", all_metrics),
        ):
            write_csv(args.output_dir / filename, rows)
        print(f"main seed={seed} W={m['W_exceedance_count']}/{args.steps} "
              f"X2_violation={m['X2_violation_count']} QP={m['QP_infeasible_count']}",
              flush=True)
        # The 1000-step record dictionaries are not needed after extraction.
        run.pop("records")
    main_summary = summarize_runs(main_runs, f"literal_variance_{args.main_seeds}seed")
    sensitivity = []
    for index in range(args.sensitivity_seeds):
        seed = args.seed_start + index
        run = collect_one(cfg, model, design, omega, domain,
                          "zanon2019_sigma_as_std_audit", seed, args.steps,
                          PAPER_VARIANCES_F1_X1_T1_T200)
        sensitivity.append(run)
        print(f"sigma-as-std seed={seed} W={run['metrics']['W_exceedance_count']}/{args.steps}",
              flush=True)
        run.pop("records")
    sensitivity_summary = summarize_runs(
        sensitivity, f"sigma_as_std_audit_{args.sensitivity_seeds}seed")
    write_csv(args.output_dir / "sigma_interpretation_sensitivity.csv",
              [r["metrics"] for r in sensitivity] +
              [main_summary, sensitivity_summary])
    ablation_summaries, ablation_rows = [], []
    for component in (*VARIABLE_NAMES, "all_four"):
        group_runs = []
        for index in range(args.ablation_seeds):
            seed = args.seed_start + index
            run = (
                main_runs[index]
                if component == "all_four" and index < len(main_runs)
                else collect_one(cfg, model, design, omega, domain,
                                 "zanon2019_stochastic", seed, args.steps,
                                 PAPER_VARIANCES_F1_X1_T1_T200,
                                 None if component == "all_four" else component)
            )
            group_runs.append(run)
            ablation_rows.append(run["metrics"])
            print(f"ablation {component} seed={seed} "
                  f"W={run['metrics']['W_exceedance_count']}/{args.steps}", flush=True)
            run.pop("records")
        ablation_summaries.append(summarize_runs(group_runs, component))
    write_csv(args.output_dir / "disturbance_ablation.csv",
              ablation_rows + ablation_summaries)
    (args.output_dir / "w_worst_case_sample.json").write_text(
        json.dumps(worst, indent=2), encoding="utf-8",
    )
    plot_results(args.output_dir, main_runs, cfg)
    certification_status, dist_pool, state_pool = report(
        args.output_dir, args, cfg, d_rows, x_rows, main_runs,
        main_summary, sensitivity_summary, ablation_summaries,
        worst, h_w, b_w,
    )
    (args.output_dir / "calibration_summary.json").write_text(json.dumps({
        "paper_source": "ECC2019 Numerical Example and Fig. 2",
        "main_disturbance": "implementation choice for reproduction: iid Gaussian with literal paper variances",
        "not_specified_in_paper": ["Gaussian distribution", "per-step independence",
                                   "initial state", "sampling time", "fixed seed"],
        "nominal_F1_X1_T1_T200": cfg.disturbance_nominal.tolist(),
        "paper_literal_variances_F1_X1_T1_T200": PAPER_VARIANCES_F1_X1_T1_T200.tolist(),
        "main_seed_count": args.main_seeds, "steps_per_seed": args.steps,
        "main_summary": main_summary,
        "sigma_as_std_sensitivity": sensitivity_summary,
        "ablation_summaries": ablation_summaries,
        "pooled_disturbance_statistics": dist_pool,
        "pooled_state_statistics": state_pool,
        "worst_W_sample": worst,
        "certification_status": certification_status,
        "formal_safety_guarantee_for_all_Gaussian_draws": False,
        "control_safety_design_changed": False,
    }, indent=2), encoding="utf-8")
    audit_existing_w_membership(args.output_dir, design)
    add_matched_sensitivity(args.output_dir, cfg, args.sensitivity_seeds,
                            args.seed_start, args.steps)
    print("calibration complete", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
