"""Pareto analysis of independently certified evaporator references.

The input CSV files come from ``certified_reference_search``.  Only candidates
whose complete design passed every original safety gate enter the frontier.
This module never trains SAC or changes the design gates.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .control import estimate_hinf_norm, point_in_convex_polygon, spectral_radius
from .config import ExperimentConfig


GATES = (
    "physical_state_feasible", "physical_input_feasible", "hinf_feasible",
    "rpi_feasible", "x_tightening_feasible", "u_tightening_feasible",
    "reference_in_tightened_state_set", "invariant_feasible",
    "reference_in_S", "s_plus_z_contained", "input_tube_contained",
    "disturbance_bound_audit_passed", "verification_qp_feasible",
    "residual_authority_feasible", "fully_certified",
)
OBJECTIVES = (
    ("cost_increase_percent", "min", 1e-9),
    ("minimum_residual_authority", "max", 1e-8),
    ("input_margin_normalized", "max", 1e-11),
    ("S_margin_normalized", "max", 1e-11),
    ("hinf_margin", "max", 1e-7),
    ("rpi_width_X2", "min", 1e-9),
    ("rpi_width_P2", "min", 1e-9),
    ("rpi_area", "min", 1e-9),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent / "evaporation_safe_sac"
    parser.add_argument(
        "--sources", nargs="+", type=Path,
        default=[
            root / "outputs_certified_economic_reference_search_07faaa1",
            root / "outputs_certified_reference_margin_audit",
            root / "outputs_certified_reference_pareto_probes",
        ],
        help="Directories containing candidate CSV and candidate_designs/ NPZ.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "outputs_certified_reference_pareto",
    )
    parser.add_argument("--dense-hinf-points", type=int, default=262144)
    return parser.parse_args()


def _load(source: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (source / "reference_search_candidates.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for raw in csv.DictReader(stream):
            if not all(raw.get(gate) == "True" for gate in GATES):
                continue
            row: dict[str, object] = dict(raw)
            for key, value in raw.items():
                if key in GATES:
                    row[key] = True
                elif key not in {"stage", "gain_seed", "failure_reason", "design_file"}:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        pass
            row["source"] = str(source.resolve())
            row["design_path"] = str(
                (source / "candidate_designs" / raw["design_file"]).resolve()
            )
            if not Path(row["design_path"]).is_file():
                raise FileNotFoundError(row["design_path"])
            state = np.array([row["X2"], row["P2"]], dtype=float)
            control = np.array([
                row["steady_P100"], row["steady_F200"]
            ], dtype=float)
            s_lower = np.array([row["S_X2_lower"], row["S_P2_lower"]])
            s_upper = np.array([row["S_X2_upper"], row["S_P2_upper"]])
            u_lower = np.array([
                row["U_minus_KZ_P100_lower"],
                row["U_minus_KZ_F200_lower"],
            ])
            u_upper = np.array([
                row["U_minus_KZ_P100_upper"],
                row["U_minus_KZ_F200_upper"],
            ])
            row["S_lower_margin_by_state"] = (state - s_lower).tolist()
            row["S_upper_margin_by_state"] = (s_upper - state).tolist()
            row["input_lower_margin_by_control"] = (control - u_lower).tolist()
            row["input_upper_margin_by_control"] = (u_upper - control).tolist()
            row["S_margin"] = float(np.min(np.r_[state - s_lower, s_upper - state]))
            row["input_margin"] = float(np.min(np.r_[
                control - u_lower, u_upper - control
            ]))
            row["S_margin_normalized"] = float(np.min(np.r_[
                (state - s_lower) / [15.0, 20.0],
                (s_upper - state) / [15.0, 20.0],
            ]))
            row["input_margin_normalized"] = float(np.min(np.r_[
                (control - u_lower) / [100.0, 100.0],
                (u_upper - control) / [100.0, 100.0],
            ]))
            row["hinf_margin"] = float(
                row["hinf_gamma"] - row["hinf_sampled_norm"]
            )
            if (
                row["minimum_residual_authority"] < 0.05 - 1e-8
                or row["input_margin"] < -1e-8
                or row["S_margin"] < -1e-8
                or row["hinf_margin"] <= 0.0
            ):
                raise RuntimeError(f"Recorded certified row failed gates: {state}")
            rows.append(row)
    return rows


def _dominates(a: dict, b: dict) -> bool:
    strictly_better = False
    for key, sense, tolerance in OBJECTIVES:
        va, vb = float(a[key]), float(b[key])
        difference = (vb - va) if sense == "min" else (va - vb)
        if difference < -tolerance:
            return False
        strictly_better |= difference > tolerance
    return strictly_better


def _dense_audit(row: dict, frequency_points: int) -> dict:
    cfg = ExperimentConfig(benchmark_profile="zanon2016", experiment_mode="proposed")
    with np.load(row["design_path"]) as saved:
        a, b, k = saved["A"], saved["B"], saved["K"]
        w = saved["W_vertices"]
        z = saved["rpi_boundary"]
        z_ref, v_ref = saved["z_ref"], saved["v_ref"]
        s_lower, s_upper = saved["invariant_lower"], saved["invariant_upper"]
        x_lower, x_upper = saved["x_lower_tight"], saved["x_upper_tight"]
        u_lower, u_upper = saved["u_lower_tight"], saved["u_upper_tight"]
        acl = a + b @ k
        sampled = estimate_hinf_norm(
            a, b, k, cfg.hinf_q, cfg.hinf_r, points=frequency_points
        )
        rpi_failures = sum(
            not point_in_convex_polygon(acl @ vertex + disturbance, z, tol=2e-8)
            for vertex in z for disturbance in w
        )
        return {
            "spectral_radius": float(spectral_radius(acl)),
            "dense_hinf_points": frequency_points,
            "dense_hinf_norm": float(sampled),
            "dense_hinf_margin": float(cfg.hinf_gamma - sampled),
            "rpi_one_step_vertex_disturbance_failures": int(rpi_failures),
            "reference_in_S": bool(np.all(z_ref >= s_lower - 1e-10)
                                   and np.all(z_ref <= s_upper + 1e-10)),
            "S_subset_X_minus_Z": bool(np.all(s_lower >= x_lower - 1e-10)
                                       and np.all(s_upper <= x_upper + 1e-10)),
            "steady_input_in_U_minus_KZ": bool(
                np.all(v_ref >= u_lower - 1e-10)
                and np.all(v_ref <= u_upper + 1e-10)
            ),
            "K": k.tolist(),
        }


def _save_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "X2", "P2", "steady_P100", "steady_F200", "steady_cost",
        "cost_increase_percent", "minimum_residual_authority",
        "input_margin", "S_margin", "input_margin_normalized",
        "S_margin_normalized", "hinf_margin", "rpi_width_X2",
        "rpi_width_P2", "rpi_area", "spectral_radius", "pareto",
        "source", "design_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def _plot(path: Path, rows: list[dict], frontier: list[dict], selected: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    por = np.array([r["cost_increase_percent"] for r in rows], dtype=float)
    authority = np.array([r["minimum_residual_authority"] for r in rows])
    margin = np.array([r["input_margin_normalized"] for r in rows], dtype=float)
    front_ids = {id(row) for row in frontier}
    is_front = np.array([id(row) in front_ids for row in rows])
    scatter = axes[0].scatter(
        por, authority, c=margin, cmap="viridis", s=32,
        edgecolor=np.where(is_front, "black", "none"), linewidth=0.7,
    )
    axes[0].axhline(0.05, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.045, 0.055)
    axes[0].set_xlabel("PoR: steady cost increase vs Paper2016 optimum (%)")
    axes[0].set_ylabel("Minimum residual authority")
    axes[0].set_title("PoR vs authority: all sampled values are at 5% floor")
    fig.colorbar(scatter, ax=axes[0], label="Normalized input margin")
    axes[1].scatter(por, margin, c="0.72", s=25, label="certified")
    axes[1].scatter(
        [r["cost_increase_percent"] for r in frontier],
        [r["input_margin_normalized"] for r in frontier],
        facecolors="none", edgecolors="black", s=46, label="Pareto",
    )
    for label, row in selected.items():
        if row is None:
            continue
        short_label = label.split("_")[0]
        if short_label == "C" and label == "C_higher_authority":
            continue
        axes[1].annotate(
            short_label,
            (row["cost_increase_percent"], row["input_margin_normalized"]),
            xytext=(4, 4), textcoords="offset points",
        )
    axes[1].set_xlabel("PoR (%)")
    axes[1].set_ylabel("Minimum normalized tightened input margin")
    axes[1].set_title("Economic and input margin tradeoff")
    axes[1].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for source in args.sources for row in _load(source)]
    unique: dict[tuple[float, float], dict] = {}
    for row in rows:
        key = (round(row["X2"], 10), round(row["P2"], 10))
        if key not in unique or row["steady_cost"] < unique[key]["steady_cost"]:
            unique[key] = row
    rows = sorted(unique.values(), key=lambda row: row["steady_cost"])
    frontier = [
        row for row in rows
        if not any(_dominates(other, row) for other in rows if other is not row)
    ]
    for row in rows:
        row["pareto"] = row in frontier
    a = rows[0]
    # A is also on the Pareto front.  B is a representative with several
    # margins improved while its PoR remains within 0.02 percentage points.
    nearby = [
        row for row in frontier
        if row["cost_increase_percent"] <= a["cost_increase_percent"] + 0.02
        and row["input_margin_normalized"] >= a["input_margin_normalized"]
        and row["S_margin_normalized"] >= a["S_margin_normalized"]
        and row["hinf_margin"] >= a["hinf_margin"]
        and row["rpi_area"] <= a["rpi_area"] * 1.02
    ]
    b = max(
        nearby,
        key=lambda row: (
            (row["input_margin_normalized"] / a["input_margin_normalized"])
            + (row["S_margin_normalized"] / a["S_margin_normalized"])
            + min(row["hinf_margin"] / max(a["hinf_margin"], 1e-12), 20.0)
            - 0.5 * (row["cost_increase_percent"]
                     - a["cost_increase_percent"]) / 0.02
        ),
    ) if nearby else None
    higher_authority = [
        row for row in frontier
        if row["minimum_residual_authority"]
        >= a["minimum_residual_authority"] + 1e-6
    ]
    c = min(
        higher_authority, key=lambda row: row["cost_increase_percent"]
    ) if higher_authority else None
    # When authority is numerically flat, retain an additional *margin*
    # tradeoff for inspection.  It must never be labelled higher-authority.
    margin_alternatives = [
        row for row in frontier
        if row is not a and row is not b
        and row["cost_increase_percent"]
        <= a["cost_increase_percent"] + 0.02
        and row["input_margin_normalized"] >= 0.0001
        and row["S_margin_normalized"] > a["S_margin_normalized"]
        and row["rpi_area"] < a["rpi_area"]
    ]
    c_margin = min(
        margin_alternatives, key=lambda row: row["rpi_area"]
    ) if c is None and margin_alternatives else None
    selected = {
        "A_minimum_cost": a,
        "B_balanced": b,
        "C_higher_authority": c,
        "C_margin_tradeoff_not_higher_authority": c_margin,
    }
    for row in selected.values():
        if row is not None:
            row["dense_audit"] = _dense_audit(row, args.dense_hinf_points)
    _save_csv(args.output_dir / "all_certified_candidates.csv", rows)
    _save_csv(args.output_dir / "pareto_candidates.csv", frontier)
    _plot(args.output_dir / "por_vs_authority.png", rows, frontier, selected)
    summary = {
        "method": "Pareto filter over existing independently fully-certified references; no gates changed",
        "scope_note": "Sampled frontier of the evaluated references, not a global continuous-set optimum",
        "source_directories": [str(source.resolve()) for source in args.sources],
        "fully_certified_count": len(rows),
        "pareto_count": len(frontier),
        "authority_range": [
            float(min(row["minimum_residual_authority"] for row in rows)),
            float(max(row["minimum_residual_authority"] for row in rows)),
        ],
        "higher_authority_threshold": 1e-6,
        "higher_authority_found": c is not None,
        "authority_indistinguishable_from_0_05_within_1e_8": bool(all(
            abs(row["minimum_residual_authority"] - 0.05) <= 1e-8
            for row in rows
        )),
        "A_minimum_cost": a,
        "B_balanced": b,
        "C_higher_authority": c,
        "C_margin_tradeoff_not_higher_authority": c_margin,
        "pareto_candidates": frontier,
        "sac_training_performed": False,
    }
    with (args.output_dir / "pareto_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps({
        "fully_certified_count": len(rows),
        "pareto_count": len(frontier),
        "authority_range": summary["authority_range"],
        "A": [a["X2"], a["P2"]],
        "B": None if b is None else [b["X2"], b["P2"]],
        "C_higher_authority": None if c is None else [c["X2"], c["P2"]],
        "C_margin_tradeoff": (
            None if c_margin is None else [c_margin["X2"], c_margin["P2"]]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
