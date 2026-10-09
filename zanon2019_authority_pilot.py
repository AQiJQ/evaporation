"""Fresh authority runs or read-only comparison; never expands to final tests.

Run training ONLY with --run. Default prepares/updates comparison from existing
results; absent results stay missing. No checkpoint loading/warm starts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .zanon2019_authority import PILOT_ALPHAS, VALIDATION_SEEDS, hierarchy_key
from .zanon2019_benchmark import write_csv
from .zanon2019_training_report import CERTIFICATION_STATUS
from .zanon2019_train import save_json

ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = ROOT / "evaporation_safe_sac"
DEFAULT_REPORT = DEFAULT_RUN_ROOT / "outputs_zanon2019_authority_comparison"
COMPARISON_FIELDS = ("best_episode", "empirical_safe", "economic_improvement_pct",
    "economic_degradation_pct", "economic_comparable_0p5pct", "economic_comparable_1pct",
    "worst_economic_improvement_pct", "X2_IAE_ratio", "P2_IAE_ratio", "X2_ISE_ratio",
    "P2_ISE_ratio", "G_X2_ratio", "G_P2_ratio", "X2_std_ratio", "P2_std_ratio",
    "minimum_X2_margin", "X2_margin_p1", "X2_margin_p5", "X2_margin_mean", "X2_margin_p50",
    "X2_margin_fraction_lt_0p05", "X2_margin_fraction_lt_0p10", "X2_margin_fraction_lt_0p20",
    "P100_TV_ratio", "F200_TV_ratio", "P100_RMS_du_ratio", "F200_RMS_du_ratio",
    "physical_violation", "input_violation", "QP_infeasible", "Omega_exit", "W_exceedance_rate",
    "raw_actor_saturation_fraction", "scaled_action_norm_mean", "residual_requested_norm_mean",
    "residual_applied_norm_mean", "residual_requested_applied_ratio", "authority_joint_candidate")


def table(path, rows, fields):
    """Empty evidence is a header-only table, never stale previous results."""
    keys = list(dict.fromkeys([*fields, *(k for r in rows for k in r)]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_directory(root, alpha):
    return Path(root) / f"outputs_zanon2019_alpha{int(round(alpha * 100)):03d}_seed42_100x1000"


def commands(root, device="cuda"):
    return [[sys.executable, "-m", "evaporation.zanon2019_train",
        "--disturbance-mode", "zanon2019_stochastic", "--authority-pilot",
        "--stochastic-residual-scale", str(alpha), "--episodes", "100",
        "--steps", "1000", "--seed", "42", "--eval-seeds", "10",
        "--eval-seed-start", "420000", "--eval-every", "5", "--device", device,
        "--output-dir", str(run_directory(root, alpha))] for alpha in PILOT_ALPHAS]


def fingerprint(manifest):
    """Only authority/output/derived selection fields may differ across pilots."""
    cfg = dict(manifest["frozen_experiment_config"])
    cfg.pop("stochastic_residual_scale", None)
    return json.dumps({k: manifest[k] for k in (
        "episodes", "steps", "seed", "mode", "observation_dim", "weights",
        "variance_F1_X1_T1_T200", "evaluation_seeds", "evaluation_every",
        "frozen_SAC_config", "frozen_geometry_sources")} | {"cfg": cfg}, sort_keys=True)


def flatten(alpha, folder, summary, best, name):
    row = {"alpha": alpha, "source_directory": str(folder), "selection": name,
        "run_state": summary["run_state"], "completed_episodes": summary["completed_episodes"],
        "early_stop": summary["run_state"] in ("aborted", "evaluation_aborted", "evaluation_setup_failed"),
        "certification_status": CERTIFICATION_STATUS, "available": best is not None}
    if best is not None:
        row.update({k: v for k, v in best.items() if not isinstance(v, (list, dict))})
        row["best_episode"] = best["episode"]
    else:
        row["reason"] = "no eligible post-warmup validation checkpoint"
    return row


def pareto(rows):
    good = [r for r in rows if r.get("empirical_safe") and r.get("post_warmup")
            and r.get("economic_comparable_0p5pct")]
    def vector(r):
        return np.array([-r["economic_improvement_pct"], -r["minimum_X2_margin"],
            -r["X2_margin_p1"], r["mean_IAE_ratio"], r["mean_ISE_ratio"],
            .5 * (r["G_X2_ratio"] + r["G_P2_ratio"]), r["mean_TV_ratio"],
            r["mean_RMS_du_ratio"], r["any_outer_10pct_fraction"]])
    for r in rows:
        r["authority_pareto_nondominated"] = r in good and not any(
            np.all(vector(t) <= vector(r) + 1e-10) and np.any(vector(t) < vector(r) - 1e-10)
            for t in good if t is not r)


def plots(rows, all_rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    available = [r for r in rows if r.get("available")]
    specs = (
        ("alpha_vs_economic_performance", ("economic_improvement_pct",), "Economic improvement (%)"),
        ("alpha_vs_X2_IAE", ("X2_IAE_ratio", "P2_IAE_ratio"), "IAE ratio"),
        ("alpha_vs_G_X2", ("G_X2_ratio", "G_P2_ratio"), "Empirical amplification ratio (not Hinf)"),
        ("alpha_vs_X2_margin", ("minimum_X2_margin", "X2_margin_p1", "X2_margin_p5"), "X2 margin"),
        ("alpha_vs_TV", ("P100_TV_ratio", "F200_TV_ratio", "P100_RMS_du_ratio", "F200_RMS_du_ratio"), "Control activity ratio"),
        ("alpha_vs_applied_residual", ("scaled_action_norm_mean", "residual_applied_norm_mean"), "Normalized action / residual magnitude"),
    )
    for filename, keys, ylabel in specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in keys:
            points = [r for r in available if r.get(key) is not None]
            if points:
                ax.plot([r["alpha"] for r in points], [r[key] for r in points], "o-", label=key)
        if available:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, "No eligible trained pilot results yet", ha="center", transform=ax.transAxes)
        ax.set(xlabel="stochastic residual authority alpha", ylabel=ylabel,
               title="Validation-selected checkpoints; missing runs are not zero")
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / f"{filename}.png", dpi=160)
        plt.close(fig)
    for filename, ykey, ylabel in (
        ("alpha_safety_economic_tradeoff", "minimum_X2_margin", "Minimum X2 margin"),
        ("alpha_disturbance_economic_tradeoff", "G_X2_ratio", "X2 empirical amplification ratio (not Hinf)")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for alpha in PILOT_ALPHAS:
            rs = [r for r in all_rows if r["alpha"] == alpha and r.get("post_warmup")]
            if rs:
                ax.scatter([r["economic_improvement_pct"] for r in rs], [r[ykey] for r in rs],
                           label=f"alpha={alpha}", alpha=.5)
        ax.axvline(-.5, color="gray", ls="--", label="0.5% degradation")
        ax.set(xlabel="Economic improvement (%)", ylabel=ylabel,
               title="All post-warmup validation checkpoints (including failures)")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out / f"{filename}.png", dpi=160)
        plt.close(fig)


def compare(root, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, all_rows, selections, fingerprints = [], [], [], []
    for alpha in PILOT_ALPHAS:
        folder = run_directory(root, alpha)
        path = folder / "run_summary.json"
        if not path.exists():
            rows.append({"alpha": alpha, "available": False, "run_state": "not_run",
                         "source_directory": str(folder), "reason": "training results missing"})
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if (summary.get("stochastic_residual_scale") != alpha or not summary.get("authority_pilot")
                or summary["evaluation_seeds"] != VALIDATION_SEEDS or summary["steps"] != 1000
                or summary["episodes"] != 100 or summary["seed"] != 42):
            raise ValueError(f"Not a matching fresh 100x1000 authority pilot: {folder}")
        fingerprints.append(fingerprint(summary))
        selected = summary["selected_checkpoints"]
        rows.append(flatten(alpha, folder, summary, selected.get("best_safety_economic_actor"),
                            "best_safety_economic_actor"))
        for name, best in selected.items():
            selections.append(flatten(alpha, folder, summary, best, name))
        with (folder / "checkpoint_pareto.csv").open(encoding="utf-8", newline="") as f:
            for raw in csv.DictReader(f):
                # Parse typed status scalars, never coerce missing data to zero.
                status = {}
                for key, value in raw.items():
                    if value in ("True", "False"):
                        status[key] = value == "True"
                    else:
                        try:
                            status[key] = float(value)
                        except (ValueError, TypeError):
                            status[key] = value or None
                status["alpha"] = alpha
                all_rows.append(status)
    if len(set(fingerprints)) > 1:
        raise ValueError("Pilots differ in frozen parameters / geometry / validation protocol")
    pareto(all_rows)
    table(out / "stochastic_authority_comparison.csv", rows, ("alpha", "available", *COMPARISON_FIELDS))
    table(out / "stochastic_authority_selected_checkpoints.csv", selections, ("alpha", "selection", *COMPARISON_FIELDS))
    table(out / "stochastic_authority_all_checkpoints.csv", all_rows, ("alpha", "episode", *COMPARISON_FIELDS))
    table(out / "stochastic_authority_pareto.csv", [r for r in all_rows if r["authority_pareto_nondominated"]],
          ("alpha", "episode", *COMPARISON_FIELDS))
    plots(rows, all_rows, out)
    completed = all(r.get("run_state") == "completed" and r.get("completed_episodes") == 100 for r in rows)
    resolved = all(r.get("run_state") in ("completed", "aborted", "evaluation_aborted") for r in rows)
    completed_alphas = {r["alpha"] for r in rows if r.get("run_state") == "completed"
                        and r.get("completed_episodes") == 100}
    # Search the full checkpoint frontier, not only the margin-first winner.
    # A different named checkpoint can be a better joint witness.
    eligible = [r for r in all_rows if r["alpha"] in completed_alphas
                and r.get("authority_joint_candidate") and r["authority_pareto_nondominated"]]
    representative = min(eligible, key=hierarchy_key) if resolved and eligible else None
    decision = {
        "certification_status": CERTIFICATION_STATUS, "final_test_performed": False,
        "frozen_config_consistency": True if len(fingerprints) >= 2 else None,
        "all_pilots_completed": completed,
        "all_pilots_resolved_including_early_stop": resolved,
        "sweet_spot_alpha": representative["alpha"] if representative else None,
        "candidate_episode": representative["episode"] if representative else None,
        "selection_basis": "hierarchical validation candidate; review full Pareto trade-off and occupancy before extending",
        "reason": "candidate for human review only" if representative else
            "unresolved pilots or no eligible completed comparable joint candidate; no authority recommendation",
        "automatically_extend_training": False, "reward_change_evidence": "not inferred automatically",
        "reference": "alpha=1 episode5 is failed 3-seed historical reference, not a 10-seed retrained pilot",
    }
    # Preserve actual historical checkpoint values, not text-rounded constants.
    ref = Path(root) / "outputs_zanon2019_stochastic_seed42_300x1000" / "checkpoint_pareto.csv"
    if ref.exists():
        with ref.open(encoding="utf-8", newline="") as f:
            reference = [dict(r, role="high_authority_failed_reference_not_selection_eligible")
                         for r in csv.DictReader(f) if float(r["episode"]) == 5]
        write_csv(out / "alpha1_failed_historical_reference.csv", reference)
    save_json(out / "authority_decision.json", decision)
    (out / "authority_comparison_summary.md").write_text(
        "# Stochastic residual authority pilot\n\n"
        + json.dumps(decision, indent=2) + "\n\n"
        + "Validation only: 420000..420009. Final seeds 430000..430049 remain unused.\n"
        + "G is empirical amplification, not Hinf. Baseline is zero residual, not the ECC2019 controller.\n"
        + "Protocol and hierarchical tie-breaking: implementation choice for reproduction; not specified in the paper.\n"
        + "Missing runs/checkpoints are unavailable, not zero. An aborted run is not a completed safe pilot.\n",
        encoding="utf-8")
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Explicitly start four fresh local pilots")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cmds = commands(args.run_root, args.device)
    # Check ALL targets before training any authority. Never resume or overwrite.
    if args.run:
        for alpha in PILOT_ALPHAS:
            folder = run_directory(args.run_root, alpha)
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError(f"Preserve existing evidence; select a fresh --run-root: {folder}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "pilot_commands.json", {"commands": cmds,
        "training_performed_by_this_call": args.run, "fresh_runs": True,
        "validation_seeds": VALIDATION_SEEDS, "never_auto_extend": True})
    if args.run:
        for alpha, cmd in zip(PILOT_ALPHAS, cmds):
            print(f"=== fresh authority pilot alpha={alpha} ===", flush=True)
            result = subprocess.run(cmd, cwd=ROOT.parent, check=False)
            folder = run_directory(args.run_root, alpha)
            if result.returncode:
                # Only a documented safety abort may proceed to a DIFFERENT
                # fresh alpha. Do not hide programming/runtime/setup errors.
                documented_safety_abort = (folder / "evaluation_abort.json").exists()
                training_abort = folder / "training_abort.json"
                if training_abort.exists():
                    reason = json.loads(training_abort.read_text(encoding="utf-8")).get("reason", "")
                    documented_safety_abort |= reason.startswith((
                        "ECC2019 empirical safety anomaly", "Interior-anchor training entered an uncertified",
                        "Omega training abort"))
                if not documented_safety_abort:
                    raise RuntimeError(f"Pilot alpha={alpha} failed without safety evidence")
                print(f"alpha={alpha}: stopped; evidence preserved; no resume", flush=True)
            compare(args.run_root, args.output_dir)
    print(json.dumps(compare(args.run_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
