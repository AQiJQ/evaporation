"""Fig.2-style plots from locked validation archives; no rollout or training.

The ECC2019 three-panel layout is a visual reference, not an external numerical
comparison. Its naive/economic NMPC traces are NOT reproduced here. The middle
panel is our paired robust zero-residual baseline. Delta cost is explicitly
defined as SAC minus baseline (negative is favorable); this is an implementation
choice for reproduction, not a claimed paper-defined subtraction formula.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
ROOT = REPO / "evaporation_safe_sac/final_stochastic_candidate_A"
REPRESENTATIVE_SEED = 420000  # predeclared validation realization; no cherry-picking


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_archives(root):
    lock_path = root / "locked_final_policies.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if [p["training_seed"] for p in lock["policies"]] != [42, 2027, 314159]:
        raise ValueError("Expected all three locked training seeds in declared order")
    archives = []
    for policy in lock["policies"]:
        if not policy["eligible_final_checkpoint"]:
            raise ValueError("Cannot plot an ineligible policy as a locked candidate")
        for path_key, hash_key in (
            ("locked_checkpoint_path", "checkpoint_sha256"),
            ("config_manifest", "config_manifest_sha256"),
            ("validation_source", "validation_source_sha256"),
        ):
            if digest(policy[path_key]) != policy[hash_key]:
                raise ValueError(f"Locked source changed: {policy[path_key]}")
        run = Path(policy["config_manifest"]).parent
        source = (run / "evaluation_trajectories" /
                  f"episode_{policy['selected_episode']:04d}" /
                  f"seed_{REPRESENTATIVE_SEED}.csv")
        with source.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        def column(key):
            return np.asarray([float(r[key]) for r in rows])
        steps = column("step")
        if len(rows) != 1000 or not np.array_equal(steps, np.arange(1000)):
            raise ValueError(f"Incomplete or unordered trajectory: {source}")
        base_cost, sac_cost = column("baseline_economic_cost"), column("SAC_economic_cost")
        delta = sac_cost - base_cost
        if not np.allclose(delta, column("delta_l_SAC_minus_baseline"), atol=1e-8, rtol=0):
            raise ValueError("Archived instantaneous cost difference has inconsistent sign/value")
        if not np.allclose(np.cumsum(delta), column("cumulative_delta_SAC_minus_baseline"), atol=1e-7, rtol=0):
            raise ValueError("Archived cumulative cost difference is inconsistent")
        config = policy["fingerprint"]["config"]
        dt_s = float(config["dt_min"]) * 60
        data = dict(training_seed=policy["training_seed"],
                    episode=policy["selected_episode"], source=str(source),
                    source_sha256=digest(source), time=steps * dt_s,
                    baseline_X2=column("baseline_state_0"), SAC_X2=column("SAC_state_0"),
                    baseline_cost=base_cost, SAC_cost=sac_cost, delta_cost=delta,
                    disturbances=np.column_stack([column(k) for k in ("F1", "X1", "T1", "T200")]),
                    dt_s=dt_s, baseline_J=float(base_cost.sum()), SAC_J=float(sac_cost.sum()))
        numeric = np.column_stack([data[k] for k in ("time", "baseline_X2", "SAC_X2", "baseline_cost", "SAC_cost")])
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite trajectory data")
        if archives:
            for key in ("time", "baseline_X2", "baseline_cost", "disturbances"):
                if not np.array_equal(data[key], archives[0][key]):
                    raise ValueError(f"Cross-policy paired baseline/path differs: {key}")
        archives.append(data)
    return lock_path, archives


def figures(archives, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 240, "pdf.fonttype": 42})
    low = min(25., min(d[k].min() for d in archives for k in ("SAC_X2", "baseline_X2"))) - .025
    high = max(d[k].max() for d in archives for k in ("SAC_X2", "baseline_X2")) + .03
    max_delta = max(abs(d["delta_cost"]).max() for d in archives) * 1.08

    def panel(axes, data, paper_axis=False):
        t = data["time"]
        axes[0].plot(t, data["SAC_X2"], color="black", lw=.65, label="Locked Residual SAC")
        axes[1].plot(t, data["baseline_X2"], color="#0072B2", lw=.65,
                     label="Paired robust zero-residual baseline")
        for ax in axes[:2]:
            ax.axhline(25, color="black", ls="--", lw=1.6, label="Quality bound: X2 = 25")
            ax.set_ylim((24, 30) if paper_axis else (low, high))
            ax.set_ylabel(r"$X_2$ (%)")
            ax.legend(loc="upper right", fontsize=7, framealpha=.9)
        axes[2].plot(t, data["delta_cost"], color="black", lw=.6)
        axes[2].axhline(0, color="#777777", ls="--", lw=.8)
        axes[2].set_ylim(-max_delta, max_delta)
        axes[2].set_ylabel(r"$\ell_{\mathrm{SAC}}-\ell_{\mathrm{base}}$")
        axes[2].set_xlabel("Time (s)")
        for ax in axes:
            ax.set_xlim(0, 1000)
            ax.set_xticks(np.arange(0, 1001, 200))
            ax.grid(alpha=.18)
        gain = (data["baseline_J"] - data["SAC_J"]) / data["baseline_J"] * 100
        axes[0].set_title(f"Training seed {data['training_seed']}; episode {data['episode']}\n"
                          f"Realization economic improvement: {gain:+.5f}%", fontsize=10)

    def save(fig, stem):
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{extension}")
        plt.close(fig)

    for paper_axis in (False, True):
        fig, axes = plt.subplots(3, 1, figsize=(8, 8.8), sharex=True)
        panel(axes, archives[0], paper_axis)
        fig.suptitle("ECC2019 Fig.2-style view of our paired stochastic simulation", fontsize=12)
        fig.text(.08, .02,
                 "Validation disturbance seed 420000; alpha = 0.10; no independent final test.\n"
                 "Negative cost difference favors SAC. Baseline is ours, not ECC2019 naive NMPC.\n"
                 "Archived 1000 steps (0-999 s). " +
                 ("X2 axis 24-30 follows the paper-style scale." if paper_axis else "X2 axis zoomed; raw trajectories, no smoothing."),
                 fontsize=8)
        fig.tight_layout(rect=(0, .085, 1, .955))
        save(fig, "fig2_seed42_paper_axis" if paper_axis else "fig2_seed42_zoom")

    fig, axes = plt.subplots(3, 3, figsize=(16, 9), sharex=True, sharey="row")
    for col, data in enumerate(archives):
        panel(axes[:, col], data)
    fig.suptitle("Locked three-training-seed policies: ECC2019 Fig.2-style paired validation", fontsize=13)
    fig.text(.055, .022,
             "Common disturbance realization 420000 and identical baseline; alpha = 0.10. "
             "Negative cost difference favors SAC.\n"
             "Our zero-residual baseline is not an authors' NMPC trajectory. Validation only; "
             "raw 1000-step archives, zoomed common X2 axes, no smoothing.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .95))
    save(fig, "fig2_three_training_seeds_zoom")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "fig2_style_validation")
    args = parser.parse_args()
    lock_path, archives = read_archives(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures(archives, args.output_dir)
    exported = []
    for data in archives:
        for k, t in enumerate(data["time"]):
            exported.append(dict(training_seed=data["training_seed"], episode=data["episode"],
                disturbance_seed=REPRESENTATIVE_SEED, time_s=float(t),
                X2_SAC=float(data["SAC_X2"][k]), X2_baseline=float(data["baseline_X2"][k]),
                cost_SAC=float(data["SAC_cost"][k]), cost_baseline=float(data["baseline_cost"][k]),
                delta_cost_SAC_minus_baseline=float(data["delta_cost"][k])))
    with (args.output_dir / "fig2_plotted_data.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader(); writer.writerows(exported)
    summary = dict(scope="validation only; not independent final test",
        layout_reference="ECC2019 Fig.2; visual layout only, no external numeric comparison",
        baseline_identity="our paired robust zero-residual controller, NOT naive/economic NMPC",
        delta_cost_definition="stored economic_cost_SAC - stored economic_cost_baseline; negative favors SAC",
        sign_provenance="explicit implementation choice for reproduction; no claimed paper subtraction formula",
        cost_convention="original archived economic stage costs; no shaping or reward substituted",
        state_convention="archived pre-step X2, steps 0...999; terminal state not present in this plot",
        realization_selection="predeclared first validation seed 420000; not selected for best performance",
        smoothing="none", source_policy_lock=str(lock_path), source_policy_lock_sha256=digest(lock_path),
        trajectories=[dict(training_seed=d["training_seed"], episode=d["episode"],
            source=d["source"], source_sha256=d["source_sha256"], sampling_time_s=d["dt_s"],
            steps=len(d["time"]), baseline_J_econ=d["baseline_J"], SAC_J_econ=d["SAC_J"],
            economic_improvement_pct=(d["baseline_J"]-d["SAC_J"])/d["baseline_J"]*100,
            sum_delta_cost=float(d["delta_cost"].sum())) for d in archives])
    (args.output_dir / "fig2_provenance.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["trajectories"], indent=2))
    print(f"Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
