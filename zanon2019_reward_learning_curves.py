"""Plot actual logged training reward, without training or changing any source.

Returns are undiscounted sums of the replay reward. They do not include an
entropy bonus or audit-only RPI penalties. Random training paths make these
curves supplementary evidence, not a replacement for fixed paired evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "evaporation_safe_sac/final_stochastic_candidate_A"
WINDOW = 5
T95_DF2 = 4.302652729911275


def trailing_mean(values, window=WINDOW):
    return np.asarray([values[max(0, i-window+1):i+1].mean() for i in range(len(values))])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(root):
    lock_path = root / "locked_final_policies.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    data = []
    for policy in lock["policies"]:
        manifest = Path(policy["config_manifest"])
        if sha(manifest) != policy["config_manifest_sha256"]:
            raise ValueError("Locked manifest changed")
        source = manifest.parent / "training_log.csv"
        with source.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        episode = np.asarray([int(r["episode"]) for r in rows])
        returns = np.asarray([float(r["reward_return"]) for r in rows])
        steps = np.asarray([int(r["steps"]) for r in rows])
        if not np.array_equal(episode, np.arange(1, 101)) or not np.all(steps == 1000):
            raise ValueError(f"Expected complete 100x1000 log: {source}")
        if not np.isfinite(returns).all():
            raise ValueError("Missing/nonfinite return cannot be represented as zero")
        # Older seed42 logs have no decomposition. Do not impute missing
        # component columns; reward_return itself is the historical evidence.
        if "reward_replay_equivalent_total" in rows[0]:
            replay = np.asarray([float(r["reward_replay_equivalent_total"]) for r in rows])
            if not np.allclose(returns, replay, atol=1e-8, rtol=0):
                raise ValueError("Training return disagrees with replay-equivalent return")
        data.append(dict(seed=policy["training_seed"], episodes=episode, returns=returns,
                         steps=steps, smooth=trailing_mean(returns), source=str(source),
                         sha256=sha(source), manifest_sha256=sha(manifest),
                         warmup_steps=int(policy["fingerprint"]["config"]["warmup_steps"])))
    if [d["seed"] for d in data] != [42, 2027, 314159]:
        raise ValueError("Expected the three predeclared training seeds")
    return lock_path, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reward_learning_curves")
    args = parser.parse_args()
    lock_path, data = load(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = data[0]["episodes"]
    matrix = np.stack([d["returns"] for d in data])
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    half = T95_DF2 * std / np.sqrt(3)
    warmup_episodes = data[0]["warmup_steps"] / 1000
    if any(d["warmup_steps"] != data[0]["warmup_steps"] for d in data):
        raise ValueError("Warmup differs across seeds")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 240, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for color, d in zip(("#0072B2", "#D55E00", "#009E73"), data):
        axes[0].plot(episodes, d["returns"], color=color, alpha=.18, lw=.8)
        axes[0].plot(episodes, d["smooth"], color=color, lw=1.6,
                     label=f"Training seed {d['seed']} (trailing {WINDOW}-episode mean)")
    axes[0].set_title("Individual training seeds: faint raw returns, solid moving averages")
    axes[0].legend(fontsize=8, loc="best")
    axes[1].fill_between(episodes, mean-half, mean+half, color="#0072B2", alpha=.16,
                         label="Pointwise descriptive 95% t interval (n=3, df=2)")
    axes[1].plot(episodes, mean, color="#0072B2", lw=.9, alpha=.65, label="Raw across-seed mean")
    axes[1].plot(episodes, trailing_mean(mean), color="black", lw=1.7,
                 label=f"Across-seed mean, trailing {WINDOW}-episode average")
    axes[1].set_title("Across-training-seed mean; interval is for raw episodic return")
    axes[1].legend(fontsize=8, loc="best")
    for ax in axes:
        ax.axvline(warmup_episodes, color="#666666", ls="--", lw=1)
        ax.axhline(0, color="#999999", lw=.6)
        ax.set_ylabel("Episode reward sum (higher is better)")
        ax.set_xlim(1, 100)
        ax.grid(alpha=.2)
    axes[1].set_xlabel("Training episode (1000 steps per episode)")
    fig.suptitle("Stochastic Residual SAC: reward-based learning curves (alpha = 0.10)", fontsize=13)
    fig.text(.075, .018,
             "Recorded reward_return = sum of actual training rewards; entropy bonus and audit-only RPI penalties excluded.\n"
             "Vertical dash: 5000-step warmup boundary. Random training paths; no new training or final test.\n"
             "CI uses 3 training seeds, not pooled steps; descriptive only, not a simultaneous band or fixed-evaluation CI.", fontsize=8)
    fig.tight_layout(rect=(0, .105, 1, .955))
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"reward_learning_curves.{extension}")
    plt.close(fig)

    rows = []
    for i, ep in enumerate(episodes):
        row = dict(episode=int(ep), mean_reward_return=float(mean[i]),
                   sample_std=float(std[i]), CI95_lower=float(mean[i]-half[i]),
                   CI95_upper=float(mean[i]+half[i]),
                   mean_trailing_5=float(trailing_mean(mean)[i]))
        for d in data:
            row[f"seed{d['seed']}_reward_return"] = float(d["returns"][i])
            row[f"seed{d['seed']}_trailing_5"] = float(d["smooth"][i])
        rows.append(row)
    with (args.output_dir / "reward_learning_curves.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(reward_source="training_log.csv: reward_return",
        definition="undiscounted episode sum of actual replay reward; no entropy bonus",
        training_steps_per_episode=1000, alpha=0.10, moving_average_window=WINDOW,
        missing_component_policy="seed42 has no historical decomposition; no imputation",
        CI="mean +/- t(0.975,df=2)*sample_std/sqrt(3), pointwise raw episodic returns",
        warning="Supplementary randomized training evidence, not fixed paired evaluation or independent final test",
        policy_lock=str(lock_path), policy_lock_sha256=sha(lock_path),
        seeds=[dict(training_seed=d["seed"], source=d["source"], sha256=d["sha256"],
                    mean_first10=float(d["returns"][:10].mean()),
                    mean_last10=float(d["returns"][-10:].mean()),
                    episode100=float(d["returns"][-1])) for d in data])
    (args.output_dir / "reward_curve_provenance.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["seeds"], indent=2))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
