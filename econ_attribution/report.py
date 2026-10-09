"""Scientific CSV/PNG outputs, with explicit partial-search decision limits."""
from __future__ import annotations
import json
import numpy as np
from . import core, study


def mean_rows(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def report():
    study.locked()
    root = study.ROOT
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .15})
    def csv(name):
        path = root / name
        return study.io.read_csv(path) if path.exists() else []
    def save(fig, name):
        fig.tight_layout(); fig.savefig(root / name, dpi=160); plt.close(fig)
    components = csv("economic_component_decomposition.csv")
    splits = csv("geometry_vs_sac_decomposition.csv")
    attrs = csv("actor_residual_economic_attribution.csv")
    geometry_summary = []
    for name in study.GEOMETRIES:
        rows = [r for r in splits if r["geometry"] == name]
        if len(rows) != 10:
            continue
        summary = dict(geometry=name, **{key: mean_rows(rows, key) for key in (
            "J_base_prod", "J_base_geometry", "J_SAC_geometry", "DeltaJ_geometry", "DeltaJ_SAC_given_geometry",
            "DeltaJ_total", "Gain_geometry_pct", "Gain_SAC_given_geometry_pct", "Gain_total_vs_prod_pct",
            "economic_win_fraction", "P100_TV", "F200_TV", "P100_RMS_du", "F200_RMS_du", "X2_std", "centered_X2_MAE")})
        summary["minimum_X2_margin"] = min(float(r["minimum_X2_margin"]) for r in rows)
        summary["component_gain"] = {c: mean_rows([r for r in components if r["geometry"] == name and r["component"] == c], "economic_gain_component") for c in core.COMPONENTS}
        geometry_summary.append(summary)
        if name in ("baseline", "C2_8"):
            # Full endpoints plus a separate signed component panel; no broken
            # axis or hidden truncated bars exaggerating a sub-0.1% difference.
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
            start = summary["J_base_prod"]
            changes = [-summary["component_gain"][c] for c in core.COMPONENTS]
            labels = ["Production baseline", *core.COMPONENTS, "Frozen SAC"]
            ends = [start]; positions = []
            current = start
            for change in changes:
                positions.append((min(current, current+change), abs(change)))
                current += change; ends.append(current)
            axes[0].bar(0, start, color="grey")
            for i, ((bottom, height), change) in enumerate(zip(positions, changes), 1):
                axes[0].bar(i, height, bottom=bottom, color="#d55e00" if change > 0 else "#009e73")
                axes[0].plot([i-1, i], [ends[i-1], ends[i-1]], color="grey", lw=.6)
            axes[0].bar(5, current, color="#0072b2")
            axes[0].set(ylim=(0, start*1.1), ylabel="Mean cumulative physical economic cost", title=f"{name}: full-scale cost waterfall")
            axes[0].set_xticks(range(6), labels, rotation=30, ha="right")
            gains = [summary["component_gain"][c] for c in core.COMPONENTS]
            axes[1].bar(core.COMPONENTS, gains, color=["#009e73" if x >= 0 else "#d55e00" for x in gains])
            axes[1].axhline(0, color="grey", lw=.7)
            axes[1].set(ylabel="Baseline minus SAC component cost", title="Separate signed contribution detail (positive = saving)")
            axes[1].tick_params(axis="x", rotation=25)
            save(fig, "economic_gain_component_waterfall.png" if name == "baseline" else "economic_gain_component_waterfall_C2_8.png")
    if geometry_summary:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(geometry_summary)); width = .24
        for offset, key, label in ((-1, "DeltaJ_geometry", "Geometry baseline change"),
                                  (0, "DeltaJ_SAC_given_geometry", "SAC given geometry"),
                                  (1, "DeltaJ_total", "Total vs production baseline")):
            ax.bar(x+offset*width, [r[key] for r in geometry_summary], width, label=label)
        ax.axhline(0, color="grey", lw=.8)
        ax.set_xticks(x, [r["geometry"] for r in geometry_summary])
        ax.set(ylabel="Mean absolute cost change (negative = saving)", title="Exact additive geometry / SAC decomposition")
        ax.legend(fontsize=8)
        save(fig, "geometry_vs_sac_economic_decomposition.png")
    production = [r for r in attrs if r["geometry"] == "baseline"]
    if production:
        for channel in ("P100", "F200"):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            x = np.array([float(r[f"applied_residual_{channel}"]) for r in production])
            y = np.array([float(r["g_t"]) for r in production])
            plot = ax.hexbin(x, y, gridsize=45, mincnt=1, cmap="viridis")
            fig.colorbar(plot, ax=ax, label="Step count")
            ax.axhline(0, color="grey", lw=.7); ax.axvline(0, color="grey", lw=.7)
            ax.set(xlabel=f"Local applied {channel} residual (physical units)", ylabel="Paired instantaneous economic advantage (%)",
                   title="Frozen Strong actor; correlation does not establish causation")
            save(fig, f"actor_{channel}_residual_vs_economic_advantage.png")
    certificates = csv("p100_certificate_sensitivity.csv")
    counterfactual = csv("p100_frozen_policy_counterfactual.csv")
    p100_summary = []
    for c in certificates:
        records = [r for r in counterfactual if r["geometry"] == c["candidate"] and r["status"] == "COMPLETE"]
        entry = dict(candidate=c["candidate"], factor=float(c["factor"]), certificate=c["certificate"], completed_paths=len(records))
        if len(records) == 10:
            entry.update(positive_P100_authority=float(c["P100_applied_residual_max"]),
                mean_gain_pct=mean_rows(records, "Gain_total_vs_prod_pct"),
                worst_gain_pct=min(float(r["Gain_total_vs_prod_pct"]) for r in records),
                win_fraction=mean_rows(records, "economic_win_fraction"),
                minimum_X2_margin=min(float(r["minimum_X2_margin"]) for r in records),
                **{k: mean_rows(records, k) for k in ("P100_TV", "F200_TV", "P100_RMS_du", "F200_RMS_du", "X2_std", "centered_X2_MAE")})
        p100_summary.append(entry)
    available = [r for r in p100_summary if "mean_gain_pct" in r]
    if available:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for r in available:
            ax.scatter(r["positive_P100_authority"], r["mean_gain_pct"])
            ax.annotate(str(r["factor"]), (r["positive_P100_authority"], r["mean_gain_pct"]), xytext=(4, 4), textcoords="offset points")
        ax.set(xlabel="Reference positive applied P100 authority (alpha=0.1)", ylabel="Mean economic gain vs production baseline (%)",
               title="Preregistered P100-only geometry sensitivity; no forced regression")
        save(fig, "p100_authority_vs_economic_improvement.png")
    witness_path = root / "state_dependent_witness_summary.json"
    witness = study.v2.load(witness_path) if witness_path.exists() else {"status": "NOT_RUN"}
    rows = csv("state_dependent_witness_results.csv")
    safe = [r for r in rows if r.get("safety_passed") == "True"]
    if safe:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        colors = ["#009e73" if r["control_quality_pass"] == "True" else "#d55e00" for r in safe]
        gains = [float(r["mean_improvement_pct"]) for r in safe]
        axes[0].scatter([float(r["X2_std"]) for r in safe], gains, c=colors, alpha=.6, s=18)
        axes[1].scatter([.5*(float(r["P100_TV_ratio"])+float(r["F200_TV_ratio"])) for r in safe], gains, c=colors, alpha=.6, s=18)
        axes[0].set(xlabel="X2 standard deviation", ylabel="Mean economic gain (%)")
        axes[1].set(xlabel="Mean P100/F200 TV ratio", ylabel="Mean economic gain (%)")
        fig.suptitle(f"Finite state-only family / {witness['status']} (green: existing TV/RMS band passes)")
        save(fig, "state_dependent_witness_economic_pareto.png")
        labels = ["Frozen SAC"]; values = [.08650902042768663]
        for key, label in (("best_safe", "Best found safe witness"), ("best_control_quality", "Best found quality witness")):
            if witness.get(key):
                labels.append(label); values.append(witness[key]["mean_improvement_pct"])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values)
        ax.set(ylabel="Mean economic gain vs production baseline (%)", title=f"{witness['status']}: development comparison, not independent generalization")
        save(fig, "state_dependent_witness_vs_sac.png")
    complete = witness.get("status") == "COMPLETE" and len(available) == sum(c["certificate"] == "CERTIFIED_PASS" for c in certificates)
    decision = dict(status="COMPLETE" if complete else "PARTIAL_PENDING_LOCAL_OFFLINE_SEARCH",
        geometry_summary=geometry_summary, p100_summary=p100_summary, witness_summary=witness,
        production_hashes_unchanged=study.locked()["source_and_production_hashes"] == study.protected(),
        no_training=True, final_seeds_used=False, commit=False, push=False,
        next_stage="Do not decide geometry vs learning before all preregistered candidates finish",
        unit="runtime model economic cost units; no unverified currency assigned",
        correlation_is_not_causation=True)
    if complete:
        best = witness.get("best_control_quality")
        improvement = max((r["mean_gain_pct"] for r in available), default=.08650902042768663)-.08650902042768663
        if best and best["mean_improvement_pct"] >= .20:
            decision["next_stage"] = "Learning headroom indicated within finite family; inspect critic/exploration/selection next, not automatic controller replacement"
        elif witness.get("best_safe") and witness["best_safe"]["mean_improvement_pct"] >= .20 and (not best or best["mean_improvement_pct"] < .20):
            decision["next_stage"] = "Economics-performance trade-off: higher raw gains do not pass existing control-quality band"
        else:
            decision["next_stage"] = "Report finite-family/geometry headroom without claiming a global upper bound; do not chase large gains by relaxing gates"
        decision["observed_P100_max_gain_increment_percentage_points"] = improvement
    study.save(root / "next_stage_decision_summary.json", decision)
    lines = ["# Economic attribution and finite state-only witness study", "", f"Status: {decision['status']}", "",
        "Physical cost and reward are separate. Cost uses nonlinear post-state and applied input.",
        "All percentages retain their specified denominators. Absolute geometry/SAC effects add exactly.",
        "The P100/steam label is indirect; 600 multiplies F100, not P100.",
        "Frozen actor comparisons include changed mapping and changed state trajectories; correlations are not causal proof.",
        "No Gaussian trajectory is labeled formally bounded-W certified.", "", "## Geometry comparison", "",
        "| Geometry | Geometry baseline gain % | SAC vs own baseline % | Total vs production % | Win fraction |", "|---|---:|---:|---:|---:|"]
    for r in geometry_summary:
        lines.append(f"|{r['geometry']}|{r['Gain_geometry_pct']:.8f}|{r['Gain_SAC_given_geometry_pct']:.8f}|{r['Gain_total_vs_prod_pct']:.8f}|{r['economic_win_fraction']:.4f}|")
    lines += ["", "## Physical component contributions (positive = saving)", ""]
    for r in geometry_summary:
        lines.append(f"{r['geometry']}: " + json.dumps(r["component_gain"]))
    lines += ["", "## P100-only scan", "", *[json.dumps(r) for r in p100_summary], "", "## Witness", "",
        json.dumps(witness, indent=2), "", decision["next_stage"], "",
        "No SAC training, no production changes, no final seeds, no commit/push. Existing Conservative partial results preserved."]
    (root / "ECONOMIC_ATTRIBUTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(decision["status"], decision["next_stage"], flush=True)
