"""Explain completed evidence without altering preregistered search sources.

This supplementary report has its own source receipt. It does not select extra
geometry candidates, change witnesses, train, or turn partial search into a
headroom conclusion. Physical-cost tables exclude legacy helper reward fields.
"""
import json
import numpy as np
from .. import study


HELPER_FIELDS = (
    "economic_reward_component", "smoothness_regularization_component",
    "lagrangian_constraint_component", "total_training_reward", "regularizer_fraction",
    "regularizer_fraction_defined", "sum_abs_economic_reward", "smoothness_penalty_raw_total",
    "joint_v2",
)


def read(name):
    path = study.ROOT / name
    return study.io.read_csv(path) if path.exists() else []


def average(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def clean_physical_tables():
    # full_metrics(..., weight=0, dual=0) is ONLY a diagnostic metrics helper.
    # Its scalar "total_training_reward" is not the locked SAC replay reward.
    # Remove these unrequested helper outputs from physical-economics CSVs.
    for name in ("geometry_vs_sac_decomposition.csv", "p100_frozen_policy_counterfactual.csv"):
        rows = read(name)
        if rows:
            rows = [{k: v for k, v in r.items() if k not in HELPER_FIELDS} for r in rows]
            study.write_csv(study.ROOT / name, rows)
    study.save(study.ROOT / "metric_field_semantics.json", dict(
        source_sha256=study.io.digest(__file__), excluded_helper_fields=list(HELPER_FIELDS),
        reason="legacy full_metrics called at zero diagnostic weights; helper return is NOT actual frozen SAC training return",
        physical_economics="Only nonlinear post-state economic cost and its components enter attribution",
        receipt_JSON_caveat="per-path diagnostic receipts retain helper fields for provenance, not reward analysis",
        production_reward_unchanged=True))


def main():
    study.locked()
    clean_physical_tables()
    root = study.ROOT
    decision = study.v2.load(root / "next_stage_decision_summary.json")
    geometry = decision["geometry_summary"]
    components = read("economic_component_decomposition.csv")
    channels = read("economic_channel_decomposition.csv")
    behaviors = read("actor_behavior_summary.csv")
    usage = read("new_authority_actual_usage.csv")
    certs = read("p100_certificate_sensitivity.csv")
    cf = read("p100_frozen_policy_counterfactual.csv")
    # Stage-level flat CSV is written on completion, while per-path receipts
    # persist earlier. Expose genuine partial progress, not misleading zeros.
    if not cf:
        cf = [study.v2.load(p) for p in sorted((root / "p100_pairs").glob("*/seed_*_result.json"))]
    lines = ["# 经济归因与有限搜索：详细结果", "",
        f"状态：{decision['status']}。未完成的 witness 不计为失败，也不用于 headroom 结论。", "",
        "## 真实经济成本", "",
        "运行时代码：`model.py:108 / EvaporatorModel.economic_cost`。",
        "`ell = 10.09*F2 + 10.09*F3 + 600*F100 + 0.6*F200`。",
        "F3=50固定；P100通过T100/Q100/F100及F4/F2间接影响代价。600乘的是F100，不能标成600*P100。",
        "代码未给成本系数明确货币/时间单位，本报告使用模型成本单位。J是原有1000步stage-cost之和。",
        "成本在nonlinear next state、最终实际输入、当前扰动上计算。与SAC replay reward严格分开。",
        "旧metrics helper的零权重return并非锁定训练reward，已从主经济CSV移除；receipt中的对应字段仅留作诊断来源。", "",
        "## 各成本分量贡献", "",
        "下表为10条相同DEV路径平均。正数表示baseline-SAC，即节省；负数表示额外成本。", "",
        "|geometry|F2|F3|F100蒸汽|F200冷却|总节省|收益%|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in geometry:
        g = r["component_gain"]
        lines.append(f"|{r['geometry']}|{g['flow_F2']:.3f}|{g['recirculation_F3']:.3f}|{g['steam_F100']:.3f}|{g['cooling_F200']:.3f}|{-r['DeltaJ_total']:.3f}|{r['Gain_total_vs_prod_pct']:.8f}|")
    lines += ["", "Production的主要收益来自蒸汽成本下降，F2与F200成本上涨抵消一部分。F2是原经济代价项，不自行解释为产品销售收益。", "",
        "## geometry与SAC分开", "",
        "绝对量：DeltaJ_total = DeltaJ_geometry + DeltaJ_SAC_given_geometry。三种百分比有不同分母，不能直接相加。", "",
        "|geometry|baseline额外成本|SAC相对自身baseline节省|最终对production收益%|自身baseline收益%|瞬时经济占优比例|",
        "|---|---:|---:|---:|---:|---:|"]
    for r in geometry:
        lines.append(f"|{r['geometry']}|{r['DeltaJ_geometry']:.3f}|{-r['DeltaJ_SAC_given_geometry']:.3f}|{r['Gain_total_vs_prod_pct']:.8f}|{r['Gain_SAC_given_geometry_pct']:.8f}|{r['economic_win_fraction']:.2%}|")
    lines += ["", "C2/C2_3的自身zero-residual baseline已变贵。必须先扣除这部分，再比较actor效果。", "",
        "|geometry|baseline F2额外成本|baseline蒸汽额外成本|baseline冷却额外成本|",
        "|---|---:|---:|---:|"]
    for r in geometry:
        rows = [x for x in components if x["geometry"] == r["geometry"]]
        values = [average([x for x in rows if x["component"] == key], "geometry_baseline_component_change")
                  for key in ("flow_F2", "steam_F100", "cooling_F200")]
        lines.append(f"|{r['geometry']}|" + "|".join(f"{v:.3f}" for v in values) + "|")
    lines += ["", "## 精确代数 input/state channel identity", "",
        "固定当步disturbance时，当前模型的cost对post-state X2/P2和实际P100/F200为仿射函数，逐步恒等式已核对。",
        "这是代数分量拆解，不是完整动态因果或离线重新优化。P100直接项节省与state项损失不能混称净P100收益。", "",
        "|geometry|post-X2贡献|post-P2贡献|direct-P100贡献|direct-F200贡献|",
        "|---|---:|---:|---:|---:|"]
    # channels uses explicit comparison roles; do not combine the three effects.
    if channels:
        for r in geometry:
            rows = [x for x in channels if x["geometry"] == r["geometry"]]
            selected = [x for x in rows if x["effect"] == "total"]
            if selected:
                vals = [average([x for x in selected if x["channel"] == k], "gain_component")
                        for k in ("post_X2", "post_P2", "direct_P100", "direct_F200")]
                lines.append(f"|{r['geometry']}|" + "|".join(f"{v:.3f}" for v in vals) + "|")
    lines += ["", "完整channel数据见economic_channel_decomposition.csv。", "",
        "## actor方向与相关性", "",
        "local residual相对于同一个SAC当前state/nominal context的zero-action输入；与配对闭环baseline输入差是两种量。", "",
        "|channel|positive|negative|zero|mean|median|p5|p25|p75|p95|Pearson(g)|Spearman(g)|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in behaviors:
        if r["geometry"] == "baseline" and r["group"] == "all":
            lines.append(f"|{r['channel']}|{float(r['positive_fraction']):.2%}|{float(r['negative_fraction']):.2%}|{float(r['zero_fraction']):.2%}|" +
                "|".join(f"{float(r[k]):.5f}" for k in ("mean", "median", "p5", "p25", "p75", "p95", "Pearson", "Spearman")) + "|")
    lines += ["", "|经济分组|channel|local residual mean|配对闭环输入差mean|negative比例|",
        "|---|---|---:|---:|---:|"]
    for r in behaviors:
        if r["geometry"] == "baseline" and r["group"] in ("economic_positive", "economic_negative"):
            lines.append(f"|{r['group']}|{r['channel']}|{float(r['mean']):.5f}|{float(r['paired_closed_loop_difference_mean']):.5f}|{float(r['negative_fraction']):.2%}|")
    lines += ["", "correlation不等于causation。状态条件分箱、绝对residual相关性与authority utilization均在actor_behavior_summary.csv。", "",
        "## 新geometry动作范围是否被使用", "",
        "在每条新geometry轨迹的同一state/z/raw actor下查询原online mapping，并检查当前输入是否属于原alpha=0.1收缩动作集合。",
        "这是局部映射干预，不传播干预后的plant，也不将local差值当成闭环经济因果效应。", "",
        "|geometry|原动作集合外步数比例|同raw映射P100绝对变化mean|同raw映射F200绝对变化mean|原QP不可行|",
        "|---|---:|---:|---:|---:|"]
    for r in usage:
        lines.append(f"|{r['geometry']}|{float(r['fraction_outside_original_action_image']):.2%}|{float(r['mean_abs_mapping_P100_change']):.5f}|{float(r['mean_abs_mapping_F200_change']):.5f}|{r['original_QP_infeasible_count']}|")
    lines += ["", "C2并非完全没有使用新增geometry：映射后的F200及可执行动作范围确实改变。不能仅归因于positive F200 authority；baseline投影、动作中心和随后state轨迹也改变。",
        "C2_8的baseline成本与production一致，新增P100范围及映射被实际使用；其净增益不是baseline减弱制造，也不是CSV回放dtype的数值差异。", "",
        "## 五个P100-only预注册候选", "",
        "|factor|certificate|positive P100 authority|negative P100 authority|F200 positive authority|完成路径|收益mean%|收益worst%|win|X2最低margin|P100 TV|F200 TV|P100 RMS|F200 RMS|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    aggregate = []
    p100_parts = []
    frozen = {int(r["validation_seed"]): r for r in read("geometry_vs_sac_decomposition.csv") if r["geometry"] == "baseline"}
    comparisons = []
    for c in certs:
        rows = [r for r in cf if r["geometry"] == c["candidate"] and r["status"] == "COMPLETE"]
        base = dict(candidate=c["candidate"], factor=float(c["factor"]), certificate=c["certificate"], completed_paths=len(rows))
        if len(rows) != 10:
            lines.append(f"|{c['factor']}|{c['certificate']}|{c.get('P100_applied_residual_max', '')}|{c.get('P100_applied_residual_min', '')}|{c.get('F200_applied_residual_max', '')}|{len(rows)}|pending|pending|pending|pending|pending|pending|pending|pending|")
            aggregate.append(base); continue
        keys = ("Gain_total_vs_prod_pct", "Gain_SAC_given_geometry_pct", "Gain_geometry_pct", "economic_win_fraction",
            "X2_std", "centered_X2_MAE", "P100_TV", "F200_TV", "P100_RMS_du", "F200_RMS_du",
            "X2_IAE_ratio", "P2_IAE_ratio", "X2_ISE_ratio", "P2_ISE_ratio")
        base.update({key: average(rows, key) for key in keys})
        base.update(worst_gain_pct=min(float(r["Gain_total_vs_prod_pct"]) for r in rows),
            minimum_X2_margin=min(float(r["minimum_X2_margin"]) for r in rows))
        increments = []
        for r in rows:
            original = frozen[int(r["validation_seed"])]
            difference = float(r["Gain_total_vs_prod_pct"])-float(original["Gain_total_vs_prod_pct"])
            increments.append(difference)
            comparisons.append(dict(candidate=c["candidate"], validation_seed=r["validation_seed"],
                economic_gain_increment_percentage_points=difference,
                **{key+"_ratio_vs_frozen_production": float(r[key])/float(original[key]) for key in (
                    "P100_TV", "F200_TV", "P100_RMS_du", "F200_RMS_du", "X2_std", "centered_X2_MAE")},
                X2_minimum_margin_change=float(r["minimum_X2_margin"])-float(original["minimum_X2_margin"])))
        base.update(mean_increment_vs_frozen_SAC_pp=float(np.mean(increments)),
            min_increment_vs_frozen_SAC_pp=float(min(increments)),
            all_development_increments_positive=all(x > 1e-10 for x in increments),
            conditional_increment_CI95_halfwidth=2.2621571628*float(np.std(increments, ddof=1))/np.sqrt(10),
            CI_scope="paired DEV paths, fixed trained actor; reused selection set, not training-seed uncertainty or independent generalization")
        for key in study.io.SAFETY:
            base[key] = sum(int(r[key]) for r in rows)
        for r in rows:
            path = root / "p100_pairs" / c["candidate"] / f"seed_{int(r['validation_seed'])}_result.json"
            p100_parts.extend(study.v2.load(path)["components"])
        aggregate.append(base)
        lines.append(f"|{c['factor']}|{c['certificate']}|{float(c['P100_applied_residual_max']):.5f}|{float(c['P100_applied_residual_min']):.5f}|{float(c['F200_applied_residual_max']):.5f}|10|{base['Gain_total_vs_prod_pct']:.8f}|{base['worst_gain_pct']:.8f}|{base['economic_win_fraction']:.2%}|{base['minimum_X2_margin']:.5f}|{base['P100_TV']:.2f}|{base['F200_TV']:.2f}|{base['P100_RMS_du']:.5f}|{base['F200_RMS_du']:.5f}|")
    study.write_csv(root / "p100_aggregate_metrics.csv", aggregate)
    if comparisons:
        study.write_csv(root / "p100_comparison_to_frozen_sac.csv", comparisons)
    complete = [r for r in aggregate if r["completed_paths"] == 10]
    if len(complete) == 5:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        factors = [r["factor"] for r in complete]
        axes[0].errorbar(factors, [r["mean_increment_vs_frozen_SAC_pp"] for r in complete],
            yerr=[r["conditional_increment_CI95_halfwidth"] for r in complete], fmt="o", capsize=4)
        axes[0].axhline(0, color="grey", lw=.7)
        axes[0].set(xlabel="P100 window multiplier (F200 fixed)", ylabel="Gain increment vs frozen SAC\n(percentage points)",
            title="Paired DEV paths: mean and conditional 95% CI")
        axes[1].scatter([r["Gain_total_vs_prod_pct"] for r in complete], [r["minimum_X2_margin"] for r in complete])
        for r in complete:
            axes[1].annotate(str(r["factor"]), (r["Gain_total_vs_prod_pct"], r["minimum_X2_margin"]), xytext=(4, 4), textcoords="offset points")
        axes[1].set(xlabel="Economic gain vs production baseline (%)", ylabel="Minimum X2 constraint margin (percentage points)",
            title="Economics / empirical margin trade-off")
        for ax in axes:
            ax.grid(alpha=.15)
        fig.tight_layout()
        fig.savefig(root / "p100_paired_increment_and_margin.png", dpi=160)
        plt.close(fig)
    if p100_parts:
        study.write_csv(root / "p100_component_decomposition.csv", p100_parts)
    lines += ["", "证书JSON保存全部gate、10256个非线性bounded-W样本、完整QP probes。有限样本覆盖不声称连续域解析证明。",
        "Gaussian DEV闭环的W超界单独记录；这些轨迹仅为empirical safety，不重新定义certified W。", "",
        "## witness与下一阶段", "", json.dumps(decision["witness_summary"], ensure_ascii=False, indent=2), "",
        "完整512个候选完成前，不能判断learning headroom、intrinsic economic upper bound或geometry方向最终优劣。",
        "有限state-only family的best-found也不是global optimum或oracle upper bound。",
        "未完成witness时不画虚构Pareto，也不填safe/positive数量为0。", "",
        "## 固定与测试", "",
        f"production hashes unchanged: {study.locked()['source_and_production_hashes'] == study.protected()}",
        "未训练SAC；未使用430000~430049；未修改生产U_R/alpha=0.1/Strong reference/K/W/Z/S/Omega/QP/actor/reward。",
        "未commit、未push；旧Conservative partial结果保留。",
        "测试结果见tests_receipt.json。可恢复执行命令见econ_attribution/README.md。"]
    (root / "ECONOMIC_ATTRIBUTION_DETAILS_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    study.save(root / "supplementary_report_receipt.json", dict(source_sha256=study.io.digest(__file__),
        production_hashes_unchanged=study.locked()["source_and_production_hashes"] == study.protected(),
        p100_complete=sum(r["completed_paths"] == 10 for r in aggregate),
        witness_status=decision["witness_summary"].get("status", "NOT_RUN")))
    print(root / "ECONOMIC_ATTRIBUTION_DETAILS_ZH.md", flush=True)


if __name__ == "__main__":
    main()
