"""Paper artifacts from archived paired validation data ONLY; never trains.

ECC2019 is a qualitative external source, never a numeric comparator here.
No solver, actor, mapping, reward or safety configuration is modified.
Run from the repository parent: python -m evaporation.paper_stochastic_results
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .ecc2019_reproduction import (
    REPO, PDF, SOURCES, DEV_SEEDS, B, REFERENCE, digest, common_model,
    read_ours_archive, validate_sources, save_json,
)
from .zanon2019_benchmark import write_csv

OUT = REPO / 'evaporation_safe_sac/paper_results_stochastic'
LABELS = ('baseline', 'A_alpha010', 'B_alpha020_rewardcal')
SCOPE = 'validation/development; not independent held-out test'


def stats(values):
    v = np.asarray(values, dtype=float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.
    # Student t(9), not a CI across independently trained SAC seeds.
    half = 2.2621571627409915 * sd / np.sqrt(len(v))
    return dict(mean=float(v.mean()), sample_std=sd, CI95_lower=float(v.mean()-half),
                CI95_upper=float(v.mean()+half), min=float(v.min()), max=float(v.max()))


def prepare():
    manifests = validate_sources()
    model = common_model()
    archives = {label: {} for label in LABELS}
    sources = list(SOURCES.values())
    for seed in DEV_SEEDS:
        base = read_ours_archive(sources[0], seed, 'baseline', model)
        other = read_ours_archive(sources[1], seed, 'baseline', model)
        for k in ('states', 'controls', 'costs', 'disturbances'):
            if not np.array_equal(base[k], other[k]):
                raise ValueError(f'Paired baseline mismatch: seed={seed} {k}')
        archives['baseline'][seed] = base
        for label, source in zip(LABELS[1:], sources):
            archives[label][seed] = read_ours_archive(source, seed, 'SAC', model)
        print(f'Archived seed {seed} verified (no closed-loop experiment)', flush=True)
    rows = []
    for label in LABELS:
        for seed, data in archives[label].items():
            m = dict(data['metrics'])
            # Existing predeclared B-reference benchmark is primary; also keep
            # nominal-optimum metrics so target/bias effects cannot be hidden.
            for state in ('X2', 'P2'):
                for metric in ('IAE', 'ISE'):
                    m[f'{state}_{metric}_nominal'] = m[f'{state}_{metric}']
                    m[f'{state}_{metric}'] = m[f'{state}_{metric}_B_diagnostic']
                i = ('X2', 'P2').index(state)
                m[f'{state}_peak_deviation_nominal'] = m[f'{state}_peak_deviation']
                m[f'{state}_peak_deviation'] = float(np.abs(data['states'][:-1, i]-B[i]).max())
            p2 = data['states'][:, 1]
            m['P2_physical_violation_count'] = int(((p2 < model.cfg.state_lower[1]-1e-8) |
                                                   (p2 > model.cfg.state_upper[1]+1e-8)).sum())
            internal = data['internal_safety']
            for k, v in internal.items():
                # Internal robust-input occupancy differs from common physical
                # bound occupancy. Keep both explicitly, never silently overwrite.
                m['internal_'+k] = v
            bm = archives['baseline'][seed]['metrics']
            m['economic_improvement_pct'] = (bm['J_econ']-m['J_econ'])/bm['J_econ']*100
            for state in ('X2', 'P2'):
                for metric in ('IAE', 'ISE'):
                    m[f'{state}_{metric}_ratio'] = m[f'{state}_{metric}']/bm[f'{state}_{metric}_B_diagnostic']
                m[f'{state}_std_ratio'] = m[f'{state}_std']/bm[f'{state}_std']
            for name in ('P100', 'F200'):
                for metric in ('TV', 'RMS_du'):
                    m[f'{name}_{metric}_ratio'] = m[f'{name}_{metric}']/bm[f'{name}_{metric}']
            rows.append(dict(method=label, seed=seed, scope=SCOPE, **m))
    return manifests, archives, rows


def seed_audit(manifests):
    rows = []
    for name, source in SOURCES.items():
        m = manifests[name]
        rows.append(dict(seed=m['seed'], role='training_model_initialization',
                         source=str(source/'experiment_manifest.json'), evidence='executed existing run', independent_test=False))
        with (source/'training_log.csv').open(encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rows.append(dict(seed=int(r['disturbance_seed']), role='training_disturbance',
                                 source=str(source/'training_log.csv'), evidence='logged episode '+r['episode'], independent_test=False))
        with (source/'fixed_evaluation.csv').open(encoding='utf-8') as f:
            evaluations = list(csv.DictReader(f))
        used = sorted({int(r['seed']) for r in evaluations})
        for seed in used:
            episodes = sorted({int(r['episode']) for r in evaluations if int(r['seed']) == seed})
            rows.append(dict(seed=seed, role='validation_and_diagnostic_reuse',
                             source=str(source/'fixed_evaluation.csv'), evidence=f'{len(episodes)} checkpoints; reused for authority/reward decisions', independent_test=False))
        for seed in m['reserved_final_test_seeds']:
            observed = seed in used or any((REPO/'evaporation_safe_sac').glob(f'**/seed_{seed}.csv'))
            rows.append(dict(seed=seed, role='reserved_test_execution_found' if observed else 'unused_test_reserved',
                             source=str(source/'experiment_manifest.json'), evidence=f'final_test_performed={m["final_test_performed"]}; archived trajectory/fixed-evaluation scan observed={observed}',
                             independent_test=False if observed else 'not evaluated'))
    audit_source = REPO/'zanon2019_sac_optimization_audit.py'
    # Existing MC diagnostic rule. This row records a seed family, not fabricated
    # evidence that every possible member was executed.
    rows.append(dict(seed='9000000 + state_id*100 + repeat', role='diagnostic_MC_seed_family',
                     source=str(audit_source), evidence='source-code rule; diagnostic, not final test', independent_test=False))
    if any(r['role']=='reserved_test_execution_found' for r in rows):
        raise ValueError('Reserved seed execution found; manually audit before labeling unused')
    return rows


def aggregate_table(rows, labels):
    result = []
    keys = [k for k, v in rows[0].items() if isinstance(v, (float, int)) and k != 'seed']
    for metric in keys:
        r = dict(metric=metric, scope=SCOPE, n_disturbance_seeds=len(DEV_SEEDS), n_training_seeds=1,
                 aggregation='mean per-seed; ratio/gain = mean paired ratios; std = mean within-seed population std',
                 reference='B=[25.39,50.125]; *_nominal=[25,49.743]; safety includes terminal state')
        for label in labels:
            values = [p[metric] for p in rows if p['method']==label]
            for key, value in stats(values).items():
                r[f'{label}_{key}'] = value
        result.append(r)
    # Worst realization gain is not a mean or a lower CI endpoint.
    for metric, source_key, op in (
        ('worst_seed_economic_improvement_pct', 'economic_improvement_pct', min),
        ('global_X2_min', 'X2_min', min), ('global_X2_max', 'X2_max', max),
        ('global_P2_min', 'P2_min', min), ('global_P2_max', 'P2_max', max),
        ('global_minimum_X2_margin', 'X2_margin_min', min),
        ('global_P100_min', 'P100_min', min), ('global_P100_max', 'P100_max', max),
        ('global_F200_min', 'F200_min', min), ('global_F200_max', 'F200_max', max),
    ):
        r = dict(metric=metric, scope=SCOPE, n_disturbance_seeds=len(DEV_SEEDS), n_training_seeds=1,
                 aggregation='observed global extremum across realizations; no CI of an extremum', reference='same seed/cost/horizon')
        for label in labels:
            r[f'{label}_mean'] = op(p[source_key] for p in rows if p['method']==label)
        result.append(r)
    return result


def candidate_table(rows, manifests):
    out = []
    for label, name in zip(LABELS[1:], SOURCES):
        subset = [r for r in rows if r['method']==label]
        r = dict(candidate=label, episode=100, scope=SCOPE,
                 authority_scale=manifests[name]['stochastic_residual_scale'],
                 reward_definition=manifests[name]['reward'],
                 reward_weights=json.dumps(manifests[name].get('weights', {})),
                 comparison_warning='authority AND reward differ; not a pure authority ablation',
                 recommendation='validation main candidate' if label==LABELS[1] else 'supplementary Pareto point')
        for k, v in subset[0].items():
            if isinstance(v, (int, float)) and k!='seed':
                r[k] = float(np.mean([p[k] for p in subset]))
        r['worst_seed_economic_improvement_pct'] = min(p['economic_improvement_pct'] for p in subset)
        r['global_minimum_X2_margin'] = min(p['X2_margin_min'] for p in subset)
        r['global_X2_min'] = min(p['X2_min'] for p in subset)
        r['global_X2_max'] = max(p['X2_max'] for p in subset)
        out.append(r)
    return out


def figures(archives, rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'savefig.dpi':200})
    seed = DEV_SEEDS[0]  # predeclared representative, NOT the best realization
    base, sac = archives['baseline'][seed], archives[LABELS[1]][seed]
    colors = ('#0072B2', '#D55E00')
    def make(name, ylabel, x, y0, y1, *, ylim=None, line=None, note=''):
        fig, ax = plt.subplots(figsize=(8,3.5))
        ax.plot(x, y0, color=colors[0], lw=.9, label='Robust zero residual')
        ax.plot(x, y1, color=colors[1], lw=.9, label='Safe Residual SAC (A)')
        if line is not None:
            ax.axhline(line, color='black', ls='--', lw=1.8, label='Quality constraint' if line==25 else 'Zero margin')
        if ylim: ax.set_ylim(*ylim)
        ax.set(xlabel='Time (s)', ylabel=ylabel, xlim=(0,1000))
        ax.grid(alpha=.2); ax.legend(loc='best', fontsize=8)
        ax.set_title(f'Validation realization {seed}; frozen episode 100', fontsize=10)
        fig.text(.02,.01, note or 'Development data, not an independent test; no external author trajectory.', fontsize=8)
        fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(out/name); plt.close(fig)
    t = np.arange(1001)
    x0, x1 = base['states'][:,0], sac['states'][:,0]
    make('paper_X2_stochastic_response.png','X2 (%)',t,x0,x1,ylim=(24.98,max(x0.max(),x1.max())+.035),line=25)
    make('paper_X2_stochastic_response_ecc_axis.png','X2 (%)',t,x0,x1,ylim=(24,30),line=25,
         note='ECC-like axis scale only; no digitization or matched external numerical comparison.')
    make('paper_X2_safety_margin.png','X2 - 25 (percentage points)',t,x0-25,x1-25,line=0,
         note=f'This realization minima: baseline={x0.min()-25:.6f}, SAC={x1.min()-25:.6f}; terminal included.')
    for i, name in enumerate(('P100','F200')):
        make(f'paper_{name}_response.png',name, t[:-1],base['controls'][:,i],sac['controls'][:,i])
    make('paper_cumulative_economic_cost.png','Cumulative economic cost',t,
         np.r_[0,np.cumsum(base['costs'])],np.r_[0,np.cumsum(sac['costs'])],
         note='Small paired economic gain; overlapping cumulative curves do not imply identical costs.')
    fig, ax = plt.subplots(figsize=(8,3.5))
    ax.plot(t[:-1],sac['costs']-base['costs'],color=colors[1],lw=.8)
    ax.axhline(0,color='black',lw=1); ax.set(xlabel='Time (s)',ylabel='Stage cost SAC - baseline',xlim=(0,1000))
    ax.set_title(f'Validation {seed}; negative = SAC cheaper'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'paper_instantaneous_economic_difference.png'); plt.close(fig)
    # Not a final-test distribution; every seed gets equal T samples. No pooled
    # IID confidence intervals are computed from correlated time samples.
    fig, axes = plt.subplots(1,2,figsize=(8,3.5))
    for label, color in zip(('baseline',LABELS[1]),colors):
        v = np.concatenate([archives[label][s]['states'][:-1,0] for s in DEV_SEEDS])
        axes[0].hist(v,bins=50,density=True,histtype='step',color=color,label=label)
    axes[0].set(xlabel='X2 (%)',ylabel='Empirical density (validation)'); axes[0].legend(fontsize=8)
    for label, color in zip(('baseline',LABELS[1]),colors):
        v=[r['X2_std'] for r in rows if r['method']==label]
        axes[1].plot(DEV_SEEDS,v,'o-',color=color,label=label)
    axes[1].ticklabel_format(useOffset=False,style='plain',axis='x')
    axes[1].set(xlabel='Validation disturbance seed',ylabel='Within-seed X2 std')
    fig.suptitle('Reused development seeds; NOT independent final-test statistics',fontsize=10)
    fig.tight_layout(); fig.savefig(out/'paper_X2_distribution.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    for label,color in zip(LABELS,('#0072B2','#D55E00','#009E73')):
        rs=[r for r in rows if r['method']==label]
        g=stats([r['economic_improvement_pct'] for r in rs]); margin=min(r['X2_margin_min'] for r in rs)
        ax.errorbar(g['mean'],margin,xerr=g['CI95_upper']-g['mean'],fmt='o',color=color,label=label,capsize=3)
    ax.set(xlabel='Mean paired economic improvement (%)',ylabel='Global minimum observed X2 margin')
    ax.set_title('Validation trade-off; A/B also differ in reward');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'paper_economics_safety_tradeoff.png');plt.close(fig)


def documents(rows, candidates, out):
    def avg(label,key): return float(np.mean([r[key] for r in rows if r['method']==label]))
    def low(label,key): return min(r[key] for r in rows if r['method']==label)
    def high(label,key): return max(r[key] for r in rows if r['method']==label)
    lines = []
    for label in LABELS:
        lines.append(f'|{label}|{avg(label,"J_econ"):.3f}|{avg(label,"economic_improvement_pct"):.6f}|'
                     f'{low(label,"economic_improvement_pct"):.6f}|{avg(label,"X2_mean"):.6f}|'
                     f'{avg(label,"X2_std"):.6f}|{low(label,"X2_min"):.6f}–{high(label,"X2_max"):.6f}|'
                     f'{avg(label,"X2_IAE_ratio"):.6f}|{avg(label,"P2_IAE_ratio"):.6f}|'
                     f'{avg(label,"P100_TV_ratio"):.6f}|{avg(label,"F200_TV_ratio"):.6f}|')
    table = ('|Method|J_econ|Gain %|Worst gain %|X2 mean|Mean within-seed std|Global X2 range|'
             'X2 IAE ratio (B)|P2 IAE ratio (B)|P100 TV ratio|F200 TV ratio|\n'
             '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|\n'+'\n'.join(lines))
    literature = '''# ECC2019 qualitative literature comparison

## Source-supported observations

Source: Practical Reinforcement Learning of Stabilizing Economic MPC, supplied PDF.
Numerical Example: PDF page 4 / printed 2261. Fig.2: PDF page 6 / printed 2263.
The process uses X2/P2 states and P100/F200 inputs; the X2 lower quality bound is 25.
The exogenous quantities fluctuate about nominal values. The reported variances
are Var(F1)=2, Var(X1)=1, Var(T1)=8, Var(T200)=5. State bounds are relaxed
through slack variables in Eq.(3e), with linear and quadratic slack costs.
PDF page 5 / printed 2262 states:
> Indeed, the constraint is violated but only rarely and by small amounts.

The paper discusses the economics/constraint trade-off and reports 14% and 12%
gains relative to its naive and nominal-economic NMPC comparators, respectively.
These are NOT gains relative to our robust controller. Fig.2 displays stochastic
X2 responses and an instantaneous economic cost difference. Its explicit algebraic
sign convention and exact percentage-gain formula are not specified in the paper.

## Our observations

See the internally paired table below. Our curves and distributions use archived
validation realizations, not recovered author data. All three internal methods
have zero observed physical state/input violations in these archived evaluations.
The concentration bands in our presented plots are narrow on a 24–30 axis.

## Comparison scope

qualitative external comparison, quantitative internal ablation.
Gaussian distribution, independent per-step sampling, 1 s sampling, initial state
B and a 1000-step horizon are implementation choices for reproduction; the exact
distribution, sampling convention, initial state and seed/realization matching
are not specified in the paper. We do not assert matched external protocols.
Do not infer ECC2019 std, IAE, violation rate or maximum violation from pixels.
No quantitative economic or disturbance-rejection superiority over ECC2019 is
established. A visibly narrower plotted band across these separate simulations
is a qualitative observation, not a controlled numerical superiority result.

## Suggested citation caption

ECC2019 Fig.2 (printed p.2263; PDF p.6): reported stochastic concentration
responses and economic-cost comparison of RL-tuned and reference NMPC controllers.
Reproduced only if the thesis author chooses to quote the original figure with
appropriate attribution/permission; no author trajectory has been redrawn here.

## Suggested body reference

ECC2019 reports rare and small violations of the X2 quality constraint, whereas
no X2<25 violation was observed in our evaluated validation trajectories. The
present curves form a visibly narrow band on a similar axis scale, but differences
in unspecified implementation details and random realizations prevent a matched
quantitative comparison. Both studies illustrate an economics/constraint trade-off.
'''
    (out/'ecc2019_literature_comparison.md').write_text(literature+'\n'+table+'\n',encoding='utf-8')
    gain=stats([r['economic_improvement_pct'] for r in rows if r['method']==LABELS[1]])
    numeric=(f'A mean gain {gain["mean"]:.6f}% (descriptive paired t95 interval '
             f'{gain["CI95_lower"]:.6f}–{gain["CI95_upper"]:.6f}%). '
             f'X2 B-reference IAE +{(avg(LABELS[1],"X2_IAE_ratio")-1)*100:.2f}%, '
             f'ISE +{(avg(LABELS[1],"X2_ISE_ratio")-1)*100:.2f}%, '
             f'P100 TV {(avg(LABELS[1],"P100_TV_ratio")-1)*100:+.2f}%, '
             f'F200 TV {(avg(LABELS[1],"F200_TV_ratio")-1)*100:+.2f}%.')
    scope = '''These are reused validation/development seeds 420000–420009, not
independent held-out test seeds or ten independently trained policies. Both
policies have training initialization seed 42. The reserved 430000–430049 have
no execution evidence in audited manifests/fixed evaluations/trajectory filenames;
no new evaluation was run. Confidence intervals describe variability across the
ten paired realizations conditional on the selected policy; adaptive tuning and
selection prevent interpreting them as unbiased final-test inference.
'''
    definitions = '''J_econ is the sum of saved next-state/final-input stage costs;
shaping and relaxed-state costs are excluded. Gain is the mean of per-seed
100*(J_base-J_SAC)/J_base, not the ratio of aggregate means. IAE/ISE and ratios in
the primary table use the prespecified balanced reference B=[25.39,50.125].
The additional *_nominal columns use [25,49.743]. Moving nearer to X2=25 may
reduce nominal-reference error while increasing B-reference error; neither is
hidden or substituted post hoc. Performance uses 1000 pre-step states; safety
uses 1001 states including the reconstructed terminal state. Input TV/RMS use
999 successive saved-input differences. X2 std is the mean of ten within-seed
population stds; extrema in the text span all realizations. CSV mean/min/max
fields distinguish per-seed averaging from global extrema. Margin quantiles are
per-seed statistics, not pooled confidence bounds. Empirical G is the L2 norm of
state deviation from B divided by the L2 norm of four exogenous deviations
normalized by sqrt(variance); it is empirical disturbance amplification, not
an H-infinity norm. Raw state statistics do not depend on the chosen reference.
'''
    en = f'''# Stochastic Experiment I — validation results

## 1. Stochastic settings

The paper reports variances [2,1,8,5] for [F1,X1,T1,T200]. We use their square
roots as standard deviations. Gaussian iid per-step draws, 1 s sampling,
1000 steps and initial B are implementation choices for reproduction, not
details specified in the paper. {scope}

## 2. Robust baseline

The zero-residual H∞/RPI/QP controller provides strong empirical disturbance
attenuation under the tested stochastic conditions. Its X2 mean is
{avg('baseline','X2_mean'):.6f}, mean within-seed std
{avg('baseline','X2_std'):.6f}, global minimum margin
{low('baseline','X2_margin_min'):.6f}. No observed state/input violation occurs.

## 3. Residual SAC

Candidate A is recommended provisionally for the validation main display: it
uses less X2 performance margin than B, despite smaller economic gain and a
larger F200 TV. B is a supplementary Pareto point, not a pure authority ablation:
reward calibration and authority both differ. Neither dominates the baseline.

## 4. Internally paired quantitative analysis

{table}

{numeric}

{definitions}
Residual SAC trades part of the robust baseline's disturbance-rejection margin
for a small positive economic gain, retaining zero observed physical violations.
Do not claim that the small X2 band is obtained without any control-activity
cost: A's F200 TV increases. No external matched economic comparison is made.

## 5. Qualitative ECC2019 comparison

See ecc2019_literature_comparison.md for source pages, original wording, caption
and citation text. ECC2019 reports rare/small X2 constraint violations with
relaxed state bounds; no such violation occurs in our displayed evaluations.
Our band appears narrow on a similar axis, but external numerical superiority
is not established because the protocols/realizations are not matched.

## 6. Safety claim boundary

Gaussian support is unbounded. Mean W exceedance is
{100*avg(LABELS[1],'internal_W_exceedance_rate'):.2f}% for A. This is empirical
stochastic evaluation, not formal Gaussian robust certification. Zero observed
physical/QP/Omega/robust-region anomalies do not certify the continuous domain.
Finite-jump Omega/Gm/Bj and conditional dwell-time certificates remain unchanged
for Experiments II and III, not transplanted into Experiment I.

## 7. Summary and readiness

Improved internal economics and preserved empirical safety coexist with an
explicit performance trade-off. These artifacts support a thesis development
results section, not an unbiased final test claim. Lock the candidate and
analysis protocol before an independently reserved paired test; no such test or
training is initiated here. Multiple training seeds would be needed for claims
about learning reproducibility. Stop ECC2019 fitting and further blind
reward/authority/critic tuning. Preserve finite-jump experiments separately.
'''
    cn=f'''# 随机工况实验 I：开发/验证集结果分析

## 1. 随机工况设置

论文给出 F1/X1/T1/T200 方差为 [2,1,8,5]，本实验使用平方根作为标准差。
Gaussian iid、逐步独立采样、1 s 采样、1000 steps、初态 B 属于
implementation choice for reproduction；相关精确实现 not specified in the paper。
当前仅使用反复参与 authority/reward 开发的 420000–420009 validation seeds。
两策略训练初始化均为 seed42，并非十个训练种子。430000–430049 仅预留，
审计范围内未发现运行证据。本轮未运行它们，也未训练。均值/95% CI 是给定
已选择策略下的配对验证 realization 描述统计，不是无偏独立 test 推断。

## 2. Baseline 结果

zero-residual H∞/RPI/QP 已具有较强经验扰动抑制。X2 均值
{avg('baseline','X2_mean'):.6f}，平均轨迹内标准差 {avg('baseline','X2_std'):.6f}；
跨 realization 最小安全余量 {low('baseline','X2_margin_min'):.6f}。
所检验轨迹的物理状态/输入违约均为零。

## 3. Residual SAC 结果

暂推荐 A 作为验证集主展示候选，B 作为补充 Pareto 点。A 的 X2 退化比 B 小，
但经济收益也较小，而且 F200 TV 的代价较大；没有任何一方全面占优。
A/B 同时改变 authority 和 reward，不能归因为纯 α 消融。
α 在此指 stochastic residual authority scale，不是 SAC entropy 系数。

## 4. 内部严格配对定量分析

{table}

{numeric}

两者共用初态、扰动 path、seed、horizon、nonlinear plant 和经济成本。
J_econ 不包含 shaping penalty；收益取每个 seed 的相对收益后再平均，
不采用 aggregate 均值的比值。主 IAE/ISE/peak 参考固定 B=[25.39,50.125]，
CSV 同时保留 *_nominal=[25,49.743] 的结果，避免事后切换 reference。
靠近 X2=25 可以减小 nominal error，却增加 B-reference error，二者并不矛盾。
性能积分使用 1000 个步前状态；安全审计含 terminal，共 1001 个状态。
输入 TV/RMS 使用 999 个相邻已执行输入差分。std 为各 seed 轨迹内总体 std 的
均值，正文极值为所有轨迹极值；CSV mean/min/max 区分两种聚合。
G_emp 是相对 B 的状态 L2 偏差除以四外扰按标准差归一化后的联合 L2 范数，
仅为 empirical disturbance amplification，并非 H∞ norm。
SAC 使用一部分抑扰余量换取小幅经济收益，不能把其解释为全面支配 baseline，
也不能忽略 A 的 F200 TV 增加。

## 5. ECC2019 定性文献比较

原文 PDF 第5页/印刷2262页报告质量约束仅罕见且小幅违反；Fig.2 位于
PDF第6页/印刷2263页。本文所显示轨迹没有 X2<25；相近宽纵轴下随机带较窄。
由于原作者 raw data、exact realization 和部分实现细节缺失，这只是不同模拟
下的定性视觉及报告行为比较，不构成受控数值优势证明。禁止计算本文相对
ECC2019 的 std/IAE/违约率/经济收益百分比。原文14%/12%的 baseline 是其
naive/nominal-economic NMPC，不是本文 robust baseline。
引用图注及正文措辞见 ecc2019_literature_comparison.md；没有重绘原图数据。

## 6. 安全结论边界

Gaussian 支撑无界，A 的 W 超界均值为
{100*avg(LABELS[1],'internal_W_exceedance_rate'):.2f}%。因此这里只能称
empirical stochastic evaluation，不能宣称 formal Gaussian robust guarantee。
观测到的物理/QP/Omega/robust-region 异常为零不意味着连续域解析认证。
Omega/Gm/Bj 与 finite-jump supervisor 保留用于实验 II/III，不将其条件证书
移植为实验 I 的 Gaussian 安全证明。

## 7. 小结与论文条件

内部经济小幅改善、经验安全保持与明确性能代价同时成立。当前材料足以形成
诚实的论文开发/验证结果章节，但尚不能冒充独立最终测试。真正必要的下一项
是先锁定候选及分析方案，再进行预留独立种子的配对评价；本轮未自动启动。
若需跨训练种子可复现结论，单 seed42 不足。停止 ECC 数值拟合、盲目 reward/
authority/critic 调参；有限跳变实验独立保留。本轮无训练、无 commit、无 push。
'''
    (out/'paper_stochastic_results_en.md').write_text(en,encoding='utf-8')
    (out/'paper_stochastic_results_cn.md').write_text(cn,encoding='utf-8')
    names=sorted(p.name for p in out.glob('paper_*.png'))
    index='# Paper figures (validation/development only)\n\n'
    index+='Representative trajectory seed=420000, fixed episode=100; not best-case selection.\n\n'
    for name in names:
        index+=f'## {name}\n\n![{name}]({name})\n\n'
    index+='The distribution is explicitly development visualization, not a held-out multi-seed test.\n'
    (out/'paper_figures_index.md').write_text(index,encoding='utf-8')


def main():
    # Hash frozen artifacts before and after reporting, including preexisting
    # user edits. Verification reads archives, never advances an online actor.
    frozen=[REPO/name for name in ('model.py','config.py','control.py','sac.py','zanon2019_train.py') if (REPO/name).exists()]
    frozen += [source/'models/evaluation/episode_0100_actor.pth' for source in SOURCES.values()]
    frozen += [PDF]
    before={str(p):digest(p) for p in frozen}
    manifests, archives, rows = prepare()
    audits=seed_audit(manifests)
    OUT.mkdir(parents=True,exist_ok=True)
    write_csv(OUT/'seed_usage_audit.csv',audits)
    write_csv(OUT/'paper_per_seed_metrics.csv',rows)
    write_csv(OUT/'paper_main_stochastic_comparison.csv',aggregate_table(rows,LABELS[:2]))
    candidates=candidate_table(rows,manifests)
    write_csv(OUT/'paper_candidate_comparison.csv',candidates)
    figures(archives,rows,OUT)
    documents(rows,candidates,OUT)
    if before!={str(p):digest(p) for p in frozen}: raise RuntimeError('Frozen source/artifact modified')
    save_json(OUT/'paper_results_manifest.json',dict(scope=SCOPE,selected_candidate=LABELS[1],
        selection='provisional validation trade-off; not final-test selection',training_run=False,
        new_closed_loop_evaluation_run=False,independent_test_available=False,source_manifests=manifests,
        frozen_artifact_sha256=before,archive_provenance=[dict(method=l,seed=s,path=a['source'],sha256=a['source_sha256'])
            for l,aa in archives.items() for s,a in aa.items()],
        external_comparison='qualitative external literature comparison only',
        internal_comparison='strict quantitative paired comparison',
        safety='empirical only; Gaussian support unbounded; no continuous-domain proof'))
    print(json.dumps(dict(output=str(OUT),candidates=candidates),ensure_ascii=False),flush=True)


if __name__=='__main__': main()
