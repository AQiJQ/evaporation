"""Read completed isolated U_R evidence; no controller calls or training."""
from collections import Counter
from . import zanon2019_ur_certificate_sensitivity as study
from . import zanon2019_paired_v2 as io

def main():
    root=study.ROOT
    summary=io.load(root/'summary.json');replay=io.load(root/'rho_search_reconstruction.json')
    rows=summary['rows'];passed=[r for r in rows if r['certificate']=='CERTIFIED_PASS'];base=rows[0]
    def best(families,key):
        group=[r for r in passed if r['family'] in families]
        return max(group,key=lambda r:r[key]) if group else None
    shared=best(['shared_rho'],'rhoF');half=best(['F200_half_width'],'hF')
    independent=best(['independent_rho','independent_2D'],'rhoF')
    overall=best([r['family'] for r in passed],'UR_F200_upper')
    econ=[r for r in passed if r.get('economic_counterfactual_status','').startswith('COMPLETE')]
    economy=max(econ,key=lambda r:r['economic_improvement_pct']) if econ else None
    failures=Counter(g for r in rows for g in r.get('failed_gates','').split(';') if g)
    def value(r,key):return 'NOT_EVALUATED / no passing candidate' if r is None or r.get(key) is None else str(r[key])
    def name(r):return value(r,'candidate')
    replies=[
        ('rho search',f"initial={replay['initial_rho']}, growth={replay['growth']}, bisection={replay['bisection_iterations']}, tolerance={replay['tolerance']}; each attempt rebuilds sampled W and optimizes local K/M/geometry. Finite numerical acceptance, not a globally maximal admissible domain."),
        ('original maximum accepted rho',f"{replay.get('max_accepted_rho')}; selected={replay.get('selected_rho')}; exact archived-scale reproduction={replay.get('matches_archived_selected_scale')}"),
        ('Family A maximum shared rho',value(shared,'rhoF')),
        ('Family A F200 actual upper',value(shared,'UR_F200_upper')),
        ('Family A F200 nominal upper',value(shared,'nominal_tight_F200_upper')),
        ('Family B maximum hF',value(half,'hF')),
        ('Family B F200 actual upper',value(half,'UR_F200_upper')),
        ('Family C pass',str(independent is not None)),
        ('Family C maximum rhoF',value(independent,'rhoF')),
        ('Corresponding rhoP',value(independent,'rhoP')),
        ('shared-rho coupling', 'At the same F200 +15% window, shared +15% fails W coverage, unchanged P100 also fails, but P100 -5% / F200 +15% passes. This demonstrates joint-domain coverage coupling in this preregistered finite test; it is not cost-free independent expansion.'),
        ('current applied residual authority',f"P100=[{value(base,'P100_applied_residual_min')},{value(base,'P100_applied_residual_max')}]; F200=[{value(base,'F200_applied_residual_min')},{value(base,'F200_applied_residual_max')}], alpha=.1, reset reference"),
        ('maximum geometry applied authority',f"{name(overall)}: F200=[{value(overall,'F200_applied_residual_min')},{value(overall,'F200_applied_residual_max')}]"),
        ('authority increment',str(overall['F200_applied_residual_max']-base['F200_applied_residual_max']) if overall and base.get('F200_applied_residual_max') is not None else 'NOT_EVALUATED'),
        ('P100 effect',f"{name(overall)} P100=[{value(overall,'P100_applied_residual_min')},{value(overall,'P100_applied_residual_max')}]; independent candidates preserve actual P100 bounds, but state-specific baseline projection can still change authority."),
        ('failed candidates',', '.join(r['candidate'] for r in rows if r['certificate']!='CERTIFIED_PASS')),
        ('most common limiting gates',str(failures.most_common())),
        ('certified expanded UR exists',str(any(r['reclaimed_F200_UR_upper']>1e-8 for r in passed))),
        ('reclaimable F200 actual headroom',value(overall,'reclaimed_F200_UR_upper')),
        ('reclaimable F200 nominal headroom',value(overall,'reclaimed_F200_nominal_upper')),
        ('current frozen-policy economic gain (%)',value(base,'economic_improvement_pct')),
        ('all passing candidate economics','See complete sensitivity CSV; NOT_EVALUATED values are never zero-imputed.'),
        ('maximum frozen-policy gain (%)',f"{name(economy)}: {value(economy,'economic_improvement_pct')}"),
        ('win fraction',f"current={value(base,'economic_win_fraction')}; maximum-gain candidate={value(economy,'economic_win_fraction')}"),
        ('X2 minimum margin',f"current={value(base,'minimum_X2_margin')}; maximum-gain candidate={value(economy,'minimum_X2_margin')}"),
        ('control TV/RMS',f"current P100 TV/RMS={value(base,'P100_TV')}/{value(base,'P100_RMS_du')}, F200={value(base,'F200_TV')}/{value(base,'F200_RMS_du')}; maximum-gain P100={value(economy,'P100_TV')}/{value(economy,'P100_RMS_du')}, F200={value(economy,'F200_TV')}/{value(economy,'F200_RMS_du')}"),
        ('QP modifications',f"current={value(base,'QP_modification_count')}, maximum-gain={value(economy,'QP_modification_count')}; per-seed counts retained."),
        ('new empirical anomalies', 'Any counterfactual exception is retained and suppresses aggregate gain; inspect frozen_policy_geometry_sensitivity.csv, never score incomplete/unsafe rollouts.'),
        ('UR limits economic headroom', 'Compare frozen-policy and geometry-only zero-residual gains separately. An improvement driven by a changed zero-residual base is geometry sensitivity, not new learning.'),
        ('alpha=.1 bottleneck', 'Alpha is held fixed. This experiment cannot causally identify alpha versus critic/reward/learning as the remaining bottleneck.'),
        ('next priority','Finish frozen-policy counterfactual after disk resources are available, before choosing geometry versus SAC changes. No production expansion is selected here. Only passing candidates with useful authority and favorable empirical quality justify a separately preregistered follow-up.'),
        ('enlarged_ur_v1','Not created; not automatically authorized by this sensitivity result.'),
        ('sensitivity CSV',str(root/'ur_certificate_sensitivity.csv')),
        ('failure CSV',str(root/'ur_certificate_failure_reasons.csv')),
        ('certificate/authority figure',str(root/'ur_certificate_vs_authority.png')),
        ('authority/economics figure',str(root/'ur_authority_vs_economic_improvement.png')),
        ('channel-scaling comparison',str(root/'shared_vs_independent_ur_scaling.csv')),
        ('tests',f"Latest full run: {io.load(root/'tests_receipt.json')['unit_tests']} tests, passed={io.load(root/'tests_receipt.json')['passed']}. Latest run had one disk-write OSError(28); all 14 new contract tests passed. Earlier clean run: 221 main + 3 optional CasADi + evaporation.test passed. Final full-suite clean rerun pending resource resolution." if (root/'tests_receipt.json').exists() else 'NOT_EVALUATED'),
        ('production preserved',str(summary['production_hashes_unchanged'] and summary['geometry_hashes_unchanged'])),
        ('training',str(summary['training'])),('final seeds used',str(summary['final_seeds_used'])),
        ('commit',str(summary['commit'])),('push',str(summary['push']))]
    text=['# Certificate-aware U_R sensitivity study','',
          f"Status: {summary['status']}. {summary.get('blocker','')}",'',
          'All values use the frozen balanced-B model/certificate. Passing means the existing bounded-W numerical protocol and sampled coverage, not a continuous nonlinear or Gaussian safety proof.',
          'This study checks the active ECC2019 Z/Omega branch. Stored finite-jump Gm/Bj are hash-preserved and inactive here. Before adopting a P100-shrinking candidate for a finite-jump experiment, recheck its downstream Gm/Bj action-existence certificates. No production adoption is made.',
          '',f"Candidates: {summary['candidates']}; passing: {summary['certified_pass']}; failed: {summary['certified_fail']}.",'',
          '## Original rho replay','', '| attempt | rho | pass | failed gate |','|---:|---:|:---:|---|']
    text += [f"|{r['attempt']}|{r['scale']:.12g}|{r['feasible']}|{r['failure_reason']}|" for r in replay['attempts']]
    text += ['','## Candidate comparison','',
             '| candidate | family | F200 U_R upper | nominal upper | certificate | positive applied F200 | frozen gain % |',
             '|---|---|---:|---:|---|---:|---:|']
    text += [f"|{r['candidate']}|{r['family']}|{r['UR_F200_upper']:.9f}|{r['nominal_tight_F200_upper']:.9f}|{r['certificate']}|{value(r,'F200_applied_residual_max')}|{value(r,'economic_improvement_pct')}|" for r in rows]
    text += ['','## Requested answers','']+[f"{i}. {key}: {answer}\n" for i,(key,answer) in enumerate(replies,1)]
    text += ['','## Resume after resource resolution','',
        'Stop the separately running stage4 Oracle yourself if appropriate and free disk space. This study never stops that historical process or deletes its results.',
        'From the repository parent, run the following offline-only commands:',
        '```powershell',
        '& ".\\evaporation\\.venv-cuda\\Scripts\\python.exe" -m evaporation.zanon2019_ur_certificate_sensitivity --tests',
        '& ".\\evaporation\\.venv-cuda\\Scripts\\python.exe" -m evaporation.zanon2019_ur_certificate_sensitivity --counterfactual --resume --workers 1',
        '& ".\\evaporation\\.venv-cuda\\Scripts\\python.exe" -m evaporation.zanon2019_ur_certificate_report',
        '```','The candidate protocol, alpha and production hashes are checked on resume. No training entry is added.']
    (root/'UR_CERTIFICATE_SENSITIVITY.md').write_text('\n'.join(text),encoding='utf-8')
    print('report',root/'UR_CERTIFICATE_SENSITIVITY.md')

if __name__=='__main__':main()
