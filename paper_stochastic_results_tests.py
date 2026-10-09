"""Reporting regression checks; no SAC training or closed-loop experiments."""
import csv
import json
import unittest
import numpy as np
from .paper_stochastic_results import OUT, LABELS, stats
from .ecc2019_reproduction import digest


def read(name):
    with (OUT/name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


class ReportingTests(unittest.TestCase):
    def test_frozen_sources_unchanged(self):
        manifest=json.loads((OUT/'paper_results_manifest.json').read_text(encoding='utf-8'))
        self.assertFalse(manifest['training_run'])
        self.assertFalse(manifest['new_closed_loop_evaluation_run'])
        self.assertFalse(manifest['independent_test_available'])
        for path, sha in manifest['frozen_artifact_sha256'].items():
            self.assertEqual(digest(path),sha)
        for row in manifest['archive_provenance']:
            self.assertEqual(digest(row['path']),row['sha256'])

    def test_table_scope_and_aggregation(self):
        rows=read('paper_main_stochastic_comparison.csv')
        self.assertTrue(all('not independent held-out' in r['scope'] for r in rows))
        self.assertTrue(all('ecc2019' not in k.lower() for k in rows[0]))
        per=read('paper_per_seed_metrics.csv')
        indexed={(r['method'],int(r['seed'])):r for r in per}
        main={r['metric']:r for r in rows}
        for label in LABELS[:2]:
            gains=[]
            for seed in range(420000,420010):
                r=indexed[label,seed];b=indexed['baseline',seed]
                gain=(float(b['J_econ'])-float(r['J_econ']))/float(b['J_econ'])*100
                self.assertAlmostEqual(gain,float(r['economic_improvement_pct']))
                self.assertAlmostEqual(float(r['X2_IAE']),float(r['X2_IAE_B_diagnostic']))
                gains.append(gain)
            self.assertAlmostEqual(np.mean(gains),float(main['economic_improvement_pct'][label+'_mean']))
            self.assertAlmostEqual(min(gains),float(main['worst_seed_economic_improvement_pct'][label+'_mean']))

    def test_seed_roles(self):
        audit=read('seed_usage_audit.csv')
        roles={r['role'] for r in audit}
        self.assertIn('validation_and_diagnostic_reuse',roles)
        self.assertIn('unused_test_reserved',roles)
        self.assertNotIn('reserved_test_execution_found',roles)
        training={int(r['seed']) for r in audit if r['role']=='training_disturbance'}
        self.assertTrue(training.isdisjoint(set(range(420000,420010))))
        self.assertTrue(training.isdisjoint(set(range(430000,430050))))

    def test_safety_preserved_in_archives(self):
        for r in read('paper_per_seed_metrics.csv'):
            for key in ('physical_state_violation_count','physical_input_violation_count',
                        'internal_QP_infeasible_count','internal_Omega_exit_count','internal_robust_region_violation_count'):
                self.assertEqual(float(r[key]),0)
            # Nonzero W exceedance is retained, not hidden as a physical violation.
            self.assertGreater(float(r['internal_W_exceedance_rate']),.9)

    def test_deliverables(self):
        self.assertEqual(len(list(OUT.glob('paper_*.png'))),9)
        for name in ('paper_stochastic_results_cn.md','paper_stochastic_results_en.md','ecc2019_literature_comparison.md'):
            self.assertTrue((OUT/name).stat().st_size>1000)
        candidates=read('paper_candidate_comparison.csv')
        self.assertEqual(len(candidates),2)
        self.assertNotEqual(candidates[0]['reward_weights'],candidates[1]['reward_weights'])

    def test_descriptive_ci(self):
        s=stats(range(10))
        self.assertEqual(s['mean'],4.5)
        self.assertLess(s['CI95_lower'],s['mean'])
        self.assertGreater(s['CI95_upper'],s['mean'])


if __name__=='__main__': unittest.main()
