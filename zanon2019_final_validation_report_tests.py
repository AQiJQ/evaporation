"""Validation-report unit tests, no training/disturbance generation."""
import unittest
from . import zanon2019_final_validation_report as r
from .zanon2019_final_experiment import ROOT,read_json
from .zanon2019_final_experiment_tests import safe_test_directory


def fixtures(n):
    policies=[];rows=[]
    for i,seed in enumerate((42,2027,314159)):
        eligible=i<n
        policies.append(dict(training_seed=seed,status='successful' if eligible else 'no_stable_positive_economic_safe_checkpoint',
                             eligible_final_checkpoint=eligible,training_completed_100x1000=True))
        row=dict(training_seed=seed,eligible_final_checkpoint=eligible,F200_TV_ratio=1.5 if eligible else None)
        if eligible:row.update(mean_economic_improvement_pct=.01*(i+1),X2_IAE_ratio=1.1+i*.1,
            X2_std_ratio=1.05,minimum_X2_margin=.18,P100_TV_ratio=.9)
        rows.append(row)
    return policies,rows


class ValidationReportTests(unittest.TestCase):
    def test_real_locked_policy_extra_metrics(self):
        p=read_json(ROOT/'seed42_validation_lock.json')
        metrics,within=r.enrich(p)
        self.assertEqual(len(within),13)
        self.assertTrue(all(s['n']==10 for s in within))
        self.assertGreater(metrics['X2_std_ratio'],0)
        self.assertGreater(metrics['G_X2_ratio'],0)
        self.assertAlmostEqual(metrics['X2_IAE_ratio'],p['validation_metrics']['X2_IAE_ratio'])
    def test_null_policies_excluded_not_zero_filled(self):
        policies,rows=fixtures(1)
        s=r.build_summary(policies,rows,[])
        self.assertEqual(s['eligible_policy_count'],1)
        self.assertAlmostEqual(s['mean_of_selected_policy_economic_improvements'],.01)
        self.assertIsNone(s['std_between_training_seed_economic_improvements'])
        self.assertEqual(s['case'],'Case C')
    def test_cases_A_B_C(self):
        for n,case in ((3,'Case A'),(2,'Case B'),(1,'Case C'),(0,'Case C')):
            p,rows=fixtures(n);s=r.build_summary(p,rows,[])
            self.assertEqual(s['case'],case)
            self.assertEqual(s['n_completed_runs'],3)
            self.assertEqual(s['eligible_policy_count'],n)
        self.assertIsNone(r.build_summary(*fixtures(0),[])['mean_X2_IAE_ratio'])
    def test_pending_is_not_failure_case(self):
        p,rows=fixtures(1);p[1]['status']='training_pending';p[1]['training_completed_100x1000']=False
        s=r.build_summary(p,rows,[])
        self.assertEqual(s['case'],'pending_training_completion')
        self.assertFalse(s['stable_positive_economic_validation_across_all_three'])
    def test_between_seed_std_not_30_iid_CI(self):
        p,rows=fixtures(3);s=r.build_summary(p,rows,[])
        self.assertAlmostEqual(s['std_between_training_seed_economic_improvements'],.01)
        self.assertNotIn('pooled_CI',s)
        self.assertTrue(s['F200_TV_increase_repeated_across_all_three'])
    def test_failed_validation_safety_is_not_hidden(self):
        p,rows=fixtures(2)
        history=[[dict(SAC_physical_state_violation_steps='2',SAC_physical_input_violation_count='1',
                       SAC_QP_infeasible_count='3',SAC_Omega_exit_count='4')]]
        s=r.build_summary(p,rows,history)
        self.assertEqual(s['total_validation_physical_violations'],3)
        self.assertEqual(s['total_validation_QP_infeasible'],3)
        self.assertEqual(s['total_validation_Omega_exit'],4)

    def test_four_plots_keep_failed_policy_visible(self):
        _,rows=fixtures(2)
        with safe_test_directory() as root:
            r.plots(root,rows,[],qa_only=True)
            self.assertEqual(len(list(root.glob('training_seed_validation_*.png'))),4)


if __name__=='__main__':unittest.main()
