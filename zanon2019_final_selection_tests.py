"""Pure final_v1 validation tests: no SAC training or final disturbance access."""
import copy
import unittest
import numpy as np
from . import zanon2019_final_selection as s
from .zanon2019_final_experiment import SOURCE42,read_json,csv_rows


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows=[r for r in csv_rows(SOURCE42/'fixed_evaluation.csv') if int(r['episode'])==15]
    def summarize(self,rows=None):return s.aggregate(15,rows or self.rows,15000,5000)
    def test_real_scan_has_no_manual_episode(self):
        audits,selected=s.scan(SOURCE42,read_json(SOURCE42/'experiment_manifest.json'))
        self.assertEqual(len(audits),21)
        self.assertGreater(sum(r['eligible_final_checkpoint'] for r in audits),0)
        self.assertEqual(selected['final_rank'],1)
        self.assertNotEqual(selected['episode'],5)
        self.assertGreater(selected['mean_economic_improvement_pct'],0)
        self.assertGreaterEqual(selected['worst_seed_economic_improvement_pct'],0)
        ep5=next(r for r in audits if r['episode']==5)
        self.assertFalse(ep5['eligible_final_checkpoint'])
    def test_all_validation_seeds_must_be_nonnegative(self):
        rows=copy.deepcopy(self.rows)
        r=rows[0];r['SAC_J_econ']=str(float(r['baseline_J_econ'])*1.000001)
        for r in rows[1:]:r['SAC_J_econ']=str(float(r['baseline_J_econ'])*.999)
        result=self.summarize(rows)
        self.assertGreater(result['mean_economic_improvement_pct'],0)
        self.assertLess(result['worst_seed_economic_improvement_pct'],0)
        self.assertFalse(result['eligible_final_checkpoint'])
    def test_zero_mean_rejected(self):
        rows=copy.deepcopy(self.rows)
        for r in rows:r['SAC_J_econ']=r['baseline_J_econ']
        self.assertFalse(self.summarize(rows)['eligible_final_checkpoint'])
    def test_negative_roundoff_is_not_tolerated(self):
        rows=copy.deepcopy(self.rows)
        rows[0]['SAC_J_econ']=str(np.nextafter(float(rows[0]['baseline_J_econ']),np.inf))
        self.assertFalse(self.summarize(rows)['eligible_final_checkpoint'])
    def test_each_safety_counter_rejects(self):
        for name in ('physical_state_violation_steps','physical_input_violation_count','QP_infeasible_count','Omega_exit_count'):
            rows=copy.deepcopy(self.rows);rows[0]['SAC_'+name]='1'
            self.assertFalse(self.summarize(rows)['eligible_final_checkpoint'])
    def test_W_and_IAE_are_not_hard_gates(self):
        result=self.summarize()
        self.assertGreater(result['W_exceedance_rate'],0)
        self.assertGreater(result['X2_IAE_ratio'],1.10)
        self.assertTrue(result['eligible_final_checkpoint'])
    def test_warmup_and_complete_pairing_required(self):
        rows=copy.deepcopy(self.rows)
        for r in rows:r['global_step']='4999'
        self.assertFalse(s.aggregate(15,rows,4999,5000)['eligible_final_checkpoint'])
        self.assertFalse(self.summarize(self.rows[:-1])['eligible_final_checkpoint'])
        rows=copy.deepcopy(self.rows);rows[0]['paired_disturbance_identity_max_error']='1e-15'
        self.assertFalse(self.summarize(rows)['eligible_final_checkpoint'])
    def test_anchor_bands_do_not_chain_or_depend_on_input_order(self):
        template=self.summarize();rr=[]
        for ep,ratio,margin in ((1,1.,.2),(2,1.009,.3),(3,1.018,.4)):
            rr.append(dict(template,episode=ep,X2_IAE_ratio=ratio,minimum_X2_margin=margin))
        order=[r['episode'] for r in s.rank(copy.deepcopy(rr))]
        self.assertEqual(order,[2,1,3])
        for permutation in (rr[::-1],[rr[1],rr[2],rr[0]]):
            self.assertEqual([r['episode'] for r in s.rank(copy.deepcopy(permutation))],order)
    def test_economics_only_breaks_last_tie(self):
        template=self.summarize()
        poor_margin=dict(template,episode=1,minimum_X2_margin=.1,mean_economic_improvement_pct=100)
        good_margin=dict(template,episode=2,minimum_X2_margin=.2,mean_economic_improvement_pct=.001)
        self.assertEqual(s.rank([poor_margin,good_margin])[0]['episode'],2)
        tied=[dict(template,episode=3,mean_economic_improvement_pct=.1),dict(template,episode=4,mean_economic_improvement_pct=.2)]
        self.assertEqual(s.rank(tied)[0]['episode'],4)


if __name__=='__main__':unittest.main()
