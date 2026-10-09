"""Static U_R audit tests; no controller/domain redesign or plant rollout."""
from copy import deepcopy
import inspect
import unittest
import numpy as np

from . import zanon2019_ur_region_audit as a


class URRegionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before=a.protected_hashes()
        cls.envs=a.prior.environments()
        cls.region,cls.active=a.reconstruct(cls.envs['Strong'])
        cls.sensitivity=a.window_sensitivity(cls.envs['Strong'],cls.region)

    def test_UR_reconstruction_reproducible(self):
        again,_=a.reconstruct(self.envs['Strong'])
        for key in ('UR_lower','UR_upper','H_physical','h_physical'):
            np.testing.assert_array_equal(again[key],self.region[key])

    def test_F200_upper_reproduces(self):
        self.assertAlmostEqual(self.region['UR_upper'][1],249.52860663022335,places=10)

    def test_P100_upper_reproduces(self):
        self.assertAlmostEqual(self.region['UR_upper'][0],233.65538828079238,places=10)

    def test_active_facet_stable(self):
        _,again=a.reconstruct(self.envs['Strong'])
        for channel,expected in [('P100',4),('F200',5)]:
            active=[row for row in self.active if row['channel']==channel and row['active']]
            repeated=[row for row in again if row['channel']==channel and row['active']]
            self.assertEqual([r['constraint_id'] for r in active],[expected])
            self.assertEqual(active,repeated)
            self.assertEqual(active[0]['audit_LP_multiplier'],1)
            self.assertIsNone(active[0]['original_builder_multiplier'])

    def test_removal_addition_nonmutating(self):
        d=self.envs['Strong'][2]
        before=deepcopy(vars(d))
        rows=a.window_sensitivity(self.envs['Strong'],self.region)
        table={row['audit_configuration']:row for row in rows}
        self.assertEqual(table['physical_only']['F200_upper'],400)
        self.assertEqual(table['remove_local_F200_upper']['F200_upper'],400)
        for key,old in before.items():
            if isinstance(old,np.ndarray):np.testing.assert_array_equal(old,getattr(d,key))
            else:self.assertEqual(old,getattr(d,key))

    def test_W_off_isolated(self):
        d=self.envs['Strong'][2];before=d.w_vertices.copy()
        rows=a.window_sensitivity(self.envs['Strong'],self.region)
        full=next(r for r in rows if r['audit_configuration']=='full_UR')
        off=next(r for r in rows if r['audit_configuration']=='remove_W_direct_term')
        self.assertEqual(full['F200_upper'],off['F200_upper'])
        np.testing.assert_array_equal(before,d.w_vertices)
        self.assertIn('indirect',off['note'])

    def test_S_Omega_dependency_classification(self):
        deps=a.dependencies()
        self.assertIn('S',deps['direct_UR_absent'])
        self.assertIn('Omega',deps['direct_UR_absent'])
        self.assertIn('S',deps['indirect_scale_gates'])
        text=inspect.getsource(a.OnlineThetaLearner._robust_region_bounds)
        for term in ('w_vertices','invariant_lower','predecessor','omega'):
            self.assertNotIn(term,text)

    def test_finite_jump_independence(self):
        self.assertTrue(a.dependencies()['finite_jump_independent'])
        self.assertFalse(self.region['derived_from_predecessor'])

    def test_Strong_Conservative_hashes_unchanged(self):
        self.assertEqual(self.before,a.protected_hashes())
        ds,dc=self.envs['Strong'][2],self.envs['Primary_Conservative'][2]
        for key in ('k','w_vertices','rpi_boundary','robust_input_lower','robust_input_upper','invariant_lower','invariant_upper'):
            np.testing.assert_array_equal(getattr(ds,key),getattr(dc,key))

    def test_residual_authority_nonmutating(self):
        contexts=[]
        for name,env in self.envs.items():
            cfg,_,d=env[:3]
            contexts.append(dict(reference=name,role='test_reference',step=0,state=cfg.robust_economic_reference_state.copy(),
                z=d.z_ref.copy(),source=None))
        before=a.protected_hashes()
        rows,ranges,isolated,snapshots,stats=a.attribution(self.envs,contexts)
        self.assertEqual(before,a.protected_hashes())
        self.assertEqual(stats['QP_infeasible_count'],0)
        self.assertEqual(stats['QP_modification_count'],0)
        self.assertTrue(all(r['F200_residual_max']>0 and r['F200_residual_min']<0 for r in ranges))

    def test_epsilon_is_rhs_sensitivity_not_tolerance_change(self):
        row=next(r for r in self.sensitivity if r['audit_configuration']=='epsilon_local_F200_upper')
        self.assertAlmostEqual(row['F200_increase_vs_full'],1e-6,places=9)

    def test_no_fabricated_scale_failure_gate(self):
        self.assertIsNone(self.region['original_limiting_scale_gate'])
        self.assertIn('not archived',self.region['original_selected_scale_failure_log_status'])

    def test_P100_more_restricted_at_UR_layer(self):
        reduction=self.region['physical_upper']-self.region['UR_upper']
        self.assertGreater(reduction[0],reduction[1])


if __name__=='__main__':unittest.main()
