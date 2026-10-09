"""Static provenance tests; no plant trajectory, SAC update or safety redesign."""
from copy import deepcopy
import unittest
import numpy as np

from . import zanon2019_input_tightening_audit as audit
from .control import project_qp_2d


class InputTighteningAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hashes=audit.protected_hashes()
        cls.envs=audit.environments()
        cls.support=audit.support_audit(cls.envs['Strong'])
        cls.details={key:audit.reference_audit(env) for key,env in cls.envs.items()}

    def test_physical_bounds(self):
        for env in self.envs.values():
            np.testing.assert_array_equal(env[0].input_lower,[100,100])
            np.testing.assert_array_equal(env[0].input_upper,[400,400])

    def test_KZ_support_reproducibility(self):
        s=self.support
        np.testing.assert_allclose(s['KZ_polygon_min'],s['KZ_polygon_LP_min'],rtol=0,atol=1e-7)
        np.testing.assert_allclose(s['KZ_polygon_max'],s['KZ_polygon_LP_max'],rtol=0,atol=1e-7)
        np.testing.assert_allclose(s['KZ_runtime_support_min'],s['saved_input_support_min'],rtol=0,atol=1e-9)
        np.testing.assert_allclose(s['KZ_runtime_support_max'],s['saved_input_support_max'],rtol=0,atol=1e-9)

    def test_pure_U_minus_KZ_formula(self):
        s=self.support
        np.testing.assert_array_equal(s['pure_physical_U_minus_KZ_design_lower'],s['physical_lower']-s['KZ_runtime_support_min'])
        np.testing.assert_array_equal(s['pure_physical_U_minus_KZ_design_upper'],s['physical_upper']-s['KZ_runtime_support_max'])
        self.assertFalse(s['comparison_to_printed_216303281']['matches_pure_U_minus_KZ'])

    def test_P100_F200_provenance_consistency(self):
        s=self.support
        np.testing.assert_allclose(s['runtime_nominal_lower'],s['recomputed_runtime_nominal_lower'],rtol=0,atol=1e-9)
        np.testing.assert_allclose(s['runtime_nominal_upper'],s['recomputed_runtime_nominal_upper'],rtol=0,atol=1e-9)
        np.testing.assert_allclose(s['robust_region_half_width_formula_error'],0,atol=1e-9)

    def test_QP_final_bound_consistency(self):
        for name,t in self.details.items():
            cfg,model,d=self.envs[name][:3]
            h,b=np.asarray(t['QP_nominal_rows']),np.asarray(t['QP_nominal_rhs'])
            np.testing.assert_array_equal(h[:4],np.vstack([np.eye(2),-np.eye(2)]))
            np.testing.assert_array_equal(b[:4],np.r_[d.u_upper_tight,-d.u_lower_tight])
            oh,ob=np.asarray(t['Omega_QP_q_rows']),np.asarray(t['Omega_QP_q_rhs'])
            np.testing.assert_allclose(model.physical_input(d.v_ref+ob[:2]),t['actual_hard_upper'],atol=1e-9)
            np.testing.assert_allclose(model.physical_input(d.v_ref-ob[2:4]),t['actual_hard_lower'],atol=1e-9)
            u=cfg.robust_economic_reference_input
            q=model.normalized_input(u)-d.v_ref
            np.testing.assert_allclose(oh@q-ob,
                np.asarray(t['Omega_QP_absolute_physical_rows'])@u-np.asarray(t['Omega_QP_absolute_physical_rhs']),atol=1e-12)

    def test_coordinate_type(self):
        cfg,model,d=self.envs['Strong'][:3]
        ep=np.array([.02,.1]); en=ep/cfg.state_scale
        np.testing.assert_allclose((d.k@en)*cfg.input_scale,self.support['K_physical']@ep,atol=1e-12)
        np.testing.assert_allclose(model.physical_input(d.v_ref+.01)-cfg.robust_economic_reference_input,.01*cfg.input_scale,atol=1e-12)
        self.assertGreater(cfg.robust_economic_reference_state[0],25)

    def test_reference_reload(self):
        envs=audit.environments()
        for name,env in envs.items():
            np.testing.assert_array_equal(env[0].robust_economic_reference_state,self.envs[name][0].robust_economic_reference_state)
            np.testing.assert_array_equal(env[2].u_upper_tight,self.envs[name][2].u_upper_tight)

    def test_no_double_tightening(self):
        s=self.support
        np.testing.assert_allclose(s['robust_upper']-s['runtime_nominal_upper'],s['KZ_runtime_support_max'],rtol=0,atol=1e-9)
        self.assertGreater(np.min(np.abs(s['runtime_nominal_upper']-(s['robust_upper']-2*s['KZ_runtime_support_max']))),1)

    def test_Strong_Conservative_comparison(self):
        ds,dc=self.envs['Strong'][2],self.envs['Primary_Conservative'][2]
        for key in ('k','w_vertices','rpi_boundary','u_lower_tight','u_upper_tight','robust_input_lower','robust_input_upper'):
            np.testing.assert_array_equal(getattr(ds,key),getattr(dc,key))
        self.assertLess(self.details['Primary_Conservative']['nominal_reference_margin_upper'][1],2e-8)
        self.assertGreater(self.details['Primary_Conservative']['robust_actual_margin_upper'][1],30)

    def test_audit_does_not_mutate_production(self):
        self.assertEqual(self.hashes,audit.protected_hashes())
        for env in self.envs.values():
            before=deepcopy(env[0].input_upper)
            audit.controller_at(env,[0,0])
            np.testing.assert_array_equal(before,env[0].input_upper)

    def test_mapping_extrema_real_controller(self):
        for detail in self.details.values():
            self.assertGreater(np.min(detail['actual_raw_actor_applied_residual_upper']),0)
            self.assertLess(np.max(detail['actual_raw_actor_applied_residual_lower']),0)
            for probe in detail['actual_raw_actor_extreme_probes']:
                self.assertTrue(probe['QP_feasible'])
                self.assertLess(probe['reconstruction_error'],1e-6)
                self.assertLess(probe['final_QP_gap'],1e-8)

    def test_boundary_enclosure_difference_not_hidden(self):
        gap=self.support['polygon_vs_design_support_max_difference']
        self.assertGreater(gap[0],1e-5)
        self.assertLess(abs(gap[1]),1e-6)

    def test_feasible_candidate_not_same_as_projection_feasible(self):
        env=self.envs['Strong']; cfg,model,d=env[:3]
        h,b=self.details['Strong']['QP_nominal_rows'],self.details['Strong']['QP_nominal_rhs']
        u=cfg.robust_economic_reference_input.copy();u[1]=300
        un=model.normalized_input(u)
        self.assertGreater(np.max(h@un-b),1e-9)
        safe,exists=project_qp_2d(un,h,b)
        self.assertTrue(exists)
        self.assertLessEqual(model.physical_input(safe)[1],model.physical_input(d.u_upper_tight)[1]+1e-7)


if __name__=='__main__': unittest.main()
