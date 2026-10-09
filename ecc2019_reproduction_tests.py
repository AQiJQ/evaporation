"""No-training checks of ECC NLP/model/metrics/data guards."""
import unittest
import numpy as np
from .ecc2019_nmpc import ECCNMPC,ECCSettings,initial_theta,theta_certificate,symbolic_step
from .ecc2019_reproduction import common_model,metrics,read_ours_archive,SOURCES,comparison
from pathlib import Path
from unittest.mock import patch


class PureTests(unittest.TestCase):
    def test_parameter_PD_and_bounds(self):
        p=initial_theta();self.assertTrue(theta_certificate(p)['passed'])
        p[12]=-1;self.assertFalse(theta_certificate(p)['passed'])

    def test_terminal_safety_not_omitted(self):
        model=common_model();d=np.tile(model.cfg.disturbance_nominal,(2,1))
        r=metrics(np.array([[25.39,50.125],[25.4,50.1],[24.9,50.1]]),np.full((2,2),200.),np.ones(2),d,model)
        self.assertEqual(r['X2_violation_count'],1)
        self.assertAlmostEqual(r['X2_max_violation_magnitude'],.1)
        self.assertEqual(r['state_safety_samples'],3)

    def test_g_is_not_named_Hinf(self):
        model=common_model();d=np.tile(model.cfg.disturbance_nominal,(2,1))+1
        r=metrics(np.array([[25.39,50.125]]*3),np.full((2,2),200.),np.ones(2),d,model)
        self.assertEqual(r['G_X2_emp'],0.)
        self.assertFalse(any('Hinf' in key for key in r))

    def test_archive_paired_nonlinear_path(self):
        model=common_model();base=read_ours_archive(SOURCES['proposed_alpha010_ep100'],420000,'baseline',model)
        sac=read_ours_archive(SOURCES['proposed_alpha020_rewardcal_ep100'],420000,'SAC',model)
        np.testing.assert_array_equal(base['disturbances'],sac['disturbances'])
        np.testing.assert_array_equal(base['states'][0],sac['states'][0])
        self.assertEqual(len(sac['states']),1001)

    def test_naive_artifact_not_comparable_as_RL(self):
        import json
        with patch('evaporation.ecc2019_reproduction.validate_sources',return_value={}),patch.object(Path,'read_text',return_value=json.dumps({'label':'ECC2019_reproduction_with_documented_assumptions','status':'naive'})):
            with self.assertRaisesRegex(ValueError,'Not a successfully trained'):
                comparison(Path('.'),Path('mock_naive.json'))


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import casadi
        except ImportError:raise unittest.SkipTest('Optional CasADi solver unavailable in this environment')
        cls.model=common_model();cls.solver=ECCNMPC(cls.model)

    def test_symbolic_RK4_matches_same_nonlinear_plant(self):
        import casadi as ca
        x=ca.MX.sym('x',2);u=ca.MX.sym('u',2)
        f=ca.Function('test_step',[x,u],[symbolic_step(ca,self.model.cfg,x,u)])
        for state,control in (([25.39,50.125],[194.861443,216.276654]),([28,53],[200,250])):
            actual=np.array(f(state,control)).ravel();expected=self.model.step(state,control,self.model.cfg.disturbance_nominal)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)

    def test_Q_equals_V_at_greedy_and_gradient(self):
        x=np.array([25.39,50.125]);v=self.solver.solve(x)
        q=self.solver.solve(x,first_action=v['action'],gradient=True)
        self.assertLess(abs(v['value']-q['value']),1e-4)
        self.assertTrue(np.all(v['action']>=100));self.assertTrue(np.all(v['action']<=400))
        self.assertEqual(q['gradient'].shape,(33,))
        self.assertTrue(np.isfinite(q['gradient']).all())
        np.testing.assert_array_equal(v['first_slack'],np.zeros(4))
        p=initial_theta();eps=1e-4
        p[26]+=eps;plus=self.solver.solve(x,p,first_action=v['action'])['value'];p[26]-=2*eps
        minus=self.solver.solve(x,p,first_action=v['action'])['value']
        self.assertAlmostEqual((plus-minus)/(2*eps),q['gradient'][26],places=3)

    def test_ECC_state_relaxation_is_not_replaced_by_hard_safety(self):
        solved=self.solver.solve(np.array([24.9,50.125]))
        self.assertAlmostEqual(solved['first_slack'][0],.1,places=7)
        self.assertTrue(np.all(solved['action']>=100))
        self.assertTrue(np.all(solved['action']<=400))


if __name__=='__main__':unittest.main()
