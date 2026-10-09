"""No-training tests for the independent frozen-policy audit."""
import unittest
from types import SimpleNamespace
import numpy as np
from .zanon2019_sac_optimization_audit import correlation, frozen_rollout, paired_policy_summary


class AuditTests(unittest.TestCase):
    def test_spearman_ties(self):
        self.assertAlmostEqual(correlation([1,2,2,4],[8,6,6,0],rank=True),-1.)

    def test_conditional_q_excludes_first_action_entropy(self):
        class Env:
            cfg=SimpleNamespace(robust_economic_reference_state=np.zeros(2))
            def obs(self): return np.zeros(2)
            def step(self,a,w):
                return dict(state=np.zeros(2),input=np.zeros(2),reward=1.,raw_eval_reward=1.,cost=2.,
                            physical_state=0,physical_input=0,QP_infeasible=0,Omega_exit=0,robust_region=0)
        class Policy:
            def action(self,obs): return np.ones(2),-2.
        result,*_=frozen_rollout(Env(),Policy(),np.zeros((3,4)),.5,.1,first_action=np.zeros(2))
        self.assertAlmostEqual(result['discounted_environment_return'],1.75)
        self.assertAlmostEqual(result['discounted_entropy_objective'],1.+.5*1.2+.25*1.2)

    def test_action_repeats_do_not_multiply_disturbance_samples(self):
        rows=[]
        for seed in (1,2):
            for kind,values in (('zero',[1.]),('deterministic',[2.]),('stochastic',[3.,5.])):
                for value in values:
                    rows.append(dict(episode=100,seed=seed,policy=kind,completed=True,
                        **{m:value for m in ('environment_return','discounted_entropy_objective','J_econ','X2_IAE','P2_IAE','P100_TV','F200_TV')}))
        result=paired_policy_summary(rows)[0]
        self.assertEqual(result['paired_disturbance_seed_count'],2)
        self.assertEqual(result['environment_return_stochastic_minus_deterministic']['mean'],2.)


if __name__=='__main__': unittest.main()
