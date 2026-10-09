"""No-training/no-test-seed-access regression tests for final orchestration."""
import copy
from contextlib import contextmanager
import uuid
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from . import zanon2019_final_experiment as f


def fake_metrics():
    m={v:1. for v in f.RATIO_METRICS.values()}
    m.update(J_econ=100.,X2_margin_min=.2,physical_state_violation_count=0,
        physical_input_violation_count=0,X2_violation_count=0,P2_physical_violation_count=0,
        QP_infeasible_count=0,Omega_exit_count=0,robust_region_violation_count=0,W_exceedance_rate=.98)
    return m


def selected():
    return [dict(training_seed=s,checkpoint_episode=5,status='successful',locked_checkpoint_path='dummy.pth') for s in f.SEEDS]


@contextmanager
def safe_test_directory():
    # Windows sandbox temp dirs created with restrictive mkdtemp permissions
    # may not be readable. Use an ordinary workspace-local test directory.
    # Synthetic results are outside evaporation_safe_sac, never audited as real
    # final-test execution. Preserve them instead of recursively deleting paths.
    path=f.REPO/'tmp'/('final_protocol_test_'+uuid.uuid4().hex)
    path.mkdir(parents=True,exist_ok=False)
    yield path


def fake_rows():
    rows=[]
    metrics=fake_metrics()
    for s,g in zip(f.SEEDS,(.01,.02,-.01)):
        for i in range(50):
            p=dict(metrics,J_econ=100*(1-g/100))
            row=f.paired_row(s,430000+i,dict(status='completed',metrics=metrics),dict(status='completed',metrics=p),selected()[f.SEEDS.index(s)])
            rows.append(row)
    return rows


class FinalProtocolTests(unittest.TestCase):
    def test_existing_config_exact_reuse(self):
        p=f.assert_frozen(f.ROOT)
        m=f.validate_run(f.SOURCE42,42,p)
        self.assertEqual(m['steps'],1000);self.assertEqual(m['episodes'],100)
        self.assertEqual(m['weights'],p['template_fingerprint']['weights'])
        self.assertEqual(m['validation_seeds'],list(range(420000,420010)))

    def test_only_training_seed_can_vary(self):
        m=f.read_json(f.SOURCE42/'experiment_manifest.json')
        changed=copy.deepcopy(m);changed['seed']=2027;changed['frozen_experiment_config']['seed']=2027
        self.assertEqual(f.fingerprint(m),f.fingerprint(changed))
        changed['frozen_SAC_config']['actor_lr']=1e-4
        self.assertNotEqual(f.fingerprint(m),f.fingerprint(changed))

    def test_final_v1_validation_selector(self):
        p=f.assert_frozen(f.ROOT)
        chosen=f.select_run(f.SOURCE42,42,p)
        existing=f.read_json(f.ROOT/'seed42_validation_lock.json')
        self.assertEqual(chosen['checkpoint_episode'],existing['checkpoint_episode'])
        self.assertEqual(chosen['global_step'],existing['global_step'])
        self.assertGreaterEqual(chosen['global_step'],p['template_fingerprint']['warmup_steps'])
        self.assertEqual(chosen['selection_rule_version'],'final_v1')
        self.assertGreater(chosen['validation_metrics']['mean_economic_improvement_pct'],0)
        self.assertGreaterEqual(chosen['validation_metrics']['worst_seed_economic_improvement_pct'],0)

    def test_missing_training_blocks_lock_before_test(self):
        with safe_test_directory() as tmp:
            root=Path(tmp);p=f.assert_frozen(f.ROOT)
            p=copy.deepcopy(p);p['run_dirs']={str(s):str(root/f's{s}') for s in f.SEEDS}
            with patch.object(f,'assert_frozen',return_value=p),patch.object(f,'load_locked_seed42',return_value=f.read_json(f.ROOT/'seed42_validation_lock.json')),patch.object(f,'sample_disturbance_path') as draw:
                with self.assertRaisesRegex(RuntimeError,'pending'):f.lock_policies(root)
                draw.assert_not_called()
                self.assertFalse((root/'locked_final_policies.json').exists())

    def test_no_reselection_after_test(self):
        with safe_test_directory() as tmp:
            root=Path(tmp);f.save(root/'test_started.json',{})
            with patch.object(f,'assert_frozen',return_value={}),patch.object(f,'select_run') as choose:
                with self.assertRaisesRegex(RuntimeError,'after test'):f.lock_policies(root)
                choose.assert_not_called()

    def test_final_passive_observer_never_safety_aborts(self):
        ctrl=f.FinalEvidenceController.__new__(f.FinalEvidenceController);ctrl.evidence=[]
        with patch.object(f.AuthorityController,'act',return_value=(np.array([90.,450.]),
             dict(qp_feasible=False,in_Omega=False,mode='outside_certified_domain'))):
            ctrl.act(np.array([24.,30.]),np.array([1.,1.]))
            ctrl.audit_next_state(np.array([23.,29.]))
        self.assertFalse(ctrl.evidence[0]['qp_feasible'])
        self.assertEqual(ctrl.evidence[0]['next_state'],[23.,29.])

    def test_passive_observer_matches_real_one_step_mapping(self):
        cfg,model,design,omega,domain,_=f.make_setup(1,42)
        cfg.stochastic_residual_scale=.1
        old=f.AuthorityController(cfg,model,design,omega,domain)
        new=f.FinalEvidenceController(cfg,model,design,omega,domain)
        state=np.asarray(cfg.robust_economic_reference_state)
        for action in (np.array([0.,0.]),np.array([.7,-.8])):
            # Each probe starts with the same controller state. It is a one-step
            # nominal-state implementation test, not a reserved stochastic path.
            old=f.AuthorityController(cfg,model,design,omega,domain)
            new=f.FinalEvidenceController(cfg,model,design,omega,domain)
            u0,info0=old.act(state,action);u1,info1=new.act(state,action)
            np.testing.assert_array_equal(u0,u1)
            self.assertEqual(info0['qp_feasible'],info1['qp_feasible'])
            np.testing.assert_array_equal(old.actor_observation_extra(state),new.actor_observation_extra(state))
            new.audit_next_state(model.step(state,u1,cfg.disturbance_nominal))

    def test_bad_economics_and_safety_are_not_filtered(self):
        b=fake_metrics();p=dict(b,J_econ=101.,QP_infeasible_count=2,Omega_exit_count=3,
                              physical_state_violation_count=1)
        r=f.paired_row(42,430000,dict(status='completed',metrics=b),dict(status='completed',metrics=p),selected()[0])
        self.assertEqual(r['improvement_pct'],-1.)
        self.assertEqual(r['QP_infeasible'],2);self.assertEqual(r['physical_violation'],1)

    def test_unavailable_does_not_become_zero_anomalies(self):
        r=f.paired_row(42,430000,dict(status='completed',metrics=fake_metrics()),
                       dict(status='program_failure_incomplete',metrics=None),selected()[0])
        self.assertNotIn('physical_violation',r)
        self.assertNotIn('improvement_pct',r)

    def test_statistics_have_two_levels(self):
        per,cross,detail=f.summaries(fake_rows(),selected())
        self.assertTrue(all(p['completed_test_seed_count']==50 for p in per))
        g=next(p for p in cross if p['metric']=='improvement_pct')
        self.assertEqual(g['n'],3)
        self.assertAlmostEqual(g['mean'],(.01+.02-.01)/3)
        self.assertGreater(g['std'],0)
        self.assertAlmostEqual(g['CI95_upper']-g['mean'],4.302652729911275*g['std']/np.sqrt(3))
        self.assertTrue(all(p['improvement_pct_n']==50 for p in per))

    def test_missing_seed_retained_and_coverage_exposed(self):
        rows=fake_rows();rows[0]=dict(training_seed=42,test_seed=430000,status='program_failure_incomplete',baseline_status='completed')
        per,cross,_=f.summaries(rows,selected())
        self.assertEqual(per[0]['test_seed_count'],50)
        self.assertEqual(per[0]['completed_test_seed_count'],49)
        self.assertFalse(per[0]['complete_coverage'])
        self.assertIsNone(per[0]['improvement_pct_CI95_lower'])

    def test_fifty_baselines_not_one_hundred_fifty(self):
        # Pure orchestration mock: no reserved disturbance is sampled and no
        # nonlinear rollout is executed. Only integer identifiers are exercised.
        with safe_test_directory() as tmp:
            root=Path(tmp);f.save(root/'locked_final_policies.json',{})
            f.save(root/'independent_test_authorization.json',dict(approved_for_independent_final_test=True,
                policy_lock_sha256=f.digest(root/'locked_final_policies.json')))
            lock=dict(policies=selected());result=dict(status='completed',metrics=fake_metrics(),records=[],evidence=[])
            with patch.object(f,'verify_policy_lock',return_value=lock),patch.object(f,'seed_usage_audit'),\
                 patch.object(f,'make_setup',return_value=(SimpleNamespace(),None,None,None,None,None)),\
                 patch.object(f,'make_agent'),patch.object(f,'load_actor'),patch.object(f,'archive_rollout'),\
                 patch.object(f,'sample_disturbance_path',return_value=(np.zeros((1000,4)),{})),\
                 patch.object(f,'rollout',return_value=result) as roll,patch.object(f,'report'),patch('builtins.print'):
                f.final_test(root,'cpu')
            self.assertEqual(roll.call_count,200)
            self.assertEqual(sum(isinstance(c.args[5],f.ZeroResidualPolicy) for c in roll.call_args_list),50)
            self.assertEqual(len(f.csv_rows(root/'final_test_per_policy.csv')),150)
            self.assertEqual(len(f.csv_rows(root/'final_test_baseline.csv')),50)

    def test_reserved_seed_audit_passed_not_tested(self):
        audit=f.read_json(f.ROOT/'seed_usage_audit.json')
        self.assertTrue(audit['passed']);self.assertEqual(audit['collisions'],[])
        self.assertFalse((f.ROOT/'test_started.json').exists())
        self.assertFalse((f.ROOT/'final_test_per_policy.csv').exists())

    def test_no_final_test_access_without_later_authorization(self):
        with patch.object(f,'sample_disturbance_path') as draw:
            with self.assertRaisesRegex(RuntimeError,'not been authorized'):f.final_test(f.ROOT,'cpu')
            draw.assert_not_called()

    def test_no_candidate_is_never_replaced_by_diagnostic(self):
        p=f.assert_frozen(f.ROOT)
        with patch.object(f,'final_selection_scan',return_value=([],None)),patch.object(f,'write_csv'):
            r=f.select_run(f.SOURCE42,42,p)
        self.assertEqual(r['status'],'no_stable_positive_economic_safe_checkpoint')
        self.assertIsNone(r['selected_checkpoint'])
        self.assertFalse(r['eligible_final_checkpoint'])
        self.assertIsNotNone(r['best_empirical_safe_checkpoint'])

    def test_permanent_rule_matches_source(self):
        self.assertEqual(f.read_json(f.ROOT/'final_checkpoint_selection_rule.json'),f.rule_document())

    def test_seed42_reporting_does_not_rescan_or_modify_lock(self):
        before=f.digest(f.ROOT/'seed42_validation_lock.json')
        audit_before=f.digest(f.ROOT/'seed42_final_selection_audit.csv')
        with patch.object(f,'final_selection_scan',side_effect=AssertionError('seed42 reselected')):
            p=f.load_locked_seed42(f.ROOT,f.assert_frozen(f.ROOT))
            from .zanon2019_final_validation_report import enrich
            metrics,_=enrich(p)
        self.assertIn('X2_std_ratio',metrics);self.assertIn('G_X2_ratio',metrics)
        self.assertEqual(before,f.digest(f.ROOT/'seed42_validation_lock.json'))
        self.assertEqual(audit_before,f.digest(f.ROOT/'seed42_final_selection_audit.csv'))


if __name__=='__main__':unittest.main()
