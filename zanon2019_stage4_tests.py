"""Boundary immutability, exact economics, full-budget oracle and local gates."""
from unittest import TestCase, mock
from types import SimpleNamespace
import inspect
import ast
import numpy as np
from . import zanon2019_stage4 as stage
from . import zanon2019_stage4_metrics as m


class Stage4Tests(TestCase):
    def test_primary_is_frozen_first_candidate(self):
        s = stage.selection()
        self.assertEqual(s['selected_delta_X2'], .01)
        np.testing.assert_allclose(s['selected_x_ref'], [25.4, 50.13307219798317], rtol=0, atol=1e-13)
        self.assertFalse(s['selection_used_SAC_results'])

    def test_boundary_trial_does_not_modify_geometry_or_reference(self):
        before = stage.previous.snapshot(stage.previous.ROOT)
        trial = stage.boundary_trial()
        self.assertTrue(trial['feasible']); self.assertTrue(trial['QP_init_feasible'])
        self.assertGreater(trial['raw_margin'], 0)
        self.assertLess(trial['raw_margin'], 2e-8)
        self.assertEqual(before, stage.previous.snapshot(stage.previous.ROOT))

    def test_recovery_beyond_Strong_and_not_clipped(self):
        result, series = m.economic_triplet([100], [110], [90])
        self.assertEqual(result['economic_recovery_ratio_pct'], 200)
        self.assertEqual(result['economic_improvement_vs_strong_pct'], 10)
        self.assertEqual(result['economic_difference_vs_strong_abs'], -10)
        self.assertAlmostEqual(result['economic_improvement_pct'], 100*20/110)
        self.assertEqual(result['economic_improvement_pct'], series['G_cum'][-1])

    def test_nonpositive_penalty_is_NA(self):
        for c in ([99], [100]):
            self.assertIsNone(m.economic_triplet([100], c, [98])[0]['economic_recovery_ratio_pct'])
            self.assertIsNone(m.economic_triplet([100], c, [98])[0]['recovery_break_even_step'])

    def test_break_even_full_horizon_not_running_penalty(self):
        r, series = m.economic_triplet([100]*4, [101]*4, [100, 100, 98, 100])
        self.assertEqual(r['recovery_break_even_step'], 3)
        self.assertEqual(r['strong_outperformance_step'], 3)
        np.testing.assert_array_equal(series['strong_cumulative_cost'], [100, 200, 300, 400])
        np.testing.assert_array_equal(series['candidate_cumulative_cost'], [100, 200, 298, 398])

    def test_break_even_missing_not_zero(self):
        r, _ = m.economic_triplet([100]*4, [101]*4, [100.9]*4)
        self.assertIsNone(r['recovery_break_even_step'])
        self.assertIsNone(r['strong_outperformance_step'])

    def test_instantaneous_and_cumulative_signs(self):
        r, s = m.economic_triplet([100]*3, [110]*3, [109, 111, 108])
        self.assertEqual(r['economic_win_fraction'], 2/3)
        np.testing.assert_allclose(s['instantaneous_economic_advantage_pct'], 100*np.array([1, -1, 2])/110)
        self.assertAlmostEqual(s['G_cum'][-1], 100*2/330)

    def test_nonfinite_and_incomplete_costs_rejected(self):
        for x in ([float('nan')], [float('inf')], []):
            with self.assertRaises(ValueError): m.economic_triplet([100], [101], x)

    def test_F200_actual_not_confused_with_Z_nominal(self):
        model = SimpleNamespace(physical_input=lambda v: np.array(v), cfg=SimpleNamespace(input_scale=np.ones(2)))
        d = SimpleNamespace(u_upper_tight=np.array([20, 10]))
        info = dict(mode='Z_mode_existing_controller', action_final_coordinate=np.array([5, 9]),
                    action_candidate_coordinate=np.array([5, 9]), action_safe_rows=np.array([[1, 0], [0, 1]]),
                    action_safe_bounds=np.array([20, 10]))
        r = m.f200_step(model, d, [5, 21], info)
        self.assertEqual(r['Z_nominal_F200_tightened_margin'], 1)
        self.assertEqual(r['actual_F200_comparison_to_nominal_tightened_upper_margin'], -11)
        self.assertFalse(r['qp_F200_input_upper_active'])

    def test_Omega_tightened_margin_not_applicable(self):
        model = SimpleNamespace(physical_input=lambda v: np.array(v), cfg=SimpleNamespace(input_scale=np.ones(2)))
        d = SimpleNamespace(u_upper_tight=np.array([20, 10]))
        info = dict(mode='Omega_safe_one_step_QP', action_final_coordinate=np.array([5, 4]),
                    action_candidate_coordinate=np.array([5, 4]), action_safe_rows=np.array([[1, 0], [0, 1]]),
                    action_safe_bounds=np.array([20, 4]))
        r = m.f200_step(model, d, [5, 14], info)
        self.assertIsNone(r['Z_nominal_F200_tightened_margin'])
        self.assertTrue(r['qp_F200_input_upper_active'])
        self.assertFalse(r['qp_Z_F200_tightened_upper_active'])
        self.assertIsNone(m.f200_summary([r])['minimum_F200_tightened_margin'])

    def test_training_gate_all_prerequisites(self):
        witness = dict(safety_passed=True, no_nonfinite=True, economic_improvement_pct=.01,
                       worst_validation_improvement_pct=0., reasonable_control_activity=True)
        summary = dict(completed_all_512=True, total_candidate_count=512, training_witness=witness)
        self.assertTrue(m.training_gate(dict(numerically_stable=True), summary))
        self.assertFalse(m.training_gate(dict(numerically_stable=False), summary))
        self.assertFalse(m.training_gate(dict(numerically_stable=True), dict(summary, total_candidate_count=511)))
        for k, value in (('safety_passed', False), ('no_nonfinite', False), ('economic_improvement_pct', 0.),
                         ('worst_validation_improvement_pct', -.001), ('reasonable_control_activity', False)):
            self.assertFalse(m.training_gate(dict(numerically_stable=True), dict(summary, training_witness=dict(witness, **{k:value}))))

    def test_oracle_must_finish_512_and_has_no_early_stop(self):
        with self.assertRaises(RuntimeError): stage.finish_summary([], dict(numerically_stable=True))
        tree = ast.parse(inspect.getsource(stage.oracle))
        self.assertFalse(any(isinstance(n, ast.Break) for n in ast.walk(tree)))
        self.assertIn('range(512)', inspect.getsource(stage.oracle))

    def test_candidate_generation_preserves_frozen_grid_and_segments(self):
        p = dict(oracle_segments=[10, 5])
        rng = np.random.default_rng(190019)
        segment, actions = stage.actions_for(0, p, rng, [])
        self.assertEqual(segment, 10); np.testing.assert_array_equal(actions, np.tile([-1, -1], (100, 1)))
        segment, actions = stage.actions_for(12, p, rng, [])
        np.testing.assert_array_equal(actions, np.zeros((100, 2)))
        segment, actions = stage.actions_for(256, p, rng, [])
        self.assertEqual(segment, 5); self.assertEqual(actions.shape, (200, 2))
        self.assertTrue(np.all(np.abs(actions) <= 1))

    def test_control_quality_band_reuses_current_1p10(self):
        self.assertEqual(m.ACTIVITY_LIMIT, 1.1)
        row = {k:1.1 for k in m.ACTIVITY_KEYS}
        self.assertTrue(m.reasonable_activity([row]))
        self.assertFalse(m.reasonable_activity([dict(row, F200_RMS_du_ratio=1.100001)]))

    def test_new_sources_do_not_change_old_lock_globs(self):
        self.assertNotIn('zanon2019_stage4.py', stage.previous.code_hashes())
        self.assertNotIn('zanon2019_stage4.py', stage.exp.code_hashes())
        stage.previous.locked(); stage.old.locked(stage.old.ROOT)

    def test_local_windows_interpreter_and_development_seeds(self):
        self.assertEqual(list(stage.DEV), list(range(420000, 420010)))
        self.assertIn('sys.executable', inspect.getsource(stage.boundary_audit))
        self.assertNotIn('training=True', inspect.getsource(stage.rollout))

    def test_readonly_observer_preserves_closed_loop(self):
        args = stage.exp.args_for(); args.steps = 3
        strong = stage.v2.guard(args)
        env, check = stage.previous.full_check(strong, stage.selection()['selected_x_ref'], [25.39, 50.125])
        path, _ = stage.previous.sample_disturbance_path(env[0], 'zanon2019_stochastic', 420000, 3,
            stage.previous.PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
        a, _ = stage.old.rollout(env, path, 420000, stage.old.v1.ConstantPolicy([-.2, .1]), [25.39, 50.125])
        b, diagnostic = stage.rollout(env, path, 420000, stage.old.v1.ConstantPolicy([-.2, .1]), [25.39, 50.125])
        for x, y in zip(a, b):
            for k in ('state', 'control', 'raw_action', 'w_hat'): np.testing.assert_array_equal(x[k], y[k])
            self.assertEqual(x['economic_cost'], y['economic_cost'])
        self.assertEqual(len(diagnostic), 3)

    def test_local_training_cannot_bypass_oracle_or_boundary_gate(self):
        from . import zanon2019_stage4_train as local
        with mock.patch.object(stage, 'locked', return_value={}), mock.patch.object(stage.old, 'load',
                side_effect=[dict(numerically_stable=False), dict(completed_all_512=False)]):
            with self.assertRaises(RuntimeError): local.require_training()

    def test_local_pilot_checkpoint_requires_post_warmup_all_paths_and_quality(self):
        from . import zanon2019_stage4_train as local
        rows = [dict(economic_improvement_pct=.01, minimum_F200_tightened_margin=1e-8,
            **{k:0 for k in stage.old.SAFETY}, **{k:1. for k in m.ACTIVITY_KEYS}) for _ in range(10)]
        self.assertFalse(local.qualified(rows, 4999, 5000))
        self.assertFalse(local.qualified(rows[:9], 5000, 5000))
        self.assertTrue(local.qualified(rows, 5000, 5000))
        rows[0]['economic_improvement_pct'] = -.001
        self.assertFalse(local.qualified(rows, 5000, 5000))
