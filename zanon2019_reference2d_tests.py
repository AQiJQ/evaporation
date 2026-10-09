"""Reference2D: frozen geometry, nearest non-economic choice and fail-closed selection."""
from unittest import TestCase, mock
import numpy as np

from . import zanon2019_reference2d as stage


class LinearSteady:
    def steady_input(self, state):
        x, p = state
        return np.array([2*x+3*p, 5*x-2*p+20])


class Reference2DTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.strong = stage.v2.guard(stage.exp.args_for())

    def test_central_difference_orientation(self):
        np.testing.assert_allclose(stage.jacobian(LinearSteady(), [2, 3]), [[2, 3], [5, -2]], atol=1e-9)

    def test_difference_invalid_steps_rejected(self):
        for steps in ([0, 1], [float('nan'), 1], [-1, 1]):
            with self.assertRaises(ValueError): stage.jacobian(LinearSteady(), [2, 3], steps)

    def test_predefined_grid_and_no_final_seed(self):
        p = stage.make_protocol(self.strong)
        self.assertEqual(p['deltas_X2'], [.01, .02, .03, .05, .1, .15, .2])
        self.assertEqual(p['validation_seeds'], list(range(420000, 420010)))
        self.assertFalse(p['training_performed']); self.assertFalse(p['final_test_performed'])
        self.assertEqual(p['difference_step_sizes'].tolist(), [1e-4, 1e-4])
        self.assertGreater(p['P2_range'][0], 40); self.assertLess(p['P2_range'][1], 80)

    def test_sensitivity_consistent(self):
        model = self.strong[1]; ref = self.strong[0].robust_economic_reference_state
        j = stage.jacobian(model, ref)
        for s in (.5, 2): np.testing.assert_allclose(stage.jacobian(model, ref, stage.STEP*s), j, rtol=1e-6, atol=1e-7)
        self.assertGreater(j[1, 0], 0); self.assertLess(j[1, 1], 0)

    def test_two_dimensional_reference_preserves_every_geometry_array(self):
        e = stage.environment(self.strong, [25.4, 50.14])
        for name in ('a', 'b', 'k', 'w_vertices', 'rpi_boundary', 'invariant_lower', 'invariant_upper',
                     'u_lower_tight', 'u_upper_tight', 'robust_state_lower', 'robust_state_upper'):
            np.testing.assert_array_equal(getattr(e[2], name), getattr(self.strong[2], name))
        np.testing.assert_allclose(e[3]+e[2].z_ref, self.strong[3]+self.strong[2].z_ref, atol=1e-12)
        np.testing.assert_array_equal(e[0].linearization_state, self.strong[0].linearization_state)
        np.testing.assert_array_equal(e[0].linearization_input, self.strong[0].linearization_input)
        np.testing.assert_allclose(e[1].derivative(e[0].robust_economic_reference_state,
            e[0].robust_economic_reference_input, e[0].disturbance_nominal), 0, atol=1e-9)

    def test_strong_reference_remains_bitwise_identical(self):
        e = stage.environment(self.strong, self.strong[0].robust_economic_reference_state)
        np.testing.assert_array_equal(e[2].affine, self.strong[2].affine)
        np.testing.assert_array_equal(e[2].nominal_policy_offset, self.strong[2].nominal_policy_offset)

    def test_root_interval_obeys_both_input_channels(self):
        interval, reason = stage.input_interval(LinearSteady(), 2, [0, 10], [7, 10], [13, 26], 1e-10, 1e-8)
        self.assertIsNone(reason)
        self.assertAlmostEqual(interval[0], 2+5e-9, places=8)
        self.assertAlmostEqual(interval[1], 3-1e-8/3, places=8)
        for p in interval:
            u = LinearSteady().steady_input([2, p])
            self.assertTrue(np.all(u >= [7, 10])); self.assertTrue(np.all(u <= [13, 26]))

    def test_empty_input_interval_is_rejected(self):
        interval, _ = stage.input_interval(LinearSteady(), 2, [0, 1], [7, 10], [13, 26], 1e-10, 1e-8)
        self.assertIsNone(interval)

    def test_nearest_P2_is_not_economic_optimum(self):
        p = stage.make_protocol(self.strong)
        e, r = stage.nearest_reference(self.strong, .01, p)
        self.assertTrue(r['reference_feasible'], r.get('reject_reason'))
        interval = r['input_feasible_P2_interval']
        self.assertAlmostEqual(r['P2_ref'], np.clip(50.125, *interval), places=8)
        self.assertGreater(r['P2_ref'], 50.125)
        self.assertLess(r['F200_tightened_margin'], 1e-6)
        self.assertGreaterEqual(r['minimum_residual_authority'], .05-1e-8)
        self.assertTrue(r['gates']['online_QP_initialization'])

    def test_root_full_gate_failure_cannot_enter_baseline(self):
        p = stage.make_protocol(self.strong); p['P2_fallback_grid_points'] = 3
        with mock.patch.object(stage, 'full_check', return_value=(self.strong, dict(passed=False, limiting_gates=['nominal_W_sample_coverage']))):
            e, r = stage.nearest_reference(self.strong, .01, p)
        self.assertIsNone(e); self.assertFalse(r['reference_feasible'])
        self.assertEqual(r['nearest_input_limiting_gates'], ['nominal_W_sample_coverage'])

    def test_eligibility_requires_both_margin_and_economic_penalty(self):
        strong = [dict(validation_seed=s, minimum_X2_margin=.2, **{k: 0 for k in stage.old.SAFETY}) for s in stage.old.DEV]
        candidate = [dict(r, minimum_X2_margin=.21) for r in strong]
        p = dict(margin_increase_tolerance=1e-4, penalty_numerical_guard_abs=2e-4)
        row = dict(reference_feasible=True, baseline_evaluation_status='completed', conservatism_penalty_abs=1.)
        self.assertTrue(stage.eligible(row, candidate, strong, p))
        for penalty in (0., -1., 2e-4):
            self.assertFalse(stage.eligible(dict(row, conservatism_penalty_abs=penalty), candidate, strong, p))
        candidate[0]['minimum_X2_margin'] = .19
        self.assertFalse(stage.eligible(row, candidate, strong, p))

    def test_frozen_hash_names_do_not_include_new_module(self):
        self.assertNotIn('zanon2019_reference2d.py', stage.exp.code_hashes())
        self.assertNotIn('zanon2019_reference2d.py', stage.old.source_hashes())
        stage.exp.locked(stage.exp.ROOT); stage.old.locked(stage.old.ROOT)

    def test_oracle_gate_rejects_unselected_or_SAC_dependent_reference(self):
        self.assertFalse(stage.oracle_gate(dict(Conservative_metrics=None)))
        s = dict(Conservative_metrics=dict(valid_Conservative_candidate=True, conservatism_penalty_abs=1.),
                 selected_before_oracle=True, selected_before_SAC_training=True, selection_used_SAC_results=False)
        self.assertTrue(stage.oracle_gate(s))
        self.assertFalse(stage.oracle_gate(dict(s, selection_used_SAC_results=True)))
        self.assertFalse(stage.oracle_gate(dict(s, selected_before_oracle=False)))

    def test_no_training_entry(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(stage))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(c.func, ast.Attribute) and c.func.attr == 'train' for c in calls))
        self.assertFalse(any(k.arg == 'training' and isinstance(k.value, ast.Constant) and k.value.value is True
                             for c in calls for k in c.keywords))
