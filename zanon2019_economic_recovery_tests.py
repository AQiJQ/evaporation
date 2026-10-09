"""Economic definitions, reference-only geometry and phase/provenance guards."""
from __future__ import annotations

from pathlib import Path
import uuid
from unittest import TestCase, mock
import numpy as np

from . import zanon2019_economic_recovery as exp
from . import zanon2019_paired_v2 as v2
from . import zanon2019_paired_experiment as v1
from .zanon2019_economic_recovery_metrics import economic_series, recovery, headroom_utilization, pair_identity, aggregate
from .zanon2019_economic_recovery_reference import reference_environment
from .zanon2019_paired_v2_objective import paired_move, components


def records(cost=10., reference=(25.39, 50.125)):
    return [dict(state=np.array(reference), economic_cost=cost, control=np.array([195., 216.]),
                 disturbance=np.array([10., 5., 40., 25.]), qp_feasible=True, in_Omega_before=True)]


class EconomicRecoveryTests(TestCase):
    def test_pair_baseline_reference_identity(self):
        pair_identity(records(), records(), [25.39, 50.125])
        with self.assertRaises(ValueError): pair_identity(records(), records(), [25.44, 50.125])

    def test_disturbance_identity(self):
        b, r = records(), records(); r[0]["disturbance"][0] += .1
        with self.assertRaises(ValueError): pair_identity(b, r, [25.39, 50.125])

    def test_economic_reward_direction(self):
        raw = components(records(9), records(10), [100., 100.], [195., 216.])
        self.assertAlmostEqual(raw[0, 0], 1/200.)

    def test_instantaneous_direction(self):
        r = economic_series([10., 20., 30.], [9., 22., 30.])
        np.testing.assert_allclose(r["instantaneous_economic_advantage_pct"], [10., -10., 0.])
        np.testing.assert_allclose(r["delta_ell"], [-1., 2., 0.])

    def test_cumulative_endpoint_not_mean_instant(self):
        r = economic_series([10., 20.], [8., 19.])
        self.assertEqual(r["G_cum"][-1], r["economic_improvement_pct"])
        self.assertEqual(r["economic_improvement_pct"], 10.)
        self.assertNotEqual(r["economic_improvement_pct"], np.mean(r["instantaneous_economic_advantage_pct"]))

    def test_rolling20_full_window(self):
        r = economic_series(np.arange(1., 26.), np.arange(1., 26.)-1)
        self.assertEqual(r["G_roll20"][:19], [None]*19)
        self.assertAlmostEqual(r["G_roll20"][19], 100.*20/sum(range(1, 21)))
        self.assertAlmostEqual(r["G_roll20"][24], 100.*20/sum(range(6, 26)))

    def test_strict_win_fraction(self):
        self.assertEqual(economic_series([10., 10., 10.], [9., 10., 11.])["economic_win_fraction"], 1/3.)

    def test_applied_move_semantics(self):
        # Uses final physical u, not actor action/residual.
        np.testing.assert_allclose(paired_move([210., 220.], [200., 200.], [205., 205.], [200., 200.], [100., 100.]), [.0075, .0375])

    def test_reference_steady_consistency(self):
        strong = v2.guard(exp.args_for())
        cfg = strong[0]; original = cfg.robust_economic_reference_state.copy()
        env, report = reference_environment(strong, .05)
        self.assertTrue(report["gates"]["nonlinear_steady"])
        self.assertTrue(report["gates"]["affine_steady"])
        np.testing.assert_array_equal(cfg.robust_economic_reference_state, original)
        np.testing.assert_allclose(env[1].derivative(env[0].robust_economic_reference_state,
                                                   env[0].robust_economic_reference_input, env[0].disturbance_nominal), 0., atol=1e-9)

    def test_reference_sweep_does_not_modify_forbidden_arrays(self):
        strong = v2.guard(exp.args_for()); env, report = reference_environment(strong, .1)
        self.assertTrue(report["gates"]["frozen_arrays_identical"])
        self.assertTrue(report["gates"]["physical_Omega_identical"])
        for key in ("k", "w_vertices", "rpi_boundary", "u_lower_tight", "u_upper_tight", "invariant_lower", "invariant_upper"):
            np.testing.assert_array_equal(getattr(strong[2], key), getattr(env[2], key))

    def test_recovery_formula_and_no_clipping(self):
        self.assertAlmostEqual(recovery(100., 102., 100.6)["economic_recovery_ratio_pct"], 70.)
        self.assertEqual(recovery(100., 102., 99.)["economic_recovery_ratio_pct"], 150.)
        self.assertEqual(recovery(100., 102., 103.)["economic_recovery_ratio_pct"], -50.)

    def test_recovery_nonpositive_denominator(self):
        self.assertIsNone(recovery(100., 100., 99.)["economic_recovery_ratio_pct"])
        self.assertIsNone(recovery(100., 99., 98.)["economic_recovery_ratio_pct"])

    def test_preserve_completed_strong_and_v1(self):
        files = [v2.ROOT/"seed_42/run_summary.json", v1.DEFAULT_OUT/"oracle10/oracle_summary.json"]
        before = [v1.digest(p) for p in files]
        self.assertEqual(exp.run_dir(exp.ROOT, "strong", 42), v2.ROOT/"seed_42")
        self.assertTrue(exp.completed(exp.ROOT, "strong", 42))
        self.assertEqual(before, [v1.digest(p) for p in files])

    def test_immutable_selection(self):
        # Windows managed workspace cannot traverse chmod(0700) tempfile dirs.
        folder = v1.REPO/"evaporation_safe_sac"/f"test_economic_recovery_{uuid.uuid4().hex}"
        folder.mkdir()
        path = folder/"selection.json"
        exp.immutable(path, dict(selected_before_SAC_training=True))
        with self.assertRaises(RuntimeError): exp.immutable(path, dict(selected_before_SAC_training=False))

    def test_phase_order_blocks_conservative_before_three_strong(self):
        with mock.patch.object(exp, "completed", side_effect=lambda root, pair, seed: seed == 42):
            with self.assertRaisesRegex(RuntimeError, "2027.*314159"): exp.require_strong_complete(exp.ROOT)

    def test_final_seeds_forbidden(self):
        exp.require_dev(list(range(420000, 420010)))
        for seed in range(430000, 430050):
            with self.assertRaises(ValueError): exp.require_dev([seed])

    def test_positive_economic_denominator_required(self):
        for value in (0., -1., np.nan):
            with self.assertRaises(ValueError): economic_series([value], [1.])

    def test_best_found_not_upper_bound(self):
        self.assertEqual(headroom_utilization(.2, .1, dict(a=1), dict(a=1)), 200.)
        self.assertIsNone(headroom_utilization(.2, .1, dict(a=1), dict(a=2)))

    def test_no_ci_for_one_training_seed(self):
        row = aggregate([dict(gain=.08)], ["gain"])[0]
        self.assertIsNone(row["sample_std"]); self.assertIsNone(row["ci95_half_width"])

    def test_zero_reference_delta_identical_controller_design(self):
        strong = v2.guard(exp.args_for()); env, _ = reference_environment(strong, 0.)
        for key in vars(strong[2]):
            np.testing.assert_equal(getattr(strong[2], key), getattr(env[2], key))

    def test_frozen_figure_source_not_training_trajectory(self):
        from .zanon2019_economic_recovery_report import figure_source
        root = v1.REPO/"evaporation_safe_sac"/f"test_economic_recovery_{uuid.uuid4().hex}"
        root.mkdir()
        self.assertIsNone(figure_source(root, "strong"))
        folder = root/"strong/frozen_presentation"; folder.mkdir(parents=True)
        exp.save(folder/"summary.json", dict(trajectory_sha256="fake"))
        exp.save(folder/"selection.json", dict(trajectory_source="training trajectory"))
        with self.assertRaises(RuntimeError): figure_source(root, "strong")

    def test_learning_source_is_fixed_validation(self):
        from .zanon2019_paired_v2_report import fixed_curve
        with self.assertRaises(ValueError): fixed_curve([dict(episode=1, metric_source="training_rollout")], "gain")

    def test_grid_and_primary_pair_are_explicit(self):
        self.assertEqual(exp.DELTAS, (0., .05, .10, .15, .20))
        self.assertEqual(exp.PAIRS["strong"], "primary_strong_baseline")
        self.assertEqual(exp.PAIRS["conservative"], "sensitivity_conservative_baseline")
