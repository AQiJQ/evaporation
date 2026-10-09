"""No SAC training. Component/unit and three-transition warmup integration tests."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from . import zanon2019_paired_v2 as v2
from .zanon2019_paired_v2_objective import (paired_move, components, decomposition, operating_metrics,
    joint, SafetyDual, EconomicReplay, paired_metrics)
from .zanon2019_paired_v2_report import fixed_curve, figure_data


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.base = [dict(state=np.array([25.39, 50.125]), control=np.array([190., 210.]),
                          disturbance=np.ones(4), economic_cost=100., raw_action=np.ones(2))]
        self.policy = copy.deepcopy(self.base)
        self.initial = np.array([190., 210.])

    def test_paired_delta_two_axes(self):
        np.testing.assert_allclose(paired_move([194, 212], [190, 210], [192, 211], [190, 210], [100, 100]), [.0012, .0003])

    def test_smaller_move_zero(self):
        np.testing.assert_array_equal(paired_move([1, -1], [0, 0], [2, 2], [0, 0], [100, 100]), [0, 0])

    def test_equal_move_zero(self):
        np.testing.assert_array_equal(paired_move([-2, 2], [0, 0], [2, -2], [0, 0], [100, 100]), [0, 0])

    def test_larger_move_positive(self):
        self.assertTrue(np.all(paired_move([3, 4], [0, 0], [2, 2], [0, 0], [100, 100]) > 0))

    def test_fixed_normalization(self):
        np.testing.assert_allclose(paired_move([1, 1], [0, 0], [0, 0], [0, 0], [100, 200]), [1e-4, 2.5e-5])

    def test_invalid_scale(self):
        with self.assertRaises(ValueError): paired_move([0, 0], [0, 0], [0, 0], [0, 0], [0, 100])

    def test_first_step_initial_applied_semantics(self):
        raw = components(self.policy, self.base, [100, 100], self.initial)
        np.testing.assert_array_equal(raw, np.zeros((1, 7)))

    def test_raw_actor_not_used(self):
        self.policy[0]["raw_action"] *= 1000
        self.assertEqual(components(self.policy, self.base, [100, 100], self.initial)[0, 1], 0)

    def test_applied_input_used(self):
        self.policy[0]["control"][1] += 10
        self.assertAlmostEqual(components(self.policy, self.base, [100, 100], self.initial)[0, 1], .01)

    def test_cost_sign_and_200(self):
        self.policy[0]["economic_cost"] = 90
        self.assertEqual(components(self.policy, self.base, [100, 100], self.initial)[0, 0], .05)

    def test_disturbance_pair_required(self):
        self.policy[0]["disturbance"][2] += 1
        with self.assertRaises(ValueError): components(self.policy, self.base, [100, 100], self.initial)

    def test_initial_state_pair_required(self):
        self.policy[0]["state"][0] += .1
        with self.assertRaises(ValueError): components(self.policy, self.base, [100, 100], self.initial)

    def test_decomposition_consistency(self):
        r = decomposition(np.array([[.2, .01, 1, 0, 0, 0, 0]]), 2., [3, 0, 0, 0, 0])
        self.assertAlmostEqual(r["total_training_reward"], -2.82)
        self.assertAlmostEqual(r["regularizer_fraction"], .1)

    def test_zero_economics_fraction_undefined(self):
        self.assertIsNone(decomposition(np.zeros((1, 7)), 1, np.zeros(5))["regularizer_fraction"])

    def test_replay_independent_components_and_current_dual(self):
        dual = SafetyDual.create()
        replay = EconomicReplay(2, 2, 2, torch.device("cpu"), dual, 2.)
        raw = [.2, .01, 1, 0, 0, 0, 0]
        replay.add_components([0, 0], [1, 1], raw, [0, 0], False)
        np.testing.assert_allclose(replay.components[0], raw)
        self.assertEqual(replay.rewards[0, 0], 0)
        self.assertAlmostEqual(replay.sample(1)[2].item(), .18, places=6)
        dual.values[0] = 3
        self.assertAlmostEqual(replay.sample(1)[2].item(), -2.82, places=6)

    def test_fixed_weight_not_settable(self):
        replay = EconomicReplay(2, 2, 2, torch.device("cpu"), SafetyDual.create(), 2.)
        with self.assertRaises(AttributeError): replay.weight = 999.

    def test_smoothness_not_dual_cost(self):
        dual = SafetyDual.create()
        dual.update(np.zeros(5))
        np.testing.assert_array_equal(dual.values, np.zeros(5))
        self.assertEqual(len(dual.values), 5)

    def test_unexcited_calibration_stops(self):
        with self.assertRaises(RuntimeError): v2.calibrate(np.ones(1000), np.zeros(1000))

    def test_calibration_five_percent(self):
        stats = v2.calibrate(np.full(1000, .2), np.full(1000, .01))
        self.assertAlmostEqual(stats["lambda_smooth"], 1.)


class ProtocolTests(unittest.TestCase):
    def test_v1_preserved_and_145_v2_reclassification(self):
        oracle = v2.v1.DEFAULT_OUT/"oracle10"
        original = v2.v1.digest(oracle/"oracle_summary.json")
        self.assertEqual(v2.load(oracle/"oracle_summary.json")["joint_witness_count"], 0)
        passed = []
        for row in v2.v1.read_csv(oracle/"all_candidates.csv"):
            candidate = int(row["candidate"])
            rows = v2.v1.read_csv(oracle/"candidate_metrics"/f"candidate_{candidate:04d}.csv")
            numerical = [dict(safety_passed=r["safety_passed"] == "True",
                **{k: float(r[k]) for k in ("economic_improvement_pct", "P100_TV_ratio", "F200_TV_ratio")}) for r in rows]
            if joint(numerical): passed.append(candidate)
        self.assertEqual(passed, [71, 76, 133])
        self.assertEqual(original, v2.v1.digest(oracle/"oracle_summary.json"))

    def test_tracking_not_hidden_gate(self):
        r = dict(safety_passed=True, economic_improvement_pct=.01, P100_TV_ratio=.9,
                 F200_TV_ratio=.99, X2_IAE_ratio=2., P2_IAE_ratio=2.)
        self.assertTrue(joint([r]))

    def test_each_realization_must_pass(self):
        r = dict(safety_passed=True, economic_improvement_pct=.01, P100_TV_ratio=.9, F200_TV_ratio=.99)
        self.assertFalse(joint([r, dict(r, economic_improvement_pct=-.001)]))
        self.assertFalse(joint([dict(r, F200_TV_ratio=1.01)]))
        self.assertFalse(joint([dict(r, safety_passed=False)]))

    def test_operating_shift_and_centered_fluctuation(self):
        r = [dict(state=[x, 50.]) for x in (25.3, 25.4, 25.5)]
        s = [dict(state=[x-.01, 50.]) for x in (25.3, 25.4, 25.5)]
        a, b = operating_metrics(r), operating_metrics(s)
        self.assertAlmostEqual(b["mean_X2"]-a["mean_X2"], -.01)
        self.assertAlmostEqual(a["centered_X2_MAE"], b["centered_X2_MAE"])
        self.assertAlmostEqual(a["std_X2"], b["std_X2"])

    def test_terminal_minimum_included(self):
        r = [dict(state=[25.3, 50.])]
        self.assertAlmostEqual(operating_metrics(r, 25.2)["minimum_X2_margin"], .2)

    def test_main_learning_curve_refuses_training_return(self):
        with self.assertRaises(ValueError): fixed_curve([dict(episode=1, episode_return=0)], "episode_return")
        x, y = fixed_curve([dict(episode=5, metric_source="fixed_paired_validation", gain=.1)], "gain")
        self.assertEqual((x, y), ([5], [.1]))

    def test_plot_cost_sign_and_time_alignment(self):
        r = dict(step="0", delta_l_SAC_minus_baseline="-1")
        for name in ("baseline", "SAC"):
            r.update({name+"_"+key: "1" for key in ("state_0", "state_1", "control_0", "control_1")})
            r[name+"_economic_cost"] = "2" if name == "baseline" else "1"
        self.assertEqual(figure_data([r])["delta"][0], -1)
        with self.assertRaises(ValueError): figure_data([dict(r, step="1")])

    def test_no_future_final_seeds_in_calibration(self):
        sources = v2.calibration_sources()
        self.assertEqual(len(sources), 69)
        self.assertTrue(all(dev in v2.DEV for _, _, dev, _ in sources))

    def test_prior_seed_continuation_gate(self):
        s = dict(run_state="completed", completed_episodes=100, global_step=100000,
                 max_action_collapse_fraction=0., replay_decomposition_checked=True, no_nonfinite=True)
        self.assertFalse(v2.pilot_gate(s, []))
        self.assertTrue(v2.pilot_gate(s, [dict(positive_economic_safe_learned=True)]))
        self.assertFalse(v2.pilot_gate(dict(s, run_state="aborted"), [dict(positive_economic_safe_learned=True)]))

    def test_frozen_parent_hashes_pass(self):
        from .zanon2019_final_experiment import verify_policy_lock
        self.assertEqual(len(verify_policy_lock(v2.v1.LOCK_ROOT)["policies"]), 3)

    def test_real_paired_replay_and_no_gradient_evaluation(self):
        from .zanon2019_benchmark import make_setup
        env0 = make_setup(3, 42)
        cfg, model, design, omega, domain = env0[:5]
        cfg.stochastic_residual_scale, cfg.abort_on_first_uncertified_step = .1, True
        env = (cfg, model, design, omega, domain)
        path, _ = v2.sample_disturbance_path(cfg, "zanon2019_stochastic", 420000, 3,
                                            v2.PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
        class NoOptimizer(v2.ZeroResidualPolicy):
            def update(self, *args): raise AssertionError("No SAC training in tests")
        with torch.no_grad():
            base, _, _ = v2.v1.rollout(env, path, 420000, NoOptimizer())
        replay = EconomicReplay(23, 2, 4, torch.device("cpu"), SafetyDual.create(), 1.)
        ctrl = v2.v1.PairedEvidenceController(cfg, model, design, omega, domain, path)
        replay.bind(ctrl, base)
        _, records, step = v2.run_episode(cfg, model, ctrl, NoOptimizer(), replay,
            np.random.default_rng(77), training=True, global_step=0, disturbance_trajectory=path,
            initial_state_override=cfg.robust_economic_reference_state)
        metric, raw = paired_metrics(records, base, env, 1., np.zeros(5))
        np.testing.assert_allclose(replay.episode_components, raw, rtol=0, atol=1e-9)
        self.assertEqual(step, 3)
        self.assertEqual(replay.size, 3)
        self.assertEqual(metric["physical_state_violation_steps"], 0)

    def test_immediate_safety_abort_retained(self):
        from .zanon2019_training_report import EvidenceController
        ctrl = object.__new__(v2.v1.PairedEvidenceController)
        ctrl.model = SimpleNamespace(normalized_state=lambda v: v, normalized_input=lambda v: v)
        ctrl.d = SimpleNamespace(robust_state_lower=np.full(2, -1.), robust_state_upper=np.ones(2),
                                 robust_input_lower=np.full(2, -1.), robust_input_upper=np.ones(2))
        ctrl.last_control, ctrl.evidence = np.zeros(2), [dict(step=0)]
        with patch.object(EvidenceController, "audit_next_state", return_value=None):
            with self.assertRaises(RuntimeError): ctrl.audit_next_state(np.array([1.1, 0.]))

    def test_actual_agent_no_gradient_evaluation(self):
        from .zanon2019_benchmark import make_setup, make_agent
        cfg = make_setup(3, 42)[0]
        agent = make_agent(cfg, "cpu")
        agent.zero_initialize_residual_mean()
        with torch.no_grad():
            a = agent.select_action(np.zeros(23), deterministic=True)
        np.testing.assert_array_equal(a, [0, 0])
        self.assertTrue(all(p.grad is None for net in (agent.actor, agent.q1, agent.q2) for p in net.parameters()))

    def test_complete_report_interfaces_without_training_or_fake_artifacts(self):
        # Existing development trace is a unit-test fixture, NEVER a v2 result.
        # Every output write/figure export is mocked. No fake run is published.
        from .zanon2019_benchmark import make_setup
        from .zanon2019_paired_v2_report import report
        from matplotlib.figure import Figure
        env = make_setup(1000, 42)
        role, seed, dev, source = v2.calibration_sources()[0]
        actual_rows = v2.v1.read_csv(source)
        base, policy = v2.archived_pair(source)
        metric, _ = paired_metrics(policy, base, env, 1., np.zeros(5))
        metrics_rows = [dict(metric, validation_seed=s) for s in v2.DEV]
        fixture = v2.assessment(metrics_rows, 100, 100000, 5000)
        root = Path("unit_fixture_NOT_a_training_run")
        original_load = v2.load
        def fake_load(path):
            if path.name == "smoothness_regularization_config.json": return dict(lambda_smooth=1.)
            if path.name == "checkpoints.json": return [fixture]
            if path.name == "run_summary.json": return dict(run_state="unit_fixture")
            if path.name == "experiment_manifest.json": return dict(smoothness_config_sha256="unit_fixture_not_real_evidence")
            return original_load(path)
        def exists(path):
            if "unit_fixture_NOT_a_training_run" not in str(path): return True
            return "seed_42" in str(path) and path.name != "training_log.csv"
        with patch.object(Path, "exists", exists), patch.object(Path, "mkdir"), \
                patch.object(v2, "load", side_effect=fake_load), \
                patch.object(v2, "locked_config", return_value=dict(lambda_smooth=1.)), \
                patch.object(v2.v1, "read_csv", return_value=actual_rows), \
                patch.object(v2.v1, "digest", return_value="unit_fixture_not_real_evidence"), \
                patch.object(v2.v1, "save"), patch("evaporation.zanon2019_paired_v2_report.write_csv"), \
                patch.object(Figure, "savefig") as export, patch.object(Path, "glob", return_value=[]):
            report(root, env)
        names = {Path(c.args[0]).name for c in export.call_args_list}
        self.assertTrue({"paper_fig2_style_combined.png", "paper_learning_curves_combined.png",
                         "paper_reward_decomposition.png", "economic_vs_F200_TV.png"} <= names)


if __name__ == "__main__":
    unittest.main()
