"""Fail-closed Conservative sweep, common resets, provenance and economics."""
from pathlib import Path
from unittest import TestCase, mock
import uuid
import numpy as np

from . import zanon2019_conservative_mechanism as stage
from . import zanon2019_paired_v2 as v2
from . import zanon2019_economic_recovery as previous
from .zanon2019_economic_recovery_reference import reference_environment
from .zanon2019_economic_recovery_metrics import recovery, economic_series
from .zanon2019_benchmark import sample_disturbance_path, PAPER_VARIANCES_F1_X1_T1_T200
from .train import ZeroResidualPolicy


def rows(margin=.21):
    return [dict(validation_seed=s, minimum_X2_margin=margin,
                 **{k: 0 for k in stage.SAFETY}) for s in stage.DEV]


class ConservativeMechanismTests(TestCase):
    def test_new_stage_does_not_modify_old_protocol_or_v2(self):
        before = {p: stage.v1.digest(p) for p in (previous.ROOT/"protocol.json", v2.ROOT/"seed_42/run_summary.json")}
        previous.locked(previous.ROOT)
        self.assertEqual(before, {p: stage.v1.digest(p) for p in before})
        self.assertFalse(any("conservative_mechanism" in p for p in previous.code_hashes()))

    def test_fixed_grid_P2_and_nonoptimized_steady_input(self):
        env = v2.guard(previous.args_for())
        for delta in stage.DELTAS:
            candidate, report = reference_environment(env, delta)
            self.assertAlmostEqual(candidate[0].robust_economic_reference_state[0], 25.39+delta)
            self.assertAlmostEqual(candidate[0].robust_economic_reference_state[1], 50.125)
            self.assertTrue(report["gates"]["nonlinear_steady"])
            np.testing.assert_array_equal(env[2].k, candidate[2].k)
            np.testing.assert_array_equal(env[2].w_vertices, candidate[2].w_vertices)
            np.testing.assert_array_equal(env[2].rpi_boundary, candidate[2].rpi_boundary)
            np.testing.assert_array_equal(env[2].u_upper_tight, candidate[2].u_upper_tight)

    def test_all_positive_fixed_P2_candidates_fail_current_tightening(self):
        env = v2.guard(previous.args_for())
        for delta in stage.DELTAS[1:]:
            _, report = reference_environment(env, delta)
            self.assertFalse(report["passed"])
            self.assertIn("reference_input_tightened", report["limiting_gates"])
            self.assertGreater(report["steady_input"][1], env[1].physical_input(env[2].u_upper_tight)[1])

    def test_margin_gate_uses_every_path_not_only_aggregate(self):
        strong, candidate = rows(), rows(.23)
        self.assertTrue(stage.margin_gate(candidate, strong, 1e-4))
        candidate[5]["minimum_X2_margin"] = .20999
        self.assertFalse(stage.margin_gate(candidate, strong, 1e-4))

    def test_exact_margin_tolerance_does_not_pass(self):
        strong, candidate = rows(0.), rows(1e-4)
        self.assertFalse(stage.margin_gate(candidate, strong, 1e-4))

    def test_safety_anomaly_blocks_margin_selection(self):
        for key in stage.SAFETY:
            candidate = rows(.23); candidate[0][key] = 1
            self.assertFalse(stage.margin_gate(candidate, rows(), 1e-4))

    def test_incomplete_or_wrong_validation_seeds_rejected(self):
        self.assertFalse(stage.margin_gate(rows(.23)[:9], rows(), 1e-4))
        candidate = rows(.23); candidate[-1]["validation_seed"] = 430000
        self.assertFalse(stage.margin_gate(candidate, rows(), 1e-4))

    def test_oracle_stops_if_no_selection(self):
        allowed, reason = stage.oracle_precondition(dict(selected_delta_X2=None), dict(meaningful=False))
        self.assertFalse(allowed); self.assertEqual(reason, "no_eligible_conservative_reference")

    def test_oracle_stops_if_penalty_nonmeaningful(self):
        selected = dict(selected_delta_X2=.05, selected_before_SAC_training=True, selection_used_SAC_results=False)
        self.assertFalse(stage.oracle_precondition(selected, dict(meaningful=False, J_strong=100., J_conservative=100.))[0])

    def test_selection_cannot_use_SAC_results(self):
        selected = dict(selected_delta_X2=.05, selected_before_SAC_training=True, selection_used_SAC_results=True)
        self.assertFalse(stage.oracle_precondition(selected, dict(meaningful=True, J_strong=100., J_conservative=101.))[0])

    def test_recovery_nonpositive_not_applicable(self):
        self.assertIsNone(recovery(100., 99., 98.)["economic_recovery_ratio_pct"])

    def test_locked_economic_sign_and_cumulative(self):
        e = economic_series([100., 110.], [99., 109.])
        self.assertTrue(np.all(e["instantaneous_economic_advantage_pct"] > 0))
        self.assertEqual(e["G_cum"][-1], e["economic_improvement_pct"])

    def test_common_initial_condition_and_paired_disturbance(self):
        args = previous.args_for(); args.steps = 3
        env = v2.guard(args)
        initial = np.array([25.39, 50.125])
        path, _ = sample_disturbance_path(env[0], "zanon2019_stochastic", 420000, 3,
            PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
        first, _ = stage.rollout(env, path, 420000, ZeroResidualPolicy(), initial)
        second, _ = stage.rollout(env, path, 420000, ZeroResidualPolicy(), initial)
        np.testing.assert_array_equal(first[0]["state"], initial)
        for a, b, disturbance in zip(first, second, path):
            np.testing.assert_array_equal(a["disturbance"], disturbance)
            np.testing.assert_array_equal(a["state"], b["state"])
            np.testing.assert_array_equal(a["control"], b["control"])
            self.assertEqual(a["economic_cost"], b["economic_cost"])

    def test_summary_minimum_is_worst_not_average(self):
        data = rows()
        for i, r in enumerate(data):
            r.update(J_econ=100+i, mean_X2=25.4, std_X2=.04, centered_X2_MAE=.03,
                     X2_IAE=1., X2_ISE=1., P2_IAE=1., P2_ISE=1., P100_TV=1.,
                     P100_delta_RMS=1., F200_TV=1., F200_delta_RMS=1., W_exceedance_rate=.98,
                     minimum_X2=25.2+i*.01, minimum_X2_margin=.2+i*.01)
        summary = stage.summarize(data)
        self.assertEqual(summary["minimum_X2_margin"], .2)
        self.assertEqual(summary["mean_J_econ"], 104.5)

    def test_no_training_entry_or_hidden_training_call(self):
        import ast
        import inspect
        source = inspect.getsource(stage.rollout)
        self.assertIn("training=False", source)
        calls = [node for node in ast.walk(ast.parse(inspect.getsource(stage)))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_episode"]
        self.assertEqual(len(calls), 1)
        keyword = next(k for k in calls[0].keywords if k.arg == "training")
        self.assertIs(keyword.value.value, False)
        self.assertNotIn("430000", inspect.getsource(stage))

    def test_blocked_oracle_has_null_benefit_not_fake_zero(self):
        folder = stage.v1.REPO/"evaporation_safe_sac"/f"test_conservative_mechanism_{uuid.uuid4().hex}"
        folder.mkdir()
        stage.save(folder/"conservative_baseline_selection.json", dict(selected_delta_X2=None))
        stage.save(folder/"conservatism_economic_penalty.json", dict(meaningful=False))
        with mock.patch.object(stage, "locked", return_value={}), mock.patch.object(stage, "rollout", side_effect=AssertionError("must not simulate")):
            stage.oracle(folder)
        result = stage.load(folder/"oracle_conservative_summary.json")
        self.assertEqual(result["total_candidates"], 0)
        self.assertIsNone(result["best_economic_improvement_found_pct"])
        self.assertFalse(result["conservative_seed42_training_allowed"])
