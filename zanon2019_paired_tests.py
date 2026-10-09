"""No-training checks for the isolated paired-objective experiment."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from .omega_constrained_replay import DualState
from .zanon2019_paired_objective import (KEYS, Limits, step_components,
    component_series, violations, PairedReplay)
from .zanon2019_paired_experiment import (ConstantPolicy, SegmentPolicy,
    nondominated, paired_result, verified_inputs, PairedEvidenceController)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([25.39, 50.125])
        self.lower, self.upper = np.array([100., 100.]), np.array([400., 400.])
        self.base = [dict(state=self.reference + [0., 0.], control=np.array([200., 210.]), economic_cost=20., disturbance=np.ones(4)),
                     dict(state=self.reference + [.2, .3], control=np.array([210., 205.]), economic_cost=21., disturbance=np.ones(4)),
                     dict(state=self.reference + [-.1, .2], control=np.array([205., 215.]), economic_cost=22., disturbance=np.ones(4))]
        self.limits = Limits(3, np.array([.3, .5]), np.array([15., 15.]), 0., np.array([300., 300.]))

    def summed(self, records):
        return np.sum([step_components(k, r["state"], r["control"], r["economic_cost"],
                       records[k-1]["control"] if k else None, self.base, self.limits,
                       self.reference, self.lower, self.upper, 200.)
                       for k, r in enumerate(records)], axis=0)

    def test_zero_baseline_identity(self):
        values = self.summed(self.base)
        self.assertAlmostEqual(values[0], 0.)
        np.testing.assert_allclose(values[1:], [-.1, -.1, -.1*15/900, -.1*15/900, 0.], atol=1e-12)

    def test_component_sum_matches_all_step_metrics(self):
        actual = copy.deepcopy(self.base)
        actual[1]["state"] += [.1, -.1]
        actual[1]["control"] += [2., 3.]
        actual[2]["economic_cost"] -= 3.
        state = np.asarray([r["state"] for r in actual])
        inputs = np.asarray([r["control"] for r in actual])
        iae = np.abs(state-self.reference).sum(axis=0)
        tv = np.abs(np.diff(inputs, axis=0)).sum(axis=0)
        m = dict(X2_IAE=iae[0], P2_IAE=iae[1], P100_TV=tv[0], F200_TV=tv[1], any_outer_10pct_steps=0)
        values = self.summed(actual)
        np.testing.assert_allclose(values[1:], violations(m, self.limits), atol=1e-12)
        self.assertAlmostEqual(values[0], .015)

    def test_zero_tv_does_not_divide_by_zero(self):
        limits = Limits(3, np.ones(2), np.zeros(2), 0., np.full(2, 300.))
        metric = dict(X2_IAE=1., P2_IAE=1., P100_TV=0., F200_TV=1., any_outer_10pct_steps=0)
        g = violations(metric, limits)
        self.assertTrue(np.isfinite(g).all())
        self.assertEqual(g[2], 0.)
        self.assertGreater(g[3], 0.)

    def test_mismatch_and_incomplete_pair_rejected(self):
        cfg = SimpleNamespace(robust_economic_reference_state=self.reference)
        model = SimpleNamespace(physical_input=lambda x: x)
        design = SimpleNamespace(robust_input_lower=self.lower, robust_input_upper=self.upper)
        actual = copy.deepcopy(self.base); actual[1]["disturbance"][0] += 1.
        with self.assertRaisesRegex(ValueError, "Unpaired disturbance"):
            component_series(actual, self.base, self.limits, cfg, model, design, 200.)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            component_series(self.base[:2], self.base, self.limits, cfg, model, design, 200.)

    def test_dynamic_replay_uses_current_multiplier(self):
        dual = DualState.create(np.ones(5), 100.)
        replay = PairedReplay(2, 2, 4, torch.device("cpu"), dual,
                              scales=np.ones(5), economic_scale=200.)
        replay.add_components([0, 0], [0, 0], [2., 1., 0., 0., 0., 0.], [1, 1], False)
        self.assertEqual(float(replay.sample(1)[2][0, 0]), 2.)
        dual.values[0] = 3.
        self.assertEqual(float(replay.sample(1)[2][0, 0]), -1.)
        self.assertEqual(float(replay.components[0, 0]), 2.)

    def test_replay_adapter_uses_final_input_and_next_state_cost(self):
        row = dict(step=0, next_state_observed=True,
                   **{f"disturbance_{i}": 1. for i in range(4)},
                   state_0=self.reference[0], state_1=self.reference[1],
                   next_state_0=25.5, next_state_1=50.2, control_0=200., control_1=210.)
        ctrl = SimpleNamespace(evidence=[row], cfg=SimpleNamespace(robust_economic_reference_state=self.reference),
            model=SimpleNamespace(physical_input=lambda v: v, economic_cost=lambda x, u, d: float(x[0])),
            d=SimpleNamespace(robust_input_lower=self.lower, robust_input_upper=self.upper))
        dual = DualState.create(np.ones(5), 100.)
        replay = PairedReplay(2, 2, 4, torch.device("cpu"), dual,
                              scales=np.ones(5), economic_scale=200.)
        replay.bind(ctrl, self.base, self.limits)
        replay.add([0, 0], [.2, -.2], -99999., [1, 1], False)
        self.assertAlmostEqual(replay.episode_components[0][0], (20.-25.5)/200.)
        self.assertAlmostEqual(float(replay.components[0, 0]), (20.-25.5)/200., places=7)
        self.assertEqual(float(replay.rewards[0, 0]), 0.)

    def test_safety_excludes_W_only_as_documented(self):
        metric = dict(J_econ=60., X2_IAE=.3, P2_IAE=.5, X2_ISE=.1, P2_ISE=.2,
            P100_TV=15., F200_TV=15., any_outer_10pct_steps=0.,
            physical_state_violation_steps=0, physical_input_violation_count=0,
            QP_infeasible_count=0, Omega_exit_count=0, robust_region_violation_count=0,
            W_exceedance_count=3)
        base = dict(metric, J_econ=61.)
        self.assertTrue(paired_result(metric, base, self.limits)["joint_candidate"])
        metric["QP_infeasible_count"] = 1
        self.assertFalse(paired_result(metric, base, self.limits)["joint_candidate"])

    def test_real_three_step_replay_adapter_without_optimizer_updates(self):
        # Only warmup transitions. No SACAgent/network or optimizer is created.
        from .zanon2019_benchmark import (make_setup, sample_disturbance_path,
                                         PAPER_VARIANCES_F1_X1_T1_T200)
        from .zanon2019_training_report import EvidenceController, metrics
        from .zanon2019_paired_objective import limits_from_metrics
        from .train import ZeroResidualPolicy, run_episode, OBS_DIM
        cfg, model, design, omega, domain, _ = make_setup(3, 42)
        cfg.stochastic_residual_scale = .1
        cfg.abort_on_first_uncertified_step = True
        path, _ = sample_disturbance_path(cfg, "zanon2019_stochastic", 420000, 3,
                                          PAPER_VARIANCES_F1_X1_T1_T200, 20, 50)
        ctrl = PairedEvidenceController(cfg, model, design, omega, domain, path)
        _, base, _ = run_episode(cfg, model, ctrl, ZeroResidualPolicy(), None,
            np.random.default_rng(1), training=False, global_step=0,
            disturbance_trajectory=path, initial_state_override=cfg.robust_economic_reference_state)
        ranges = cfg.input_scale * (design.robust_input_upper-design.robust_input_lower)
        limits = limits_from_metrics(metrics(base, cfg, model, design, omega), 3, ranges)
        dual = DualState.create(np.ones(5), 100.)
        replay = PairedReplay(OBS_DIM+4, 2, 4, torch.device("cpu"), dual,
                              scales=np.ones(5), economic_scale=200.)
        train_ctrl = PairedEvidenceController(cfg, model, design, omega, domain, path)
        replay.bind(train_ctrl, base, limits)
        class NoOptimizerPolicy(ZeroResidualPolicy):
            def update(self, *args): raise AssertionError("Optimizer update forbidden in this test")
        _, records, global_step = run_episode(cfg, model, train_ctrl, NoOptimizerPolicy(), replay,
            np.random.default_rng(2), training=True, global_step=0,
            disturbance_trajectory=path, initial_state_override=cfg.robust_economic_reference_state)
        expected = component_series(records, base, limits, cfg, model, design, 200.)
        np.testing.assert_allclose(replay.episode_components, expected, atol=1e-9, rtol=0)
        np.testing.assert_allclose(expected[:, 1:].sum(axis=0),
                                  violations(metrics(records, cfg, model, design, omega), limits), atol=1e-9)
        self.assertEqual(global_step, 3)
        self.assertEqual(replay.size, 3)


class SearchTests(unittest.TestCase):
    def test_immediate_robust_region_abort(self):
        from .zanon2019_training_report import EvidenceController
        ctrl = object.__new__(PairedEvidenceController)
        ctrl.model = SimpleNamespace(normalized_state=lambda v: np.asarray(v),
                                     normalized_input=lambda v: np.asarray(v))
        ctrl.d = SimpleNamespace(robust_state_lower=np.full(2, -1.), robust_state_upper=np.ones(2),
                                 robust_input_lower=np.full(2, -1.), robust_input_upper=np.ones(2))
        ctrl.last_control = np.zeros(2)
        ctrl.evidence = [dict(step=0)]
        with patch.object(EvidenceController, "audit_next_state", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "robust-region"):
                ctrl.audit_next_state(np.array([1.1, 0.]))
        self.assertEqual(ctrl.evidence[-1]["robust_region_violation"], 1)

    def test_segment_semantics(self):
        p = SegmentPolicy([[.1, .2], [.3, .4]], 2)
        np.testing.assert_allclose([p.select_action(None) for _ in range(4)],
                                  [[.1, .2], [.1, .2], [.3, .4], [.3, .4]])
        np.testing.assert_array_equal(ConstantPolicy().select_action(None), [0, 0])

    def test_pareto_does_not_optimize_only_economy(self):
        def row(i, gain, ratio, safe=True):
            return dict(candidate=i, economic_improvement_pct=gain,
                safety_passed=safe, boundary_fraction=0.,
                **{k+"_ratio": ratio for k in ("X2_IAE", "P2_IAE", "X2_ISE", "P2_ISE", "P100_TV", "F200_TV")})
        a, b, c, d = row(0, .01, 1.), row(1, .02, 1.2), row(2, .001, 1.1), row(3, 1., .9, False)
        self.assertEqual([r["candidate"] for r in nondominated([a, b, c, d])], [0, 1])

    def test_training_refuses_no_oracle_witness(self):
        # No agent/replay allocation or training takes place in this test.
        args = SimpleNamespace(constraint_audit=Path("unused_audit.json"),
                               oracle_summary=Path("unused_oracle.json"))
        with patch.object(Path, "read_text", side_effect=[
                '{"protocol":{}}', '{"protocol":{},"joint_witness_count":0}']), \
             patch("evaporation.zanon2019_paired_experiment.protocol", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "No joint"):
                verified_inputs(args, None)


if __name__ == "__main__":
    unittest.main()
