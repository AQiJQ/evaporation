"""Authority adapter tests: no SAC updates, no authority pilot training."""
from copy import copy
import csv
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from .sac import set_seed
from .train import run_episode, ZeroResidualPolicy
from .zanon2019_benchmark import make_setup, make_agent, StochasticInterior23
from .zanon2019_training_report import (FixedPairedEvaluator, EvidenceController,
    assessment, evaluation_safety_anomalies)
from .zanon2019_authority import (AuthorityController, AuthoritySelection,
    authority_assessment, PILOT_ALPHAS)
from .zanon2019_authority_pilot import commands, compare


class AuthorityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42)
        cls.cfg, cls.model, cls.design, cls.omega, cls.domain, _ = make_setup(20, 42)
        cls.agent = make_agent(cls.cfg, "cpu")
        cls.agent.zero_initialize_residual_mean()
        cls.out = Path(__file__).parent / "evaporation_safe_sac" / ("test_authority_no_training_" + uuid4().hex[:8])
        cls.evaluator = FixedPairedEvaluator(cls.cfg, cls.model, cls.design, cls.omega,
            cls.domain, "zanon2019_stochastic", [2, 1, 8, 5], 20, 50,
            [420000, 420001, 420002], cls.out)
        cls.path = cls.evaluator.cache[420000][0]

    def cfg_at(self, alpha):
        cfg = copy(self.cfg)
        cfg.stochastic_residual_scale = alpha
        return cfg

    def controller(self, kind, cfg):
        return kind(cfg, self.model, self.design, self.omega, self.domain)

    def rollout(self, cfg, ctrl, action):
        class Frozen:
            def select_action(self, obs, deterministic=True):
                return np.array(action, dtype=np.float32)
        with patch.object(self.agent, "update", side_effect=AssertionError("No training")):
            return run_episode(cfg, self.model, ctrl, Frozen(), None, np.random.default_rng(1),
                training=False, global_step=0, disturbance_trajectory=self.path,
                initial_state_override=cfg.robust_economic_reference_state)[1]

    def test_scaling_exact_original_and_zero(self):
        raw = np.array([.15, -.05], dtype=np.float32)
        for alpha in (0., .1, .2, .3, .5, 1.):
            cfg = self.cfg_at(alpha)
            adapted = self.rollout(cfg, self.controller(AuthorityController, cfg), raw)
            original = self.rollout(cfg, self.controller(StochasticInterior23, cfg), alpha * raw)
            for key in ("state", "control", "residual", "applied_residual"):
                np.testing.assert_array_equal([r[key] for r in adapted], [r[key] for r in original])
            np.testing.assert_array_equal([r["raw_action"] for r in adapted], np.tile(raw, (20, 1)))
            if alpha == 0:
                np.testing.assert_array_equal([r["control"] for r in adapted],
                    [r["control"] for r in self.evaluator.cache[420000][1]])
        self.assertEqual(self.agent.total_updates, 0)

    def test_replay_keeps_full_raw_warmup_action(self):
        cfg = self.cfg_at(.1)
        class AuditReplay:
            def __init__(self): self.actions, self.size = [], 0
            def add(self, obs, action, *args, **kwargs):
                self.actions.append(action.copy())
                self.size += 1
        replay = AuditReplay()
        ctrl = EvidenceController(cfg, self.model, self.design, self.omega, self.domain, self.path)
        with patch.object(self.agent, "update", side_effect=AssertionError("No SAC updates")):
            _, records, _ = run_episode(cfg, self.model, ctrl, self.agent, replay,
                np.random.default_rng(77), training=True, global_step=0,
                disturbance_trajectory=self.path,
                initial_state_override=cfg.robust_economic_reference_state)
        np.testing.assert_array_equal(replay.actions, [r["raw_action"] for r in records])
        self.assertGreater(np.max(np.abs(replay.actions)), .5)
        np.testing.assert_allclose([[r[f"scaled_action_{i}"] for i in range(2)] for r in ctrl.evidence],
                                   .1 * np.asarray(replay.actions), rtol=0, atol=1e-8)
        self.assertEqual(self.agent.total_updates, 0)

    def test_paired_metrics_and_hierarchy(self):
        cfg = self.cfg_at(.2)
        cfg.authority_pilot = True
        evaluator = FixedPairedEvaluator(cfg, self.model, self.design, self.omega,
            self.domain, "zanon2019_stochastic", [2, 1, 8, 5], 20, 50,
            [420000], self.out / "paired_alpha02")
        rows, status = evaluator.evaluate(self.agent, 0, cfg.warmup_steps)
        self.assertTrue(status["empirical_safe"])
        self.assertTrue(status["economic_comparable_0p5pct"])
        self.assertEqual(status["scaled_action_norm_mean"], 0.)
        self.assertEqual(status["economic_improvement_pct"], 0.)
        self.assertFalse(evaluation_safety_anomalies(rows))
        class Saver:
            def __init__(self): self.saved = []
            def save_actor(self, path): self.saved.append(path.name)
        selector, saver = AuthoritySelection(self.out), Saver()
        selector.consider(saver, {**status, "post_warmup": False})
        self.assertEqual(saver.saved, [])
        selector.consider(saver, status)
        self.assertIn("best_safety_economic_actor.pth", saver.saved)
        episode_before = selector.best["best_safety_economic_actor"]["episode"]
        # More economics cannot outrank less physical margin.
        selector.consider(saver, {**status, "episode": 5,
            "minimum_X2_margin": status["minimum_X2_margin"] - .1,
            "economic_improvement_pct": 5.})
        self.assertEqual(selector.best["best_safety_economic_actor"]["episode"], episode_before)
        invalid = [{**rows[0], "SAC_Omega_exit_count": 1}]
        bad = authority_assessment(invalid, assessment(invalid, 5, cfg.warmup_steps, cfg.warmup_steps))
        self.assertFalse(bad["empirical_safe"])
        n = len(saver.saved)
        selector.consider(saver, bad)
        self.assertEqual(len(saver.saved), n)

    def test_commands_and_missing_results(self):
        for alpha, cmd in zip(PILOT_ALPHAS, commands(self.out)):
            self.assertEqual(cmd[cmd.index("--stochastic-residual-scale") + 1], str(alpha))
            self.assertEqual(cmd[cmd.index("--episodes") + 1], "100")
            self.assertEqual(cmd[cmd.index("--eval-seeds") + 1], "10")
            self.assertNotIn("--resume", cmd)
        decision = compare(self.out / "absent_runs", self.out / "missing_results")
        self.assertIsNone(decision["sweet_spot_alpha"])
        self.assertFalse(decision["final_test_performed"])
        with (self.out / "missing_results" / "stochastic_authority_comparison.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["economic_improvement_pct"] == "" for r in rows))
        self.assertEqual(json.loads((self.out / "missing_results" / "authority_decision.json").read_text())["all_pilots_completed"], False)

    def test_scale_validation(self):
        for alpha in (-.1, 1.1, float("nan")):
            cfg = self.cfg_at(alpha)
            with self.assertRaises(ValueError):
                self.controller(AuthorityController, cfg).act(cfg.robust_economic_reference_state, np.zeros(2))
        cfg = self.cfg_at(.1)
        cfg.disturbance_mode = "piecewise_stochastic"
        with self.assertRaises(ValueError):
            self.controller(AuthorityController, cfg).act(cfg.robust_economic_reference_state, np.zeros(2))

    def test_pilot_orchestration_evaluation_only(self):
        from . import zanon2019_train as entry
        from .sac import SACAgent
        out = self.out / "pilot_orchestration_NO_TRAINING"
        def no_training(*args, **kwargs):
            args = list(args)
            args[4] = None
            kwargs["training"] = False
            stat, records, _ = run_episode(*args, **kwargs)
            return stat, records, 5000  # TEST ONLY synthetic learned-gate branch
        argv = ["pilot_TEST_ONLY", "--authority-pilot", "--stochastic-residual-scale", ".3",
            "--episodes", "1", "--steps", "20", "--device", "cpu", "--output-dir", str(out)]
        with patch.object(sys, "argv", argv), patch.object(entry, "run_episode", no_training), \
                patch.object(SACAgent, "update", side_effect=AssertionError("No SAC training")), \
                patch("builtins.print"):
            entry.main()
        summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["evaluation_seeds"], list(range(420000, 420010)))
        self.assertEqual(summary["stochastic_residual_scale"], .3)
        self.assertEqual(summary["run_state"], "completed")
        self.assertFalse(summary["final_test_performed"])
        for name in ("best_safety_economic_actor", "best_economic_actor", "best_margin_actor",
                     "best_disturbance_rejection_actor", "best_low_activity_actor", "final_actor"):
            self.assertTrue((out / "models" / f"{name}.pth").exists())
        summary["TEST_ONLY_NO_TRAINING_SYNTHETIC_GLOBAL_STEP"] = True
        entry.save_json(out / "run_summary.json", summary)


if __name__ == "__main__":
    unittest.main()
