"""Evaluation-only and selection regression checks; no replay or SAC updates."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from .sac import set_seed
from .train import run_episode
from .zanon2019_benchmark import make_setup, make_agent, sample_disturbance_path
from .zanon2019_training_report import (
    CheckpointSelection, EvidenceController, FixedPairedEvaluator,
    assessment, classify, plot_learning, update_pareto,
    evaluation_safety_anomalies,
)


class TrainingReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(42)
        cls.cfg, cls.model, cls.design, cls.omega, cls.domain, _ = make_setup(20, 42)
        cls.agent = make_agent(cls.cfg, "cpu")
        cls.agent.zero_initialize_residual_mean()
        cls.out = Path(__file__).parent / "evaporation_safe_sac" / (
            "test_zanon2019_training_report_" + uuid4().hex[:8])
        cls.evaluator = FixedPairedEvaluator(cls.cfg, cls.model, cls.design,
            cls.omega, cls.domain, "zanon2019_stochastic", [2, 1, 8, 5],
            20, 50, [420000, 420001, 420002], cls.out)
        cls.rows, cls.status = cls.evaluator.evaluate(cls.agent, 0, 0)

    def test_fixed_identity_and_report(self):
        self.assertEqual(len(self.rows), 3)
        for row in self.rows:
            self.assertAlmostEqual(row["economic_improvement_percent"], 0., places=9)
            for key in ("X2_IAE_ratio", "P2_IAE_ratio", "X2_ISE_ratio", "P2_ISE_ratio",
                        "P100_TV_ratio", "F200_TV_ratio", "G_X2_ratio", "G_P2_ratio"):
                self.assertAlmostEqual(row[key], 1., places=9)
            self.assertEqual(row["SAC_physical_state_violation_steps"], 0)
            self.assertEqual(row["SAC_QP_infeasible_count"], 0)
        repeated, _ = self.evaluator.evaluate(self.agent, 0, 0, label="repeat")
        self.assertEqual(self.rows[0]["SAC_J_econ"], repeated[0]["SAC_J_econ"])
        plot_learning(self.rows, self.out)

    def test_gate_and_selection(self):
        rows = [{**r, "economic_improvement_percent": 2.,
                 "SAC_W_exceedance_count": 20, "SAC_W_exceedance_rate": 1.}
                for r in self.rows]
        post = assessment(rows, 5, self.cfg.warmup_steps, self.cfg.warmup_steps)
        self.assertTrue(post["eligible_empirical_joint_checkpoint"])
        pre = assessment(rows, 0, 0, self.cfg.warmup_steps)
        self.assertFalse(pre["eligible_empirical_joint_checkpoint"])
        class DummyActor:
            def __init__(self): self.saved = []
            def save_actor(self, path): self.saved.append(path.name)
        actor = DummyActor()
        selection = CheckpointSelection(self.out)
        selection.consider(actor, pre)
        self.assertEqual(actor.saved, [])
        selection.consider(actor, post)
        self.assertIn("best_empirical_joint_actor.pth", actor.saved)
        omega_rows = [{**r, "SAC_Omega_exit_count": 1} for r in rows]
        os = assessment(omega_rows, 10, self.cfg.warmup_steps, self.cfg.warmup_steps)
        self.assertTrue(os["empirical_safe"])
        self.assertFalse(os["eligible_empirical_joint_checkpoint"])
        self.assertEqual(classify([post, os]), "D")
        bad = assessment([{**r, "SAC_QP_infeasible_count": 1} for r in rows],
                         10, self.cfg.warmup_steps, self.cfg.warmup_steps)
        before = len(actor.saved)
        selection.consider(actor, bad)
        self.assertEqual(before, len(actor.saved))
        worse = {**post, "economic_improvement_pct": 1., "mean_IAE_ratio": 1.05}
        statuses = [post, worse, pre, os, bad]
        update_pareto(statuses)
        self.assertTrue(post["pareto_nondominated"])
        self.assertFalse(worse["pareto_nondominated"])

    def test_evidence_hook_preserves_baseline_and_aborts(self):
        path = self.evaluator.cache[420000][0]
        ctrl = EvidenceController(self.cfg, self.model, self.design,
                                  self.omega, self.domain, path)
        _, records, _ = run_episode(self.cfg, self.model, ctrl, self.agent, None,
            np.random.default_rng(1), training=False, global_step=0,
            disturbance_trajectory=path,
            initial_state_override=self.cfg.robust_economic_reference_state)
        base = self.evaluator.cache[420000][1]
        np.testing.assert_array_equal([r["control"] for r in records],
                                      [r["control"] for r in base])
        self.assertEqual(len(ctrl.evidence), 20)
        self.assertTrue(ctrl.evidence[-1]["next_state_observed"])
        self.assertIsNotNone(ctrl.evidence[-1]["applied_displacement_from_baseline"])
        # Synthetic fault injection is an audit unit test, not a plant rollout.
        with self.assertRaisesRegex(RuntimeError, "empirical safety anomaly"):
            ctrl.audit_next_state(np.array([24., 50.]))
        self.assertEqual(ctrl.evidence[-1]["physical_state_violation"], 1)

    def test_orchestration_without_training(self):
        from . import zanon2019_train as entry
        from .sac import SACAgent
        out = self.out / ("orchestration_evaluation_only_" + uuid4().hex[:8])
        def evaluation_only(*args, **kwargs):
            # Exercise reporting/checkpoint I/O only. Synthetic global step is
            # used to test selection branches; it is NOT a learned experiment.
            args = list(args)
            args[4] = None
            kwargs["training"] = False
            stat, records, _ = run_episode(*args, **kwargs)
            return stat, records, 5000
        argv = ["zanon2019_train", "--episodes", "1", "--steps", "20",
                "--output-dir", str(out), "--device", "cpu"]
        with patch.object(sys, "argv", argv), patch.object(entry, "run_episode", evaluation_only), \
                patch.object(SACAgent, "update", side_effect=AssertionError("No training allowed")), \
                patch("builtins.print"):
            entry.main()
        summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["run_state"], "completed")
        self.assertFalse(summary["formal_safety_claim"])
        self.assertEqual(len(summary["comparison"]), 5)
        self.assertTrue((out / "models" / "final_actor.pth").exists())
        self.assertFalse((out / "models" / "best_empirical_joint_actor.pth").exists())
        self.assertTrue((out / "checkpoint_pareto.csv").exists())
        summary["TEST_ONLY_NO_SAC_UPDATES_SYNTHETIC_GLOBAL_STEP"] = True
        entry.save_json(out / "run_summary.json", summary)

    def test_evaluation_stop_gate_W_is_diagnostic(self):
        rows = [{**r, "SAC_W_exceedance_count": 1000,
                 "SAC_W_exceedance_rate": 1.0} for r in self.rows]
        self.assertEqual(evaluation_safety_anomalies(rows), [])
        for key in ("physical_state_violation_steps", "physical_input_violation_count",
                    "X2_violation_count", "P2_physical_violation_count",
                    "QP_infeasible_count", "Omega_exit_count", "Omega_exit_events"):
            for controller in ("baseline", "SAC"):
                with self.subTest(key=key, controller=controller):
                    changed = [{**rows[0], f"{controller}_{key}": 1}, *rows[1:]]
                    failures = evaluation_safety_anomalies(changed)
                    self.assertEqual(len(failures), 1)
                    self.assertEqual(failures[0]["controller"], controller)
                    self.assertEqual(failures[0]["seed"], rows[0]["seed"])

    def test_evaluation_failure_stops_before_next_episode(self):
        from . import zanon2019_train as entry
        from .sac import SACAgent
        original_evaluate = FixedPairedEvaluator.evaluate
        # All faults are injected into metrics after a safe evaluation-only
        # rollout. Test outputs are explicitly separate from real experiments.
        for target_episode, key, controller in (
                (0, "Omega_exit_count", "SAC"),
                (1, "QP_infeasible_count", "SAC"),
                (1, "physical_state_violation_steps", "SAC"),
                (0, "physical_input_violation_count", "baseline")):
            with self.subTest(episode=target_episode, key=key, controller=controller):
                out = self.out / ("fault_injection_evaluation_only_" + uuid4().hex[:8])
                calls = []
                def evaluation_only(*args, **kwargs):
                    calls.append(1)
                    args = list(args)
                    args[4] = None
                    kwargs["training"] = False
                    stat, records, _ = run_episode(*args, **kwargs)
                    return stat, records, 5000
                def inject(evaluator, agent, episode, global_step, label=None):
                    rows, status = original_evaluate(evaluator, agent, episode, global_step, label)
                    if episode == target_episode:
                        rows[0][f"{controller}_{key}"] = 1
                        status = assessment(rows, episode, global_step, evaluator.cfg.warmup_steps)
                    return rows, status
                argv = ["zanon2019_train", "--episodes", "2", "--steps", "20",
                        "--eval-every", "1", "--output-dir", str(out), "--device", "cpu"]
                with patch.object(sys, "argv", argv), \
                        patch.object(entry, "run_episode", evaluation_only), \
                        patch.object(FixedPairedEvaluator, "evaluate", inject), \
                        patch.object(SACAgent, "update", side_effect=AssertionError("No training allowed")), \
                        patch("builtins.print"):
                    with self.assertRaises(entry.EvaluationSafetyAbort):
                        entry.main()
                self.assertEqual(len(calls), target_episode)
                summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["run_state"], "evaluation_aborted")
                self.assertEqual(summary["case"], "D")
                failure = json.loads((out / "evaluation_abort.json").read_text(encoding="utf-8"))
                self.assertEqual(failure["episode"], target_episode)
                self.assertEqual(failure["phase"], "fixed_evaluation")
                self.assertEqual(failure["anomalies"][0]["controller"], controller)
                self.assertFalse((out / "training_abort.json").exists())
                self.assertFalse((out / "models" / "final_actor.pth").exists())
                self.assertTrue((out / "models" / "evaluation_failure_actor.pth").exists())
                self.assertTrue((out / "models" / "evaluation_failure_checkpoint.pth").exists())
                self.assertTrue((out / "evaluation_failure_metrics.csv").exists())
                self.assertTrue((out / "checkpoint_pareto.csv").exists())
                self.assertTrue((out / "evaluation_trajectories" /
                                 f"episode_{target_episode:04d}" / "seed_420000.csv").exists())
                summary["TEST_ONLY_SYNTHETIC_EVALUATION_FAULT_NO_SAC_UPDATES"] = True
                entry.save_json(out / "run_summary.json", summary)


if __name__ == "__main__":
    unittest.main()
