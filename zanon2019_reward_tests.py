"""Reward calibration regression tests; no SAC optimizer updates or training."""
from copy import copy
from pathlib import Path
import json
import unittest
from unittest.mock import patch

import numpy as np

from .train import run_episode
from .zanon2019_authority import AuthorityController
from .zanon2019_benchmark import make_setup, sample_disturbance_path
from .zanon2019_reward_audit import (
    DEFAULT_OUTPUT, WEIGHT_FIELDS, calibrate, apply_calibration, reward_metrics,
)


class RewardCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.model, cls.design, cls.omega, cls.domain, cls.old = make_setup(20, 42)
        cls.cfg.stochastic_residual_scale = .2
        cls.path = DEFAULT_OUTPUT / "reward_calibration.json"
        cls.data = json.loads(cls.path.read_text(encoding="utf-8"))
        source = Path(cls.data["source_directory"])
        cls.geometry = list(json.loads((source / "experiment_manifest.json").read_text(
            encoding="utf-8"))["frozen_geometry_sources"])

    def test_calibration_targets_and_zero_guard(self):
        means = self.data["baseline_means"]
        econ = means["mean_absolute_economic_reward"]
        weights = calibrate(means, econ, self.old)
        for name, target in self.data["target_fractions"].items():
            self.assertAlmostEqual(weights[name] * means[name] / econ, target)
        self.assertEqual(weights["saturation"], self.old["saturation"])
        with self.assertRaises(ValueError):
            calibrate({**means, "F200_move": 0}, econ, self.old)

    def test_opt_in_only_changes_three_weights(self):
        cfg = copy(self.cfg)
        weights, _ = apply_calibration(cfg, self.path, geometry_sources=self.geometry)
        changed = {k for k in vars(cfg) if not np.array_equal(
            np.asarray(vars(cfg)[k]), np.asarray(vars(self.cfg)[k]))}
        self.assertEqual(changed, {WEIGHT_FIELDS[n] for n in ("state_recovery", "P100_move", "F200_move")})
        self.assertEqual(weights, self.data["calibrated_weights"])
        cfg = copy(self.cfg); cfg.stochastic_residual_scale = .3
        with self.assertRaises(ValueError):
            apply_calibration(cfg, self.path, geometry_sources=self.geometry)
        cfg = copy(self.cfg); cfg.reward_cost_scale *= 2
        with self.assertRaises(ValueError):
            apply_calibration(cfg, self.path, geometry_sources=self.geometry)
        bad = {**self.data, "calibrated_weights": {**weights, "F200_move": -1}}
        with patch.object(Path, "read_text", return_value=json.dumps(bad)):
            with self.assertRaises(ValueError):
                apply_calibration(copy(self.cfg), self.path, geometry_sources=self.geometry)

    def test_fixed_policy_controls_unchanged_reward_changes_exactly(self):
        class FrozenPolicy:
            def select_action(self, obs, deterministic=True):
                assert len(obs) == 23
                return np.array([.2, -.1], dtype=np.float32)
        disturbance, _ = sample_disturbance_path(self.cfg, "zanon2019_stochastic", 420000,
            20, [2, 1, 8, 5], 20, 50)
        new = copy(self.cfg)
        apply_calibration(new, self.path, geometry_sources=self.geometry)
        def rollout(cfg):
            ctrl = AuthorityController(cfg, self.model, self.design, self.omega, self.domain)
            return run_episode(cfg, self.model, ctrl, FrozenPolicy(), None,
                np.random.default_rng(1), training=False, global_step=0,
                disturbance_trajectory=disturbance,
                initial_state_override=cfg.robust_economic_reference_state)[1]
        # Fail loudly if any optimizer is accidentally invoked by this test.
        with patch("evaporation.sac.SACAgent.update", side_effect=AssertionError("No training")):
            old_records, new_records = rollout(self.cfg), rollout(new)
        losses = {"state_recovery": "state_recovery_loss", "P100_move": "p100_move_loss",
                  "F200_move": "f200_move_loss", "saturation": "saturation_loss"}
        for old, now in zip(old_records, new_records):
            for field in ("state", "control", "raw_action", "residual", "applied_residual"):
                np.testing.assert_array_equal(old[field], now[field])
            delta = -sum((self.data["calibrated_weights"][k] - self.old[k]) * old[v]
                         for k, v in losses.items())
            self.assertAlmostEqual(now["total_reward"] - old["total_reward"], delta, places=8)
        report = reward_metrics(new_records)
        self.assertAlmostEqual(report["reward_replay_equivalent_total"], sum(
            r["total_reward"] + r["rpi_violation_event_penalty"] + r["rpi_excess_penalty"]
            for r in new_records))
        # Applying the same exclusion twice must not add RPI terms again.
        replay_records = [{**r, "total_reward": r["total_reward"] + r["rpi_violation_event_penalty"]
            + r["rpi_excess_penalty"], "rpi_penalty_excluded_from_training_reward": True}
            for r in new_records]
        self.assertAlmostEqual(reward_metrics(replay_records)["reward_replay_equivalent_total"],
                               report["reward_replay_equivalent_total"])


if __name__ == "__main__":
    unittest.main()
