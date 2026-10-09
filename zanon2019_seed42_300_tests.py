"""No-training contracts for the isolated 300-episode launcher and return plots."""
import ast
import inspect
from pathlib import Path
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
import uuid
import unittest
from unittest.mock import patch, Mock

import numpy as np

from . import zanon2019_seed42_300 as run


@contextmanager
def scratch_folder():
    # Windows sandbox TemporaryDirectory ACLs can deny writes to its children.
    # Use a uniquely named, ordinary workspace directory. Keep only tiny test
    # fixtures here; never put synthetic data in the real experiment output.
    directory = run.v1.REPO / "tmp" / ("seed42_300_tests_" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    yield str(directory)


class Seed42ExtensionTests(unittest.TestCase):
    def test_defaults_are_single_seed300_cuda_and_original_horizon(self):
        args = run.parser().parse_args(["train"])
        self.assertEqual((args.seed, args.episodes, args.steps, args.device), (42, 300, 1000, "cuda"))
        self.assertEqual(args.output_dir, run.DEFAULT_OUTPUT)

    def test_no_second_seed_or_changed_horizon(self):
        for option, value in (("--seed", "2027"), ("--episodes", "100"), ("--steps", "300")):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                run.parser().parse_args(["train", option, value])

    def test_output_preserves_existing_roots(self):
        self.assertEqual(run.output_path(run.DEFAULT_OUTPUT), run.DEFAULT_OUTPUT.resolve())
        for path in (run.v2.ROOT / "seed_42", run.ROOT, run.ROOT / ".." / "old_run"):
            with self.assertRaises(ValueError):
                run.output_path(path)

    def test_identical_first100_path_seeds_and_distinct_extension(self):
        self.assertEqual([run.episode_seed(i) for i in range(1, 101)],
                         [42 * 1000000 + i for i in range(1, 101)])
        self.assertEqual(run.episode_seed(300), 42000300)

    def test_cuda_has_no_silent_fallback(self):
        args = run.parser().parse_args(["preflight"])
        with patch.object(run.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no silent CPU fallback"):
                run.preflight(args)

    def test_nonempty_directory_is_never_overwritten(self):
        with scratch_folder() as tmp:
            root = Path(tmp)
            evidence = root / "partial"
            evidence.mkdir()
            run.v1.save(evidence / "run_summary.json", {"run_state": "aborted"})
            with patch.object(run, "ROOT", root):
                args = run.parser().parse_args(["preflight", "--output-dir", str(evidence)])
                with self.assertRaisesRegex(RuntimeError, "Preserve the existing run"):
                    run.preflight(args)
            self.assertTrue((evidence / "run_summary.json").exists())

    def test_trailing_mean_does_not_invent_points_or_use_future(self):
        out = run.trailing_mean([1, 2, 3, 4], 3)
        self.assertTrue(np.isnan(out[:2]).all())
        np.testing.assert_allclose(out[2:], [2, 3])
        self.assertTrue(np.isnan(run.trailing_mean([1, 2], 10)).all())
        with self.assertRaises(ValueError):
            run.trailing_mean([np.inf], 1)

    def test_return_matches_replay_not_legacy_diagnostic(self):
        with scratch_folder() as tmp:
            folder = Path(tmp)
            run.write_csv(folder / "training_log.csv", [dict(episode=1, episode_return=2,
                economic_reward_component=3, smoothness_regularization_component=-1,
                lagrangian_constraint_component=0, legacy_return_diagnostic=-9999)])
            training, fixed = run.learning_data(folder)
            self.assertEqual(training, [(1, 2)])
            self.assertEqual(fixed, [])

    def test_inconsistent_return_and_duplicate_episodes_fail(self):
        with scratch_folder() as tmp:
            folder = Path(tmp)
            row = dict(episode=1, episode_return=2, economic_reward_component=1,
                       smoothness_regularization_component=0, lagrangian_constraint_component=0)
            run.write_csv(folder / "training_log.csv", [row])
            with self.assertRaisesRegex(ValueError, "actual paired replay"):
                run.learning_data(folder)
            row["episode_return"] = 1
            run.write_csv(folder / "training_log.csv", [row, row])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                run.learning_data(folder)

    def test_real_plot_smoke_and_partial_status(self):
        with scratch_folder() as tmp:
            root = Path(tmp)
            folder = root / "plot_fixture"
            folder.mkdir()
            run.write_csv(folder / "training_log.csv", [dict(episode=i, episode_return=float(i-5),
                economic_reward_component=float(i-4), smoothness_regularization_component=-1.,
                lagrangian_constraint_component=0.) for i in range(1, 13)])
            run.v1.save(folder / "checkpoints.json", [dict(episode=0, total_training_reward=0,
                mean_economic_improvement_pct=0), dict(episode=5, total_training_reward=-1,
                mean_economic_improvement_pct=-.1)])
            run.v1.save(folder / "run_summary.json", {"run_state": "aborted"})
            with patch.object(run, "ROOT", root):
                figures = run.plot_returns(folder)
            for name in ("training_return_learning_curve.png", "fixed_eval_return_learning_curve.png",
                         "fixed_eval_economic_learning_curve.png"):
                self.assertGreater((figures / name).stat().st_size, 1000)
            self.assertEqual(run.v2.load(folder / "run_summary.json")["run_state"], "aborted")

    def test_plot_before_training_has_no_synthetic_output(self):
        with scratch_folder() as tmp:
            root = Path(tmp)
            folder = root / "empty"
            folder.mkdir()
            with patch.object(run, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "No recorded return"):
                    run.plot_returns(folder)
            self.assertFalse((folder / "figures").exists())

    def test_preflight_and_plot_never_call_training(self):
        tree = ast.parse(inspect.getsource(run.preflight))
        called = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)]
        self.assertNotIn("run_episode", called)
        self.assertNotIn("train", called)
        with patch.object(run, "train") as training, patch.object(run, "preflight") as preflight:
            with patch("sys.argv", ["launcher", "preflight"]):
                run.main()
            preflight.assert_called_once()
            training.assert_not_called()

    def test_all300_episodes_and61_evaluations_with_mocked_plant_and_sac(self):
        # Exercise orchestration, not learning: no nonlinear simulation or
        # SAC network/optimizer is created, stepped or trained by this test.
        with scratch_folder() as tmp, ExitStack() as stack:
            root = Path(tmp)
            output = root / "mock_orchestration"
            cfg = SimpleNamespace(replay_capacity=20, warmup_steps=1000,
                                  robust_economic_reference_state=np.array([25.39, 50.125]))
            env = (cfg, None, None, None, None)
            net = SimpleNamespace(parameters=lambda: [])
            agent = SimpleNamespace(device="cpu", cfg=SimpleNamespace(), alpha=.2,
                actor=net, q1=net, q2=net, zero_initialize_residual_mean=Mock(),
                save_actor=Mock(), save_checkpoint=Mock())
            replay = SimpleNamespace(weight=17.613805207960148, size=0,
                                     episode_components=np.zeros((1, 7)), bind=Mock())
            cfg.batch_size = 256
            metric = dict(total_training_reward=0., economic_reward_component=0.,
                smoothness_regularization_component=0., lagrangian_constraint_component=0.,
                sum_abs_economic_reward=0., economic_improvement_pct=0.,
                minimum_X2_margin=.3, action_collapse_fraction=0.,
                **{key: 0 for key in run.v1.SAFETY})
            paths = {seed: (np.zeros((1, 4)), [{}], {}, None) for seed in run.v2.DEV}
            controller = SimpleNamespace(evidence=[])
            mocks = {
                "preflight": lambda args: (env, {"lambda_smooth": replay.weight}, output, None),
                "make_agent": lambda cfg, device: agent,
                "EconomicReplay": lambda *args: replay,
                "frozen_hashes": lambda: {},
                "sample_disturbance_path": lambda *args: (np.zeros((1, 4)), {}),
                "run_episode": lambda *args, **kwargs: ({"return": 0}, [{}], kwargs["global_step"] + 1000),
                "plot_returns": lambda path: path / "figures",
                "set_seed": lambda seed: None,
            }
            for name, value in mocks.items():
                stack.enter_context(patch.object(run, name, side_effect=value))
            stack.enter_context(patch.object(run.v1, "protocol", return_value={}))
            stack.enter_context(patch.object(run.v1, "paths_and_baselines", return_value=paths))
            stack.enter_context(patch.object(run.v1, "rollout", return_value=([{}], {}, controller)))
            stack.enter_context(patch.object(run.v1, "PairedEvidenceController", return_value=controller))
            stack.enter_context(patch.object(run.recovery, "full_metrics", side_effect=
                                           lambda *args: (metric.copy(), np.zeros((1, 7)), {})))
            stack.enter_context(patch.object(run.recovery, "trace", return_value=[{}]))
            stack.enter_context(patch("builtins.print"))
            run.train(run.parser().parse_args(["train", "--device", "cpu"]))
            log = run.v1.read_csv(output / "training_log.csv")
            fixed = run.v2.load(output / "checkpoints.json")
            summary = run.v2.load(output / "run_summary.json")
            self.assertEqual(len(log), 300)
            self.assertEqual([int(row["disturbance_seed"]) for row in log],
                             [42000000 + i for i in range(1, 301)])
            self.assertEqual([row["episode"] for row in fixed], list(range(0, 301, 5)))
            self.assertEqual(summary["global_step"], 300000)
            self.assertEqual(summary["run_state"], "completed")
            self.assertFalse(summary["automatic_next_seed"])
            self.assertEqual(agent.save_actor.call_count, 61)


if __name__ == "__main__":
    unittest.main()
