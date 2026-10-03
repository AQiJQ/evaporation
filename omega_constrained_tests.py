"""Fast checks for frozen shock sequence and dynamic component replay."""
from __future__ import annotations

import csv

import numpy as np
import torch

from .config import ExperimentConfig
from .omega_constrained_replay import ConstrainedReplayBuffer, DualState
from .residual_action_space_diagnosis import REPO_DIR
from .train import (
    ACTION_DIM, balanced_random_paper2016_schedule,
    sample_paper2016_training_shock_scale,
)


def test_dynamic_replay():
    dual = DualState.create(np.ones(5), 10.0)
    buffer = ConstrainedReplayBuffer(23, 2, 4, torch.device("cpu"), dual)
    components = np.array([2.0, 1.0, 0.5, 0.25, -0.1, 0.0])
    buffer.add_components(np.zeros(23), np.zeros(2), components,
                          np.zeros(23), False)
    first = float(buffer.sample(1)[2][0, 0])
    dual.update(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    second = float(buffer.sample(1)[2][0, 0])
    if abs(first - 2.0) > 1e-7 or abs(second - 1.0) > 1e-7:
        raise AssertionError("Stored replay did not use current dual multipliers")
    if float(buffer.rewards[0, 0]) != 0.0:
        raise AssertionError("Replay unexpectedly stored a weighted scalar")


def test_full_shock_sequence():
    path = REPO_DIR / (
        "evaporation_safe_sac/outputs_omega_safe_B_dwell20_interior_obs23_"
        "seed42_300x300_local_full/training_log.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))
    if len(reference) != 300:
        raise AssertionError("Need complete prior 23D training log")
    cfg = ExperimentConfig(
        benchmark_profile="zanon2016", experiment_mode="proposed",
        episodes=300, steps_per_episode=300, seed=42,
    )
    rng = np.random.default_rng(42)
    schedule = balanced_random_paper2016_schedule(300, 42)
    global_step = 0
    for episode, scenario in enumerate(schedule):
        shock_scale = sample_paper2016_training_shock_scale(cfg, rng)
        old = reference[episode]
        if scenario != old["scenario_type"] or abs(
            shock_scale - float(old["shock_scale_lambda"])
        ) > 1e-12:
            raise AssertionError(f"Shock sequence mismatch at episode {episode + 1}")
        for _ in range(300):
            if global_step < cfg.warmup_steps:
                rng.uniform(-1.0, 1.0, ACTION_DIM)
            global_step += 1


if __name__ == "__main__":
    test_dynamic_replay()
    test_full_shock_sequence()
    print("constrained replay and 300-episode shock sequence checks passed")
