"""Component replay with current-episode Lagrange multipliers at sample time."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .omega_constrained_objective import KEYS
from .sac import ReplayBuffer


@dataclass
class DualState:
    values: np.ndarray
    learning_rates: np.ndarray
    lambda_max: float

    @classmethod
    def create(cls, learning_rates, lambda_max):
        rates = np.asarray(learning_rates, dtype=float)
        if rates.shape != (len(KEYS),) or np.any(rates <= 0):
            raise ValueError("Need one positive dual learning rate per constraint")
        if not np.isfinite(lambda_max) or lambda_max <= 0:
            raise ValueError("lambda_max must be positive")
        return cls(np.zeros(len(KEYS), dtype=float), rates, float(lambda_max))

    def update(self, violations):
        g = np.asarray(violations, dtype=float)
        if g.shape != self.values.shape or not np.all(np.isfinite(g)):
            raise ValueError("Invalid episode constraint vector")
        self.values = np.clip(
            self.values + self.learning_rates * g,
            0.0, self.lambda_max,
        )
        return self.values.copy()


class ConstrainedReplayBuffer(ReplayBuffer):
    """Never stores a lambda-weighted scalar as the source of truth."""

    def __init__(self, obs_dim, action_dim, capacity, device, dual: DualState):
        super().__init__(obs_dim, action_dim, capacity, device)
        self.components = np.zeros((capacity, 1 + len(KEYS)), dtype=np.float32)
        self.dual = dual

    @property
    def allocated_bytes(self):
        return super().allocated_bytes + self.components.nbytes

    def add_components(self, obs, action, components, next_obs, done,
                       execution_mask=1.0):
        values = np.asarray(components, dtype=np.float32)
        if values.shape != (1 + len(KEYS),) or not np.all(np.isfinite(values)):
            raise ValueError("Expected six finite independent reward components")
        position = self.ptr
        super().add(obs, action, 0.0, next_obs, done, execution_mask)
        self.components[position] = values

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        components = torch.as_tensor(self.components[idx], device=self.device)
        dual = torch.as_tensor(
            self.dual.values, device=self.device, dtype=components.dtype
        )
        reward = components[:, :1] - (
            components[:, 1:] * dual.unsqueeze(0)
        ).sum(dim=1, keepdim=True)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            reward,
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.dones[idx], device=self.device),
            torch.as_tensor(self.execution_masks[idx], device=self.device),
        )
