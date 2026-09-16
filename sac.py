"""PyTorch continuous Soft Actor-Critic aligned with the original CSTR-SAC."""
from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required by the continuous evaporator SAC. "
        "Install torch in the Python environment used to launch training."
    ) from exc


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReplayBuffer:
    """Continuous-action replay buffer matching the CSTR-SAC data path."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.execution_masks = np.ones((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    @property
    def allocated_bytes(self) -> int:
        return int(
            self.obs.nbytes + self.next_obs.nbytes + self.actions.nbytes
            + self.rewards.nbytes + self.dones.nbytes
            + self.execution_masks.nbytes
        )

    def add(self, obs, action, reward, next_obs, done, execution_mask: float = 1.0) -> None:
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32)
        self.rewards[self.ptr, 0] = float(reward)
        self.next_obs[self.ptr] = np.asarray(next_obs, dtype=np.float32)
        self.dones[self.ptr, 0] = float(done)
        self.execution_masks[self.ptr, 0] = float(np.clip(execution_mask, 0.0, 1.0))
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=self.device),
            torch.as_tensor(self.actions[idx], device=self.device),
            torch.as_tensor(self.rewards[idx], device=self.device),
            torch.as_tensor(self.next_obs[idx], device=self.device),
            torch.as_tensor(self.dones[idx], device=self.device),
            torch.as_tensor(self.execution_masks[idx], device=self.device),
        )


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Tanh-squashed Gaussian actor with reparameterized sampling."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        return self.mean(hidden), torch.clamp(self.log_std(hidden), LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample()
        action = torch.tanh(raw)
        log_prob = normal.log_prob(raw) - torch.log(1.0 - action.pow(2) + EPS)
        return action, log_prob.sum(dim=-1, keepdim=True), torch.tanh(mean)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool, device: torch.device) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if deterministic:
            mean, _ = self(obs_t)
            action = torch.tanh(mean)
        else:
            action, _, _ = self.sample(obs_t)
        return action.cpu().numpy()[0]


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.q = MLP(obs_dim + action_dim, 1, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=-1))


@dataclass
class SACConfig:
    gamma: float = 0.995
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    hidden_dim: int = 256
    alpha: float = 0.2
    alpha_min: float = 0.015
    entropy_tuning_warmup_updates: int = 10000
    auto_entropy_tuning: bool = True
    target_entropy: float = -2.0
    actor_ema_decay: float = 0.0


class SACAgent:
    """Continuous twin-Q SAC implementation used only by the evaporator code."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: SACConfig, device: str | torch.device = "auto"):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.actor = GaussianPolicy(obs_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.actor_ema = copy.deepcopy(self.actor).to(self.device)
        self.actor_ema.requires_grad_(False)
        self.q1 = QNetwork(obs_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        self.q1_target.requires_grad_(False)
        self.q2_target.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)
        self.auto_entropy_tuning = bool(cfg.auto_entropy_tuning)
        self.target_entropy = float(cfg.target_entropy)
        if self.auto_entropy_tuning:
            self.log_alpha = torch.tensor(
                np.log(max(cfg.alpha, cfg.alpha_min)), dtype=torch.float32,
                requires_grad=True, device=self.device,
            )
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        else:
            self.log_alpha = None
            self.alpha_opt = None
        self.alpha = float(max(cfg.alpha, cfg.alpha_min))
        self.total_updates = 0
        self.policy_output_scale = 1.0

    def zero_initialize_residual_mean(self) -> None:
        nn.init.zeros_(self.actor.mean.weight)
        nn.init.zeros_(self.actor.mean.bias)
        self.actor_ema.load_state_dict(self.actor.state_dict())

    def select_action(self, obs: np.ndarray, deterministic: bool = False, *, use_ema: bool = False) -> np.ndarray:
        policy = self.actor_ema if use_ema else self.actor
        return self.policy_output_scale * policy.act(obs, deterministic, self.device)

    def set_policy_output_scale(self, scale: float) -> None:
        scale = float(scale)
        if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("policy output scale must be finite and in [0, 1]")
        self.policy_output_scale = scale

    def update_actor_ema(self) -> None:
        decay = float(self.cfg.actor_ema_decay)
        if not 0.0 < decay < 1.0:
            self.actor_ema.load_state_dict(self.actor.state_dict())
            return
        with torch.no_grad():
            for ema, current in zip(self.actor_ema.parameters(), self.actor.parameters()):
                ema.mul_(decay).add_(current, alpha=1.0 - decay)

    def update(self, replay: ReplayBuffer, batch_size: int) -> Dict[str, float]:
        obs, action, reward, next_obs, done, execution_mask = replay.sample(batch_size)
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_obs)
            q_next = torch.min(
                self.q1_target(next_obs, next_action),
                self.q2_target(next_obs, next_action),
            ) - self.alpha * next_logp
            target_q = reward + (1.0 - done) * self.cfg.gamma * q_next

        mask_sum = execution_mask.sum().clamp_min(1.0)
        q1_loss = (
            F.mse_loss(self.q1(obs, action), target_q, reduction="none")
            * execution_mask
        ).sum() / mask_sum
        q2_loss = (
            F.mse_loss(self.q2(obs, action), target_q, reduction="none")
            * execution_mask
        ).sum() / mask_sum
        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        new_action, logp, _ = self.actor.sample(obs)
        q_pi = torch.min(self.q1(obs, new_action), self.q2(obs, new_action))
        actor_loss = ((self.alpha * logp - q_pi) * execution_mask).sum() / mask_sum
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss_value = 0.0
        tune_entropy = (
            self.auto_entropy_tuning
            and self.log_alpha is not None
            and self.alpha_opt is not None
            and self.total_updates >= self.cfg.entropy_tuning_warmup_updates
        )
        if tune_entropy:
            alpha_loss = -(
                self.log_alpha * (logp + self.target_entropy).detach()
                * execution_mask
            ).sum() / mask_sum
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(min=float(np.log(self.cfg.alpha_min)))
            self.alpha = float(self.log_alpha.exp().detach().cpu().item())
            alpha_loss_value = float(alpha_loss.detach().cpu().item())

        with torch.no_grad():
            for source, target in zip(self.q1.parameters(), self.q1_target.parameters()):
                target.mul_(1.0 - self.cfg.tau).add_(source, alpha=self.cfg.tau)
            for source, target in zip(self.q2.parameters(), self.q2_target.parameters()):
                target.mul_(1.0 - self.cfg.tau).add_(source, alpha=self.cfg.tau)

        self.total_updates += 1
        return {
            "q1_loss": float(q1_loss.detach().cpu().item()),
            "q2_loss": float(q2_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha),
            "mean_logp": float(logp.detach().mean().cpu().item()),
        }

    def save_actor(self, path: str | Path, *, use_ema: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": (self.actor_ema if use_ema else self.actor).state_dict(),
            "alpha": self.alpha,
            "policy_output_scale": self.policy_output_scale,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }, path)

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_ema": self.actor_ema.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "alpha": self.alpha,
            "cfg": dict(self.cfg.__dict__),
            "total_updates": self.total_updates,
            "policy_output_scale": self.policy_output_scale,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }
        if self.log_alpha is not None and self.alpha_opt is not None:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["alpha_opt"] = self.alpha_opt.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path, load_optimizers: bool = True) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        if int(checkpoint.get("obs_dim", self.obs_dim)) != self.obs_dim:
            raise ValueError("Checkpoint observation dimension does not match this evaporator.")
        if int(checkpoint.get("action_dim", self.action_dim)) != self.action_dim:
            raise ValueError("Checkpoint action dimension does not match this evaporator.")
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_ema.load_state_dict(checkpoint.get("actor_ema", checkpoint["actor"]))
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint.get("q1_target", checkpoint["q1"]))
        self.q2_target.load_state_dict(checkpoint.get("q2_target", checkpoint["q2"]))
        if load_optimizers:
            self.actor_opt.load_state_dict(checkpoint["actor_opt"])
            self.q1_opt.load_state_dict(checkpoint["q1_opt"])
            self.q2_opt.load_state_dict(checkpoint["q2_opt"])
            # A resumed run follows the learning rates requested by the current
            # experiment rather than silently inheriting rates from an older run.
            for group in self.actor_opt.param_groups:
                group["lr"] = self.cfg.actor_lr
            for optimizer in (self.q1_opt, self.q2_opt):
                for group in optimizer.param_groups:
                    group["lr"] = self.cfg.critic_lr
        self.alpha = max(
            float(checkpoint.get("alpha", self.alpha)),
            float(self.cfg.alpha_min),
        )
        self.total_updates = int(checkpoint.get("total_updates", 0))
        self.policy_output_scale = float(checkpoint.get("policy_output_scale", 1.0))
        if self.log_alpha is not None and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
            with torch.no_grad():
                self.log_alpha.clamp_(min=float(np.log(self.cfg.alpha_min)))
            self.alpha = float(self.log_alpha.exp().detach().cpu().item())
            if load_optimizers and self.alpha_opt is not None and "alpha_opt" in checkpoint:
                self.alpha_opt.load_state_dict(checkpoint["alpha_opt"])
                for group in self.alpha_opt.param_groups:
                    group["lr"] = self.cfg.alpha_lr
