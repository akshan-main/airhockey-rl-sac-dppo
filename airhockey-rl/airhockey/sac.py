"""Soft Actor-Critic (Haarnoja et al. 2018).

Components:
  - GaussianActor: MLP with tanh-squashed output
  - TwinCritic: two Q networks to reduce overestimation
  - ReplayBuffer: ring buffer
  - SACAgent: owns the actor, critics, target critics, log_alpha, and
    the three optimizers. update() runs one gradient step.

Automatic entropy tuning: log_alpha is a trainable scalar optimized
toward target_entropy (defaults to -action_dim).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN = -2.0
LOG_STD_MAX = 1.0


# ── Actor and Critic networks ─────────────────────────────────
class GaussianActor(nn.Module):
    """Outputs (mean, log_std) of a pre-squash Gaussian, samples via
    reparameterization, then applies tanh. log_prob includes the tanh
    change-of-variable correction.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action in [-1, 1], log_prob)."""
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        pre = normal.rsample()
        action = torch.tanh(pre)
        # log p(a) = log N(pre) - sum_i log(1 - tanh(pre_i)^2)
        log_prob = normal.log_prob(pre) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean)


class TwinCritic(nn.Module):
    """Two Q networks, each (obs, act) -> scalar."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = self._build(obs_dim, act_dim, hidden)
        self.q2 = self._build(obs_dim, act_dim, hidden)

    @staticmethod
    def _build(obs_dim: int, act_dim: int, hidden: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


# ── Replay buffer ─────────────────────────────────────────────
class ReplayBuffer:
    """Ring buffer of (obs, act, reward, next_obs, done) on CPU."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def push(self, o: np.ndarray, a: np.ndarray, r: float, no: np.ndarray, d: bool) -> None:
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = no
        self.done[self.ptr] = 1.0 if d else 0.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch)
        return (
            torch.from_numpy(self.obs[idx]).to(device),
            torch.from_numpy(self.act[idx]).to(device),
            torch.from_numpy(self.rew[idx]).to(device),
            torch.from_numpy(self.next_obs[idx]).to(device),
            torch.from_numpy(self.done[idx]).to(device),
        )


# ── SAC agent ─────────────────────────────────────────────────
@dataclass
class SACConfig:
    obs_dim: int = 10
    act_dim: int = 2
    hidden: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 1e-4
    init_log_alpha: float = -2.3  # alpha ~= 0.1
    # Hard floor so auto-tuning can never collapse entropy. log(0.05) ~= -3.0.
    min_log_alpha: float = -3.0
    # Defaults to -0.5 * act_dim (half the pre-squash target), which
    # keeps the tanh-squashed Gaussian off the saturation walls.
    target_entropy: float | None = None


class SACAgent:
    """Owns the actor, critics, target critics, log_alpha, and three
    optimizers. Call act(obs) for rollout, update(batch) for training."""

    def __init__(self, cfg: SACConfig, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = GaussianActor(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(self.device)
        self.critic = TwinCritic(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(self.device)
        self.critic_target = TwinCritic(cfg.obs_dim, cfg.act_dim, cfg.hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.log_alpha = torch.tensor(
            cfg.init_log_alpha, device=self.device, requires_grad=True
        )
        self.target_entropy = (
            -0.5 * float(cfg.act_dim) if cfg.target_entropy is None else cfg.target_entropy
        )

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    # ── Rollout ──────────────────────────────────────────────
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        if deterministic:
            a = self.actor.act_deterministic(obs_t)
        else:
            a, _ = self.actor.sample(obs_t)
        return a.squeeze(0).cpu().numpy()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ── Update ───────────────────────────────────────────────
    def update(self, batch) -> dict[str, float]:
        obs, act, rew, next_obs, done = batch

        # ── Critic update ────────────────────────────────────
        with torch.no_grad():
            next_act, next_logp = self.actor.sample(next_obs)
            tq1, tq2 = self.critic_target(next_obs, next_act)
            target_q = torch.min(tq1, tq2) - self.alpha * next_logp.squeeze(-1)
            target = rew + self.cfg.gamma * (1.0 - done) * target_q

        q1, q2 = self.critic(obs, act)
        loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_critic.zero_grad(set_to_none=True)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_critic.step()

        # ── Actor update ─────────────────────────────────────
        new_act, logp = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_act)
        q_new = torch.min(q1_new, q2_new)
        loss_pi = (self.alpha.detach() * logp.squeeze(-1) - q_new).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_actor.step()

        # ── Entropy tuning ───────────────────────────────────
        loss_alpha = (
            -(self.log_alpha * (logp.detach() + self.target_entropy).squeeze(-1))
        ).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()
        # Clamp log_alpha to its floor so entropy can't collapse to zero.
        with torch.no_grad():
            self.log_alpha.clamp_(min=self.cfg.min_log_alpha)

        # ── Polyak target update ─────────────────────────────
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {
            "q_loss": float(loss_q.item()),
            "pi_loss": float(loss_pi.item()),
            "alpha": float(self.alpha.item()),
            "entropy": float(-logp.mean().item()),
        }

    # ── Save / load ──────────────────────────────────────────
    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "config": self.cfg.__dict__,
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
