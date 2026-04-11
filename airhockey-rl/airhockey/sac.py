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

    def sample_mixed(
        self,
        batch: int,
        device: torch.device,
        demo_buffer: "ReplayBuffer",
        demo_fraction: float = 0.25,
    ):
        """Sample a batch that's `demo_fraction` from a persistent demo
        buffer and the rest from this (online) buffer. This prevents
        demo transitions from being diluted as the online buffer grows,
        which is the SACfD trick (Vecerik et al. 2017)."""
        n_demo = min(int(batch * demo_fraction), demo_buffer.size)
        n_online = batch - n_demo
        d_idx = np.random.randint(0, demo_buffer.size, size=n_demo)
        o_idx = np.random.randint(0, self.size, size=n_online)
        def cat(arr_online, arr_demo):
            return torch.from_numpy(
                np.concatenate([arr_online[o_idx], arr_demo[d_idx]], axis=0)
            ).to(device)
        return (
            cat(self.obs, demo_buffer.obs),
            cat(self.act, demo_buffer.act),
            cat(self.rew, demo_buffer.rew),
            cat(self.next_obs, demo_buffer.next_obs),
            cat(self.done, demo_buffer.done),
        )


# ── N-step replay buffer ─────────────────────────────────────
class NStepReplayBuffer:
    """Stores sequences of length n+1 for SACn. Each entry contains
    obs[0..n], act[0..n], rew[0..n-1], done[0..n-1], logp[0..n-1].
    logp[i] is log π_old(a_{i+1} | s_{i+1}) at collection time.
    Bootstrap values for each future state are cached and refreshed
    periodically to avoid the expensive target-critic forward pass
    on every update."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, n: int):
        self.capacity = capacity
        self.n = n
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, n + 1, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, n + 1, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, n), dtype=np.float32)
        self.done = np.zeros((capacity, n), dtype=np.float32)
        self.logp = np.zeros((capacity, n), dtype=np.float32)
        self.bootstrap = np.zeros((capacity, n), dtype=np.float32)

    def push(self, obs_seq, act_seq, rew_seq, done_seq, logp_seq,
             bootstrap_vals=None):
        self.obs[self.ptr] = obs_seq
        self.act[self.ptr] = act_seq
        self.rew[self.ptr] = rew_seq
        self.done[self.ptr] = done_seq
        self.logp[self.ptr] = logp_seq
        if bootstrap_vals is not None:
            self.bootstrap[self.ptr] = bootstrap_vals
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch)
        return (
            torch.from_numpy(self.obs[idx]).to(device),
            torch.from_numpy(self.act[idx]).to(device),
            torch.from_numpy(self.rew[idx]).to(device),
            torch.from_numpy(self.done[idx]).to(device),
            torch.from_numpy(self.logp[idx]).to(device),
            torch.from_numpy(self.bootstrap[idx]).to(device),
        )

    @torch.no_grad()
    def refresh_bootstrap(self, agent, device: torch.device,
                          batch_size: int = 4096):
        """Recompute cached bootstrap V(s_{t+τ}) for all stored
        sequences using the current target critic + actor."""
        if self.size == 0:
            return
        n = self.n
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            sz = end - start
            future_obs = torch.from_numpy(
                self.obs[start:end, 1:n + 1].reshape(sz * n, -1)
            ).to(device)
            fa, flp = agent.actor.sample(future_obs)
            tq1, tq2 = agent.critic_target(future_obs, fa)
            v = torch.min(tq1, tq2) - agent.alpha * flp.squeeze(-1)
            self.bootstrap[start:end] = v.reshape(sz, n).cpu().numpy()


# ── SAC agent ─────────────────────────────────────────────────
@dataclass
class SACConfig:
    obs_dim: int = 10
    act_dim: int = 2
    hidden: int = 512
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

    # ── SACn update (Łyskawa et al. 2025) — vectorized ─────────
    def update_sacn(self, seq_batch, n: int = 8, q_b: float = 0.75) -> dict[str, float]:
        """Full SACn: n-step returns averaged over τ=1..n, importance
        sampling with quantile clipping, τ-sampled entropy.

        Fully vectorized — no Python loops over τ or i. All n targets
        are computed in parallel via cumulative products and batch
        forward passes.

        seq_batch: (obs_seq, act_seq, rew_seq, done_seq, logp_old_seq, bootstrap_cached)
          obs_seq: (B, n+1, obs_dim)
          act_seq: (B, n+1, act_dim)  — act_seq[:,n] unused (padding)
          rew_seq: (B, n)
          done_seq: (B, n)
          logp_old_seq: (B, n) — log π_old(a_i | s_i) at collection time
          bootstrap_cached: (B, n) — cached V(s_{t+τ}) from refresh_bootstrap
        """
        obs_seq, act_seq, rew_seq, done_seq, logp_old_seq, bootstrap_cached = seq_batch
        B = obs_seq.shape[0]
        gamma = self.cfg.gamma
        device = self.device

        obs_0 = obs_seq[:, 0]  # (B, O)
        act_0 = act_seq[:, 0]  # (B, A)

        # ── Precompute discount and survival masks ───────────
        # gamma_pow[i] = γ^i for i=0..n-1
        gamma_pow = torch.tensor(
            [gamma ** i for i in range(n)], device=device
        )  # (n,)

        # not_done_cumprod[i] = Π_{j=0}^{i} (1-done_j)
        # i.e. "still alive at step i"
        not_done = 1.0 - done_seq  # (B, n)
        survival = torch.cumprod(not_done, dim=1)  # (B, n)
        # Prepend 1.0 for step 0 (always alive)
        survival_with_start = torch.cat(
            [torch.ones(B, 1, device=device), survival], dim=1
        )  # (B, n+1)

        # Discounted rewards: disc_rew[b, i] = γ^i * r_i * alive_at_i
        disc_rew = gamma_pow.unsqueeze(0) * rew_seq * survival_with_start[:, :n]  # (B, n)

        # Cumulative reward sums: cum_rew[b, τ-1] = Σ_{i=0}^{τ-1} disc_rew[b,i]
        cum_rew = torch.cumsum(disc_rew, dim=1)  # (B, n)

        # ── Bootstrap values (cached) ─────────────────────────
        v_all = bootstrap_cached  # (B, n) — precomputed by refresh_bootstrap

        # Bootstrap term for each τ: γ^τ * alive_at_τ * V(s_{t+τ})
        gamma_tau = torch.tensor(
            [gamma ** (i + 1) for i in range(n)], device=device
        )  # (n,)
        bootstrap = gamma_tau.unsqueeze(0) * survival[:, :n] * v_all  # (B, n)

        # ── N-step targets: R^τ = cum_rew[:, τ-1] + bootstrap[:, τ-1]
        R_all = cum_rew + bootstrap  # (B, n) — R_all[:, τ-1] is R^τ

        # ── Single actor forward pass for entropy + IS weights ──
        # Both need the actor at future states. One forward pass on
        # all n future obs, then split for entropy vs IS.
        with torch.no_grad():
            all_future_obs = obs_seq[:, 1:n + 1].reshape(B * n, -1)  # (B*n, O)
            all_future_act = act_seq[:, 1:n + 1].reshape(B * n, -1)  # (B*n, A)
            mean_all, log_std_all = self.actor(all_future_obs)
            std_all = log_std_all.exp()

            # Entropy: sample fresh actions, compute log prob
            dist_all = torch.distributions.Normal(mean_all, std_all)
            fresh_pre = dist_all.rsample()
            fresh_act = torch.tanh(fresh_pre)
            fresh_lp = (
                dist_all.log_prob(fresh_pre)
                - torch.log(1 - fresh_act.pow(2) + 1e-6)
            ).sum(-1).reshape(B, n)

            ent_per_state = -fresh_lp  # (B, n)
            cum_ent = torch.cumsum(ent_per_state, dim=1)
            tau_indices = torch.arange(1, n + 1, device=device, dtype=torch.float32)
            avg_ent = cum_ent / tau_indices.unsqueeze(0)
            ent_bonus = self.alpha * avg_ent
            R_all = R_all + ent_bonus * survival[:, :n]

            # IS weights: log prob of stored actions under current policy
            # Only need positions 1..n-1 (not position n)
            mean_inter = mean_all.reshape(B, n, -1)[:, :n - 1].reshape(B * (n - 1), -1)
            std_inter = std_all.reshape(B, n, -1)[:, :n - 1].reshape(B * (n - 1), -1)
            inter_act = all_future_act.reshape(B, n, -1)[:, :n - 1].reshape(B * (n - 1), -1)
            pre_stored = torch.atanh(inter_act.clamp(-0.999, 0.999))
            dist_inter = torch.distributions.Normal(mean_inter, std_inter)
            lp_curr = (
                dist_inter.log_prob(pre_stored)
                - torch.log(1 - inter_act.pow(2) + 1e-6)
            ).sum(-1).reshape(B, n - 1)

            lp_old = logp_old_seq[:, 1:n]  # (B, n-1)
            log_ratios = lp_curr - lp_old  # (B, n-1)

            # Cumulative log ratio: for τ, sum log_ratios[0..τ-2]
            # τ=1: no IS (weight=1). τ=2: log_ratio[0]. τ=3: log_ratio[0]+[1]. etc.
            cum_log_ratios = torch.cumsum(log_ratios, dim=1)  # (B, n-1)
            # Prepend 0 for τ=1 (no correction needed)
            cum_log_ratios = torch.cat(
                [torch.zeros(B, 1, device=device), cum_log_ratios], dim=1
            )  # (B, n)

            raw_w = torch.exp(cum_log_ratios)  # (B, n)
            # Quantile clipping per τ (paper eq 11) — vectorized
            b_vals = torch.quantile(raw_w, q_b, dim=0)  # (n,)
            w_clipped = torch.clamp(raw_w, max=b_vals.unsqueeze(0).expand_as(raw_w))
            max_per_col = w_clipped.max(dim=0).values.clamp(min=1e-8)  # (n,)
            w_clipped = w_clipped / max_per_col.unsqueeze(0)

        # ── Critic loss: average weighted MSE over all τ ──────
        q1, q2 = self.critic(obs_0, act_0)  # (B,) each — one forward pass
        q1 = q1.unsqueeze(1).expand_as(R_all)  # (B, n)
        q2 = q2.unsqueeze(1).expand_as(R_all)  # (B, n)
        R_det = R_all.detach()

        loss_q = (
            (w_clipped * (q1 - R_det) ** 2).mean()
            + (w_clipped * (q2 - R_det) ** 2).mean()
        ) / n

        self.opt_critic.zero_grad(set_to_none=True)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.opt_critic.step()

        # ── Actor update (standard SAC) ───────────────────────
        new_act, logp = self.actor.sample(obs_0)
        q1_new, q2_new = self.critic(obs_0, new_act)
        q_new = torch.min(q1_new, q2_new)
        loss_pi = (self.alpha.detach() * logp.squeeze(-1) - q_new).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.opt_actor.step()

        # ── Entropy tuning ────────────────────────────────────
        loss_alpha = (
            -(self.log_alpha * (logp.detach() + self.target_entropy).squeeze(-1))
        ).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()
        with torch.no_grad():
            self.log_alpha.clamp_(min=self.cfg.min_log_alpha)

        # ── Polyak ────────────────────────────────────────────
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
