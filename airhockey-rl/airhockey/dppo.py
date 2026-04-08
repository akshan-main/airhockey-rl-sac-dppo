"""DPPO — Diffusion Policy Policy Optimization (Ren et al. 2024).

PPO applied to the DDIM sampling chain of a Diffusion Policy. The
K denoising steps are treated as sub-actions of the outer policy; the
clipped surrogate is computed over per-denoising-step likelihood
ratios. Advantages come from the outer-env trajectory.

The training driver lives in train_dppo.py. This module exposes:
  - Critic: value MLP
  - sample_with_chain: DDIM sampling that records (a^k, a^(k-1), t)
  - per_step_logprob: Gaussian log p(a^(k-1) | a^(k), t, obs)
  - compute_gae / compute_gae_per_env: GAE over 1D / (T, N) trajectories
  - dppo_update: PPO step, optionally with a KL-to-BC penalty

Reference: https://arxiv.org/abs/2409.00588
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from airhockey.policy import DiffusionPolicyConfig, NoiseScheduler, UNet1D


class Critic(nn.Module):
    """MLP value function V(obs) -> scalar."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


@dataclass
class SampleResult:
    actions: torch.Tensor              # (B, H, A) final action chunk
    chain: list[torch.Tensor]          # K+1 entries of (B, H, A): a^K, a^(K-1), ..., a^0
    eps_pred: list[torch.Tensor]       # K entries of (B, H, A): predicted noise per step
    timesteps: list[torch.Tensor]      # K entries of (B,): diffusion timesteps used


@torch.no_grad()
def sample_with_chain(
    model: UNet1D,
    scheduler: NoiseScheduler,
    obs: torch.Tensor,
    n_steps: Optional[int] = None,
) -> SampleResult:
    """DDIM sampling that records the full denoising chain for later
    per-step log-prob computation."""
    cfg = scheduler.cfg
    device = obs.device
    n_steps = n_steps or cfg.n_inference_steps
    T = cfg.n_train_diffusion_steps
    ts = torch.linspace(T - 1, 0, n_steps + 1, dtype=torch.long, device=device)
    B = obs.shape[0]
    a = torch.randn(B, cfg.horizon, cfg.act_dim, device=device)
    chain = [a.clone()]
    eps_pred_list: list[torch.Tensor] = []
    timesteps: list[torch.Tensor] = []
    for i in range(n_steps):
        t_now = ts[i]
        t_next = ts[i + 1]
        t_batch = t_now.expand(B)
        eps = model(a, t_batch, obs)
        ab_now = scheduler.alpha_bars[t_now]
        ab_next = scheduler.alpha_bars[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
        a0_pred = (a - torch.sqrt(1.0 - ab_now) * eps) / torch.sqrt(ab_now)
        a0_pred = a0_pred.clamp(-1.5, 1.5)
        a = torch.sqrt(ab_next) * a0_pred + torch.sqrt(1.0 - ab_next) * eps
        chain.append(a.clone())
        eps_pred_list.append(eps)
        timesteps.append(t_batch)
    return SampleResult(
        actions=a.clamp(-1.0, 1.0),
        chain=chain,
        eps_pred=eps_pred_list,
        timesteps=timesteps,
    )


def per_step_logprob(
    model: UNet1D,
    scheduler: NoiseScheduler,
    obs: torch.Tensor,
    a_curr: torch.Tensor,
    a_next: torch.Tensor,
    t: torch.Tensor,
    sigma: float = 0.1,
) -> torch.Tensor:
    """log p(a_next | a_curr, t, obs) treating the DDIM step as a
    Gaussian N(mean=a0_pred, std=sigma). Gradient flows through eps_theta.
    Returns (B,) after summing over horizon and action dims.
    """
    eps = model(a_curr, t, obs)
    ab_now = scheduler.alpha_bars[t].view(-1, 1, 1)
    a0_pred = (a_curr - torch.sqrt(1.0 - ab_now) * eps) / torch.sqrt(ab_now)
    a0_pred = a0_pred.clamp(-1.5, 1.5)
    log_prob = -0.5 * ((a_next - a0_pred) / sigma) ** 2 - 0.5 * np.log(2 * np.pi * sigma ** 2)
    return log_prob.sum(dim=(-1, -2))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE over a single contiguous trajectory. Inputs are (T,). Returns
    (advantages, returns), both (T,). For multi-env rollouts use
    compute_gae_per_env — passing mixed-env data through this function
    will leak episode boundaries across envs.
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    returns = adv + values[:T]
    return adv, returns


def compute_gae_per_env(
    rewards: np.ndarray,       # (T, N)
    values: np.ndarray,         # (T, N)
    dones: np.ndarray,          # (T, N)
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE computed per env. Inputs are (T, N). Returns (T, N)."""
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    for env_i in range(N):
        a, _ = compute_gae(rewards[:, env_i], values[:, env_i], dones[:, env_i],
                           gamma=gamma, lam=lam)
        adv[:, env_i] = a
    returns = adv + values
    return adv, returns


def dppo_update(
    model: UNet1D,
    critic: Critic,
    scheduler: NoiseScheduler,
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
    *,
    obs: torch.Tensor,                  # (B, O)
    chains_curr: torch.Tensor,           # (B, K, H, A): a^k per step
    chains_next: torch.Tensor,           # (B, K, H, A): a^(k-1) per step
    timesteps: torch.Tensor,             # (B, K)
    advantages: torch.Tensor,            # (B,)
    returns: torch.Tensor,               # (B,)
    old_logprobs: torch.Tensor,          # (B, K)
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    n_epochs: int = 4,
    minibatch: int = 256,
    sigma: float = 0.1,
    bc_kl_coef: float = 0.0,
    actor_ref: Optional[UNet1D] = None,
) -> dict[str, float]:
    """PPO epochs over the recorded denoising chain.

    If bc_kl_coef > 0 and actor_ref is provided, adds a KL-to-reference
    penalty term of the form `bc_kl_coef * (new_lp - ref_lp).mean()`
    to the actor loss.
    """
    B, K = chains_curr.shape[:2]
    device = obs.device
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    metrics = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "bc_kl": 0.0,
        "kl": 0.0,
        "clip_frac": 0.0,
    }
    n_updates = 0

    for _ in range(n_epochs):
        idx = torch.randperm(B, device=device)
        for start in range(0, B, minibatch):
            mb = idx[start : start + minibatch]
            mb_obs = obs[mb]
            mb_curr = chains_curr[mb]
            mb_next = chains_next[mb]
            mb_t = timesteps[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]
            mb_old_lp = old_logprobs[mb]

            new_lp_per_k = []
            for k in range(K):
                lp_k = per_step_logprob(
                    model, scheduler, mb_obs,
                    mb_curr[:, k], mb_next[:, k], mb_t[:, k], sigma=sigma,
                )
                new_lp_per_k.append(lp_k)
            new_lp = torch.stack(new_lp_per_k, dim=1)   # (B', K)
            old_lp = mb_old_lp

            new_chain_lp = new_lp.sum(dim=1)
            old_chain_lp = old_lp.sum(dim=1)

            ratio = torch.exp(new_chain_lp - old_chain_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            ppo_loss = -torch.min(surr1, surr2).mean()

            bc_kl_val = torch.tensor(0.0, device=device)
            if bc_kl_coef > 0.0 and actor_ref is not None:
                with torch.no_grad():
                    ref_lp_per_k = []
                    for k in range(K):
                        ref_lp_k = per_step_logprob(
                            actor_ref, scheduler, mb_obs,
                            mb_curr[:, k], mb_next[:, k], mb_t[:, k], sigma=sigma,
                        )
                        ref_lp_per_k.append(ref_lp_k)
                    ref_chain_lp = torch.stack(ref_lp_per_k, dim=1).sum(dim=1)
                bc_kl_val = (new_chain_lp - ref_chain_lp).mean()

            actor_loss = ppo_loss + bc_kl_coef * bc_kl_val

            value = critic(mb_obs)
            critic_loss = F.mse_loss(value, mb_ret)

            optim_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_actor.step()

            optim_critic.zero_grad(set_to_none=True)
            (vf_coef * critic_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optim_critic.step()

            with torch.no_grad():
                approx_kl = (old_chain_lp - new_chain_lp).mean().item()
                clip_frac = ((ratio - 1).abs() > clip_eps).float().mean().item()
            metrics["actor_loss"] += float(ppo_loss.item())
            metrics["critic_loss"] += float(critic_loss.item())
            metrics["bc_kl"] += float(bc_kl_val.item()) if isinstance(bc_kl_val, torch.Tensor) else 0.0
            metrics["kl"] += approx_kl
            metrics["clip_frac"] += clip_frac
            n_updates += 1

    if n_updates > 0:
        for k in metrics:
            metrics[k] /= n_updates
    return metrics
