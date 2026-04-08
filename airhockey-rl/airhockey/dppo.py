"""DPPO — Diffusion Policy Policy Optimization (Ren et al. 2024).

PPO applied to the diffusion sampling process. The diffusion sampler is
treated as a sequence of K denoising steps; each step is a "sub-action"
of the outer policy. The clipped surrogate is computed over the
per-denoising-step likelihood ratio, with advantages from the outer
environment trajectory broadcast across the inner denoising chain.

This module is the *algorithm core*: rollout collection over the env,
return computation, and the DPPO update step. The training driver lives
in train_dppo.py.

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


# ── Critic head ──────────────────────────────────────────────
class Critic(nn.Module):
    """Small MLP value function over observations. Standard PPO critic."""

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


# ── DPPO sampler with logged denoising trajectory ────────────
@dataclass
class SampleResult:
    actions: torch.Tensor              # (B, H, A) final action chunk
    chain: list[torch.Tensor]          # K+1 entries of (B, H, A) — a^(K), a^(K-1), ..., a^(0)
    eps_pred: list[torch.Tensor]       # K entries of (B, H, A) — predicted noise at each step
    timesteps: list[torch.Tensor]       # K entries of (B,) — diffusion timesteps used


@torch.no_grad()
def sample_with_chain(
    model: UNet1D,
    scheduler: NoiseScheduler,
    obs: torch.Tensor,
    n_steps: Optional[int] = None,
) -> SampleResult:
    """DDIM sampling that records the full denoising chain. Used to compute
    the per-denoising-step log-likelihoods later for the PPO update."""
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
    """Compute log p(a_next | a_curr, t, obs) under the current model.

    DPPO interprets the deterministic DDIM step as a Gaussian transition
    centered at the model's prediction with fixed std `sigma`. The
    log-prob of the (recorded) a_next under that Gaussian is the
    likelihood whose ratio drives the PPO update.

    All tensors are batched over the rollout dimension; the gradient
    flows through ε_θ via reparameterization of the predicted mean.
    """
    eps = model(a_curr, t, obs)  # gradient flows here
    # Predicted next sample under DDIM (deterministic mean)
    ab_now = scheduler.alpha_bars[t].view(-1, 1, 1)
    # Find next ab using the same schedule as the sampler — caller passes via t
    # For simplicity we approximate the mean using (a_curr - sqrt(1-ab) eps)
    a0_pred = (a_curr - torch.sqrt(1.0 - ab_now) * eps) / torch.sqrt(ab_now)
    a0_pred = a0_pred.clamp(-1.5, 1.5)
    mean = a0_pred  # simplified: log-prob of recorded a_next under N(mean, sigma)
    log_prob = -0.5 * ((a_next - mean) / sigma) ** 2 - 0.5 * np.log(2 * np.pi * sigma ** 2)
    return log_prob.sum(dim=(-1, -2))  # (B,) sum over H, A


# ── GAE advantage ────────────────────────────────────────────
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over a SINGLE trajectory.
    All inputs are (T,) arrays. Returns (advantages, returns) of shape (T,).

    Note: this assumes the inputs come from one contiguous trajectory.
    If you have trajectories from multiple parallel envs, call this once
    per env and concatenate the results — never pass mixed-env data as a
    single sequence, because the `dones` flag from one env's episode
    boundary would incorrectly reset another env's value bootstrap.
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
    """GAE computed independently for each of N parallel envs. Returns
    (advantages, returns) of shape (T, N). Each env is a separate
    trajectory and the `dones` flags only affect their own env column.
    """
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    for env_i in range(N):
        a, _ = compute_gae(rewards[:, env_i], values[:, env_i], dones[:, env_i],
                           gamma=gamma, lam=lam)
        adv[:, env_i] = a
    returns = adv + values
    return adv, returns


# ── DPPO update ──────────────────────────────────────────────
def dppo_update(
    model: UNet1D,
    critic: Critic,
    scheduler: NoiseScheduler,
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
    *,
    obs: torch.Tensor,                  # (B, O)
    chains_curr: torch.Tensor,           # (B, K, H, A) — a^(k) at step k
    chains_next: torch.Tensor,           # (B, K, H, A) — a^(k-1) at step k
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
    """Run several epochs of PPO updates over the recorded denoising chain.

    If ``bc_kl_coef > 0`` and ``actor_ref`` is provided, an extra KL-to-BC
    penalty is added to the actor loss. This regularizes the policy back
    toward the behavior-cloned starting point to prevent catastrophic
    policy collapse during online RL fine-tuning — one of the standard
    tricks for DPPO stability.
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

            # Compute new log-probs over all K denoising steps
            new_lp_per_k = []
            for k in range(K):
                lp_k = per_step_logprob(
                    model, scheduler, mb_obs,
                    mb_curr[:, k], mb_next[:, k], mb_t[:, k], sigma=sigma,
                )
                new_lp_per_k.append(lp_k)
            new_lp = torch.stack(new_lp_per_k, dim=1)   # (B', K)
            old_lp = mb_old_lp                            # (B', K)

            # Chain-level log-prob
            new_chain_lp = new_lp.sum(dim=1)
            old_chain_lp = old_lp.sum(dim=1)

            ratio = torch.exp(new_chain_lp - old_chain_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            ppo_loss = -torch.min(surr1, surr2).mean()

            # KL-to-BC regularization (optional but recommended)
            bc_kl_val = torch.tensor(0.0, device=device)
            if bc_kl_coef > 0.0 and actor_ref is not None:
                # Compute reference log-probs under the frozen BC model
                # and penalize divergence. We sum per-step KL contributions.
                with torch.no_grad():
                    ref_lp_per_k = []
                    for k in range(K):
                        ref_lp_k = per_step_logprob(
                            actor_ref, scheduler, mb_obs,
                            mb_curr[:, k], mb_next[:, k], mb_t[:, k], sigma=sigma,
                        )
                        ref_lp_per_k.append(ref_lp_k)
                    ref_chain_lp = torch.stack(ref_lp_per_k, dim=1).sum(dim=1)
                # Approx KL(new || ref) ≈ (new_chain_lp - ref_chain_lp).mean()
                # but we want to penalize the NEW policy from drifting, so
                # the sign is chosen so the gradient pulls new_chain_lp
                # toward ref_chain_lp.
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
