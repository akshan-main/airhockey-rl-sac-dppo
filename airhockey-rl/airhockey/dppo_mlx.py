"""DPPO in MLX — faithful to the official irom-princeton/dppo repo.

Key design choices from the reference:
  - No explicit KL penalty. PPO clipping + target_kl early stopping.
  - Log-probs clamped to [-5, 2] before ratio computation.
  - Std floored at min_logprob_std, derived from the same DDIM schedule
    used during sampling (not DDPM t-1).
  - Denoising discount: gamma_denoising^(K-k-1) per step.
  - Noise clamped to [-3, 3] during sampling.
  - Clip coeff exponentially interpolated across denoising steps.
  - Log-prob uses mean() over action dims, not sum().
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from airhockey.policy_mlx import DiffusionPolicyConfig, NoiseScheduler, DiffusionMLP


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def __call__(self, obs: mx.array) -> mx.array:
        x = nn.mish(self.l1(obs))
        x = nn.mish(self.l2(x))
        return self.l3(x).squeeze(-1)


@dataclass
class SampleResult:
    actions: mx.array          # (B, H, A) final
    chain: list[mx.array]      # K+1 entries of (B, H, A)
    timesteps: list[mx.array]  # K entries of (B,)
    stds: list[mx.array]       # K entries of (B, 1, 1)
    ab_nexts: list[mx.array]   # K entries of (B, 1, 1) — alpha_bar at next step


def _ddim_schedule(T: int, n_steps: int):
    """Return the DDIM timestep indices, evenly spaced."""
    return [int(round(x)) for x in np.linspace(T - 1, 0, n_steps + 1)]


def sample_with_chain(
    model: DiffusionMLP,
    scheduler: NoiseScheduler,
    obs: mx.array,
    n_steps: int | None = None,
    min_sampling_std: float = 0.1,
    randn_clip: float = 3.0,
) -> SampleResult:
    cfg = scheduler.cfg
    n_steps = n_steps or cfg.n_inference_steps
    T = cfg.n_train_diffusion_steps
    ts = _ddim_schedule(T, n_steps)
    B = obs.shape[0]
    a = mx.clip(mx.random.normal((B, cfg.horizon, cfg.act_dim)), -randn_clip, randn_clip)
    chain = [a]
    timesteps_out = []
    stds_out = []
    ab_nexts_out = []
    for i in range(n_steps):
        t_now = ts[i]
        t_next = ts[i + 1]
        t_batch = mx.full((B,), t_now, dtype=mx.int32)
        eps = model(a, t_batch, obs)
        ab_now = float(scheduler.alpha_bars[t_now].item())
        ab_next = float(scheduler.alpha_bars[t_next].item()) if t_next >= 0 else 1.0
        a0_pred = (a - math.sqrt(1.0 - ab_now) * eps) / math.sqrt(ab_now)
        a0_pred = mx.clip(a0_pred, -1.5, 1.5)
        mean = math.sqrt(ab_next) * a0_pred + math.sqrt(1.0 - ab_next) * eps
        raw_std = math.sqrt(max(1.0 - ab_next, 0.0))
        std_val = max(raw_std, min_sampling_std)
        noise = mx.clip(mx.random.normal(mean.shape), -randn_clip, randn_clip)
        a = mean + std_val * noise
        chain.append(a)
        timesteps_out.append(t_batch)
        stds_out.append(mx.full((B, 1, 1), std_val))
        ab_nexts_out.append(mx.full((B, 1, 1), ab_next))
    return SampleResult(
        actions=mx.clip(a, -1.0, 1.0),
        chain=chain,
        timesteps=timesteps_out,
        stds=stds_out,
        ab_nexts=ab_nexts_out,
    )


def per_step_logprob(
    model: DiffusionMLP,
    scheduler: NoiseScheduler,
    obs: mx.array,
    a_curr: mx.array,
    a_next: mx.array,
    t_now: mx.array,
    ab_next: mx.array,
    std: mx.array,
    min_logprob_std: float = 0.1,
) -> mx.array:
    """Gaussian log p(a_next | a_curr, t, obs).

    Recomputes the EXACT same DDIM mean used during sampling:
      mean = sqrt(ab_next) * a0_pred + sqrt(1 - ab_next) * eps
    Then evaluates N(mean, std).log_prob(a_next).

    ab_next and std are passed from the stored chain to guarantee
    the log-prob matches the sampling distribution exactly.
    """
    eps = model(a_curr, t_now, obs)
    ab_now = scheduler.alpha_bars[t_now].reshape(-1, 1, 1)
    a0_pred = (a_curr - mx.sqrt(1.0 - ab_now) * eps) / mx.sqrt(ab_now)
    a0_pred = mx.clip(a0_pred, -1.5, 1.5)
    # Exact same mean as sample_with_chain computed
    mean = mx.sqrt(ab_next) * a0_pred + mx.sqrt(1.0 - ab_next) * eps
    # Floor std for numerical stability
    std_floored = mx.maximum(std, mx.array(min_logprob_std))
    log_prob = -0.5 * ((a_next - mean) / std_floored) ** 2 - 0.5 * mx.log(2 * np.pi * std_floored ** 2)
    return mx.mean(log_prob, axis=(-1, -2))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
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


def dppo_update(
    model: DiffusionMLP,
    critic: Critic,
    scheduler: NoiseScheduler,
    optim_actor,
    optim_critic,
    *,
    obs: mx.array,
    chains_curr: mx.array,        # (B, K, H, A)
    chains_next: mx.array,        # (B, K, H, A)
    timesteps: mx.array,          # (B, K)
    chain_stds: mx.array,         # (B, K, 1, 1)
    chain_ab_nexts: mx.array,     # (B, K, 1, 1)
    advantages: mx.array,         # (B,)
    returns: mx.array,            # (B,)
    old_logprobs: mx.array,       # (B, K)
    clip_eps: float = 0.01,
    clip_eps_base: float = 0.01,
    clip_eps_rate: float = 3.0,
    vf_coef: float = 0.5,
    n_epochs: int = 5,
    minibatch: int = 256,
    min_logprob_std: float = 0.1,
    gamma_denoising: float = 0.99,
    target_kl: float = 1.0,
) -> dict[str, float]:
    B, K = chains_curr.shape[:2]
    advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)

    denoise_discount = mx.array(
        [gamma_denoising ** (K - k - 1) for k in range(K)]
    )

    clip_coeffs = []
    for k in range(K):
        t_norm = k / max(K - 1, 1)
        c = clip_eps_base + (clip_eps - clip_eps_base) * (
            math.exp(clip_eps_rate * t_norm) - 1
        ) / (math.exp(clip_eps_rate) - 1)
        clip_coeffs.append(c)
    clip_coeffs = mx.array(clip_coeffs)

    metrics = {"actor_loss": 0.0, "critic_loss": 0.0,
               "kl": 0.0, "clip_frac": 0.0}
    n_updates = 0
    early_stop = False

    def actor_loss_fn(model, mb_obs, mb_curr, mb_next, mb_t, mb_stds, mb_ab_nexts, mb_adv, mb_old_lp):
        new_lp_list = []
        for k in range(K):
            lp_k = per_step_logprob(
                model, scheduler, mb_obs,
                mb_curr[:, k], mb_next[:, k], mb_t[:, k],
                ab_next=mb_ab_nexts[:, k],
                std=mb_stds[:, k],
                min_logprob_std=min_logprob_std,
            )
            new_lp_list.append(lp_k)
        new_lp = mx.stack(new_lp_list, axis=1)

        new_lp_clamped = mx.clip(new_lp, -5.0, 2.0)
        old_lp_clamped = mx.clip(mb_old_lp, -5.0, 2.0)

        log_ratio = new_lp_clamped - old_lp_clamped
        ratio = mx.exp(log_ratio)

        weighted_adv = mb_adv[:, None] * denoise_discount[None, :]

        surr1 = ratio * weighted_adv
        surr2 = mx.clip(ratio, 1 - clip_coeffs[None, :], 1 + clip_coeffs[None, :]) * weighted_adv
        ppo_loss = -mx.mean(mx.minimum(surr1, surr2))

        return ppo_loss, (ratio, new_lp_clamped, old_lp_clamped)

    def critic_loss_fn(critic, mb_obs, mb_ret):
        value = critic(mb_obs)
        return mx.mean((value - mb_ret) ** 2)

    actor_loss_grad = nn.value_and_grad(model, actor_loss_fn)
    critic_loss_grad = nn.value_and_grad(critic, critic_loss_fn)

    for epoch in range(n_epochs):
        if early_stop:
            break
        perm = np.random.permutation(B)
        for start in range(0, B, minibatch):
            mb = mx.array(perm[start:start + minibatch])
            mb_obs = obs[mb]
            mb_curr = chains_curr[mb]
            mb_next = chains_next[mb]
            mb_t = timesteps[mb]
            mb_stds = chain_stds[mb]
            mb_ab_nexts = chain_ab_nexts[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]
            mb_old_lp = old_logprobs[mb]

            (ppo_loss, (ratio, new_clp, old_clp)), actor_grads = actor_loss_grad(
                model, mb_obs, mb_curr, mb_next, mb_t, mb_stds, mb_ab_nexts, mb_adv, mb_old_lp
            )
            optim_actor.update(model, actor_grads)

            c_loss, critic_grads = critic_loss_grad(critic, mb_obs, mb_ret)
            optim_critic.update(critic, critic_grads)

            mx.eval(model.parameters(), critic.parameters(),
                    optim_actor.state, optim_critic.state)

            log_ratio = new_clp - old_clp
            approx_kl = float(mx.mean((ratio - 1) - log_ratio).item())
            clip_frac = float(mx.mean(
                (mx.abs(ratio - 1) > clip_coeffs[None, :]).astype(mx.float32)
            ).item())

            metrics["actor_loss"] += float(ppo_loss.item())
            metrics["critic_loss"] += float(c_loss.item())
            metrics["kl"] += approx_kl
            metrics["clip_frac"] += clip_frac
            n_updates += 1

            if approx_kl > target_kl:
                early_stop = True
                break

    if n_updates > 0:
        for k in metrics:
            metrics[k] /= n_updates
    return metrics
