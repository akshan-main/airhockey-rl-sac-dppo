"""DPPO components in MLX

Critic, per_step_logprob, compute_gae, and dppo_update ported to MLX.
sample_with_chain stays in the training script since it's tightly
coupled with the env rollout loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from airhockey.policy_mlx import DiffusionPolicyConfig, NoiseScheduler, UNet1D


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
    actions: mx.array
    chain: list[mx.array]
    eps_pred: list[mx.array]
    timesteps: list[mx.array]


def sample_with_chain(
    model: UNet1D,
    scheduler: NoiseScheduler,
    obs: mx.array,
    n_steps: int | None = None,
) -> SampleResult:
    cfg = scheduler.cfg
    n_steps = n_steps or cfg.n_inference_steps
    T = cfg.n_train_diffusion_steps
    ts = mx.linspace(T - 1, 0, n_steps + 1).astype(mx.int32)
    B = obs.shape[0]
    a = mx.random.normal((B, cfg.horizon, cfg.act_dim))
    chain = [a]
    eps_pred_list = []
    timesteps = []
    for i in range(n_steps):
        t_now = int(ts[i].item())
        t_next = int(ts[i + 1].item())
        t_batch = mx.full((B,), t_now, dtype=mx.int32)
        eps = model(a, t_batch, obs)
        ab_now = scheduler.alpha_bars[t_now]
        ab_next = scheduler.alpha_bars[t_next] if t_next >= 0 else mx.array(1.0)
        a0_pred = (a - mx.sqrt(1.0 - ab_now) * eps) / mx.sqrt(ab_now)
        a0_pred = mx.clip(a0_pred, -1.5, 1.5)
        a = mx.sqrt(ab_next) * a0_pred + mx.sqrt(1.0 - ab_next) * eps
        chain.append(a)
        eps_pred_list.append(eps)
        timesteps.append(t_batch)
    return SampleResult(
        actions=mx.clip(a, -1.0, 1.0),
        chain=chain,
        eps_pred=eps_pred_list,
        timesteps=timesteps,
    )


def per_step_logprob(
    model: UNet1D,
    scheduler: NoiseScheduler,
    obs: mx.array,
    a_curr: mx.array,
    a_next: mx.array,
    t: mx.array,
    sigma: float = 0.1,
) -> mx.array:
    eps = model(a_curr, t, obs)
    ab_now = scheduler.alpha_bars[t].reshape(-1, 1, 1)
    a0_pred = (a_curr - mx.sqrt(1.0 - ab_now) * eps) / mx.sqrt(ab_now)
    a0_pred = mx.clip(a0_pred, -1.5, 1.5)
    log_prob = -0.5 * ((a_next - a0_pred) / sigma) ** 2 - 0.5 * np.log(2 * np.pi * sigma ** 2)
    return mx.sum(log_prob, axis=(-1, -2))


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
    model: UNet1D,
    critic: Critic,
    scheduler: NoiseScheduler,
    optim_actor,
    optim_critic,
    *,
    obs: mx.array,
    chains_curr: mx.array,
    chains_next: mx.array,
    timesteps: mx.array,
    advantages: mx.array,
    returns: mx.array,
    old_logprobs: mx.array,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    n_epochs: int = 4,
    minibatch: int = 256,
    sigma: float = 0.1,
    bc_kl_coef: float = 0.0,
    actor_ref=None,
) -> dict[str, float]:
    B, K = chains_curr.shape[:2]
    advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-8)

    metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "bc_kl": 0.0,
               "kl": 0.0, "clip_frac": 0.0}
    n_updates = 0

    def actor_loss_fn(model, mb_obs, mb_curr, mb_next, mb_t, mb_adv, mb_old_lp):
        new_lp_list = []
        for k in range(K):
            lp_k = per_step_logprob(
                model, scheduler, mb_obs,
                mb_curr[:, k], mb_next[:, k], mb_t[:, k], sigma=sigma,
            )
            new_lp_list.append(lp_k)
        new_lp = mx.stack(new_lp_list, axis=1)
        new_chain_lp = mx.sum(new_lp, axis=1)
        old_chain_lp = mx.sum(mb_old_lp, axis=1)
        ratio = mx.exp(new_chain_lp - old_chain_lp)
        surr1 = ratio * mb_adv
        surr2 = mx.clip(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
        ppo_loss = -mx.mean(mx.minimum(surr1, surr2))

        bc_kl = mx.array(0.0)
        if bc_kl_coef > 0.0 and actor_ref is not None:
            mx.stop_gradient(actor_ref)
            ref_lp_list = []
            for k in range(K):
                ref_lp_k = per_step_logprob(
                    actor_ref, scheduler, mb_obs,
                    mb_curr[:, k], mb_next[:, k], mb_t[:, k], sigma=sigma,
                )
                ref_lp_list.append(ref_lp_k)
            ref_chain_lp = mx.sum(mx.stack(ref_lp_list, axis=1), axis=1)
            bc_kl = mx.mean(new_chain_lp - ref_chain_lp)

        return ppo_loss + bc_kl_coef * bc_kl, (ppo_loss, bc_kl, ratio, new_chain_lp, old_chain_lp)

    def critic_loss_fn(critic, mb_obs, mb_ret):
        value = critic(mb_obs)
        return mx.mean((value - mb_ret) ** 2)

    actor_loss_grad = nn.value_and_grad(model, actor_loss_fn)
    critic_loss_grad = nn.value_and_grad(critic, critic_loss_fn)

    for _ in range(n_epochs):
        perm = np.random.permutation(B)
        for start in range(0, B, minibatch):
            mb = perm[start:start + minibatch]
            mb_obs = obs[mb]
            mb_curr = chains_curr[mb]
            mb_next = chains_next[mb]
            mb_t = timesteps[mb]
            mb_adv = advantages[mb]
            mb_ret = returns[mb]
            mb_old_lp = old_logprobs[mb]

            (total_loss, (ppo_loss, bc_kl, ratio, new_clp, old_clp)), actor_grads = actor_loss_grad(
                model, mb_obs, mb_curr, mb_next, mb_t, mb_adv, mb_old_lp
            )
            optim_actor.update(model, actor_grads)

            (c_loss,), critic_grads = critic_loss_grad(critic, mb_obs, mb_ret)
            optim_critic.update(critic, critic_grads)

            mx.eval(model.parameters(), critic.parameters(),
                    optim_actor.state, optim_critic.state)

            approx_kl = float(mx.mean(old_clp - new_clp).item())
            clip_frac = float(mx.mean((mx.abs(ratio - 1) > clip_eps).astype(mx.float32)).item())
            metrics["actor_loss"] += float(ppo_loss.item())
            metrics["critic_loss"] += float(c_loss.item())
            metrics["bc_kl"] += float(bc_kl.item())
            metrics["kl"] += approx_kl
            metrics["clip_frac"] += clip_frac
            n_updates += 1

    if n_updates > 0:
        for k in metrics:
            metrics[k] /= n_updates
    return metrics
