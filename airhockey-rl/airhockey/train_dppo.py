"""Stage 2 — DPPO fine-tune of the BC-pretrained Diffusion Policy.

Online on-policy RL that:
  1. Rolls out the current diffusion policy across N parallel envs
  2. Records the full denoising chain at every action sampling event
  3. Computes GAE advantages from environment rewards
  4. Runs PPO updates over the per-denoising-step likelihood ratio

Trained from a BC checkpoint produced by `train_bc.py`. Saves the
fine-tuned actor + critic to ckpt/dppo.pt.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from airhockey.dppo import (
    Critic,
    compute_gae_per_env,
    dppo_update,
    sample_with_chain,
    per_step_logprob,
)
from airhockey.env import AirHockeyEnv
from airhockey.physics import PhysicsConfig
from airhockey.policy import DiffusionPolicyConfig, NoiseScheduler, UNet1D
from airhockey.snapshot_opponent import load_opponent


@dataclass
class DPPOArgs:
    init: str = "ckpt/bc.pt"
    opponent: str = "ckpt/sac_expert.pt"   # SAC checkpoint for the top-paddle opponent
    out: str = "ckpt/dppo.pt"
    total_steps: int = 2_000_000
    n_envs: int = 16
    rollout_steps: int = 256
    n_epochs: int = 4
    minibatch: int = 256
    actor_lr: float = 3e-5
    critic_lr: float = 3e-4
    gamma: float = 0.99
    gae_lam: float = 0.95
    clip_eps: float = 0.2
    sigma: float = 0.1
    bc_kl_coef: float = 0.05  # KL-to-BC regularization to stop policy collapse
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_envs(n: int, seed: int, opponent_ckpt: str, device: str):
    """Build a list of envs, each with a frozen SAC opponent loaded
    from the given checkpoint. We run envs sequentially (not via
    AsyncVectorEnv) because the env is tiny and not the bottleneck.
    """
    # Load the opponent ONCE — all envs can share the same callable
    # because the SAC actor is stateless and the opponent function
    # only reads the obs it's given (no shared physics handle).
    opponent_fn = load_opponent(opponent_ckpt, device=device, deterministic=True)
    envs = []
    for i in range(n):
        env = AirHockeyEnv(physics_config=PhysicsConfig(), seed=seed + i)
        env.opponent = opponent_fn
        envs.append(env)
    return envs


def reset_all(envs, seed: int):
    obs = []
    for i, env in enumerate(envs):
        o, _ = env.reset(seed=seed + i)
        obs.append(o)
    return np.stack(obs).astype(np.float32)


def step_all(envs, actions: np.ndarray):
    """actions: (N, A) — one normalized action per env."""
    next_obs = []
    rewards = []
    dones = []
    infos = []
    for env, a in zip(envs, actions):
        o, r, term, trunc, info = env.step(a)
        if term or trunc:
            o, _ = env.reset()
        next_obs.append(o)
        rewards.append(r)
        dones.append(float(term or trunc))
        infos.append(info)
    return (
        np.stack(next_obs).astype(np.float32),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=np.float32),
        infos,
    )


def main(args: DPPOArgs):
    device = torch.device(args.device)

    # ── Load BC checkpoint ────────────────────────────────────
    ckpt = torch.load(args.init, map_location=device)
    cfg = DiffusionPolicyConfig(**ckpt["config"])
    actor = UNet1D(cfg).to(device)
    actor.load_state_dict(ckpt["model"])
    # Frozen reference for KL-to-BC regularization
    actor_ref = UNet1D(cfg).to(device)
    actor_ref.load_state_dict(ckpt["model"])
    for p in actor_ref.parameters():
        p.requires_grad = False
    critic = Critic(obs_dim=cfg.obs_dim).to(device)

    scheduler = NoiseScheduler(cfg, device=device)

    optim_actor = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr, weight_decay=1e-6)
    optim_critic = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr, weight_decay=1e-6)

    envs = make_envs(args.n_envs, args.seed, args.opponent, args.device)
    obs = reset_all(envs, args.seed)
    obs_t = torch.from_numpy(obs).to(device)

    H = cfg.horizon
    K = cfg.n_inference_steps
    A = cfg.act_dim

    # Per-env action chunk pointer: how many steps of the current chunk
    # have already been executed. We sample a new chunk when pointer == H.
    chunk_pointer = np.zeros(args.n_envs, dtype=np.int32)
    current_chunks = np.zeros((args.n_envs, H, A), dtype=np.float32)
    # Per-env stored chain + log-probs from the most recent sampling event,
    # used as the rollout-time fixed snapshot for the PPO update.
    pending_chain_curr = [None] * args.n_envs   # each: (K, H, A) tensor
    pending_chain_next = [None] * args.n_envs
    pending_timesteps = [None] * args.n_envs
    pending_old_lp = [None] * args.n_envs
    pending_obs = [None] * args.n_envs

    # Rollout buffer — stores per-env sequences so we can compute GAE
    # correctly (independently per env trajectory). Each list has length T,
    # where each element is either a (N,)-shaped array (scalars) or a
    # (N, H, A) / (N, K, H, A) / (N, K) stacked tensor.
    buf_obs: list[torch.Tensor] = []          # each: (N, O)
    buf_chain_curr: list[torch.Tensor] = []   # each: (N, K, H, A)
    buf_chain_next: list[torch.Tensor] = []
    buf_timesteps: list[torch.Tensor] = []
    buf_old_lp: list[torch.Tensor] = []       # each: (N, K)
    buf_rewards: list[np.ndarray] = []         # each: (N,)
    buf_dones: list[np.ndarray] = []
    buf_values: list[np.ndarray] = []

    # Per-env scratch reward accumulator for the current chunk
    chunk_reward_acc = np.zeros(args.n_envs, dtype=np.float32)
    chunk_done_acc = np.zeros(args.n_envs, dtype=np.float32)

    return_window = deque(maxlen=50)
    win_window = deque(maxlen=50)
    cur_ep_return = np.zeros(args.n_envs, dtype=np.float32)

    n_total_steps = 0
    n_updates = 0
    pbar = tqdm(total=args.total_steps, desc="DPPO")

    while n_total_steps < args.total_steps:
        # ── Roll out one rollout_steps batch ───────────────────
        for _ in range(args.rollout_steps):
            # For each env: if chunk_pointer == 0 we need to sample a new chunk
            need_sample = chunk_pointer == 0
            if need_sample.any():
                idx = np.where(need_sample)[0]
                with torch.no_grad():
                    sub_obs = obs_t[idx]
                    res = sample_with_chain(actor, scheduler, sub_obs, n_steps=K)
                    new_chunks = res.actions.cpu().numpy()
                    # Compute old log-probs along the chain (per-step)
                    chain = res.chain  # K+1 tensors of (b, H, A)
                    chain_curr = torch.stack(chain[:-1], dim=1)  # (b, K, H, A)
                    chain_next = torch.stack(chain[1:], dim=1)
                    timesteps = torch.stack(res.timesteps, dim=1)  # (b, K)
                    old_lp_per_k = []
                    for k in range(K):
                        lp_k = per_step_logprob(
                            actor, scheduler, sub_obs,
                            chain_curr[:, k], chain_next[:, k], timesteps[:, k],
                            sigma=args.sigma,
                        )
                        old_lp_per_k.append(lp_k)
                    old_lp = torch.stack(old_lp_per_k, dim=1)  # (b, K)

                for j, env_i in enumerate(idx):
                    current_chunks[env_i] = new_chunks[j]
                    pending_chain_curr[env_i] = chain_curr[j].detach()
                    pending_chain_next[env_i] = chain_next[j].detach()
                    pending_timesteps[env_i] = timesteps[j].detach()
                    pending_old_lp[env_i] = old_lp[j].detach()
                    pending_obs[env_i] = obs_t[env_i].detach().clone()
                    chunk_reward_acc[env_i] = 0.0
                    chunk_done_acc[env_i] = 0.0

            # Take the next action from each env's chunk
            actions = current_chunks[np.arange(args.n_envs), chunk_pointer]
            next_obs_np, rewards, dones, infos = step_all(envs, actions)
            chunk_reward_acc += rewards
            chunk_done_acc = np.maximum(chunk_done_acc, dones)
            cur_ep_return += rewards

            for i, d in enumerate(dones):
                if d:
                    return_window.append(float(cur_ep_return[i]))
                    cur_ep_return[i] = 0.0
                    # Win = scored a goal in the last episode (top_score increased
                    # for our paddle = bot_score in env terms... see env mapping)
                    # We use a simple proxy: episode return > 0 means we likely
                    # finished with a hit/goal advantage.
                    win_window.append(1.0 if rewards[i] > 0 else 0.0)

            obs = next_obs_np
            obs_t = torch.from_numpy(obs).to(device)
            chunk_pointer = (chunk_pointer + 1) % H

            # When ALL envs finish their chunk synchronously, push a
            # batch to the buffer. This keeps env trajectories aligned
            # per timestep so per-env GAE is well-defined.
            all_done = (chunk_pointer == 0).all()
            if all_done:
                with torch.no_grad():
                    stacked_obs = torch.stack([pending_obs[i] for i in range(args.n_envs)])
                    vals = critic(stacked_obs).cpu().numpy()
                buf_obs.append(stacked_obs)
                buf_chain_curr.append(torch.stack([pending_chain_curr[i] for i in range(args.n_envs)]))
                buf_chain_next.append(torch.stack([pending_chain_next[i] for i in range(args.n_envs)]))
                buf_timesteps.append(torch.stack([pending_timesteps[i] for i in range(args.n_envs)]))
                buf_old_lp.append(torch.stack([pending_old_lp[i] for i in range(args.n_envs)]))
                buf_rewards.append(chunk_reward_acc.copy())
                buf_dones.append(chunk_done_acc.copy())
                buf_values.append(vals)

            n_total_steps += args.n_envs
            pbar.update(args.n_envs)
            if n_total_steps >= args.total_steps:
                break

        if not buf_obs:
            continue

        # ── Compute GAE per-env, then flatten ────────────────
        # Stack along time axis: shapes become (T, N, ...)
        rewards_tn = np.stack(buf_rewards)          # (T, N)
        values_tn = np.stack(buf_values)             # (T, N)
        dones_tn = np.stack(buf_dones)               # (T, N)
        adv_tn, returns_tn = compute_gae_per_env(
            rewards_tn, values_tn, dones_tn,
            gamma=args.gamma, lam=args.gae_lam,
        )
        adv_flat = adv_tn.reshape(-1)                # (T*N,)
        returns_flat = returns_tn.reshape(-1)

        obs_batch = torch.cat(buf_obs, dim=0).to(device)            # (T*N, O)
        chain_curr_batch = torch.cat(buf_chain_curr, dim=0).to(device)
        chain_next_batch = torch.cat(buf_chain_next, dim=0).to(device)
        ts_batch = torch.cat(buf_timesteps, dim=0).to(device)
        old_lp_batch = torch.cat(buf_old_lp, dim=0).to(device)
        adv_t = torch.from_numpy(adv_flat).to(device)
        ret_t = torch.from_numpy(returns_flat).to(device)

        metrics = dppo_update(
            actor, critic, scheduler,
            optim_actor, optim_critic,
            obs=obs_batch,
            chains_curr=chain_curr_batch,
            chains_next=chain_next_batch,
            timesteps=ts_batch,
            advantages=adv_t,
            returns=ret_t,
            old_logprobs=old_lp_batch,
            clip_eps=args.clip_eps,
            n_epochs=args.n_epochs,
            minibatch=args.minibatch,
            sigma=args.sigma,
            bc_kl_coef=args.bc_kl_coef,
            actor_ref=actor_ref,
        )
        n_updates += 1

        # Clear buffer
        buf_obs.clear()
        buf_chain_curr.clear()
        buf_chain_next.clear()
        buf_timesteps.clear()
        buf_old_lp.clear()
        buf_rewards.clear()
        buf_dones.clear()
        buf_values.clear()

        avg_ret = float(np.mean(return_window)) if return_window else 0.0
        avg_win = float(np.mean(win_window)) if win_window else 0.0
        pbar.set_postfix(
            ret=f"{avg_ret:+.2f}",
            win=f"{avg_win:.2f}",
            kl=f"{metrics['kl']:+.4f}",
            clip=f"{metrics['clip_frac']:.2f}",
        )

    pbar.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": actor.state_dict(),
            "critic": critic.state_dict(),
            "config": cfg.__dict__,
        },
        out,
    )
    print(f"Saved DPPO checkpoint to {out}")


def parse_args() -> DPPOArgs:
    p = argparse.ArgumentParser()
    for f in DPPOArgs.__dataclass_fields__.values():
        if f.type is bool:
            p.add_argument(f"--{f.name.replace('_', '-')}", action="store_true")
        else:
            p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    a = p.parse_args()
    return DPPOArgs(**{k: getattr(a, k) for k in DPPOArgs.__dataclass_fields__.keys()})


if __name__ == "__main__":
    main(parse_args())
