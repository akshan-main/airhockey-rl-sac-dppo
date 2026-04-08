"""Stage 3 — DPPO fine-tune on top of a BC-pretrained Diffusion Policy.

Loop per rollout batch:
  1. Sample action chunks for all envs (recording the full DDIM chain)
  2. Execute each chunk step-by-step across parallel envs
  3. On chunk completion OR episode reset, flush the chunk into a buffer
     keyed by env_id
  4. Compute per-env GAE, then run PPO epochs with optional KL-to-BC
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from airhockey.dppo import (
    Critic,
    compute_gae,
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
    opponent: str = "ckpt/sac_expert.pt"
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
    bc_kl_coef: float = 0.05
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_envs(n: int, seed: int, opponent_ckpt: str, device: str):
    """Build `n` envs sharing a single opponent callable."""
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
    """Step every env one tick. On termination the env is reset and the
    returned next_obs is the post-reset observation."""
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

    ckpt = torch.load(args.init, map_location=device)
    cfg = DiffusionPolicyConfig(**ckpt["config"])
    actor = UNet1D(cfg).to(device)
    actor.load_state_dict(ckpt["model"])
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

    chunk_pointer = np.zeros(args.n_envs, dtype=np.int32)
    current_chunks = np.zeros((args.n_envs, H, A), dtype=np.float32)
    pending_obs: list[Optional[torch.Tensor]] = [None] * args.n_envs
    pending_chain_curr: list[Optional[torch.Tensor]] = [None] * args.n_envs
    pending_chain_next: list[Optional[torch.Tensor]] = [None] * args.n_envs
    pending_timesteps: list[Optional[torch.Tensor]] = [None] * args.n_envs
    pending_old_lp: list[Optional[torch.Tensor]] = [None] * args.n_envs
    chunk_reward_acc = np.zeros(args.n_envs, dtype=np.float32)
    chunk_done_acc = np.zeros(args.n_envs, dtype=np.float32)

    # One entry per completed chunk per env; env_id lets us compute GAE
    # on each env's own trajectory before concatenating for the PPO step.
    buf_obs: list[torch.Tensor] = []
    buf_chain_curr: list[torch.Tensor] = []
    buf_chain_next: list[torch.Tensor] = []
    buf_timesteps: list[torch.Tensor] = []
    buf_old_lp: list[torch.Tensor] = []
    buf_rewards: list[float] = []
    buf_dones: list[float] = []
    buf_values: list[float] = []
    buf_env_id: list[int] = []

    return_window = deque(maxlen=50)
    win_window = deque(maxlen=50)
    cur_ep_return = np.zeros(args.n_envs, dtype=np.float32)

    n_total_steps = 0
    n_updates = 0
    metrics: dict = {"kl": 0.0, "clip_frac": 0.0}
    pbar = tqdm(total=args.total_steps, desc="DPPO")

    while n_total_steps < args.total_steps:
        # ── Roll out one rollout_steps batch ───────────────────
        for _ in range(args.rollout_steps):
            # pointer == 0 means: start of rollout, end of previous chunk,
            # or env just reset. In all three cases sample a fresh chunk.
            need_sample = chunk_pointer == 0
            if need_sample.any():
                idx = np.where(need_sample)[0]
                with torch.no_grad():
                    sub_obs = obs_t[idx]
                    res = sample_with_chain(actor, scheduler, sub_obs, n_steps=K)
                    new_chunks = res.actions.cpu().numpy()
                    chain_curr = torch.stack(res.chain[:-1], dim=1)  # (b, K, H, A)
                    chain_next = torch.stack(res.chain[1:], dim=1)
                    timesteps = torch.stack(res.timesteps, dim=1)    # (b, K)
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

            actions = current_chunks[np.arange(args.n_envs), chunk_pointer]
            next_obs_np, rewards, dones, infos = step_all(envs, actions)
            chunk_reward_acc += rewards
            chunk_done_acc = np.maximum(chunk_done_acc, dones)
            cur_ep_return += rewards

            for i, d in enumerate(dones):
                if d:
                    return_window.append(float(cur_ep_return[i]))
                    cur_ep_return[i] = 0.0
                    ev = infos[i].get("event", "")
                    win_window.append(1.0 if ev == "goal_bot" else 0.0)

            obs = next_obs_np
            obs_t = torch.from_numpy(obs).to(device)

            # Flush chunks that just completed OR whose env just reset,
            # and set their pointer back to 0 to trigger a resample.
            for env_i in range(args.n_envs):
                advanced_ptr = (chunk_pointer[env_i] + 1) % H
                if advanced_ptr == 0 or bool(dones[env_i]):
                    with torch.no_grad():
                        v = critic(pending_obs[env_i].unsqueeze(0)).item()
                    buf_obs.append(pending_obs[env_i])
                    buf_chain_curr.append(pending_chain_curr[env_i])
                    buf_chain_next.append(pending_chain_next[env_i])
                    buf_timesteps.append(pending_timesteps[env_i])
                    buf_old_lp.append(pending_old_lp[env_i])
                    buf_rewards.append(float(chunk_reward_acc[env_i]))
                    buf_dones.append(float(chunk_done_acc[env_i]))
                    buf_values.append(v)
                    buf_env_id.append(env_i)
                    chunk_pointer[env_i] = 0
                else:
                    chunk_pointer[env_i] = advanced_ptr

            n_total_steps += args.n_envs
            pbar.update(args.n_envs)
            if n_total_steps >= args.total_steps:
                break

        if not buf_obs:
            continue

        # GAE per env on each env's own chunk sequence, then concat.
        env_ids = np.array(buf_env_id, dtype=np.int32)
        rewards_np = np.array(buf_rewards, dtype=np.float32)
        values_np = np.array(buf_values, dtype=np.float32)
        dones_np = np.array(buf_dones, dtype=np.float32)
        adv_np = np.zeros_like(rewards_np)
        ret_np = np.zeros_like(rewards_np)
        for env_i in range(args.n_envs):
            mask = env_ids == env_i
            if not mask.any():
                continue
            idxs = np.where(mask)[0]
            a, r = compute_gae(
                rewards_np[idxs], values_np[idxs], dones_np[idxs],
                gamma=args.gamma, lam=args.gae_lam,
            )
            adv_np[idxs] = a
            ret_np[idxs] = r

        obs_batch = torch.stack(buf_obs).to(device)
        chain_curr_batch = torch.stack(buf_chain_curr).to(device)
        chain_next_batch = torch.stack(buf_chain_next).to(device)
        ts_batch = torch.stack(buf_timesteps).to(device)
        old_lp_batch = torch.stack(buf_old_lp).to(device)
        adv_t = torch.from_numpy(adv_np).to(device)
        ret_t = torch.from_numpy(ret_np).to(device)

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

        buf_obs.clear()
        buf_chain_curr.clear()
        buf_chain_next.clear()
        buf_timesteps.clear()
        buf_old_lp.clear()
        buf_rewards.clear()
        buf_dones.clear()
        buf_values.clear()
        buf_env_id.clear()

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
