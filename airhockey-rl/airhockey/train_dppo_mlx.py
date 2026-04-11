"""DPPO fine-tune using MLX for the diffusion policy updates.

The env rollout stays in NumPy/PyTorch (env.step is NumPy, opponent
loading is PyTorch). Only the diffusion policy forward/backward and
critic update run in MLX for speed.

Checkpoints save in PyTorch format for ONNX export compatibility.
"""
from __future__ import annotations

import argparse
import copy
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as mnn
import mlx.optimizers as mopt
import numpy as np
import torch
from tqdm import tqdm

from airhockey.dppo_mlx import (
    Critic,
    SampleResult,
    compute_gae,
    dppo_update,
    per_step_logprob,
    sample_with_chain,
)
from airhockey.env import AirHockeyEnv
from airhockey.eval_sac import scripted_attacker, scripted_tracker, with_action_noise
from airhockey.physics import PhysicsConfig
from airhockey.policy_mlx import (
    DiffusionPolicyConfig,
    NoiseScheduler,
    UNet1D,
    convert_mlx_to_torch,
    convert_torch_to_mlx,
)
from airhockey.snapshot_opponent import load_opponent

SCRIPTED_NOISE_STD = 0.05


@dataclass
class DPPOArgs:
    init: str = "ckpt/bc.pt"
    opponent: str = "ckpt_v2/sacn_expert.best.pt"
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


def _make_league_opponent(opponent_ckpt: str, seed: int):
    """Opponent league: 50% SAC expert, 25% attacker, 15% tracker, 10% stationary."""
    expert_fn = load_opponent(opponent_ckpt, device="cpu", deterministic=True)
    noisy_tracker = with_action_noise(scripted_tracker, SCRIPTED_NOISE_STD, seed=seed + 11)
    noisy_attacker = with_action_noise(scripted_attacker, SCRIPTED_NOISE_STD, seed=seed + 22)
    rng = np.random.default_rng(seed + 33)

    def league(obs_top):
        r = rng.random()
        if r < 0.10:
            return np.zeros(2, dtype=np.float32)
        if r < 0.25:
            return noisy_tracker(obs_top)
        if r < 0.50:
            return noisy_attacker(obs_top)
        return expert_fn(obs_top)

    return league


def make_envs(n, seed, opponent_ckpt):
    envs = []
    for i in range(n):
        env = AirHockeyEnv(physics_config=PhysicsConfig(), seed=seed + i)
        env.opponent = _make_league_opponent(opponent_ckpt, seed=seed + i)
        envs.append(env)
    return envs


def reset_all(envs, seed):
    return np.stack([env.reset(seed=seed + i)[0] for i, env in enumerate(envs)]).astype(np.float32)


def step_all(envs, actions):
    next_obs, rewards, dones, infos = [], [], [], []
    for env, a in zip(envs, actions):
        o, r, term, trunc, info = env.step(a)
        if term or trunc:
            o, _ = env.reset()
        next_obs.append(o)
        rewards.append(r)
        dones.append(float(term or trunc))
        infos.append(info)
    return (np.stack(next_obs).astype(np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32), infos)


def main(args: DPPOArgs):
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # Load BC checkpoint into MLX
    torch_ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    cfg = DiffusionPolicyConfig(**torch_ckpt["config"])
    actor = UNet1D(cfg)
    mlx_params = convert_torch_to_mlx(torch_ckpt["model"], cfg)
    actor.load_weights(list(mlx_params.items()))

    # Reference actor (frozen BC) for KL penalty
    actor_ref = UNet1D(cfg)
    actor_ref.load_weights(list(mlx_params.items()))
    actor_ref.freeze()

    critic = Critic(obs_dim=cfg.obs_dim)
    scheduler = NoiseScheduler(cfg)

    optim_actor = mopt.AdamW(learning_rate=args.actor_lr, weight_decay=1e-6)
    optim_critic = mopt.AdamW(learning_rate=args.critic_lr, weight_decay=1e-6)

    envs = make_envs(args.n_envs, args.seed, args.opponent)
    obs = reset_all(envs, args.seed)

    H = cfg.horizon
    K = cfg.n_inference_steps
    A = cfg.act_dim

    chunk_pointer = np.zeros(args.n_envs, dtype=np.int32)
    current_chunks = np.zeros((args.n_envs, H, A), dtype=np.float32)

    # Per-chunk storage
    pending_obs = [None] * args.n_envs
    pending_chain_curr = [None] * args.n_envs
    pending_chain_next = [None] * args.n_envs
    pending_timesteps = [None] * args.n_envs
    pending_old_lp = [None] * args.n_envs
    chunk_reward_acc = np.zeros(args.n_envs, dtype=np.float32)
    chunk_done_acc = np.zeros(args.n_envs, dtype=np.float32)

    buf_obs, buf_cc, buf_cn, buf_ts, buf_olp = [], [], [], [], []
    buf_rewards, buf_dones, buf_values, buf_env_id = [], [], [], []

    return_window = deque(maxlen=50)
    win_window = deque(maxlen=50)
    cur_ep_return = np.zeros(args.n_envs, dtype=np.float32)

    n_total_steps = 0
    metrics = {"kl": 0.0, "clip_frac": 0.0}
    pbar = tqdm(total=args.total_steps, desc="DPPO-MLX")

    while n_total_steps < args.total_steps:
        for _ in range(args.rollout_steps):
            need_sample = chunk_pointer == 0
            if need_sample.any():
                idx = np.where(need_sample)[0]
                sub_obs = mx.array(obs[idx])
                res = sample_with_chain(actor, scheduler, sub_obs, n_steps=K)
                mx.eval(res.actions)
                new_chunks = np.array(res.actions)

                chain_curr = mx.stack(res.chain[:-1], axis=1)
                chain_next = mx.stack(res.chain[1:], axis=1)
                timesteps = mx.stack(res.timesteps, axis=1)

                old_lp_list = []
                for k in range(K):
                    lp_k = per_step_logprob(
                        actor, scheduler, sub_obs,
                        chain_curr[:, k], chain_next[:, k], timesteps[:, k],
                        sigma=args.sigma,
                    )
                    old_lp_list.append(lp_k)
                old_lp = mx.stack(old_lp_list, axis=1)
                mx.eval(old_lp)

                for j, env_i in enumerate(idx):
                    current_chunks[env_i] = new_chunks[j]
                    pending_chain_curr[env_i] = chain_curr[j]
                    pending_chain_next[env_i] = chain_next[j]
                    pending_timesteps[env_i] = timesteps[j]
                    pending_old_lp[env_i] = old_lp[j]
                    pending_obs[env_i] = mx.array(obs[env_i])
                    chunk_reward_acc[env_i] = 0.0
                    chunk_done_acc[env_i] = 0.0

            actions = current_chunks[np.arange(args.n_envs), chunk_pointer]
            next_obs, rewards, dones, infos = step_all(envs, actions)
            chunk_reward_acc += rewards
            chunk_done_acc = np.maximum(chunk_done_acc, dones)
            cur_ep_return += rewards

            for i, d in enumerate(dones):
                if d:
                    return_window.append(float(cur_ep_return[i]))
                    cur_ep_return[i] = 0.0
                    win_window.append(1.0 if infos[i].get("event") == "goal_bot" else 0.0)

            obs = next_obs

            for env_i in range(args.n_envs):
                advanced_ptr = (chunk_pointer[env_i] + 1) % H
                if advanced_ptr == 0 or bool(dones[env_i]):
                    v = float(critic(pending_obs[env_i][None]).item())
                    buf_obs.append(pending_obs[env_i])
                    buf_cc.append(pending_chain_curr[env_i])
                    buf_cn.append(pending_chain_next[env_i])
                    buf_ts.append(pending_timesteps[env_i])
                    buf_olp.append(pending_old_lp[env_i])
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

        # GAE per env
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
            a, r = compute_gae(rewards_np[idxs], values_np[idxs], dones_np[idxs],
                               gamma=args.gamma, lam=args.gae_lam)
            adv_np[idxs] = a
            ret_np[idxs] = r

        obs_batch = mx.stack(buf_obs)
        cc_batch = mx.stack(buf_cc)
        cn_batch = mx.stack(buf_cn)
        ts_batch = mx.stack(buf_ts)
        olp_batch = mx.stack(buf_olp)
        adv_t = mx.array(adv_np)
        ret_t = mx.array(ret_np)

        metrics = dppo_update(
            actor, critic, scheduler,
            optim_actor, optim_critic,
            obs=obs_batch, chains_curr=cc_batch, chains_next=cn_batch,
            timesteps=ts_batch, advantages=adv_t, returns=ret_t,
            old_logprobs=olp_batch, clip_eps=args.clip_eps,
            n_epochs=args.n_epochs, minibatch=args.minibatch,
            sigma=args.sigma, bc_kl_coef=args.bc_kl_coef,
            actor_ref=actor_ref,
        )

        buf_obs.clear(); buf_cc.clear(); buf_cn.clear()
        buf_ts.clear(); buf_olp.clear(); buf_rewards.clear()
        buf_dones.clear(); buf_values.clear(); buf_env_id.clear()

        avg_ret = float(np.mean(return_window)) if return_window else 0.0
        avg_win = float(np.mean(win_window)) if win_window else 0.0
        pbar.set_postfix(
            ret=f"{avg_ret:+.2f}", win=f"{avg_win:.2f}",
            kl=f"{metrics['kl']:+.4f}", clip=f"{metrics['clip_frac']:.2f}",
        )

    pbar.close()

    # Save as PyTorch for ONNX compatibility
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch_state = convert_mlx_to_torch(actor.parameters())
    torch.save(
        {"model": torch_state, "config": cfg.__dict__},
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
