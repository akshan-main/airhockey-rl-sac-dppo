"""Collect demonstration trajectories from a trained SAC expert.

Stage 2a of the pipeline. After training a SAC agent in Stage 1, we use
it as the "expert" and record its actions over many self-play episodes.
The resulting (obs, action) dataset is then used to train the Diffusion
Policy via behavior cloning.

Unlike a hand-engineered heuristic, the expert here is a real RL agent,
so the demonstrations are of genuine competent play — not of whatever
rules the author happened to encode.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from airhockey.env import AirHockeyEnv
from airhockey.physics import PhysicsConfig
from airhockey.sac import SACAgent, SACConfig
from airhockey.snapshot_opponent import load_opponent


def collect(
    expert_ckpt: str,
    episodes: int,
    out_path: str,
    max_steps: int = 1500,
    seed: int = 0,
    device: str = "cpu",
) -> None:
    device_t = torch.device(device)
    # Load the SAC expert as the bottom-paddle policy
    ckpt = torch.load(expert_ckpt, map_location=device_t, weights_only=False)
    cfg = SACConfig(**ckpt["config"])
    agent = SACAgent(cfg, device=device_t)
    agent.load_state_dict(ckpt)
    agent.actor.eval()

    env = AirHockeyEnv(physics_config=PhysicsConfig(), seed=seed, max_episode_steps=max_steps)
    # Top paddle opponent: a second copy of the same SAC expert. This is
    # self-play at "final skill" level — both sides are the same agent.
    env.opponent = load_opponent(expert_ckpt, device=device, deterministic=True)

    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    ep_buf: list[int] = []
    t_buf: list[int] = []

    for ep in tqdm(range(episodes), desc="Collecting"):
        obs, _ = env.reset(seed=seed + ep)
        for t in range(max_steps):
            # Deterministic expert action
            action = agent.act(obs, deterministic=True).astype(np.float32)
            obs_buf.append(obs.astype(np.float32))
            act_buf.append(action)
            ep_buf.append(ep)
            t_buf.append(t)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=np.stack(obs_buf),
        act=np.stack(act_buf),
        episode=np.array(ep_buf, dtype=np.int32),
        timestep=np.array(t_buf, dtype=np.int32),
    )
    print(f"Saved {len(obs_buf):,} transitions across {episodes} episodes to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", type=str, required=True, help="Path to trained SAC checkpoint")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--out", type=str, default="data/demos.npz")
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    collect(args.expert, args.episodes, args.out, args.max_steps, args.seed, args.device)


if __name__ == "__main__":
    main()
