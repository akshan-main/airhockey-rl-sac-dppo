"""Head-to-head evaluation: trained Diffusion Policy vs a configurable opponent.

Loads a Diffusion Policy checkpoint (from BC or DPPO), runs N episodes
against an opponent loaded from a separate checkpoint (typically the SAC
expert), reports per-episode return, win rate, and average episode length.

If no opponent checkpoint is provided, the top paddle sits still — useful
as a pure "can the agent score at all" sanity check.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from airhockey.env import AirHockeyEnv
from airhockey.physics import PhysicsConfig
from airhockey.policy import DiffusionPolicyConfig, NoiseScheduler, UNet1D
from airhockey.snapshot_opponent import load_opponent


@torch.no_grad()
def evaluate(
    ckpt_path: str,
    episodes: int = 200,
    n_inference_steps: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
    opponent_ckpt: str | None = None,
) -> dict[str, float]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = DiffusionPolicyConfig(**ckpt["config"])
    if n_inference_steps is not None:
        cfg.n_inference_steps = n_inference_steps
    model = UNet1D(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    scheduler = NoiseScheduler(cfg, device=device)

    env = AirHockeyEnv(physics_config=PhysicsConfig(), seed=seed)
    if opponent_ckpt is not None:
        env.opponent = load_opponent(opponent_ckpt, device=device, deterministic=True)
    else:
        # No opponent: top paddle sits still. Useful as a pure
        # "can the agent score at all" sanity check.
        env.opponent = None

    returns: list[float] = []
    lengths: list[int] = []
    wins: list[int] = []

    for ep in tqdm(range(episodes), desc="Eval"):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        steps = 0
        chunk = None
        chunk_pointer = 0
        last_event = ""
        while True:
            if chunk is None or chunk_pointer == cfg.horizon:
                obs_t = torch.from_numpy(obs[None]).to(device)
                chunk = scheduler.ddim_sample(model, obs_t, n_steps=cfg.n_inference_steps)[0].cpu().numpy()
                chunk_pointer = 0
            action = chunk[chunk_pointer]
            chunk_pointer += 1
            obs, reward, term, trunc, info = env.step(action)
            last_event = info.get("event", "") or last_event
            ep_return += reward
            steps += 1
            if term or trunc:
                returns.append(ep_return)
                lengths.append(steps)
                # Win iff the terminating event was us scoring on the opponent.
                wins.append(1 if last_event == "goal_top" else 0)
                break

    metrics = {
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "win_rate": float(np.mean(wins)),
        "mean_length": float(np.mean(lengths)),
        "n_episodes": len(returns),
    }
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=None, help="Override DDIM inference steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--opponent", type=str, default=None,
                   help="Checkpoint for the top-paddle opponent (SAC or Diffusion Policy). "
                        "If omitted, the opponent sits still.")
    args = p.parse_args()

    m = evaluate(args.ckpt, args.episodes, args.steps, seed=args.seed,
                 opponent_ckpt=args.opponent)
    print()
    print(f"Checkpoint: {args.ckpt}")
    print(f"Episodes:   {m['n_episodes']}")
    print(f"Win rate:   {m['win_rate']:.1%}")
    print(f"Mean return: {m['mean_return']:+.2f}")
    print(f"Median return: {m['median_return']:+.2f}")
    print(f"Mean length: {m['mean_length']:.0f} steps")


if __name__ == "__main__":
    main()
