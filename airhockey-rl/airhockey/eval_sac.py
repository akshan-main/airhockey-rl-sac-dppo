"""Evaluate a SAC checkpoint as the bottom-paddle agent.

Counterpart to airhockey.eval (which is hardcoded for diffusion-policy
checkpoints). Loads a SAC actor, runs N episodes against a configurable
opponent (default: no-op), reports win rate / mean return / mean length.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm

from airhockey.env import AirHockeyEnv
from airhockey.physics import PhysicsConfig
from airhockey.sac import SACAgent, SACConfig
from airhockey.snapshot_opponent import load_opponent


def scripted_tracker(obs: np.ndarray) -> np.ndarray:
    """Deterministic baseline opponent. Tracks the puck on x; defends
    near home y when the puck is on the agent's side of the field, and
    advances slightly when the puck is on its own side.

    Receives `obs` in this paddle's mirrored perspective (it sees itself
    as if it were the bottom paddle), so the same code works for either
    side. Layout matches physics.get_obs:
      [0] puck_x       [1] puck_y       [2] puck_vx    [3] puck_vy
      [4] own_x        [5] own_y        [6] own_vx     [7] own_vy
      [8] other_x      [9] other_y
    All positions are normalized to [0, 1]; in this paddle's mirrored
    frame, larger y is closer to its own goal (home), smaller y is
    toward the opponent.
    """
    puck_x, puck_y = float(obs[0]), float(obs[1])
    pad_x, pad_y = float(obs[4]), float(obs[5])

    # Horizontal: track puck with a P-controller.
    ax = np.clip((puck_x - pad_x) * 8.0, -1.0, 1.0)

    # Vertical: defend near home (y ≈ 0.85). When the puck is on this
    # paddle's own side (mirrored y > 0.5), advance to meet it but stop
    # short so we don't overshoot past it.
    if puck_y > 0.5:
        target_y = max(0.55, puck_y - 0.08)
    else:
        target_y = 0.85
    ay = np.clip((target_y - pad_y) * 8.0, -1.0, 1.0)

    return np.array([ax, ay], dtype=np.float32)


@torch.no_grad()
def evaluate_sac(
    ckpt_path: str,
    episodes: int = 200,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 0,
    opponent_ckpt: str | None = None,
) -> dict[str, float]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = SACConfig(**ckpt["config"])
    agent = SACAgent(cfg, device=device)
    agent.load_state_dict(ckpt)
    agent.actor.eval()

    env = AirHockeyEnv(physics_config=PhysicsConfig(), seed=seed)
    if opponent_ckpt is None or opponent_ckpt == "scripted":
        env.opponent = scripted_tracker
    elif opponent_ckpt == "none":
        env.opponent = None
    elif opponent_ckpt == "self":
        env.opponent = load_opponent(ckpt_path, device=device, deterministic=True)
    else:
        env.opponent = load_opponent(opponent_ckpt, device=device, deterministic=True)

    returns: list[float] = []
    lengths: list[int] = []
    wins: list[int] = []

    for ep in tqdm(range(episodes), desc="Eval-SAC"):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        steps = 0
        last_event = ""
        while True:
            action = agent.act(obs.astype(np.float32), deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            last_event = info.get("event", "") or last_event
            ep_return += reward
            steps += 1
            if term or trunc:
                returns.append(ep_return)
                lengths.append(steps)
                wins.append(1 if last_event == "goal_bot" else 0)
                break

    return {
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "win_rate": float(np.mean(wins)),
        "mean_length": float(np.mean(lengths)),
        "n_episodes": len(returns),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--opponent", type=str, default=None,
                   help="Opponent: a ckpt path, 'scripted' (default), "
                        "'self' (eval vs same ckpt), or 'none' (stationary).")
    args = p.parse_args()

    m = evaluate_sac(args.ckpt, args.episodes, seed=args.seed,
                     opponent_ckpt=args.opponent)
    print()
    print(f"Checkpoint: {args.ckpt}")
    print(f"Opponent:   {args.opponent or 'no-op (stationary)'}")
    print(f"Episodes:   {m['n_episodes']}")
    print(f"Win rate:   {m['win_rate']:.1%}")
    print(f"Mean return: {m['mean_return']:+.2f}")
    print(f"Median return: {m['median_return']:+.2f}")
    print(f"Mean length: {m['mean_length']:.0f} steps")


if __name__ == "__main__":
    main()
