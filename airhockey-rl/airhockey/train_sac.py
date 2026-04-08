"""Stage 1 — Train the SAC expert via self-play.

This is the expert that Stage 2 will distill into a Diffusion Policy.

Self-play schedule:
  • Start with a no-op opponent (keeps the puck in play long enough for
    the agent to learn basic hitting).
  • After `opponent_warmup_steps`, switch the opponent to a frozen snapshot
    of the current actor.
  • Every `opponent_refresh_steps`, refresh the frozen opponent to the
    latest actor weights.

This is the standard self-play loop. It's not perfect (no league, no
population) but it works well enough for a single-agent target.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from airhockey.env import AirHockeyEnv
from airhockey.physics import PhysicsConfig
from airhockey.sac import ReplayBuffer, SACAgent, SACConfig


@dataclass
class TrainArgs:
    out: str = "ckpt/sac_expert.pt"
    total_steps: int = 1_000_000
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 5_000
    update_every: int = 1
    updates_per_step: int = 1
    opponent_warmup_steps: int = 50_000
    opponent_refresh_steps: int = 50_000
    eval_every_steps: int = 50_000
    eval_episodes: int = 20
    log_every_steps: int = 1_000
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: TrainArgs) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Env starts with a no-op opponent so the agent can learn basic physics
    env = AirHockeyEnv(physics_config=PhysicsConfig(), seed=args.seed, opponent=None)

    cfg = SACConfig(obs_dim=env.observation_space.shape[0],
                    act_dim=env.action_space.shape[0])
    agent = SACAgent(cfg, device=device)
    buffer = ReplayBuffer(args.buffer_size, cfg.obs_dim, cfg.act_dim)

    # Frozen opponent — starts as None, gets populated after warmup
    frozen_actor: Optional[torch.nn.Module] = None

    def frozen_opponent_fn(obs_top: np.ndarray) -> np.ndarray:
        if frozen_actor is None:
            return np.zeros(cfg.act_dim, dtype=np.float32)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs_top.astype(np.float32)).unsqueeze(0).to(device)
            action = frozen_actor.act_deterministic(obs_t)
        return action.squeeze(0).cpu().numpy()

    def refresh_frozen_opponent() -> None:
        nonlocal frozen_actor
        frozen_actor = copy.deepcopy(agent.actor).eval()
        for p in frozen_actor.parameters():
            p.requires_grad = False
        env.opponent = frozen_opponent_fn

    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_length = 0
    ep_returns: list[float] = []
    ep_lengths: list[int] = []

    pbar = tqdm(total=args.total_steps, desc="SAC")
    for step in range(args.total_steps):
        # ── Action ─────────────────────────────────────────
        if step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        # ── Env step ───────────────────────────────────────
        next_obs, reward, term, trunc, info = env.step(action)
        done = bool(term)  # do not bootstrap on truncation
        buffer.push(obs, action, reward, next_obs, done)

        ep_return += reward
        ep_length += 1
        if term or trunc:
            ep_returns.append(ep_return)
            ep_lengths.append(ep_length)
            ep_return = 0.0
            ep_length = 0
            obs, _ = env.reset()
        else:
            obs = next_obs

        # ── Update ─────────────────────────────────────────
        if step >= args.learning_starts and step % args.update_every == 0:
            for _ in range(args.updates_per_step):
                batch = buffer.sample(args.batch_size, device)
                metrics = agent.update(batch)

        # ── Opponent schedule ─────────────────────────────
        if step == args.opponent_warmup_steps:
            refresh_frozen_opponent()
            tqdm.write(f"[step {step}] switched to self-play opponent")
        elif (
            frozen_actor is not None
            and step > args.opponent_warmup_steps
            and step % args.opponent_refresh_steps == 0
        ):
            refresh_frozen_opponent()

        # ── Logging ────────────────────────────────────────
        if (step + 1) % args.log_every_steps == 0 and ep_returns:
            avg_ret = float(np.mean(ep_returns[-20:]))
            avg_len = float(np.mean(ep_lengths[-20:]))
            pbar.set_postfix(ret=f"{avg_ret:+.2f}", len=f"{avg_len:.0f}")

        pbar.update(1)

    pbar.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), out)
    print(f"Saved SAC checkpoint to {out}")


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser()
    for f in TrainArgs.__dataclass_fields__.values():
        p.add_argument(
            f"--{f.name.replace('_', '-')}",
            type=type(f.default),
            default=f.default,
        )
    a = p.parse_args()
    return TrainArgs(**{k: getattr(a, k) for k in TrainArgs.__dataclass_fields__.keys()})


if __name__ == "__main__":
    main(parse_args())
