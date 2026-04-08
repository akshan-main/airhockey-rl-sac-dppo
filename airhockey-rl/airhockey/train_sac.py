"""Stage 1 — Train the SAC expert via self-play.

Opponent schedule:
  - No-op opponent for the first `opponent_warmup_steps` steps.
  - After warmup, a frozen deepcopy of the current actor.
  - Refreshed every `opponent_refresh_steps` steps.
"""
from __future__ import annotations

import argparse
import copy
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from airhockey.env import AirHockeyEnv
from airhockey.eval_sac import scripted_tracker
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
    opponent_refresh_steps: int = 25_000
    opponent_league_size: int = 5
    eval_every_steps: int = 50_000
    eval_episodes: int = 20
    log_every_steps: int = 1_000
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: TrainArgs) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Start against the scripted tracker so the agent has a moving
    # opponent to learn against from step 0. After warmup, switch to a
    # self-play league (frozen snapshots of the agent itself).
    env = AirHockeyEnv(physics_config=PhysicsConfig(), seed=args.seed,
                       opponent=scripted_tracker)

    cfg = SACConfig(obs_dim=env.observation_space.shape[0],
                    act_dim=env.action_space.shape[0])
    agent = SACAgent(cfg, device=device)
    buffer = ReplayBuffer(args.buffer_size, cfg.obs_dim, cfg.act_dim)

    # Opponent league: a mix of recent actor snapshots plus the scripted
    # tracker baseline. Each episode picks one uniformly at random.
    opponent_league: deque = deque(maxlen=args.opponent_league_size)
    league_rng = random.Random(args.seed + 17)

    def _sampled_policy_fn(snapshot):
        def fn(obs_top: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_top.astype(np.float32)).unsqueeze(0).to(device)
                action, _ = snapshot.sample(obs_t)
            return action.squeeze(0).cpu().numpy()
        return fn

    def league_opponent_fn(obs_top: np.ndarray) -> np.ndarray:
        # 30% scripted, 70% a random policy snapshot (if any exist).
        if opponent_league and league_rng.random() < 0.7:
            snapshot = league_rng.choice(list(opponent_league))
            return _sampled_policy_fn(snapshot)(obs_top)
        return scripted_tracker(obs_top)

    def add_snapshot_to_league() -> None:
        snap = copy.deepcopy(agent.actor).eval()
        for p in snap.parameters():
            p.requires_grad = False
        opponent_league.append(snap)
        env.opponent = league_opponent_fn

    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_length = 0
    ep_returns: list[float] = []
    ep_lengths: list[int] = []

    latest_metrics: dict[str, float] = {}
    pbar = tqdm(total=args.total_steps, desc="SAC")
    for step in range(args.total_steps):
        if step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, term, trunc, info = env.step(action)
        # Do not bootstrap past truncation — only use `term` for the done flag.
        done = bool(term)
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

        if step >= args.learning_starts and step % args.update_every == 0:
            for _ in range(args.updates_per_step):
                batch = buffer.sample(args.batch_size, device)
                latest_metrics = agent.update(batch)

        if step == args.opponent_warmup_steps:
            add_snapshot_to_league()
            tqdm.write(f"[step {step}] switched to self-play league")
        elif (
            len(opponent_league) > 0
            and step > args.opponent_warmup_steps
            and step % args.opponent_refresh_steps == 0
        ):
            add_snapshot_to_league()

        if (step + 1) % args.log_every_steps == 0 and ep_returns:
            avg_ret = float(np.mean(ep_returns[-20:]))
            avg_len = float(np.mean(ep_lengths[-20:]))
            postfix = {"ret": f"{avg_ret:+.2f}", "len": f"{avg_len:.0f}"}
            if latest_metrics:
                postfix["qL"] = f"{latest_metrics.get('q_loss', 0):.2f}"
                postfix["piL"] = f"{latest_metrics.get('pi_loss', 0):+.2f}"
                postfix["a"] = f"{latest_metrics.get('alpha', 0):.2f}"
                postfix["H"] = f"{latest_metrics.get('entropy', 0):.2f}"
            pbar.set_postfix(**postfix)

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
