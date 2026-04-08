"""Roll out a trained SAC expert against a diverse opponent league and
save every (obs, action) step as a .npz. Used as the Stage 2 BC dataset.

The opponent mix matches `train_sac.py` so the BC dataset has the same
distribution of game situations the SAC agent was trained on. Rollout
actions are stochastic by default — the diffusion policy needs to learn
the spread of the SAC actor's distribution, not just its mean.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from airhockey.env import AirHockeyEnv
from airhockey.eval_sac import (
    scripted_attacker,
    scripted_tracker,
    with_action_noise,
)
from airhockey.physics import PhysicsConfig
from airhockey.sac import SACAgent, SACConfig

# Match train_sac.py league mix exactly so the BC dataset comes from the
# same distribution the agent was trained on.
LEAGUE_TRACKER_PROB = 0.30
LEAGUE_ATTACKER_PROB = 0.30
LEAGUE_STATIONARY_PROB = 0.15
LEAGUE_SNAPSHOT_PROB = 0.25
SCRIPTED_NOISE_STD = 0.05


def collect(
    expert_ckpt: str,
    episodes: int,
    out_path: str,
    max_steps: int = 800,
    seed: int = 0,
    device: str = "cpu",
    deterministic: bool = False,
) -> None:
    device_t = torch.device(device)
    ckpt = torch.load(expert_ckpt, map_location=device_t, weights_only=False)
    cfg = SACConfig(**ckpt["config"])
    agent = SACAgent(cfg, device=device_t)
    agent.load_state_dict(ckpt)
    agent.actor.eval()

    # Build the same opponent league train_sac.py uses. The "snapshot"
    # slot is filled with a frozen copy of the SAC expert itself.
    noisy_tracker = with_action_noise(
        scripted_tracker, SCRIPTED_NOISE_STD, seed=seed + 101,
    )
    noisy_attacker = with_action_noise(
        scripted_attacker, SCRIPTED_NOISE_STD, seed=seed + 202,
    )

    @torch.no_grad()
    def snapshot_opponent(obs_top: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs_top.astype(np.float32)).unsqueeze(0).to(device_t)
        action, _ = agent.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy()

    def stationary_opponent(_obs_top: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

    league_rng = random.Random(seed + 999)

    def pick_opponent():
        r = league_rng.random()
        if r < LEAGUE_STATIONARY_PROB:
            return "stationary", stationary_opponent
        if r < LEAGUE_STATIONARY_PROB + LEAGUE_TRACKER_PROB:
            return "tracker", noisy_tracker
        if r < LEAGUE_STATIONARY_PROB + LEAGUE_TRACKER_PROB + LEAGUE_ATTACKER_PROB:
            return "attacker", noisy_attacker
        return "snapshot", snapshot_opponent

    env = AirHockeyEnv(
        physics_config=PhysicsConfig(),
        seed=seed,
        max_episode_steps=max_steps,
    )

    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    ep_buf: list[int] = []
    t_buf: list[int] = []

    opponent_counts: Counter = Counter()
    outcomes: Counter = Counter()
    ep_lengths: list[int] = []

    for ep in tqdm(range(episodes), desc="Collecting"):
        opp_name, opp_fn = pick_opponent()
        env.opponent = opp_fn
        opponent_counts[opp_name] += 1
        obs, _ = env.reset(seed=seed + ep)
        steps = 0
        last_event = ""
        for t in range(max_steps):
            action = agent.act(obs, deterministic=deterministic).astype(np.float32)
            obs_buf.append(obs.astype(np.float32))
            act_buf.append(action)
            ep_buf.append(ep)
            t_buf.append(t)
            obs, _r, terminated, truncated, info = env.step(action)
            steps += 1
            evt = info.get("event", "")
            if evt.startswith("goal"):
                last_event = evt
            if terminated or truncated:
                break
        ep_lengths.append(steps)
        if last_event == "goal_bot":
            outcomes["win"] += 1
        elif last_event == "goal_top":
            outcomes["loss"] += 1
        else:
            outcomes["draw"] += 1

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    obs_arr = np.stack(obs_buf)
    act_arr = np.stack(act_buf)
    np.savez_compressed(
        out,
        obs=obs_arr,
        act=act_arr,
        episode=np.array(ep_buf, dtype=np.int32),
        timestep=np.array(t_buf, dtype=np.int32),
    )

    n_ep = len(ep_lengths)
    print()
    print(f"Saved {len(obs_buf):,} transitions across {n_ep} episodes to {out}")
    print(f"Mean episode length: {np.mean(ep_lengths):.0f} steps")
    print(f"Outcomes: "
          f"{outcomes['win']} wins ({outcomes['win']/n_ep:.1%}), "
          f"{outcomes['loss']} losses ({outcomes['loss']/n_ep:.1%}), "
          f"{outcomes['draw']} draws ({outcomes['draw']/n_ep:.1%})")
    print(f"Opponent mix: " + ", ".join(
        f"{k} {v/n_ep:.0%}" for k, v in sorted(opponent_counts.items())
    ))
    print(f"Action stats: mean {act_arr.mean(0)}, std {act_arr.std(0)}, "
          f"abs-mean {np.abs(act_arr).mean():.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", type=str, required=True, help="Path to trained SAC checkpoint")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--out", type=str, default="data/demos.npz")
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="Sample deterministic actions (default: stochastic)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    collect(
        args.expert, args.episodes, args.out,
        max_steps=args.max_steps, seed=args.seed, device=args.device,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
