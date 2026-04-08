"""Stage 1 — Train the SAC expert via self-play.

Opponent schedule:
  - Scripted tracker for the first `opponent_warmup_steps` steps so the
    agent sees a moving opponent immediately.
  - After warmup, a league: 40% scripted, 15% stationary (no-op), 45%
    stochastic snapshots of past selves. The league is refreshed every
    `opponent_refresh_steps` steps.

Periodic eval: every `eval_every_steps` steps the current actor is
evaluated (deterministic) against scripted and stationary opponents for
`eval_episodes` episodes each. The average win rate across both is the
selection metric — the best-so-far is saved to `<out>.best.pt`. The
final checkpoint is saved to `<out>`.

Logging: `train.csv` (per log interval) and `eval.csv` (per eval) in
the output directory, plus a tqdm progress bar.
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from airhockey.env import AirHockeyEnv
from airhockey.eval_sac import (
    run_eval,
    scripted_attacker,
    scripted_tracker,
    with_action_noise,
)
from airhockey.physics import PhysicsConfig
from airhockey.sac import ReplayBuffer, SACAgent, SACConfig


# ── Opponent mix (must sum to 1.0) ─────────────────────────────
LEAGUE_TRACKER_PROB = 0.30      # passive defender
LEAGUE_ATTACKER_PROB = 0.30     # active scorer — generates pressure
LEAGUE_STATIONARY_PROB = 0.15
LEAGUE_SNAPSHOT_PROB = 0.25

# Noise on the scripted opponent (training league + eval) so the agent
# can't memorize a fixed action sequence.
SCRIPTED_NOISE_STD = 0.05


# ── Horizontal mirroring (data augmentation) ───────────────────
# The env is left-right symmetric. Every (obs, action, reward, next_obs)
# tuple has a valid mirrored counterpart we can also push to the buffer
# — doubles effective data per env step at no rollout cost.
#
# Obs layout from physics.get_obs (perspective="bot"):
#   [0] puck_x  [1] puck_y  [2] puck_vx  [3] puck_vy
#   [4] bot_x   [5] bot_y   [6] bot_vx   [7] bot_vy
#   [8] top_x   [9] top_y
# Mirroring x: positions 0,4,8 → 1 - p; velocities 2,6 → -v.
_MIRROR_POS_IDX = (0, 4, 8)
_MIRROR_VEL_IDX = (2, 6)


def mirror_obs(obs: np.ndarray) -> np.ndarray:
    o = obs.copy()
    for i in _MIRROR_POS_IDX:
        o[i] = 1.0 - o[i]
    for i in _MIRROR_VEL_IDX:
        o[i] = -o[i]
    return o


def mirror_action(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[0] = -out[0]
    return out


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
    eval_episodes: int = 30
    log_every_steps: int = 1_000
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _stationary_opponent(obs_top: np.ndarray) -> np.ndarray:
    return np.zeros(2, dtype=np.float32)


def main(args: TrainArgs) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Noisy scripted opponents — used in the league. Noise prevents the
    # agent from memorizing a fixed action sequence.
    noisy_tracker = with_action_noise(
        scripted_tracker, SCRIPTED_NOISE_STD, seed=args.seed + 101,
    )
    noisy_attacker = with_action_noise(
        scripted_attacker, SCRIPTED_NOISE_STD, seed=args.seed + 202,
    )

    # Training env. Warmup opponent is the attacker so goal events fire
    # quickly under the pure-sparse reward; swapped to the league after.
    env = AirHockeyEnv(physics_config=PhysicsConfig(), seed=args.seed,
                       opponent=noisy_attacker)

    cfg = SACConfig(obs_dim=env.observation_space.shape[0],
                    act_dim=env.action_space.shape[0])
    agent = SACAgent(cfg, device=device)
    buffer = ReplayBuffer(args.buffer_size, cfg.obs_dim, cfg.act_dim)

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
        r = league_rng.random()
        if r < LEAGUE_STATIONARY_PROB:
            return _stationary_opponent(obs_top)
        if r < LEAGUE_STATIONARY_PROB + LEAGUE_TRACKER_PROB:
            return noisy_tracker(obs_top)
        if r < LEAGUE_STATIONARY_PROB + LEAGUE_TRACKER_PROB + LEAGUE_ATTACKER_PROB:
            return noisy_attacker(obs_top)
        if opponent_league:
            snapshot = league_rng.choice(list(opponent_league))
            return _sampled_policy_fn(snapshot)(obs_top)
        # Fallback before any snapshot exists.
        return noisy_attacker(obs_top)

    def add_snapshot_to_league() -> None:
        snap = copy.deepcopy(agent.actor).eval()
        for p in snap.parameters():
            p.requires_grad = False
        opponent_league.append(snap)
        env.opponent = league_opponent_fn

    # Output paths + CSV loggers.
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_path.with_suffix(".best.pt")
    train_csv_path = out_dir / "train.csv"
    eval_csv_path = out_dir / "eval.csv"
    goals_csv_path = out_dir / "goals.csv"
    train_csv = open(train_csv_path, "w", newline="")
    eval_csv = open(eval_csv_path, "w", newline="")
    goals_csv = open(goals_csv_path, "w", newline="")
    train_writer = csv.writer(train_csv)
    eval_writer = csv.writer(eval_csv)
    goals_writer = csv.writer(goals_csv)
    train_writer.writerow([
        "step", "ep_return_mean", "ep_length_mean",
        "q_loss", "pi_loss", "alpha", "entropy",
    ])
    eval_writer.writerow([
        "step", "opponent",
        "win_rate", "loss_rate", "draw_rate",
        "mean_return", "mean_length", "n_episodes",
    ])
    goals_writer.writerow([
        "step", "scored", "puck_x", "puck_y", "puck_vx", "puck_vy",
    ])

    best_score = -float("inf")

    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_length = 0
    ep_returns: list[float] = []
    ep_lengths: list[int] = []

    latest_metrics: dict[str, float] = {}
    pbar = tqdm(total=args.total_steps, desc="SAC")

    def do_eval(step: int) -> None:
        nonlocal best_score
        metrics_scripted = run_eval(
            agent, scripted_tracker, episodes=args.eval_episodes,
            seed=10_000 + step, opponent_noise_std=SCRIPTED_NOISE_STD,
        )
        metrics_attacker = run_eval(
            agent, scripted_attacker, episodes=args.eval_episodes,
            seed=15_000 + step, opponent_noise_std=SCRIPTED_NOISE_STD,
        )
        metrics_none = run_eval(
            agent, None, episodes=args.eval_episodes,
            seed=20_000 + step,
        )
        # Put training env's actor back in train mode — run_eval flips it
        # to eval(), which disables dropout/batchnorm. We have neither,
        # but be explicit so future refactors don't silently break.
        agent.actor.train()

        named = (
            ("scripted", metrics_scripted),
            ("attacker", metrics_attacker),
            ("none", metrics_none),
        )
        for name, m in named:
            eval_writer.writerow([
                step, name,
                f"{m['win_rate']:.4f}",
                f"{m['loss_rate']:.4f}",
                f"{m['draw_rate']:.4f}",
                f"{m['mean_return']:.4f}",
                f"{m['mean_length']:.2f}",
                m["n_episodes"],
            ])
        eval_csv.flush()

        # Selection score: average win rate across all three opponents.
        score = (
            metrics_scripted["win_rate"]
            + metrics_attacker["win_rate"]
            + metrics_none["win_rate"]
        ) / 3.0
        tqdm.write(
            f"[step {step}] eval  scripted {metrics_scripted['win_rate']:.1%}"
            f" / attacker {metrics_attacker['win_rate']:.1%}"
            f" / none {metrics_none['win_rate']:.1%}  (score {score:.3f})"
        )
        if score > best_score:
            best_score = score
            torch.save(agent.state_dict(), best_path)
            tqdm.write(f"[step {step}] new best checkpoint → {best_path}")

    for step in range(args.total_steps):
        if step < args.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, term, trunc, info = env.step(action)
        # Do not bootstrap past truncation — only `term` flags the done.
        done = bool(term)
        action_arr = np.asarray(action, dtype=np.float32)
        buffer.push(obs, action_arr, reward, next_obs, done)
        # Mirrored counterpart — env is left-right symmetric, so this is
        # a valid extra training transition.
        buffer.push(
            mirror_obs(obs), mirror_action(action_arr),
            reward, mirror_obs(next_obs), done,
        )

        # Goal location logging.
        if "goal_puck" in info:
            gx, gy, gvx, gvy = info["goal_puck"]
            goals_writer.writerow([
                step + 1,
                1 if info["event"] == "goal_bot" else 0,
                f"{gx:.2f}", f"{gy:.2f}", f"{gvx:.2f}", f"{gvy:.2f}",
            ])

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
            train_writer.writerow([
                step + 1,
                f"{avg_ret:.4f}",
                f"{avg_len:.2f}",
                f"{latest_metrics.get('q_loss', 0.0):.4f}",
                f"{latest_metrics.get('pi_loss', 0.0):.4f}",
                f"{latest_metrics.get('alpha', 0.0):.4f}",
                f"{latest_metrics.get('entropy', 0.0):.4f}",
            ])
            train_csv.flush()
            postfix = {"ret": f"{avg_ret:+.2f}", "len": f"{avg_len:.0f}"}
            if latest_metrics:
                postfix["qL"] = f"{latest_metrics.get('q_loss', 0):.2f}"
                postfix["piL"] = f"{latest_metrics.get('pi_loss', 0):+.2f}"
                postfix["a"] = f"{latest_metrics.get('alpha', 0):.3f}"
                postfix["H"] = f"{latest_metrics.get('entropy', 0):.2f}"
            pbar.set_postfix(**postfix)

        if (
            args.eval_every_steps > 0
            and step >= args.learning_starts
            and (step + 1) % args.eval_every_steps == 0
        ):
            do_eval(step + 1)

        pbar.update(1)

    pbar.close()
    train_csv.close()
    goals_csv.close()

    # Final checkpoint. If we never evaluated, `best` wasn't written —
    # do one final eval so the report is complete and `best.pt` exists.
    if best_score == -float("inf"):
        do_eval(args.total_steps)
    eval_csv.close()

    torch.save(agent.state_dict(), out_path)
    print(f"Saved final SAC checkpoint to {out_path}")
    print(f"Best checkpoint (score {best_score:.3f}) at {best_path}")
    print(f"Training log: {train_csv_path}")
    print(f"Eval log:     {eval_csv_path}")


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
