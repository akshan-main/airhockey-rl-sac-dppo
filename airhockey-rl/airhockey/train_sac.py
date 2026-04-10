"""Stage 1 — Train the SAC expert.

Simple setup:
  - Dense micro-rewards from the env (approach, puck direction, hit)
  - Opponent league: scripted tracker, scripted attacker, snapshots
  - Winners-only buffer: only episodes where the bot scored go into
    the persistent "good" buffer. 25% of each training batch is
    sampled from the good buffer so the agent always learns from
    what worked.
  - Demo episodes from scripted_attacker seed the good buffer before
    training starts.
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

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

SCRIPTED_NOISE_STD = 0.05


def mirror_obs(obs: np.ndarray) -> np.ndarray:
    o = obs.copy()
    o[0] = 1.0 - o[0]  # puck_x
    o[4] = 1.0 - o[4]  # bot_x
    o[8] = 1.0 - o[8]  # top_x
    o[2] = -o[2]        # puck_vx
    o[6] = -o[6]        # bot_vx
    return o


def mirror_action(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[0] = -out[0]
    return out


@dataclass
class TrainArgs:
    out: str = "ckpt/sac_expert.pt"
    total_steps: int = 2_000_000
    batch_size: int = 256
    buffer_size: int = 2_000_000
    learning_starts: int = 5_000
    updates_per_step: int = 1
    opponent_refresh_steps: int = 50_000
    opponent_league_size: int = 5
    demo_episodes: int = 1000
    eval_every_steps: int = 100_000
    eval_episodes: int = 50
    log_every_steps: int = 5_000
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def main(args: TrainArgs) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    noisy_tracker = with_action_noise(scripted_tracker, SCRIPTED_NOISE_STD, seed=args.seed + 101)
    noisy_attacker = with_action_noise(scripted_attacker, SCRIPTED_NOISE_STD, seed=args.seed + 202)

    opponent_league: deque = deque(maxlen=args.opponent_league_size)
    league_rng = random.Random(args.seed + 17)

    def _snapshot_fn(snapshot):
        def fn(obs_top):
            with torch.no_grad():
                obs_t = torch.from_numpy(obs_top.astype(np.float32)).unsqueeze(0).to(device)
                action, _ = snapshot.sample(obs_t)
            return action.squeeze(0).cpu().numpy()
        return fn

    def league_opponent(obs_top):
        r = league_rng.random()
        if r < 0.35:
            return noisy_tracker(obs_top)
        if r < 0.70:
            return noisy_attacker(obs_top)
        if opponent_league:
            snap = league_rng.choice(list(opponent_league))
            return _snapshot_fn(snap)(obs_top)
        return noisy_attacker(obs_top)

    env = AirHockeyEnv(physics_config=PhysicsConfig(), seed=args.seed,
                       opponent=league_opponent)

    cfg = SACConfig(obs_dim=env.observation_space.shape[0],
                    act_dim=env.action_space.shape[0])
    agent = SACAgent(cfg, device=device)
    buffer = ReplayBuffer(args.buffer_size, cfg.obs_dim, cfg.act_dim)
    good_buffer = ReplayBuffer(500_000, cfg.obs_dim, cfg.act_dim)

    def add_snapshot():
        snap = copy.deepcopy(agent.actor).eval()
        for p in snap.parameters():
            p.requires_grad = False
        opponent_league.append(snap)

    # ── Collect demos from scripted_attacker ──────────────────
    if args.demo_episodes > 0:
        demo_rng = random.Random(args.seed + 333)
        demo_wins = demo_total = 0
        for ep in tqdm(range(args.demo_episodes), desc="Demos"):
            opp = noisy_attacker if demo_rng.random() < 0.5 else noisy_tracker
            env.opponent = opp
            obs_d, _ = env.reset(seed=args.seed + 50_000 + ep)
            ep_transitions = []
            last_event = ""
            while True:
                action_d = scripted_attacker(obs_d.astype(np.float32))
                next_obs_d, rew_d, term_d, trunc_d, info_d = env.step(action_d)
                ep_transitions.append((obs_d.copy(), action_d.copy(), rew_d, next_obs_d.copy(), bool(term_d)))
                evt = info_d.get("event", "")
                if evt.startswith("goal"):
                    last_event = evt
                obs_d = next_obs_d
                if term_d or trunc_d:
                    break
            # All demo transitions go into the main buffer for early learning.
            # Winning episodes also go into the good buffer.
            for o, a, r, no, d in ep_transitions:
                buffer.push(o, a, r, no, d)
                buffer.push(mirror_obs(o), mirror_action(a), r, mirror_obs(no), d)
            if last_event == "goal_bot":
                demo_wins += 1
                for o, a, r, no, d in ep_transitions:
                    good_buffer.push(o, a, r, no, d)
                    good_buffer.push(mirror_obs(o), mirror_action(a), r, mirror_obs(no), d)
            demo_total += 1
        env.opponent = league_opponent
        print(f"Demos: {demo_wins}W / {demo_total} episodes, "
              f"good_buffer={good_buffer.size:,}, buffer={buffer.size:,}")

    # ── BC pre-train actor on demo actions ────────────────────
    if good_buffer.size > 0:
        bc_optim = torch.optim.Adam(agent.actor.parameters(), lr=1e-3)
        agent.actor.train()
        for _ in tqdm(range(1000), desc="BC pre-train"):
            idx = np.random.randint(0, good_buffer.size, size=min(512, good_buffer.size))
            obs_bc = torch.from_numpy(good_buffer.obs[idx]).to(device)
            act_bc = torch.from_numpy(good_buffer.act[idx]).to(device)
            mean, _ = agent.actor(obs_bc)
            loss = torch.nn.functional.mse_loss(torch.tanh(mean), act_bc)
            bc_optim.zero_grad()
            loss.backward()
            bc_optim.step()
        print(f"BC pre-train done, loss={loss.item():.4f}")
        agent.opt_actor = torch.optim.Adam(agent.actor.parameters(), lr=cfg.actor_lr)

    # ── Output paths + CSV loggers ────────────────────────────
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_path.with_suffix(".best.pt")
    train_csv = open(out_dir / "train.csv", "w", newline="")
    eval_csv = open(out_dir / "eval.csv", "w", newline="")
    goals_csv = open(out_dir / "goals.csv", "w", newline="")
    tw = csv.writer(train_csv)
    ew = csv.writer(eval_csv)
    gw = csv.writer(goals_csv)
    tw.writerow(["step", "ep_return_mean", "ep_length_mean", "q_loss", "pi_loss", "alpha", "entropy"])
    ew.writerow(["step", "opponent", "win_rate", "loss_rate", "draw_rate", "mean_return", "mean_length", "n_episodes"])
    gw.writerow(["step", "scored", "puck_x", "puck_y", "puck_vx", "puck_vy"])

    best_score = -float("inf")
    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_length = 0
    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    ep_transitions: list[tuple] = []
    latest_metrics: dict[str, float] = {}

    def do_eval(step):
        nonlocal best_score
        ms = run_eval(agent, scripted_tracker, episodes=args.eval_episodes,
                      seed=10_000 + step, opponent_noise_std=SCRIPTED_NOISE_STD)
        ma = run_eval(agent, scripted_attacker, episodes=args.eval_episodes,
                      seed=15_000 + step, opponent_noise_std=SCRIPTED_NOISE_STD)
        agent.actor.train()
        for name, m in [("scripted", ms), ("attacker", ma)]:
            ew.writerow([step, name, f"{m['win_rate']:.4f}", f"{m['loss_rate']:.4f}",
                         f"{m['draw_rate']:.4f}", f"{m['mean_return']:.4f}",
                         f"{m['mean_length']:.2f}", m["n_episodes"]])
        eval_csv.flush()
        score = (ms["win_rate"] + ma["win_rate"]) / 2.0
        tqdm.write(f"[step {step}] eval  scripted {ms['win_rate']:.1%} / attacker {ma['win_rate']:.1%}  (score {score:.3f})")
        if score > best_score:
            best_score = score
            torch.save(agent.state_dict(), best_path)
            tqdm.write(f"[step {step}] new best → {best_path}")

    effective_starts = 0 if args.demo_episodes > 0 else args.learning_starts
    pbar = tqdm(total=args.total_steps, desc="SAC")

    for step in range(args.total_steps):
        if step < effective_starts:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, deterministic=False)

        next_obs, reward, term, trunc, info = env.step(action)
        done = bool(term)
        action_arr = np.asarray(action, dtype=np.float32)

        # All transitions go into the main buffer
        buffer.push(obs, action_arr, reward, next_obs, done)
        buffer.push(mirror_obs(obs), mirror_action(action_arr), reward, mirror_obs(next_obs), done)

        # Track episode transitions for winners-only buffer
        ep_transitions.append((obs.copy(), action_arr.copy(), reward, next_obs.copy(), done))

        if "goal_puck" in info:
            gx, gy, gvx, gvy = info["goal_puck"]
            gw.writerow([step + 1, 1 if info["event"] == "goal_bot" else 0,
                         f"{gx:.2f}", f"{gy:.2f}", f"{gvx:.2f}", f"{gvy:.2f}"])

        ep_return += reward
        ep_length += 1
        if term or trunc:
            ep_returns.append(ep_return)
            ep_lengths.append(ep_length)

            # Winners go into the good buffer
            if info.get("event") == "goal_bot":
                for o, a, r, no, d in ep_transitions:
                    good_buffer.push(o, a, r, no, d)
                    good_buffer.push(mirror_obs(o), mirror_action(a), r, mirror_obs(no), d)

            ep_return = 0.0
            ep_length = 0
            ep_transitions = []
            obs, _ = env.reset()
        else:
            obs = next_obs

        if step >= effective_starts:
            for _ in range(args.updates_per_step):
                if good_buffer.size > 0:
                    batch = buffer.sample_mixed(args.batch_size, device, good_buffer, demo_fraction=0.25)
                else:
                    batch = buffer.sample(args.batch_size, device)
                latest_metrics = agent.update(batch)

        # Refresh snapshots
        if step > 0 and step % args.opponent_refresh_steps == 0:
            add_snapshot()

        if (step + 1) % args.log_every_steps == 0 and ep_returns:
            avg_ret = float(np.mean(ep_returns[-20:]))
            avg_len = float(np.mean(ep_lengths[-20:]))
            tw.writerow([step + 1, f"{avg_ret:.4f}", f"{avg_len:.2f}",
                         f"{latest_metrics.get('q_loss', 0):.4f}",
                         f"{latest_metrics.get('pi_loss', 0):.4f}",
                         f"{latest_metrics.get('alpha', 0):.4f}",
                         f"{latest_metrics.get('entropy', 0):.4f}"])
            train_csv.flush()
            postfix = {"ret": f"{avg_ret:+.2f}", "len": f"{avg_len:.0f}",
                       "good": f"{good_buffer.size:,}"}
            if latest_metrics:
                postfix["qL"] = f"{latest_metrics.get('q_loss', 0):.2f}"
                postfix["a"] = f"{latest_metrics.get('alpha', 0):.3f}"
            pbar.set_postfix(**postfix)

        if args.eval_every_steps > 0 and step >= effective_starts and (step + 1) % args.eval_every_steps == 0:
            do_eval(step + 1)

        pbar.update(1)

    pbar.close()
    train_csv.close()
    goals_csv.close()
    if best_score == -float("inf"):
        do_eval(args.total_steps)
    eval_csv.close()

    torch.save(agent.state_dict(), out_path)
    print(f"Saved to {out_path}, best at {best_path}")


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser()
    for f in TrainArgs.__dataclass_fields__.values():
        p.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    a = p.parse_args()
    return TrainArgs(**{k: getattr(a, k) for k in TrainArgs.__dataclass_fields__.keys()})


if __name__ == "__main__":
    main(parse_args())
