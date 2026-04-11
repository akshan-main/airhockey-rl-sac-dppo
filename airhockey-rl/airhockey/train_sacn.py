"""SACn training — SAC with n-step returns (Łyskawa et al. 2025).

Same env/opponent/demo/good-buffer setup as train_sac.py, but the
critic uses n-step returns with importance sampling and quantile
clipping instead of 1-step TD.
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
from airhockey.sac import NStepReplayBuffer, ReplayBuffer, SACAgent, SACConfig

SCRIPTED_NOISE_STD = 0.05
SACN_N = 8
SACN_QB = 0.75


def mirror_obs(obs):
    o = obs.copy()
    o[0] = 1.0 - o[0]
    o[4] = 1.0 - o[4]
    o[8] = 1.0 - o[8]
    o[2] = -o[2]
    o[6] = -o[6]
    return o


def mirror_action(a):
    out = a.copy()
    out[0] = -out[0]
    return out


@dataclass
class TrainArgs:
    out: str = "ckpt_v2/sacn_expert.pt"
    total_steps: int = 2_000_000
    batch_size: int = 256
    buffer_size: int = 500_000
    nstep: int = 8
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
    N = args.nstep

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

    # N-step buffer for SACn
    nstep_buf = NStepReplayBuffer(args.buffer_size, cfg.obs_dim, cfg.act_dim, N)
    # Regular buffer for good (winning) episodes — still 1-step for mixed sampling
    good_buffer = ReplayBuffer(500_000, cfg.obs_dim, cfg.act_dim)

    def add_snapshot():
        snap = copy.deepcopy(agent.actor).eval()
        for p in snap.parameters():
            p.requires_grad = False
        opponent_league.append(snap)

    # ── Collect demos ─────────────────────────────────────────
    if args.demo_episodes > 0:
        demo_rng = random.Random(args.seed + 333)
        demo_wins = 0
        for ep in tqdm(range(args.demo_episodes), desc="Demos"):
            opp = noisy_attacker if demo_rng.random() < 0.5 else noisy_tracker
            env.opponent = opp
            obs_d, _ = env.reset(seed=args.seed + 50_000 + ep)

            # Collect full episode
            ep_obs, ep_act, ep_rew, ep_done, ep_logp = [], [], [], [], []
            last_event = ""
            while True:
                action_d = scripted_attacker(obs_d.astype(np.float32))
                next_obs_d, rew_d, term_d, trunc_d, info_d = env.step(action_d)

                # Store log prob of action under current policy (for IS)
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs_d.astype(np.float32)).unsqueeze(0).to(device)
                    mean, log_std = agent.actor(obs_t)
                    std = log_std.exp()
                    pre = torch.atanh(torch.from_numpy(action_d).float().unsqueeze(0).clamp(-0.999, 0.999))
                    dist = torch.distributions.Normal(mean, std)
                    lp = (dist.log_prob(pre) - torch.log(1 - torch.from_numpy(action_d).float().unsqueeze(0).pow(2) + 1e-6)).sum(-1)
                    logp_val = lp.item()

                ep_obs.append(obs_d.copy())
                ep_act.append(action_d.copy())
                ep_rew.append(rew_d)
                ep_done.append(float(term_d))
                ep_logp.append(logp_val)

                evt = info_d.get("event", "")
                if evt.startswith("goal"):
                    last_event = evt
                obs_d = next_obs_d
                if term_d or trunc_d:
                    # Append final obs
                    ep_obs.append(next_obs_d.copy())
                    ep_act.append(np.zeros(cfg.act_dim, dtype=np.float32))
                    break

            # Build n-step sequences from episode
            T = len(ep_rew)
            for t in range(T - N):
                obs_seq = np.stack(ep_obs[t:t + N + 1])
                act_seq = np.stack(ep_act[t:t + N + 1])
                rew_seq = np.array(ep_rew[t:t + N], dtype=np.float32)
                done_seq = np.array(ep_done[t:t + N], dtype=np.float32)
                logp_seq = np.array(ep_logp[t:t + N], dtype=np.float32)
                nstep_buf.push(obs_seq, act_seq, rew_seq, done_seq, logp_seq)

            # Winning episodes → good buffer (1-step)
            if last_event == "goal_bot":
                demo_wins += 1
                for i in range(T):
                    good_buffer.push(ep_obs[i], ep_act[i], ep_rew[i], ep_obs[i + 1] if i + 1 < len(ep_obs) else ep_obs[i], ep_done[i] > 0.5)

        env.opponent = league_opponent
        print(f"Demos: {demo_wins}W / {args.demo_episodes} ep, nstep_buf={nstep_buf.size:,}, good={good_buffer.size:,}")

    # ── BC pre-train ──────────────────────────────────────────
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

    # ── Output ────────────────────────────────────────────────
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_path.with_suffix(".best.pt")
    train_csv = open(out_dir / "train.csv", "w", newline="")
    eval_csv = open(out_dir / "eval.csv", "w", newline="")
    tw = csv.writer(train_csv)
    ew = csv.writer(eval_csv)
    tw.writerow(["step", "ep_return_mean", "ep_length_mean", "q_loss", "pi_loss", "alpha", "entropy"])
    ew.writerow(["step", "opponent", "win_rate", "loss_rate", "draw_rate", "mean_return", "mean_length", "n_episodes"])

    best_score = -float("inf")
    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_length = 0
    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    # Rolling window for n-step sequence building
    ep_obs_win: list[np.ndarray] = [obs.copy()]
    ep_act_win: list[np.ndarray] = []
    ep_rew_win: list[float] = []
    ep_done_win: list[float] = []
    ep_logp_win: list[float] = []
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

    pbar = tqdm(total=args.total_steps, desc="SACn")

    for step in range(args.total_steps):
        action = agent.act(obs, deterministic=False)
        action_arr = np.asarray(action, dtype=np.float32)

        # Log prob under current policy for importance sampling later
        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(device)
            mean, log_std = agent.actor(obs_t)
            std = log_std.exp()
            pre = torch.atanh(torch.from_numpy(action_arr).float().unsqueeze(0).clamp(-0.999, 0.999))
            dist = torch.distributions.Normal(mean, std)
            lp = (dist.log_prob(pre) - torch.log(1 - torch.from_numpy(action_arr).float().unsqueeze(0).pow(2) + 1e-6)).sum(-1)
            logp_val = lp.item()

        next_obs, reward, term, trunc, info = env.step(action)
        done = bool(term)

        ep_act_win.append(action_arr.copy())
        ep_rew_win.append(reward)
        ep_done_win.append(float(done))
        ep_logp_win.append(logp_val)
        ep_obs_win.append(next_obs.copy())
        ep_transitions.append((obs.copy(), action_arr.copy(), reward, next_obs.copy(), done))

        # Build n-step sequence when we have enough
        if len(ep_rew_win) >= N:
            idx = len(ep_rew_win) - N
            obs_seq = np.stack(ep_obs_win[idx:idx + N + 1])
            act_seq_list = ep_act_win[idx:idx + N] + [np.zeros(cfg.act_dim, dtype=np.float32)]
            act_seq = np.stack(act_seq_list)
            rew_seq = np.array(ep_rew_win[idx:idx + N], dtype=np.float32)
            done_seq = np.array(ep_done_win[idx:idx + N], dtype=np.float32)
            logp_seq = np.array(ep_logp_win[idx:idx + N], dtype=np.float32)
            nstep_buf.push(obs_seq, act_seq, rew_seq, done_seq, logp_seq)

        ep_return += reward
        ep_length += 1

        if term or trunc:
            ep_returns.append(ep_return)
            ep_lengths.append(ep_length)

            if info.get("event") == "goal_bot":
                for o, a, r, no, d in ep_transitions:
                    good_buffer.push(o, a, r, no, d)

            ep_return = 0.0
            ep_length = 0
            ep_transitions = []
            obs, _ = env.reset()
            ep_obs_win = [obs.copy()]
            ep_act_win = []
            ep_rew_win = []
            ep_done_win = []
            ep_logp_win = []
        else:
            obs = next_obs

        # Update
        if nstep_buf.size >= args.batch_size:
            for _ in range(args.updates_per_step):
                batch = nstep_buf.sample(args.batch_size, device)
                latest_metrics = agent.update_sacn(batch, n=N, q_b=SACN_QB)

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
            pbar.set_postfix(ret=f"{avg_ret:+.2f}", len=f"{avg_len:.0f}",
                             qL=f"{latest_metrics.get('q_loss', 0):.2f}",
                             a=f"{latest_metrics.get('alpha', 0):.3f}")

        if args.eval_every_steps > 0 and (step + 1) % args.eval_every_steps == 0:
            do_eval(step + 1)

        pbar.update(1)

    pbar.close()
    train_csv.close()
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
