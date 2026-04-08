"""Periodic retraining entry point.

Triggered by a GitHub Actions cron job (every 6 hours). Steps:

  1. Download the current Diffusion Policy checkpoint from HF Hub
  2. List unprocessed trajectory shards from HF Buckets
  3. Download them, filter for *winning* trajectories only
  4. Run a small behavior-cloning update on the winning trajectories
  5. Evaluate the candidate against the SAC expert
  6. Eval gate: only promote if the new model did not regress >5% in win rate
  7. If promoted, export to ONNX and push to HF Hub
  8. Mark the consumed shards as processed

Why BC on winners rather than DPPO:
  The browser only sends (obs, action, reward, done) per step. The
  diffusion sampling chain that produced each action is not recorded.
  Without the chain, DPPO has no per-denoising-step likelihood ratio
  to compute, so the update reduces to "resample under the current
  policy," which is a no-op. BC on winners is the honest fallback:
  treat the agent's own winning trajectories as a small demo dataset
  and imitate them, which genuinely improves the policy without
  needing any server-side rollouts.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from airhockey.eval import evaluate
from airhockey.export_onnx import export as export_onnx
from airhockey.policy import (
    DiffusionPolicyConfig,
    NoiseScheduler,
    UNet1D,
    diffusion_loss,
)
from airhockey.storage import HFBucketStore


# Gate: how much the new model is allowed to drop in win rate vs the
# previous one before we refuse to promote it.
EVAL_REGRESSION_TOLERANCE = 0.05
EVAL_EPISODES = 100


HF_REPO = os.environ.get("HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LOCAL_CKPT = "ckpt/dppo.pt"
LOCAL_ONNX = "onnx"
OPPONENT_CKPT = os.environ.get("AIRHOCKEY_OPPONENT_CKPT", "ckpt/sac_expert.pt")


def download_latest_checkpoint() -> None:
    """Pull the most recent Diffusion Policy checkpoint from HF Hub."""
    try:
        from huggingface_hub import hf_hub_download
        Path("ckpt").mkdir(exist_ok=True)
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="dppo.pt",
            token=HF_TOKEN,
            local_dir="ckpt",
        )
        print(f"Downloaded checkpoint from HF Hub: {path}")
    except Exception as e:
        print(f"HF Hub download failed: {e}")
        if not Path(LOCAL_CKPT).exists():
            raise RuntimeError(
                "No checkpoint available locally and HF Hub download failed."
            )


def upload_to_hub(local_ckpt: str, local_onnx_dir: str) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=local_ckpt,
        path_in_repo="dppo.pt",
        repo_id=HF_REPO,
        commit_message="Periodic retrain",
    )
    for f in Path(local_onnx_dir).glob("*"):
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=HF_REPO,
            commit_message="Periodic retrain (ONNX)",
        )
    print(f"Pushed model to {HF_REPO}")


def extract_winning_episodes(
    obs: np.ndarray,
    act: np.ndarray,
    rew: np.ndarray,
    done: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter for episodes that ended in a win (terminal reward > 0).
    Slice each winning episode into action-chunk windows of length
    `horizon` and return (obs_windows, action_chunks).

    A "win" is any terminal step with reward above a positive threshold
    (the env gives +10 on score, so anything > 5 is a safe threshold).
    """
    N = len(obs)
    # Find episode boundaries
    boundaries = [0]
    for i in range(N):
        if done[i]:
            boundaries.append(i + 1)
    if boundaries[-1] != N:
        boundaries.append(N)

    obs_windows: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ep_len = end - start
        if ep_len < horizon + 1:
            continue
        # Check if this episode was a win (last step has big reward)
        terminal_reward = rew[end - 1]
        if terminal_reward < 5.0:
            continue
        # Slice into (obs, action_chunk) windows
        for i in range(start, end - horizon):
            obs_windows.append(obs[i])
            act_chunks.append(act[i : i + horizon])

    if not obs_windows:
        return np.zeros((0, obs.shape[1]), dtype=np.float32), np.zeros((0, horizon, act.shape[1]), dtype=np.float32)
    return np.stack(obs_windows), np.stack(act_chunks)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    download_latest_checkpoint()

    ckpt = torch.load(LOCAL_CKPT, map_location=device)
    cfg = DiffusionPolicyConfig(**ckpt["config"])
    actor = UNet1D(cfg).to(device)
    actor.load_state_dict(ckpt["model"])
    scheduler = NoiseScheduler(cfg, device=device)
    optim = torch.optim.AdamW(actor.parameters(), lr=1e-5, weight_decay=1e-6)

    store = HFBucketStore()
    keys = store.list_unprocessed_shards()
    if not keys:
        print("No new shards to process. Exiting cleanly.")
        return
    print(f"Found {len(keys)} unprocessed shards")

    # Concatenate all shards
    obs_list, act_list, rew_list, done_list = [], [], [], []
    for k in keys:
        d = store.download_shard(k)
        obs_list.append(d["obs"])
        act_list.append(d["act"])
        rew_list.append(d["reward"])
        done_list.append(d["done"])
    obs = np.concatenate(obs_list).astype(np.float32)
    act = np.concatenate(act_list).astype(np.float32)
    rew = np.concatenate(rew_list).astype(np.float32)
    done = np.concatenate(done_list).astype(np.float32)
    print(f"Loaded {len(obs):,} transitions from {len(keys)} shards")

    # Filter for winning episodes only
    win_obs, win_chunks = extract_winning_episodes(obs, act, rew, done, cfg.horizon)
    print(f"Extracted {len(win_obs):,} winning-episode chunks")

    if len(win_obs) < 128:
        print("Not enough winning data to update meaningfully. Skipping.")
        for k in keys:
            store.mark_processed(k)
        return

    # ── Behavior-cloning update on winning chunks ─────────────
    obs_t = torch.from_numpy(win_obs).to(device)
    act_t = torch.from_numpy(win_chunks).to(device)
    dataset = TensorDataset(obs_t, act_t)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False)

    actor.train()
    EPOCHS = 3
    total_loss = 0.0
    n_batches = 0
    for epoch in range(EPOCHS):
        for batch_obs, batch_act in loader:
            loss = diffusion_loss(actor, scheduler, batch_obs, batch_act)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1
    actor.eval()
    avg_loss = total_loss / max(1, n_batches)
    print(f"BC-on-winners update: {n_batches} batches, avg loss {avg_loss:.5f}")

    # Save candidate
    Path("ckpt").mkdir(exist_ok=True)
    candidate_path = "ckpt/candidate.pt"
    torch.save(
        {"model": actor.state_dict(), "config": cfg.__dict__},
        candidate_path,
    )

    # ── Eval gate ─────────────────────────────────────────────
    opponent = OPPONENT_CKPT if Path(OPPONENT_CKPT).exists() else None
    print(f"Evaluating previous model ({LOCAL_CKPT}) against {opponent or 'no opponent'}…")
    prev_metrics = evaluate(LOCAL_CKPT, episodes=EVAL_EPISODES, seed=12345,
                            opponent_ckpt=opponent)
    print(f"Evaluating new candidate ({candidate_path})…")
    new_metrics = evaluate(candidate_path, episodes=EVAL_EPISODES, seed=12345,
                           opponent_ckpt=opponent)

    print(
        f"Eval gate: prev win {prev_metrics['win_rate']:.2f}, "
        f"new win {new_metrics['win_rate']:.2f}, "
        f"delta {new_metrics['win_rate'] - prev_metrics['win_rate']:+.2f}"
    )

    promoted = (
        new_metrics["win_rate"] >= prev_metrics["win_rate"] - EVAL_REGRESSION_TOLERANCE
    )

    if promoted:
        print("✓ Candidate passed eval gate, promoting to served checkpoint")
        shutil.copy(candidate_path, LOCAL_CKPT)
        Path(LOCAL_ONNX).mkdir(exist_ok=True)
        export_onnx(LOCAL_CKPT, LOCAL_ONNX)
        if HF_REPO and HF_TOKEN:
            upload_to_hub(LOCAL_CKPT, LOCAL_ONNX)
        for k in keys:
            store.mark_processed(k)
        print(f"Marked {len(keys)} shards as processed")
    else:
        print(
            "✗ Candidate FAILED eval gate "
            f"(win rate dropped by more than {EVAL_REGRESSION_TOLERANCE:.2f})"
        )
        print("Keeping the previous served checkpoint.")
        if HF_REPO and HF_TOKEN:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=HF_TOKEN)
                api.upload_file(
                    path_or_fileobj=candidate_path,
                    path_in_repo="rejected_candidate.pt",
                    repo_id=HF_REPO,
                    commit_message=(
                        f"Rejected candidate: win {new_metrics['win_rate']:.2f}"
                        f" vs prev {prev_metrics['win_rate']:.2f}"
                    ),
                )
            except Exception as e:
                print(f"Failed to upload rejected candidate: {e}")
        # IMPORTANT: do NOT mark shards as processed if we rejected.
        # They will be re-tried in the next cycle with a fresh policy.


if __name__ == "__main__":
    main()
