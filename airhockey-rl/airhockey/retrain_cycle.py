"""Periodic retrain invoked by the GitHub Actions cron.

Steps:
  1. Download the current Diffusion Policy checkpoint from HF Hub
  2. List unprocessed trajectory shards from HF Buckets
  3. Filter for winning episodes (terminal reward > 5)
  4. Run a BC update (DDPM loss) on the winning chunks
  5. Evaluate the candidate against the SAC expert
  6. Promote only if win rate did not regress by more than
     EVAL_REGRESSION_TOLERANCE; on promotion, export ONNX and push
     to HF Hub
  7. Mark consumed shards as processed (HF Buckets lifecycle deletes
     them after N days)

The browser only POSTs (obs, action, reward, done), not the full
denoising chain, so a DPPO-style update is not possible here. BC on
winners is what we can do with the data we have.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch
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


# Maximum allowed win-rate drop (vs previous model) before the gate
# refuses to promote the candidate.
EVAL_REGRESSION_TOLERANCE = 0.05
EVAL_EPISODES = 100


HF_REPO = os.environ.get("HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
LOCAL_CKPT = "ckpt/dppo.pt"
LOCAL_ONNX = "onnx"
OPPONENT_CKPT = os.environ.get("AIRHOCKEY_OPPONENT_CKPT", "ckpt/sac_expert.best.pt")


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
    """Slice winning episodes into (obs, action_chunk) windows of length
    `horizon`. A "win" is a terminal step with reward > 5 (the env gives
    +10 on score).
    """
    N = len(obs)
    boundaries = [0]
    for i in range(N):
        if done[i]:
            boundaries.append(i + 1)
    if boundaries[-1] != N:
        boundaries.append(N)

    obs_windows: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < horizon + 1:
            continue
        if rew[end - 1] < 5.0:
            continue
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
        print("No new shards to process. Exiting.")
        return
    print(f"Found {len(keys)} unprocessed shards")

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

    win_obs, win_chunks = extract_winning_episodes(obs, act, rew, done, cfg.horizon)
    print(f"Extracted {len(win_obs):,} winning-episode chunks")

    if len(win_obs) < 128:
        print("Not enough winning data to update meaningfully. Skipping.")
        for k in keys:
            store.mark_processed(k)
        return

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

    Path("ckpt").mkdir(exist_ok=True)
    candidate_path = "ckpt/candidate.pt"
    torch.save(
        {"model": actor.state_dict(), "config": cfg.__dict__},
        candidate_path,
    )

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
        print("Candidate passed eval gate, promoting.")
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
            f"Candidate failed eval gate "
            f"(win rate dropped by more than {EVAL_REGRESSION_TOLERANCE:.2f}). "
            f"Keeping previous served checkpoint."
        )
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
        # Shards are NOT marked processed when the gate rejects, so the
        # next cycle will see them again.


if __name__ == "__main__":
    main()
