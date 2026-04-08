"""Stage 2 — Behavior cloning of the SAC expert into a Diffusion Policy.

Loads the demonstration dataset produced by `airhockey.collect` from a
trained SAC checkpoint, slices it into (obs, action_chunk) windows of
horizon H, and trains the noise predictor with the standard DDPM loss.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from airhockey.policy import (
    DiffusionPolicyConfig,
    NoiseScheduler,
    UNet1D,
    diffusion_loss,
)


class ChunkDataset(Dataset):
    """Slices contiguous (obs, action_chunk) windows from raw demos.
    For each step i where the next H steps are within the same episode,
    yields (obs[i], act[i:i+H])."""

    def __init__(self, npz_path: str, horizon: int):
        d = np.load(npz_path)
        obs = d["obs"]                # (N, 10)
        act = d["act"]                # (N, 2)
        episode = d["episode"]        # (N,)
        timestep = d["timestep"]      # (N,)
        N = len(obs)
        # Build a list of valid window starts (next H all in the same episode)
        valid: list[int] = []
        for i in range(N - horizon):
            if episode[i + horizon - 1] == episode[i]:
                valid.append(i)
        self.obs = torch.from_numpy(obs)
        self.act = torch.from_numpy(act)
        self.starts = np.array(valid, dtype=np.int64)
        self.horizon = horizon
        print(f"Loaded {len(self.starts):,} windows from {N:,} transitions "
              f"({len(self.starts) / N:.1%} valid).")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        i = int(self.starts[idx])
        return self.obs[i], self.act[i : i + self.horizon]


def train(
    data_path: str,
    out_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    horizon: int,
    device: str,
    init_path: str | None = None,
    val_frac: float = 0.1,
):
    cfg = DiffusionPolicyConfig(horizon=horizon)
    full_ds = ChunkDataset(data_path, horizon=horizon)

    # Hold-out a deterministic validation split to detect overfitting.
    n_total = len(full_ds)
    n_val = max(1, int(val_frac * n_total))
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_ds = torch.utils.data.Subset(full_ds, train_idx.tolist())
    val_ds = torch.utils.data.Subset(full_ds, val_idx.tolist())
    print(f"Split: {len(train_ds):,} train / {len(val_ds):,} val")

    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False,
        pin_memory=(device == "cuda"),
    )

    model = UNet1D(cfg).to(device)

    # Continue-training: warm-start from an existing diffusion policy
    # checkpoint. The retrain loop relies on this so each cycle's BC
    # update doesn't forget everything from the previous cycle.
    if init_path is not None:
        init_ckpt = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt["model"])
        print(f"Warm-started from {init_path}")

    scheduler = NoiseScheduler(cfg, device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Diffusion Policy: {n_params/1e6:.2f}M params")

    for epoch in range(epochs):
        model.train()
        losses = []
        for obs, act_chunk in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            obs = obs.to(device, non_blocking=True)
            act_chunk = act_chunk.to(device, non_blocking=True)
            loss = diffusion_loss(model, scheduler, obs, act_chunk)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs, act_chunk in val_loader:
                obs = obs.to(device, non_blocking=True)
                act_chunk = act_chunk.to(device, non_blocking=True)
                vloss = diffusion_loss(model, scheduler, obs, act_chunk)
                val_losses.append(vloss.item())
        print(
            f"  train_loss={np.mean(losses):.5f}  "
            f"val_loss={np.mean(val_losses):.5f}  "
            f"lr={sched.get_last_lr()[0]:.2e}"
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg.__dict__,
        },
        out,
    )
    print(f"Saved checkpoint to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/demos.npz")
    p.add_argument("--out", type=str, default="ckpt/bc.pt")
    p.add_argument("--init", type=str, default=None,
                   help="Warm-start from an existing diffusion policy checkpoint.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train(
        args.data, args.out, args.epochs, args.batch_size, args.lr,
        args.horizon, args.device,
        init_path=args.init, val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
