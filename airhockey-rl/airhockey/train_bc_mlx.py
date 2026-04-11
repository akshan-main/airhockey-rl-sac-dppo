"""BC training with MLX — mirrors train_bc.py exactly."""
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from airhockey.policy_mlx import (
    DiffusionPolicyConfig,
    NoiseScheduler,
    UNet1D,
    diffusion_loss,
    convert_mlx_to_torch,
)


class ChunkDataset:
    def __init__(self, npz_path: str, horizon: int):
        d = np.load(npz_path)
        obs = d["obs"]
        act = d["act"]
        episode = d["episode"]
        N = len(obs)
        valid = []
        for i in range(N - horizon):
            if episode[i + horizon - 1] == episode[i]:
                valid.append(i)
        self.obs = obs
        self.act = act
        self.starts = np.array(valid, dtype=np.int64)
        self.horizon = horizon
        print(f"Loaded {len(self.starts):,} windows from {N:,} transitions "
              f"({len(self.starts) / N:.1%} valid).")

    def __len__(self):
        return len(self.starts)

    def sample_batch(self, batch_size: int, rng_key=None):
        idx = np.random.randint(0, len(self.starts), size=batch_size)
        starts = self.starts[idx]
        obs_batch = self.obs[starts]
        act_batch = np.stack([self.act[s:s + self.horizon] for s in starts])
        return mx.array(obs_batch), mx.array(act_batch)


def train(
    data_path: str,
    out_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    horizon: int,
    init_path: str | None = None,
    val_frac: float = 0.1,
):
    cfg = DiffusionPolicyConfig(horizon=horizon)
    ds = ChunkDataset(data_path, horizon=horizon)

    # Val split
    n_total = len(ds)
    n_val = max(1, int(val_frac * n_total))
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_total)
    val_starts = ds.starts[perm[:n_val]]
    train_starts = ds.starts[perm[n_val:]]
    print(f"Split: {len(train_starts):,} train / {len(val_starts):,} val")

    model = UNet1D(cfg)
    scheduler = NoiseScheduler(cfg)

    if init_path is not None:
        import torch
        ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        from airhockey.policy_mlx import convert_torch_to_mlx
        mlx_params = convert_torch_to_mlx(ckpt["model"], cfg)
        model.load_weights(list(mlx_params.items()))
        print(f"Warm-started from {init_path}")

    def _count(d):
        if isinstance(d, mx.array): return d.size
        if isinstance(d, dict): return sum(_count(v) for v in d.values())
        if isinstance(d, list): return sum(_count(v) for v in d)
        return 0
    n_params = _count(model.parameters())
    print(f"Diffusion Policy (MLX): {n_params / 1e6:.2f}M params")

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-6)

    def loss_fn(model, obs, act_chunk):
        return diffusion_loss(model, scheduler, obs, act_chunk)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    steps_per_epoch = len(train_starts) // batch_size

    for epoch in range(epochs):
        # Shuffle train indices
        np.random.shuffle(train_starts)
        train_losses = []

        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}"):
            idx = train_starts[step * batch_size:(step + 1) * batch_size]
            obs_batch = mx.array(ds.obs[idx])
            act_batch = mx.array(np.stack([ds.act[s:s + horizon] for s in idx]))

            loss, grads = loss_and_grad(model, obs_batch, act_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            train_losses.append(loss.item())

        # Val loss
        val_losses = []
        for vs in range(0, len(val_starts), batch_size):
            ve = min(vs + batch_size, len(val_starts))
            vidx = val_starts[vs:ve]
            vobs = mx.array(ds.obs[vidx])
            vact = mx.array(np.stack([ds.act[s:s + horizon] for s in vidx]))
            vloss = diffusion_loss(model, scheduler, vobs, vact)
            mx.eval(vloss)
            val_losses.append(vloss.item())

        # LR schedule (cosine)
        progress = (epoch + 1) / epochs
        new_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        optimizer.learning_rate = new_lr

        print(f"  train_loss={np.mean(train_losses):.5f}  "
              f"val_loss={np.mean(val_losses):.5f}  "
              f"lr={new_lr:.2e}")

    # Save as PyTorch checkpoint for compatibility with downstream stages
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch_state = convert_mlx_to_torch(model.parameters())
    import torch
    torch.save(
        {"model": torch_state, "config": cfg.__dict__},
        out,
    )
    print(f"Saved checkpoint to {out} (PyTorch format)")


import math


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/demos.npz")
    p.add_argument("--out", type=str, default="ckpt/bc.pt")
    p.add_argument("--init", type=str, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--val-frac", type=float, default=0.1)
    args = p.parse_args()
    train(args.data, args.out, args.epochs, args.batch_size, args.lr,
          args.horizon, init_path=args.init, val_frac=args.val_frac)


if __name__ == "__main__":
    main()
