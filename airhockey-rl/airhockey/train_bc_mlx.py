"""BC training with MLX — faithful to official DPPO/Chi et al. references.

Fixes applied from audit against irom-princeton/dppo and
real-stanford/diffusion_policy:
  - EMA with decay 0.995, used at inference (not raw weights)
  - Data normalization (zero-mean, unit-variance on obs + act)
  - LR 1e-3 (DPPO reference, 10x higher than our previous 1e-4)
  - AdamW betas (0.95, 0.999) per Chi et al.
  - Cosine beta schedule (via NoiseScheduler)
  - MLP architecture (via DiffusionMLP)
  - Intermediate clip ±1.0 (clip_sample=True behavior)
  - Target loss: 0.02-0.08 (if >0.15 after 200 epochs = underfitting)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from airhockey.policy_mlx import (
    DiffusionPolicyConfig,
    DiffusionMLP,
    EMA,
    NoiseScheduler,
    Normalizer,
    convert_mlx_to_torch,
    convert_torch_to_mlx,
    diffusion_loss,
)


class ChunkDataset:
    """Precomputes all valid (stacked_obs, action_chunk) pairs.

    For each valid window starting at position i:
      obs_window = obs[i-n_obs+1 : i+1] flattened → (n_obs * obs_dim,)
      act_chunk  = act[i : i+horizon]              → (horizon, act_dim)

    Vectorized validity + precomputed arrays. Zero Python loops at
    load or sample time.
    """
    def __init__(self, npz_path: str, horizon: int, n_obs_steps: int = 2):
        d = np.load(npz_path)
        obs = d["obs"].astype(np.float32)
        act = d["act"].astype(np.float32)
        episode = d["episode"]
        timestep = d["timestep"]
        N = len(obs)
        self.n_obs_steps = n_obs_steps

        # Valid window: need n_obs_steps-1 obs before + horizon acts after.
        # Starting position i must have:
        #   - ep[i-n_obs+1] == ep[i+horizon-1]  (all same episode)
        #   - ts[i+horizon-1] == ts[i] + horizon - 1  (contiguous actions)
        #   - i >= n_obs_steps - 1  (enough history)
        start_min = n_obs_steps - 1
        idx = np.arange(start_min, N - horizon + 1)
        end = idx + horizon - 1
        hist_start = idx - (n_obs_steps - 1)

        same_ep = (episode[hist_start] == episode[end])
        contiguous = (timestep[end] == timestep[idx] + horizon - 1)
        valid_mask = same_ep & contiguous
        self.starts = idx[valid_mask].astype(np.int64)

        # Precompute stacked obs: (N_valid, n_obs_steps * obs_dim)
        obs_idx = self.starts[:, None] - np.arange(n_obs_steps - 1, -1, -1)[None, :]
        # obs_idx shape: (N_valid, n_obs_steps) — indices for obs history
        obs_windows = obs[obs_idx]  # (N_valid, n_obs_steps, obs_dim)
        self.obs_stacked = obs_windows.reshape(len(self.starts), -1)  # (N_valid, n_obs * obs_dim)

        # Precompute action chunks: (N_valid, horizon, act_dim)
        chunk_idx = self.starts[:, None] + np.arange(horizon)[None, :]
        self.act_chunks = act[chunk_idx]

        # Keep raw obs for normalization fitting
        self.obs_raw = obs
        self.act_raw = act
        self.horizon = horizon
        print(f"Loaded {len(self.starts):,} windows from {N:,} transitions "
              f"({len(self.starts) / N:.1%} valid, n_obs_steps={n_obs_steps}).")

    def __len__(self):
        return len(self.starts)


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
    ds = ChunkDataset(data_path, horizon=horizon, n_obs_steps=cfg.n_obs_steps)

    # ── Data normalization ────────────────────────────────────
    normalizer = Normalizer()
    normalizer.fit(ds.obs_raw, ds.act_raw)
    print(f"Normalizer: obs_mean[:3]={normalizer.obs_mean[:3]}  act_mean={normalizer.act_mean}")

    # Normalize stacked obs: each frame in the stack gets the same norm
    # obs_stacked is (N, n_obs * obs_dim) — normalize per-obs-dim (tiled)
    obs_mean_tiled = np.tile(normalizer.obs_mean, cfg.n_obs_steps)
    obs_std_tiled = np.tile(normalizer.obs_std, cfg.n_obs_steps)
    ds.obs_stacked = (ds.obs_stacked - obs_mean_tiled) / obs_std_tiled
    ds.act_chunks = (ds.act_chunks - normalizer.act_mean) / normalizer.act_std
    print(f"Normalized: obs range [{ds.obs_stacked.min():.2f}, {ds.obs_stacked.max():.2f}]  "
          f"act range [{ds.act_chunks.min():.2f}, {ds.act_chunks.max():.2f}]")

    # ── Val split (indices into ds.starts / ds.act_chunks) ────
    n_total = len(ds)
    n_val = max(1, int(val_frac * n_total))
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_total)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"Split: {len(train_idx):,} train / {len(val_idx):,} val")

    # ── Model ─────────────────────────────────────────────────
    model = DiffusionMLP(cfg)
    scheduler = NoiseScheduler(cfg)

    if init_path is not None:
        import torch
        ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        mlx_params = convert_torch_to_mlx(ckpt["model"], cfg)
        model.load_weights(list(mlx_params.items()))
        print(f"Warm-started from {init_path}")

    def _count(d):
        if isinstance(d, mx.array): return d.size
        if isinstance(d, dict): return sum(_count(v) for v in d.values())
        if isinstance(d, list): return sum(_count(v) for v in d)
        return 0
    n_params = _count(model.parameters())
    print(f"DiffusionMLP (MLX): {n_params / 1e6:.2f}M params")

    # ── EMA ───────────────────────────────────────────────────
    ema = EMA(model, decay=cfg.ema_decay)
    ema_start_epoch = 20  # start updating EMA after epoch 20

    # ── Optimizer: AdamW with betas=(0.95, 0.999) per Chi et al.
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=1e-6,
                            betas=[0.95, 0.999])

    def loss_fn(model, obs, act_chunk):
        return diffusion_loss(model, scheduler, obs, act_chunk)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    steps_per_epoch = len(train_idx) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = 500
    print(f"Total gradient steps: {total_steps:,}, warmup: {warmup_steps}")

    def lr_at(step_global):
        """Linear warmup then cosine decay to lr/10."""
        if step_global < warmup_steps:
            return lr * (step_global + 1) / warmup_steps
        progress = (step_global - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        min_lr = lr * 0.1
        return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    global_step = 0
    for epoch in range(epochs):
        np.random.shuffle(train_idx)
        train_losses = []

        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}"):
            batch_idx = train_idx[step * batch_size:(step + 1) * batch_size]
            obs_batch = mx.array(ds.obs_stacked[batch_idx])
            act_batch = mx.array(ds.act_chunks[batch_idx])

            # Set LR
            optimizer.learning_rate = lr_at(global_step)

            loss, grads = loss_and_grad(model, obs_batch, act_batch)
            # Gradient clipping (max_norm=1.0, matching reference)
            grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            train_losses.append(loss.item())
            global_step += 1

            # EMA update every 10 batches after ema_start_epoch
            if epoch >= ema_start_epoch and step % 10 == 0:
                ema.update(model)
                mx.eval(list(ema.shadow.values()))

        # Val loss
        val_losses = []
        for vs in range(0, len(val_idx), batch_size):
            ve = min(vs + batch_size, len(val_idx))
            vb = val_idx[vs:ve]
            vobs = mx.array(ds.obs_stacked[vb])
            vact = mx.array(ds.act_chunks[vb])
            vloss = diffusion_loss(model, scheduler, vobs, vact)
            mx.eval(vloss)
            val_losses.append(vloss.item())

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        print(f"  train_loss={tl:.5f}  val_loss={vl:.5f}  lr={optimizer.learning_rate:.2e}")

        if epoch == epochs - 1 and tl > 0.15:
            print(f"  WARNING: final train_loss {tl:.4f} > 0.15 — model is underfitting")

    # ── Apply EMA weights for saving ──────────────────────────
    ema.apply_to(model)
    print("Applied EMA weights to model")

    # ── Save as PyTorch checkpoint ────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch_state = convert_mlx_to_torch(model.parameters())
    import torch
    torch.save({
        "model": torch_state,
        "config": cfg.__dict__,
        "normalizer": normalizer.state_dict(),
    }, out)
    print(f"Saved checkpoint to {out} (PyTorch format, with normalizer)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/demos.npz")
    p.add_argument("--out", type=str, default="ckpt/bc.pt")
    p.add_argument("--init", type=str, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--val-frac", type=float, default=0.1)
    args = p.parse_args()
    train(args.data, args.out, args.epochs, args.batch_size, args.lr,
          args.horizon, init_path=args.init, val_frac=args.val_frac)


if __name__ == "__main__":
    main()
