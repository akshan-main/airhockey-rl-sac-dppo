"""Diffusion Policy in MLX — mirrors policy.py (PyTorch) exactly.

Same architecture, same shapes, same hyperparameters. The only
difference is the framework. Checkpoints convert at boundaries via
convert_torch_to_mlx() / convert_mlx_to_torch().
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DiffusionPolicyConfig:
    obs_dim: int = 10
    act_dim: int = 2
    horizon: int = 8
    hidden: int = 128
    obs_embed_dim: int = 64
    n_train_diffusion_steps: int = 100
    n_inference_steps: int = 10
    beta_start: float = 1e-4
    beta_end: float = 0.02


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        half = self.dim // 2
        emb = math.log(10_000) / (half - 1)
        emb = mx.exp(mx.arange(half) * -emb)
        emb = x[:, None].astype(mx.float32) * emb[None, :]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.norm = nn.GroupNorm(8, out_ch)

    def __call__(self, x: mx.array) -> mx.array:
        # MLX Conv1d expects (B, L, C), same as our (B, H, ch)
        x = self.conv(x)
        # GroupNorm expects (B, ..., C) — MLX handles this
        x = self.norm(x)
        x = nn.mish(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.block1 = Conv1dBlock(in_ch, out_ch)
        self.block2 = Conv1dBlock(out_ch, out_ch)
        self.cond_linear = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        h = self.block1(x)
        c = nn.mish(cond)
        c = self.cond_linear(c)[:, None, :]  # (B, 1, out_ch)
        h = h + c
        h = self.block2(h)
        skip = self.skip(x) if self.skip is not None else x
        return h + skip


class UNet1D(nn.Module):
    def __init__(self, cfg: DiffusionPolicyConfig):
        super().__init__()
        self.cfg = cfg
        H, A, D = cfg.horizon, cfg.act_dim, cfg.hidden

        self.time_emb_sin = SinusoidalPosEmb(D)
        self.time_emb_l1 = nn.Linear(D, D * 2)
        self.time_emb_l2 = nn.Linear(D * 2, D)

        self.obs_emb_l1 = nn.Linear(cfg.obs_dim, cfg.obs_embed_dim)
        self.obs_emb_l2 = nn.Linear(cfg.obs_embed_dim, D)

        cond_dim = D
        self.down1 = ResidualBlock(A, D, cond_dim)
        self.down2 = ResidualBlock(D, D * 2, cond_dim)
        self.mid = ResidualBlock(D * 2, D * 2, cond_dim)
        self.up2 = ResidualBlock(D * 4, D, cond_dim)
        self.up1 = ResidualBlock(D * 2, D, cond_dim)
        self.out_conv = nn.Conv1d(D, A, 1)

    def __call__(self, a: mx.array, t: mx.array, obs: mx.array) -> mx.array:
        """a: (B,H,A), t: (B,), obs: (B,O) → noise pred (B,H,A)"""
        # Time embedding
        te = self.time_emb_sin(t)
        te = nn.mish(self.time_emb_l1(te))
        te = self.time_emb_l2(te)
        # Obs embedding
        oe = nn.mish(self.obs_emb_l1(obs))
        oe = self.obs_emb_l2(oe)
        cond = te + oe

        # MLX Conv1d: (B, L, C) — a is already (B, H, A) which is (B, L, C)
        x = a
        d1 = self.down1(x, cond)
        d2 = self.down2(d1, cond)
        m = self.mid(d2, cond)
        u2 = self.up2(mx.concatenate([m, d2], axis=-1), cond)
        u1 = self.up1(mx.concatenate([u2, d1], axis=-1), cond)
        out = self.out_conv(u1)
        return out


class NoiseScheduler:
    def __init__(self, cfg: DiffusionPolicyConfig):
        T = cfg.n_train_diffusion_steps
        betas = mx.linspace(cfg.beta_start, cfg.beta_end, T)
        alphas = 1.0 - betas
        alpha_bars = mx.cumprod(alphas)
        self.cfg = cfg
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def add_noise(self, a0: mx.array, t: mx.array, noise: mx.array) -> mx.array:
        ab = self.alpha_bars[t].reshape(-1, 1, 1)
        return a0 * mx.sqrt(ab) + noise * mx.sqrt(1.0 - ab)

    def ddim_sample(self, model: UNet1D, obs: mx.array,
                    n_steps: int | None = None) -> mx.array:
        cfg = self.cfg
        n_steps = n_steps or cfg.n_inference_steps
        T = cfg.n_train_diffusion_steps
        ts = mx.linspace(T - 1, 0, n_steps + 1).astype(mx.int32)
        B = obs.shape[0]
        a = mx.random.normal((B, cfg.horizon, cfg.act_dim))
        for i in range(n_steps):
            t_now = int(ts[i].item())
            t_next = int(ts[i + 1].item())
            t_batch = mx.full((B,), t_now, dtype=mx.int32)
            eps = model(a, t_batch, obs)
            ab_now = self.alpha_bars[t_now]
            ab_next = self.alpha_bars[t_next] if t_next >= 0 else mx.array(1.0)
            a0_pred = (a - mx.sqrt(1.0 - ab_now) * eps) / mx.sqrt(ab_now)
            a0_pred = mx.clip(a0_pred, -1.5, 1.5)
            a = mx.sqrt(ab_next) * a0_pred + mx.sqrt(1.0 - ab_next) * eps
        return mx.clip(a, -1.0, 1.0)


def diffusion_loss(model: UNet1D, scheduler: NoiseScheduler,
                   obs: mx.array, a0: mx.array) -> mx.array:
    B = obs.shape[0]
    T = scheduler.cfg.n_train_diffusion_steps
    t = mx.random.randint(0, T, (B,))
    noise = mx.random.normal(a0.shape)
    noised = scheduler.add_noise(a0, t, noise)
    eps_pred = model(noised, t, obs)
    return mx.mean((eps_pred - noise) ** 2)


# ── Checkpoint conversion ─────────────────────────────────────
def convert_torch_to_mlx(torch_state: dict, cfg: DiffusionPolicyConfig) -> dict:
    """Convert a PyTorch UNet1D state_dict to MLX parameter dict.

    The main difference: PyTorch Conv1d weight is (out, in, kernel),
    MLX Conv1d weight is (out, kernel, in). Need to transpose axes 1,2.
    """
    import numpy as np
    mlx_params = {}
    for k, v in torch_state.items():
        arr = v.cpu().numpy()
        # Conv1d weights: PyTorch (out, in, K) → MLX (out, K, in)
        if "conv" in k.lower() and "weight" in k and arr.ndim == 3:
            arr = arr.transpose(0, 2, 1)
        mlx_params[k] = mx.array(arr)
    return mlx_params


def _flatten_dict(d, prefix=""):
    """Flatten a nested dict into dot-separated keys."""
    items = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        elif isinstance(v, mx.array):
            items[key] = v
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.update(_flatten_dict(item, f"{key}.{i}"))
                elif isinstance(item, mx.array):
                    items[f"{key}.{i}"] = item
    return items


def convert_mlx_to_torch(mlx_params: dict) -> dict:
    """Convert MLX parameters back to PyTorch state_dict format.

    Handles the key name mapping between MLX and PyTorch architectures:
    - MLX Conv1dBlock has .conv and .norm; PyTorch has .block.0 and .block.1
    - MLX ResidualBlock has .cond_linear; PyTorch has .cond_mlp.1
    - MLX UNet1D has .time_emb_l1/l2, .obs_emb_l1/l2, .out_conv;
      PyTorch has .time_emb.1/.3, .obs_emb.0/.2, .out
    """
    import torch
    import numpy as np
    flat = _flatten_dict(mlx_params)

    # Build the key mapping from MLX names to PyTorch names
    KEY_MAP = {
        "time_emb_l1": "time_emb.1",
        "time_emb_l2": "time_emb.3",
        "obs_emb_l1": "obs_emb.0",
        "obs_emb_l2": "obs_emb.2",
        "out_conv": "out",
        "cond_linear": "cond_mlp.1",
    }
    # Conv1dBlock: .conv → .block.0, .norm → .block.1
    BLOCK_MAP = {
        ".conv.": ".block.0.",
        ".norm.": ".block.1.",
    }

    torch_state = {}
    for k, v in flat.items():
        tk = k
        # Apply key mappings
        for mlx_name, torch_name in KEY_MAP.items():
            tk = tk.replace(mlx_name, torch_name)
        for mlx_pat, torch_pat in BLOCK_MAP.items():
            tk = tk.replace(mlx_pat, torch_pat)

        arr = np.array(v)
        # Conv1d weight: MLX (out, K, in) → PyTorch (out, in, K)
        if "weight" in tk and arr.ndim == 3:
            arr = arr.transpose(0, 2, 1)
        torch_state[tk] = torch.from_numpy(arr.copy())
    return torch_state
