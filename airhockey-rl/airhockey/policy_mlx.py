"""Diffusion Policy in MLX — MLP architecture with cosine schedule.

Matches the official irom-princeton/dppo implementation:
  - DiffusionMLP (not Conv1d UNet) for state-based tasks
  - Cosine beta schedule (not linear)
  - Standard DDPM epsilon-prediction loss
  - DDIM sampling for inference

Checkpoints convert between MLX and PyTorch at save/load boundaries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class DiffusionPolicyConfig:
    obs_dim: int = 10
    act_dim: int = 2
    horizon: int = 8
    hidden: int = 256
    n_layers: int = 3
    time_dim: int = 32
    n_train_diffusion_steps: int = 100
    n_inference_steps: int = 10
    ema_decay: float = 0.995


# ── Cosine beta schedule (Nichol & Dhariwal 2021) ─────────────
def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999).astype(np.float32)


# ── Sinusoidal time embedding ─────────────────────────────────
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


# ── Diffusion MLP with FiLM conditioning (state-based) ───────
class DiffusionMLP(nn.Module):
    """MLP noise predictor with FiLM (scale+bias) conditioning from
    time embedding and obs. Matches Chi et al. cond_predict_scale=True.

    Input: flattened action chunk (B, H*A), time t, obs
    Output: predicted noise (B, H, A)

    Each hidden layer is modulated by an affine transform derived
    from (time_emb, obs): h <- scale * h + bias, where scale/bias
    are learned linear projections of the concatenated conditioning.
    This is strictly more expressive than additive concatenation.
    """
    def __init__(self, cfg: DiffusionPolicyConfig):
        super().__init__()
        self.cfg = cfg
        H, A = cfg.horizon, cfg.act_dim
        input_dim = H * A
        time_dim = cfg.time_dim
        cond_dim = cfg.obs_dim
        hidden = cfg.hidden
        n_layers = cfg.n_layers

        # Time embedding
        self.time_emb = SinusoidalPosEmb(time_dim)
        self.time_mlp1 = nn.Linear(time_dim, time_dim * 4)
        self.time_mlp2 = nn.Linear(time_dim * 4, time_dim)

        # Conditioning encoder: (time_dim + obs_dim) -> conditioning vec
        self.cond_encoder = nn.Linear(time_dim + cond_dim, hidden)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden)

        # FiLM layers: each produces (scale, bias) per hidden unit
        self.layers = [nn.Linear(hidden, hidden) for _ in range(n_layers)]
        self.film_layers = [nn.Linear(hidden, 2 * hidden) for _ in range(n_layers)]

        # Output projection
        self.output_proj = nn.Linear(hidden, input_dim)

    def __call__(self, a: mx.array, t: mx.array, obs: mx.array) -> mx.array:
        B = a.shape[0]
        a_flat = a.reshape(B, -1)

        # Time embedding
        te = self.time_emb(t)
        te = nn.mish(self.time_mlp1(te))
        te = self.time_mlp2(te)

        # Conditioning = concat(time_emb, obs)
        cond = mx.concatenate([te, obs], axis=-1)
        cond = nn.mish(self.cond_encoder(cond))  # (B, hidden)

        # Input projection
        h = nn.mish(self.input_proj(a_flat))

        # FiLM-modulated residual MLP
        for layer, film in zip(self.layers, self.film_layers):
            h_in = h
            h = layer(h)
            # FiLM: split film output into scale and bias
            sb = film(cond)
            scale = sb[:, :self.cfg.hidden]
            bias = sb[:, self.cfg.hidden:]
            h = (1.0 + scale) * h + bias
            h = nn.mish(h)
            h = h + h_in  # residual

        out = self.output_proj(h)
        return out.reshape(B, self.cfg.horizon, self.cfg.act_dim)


# ── Noise scheduler with cosine schedule ──────────────────────
class NoiseScheduler:
    def __init__(self, cfg: DiffusionPolicyConfig):
        T = cfg.n_train_diffusion_steps
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)
        self.cfg = cfg
        self.betas = mx.array(betas)
        self.alphas = mx.array(alphas)
        self.alpha_bars = mx.array(alpha_bars)

    def add_noise(self, a0: mx.array, t: mx.array, noise: mx.array) -> mx.array:
        ab = self.alpha_bars[t].reshape(-1, 1, 1)
        return a0 * mx.sqrt(ab) + noise * mx.sqrt(1.0 - ab)

    def ddim_sample(self, model, obs: mx.array,
                    n_steps: int | None = None) -> mx.array:
        cfg = self.cfg
        n_steps = n_steps or cfg.n_inference_steps
        T = cfg.n_train_diffusion_steps
        ts = [int(round(x)) for x in np.linspace(T - 1, 0, n_steps + 1)]
        B = obs.shape[0]
        a = mx.random.normal((B, cfg.horizon, cfg.act_dim))
        for i in range(n_steps):
            t_now = ts[i]
            t_next = ts[i + 1]
            t_batch = mx.full((B,), t_now, dtype=mx.int32)
            eps = model(a, t_batch, obs)
            ab_now = float(self.alpha_bars[t_now].item())
            ab_next = float(self.alpha_bars[t_next].item()) if t_next >= 0 else 1.0
            a0_pred = (a - math.sqrt(1.0 - ab_now) * eps) / math.sqrt(ab_now)
            # Clip in NORMALIZED action space (actions are z-scored to
            # N(0,1), so ±3 covers 99.7%). Denormalization + final
            # clip to [-1, 1] happens at inference boundary.
            a0_pred = mx.clip(a0_pred, -3.0, 3.0)
            a = math.sqrt(ab_next) * a0_pred + math.sqrt(1.0 - ab_next) * eps
        return a  # caller handles denormalization + final clip


def diffusion_loss(model, scheduler: NoiseScheduler,
                   obs: mx.array, a0: mx.array) -> mx.array:
    B = obs.shape[0]
    T = scheduler.cfg.n_train_diffusion_steps
    t = mx.random.randint(0, T, (B,))
    noise = mx.random.normal(a0.shape)
    noised = scheduler.add_noise(a0, t, noise)
    eps_pred = model(noised, t, obs)
    return mx.mean((eps_pred - noise) ** 2)


# ── EMA (Exponential Moving Average) ──────────────────────────
class EMA:
    """Fixed-decay EMA of model parameters. The EMA copy is what
    gets evaluated and deployed — the raw model weights are noisier."""
    def __init__(self, model: DiffusionMLP, decay: float = 0.995):
        self.decay = decay
        # Deep copy of all parameters
        self.shadow = {}
        flat = _flatten_dict(model.parameters())
        for k, v in flat.items():
            self.shadow[k] = mx.array(np.array(v))

    def update(self, model: DiffusionMLP):
        flat = _flatten_dict(model.parameters())
        for k, v in flat.items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply_to(self, model: DiffusionMLP):
        """Load EMA weights into model for inference."""
        model.load_weights(list(self.shadow.items()))


# ── Data normalization ────────────────────────────────────────
class Normalizer:
    """Zero-mean unit-variance normalization. Stores stats in the
    checkpoint so inference uses the same normalization."""
    def __init__(self):
        self.obs_mean = None
        self.obs_std = None
        self.act_mean = None
        self.act_std = None

    def fit(self, obs: np.ndarray, act: np.ndarray):
        self.obs_mean = obs.mean(axis=0).astype(np.float32)
        self.obs_std = obs.std(axis=0).astype(np.float32) + 1e-6
        self.act_mean = act.mean(axis=0).astype(np.float32)
        self.act_std = act.std(axis=0).astype(np.float32) + 1e-6

    def normalize_obs(self, obs):
        if isinstance(obs, mx.array):
            return (obs - mx.array(self.obs_mean)) / mx.array(self.obs_std)
        return (obs - self.obs_mean) / self.obs_std

    def normalize_act(self, act):
        if isinstance(act, mx.array):
            return (act - mx.array(self.act_mean)) / mx.array(self.act_std)
        return (act - self.act_mean) / self.act_std

    def denormalize_act(self, act):
        if isinstance(act, mx.array):
            return act * mx.array(self.act_std) + mx.array(self.act_mean)
        return act * self.act_std + self.act_mean

    def state_dict(self):
        return {
            "obs_mean": self.obs_mean, "obs_std": self.obs_std,
            "act_mean": self.act_mean, "act_std": self.act_std,
        }

    def load_state_dict(self, d):
        self.obs_mean = d["obs_mean"]
        self.obs_std = d["obs_std"]
        self.act_mean = d["act_mean"]
        self.act_std = d["act_std"]


# ── Checkpoint conversion ─────────────────────────────────────
def _flatten_dict(d, prefix=""):
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
    """Convert MLX DiffusionMLP parameters to PyTorch state_dict."""
    import torch
    flat = _flatten_dict(mlx_params)
    torch_state = {}
    for k, v in flat.items():
        arr = np.array(v)
        torch_state[k] = torch.from_numpy(arr.copy())
    return torch_state


def convert_torch_to_mlx(torch_state: dict, cfg: DiffusionPolicyConfig) -> dict:
    """Convert PyTorch DiffusionMLP state_dict to MLX parameters."""
    mlx_params = {}
    for k, v in torch_state.items():
        arr = v.cpu().numpy()
        mlx_params[k] = mx.array(arr)
    return mlx_params


# ── Unified inference-time loader ─────────────────────────────
def load_diffusion_policy(ckpt_path: str):
    """Load a diffusion policy checkpoint and return (model, scheduler,
    normalizer, cfg). The checkpoint must contain model weights,
    config, and normalizer state (saved by train_bc_mlx / train_dppo_mlx).

    Returns an MLX model with EMA weights already applied at save time.
    """
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = DiffusionPolicyConfig(**ckpt["config"])
    model = DiffusionMLP(cfg)
    mlx_params = convert_torch_to_mlx(ckpt["model"], cfg)
    model.load_weights(list(mlx_params.items()))
    scheduler = NoiseScheduler(cfg)
    normalizer = Normalizer()
    if "normalizer" in ckpt:
        normalizer.load_state_dict(ckpt["normalizer"])
    else:
        raise ValueError(
            f"Checkpoint at {ckpt_path} has no 'normalizer' key. "
            "Must be saved by train_bc_mlx / train_dppo_mlx."
        )
    return model, scheduler, normalizer, cfg


def diffusion_act(model: DiffusionMLP, scheduler: NoiseScheduler,
                  normalizer: Normalizer, obs: np.ndarray) -> np.ndarray:
    """Sample one action chunk from the diffusion policy and return
    the first action (denormalized)."""
    obs_norm = normalizer.normalize_obs(obs.astype(np.float32))
    obs_mx = mx.array(obs_norm).reshape(1, -1)
    chunk = scheduler.ddim_sample(model, obs_mx)
    mx.eval(chunk)
    action_norm = np.array(chunk[0, 0])
    action = normalizer.denormalize_act(action_norm)
    return np.clip(action, -1.0, 1.0).astype(np.float32)
