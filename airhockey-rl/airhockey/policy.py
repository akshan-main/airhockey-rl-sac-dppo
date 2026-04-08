"""Diffusion Policy: 1D U-Net noise predictor over action chunks.

Architecture: small Conv1D U-Net (Chi et al. 2023). Inputs:
    a_t   (B, H, A)         noised action chunk
    t     (B,)              diffusion timestep
    obs   (B, O)             observation conditioning

Output: predicted noise of shape (B, H, A).

Sampling: DDIM with K steps. Training: standard DDPM ε-prediction loss.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionPolicyConfig:
    obs_dim: int = 10
    act_dim: int = 2
    horizon: int = 8                # action chunk length H
    hidden: int = 128
    obs_embed_dim: int = 64
    n_train_diffusion_steps: int = 100
    n_inference_steps: int = 10
    beta_start: float = 1e-4
    beta_end: float = 0.02


# ── Sinusoidal time embedding (DDPM standard) ──────────────────
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half = self.dim // 2
        emb = math.log(10_000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ── 1D residual conv block ─────────────────────────────────────
class Conv1dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.GroupNorm(8, out_ch),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.block1 = Conv1dBlock(in_ch, out_ch)
        self.block2 = Conv1dBlock(out_ch, out_ch)
        self.cond_mlp = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_ch))
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, cond):
        h = self.block1(x)
        c = self.cond_mlp(cond).unsqueeze(-1)
        h = h + c
        h = self.block2(h)
        return h + self.skip(x)


# ── 1D U-Net noise predictor ───────────────────────────────────
class UNet1D(nn.Module):
    def __init__(self, cfg: DiffusionPolicyConfig):
        super().__init__()
        self.cfg = cfg
        H, A, D = cfg.horizon, cfg.act_dim, cfg.hidden
        # Conditioning MLP fuses observation + diffusion timestep
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(D),
            nn.Linear(D, D * 2),
            nn.Mish(),
            nn.Linear(D * 2, D),
        )
        self.obs_emb = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.obs_embed_dim),
            nn.Mish(),
            nn.Linear(cfg.obs_embed_dim, D),
        )
        cond_dim = D
        self.down1 = ResidualBlock(A, D, cond_dim)
        self.down2 = ResidualBlock(D, D * 2, cond_dim)
        self.mid = ResidualBlock(D * 2, D * 2, cond_dim)
        self.up2 = ResidualBlock(D * 4, D, cond_dim)
        self.up1 = ResidualBlock(D * 2, D, cond_dim)
        self.out = nn.Conv1d(D, A, 1)

    def forward(self, a: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
        a:   (B, H, A)
        t:   (B,)
        obs: (B, O)
        returns predicted noise (B, H, A)
        """
        cond = self.time_emb(t) + self.obs_emb(obs)
        x = a.transpose(1, 2)  # (B, A, H)
        d1 = self.down1(x, cond)
        d2 = self.down2(d1, cond)
        m = self.mid(d2, cond)
        u2 = self.up2(torch.cat([m, d2], dim=1), cond)
        u1 = self.up1(torch.cat([u2, d1], dim=1), cond)
        out = self.out(u1)
        return out.transpose(1, 2)  # back to (B, H, A)


# ── DDPM/DDIM scheduler utilities ──────────────────────────────
class NoiseScheduler:
    """Linear-beta DDPM scheduler with DDIM sampling."""

    def __init__(self, cfg: DiffusionPolicyConfig, device: torch.device | str = "cpu"):
        self.cfg = cfg
        T = cfg.n_train_diffusion_steps
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.device = device

    def add_noise(self, a0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bars[t].view(-1, 1, 1)
        return a0 * torch.sqrt(ab) + noise * torch.sqrt(1.0 - ab)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: UNet1D,
        obs: torch.Tensor,
        n_steps: int | None = None,
    ) -> torch.Tensor:
        """DDIM deterministic sampling. Returns (B, H, A)."""
        cfg = self.cfg
        device = obs.device
        n_steps = n_steps or cfg.n_inference_steps
        T = cfg.n_train_diffusion_steps
        # Schedule: indices spaced over [0, T-1]
        ts = torch.linspace(T - 1, 0, n_steps + 1, dtype=torch.long, device=device)
        B = obs.shape[0]
        a = torch.randn(B, cfg.horizon, cfg.act_dim, device=device)
        for i in range(n_steps):
            t_now = ts[i]
            t_next = ts[i + 1]
            t_batch = t_now.expand(B)
            eps = model(a, t_batch, obs)
            ab_now = self.alpha_bars[t_now]
            ab_next = self.alpha_bars[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
            a0_pred = (a - torch.sqrt(1.0 - ab_now) * eps) / torch.sqrt(ab_now)
            a0_pred = a0_pred.clamp(-1.5, 1.5)
            a = torch.sqrt(ab_next) * a0_pred + torch.sqrt(1.0 - ab_next) * eps
        return a.clamp(-1.0, 1.0)


def diffusion_loss(
    model: UNet1D,
    scheduler: NoiseScheduler,
    obs: torch.Tensor,
    a0: torch.Tensor,
) -> torch.Tensor:
    """Standard DDPM ε-prediction loss."""
    B = obs.shape[0]
    T = scheduler.cfg.n_train_diffusion_steps
    t = torch.randint(0, T, (B,), device=obs.device)
    noise = torch.randn_like(a0)
    noised = scheduler.add_noise(a0, t, noise)
    eps_pred = model(noised, t, obs)
    return F.mse_loss(eps_pred, noise)
