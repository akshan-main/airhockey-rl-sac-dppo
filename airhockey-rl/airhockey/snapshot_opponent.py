"""Frozen-checkpoint opponent for self-play.

Given a saved checkpoint, returns an opponent callable that matches the
`env.OpponentFn` signature: takes a 10-D observation (top paddle frame)
and returns a 2-D normalized action in [-1, 1].

Supports both checkpoint kinds:
  • SAC actor-only checkpoint — file produced by train_sac.py
  • Diffusion Policy checkpoint — file produced by train_bc.py / train_dppo.py

The loader auto-detects the kind by peeking at the saved state dict keys.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from airhockey.policy import DiffusionPolicyConfig, NoiseScheduler, UNet1D
from airhockey.sac import SACAgent, SACConfig


OpponentFn = Callable[[np.ndarray], np.ndarray]


def _is_sac_checkpoint(ckpt: dict) -> bool:
    return "actor" in ckpt and "critic" in ckpt and "log_alpha" in ckpt


def _is_diffusion_checkpoint(ckpt: dict) -> bool:
    return "model" in ckpt and "config" in ckpt and isinstance(ckpt["config"], dict)


def load_opponent(
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
    deterministic: bool = True,
) -> OpponentFn:
    """Load any trained checkpoint and return a (obs -> action) callable.

    Parameters
    ----------
    ckpt_path : path to a .pt file saved by train_sac, train_bc, or train_dppo
    device    : 'cpu' or 'cuda'
    deterministic : whether to use deterministic actions (no sampling noise)
    """
    device = torch.device(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # ── SAC checkpoint ────────────────────────────────────────
    if _is_sac_checkpoint(ckpt):
        cfg = SACConfig(**ckpt["config"])
        agent = SACAgent(cfg, device=device)
        agent.load_state_dict(ckpt)
        agent.actor.eval()

        def sac_opponent(obs_top: np.ndarray) -> np.ndarray:
            return agent.act(obs_top.astype(np.float32), deterministic=deterministic)

        return sac_opponent

    # ── Diffusion Policy checkpoint ───────────────────────────
    if _is_diffusion_checkpoint(ckpt):
        cfg = DiffusionPolicyConfig(**ckpt["config"])
        model = UNet1D(cfg).to(device).eval()
        model.load_state_dict(ckpt["model"])
        scheduler = NoiseScheduler(cfg, device=device)

        # Each call samples a fresh action chunk and returns the first action.
        # This is slower than SAC but correct; used only for eval / snapshot
        # opponents where latency isn't the bottleneck.
        @torch.no_grad()
        def diffusion_opponent(obs_top: np.ndarray) -> np.ndarray:
            obs_t = torch.from_numpy(obs_top.astype(np.float32)).unsqueeze(0).to(device)
            chunk = scheduler.ddim_sample(model, obs_t, n_steps=cfg.n_inference_steps)
            return chunk[0, 0].cpu().numpy()

        return diffusion_opponent

    raise ValueError(
        f"Unrecognized checkpoint format at {ckpt_path}. "
        f"Keys: {list(ckpt.keys())}"
    )
