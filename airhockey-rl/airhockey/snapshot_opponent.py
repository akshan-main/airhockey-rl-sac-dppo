"""load_opponent(ckpt_path) -> (obs -> action) callable.

Auto-detects SAC vs Diffusion Policy checkpoints. For diffusion
policies, checks for the 'normalizer' key — new MLX-trained checkpoints
have it, legacy Conv1d-UNet checkpoints don't.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from airhockey.sac import SACAgent, SACConfig


OpponentFn = Callable[[np.ndarray], np.ndarray]


def _is_sac_checkpoint(ckpt: dict) -> bool:
    return "actor" in ckpt and "critic" in ckpt and "log_alpha" in ckpt


def _is_mlx_diffusion_checkpoint(ckpt: dict) -> bool:
    return "model" in ckpt and "config" in ckpt and "normalizer" in ckpt


def _is_legacy_diffusion_checkpoint(ckpt: dict) -> bool:
    return ("model" in ckpt and "config" in ckpt
            and "normalizer" not in ckpt)


def load_opponent(
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
    deterministic: bool = True,
) -> OpponentFn:
    """Load a SAC or Diffusion Policy checkpoint and return an OpponentFn."""
    device = torch.device(device)
    # weights_only=False is required because our checkpoints contain
    # dicts (config, normalizer). Only load from trusted paths (our
    # own ckpt directory, not user-uploaded files).
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    if _is_sac_checkpoint(ckpt):
        cfg = SACConfig(**ckpt["config"])
        agent = SACAgent(cfg, device=device)
        agent.load_state_dict(ckpt)
        agent.actor.eval()

        def sac_opponent(obs_top: np.ndarray) -> np.ndarray:
            return agent.act(obs_top.astype(np.float32), deterministic=deterministic)

        return sac_opponent

    if _is_mlx_diffusion_checkpoint(ckpt):
        # New MLX diffusion policy (MLP + FiLM + cosine schedule + normalizer)
        from airhockey.policy_mlx import load_diffusion_policy, diffusion_act
        model, scheduler, normalizer, cfg = load_diffusion_policy(str(ckpt_path))

        def diffusion_opponent(obs_top: np.ndarray) -> np.ndarray:
            return diffusion_act(model, scheduler, normalizer, obs_top)

        return diffusion_opponent

    if _is_legacy_diffusion_checkpoint(ckpt):
        # Legacy Conv1d UNet diffusion policy (no normalizer, older format)
        from airhockey.policy import DiffusionPolicyConfig, NoiseScheduler, UNet1D
        cfg = DiffusionPolicyConfig(**ckpt["config"])
        model = UNet1D(cfg).to(device).eval()
        model.load_state_dict(ckpt["model"])
        scheduler = NoiseScheduler(cfg, device=device)

        @torch.no_grad()
        def legacy_diffusion_opponent(obs_top: np.ndarray) -> np.ndarray:
            obs_t = torch.from_numpy(obs_top.astype(np.float32)).unsqueeze(0).to(device)
            chunk = scheduler.ddim_sample(model, obs_t, n_steps=cfg.n_inference_steps)
            return chunk[0, 0].cpu().numpy()

        return legacy_diffusion_opponent

    raise ValueError(
        f"Unrecognized checkpoint format at {ckpt_path}. "
        f"Keys: {list(ckpt.keys())}"
    )
