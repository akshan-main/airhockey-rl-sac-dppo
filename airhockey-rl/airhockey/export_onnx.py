"""Export the Diffusion Policy noise predictor to ONNX.

The graph is a single noise-prediction call:
    inputs:  a (B, H, A) float32, t (B,) int64, obs (B, O) float32
    output:  eps (B, H, A) float32

Also writes policy.json with the DDIM schedule the JS sampler uses.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from airhockey.policy import DiffusionPolicyConfig, NoiseScheduler, UNet1D


class NoisePredictorWrapper(torch.nn.Module):
    """Exposes the UNet as a single (a, t, obs) -> eps function."""

    def __init__(self, model: UNet1D):
        super().__init__()
        self.model = model

    def forward(self, a: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.model(a, t, obs)


def export(ckpt_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = DiffusionPolicyConfig(**ckpt["config"])
    model = UNet1D(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapper = NoisePredictorWrapper(model)

    H, A, O = cfg.horizon, cfg.act_dim, cfg.obs_dim
    dummy_a = torch.zeros(1, H, A, dtype=torch.float32)
    dummy_t = torch.zeros(1, dtype=torch.int64)
    dummy_obs = torch.zeros(1, O, dtype=torch.float32)

    onnx_path = out / "policy.onnx"
    # dynamo=False because torch >= 2.9's dynamo exporter fails on
    # this model's GroupNorm + Conv1d + cat pattern. The TorchScript
    # path produces ONNX that onnxruntime-web loads correctly.
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*legacy TorchScript-based ONNX export.*",
    )
    torch.onnx.export(
        wrapper,
        (dummy_a, dummy_t, dummy_obs),
        str(onnx_path),
        input_names=["a", "t", "obs"],
        output_names=["eps"],
        dynamic_axes={
            "a": {0: "batch"},
            "t": {0: "batch"},
            "obs": {0: "batch"},
            "eps": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"Wrote {onnx_path}")

    # Build the per-inference-step schedule the JS sampler needs.
    scheduler = NoiseScheduler(cfg, device="cpu")
    T = cfg.n_train_diffusion_steps
    K = cfg.n_inference_steps
    ts = torch.linspace(T - 1, 0, K + 1, dtype=torch.long).tolist()
    ab = scheduler.alpha_bars.cpu().numpy()
    schedule = {
        "horizon": H,
        "act_dim": A,
        "obs_dim": O,
        "n_train_steps": T,
        "n_inference_steps": K,
        "timesteps": ts,                              # length K+1
        "alpha_bars": [float(ab[t]) for t in ts[:-1]],  # length K
        "alpha_bars_next": [float(ab[t]) if t >= 0 else 1.0 for t in ts[1:]],  # length K
    }
    meta_path = out / "policy.json"
    with open(meta_path, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"Wrote {meta_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="onnx")
    args = p.parse_args()
    export(args.ckpt, args.out)


if __name__ == "__main__":
    main()
