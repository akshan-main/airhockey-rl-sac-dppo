---
title: Air Hockey RL — SAC → Diffusion Policy → DPPO
emoji: 🏒
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Air Hockey — SAC → Diffusion Policy → DPPO (Live Demo)

A custom 2D air hockey environment with a Diffusion Policy trained in
three stages:

1. **SAC** ([Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290))
   self-play expert
2. **Behavior cloning** of the SAC expert into a Diffusion Policy
   ([Chi et al. 2023](https://arxiv.org/abs/2303.04137))
3. **DPPO fine-tune** ([Ren et al. 2024](https://arxiv.org/abs/2409.00588))
   with KL-to-BC regularization

The trained policy is served from this Space and runs entirely in the
browser via ONNX runtime. Players connect to the live demo and play
against the policy; their gameplay trajectories are buffered to HF
Buckets and periodically used for a BC-on-winners fine-tune by a
GitHub Actions cron job, which pushes the updated model back to HF
Hub. This Space polls HF Hub and hot-reloads the served ONNX when a
new revision appears.

This Space is the FastAPI backend only. The static frontend (`play.html`)
lives at [github.com/akshan-main/airhockey-rl](https://github.com/akshan-main/airhockey-rl).

## Endpoints

- `GET  /health` — liveness check
- `GET  /model/version` — current served model version + buffer state
- `GET  /model/policy.onnx` — current model bytes (for `onnxruntime-web`)
- `GET  /model/policy.json` — DDIM schedule metadata for the JS sampler
- `POST /trajectory` — submit a recorded human-vs-AI trajectory

## Stack

`FastAPI · Pydantic · Uvicorn · boto3 · huggingface_hub · ONNX · onnxruntime-web · Docker · Hugging Face Spaces`

No PyTorch in this image — all training runs offline. This Space is
purely an I/O + coordination layer between the browser, HF Buckets,
and HF Hub.
