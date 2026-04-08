---
license: mit
tags:
  - reinforcement-learning
  - diffusion-policy
  - dppo
  - imitation-learning
  - robotics
  - gymnasium
  - air-hockey
library_name: pytorch
pipeline_tag: reinforcement-learning
datasets:
  - akshan-main/airhockey-demos
model-index:
  - name: airhockey-dppo
    results:
      - task:
          type: reinforcement-learning
          name: Air Hockey 2D
        dataset:
          name: airhockey-2d-v0
          type: custom
        metrics:
          - type: win_rate_vs_sac
            value: <FILL_AFTER_TRAINING>
          - type: mean_episode_return
            value: <FILL_AFTER_TRAINING>
---

# AirHockey-DPPO

A Diffusion Policy trained in three stages on a custom 2D air hockey
environment:

1. **SAC expert** ([Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290))
   trained via self-play
2. **Behavior cloning** of the SAC expert into a Diffusion Policy
   ([Chi et al. 2023](https://arxiv.org/abs/2303.04137))
3. **DPPO fine-tune** ([Ren et al. 2024](https://arxiv.org/abs/2409.00588))
   with KL-to-BC regularization against the frozen BC checkpoint

The model is the noise-predictor 1D U-Net component of the Diffusion
Policy. At inference, DDIM sampling with 10 denoising steps generates
an action chunk of length 8.

## Files

| File          | Description                                                |
|---------------|------------------------------------------------------------|
| `dppo.pt`     | PyTorch checkpoint (actor + critic + config)               |
| `policy.onnx` | ONNX export of the noise predictor for browser inference   |
| `policy.json` | DDIM schedule constants used by the JS sampler             |

## Architecture

- **Backbone**: 1D Conv U-Net noise predictor
- **Parameters**: ~1.0 M
- **Action chunk horizon**: 8
- **Action dimension**: 2 (paddle accel x, accel y) normalized to [-1, 1]
- **Observation dimension**: 10 (puck pose+vel, both paddle pose+vel)
- **Training diffusion steps**: 100 (DDPM linear beta schedule)
- **Inference diffusion steps**: 10 (DDIM, deterministic)

## Training

### Stage 1 — SAC expert (self-play)

Soft Actor-Critic with a Gaussian actor and twin critics, plus automatic
entropy tuning. Trained on the air hockey env with a self-play opponent
schedule: no-op opponent for the first 50K steps, then a frozen snapshot
of the current actor that refreshes every 50K steps. Trained for 1M
environment steps on a single T4 (~1-2 hours).

### Stage 2 — Diffusion Policy via Behavior Cloning

The SAC expert rolls out 5K episodes of self-play against a frozen copy
of itself. Each step is recorded as an (obs, action) pair. The resulting
dataset is sliced into overlapping (obs, action_chunk) windows of
length 8 and used to train a 1D U-Net noise predictor via the standard
DDPM ε-prediction loss for 200 epochs.

### Stage 3 — DPPO fine-tune

PPO applied to the diffusion sampling chain. The clipped surrogate is
computed over per-denoising-step likelihood ratios, with advantages from
per-environment GAE on the outer environment trajectory. A KL-to-BC
regularization term (β=0.05) penalizes divergence from the frozen BC
checkpoint to prevent catastrophic policy collapse. Opponent: the
Stage 1 SAC expert, held fixed. Trained for 2M environment steps.

Reward: +10 / -10 per goal, +0.1 per puck contact, 0.01 · v_puck_y
shaping toward opponent goal.

### Online retraining

After the initial training, the model is periodically refined from human
gameplay. The HF Spaces backend buffers human-vs-RL trajectories to HF
Buckets; a GitHub Actions cron job runs DPPO updates every 6 hours and
pushes the new revision to this repo.

## Evaluation

Head-to-head against the Stage 1 SAC expert on the same env:

| Stage                    | Win rate vs SAC | Mean return | Mean episode length |
|--------------------------|-----------------|-------------|---------------------|
| SAC expert (vs itself)   | ~0.50           | <FILL>      | <FILL>              |
| BC Diffusion Policy      | <FILL>          | <FILL>      | <FILL>              |
| DPPO Diffusion Policy    | <FILL>          | <FILL>      | <FILL>              |

Numbers will be filled in after the first end-to-end training run. See
`airhockey/eval.py` in the source repo for the evaluation harness.

## Intended use

- Research and demonstration of Diffusion Policy + DPPO on a small,
  fully-observed 2D continuous control task.
- Live in-browser play via the [Hugging Face Space](https://huggingface.co/spaces/akshan-main/airhockey-rl-backend).

## Limitations

- 2D top-down env, fully-observed state. Does not transfer to image
  observations or 3D robot tasks without retraining.
- Trained against the Stage 1 SAC expert as the opponent; performance
  against human players may vary, which is why the online BC-on-winners
  fine-tune loop exists.
- DPPO is a 2024 method; the implementation here is a simplified
  variant intended for a small environment, not the full reference
  from the paper.

## Source code

[github.com/akshan-main/airhockey-rl](https://github.com/akshan-main/airhockey-rl)

## Citations

```bibtex
@article{haarnoja2018sac,
  title={Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor},
  author={Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  journal={ICML},
  year={2018}
}

@article{chi2023diffusion,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and
          Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  journal={Robotics: Science and Systems},
  year={2023}
}

@article{ren2024dppo,
  title={Diffusion Policy Policy Optimization},
  author={Ren, Allen Z. and others},
  journal={arXiv:2409.00588},
  year={2024}
}
```
