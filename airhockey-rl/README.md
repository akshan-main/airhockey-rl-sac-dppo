# Air Hockey — SAC → Diffusion Policy → DPPO

A custom 2D air hockey environment, a three-stage RL training pipeline,
and cross-language browser deployment via ONNX runtime.

## Links

[![GitHub](https://img.shields.io/badge/code-GitHub-181717?logo=github)](https://github.com/akshan-main/airhockey-rl)
[![Model](https://img.shields.io/badge/model-🤗%20Hub-yellow)](https://huggingface.co/akshan-main/airhockey-dppo)
[![Dataset](https://img.shields.io/badge/dataset-🤗%20Datasets-orange)](https://huggingface.co/datasets/akshan-main/airhockey-demos)
[![Demo](https://img.shields.io/badge/demo-🤗%20Spaces-blue)](https://huggingface.co/spaces/akshan-main/airhockey-rl-backend)
[![Minari](https://img.shields.io/badge/dataset-Minari-green)](https://minari.farama.org/)

## What's in here

```
airhockey-rl/
├── airhockey/
│   ├── physics.py            Semi-implicit Euler + elastic collisions (NumPy sim)
│   ├── env.py                Gymnasium env wrapping the physics
│   ├── sac.py                Stage 1 — Soft Actor-Critic implementation
│   ├── train_sac.py          Stage 1 driver — self-play SAC
│   ├── snapshot_opponent.py  Load any trained ckpt as an env opponent
│   ├── collect.py            Stage 2a — collect demos from a SAC expert
│   ├── policy.py             Diffusion Policy (1D UNet noise predictor)
│   ├── train_bc.py           Stage 2b — DDPM behavior cloning on SAC demos
│   ├── dppo.py               DPPO algorithm (PPO over diffusion chain)
│   ├── train_dppo.py         Stage 3 driver — RL fine-tune with BC-KL regularization
│   ├── eval.py               Head-to-head eval vs a configurable opponent
│   ├── export_onnx.py        Export the noise predictor for browser inference
│   ├── storage.py            HF Buckets boto3 client
│   └── retrain_cycle.py      Periodic BC-on-winners update from human gameplay
├── app/
│   ├── server.py             FastAPI backend (model serving + trajectory intake)
│   ├── server_reference.py   Reference implementation (hidden during learning)
│   ├── SPEC.md               The spec for writing server.py from scratch
│   ├── Dockerfile            HF Spaces Docker SDK image
│   └── README_SPACE.md       Space metadata
├── claude_skills/
│   └── airhockey-rl-doctor/
│       └── SKILL.md          Diagnostic Skill for RL training failures
├── scripts/
│   ├── colab_train.md        Cell-by-cell Colab walkthrough (run this to train)
│   ├── setup_hf_bucket.md    One-time HF Buckets setup
│   ├── setup_hf_space.md     One-time HF Space setup
│   ├── upload_to_hub.py      Push trained model + card to HF Hub
│   ├── upload_demos_to_dataset.py  Push demonstrations to HF Datasets
│   ├── build_minari_dataset.py     Convert demos to Minari format
│   └── gymnasium_robotics_pr/      Submission artifacts for Farama registry
├── data/                     Demonstration trajectories (.gitignored)
├── ckpt/                     Saved checkpoints (.gitignored)
├── onnx/                     ONNX exports (.gitignored)
├── MODEL_CARD.md             HF Hub model card
├── DATASET_CARD.md           HF Datasets card
├── MINARI_CARD.md            Minari dataset card
├── pyproject.toml            Dependencies
├── LICENSE                   MIT
└── README.md                 This file
```

## Pipeline

```bash
# Install
pip install -e ".[storage,app,hub]"

# Stage 1 — SAC expert via self-play (~1-2h on T4)
python -m airhockey.train_sac --total-steps 1000000 --out ckpt/sac_expert.pt

# Stage 2a — Collect demos from the SAC expert (~10 min)
python -m airhockey.collect --expert ckpt/sac_expert.pt --episodes 5000

# Stage 2b — BC train Diffusion Policy on SAC demos (~30 min)
python -m airhockey.train_bc --data data/demos.npz --epochs 200 --out ckpt/bc.pt

# Stage 3 — DPPO fine-tune (~3-4h)
python -m airhockey.train_dppo \
    --init ckpt/bc.pt \
    --opponent ckpt/sac_expert.pt \
    --bc-kl-coef 0.05 \
    --total-steps 2000000 \
    --out ckpt/dppo.pt

# Evaluate head-to-head vs the SAC expert
python -m airhockey.eval --ckpt ckpt/dppo.pt --opponent ckpt/sac_expert.pt --episodes 200

# Export to ONNX for the browser
python -m airhockey.export_onnx --ckpt ckpt/dppo.pt --out onnx

# Push to HF Hub
HF_TOKEN=hf_xxx HF_REPO_ID=akshan-main/airhockey-dppo python scripts/upload_to_hub.py
```

For a full walkthrough on a free Colab T4 session, see
[`scripts/colab_train.md`](scripts/colab_train.md).

## Algorithm summary

### Stage 1 — Soft Actor-Critic (self-play)

Standard SAC with a Gaussian actor and twin critics, plus automatic
entropy tuning. Self-play schedule:

- Start with a no-op opponent for `opponent_warmup_steps` (default 50K)
  so the agent can learn basic puck-hitting.
- After warmup, freeze a snapshot of the current actor and use it as
  the top-paddle opponent.
- Refresh the frozen opponent every 50K steps.

Output: `ckpt/sac_expert.pt`.

### Stage 2 — Diffusion Policy via Behavior Cloning

A 1D U-Net noise predictor `ε_θ(a_t, t, obs)` is trained with the
standard DDPM loss on action chunks of length H=8:

```
L_BC = 𝔼_{(o,a₀), t∼U(1..T), ε∼N(0,I)} [ ‖ε - ε_θ(√ᾱ_t a₀ + √(1-ᾱ_t) ε, t, o)‖² ]
```

Trained on demonstrations generated by rolling out the Stage 1 SAC
agent against itself for 5K episodes. T=100 training diffusion steps,
DDIM sampling at inference with K=10 denoising steps.

Output: `ckpt/bc.pt` and `data/demos.npz`.

### Stage 3 — DPPO fine-tune

PPO applied to the diffusion sampling chain ([Ren et al. 2024](https://arxiv.org/abs/2409.00588)).
The clipped surrogate is computed over per-denoising-step likelihood
ratios, with advantages from per-env GAE on the outer environment
trajectory.

A KL-to-BC regularization term penalizes divergence from the frozen
BC checkpoint, preventing catastrophic policy collapse during online
RL fine-tuning:

```
L = L_PPO + β_BC · KL(π_θ || π_BC)
```

Opponent: the Stage 1 SAC expert (held fixed), so the Diffusion Policy
learns to exceed the SAC baseline while staying close to its distilled
starting point.

Output: `ckpt/dppo.pt`.

## Online fine-tuning from human gameplay

After deployment, the policy continues to improve from real humans
playing against it in the browser:

1. `play.html` runs the policy via `onnxruntime-web` and streams
   trajectories to the backend
2. The backend buffers trajectories and flushes them to HF Buckets
   every 30 minutes
3. A GitHub Actions cron job runs every 6 hours:
   - Pulls new trajectory shards from HF Buckets
   - Filters for *winning* episodes
   - Runs a small behavior-cloning update on winning trajectories
   - Evaluates the candidate against the SAC expert
   - Promotes the new model only if win rate didn't regress >5% (eval gate)
   - Pushes the promoted model to HF Hub
4. The backend's HF Hub poller detects the new revision within 60s and
   hot-reloads the served ONNX
5. The browser polls `/model/version` every 10s and re-initializes its
   inference session when the version increments

The result is a genuinely self-improving system: the agent gets better
the more people play against it, gated by an automatic eval safeguard.

## Stack

`PyTorch · Gymnasium · NumPy · ONNX · onnxruntime-web · FastAPI · Pydantic · Docker · boto3 · Hugging Face Hub · Hugging Face Spaces · HF Buckets · Hugging Face Datasets · Minari · GitHub Actions`

## Math used (no decoration)

- **Semi-implicit Euler integration** of puck + paddle physics
- **Vector reflection with restitution** for wall bounces
- **Elastic line-of-impact collision** with proper mass ratio (paddle vs puck)
- **Gaussian policy with tanh squash** + reparameterization trick (SAC)
- **DDPM ε-prediction loss** (Diffusion Policy)
- **DDIM sampling** at inference (deterministic)
- **GAE** (per-environment, correct across episode boundaries)
- **PPO clipped surrogate** applied to diffusion denoising-step ratios
- **KL-to-BC regularization** to prevent policy collapse
- **Polyak target network averaging** (SAC)

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

## License

MIT.
