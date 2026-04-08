# Train the full pipeline on Colab (free T4)

This is the exact sequence of cells to run in a Colab notebook with
a T4 GPU to train all three stages end-to-end and push the result
to Hugging Face Hub.

Total wall-clock time: **~5-7 hours**. Colab free tier gives ~12 hours
of T4 time per session so it fits in one sitting.

## Setup

1. Open https://colab.research.google.com/
2. **Runtime → Change runtime type → T4 GPU**
3. Create a new notebook
4. Copy each cell below in order. Run them one at a time and verify
   the output before moving on.

## Cell 1 — Clone and install

```python
!git clone https://github.com/akshan-main/airhockey-rl.git
%cd airhockey-rl
!pip install -q -e ".[storage,app,hub]"
!pip install -q torch  # Colab usually has torch already
import torch
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
```

Expected: `cuda available: True`. If it says False, go to Runtime and
switch to a T4 GPU.

## Cell 2 — Quick sanity check

```python
# Make sure imports work and the env runs
from airhockey.env import AirHockeyEnv
from airhockey.sac import SACAgent, SACConfig
from airhockey.policy import DiffusionPolicyConfig, UNet1D
env = AirHockeyEnv(seed=0)
obs, _ = env.reset()
print(f"obs: {obs.shape}, action_space: {env.action_space}")
```

Expected: `obs: (10,), action_space: Box(-1.0, 1.0, (2,), float32)`.

## Cell 3 — Stage 1: Train the SAC expert (~1-2 hours)

```python
!python -m airhockey.train_sac \
    --total-steps 1000000 \
    --opponent-warmup-steps 50000 \
    --opponent-refresh-steps 50000 \
    --out ckpt/sac_expert.pt
```

Watch the tqdm bar. The `ret` metric should climb from ~-1 to ~+5 by
the end. The `len` metric (episode length) should first shrink (as the
agent learns to score fast) then stabilize around 200-400.

If `ret` is stuck near 0 or negative after 200K steps, something is
wrong — stop and check the SAC logs.

## Cell 4 — Stage 2a: Collect demonstrations from the SAC expert (~10-15 min)

```python
!python -m airhockey.collect \
    --expert ckpt/sac_expert.pt \
    --episodes 5000 \
    --out data/demos.npz \
    --device cuda
```

Expected: `Saved ~2.5M transitions across 5000 episodes to data/demos.npz`
(exact number varies based on episode lengths).

Check the file exists:

```python
!ls -lh data/demos.npz
```

Should be ~50-150 MB depending on episode lengths.

## Cell 5 — Stage 2b: Train Diffusion Policy via BC on SAC demos (~30-45 min)

```python
!python -m airhockey.train_bc \
    --data data/demos.npz \
    --epochs 200 \
    --batch-size 512 \
    --lr 1e-4 \
    --horizon 8 \
    --out ckpt/bc.pt
```

Watch the loss. It should start around 1.0, drop fast to ~0.1 in the
first 20 epochs, and end near 0.01-0.03 by epoch 200. If it stalls at
>0.1, the dataset isn't providing enough signal (check that Stage 1's
SAC actually learned to play).

## Cell 6 — Stage 3: DPPO fine-tune (~3-4 hours)

```python
!python -m airhockey.train_dppo \
    --init ckpt/bc.pt \
    --opponent ckpt/sac_expert.pt \
    --total-steps 2000000 \
    --n-envs 16 \
    --rollout-steps 128 \
    --actor-lr 3e-5 \
    --critic-lr 3e-4 \
    --bc-kl-coef 0.05 \
    --out ckpt/dppo.pt
```

This is the long one. Watch four metrics in the tqdm postfix:

- `ret`: should climb over the course of training
- `win`: should climb from ~0.5 (parity with SAC) upward
- `kl`: should stay under 0.05 — if it spikes, lower `--actor-lr`
- `clip`: should stay under 0.15 — if it exceeds 0.3, raise `--sigma`

If it's clearly failing (win rate dropping), stop it with Ctrl+C.
You can use the BC checkpoint as your final model instead.

## Cell 7 — Evaluate all three checkpoints head-to-head vs the SAC expert

```python
# Eval SAC against itself (sanity — should be around 0.5)
!python -m airhockey.eval --ckpt ckpt/bc.pt --opponent ckpt/sac_expert.pt --episodes 200

# Eval BC against SAC
!python -m airhockey.eval --ckpt ckpt/bc.pt --opponent ckpt/sac_expert.pt --episodes 200

# Eval DPPO against SAC
!python -m airhockey.eval --ckpt ckpt/dppo.pt --opponent ckpt/sac_expert.pt --episodes 200
```

The headline number is DPPO's win rate vs SAC. Anything ≥ 0.55 is a
real improvement over BC.

Write down the numbers — they go into `MODEL_CARD.md`:

```
SAC expert win rate vs itself:      ~0.50
BC diffusion policy vs SAC:         0.??
DPPO diffusion policy vs SAC:       0.??
```

## Cell 8 — Export to ONNX

```python
!python -m airhockey.export_onnx --ckpt ckpt/dppo.pt --out onnx
!ls -lh onnx/
```

Expected: `policy.onnx` (~2-6 MB) + `policy.json` (~1 KB).

## Cell 9 — Push to Hugging Face Hub

First set your HF token as a Colab secret:
- Click the key icon in the left sidebar
- Add secret `HF_TOKEN` with your write-scope token
- Enable "Notebook access"

Then:

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HF_REPO_ID"] = "akshan-main/airhockey-dppo"
!python scripts/upload_to_hub.py
```

Also push the demonstration dataset:

```python
os.environ["HF_DATASET_REPO"] = "akshan-main/airhockey-demos"
!python scripts/upload_demos_to_dataset.py
```

## Cell 10 — Download artifacts locally

```python
from google.colab import files
files.download("ckpt/sac_expert.pt")
files.download("ckpt/bc.pt")
files.download("ckpt/dppo.pt")
files.download("onnx/policy.onnx")
files.download("onnx/policy.json")
```

Move these into your local `airhockey-rl/ckpt/` and `airhockey-rl/onnx/`
directories so you can run the local backend against the trained model.

## Cell 11 — Optional: also build the Minari dataset

```python
!python scripts/build_minari_dataset.py --data data/demos.npz --id airhockey-sac-v0
# To publish (requires an HF / Farama account):
# !minari upload airhockey-sac-v0
```

## After Colab

1. **Create the HF Space** for the FastAPI backend:
   - https://huggingface.co/new-space
   - Name: `airhockey-rl-backend`
   - SDK: Docker
   - Set secrets: `HF_TOKEN`, `HF_REPO_ID`, `HF_BUCKET_*`
2. **Push your airhockey-rl repo to the Space's git remote** so it
   auto-builds the backend Docker image.
3. **Update `play.html`** with the Space URL:
   ```javascript
   window.AIRHOCKEY_BACKEND = "https://akshan-main-airhockey-rl-backend.hf.space";
   ```
4. **Open `play.html`** and play a few games. Trajectories should POST
   to the backend and flow through the periodic retrain loop.

## If something goes wrong

Use the `airhockey-rl-doctor` Claude Skill (`claude_skills/airhockey-rl-doctor/SKILL.md`)
to diagnose training issues. Paste the last 30 lines of the tqdm
progress bar into Claude Code with the Skill loaded and it will walk
through the failure modes encoded in the Skill.
