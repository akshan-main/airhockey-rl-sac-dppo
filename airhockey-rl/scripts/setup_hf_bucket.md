# Setting up HF Buckets + HF Hub for the air hockey project

This is a one-time ~15 minute setup. After this the periodic retrain
loop runs on its own.

## 1. Create the Hugging Face model repo

The trained policy lives here. The Spaces backend pulls from this repo
and the GH Actions retrain workflow pushes to it.

```bash
huggingface-cli login
huggingface-cli repo create airhockey-dppo --type model
```

Or via the website: https://huggingface.co/new

Note the full repo id, e.g. `username/airhockey-dppo`.

## 2. Create the Hugging Face Bucket

The bucket holds buffered trajectory shards from human-vs-RL gameplay
between retrain cycles. Lifecycle policy auto-deletes old shards.

1. Go to https://huggingface.co/buckets (or your account → Buckets)
2. Click **New bucket**, name it `airhockey-rl-buffer`
3. Set the bucket region (whichever is closest to where you'll run
   GH Actions — `us-east` is the safe default)
4. Set a **lifecycle rule**:
   - Prefix: `buffered/`
   - Action: Delete after 7 days
   - This makes "dispose old shards" automatic
5. **Generate access keys** for the bucket: copy the Access Key ID and
   the Secret Access Key — you'll only see the secret once.

## 3. Generate a Hugging Face access token

The retrain workflow needs write scope on the model repo.

1. Go to https://huggingface.co/settings/tokens
2. **New token** → name `airhockey-rl-write`, role `write`
3. Copy the token.

## 4. Set GitHub repository secrets

In the GitHub repo settings for `readmeskillsection`, under
**Settings → Secrets and variables → Actions → New repository secret**,
add:

| Name              | Value                                            |
|-------------------|--------------------------------------------------|
| `HF_TOKEN`        | the write token from step 3                       |
| `HF_REPO_ID`      | `username/airhockey-dppo`                         |
| `HF_BUCKET_NAME`  | `airhockey-rl-buffer`                             |
| `HF_BUCKET_KEY_ID`| access key id from step 2                         |
| `HF_BUCKET_SECRET`| secret access key from step 2                     |
| `HF_SPACE`        | `username/airhockey-rl-backend` (for app deploy)  |

## 5. Bootstrap the model repo

The retrain workflow expects an existing checkpoint to update. Train one
locally first and push it once:

```bash
cd airhockey-rl
python -m airhockey.collect --episodes 5000
python -m airhockey.train_bc --epochs 200
python -m airhockey.train_dppo --total-steps 1_000_000
python -m airhockey.export_onnx --ckpt ckpt/dppo.pt --out onnx

huggingface-cli upload username/airhockey-dppo ckpt/dppo.pt dppo.pt
huggingface-cli upload username/airhockey-dppo onnx/policy.onnx policy.onnx
huggingface-cli upload username/airhockey-dppo onnx/policy.json policy.json
```

## 6. Create the HF Space (Docker SDK)

The Space hosts the FastAPI backend. It serves the ONNX model and accepts
trajectory POSTs.

1. Go to https://huggingface.co/new-space
2. Owner = your username
3. Name = `airhockey-rl-backend`
4. SDK = **Docker**
5. Visibility = public
6. Hardware = CPU basic (free)
7. After creation, go to the Space's settings and add the secrets:
   - `HF_REPO_ID`, `HF_TOKEN`
   - `HF_BUCKET_NAME`, `HF_BUCKET_KEY_ID`, `HF_BUCKET_SECRET`
   - `HF_BUCKET_ENDPOINT` = `https://buckets.hf.co`

The Space will auto-deploy from `airhockey-rl/app/Dockerfile` once you
push the repo. The GitHub Actions `deploy.yml` workflow handles this on
each push to `main`.

## 7. Verify the loop

After everything is set up:

1. Open `play.html` locally (or hosted on GitHub Pages) and play a few
   games against the model.
2. The browser POSTs trajectories to the Space.
3. Within ~30 minutes (or 200 buffered steps) the Space flushes a shard
   to HF Buckets under `buffered/`.
4. Within 6 hours the GH Actions cron runs `retrain_cycle.py`, which
   pulls the shards, runs DPPO, exports a new ONNX, and pushes to HF Hub.
5. Within 60 seconds of the push, the Space's HF Hub poller detects the
   new revision and re-downloads the model.
6. The browser, polling `/model/version` every 10 seconds, sees the
   bumped version and re-initializes its ONNX session with the new
   model file.

## Architecture summary

```
Browser (play.html)
    │
    │ POST /trajectory
    ▼
HF Spaces backend (FastAPI, Docker SDK)
    │
    │ flush every 30 min or 200 steps
    ▼
HF Buckets (s3://airhockey-rl-buffer/buffered/...)
    │
    │ lifecycle: delete after 7 days
    │
    │ list + download new shards every 6h
    ▼
GitHub Actions cron (.github/workflows/retrain.yml)
    │
    │ run DPPO update, export ONNX
    │
    │ push to HF Hub
    ▼
HF Hub model repo (username/airhockey-dppo)
    │
    │ poll for new revision every 60s
    ▼
HF Spaces backend re-downloads ONNX, bumps internal version
    │
    │ /model/version reports new version
    ▼
Browser hot-reloads its ONNX session and keeps playing.
```

That's the full self-improving loop. Every component is real, every
piece does the thing it's actually best at, no theater.
