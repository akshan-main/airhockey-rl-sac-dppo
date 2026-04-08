# Setting up the HF Space (FastAPI backend)

This is a one-time ~10-minute setup to host the FastAPI backend
that serves the trained ONNX model and accepts trajectory POSTs from
the browser play page.

## Prerequisites

- HF account with username `akshan-main`
- A write-scope token at https://huggingface.co/settings/tokens
- Local copy of `airhockey-rl/` ready to push

## 1. Create the Space

1. Go to https://huggingface.co/new-space
2. Owner: `akshan-main`
3. Name: `airhockey-rl-backend`
4. License: `mit`
5. SDK: **Docker**
6. Hardware: `cpu-basic` (free tier — sufficient for inference)
7. Visibility: `public`
8. Click **Create Space**

Note the resulting URL: `https://huggingface.co/spaces/akshan-main/airhockey-rl-backend`.

## 2. Set the Space secrets

In the Space's settings tab → **Variables and secrets** → add the
following as **secrets** (not variables):

| Name                | Value                                     |
|---------------------|-------------------------------------------|
| `HF_TOKEN`          | your write-scope token                    |
| `HF_REPO_ID`        | `akshan-main/airhockey-dppo`              |
| `HF_BUCKET_NAME`    | `airhockey-rl-buffer`                     |
| `HF_BUCKET_KEY_ID`  | bucket access key id (from setup_hf_bucket.md) |
| `HF_BUCKET_SECRET`  | bucket secret access key                  |
| `HF_BUCKET_ENDPOINT`| `https://buckets.hf.co`                   |

These are visible to the running container as environment variables.
The FastAPI backend reads them at startup in `app/server.py`.

## 3. Push the code

The Space deploys via `app/Dockerfile`. Push the `airhockey-rl/`
contents to the Space's git remote:

```bash
git clone https://huggingface.co/spaces/akshan-main/airhockey-rl-backend hf-space
cp -r airhockey-rl/* hf-space/
cd hf-space
git add .
git commit -m "Deploy backend"
git push
```

Or wire it up via the existing GitHub Actions workflow at
`.github/workflows/deploy.yml` — push to `main` and the workflow
auto-mirrors to the HF Space remote.

## 4. Watch the build

The Space dashboard shows the Docker build log. First build takes
~5 min (installing PyTorch CPU + boto3 + huggingface_hub). Subsequent
builds are cached.

When the build succeeds, the Space's URL responds with:

```bash
curl https://akshan-main-airhockey-rl-backend.hf.space/health
# {"ok": true, "version": 1}
```

## 5. Bootstrap the model

Before the first useful request, the Space needs a model in HF Hub
to download. Make sure you've already run:

```bash
python -m airhockey.collect ...
python -m airhockey.train_bc ...
python -m airhockey.train_dppo ...
python -m airhockey.export_onnx ...
python scripts/upload_to_hub.py
```

Within ~60 seconds of the upload, the Space's HF Hub poller picks it
up and starts serving the model at `/model/policy.onnx`.

## 6. Point the browser at it

In `play.html`, set:

```javascript
window.AIRHOCKEY_BACKEND = "https://akshan-main-airhockey-rl-backend.hf.space";
```

Open `play.html` locally or host it on GitHub Pages. Play. Trajectories
flow to the Space, get buffered, get flushed to HF Buckets, and the
periodic GitHub Actions retrain pulls them in for the next DPPO update.

## Troubleshooting

- **Space build fails on `pip install torch`** — the Dockerfile uses the
  CPU index URL `--index-url https://download.pytorch.org/whl/cpu`. If
  your build is slow or runs out of memory, switch to a smaller torch
  variant or upgrade to a paid CPU tier.
- **`/model/policy.onnx` returns 404** — the HF Hub poller hasn't run yet
  or `HF_REPO_ID` is wrong. Check the Space logs for `Hot-reloaded model`
  lines. Verify the model repo at https://huggingface.co/akshan-main/airhockey-dppo
  actually contains `policy.onnx`.
- **CORS errors in the browser** — the FastAPI backend has `allow_origins=["*"]`
  by default. If you've tightened CORS, add the play.html origin to the
  allowed list.
- **Trajectory POSTs return 200 but `n_buffered` doesn't grow** — check
  that the Space's secrets are set. If `HF_BUCKET_*` are missing, the
  flush loop drops the data instead of erroring.
