# Backend spec — write this yourself

You are writing `app/server.py` from scratch as a FastAPI app. The goal
is to learn FastAPI by building something that has to actually work end-
to-end, not by reading tutorials.

There is a finished reference implementation at `app/server_reference.py`.
**Do not look at it** while you're writing your own. Look at it only
after you have something working — or after you get genuinely stuck for
30+ minutes on the same thing.

## Where the backend sits in the bigger system

The project trains a Diffusion Policy on a custom 2D air hockey env in
three stages (SAC → Behavior Cloning → DPPO). That happens offline in a
Colab notebook. The trained model is pushed to Hugging Face Hub as an
ONNX export.

**Your backend is the live-deployment piece.** It sits between two
clients:

- The **browser** (`play.html`) — loads the latest ONNX model via
  `onnxruntime-web`, runs inference locally, sends trajectories back,
  and polls for new model versions.
- **HF Hub + HF Buckets** — the persistent storage. HF Hub holds the
  trained model. HF Buckets is where buffered trajectories land for the
  periodic retrain job.

The server itself does not train anything. It is a coordination layer:
serves the model, buffers writes, and hot-reloads when the upstream
model repository changes.

## The periodic retrain loop (context only — your backend doesn't run this)

A GitHub Actions cron job runs every 6 hours in a separate process:

1. Pulls new trajectory shards from HF Buckets via boto3
2. Filters for *winning* episodes (those where the agent scored)
3. Runs a small behavior-cloning update on those winning trajectories
4. Evaluates the candidate against the SAC expert
5. Promotes the new model to HF Hub only if win rate didn't regress >5%

Your backend detects the new HF Hub revision (within 60s of the push),
re-downloads the ONNX file, and bumps an internal version counter. The
browser, polling `/model/version` every 10s, then hot-swaps its in-
memory inference session.

You are implementing the coordination layer for this loop — not the
loop itself.

## Endpoints you need to build

### `GET /health`
Returns `{"ok": True, "version": <int>}`. Used by HF Spaces' healthcheck
and your own sanity checks. Trivial — write this first.

### `GET /model/policy.onnx`
Returns the raw bytes of the trained ONNX model file. The browser
fetches this once on load and again whenever the version increments.

The file lives on disk at `onnx/policy.onnx`. If it doesn't exist yet
(server just started and hasn't pulled from HF Hub yet), return 404
with a helpful detail message.

Use `fastapi.responses.FileResponse`.

### `GET /model/policy.json`
Same as above, but for the diffusion schedule metadata file at
`onnx/policy.json`. The browser's JS DDIM sampler needs this file to
know the alpha_bars schedule; without it the sampler can't step through
denoising correctly.

### `GET /model/version`
Returns a JSON object with the current model version int and how many
trajectory steps are sitting in the in-memory buffer:

```json
{
  "version": 3,
  "n_buffered": 184,
  "last_flush_unix": 1717891234.5
}
```

Use a Pydantic `BaseModel` for the response (this is one of the things
you should learn — FastAPI uses Pydantic to auto-generate response
schemas in the OpenAPI docs).

### `POST /trajectory`
Accepts a JSON body shaped like this:

```json
{
  "steps": [
    {
      "obs": [0.5, 0.7, 0.0, 0.1, 0.5, 0.9, 0.0, 0.0, 0.5, 0.1],
      "action": [0.2, -0.3],
      "reward": 0.01,
      "done": false
    },
    ...
  ],
  "client_id": "anonymous-uuid",
  "model_version": 3
}
```

Define this with nested Pydantic models — `Step` and `Trajectory`.
Pydantic handles validation automatically; if the browser sends garbage,
FastAPI returns a 422 with a clear error message.

The server appends every step to an in-memory `deque` with a bounded
max size (say 20,000) so memory can't grow unbounded even if flushing
fails. Returns:

```json
{"ok": true, "n_buffered": <count>, "version": <current>}
```

**This is the write side** — the endpoint and the background flush
loop both touch the buffer, so the server has to be thread-safe. Use a
`threading.Lock`.

## Background loops you need

These run forever in their own daemon threads, started in the lifespan
startup block. **Both** are essential to the project.

### Loop 1: Buffer flush to HF Buckets

Every ~10 seconds the loop wakes up and checks:

- Is the buffer length ≥ a threshold (say 200)?
- OR has it been more than 30 minutes since the last flush?

If either is true and the buffer is non-empty:

1. Drain the buffer into a local list (under the lock)
2. Pass that list to `HFBucketStore.flush_buffer(...)` from
   `airhockey/storage.py`
3. Update `last_flush` to `time.time()`
4. Print a log line with the shard key

If `HFBucketStore` is unavailable (no env vars set, or the credentials
are wrong), still drain the buffer so memory doesn't grow — just don't
upload. Print a warning instead.

### Loop 2: HF Hub model version polling

Every ~60 seconds, ask HF Hub if the model repo has a new commit. If
yes:

1. Use `huggingface_hub.HfApi.repo_info(repo_id)` to get the current SHA
2. Compare to the SHA from the previous poll
3. If changed, use `huggingface_hub.hf_hub_download(...)` to download
   `policy.onnx` and `policy.json` into the `onnx/` directory
4. Increment the in-memory version counter
5. Print a log line

If `HF_REPO_ID` env var is not set, skip this loop entirely (the version
counter stays at whatever it was at startup). This is useful for local
testing without HF credentials.

## State management

You need a small amount of mutable shared state across all the request
handlers and the background loops:

- The buffer (a `collections.deque` with `maxlen=MAX_BUFFER`)
- A `threading.Lock` to protect the buffer
- The current version int
- The last flush timestamp (unix seconds)
- The `HFBucketStore` instance (or `None` if unavailable)

The cleanest pattern is a single class instance (`ServerState`) created
once at module load time. Every endpoint and both background loops read
from this shared state.

## Configuration via environment variables

Read these at startup. All are optional — the server should start even
if none are set (it just won't be able to talk to HF).

| Var | Default | What it does |
|---|---|---|
| `HF_REPO_ID` | `""` | HF Hub model repo, e.g. `akshan-main/airhockey-dppo` |
| `HF_TOKEN` | `""` | HF auth token (used for Hub + bucket client) |
| `AIRHOCKEY_ONNX_DIR` | `"onnx"` | Local dir where the model file lives |
| `AIRHOCKEY_FLUSH_THRESHOLD` | `"200"` | Buffer size that triggers a flush |
| `AIRHOCKEY_FLUSH_INTERVAL` | `"1800"` | Max seconds between flushes |
| `AIRHOCKEY_MAX_BUFFER` | `"20000"` | Hard cap on in-memory buffer size |
| `AIRHOCKEY_POLL_INTERVAL` | `"60"` | Seconds between HF Hub poll calls |
| `HF_BUCKET_NAME` | from storage.py | HF Bucket name |
| `HF_BUCKET_ENDPOINT` | `https://buckets.hf.co` | Bucket endpoint |
| `HF_BUCKET_KEY_ID` | `""` | S3-compatible access key |
| `HF_BUCKET_SECRET` | `""` | S3-compatible secret key |

Use `os.environ.get(name, default)` and `int(...)` for the numeric ones.

## CORS

The browser will be hosted at a different origin from the Space (e.g.
your GitHub Pages site, or a local `file://`). FastAPI doesn't allow
cross-origin requests by default, so you need to add the CORS middle-
ware. For a public demo, the simplest permissive setup:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

In a real production system you'd lock this down to a specific origin,
but for a public demo `*` is fine.

## Lifespan / startup hook (use the modern pattern)

FastAPI used to use `@app.on_event("startup")` but that is deprecated
in modern FastAPI. The new way is the **lifespan context manager**:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup code here — start background threads, init state,
    # eagerly download the model from HF Hub so the server can
    # serve it from second 0.
    yield
    # shutdown code here (we don't really need any — daemon threads
    # die with the process)

app = FastAPI(lifespan=lifespan)
```

Use the lifespan pattern. It's the modern API.

Inside `lifespan`, do in this order:

1. Initialize the `ServerState` and its `HFBucketStore`
2. **Eagerly download `policy.onnx` and `policy.json` from HF Hub**
   so the server is serving immediately, not waiting 60 seconds for
   the poll loop to catch up. This is important — without it, the
   first minute of uptime returns 404 for every model fetch.
3. Start the two background threads (use `threading.Thread(target=...,
   daemon=True).start()`)

## Recommended order to build it in

Don't try to build the whole thing at once. Go in this order; test each
step before moving on. Each step adds one capability.

1. **Hello world.** A FastAPI app with one route `/health` that returns
   `{"ok": True}`. Run it with `uvicorn app.server:app --reload --host
   0.0.0.0 --port 7860`. Hit `http://localhost:7860/health` in a
   browser. **Confirm this works before doing anything else.**

2. **Pydantic models.** Define the `Step`, `Trajectory`, and
   `VersionResponse` Pydantic models. Add the `POST /trajectory`
   endpoint that just accepts the body and returns
   `{"received": True, "n_steps": len(traj.steps)}`. Test it with:
   ```
   curl -X POST -H "Content-Type: application/json" \
        -d '{"steps": [], "model_version": 1}' \
        http://localhost:7860/trajectory
   ```

3. **In-memory state.** Create the `ServerState` class with the buffer
   and lock. Update `POST /trajectory` to actually append to the buffer.
   Add `GET /model/version` that reports the buffer size using
   `VersionResponse`.

4. **File serving.** Add `GET /model/policy.onnx` and
   `GET /model/policy.json` using `FileResponse`. Test by manually
   creating the `onnx/` directory and dropping a small placeholder
   file at `onnx/policy.onnx`, then curling the endpoint.

5. **CORS.** Add the CORS middleware. Test by opening `play.html` in a
   browser and hitting the endpoints from JavaScript `fetch`.

6. **Lifespan + background threads.** Set up the lifespan context
   manager. Start the background flush loop (you can stub out the
   actual `HFBucketStore` call to just
   `print(f"would flush {len(steps)} steps")` for now).

7. **HF Buckets integration.** Replace the stub in the flush loop with
   the real `HFBucketStore.flush_buffer(...)` call. Test by setting the
   env vars and watching for shards to appear in the bucket.

8. **HF Hub polling.** Add the second background thread. Test by
   manually pushing a new commit to your HF Hub repo and watching the
   server log for `New model revision detected`.

9. **Eager model download at startup.** Add a function that pulls
   `policy.onnx` + `policy.json` from HF Hub inside the lifespan
   `startup` block, BEFORE the poll loop starts. So the server can
   serve `/model/policy.onnx` from second 0, not second 60.

After step 9 you have the full backend.

## What you should learn from each step

- **Step 1**: how to declare a FastAPI app, how to run it with uvicorn,
  how routing works
- **Step 2**: Pydantic — how request bodies get validated, how nested
  models work, how response models become docs
- **Step 3**: shared state in a request handler, why thread safety
  matters when a background loop touches the same data
- **Step 4**: serving files, content types, when to return 404
- **Step 5**: CORS, browser fetch behavior, why `*` is OK for demos
- **Step 6**: lifespan pattern, background threads, `daemon=True`
- **Step 7**: integrating an external client (boto3 → HF Buckets),
  error handling, graceful degradation when credentials are missing
- **Step 8**: polling external APIs, tracking state across iterations,
  why we compare SHAs
- **Step 9**: startup ordering — why the order things happen at boot
  matters for serving requests immediately

By the end of step 9 you will know FastAPI well enough to build other
small backends without referring to the docs for every line.

## Tools you'll be using

- `fastapi` — the framework
- `uvicorn` — the ASGI server that runs your app
- `pydantic` — request/response validation (comes with FastAPI)
- `huggingface_hub` — `HfApi`, `hf_hub_download` for the model registry
- `boto3` (via `airhockey/storage.py`) — for HF Buckets
- Standard library: `threading`, `collections.deque`, `os`, `time`,
  `pathlib`, `contextlib`, `typing`

## How to run it locally

```bash
cd airhockey-rl
pip install -e ".[app]"           # if you haven't already
uvicorn app.server:app --reload --host 0.0.0.0 --port 7860
```

Open http://localhost:7860/docs in your browser. FastAPI auto-generates
an interactive API documentation page from your Pydantic models and
endpoints. You can test every endpoint from there. **This is one of
the things FastAPI is most loved for** — try every endpoint from the
docs UI as you build each step.

For local testing without HF credentials, just leave all the `HF_*`
env vars unset. The server should still start; it just won't be able
to flush to buckets or poll the hub.

## When to ask for help

Ask me anything anytime, but I won't write the code for you. The kinds
of questions where I'll be most useful:

- "I'm getting this error, what does it mean" → paste the error
- "Why does X work this way" → conceptual questions
- "Is this the right pattern for Y" → design questions
- "Review this code" → I'll read what you wrote and point out real bugs

The kinds of questions I'll redirect you to figure out yourself:

- "Write the X endpoint" → no
- "What goes here" → look at FastAPI's docs at
  https://fastapi.tiangolo.com/

## When you're done

Tell me, and I'll do an honest review against the same checklist I
used on the rest of the codebase. Bugs called out, then we move on to
training the model on Colab.

## A few pitfalls the reference implementation originally hit

These are real mistakes I made while writing `server_reference.py` the
first time. The reference has since been corrected to avoid them, but
you should know about them so you don't repeat them:

1. **Used the deprecated `@app.on_event("startup")`** instead of the
   modern lifespan pattern. Causes warnings on modern FastAPI. The
   reference now uses `lifespan`.
2. **Did not eagerly download the model at startup**, so for the first
   60 seconds after a Space wake-up, `/model/policy.onnx` returned 404.
   The reference now does an eager download inside `lifespan`.
3. **Imported `asyncio` and `io` and never used them.** Cosmetic but
   shows up in linters. The reference no longer has dead imports.
4. **Used `datetime.utcnow()`** which is deprecated in Python 3.12+.
   Use `datetime.now(timezone.utc)` instead. This was in `storage.py`,
   not `server.py` directly, but if you do any timestamps in your
   server, don't use `utcnow()`.

If you avoid those four, your version will be at least as good as the
reference.

## What's NOT in scope for your server

Don't try to implement any of these — they live elsewhere:

- **Training loops** — all training runs in `airhockey/*.py`, executed
  on Colab via the walkthrough in `scripts/colab_train.md`. Your
  server never touches PyTorch.
- **RL updates on browser-submitted trajectories** — that's the
  periodic retrain cron job in `.github/workflows/retrain.yml`, which
  runs `airhockey/retrain_cycle.py`. Your server just buffers the
  trajectories so the cron can consume them.
- **Eval / eval gate** — also in `retrain_cycle.py`, not your server.
- **ONNX export** — already done by `airhockey/export_onnx.py` before
  the model hits HF Hub.

Your server is purely I/O and coordination. If you find yourself
importing `torch` or anything from `airhockey.policy` / `airhockey.sac`
/ `airhockey.dppo`, you're out of scope — stop and rethink.

Good luck. Build small, test each step, ask when stuck.
