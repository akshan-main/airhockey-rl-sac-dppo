"""FastAPI backend. Runs on HF Spaces (Docker SDK).

Responsibilities:
  1. Serve the current ONNX model + DDIM schedule JSON to the browser.
  2. Accept trajectory POSTs, buffer them, flush to HF Buckets.
  3. Poll HF Hub for new model revisions and hot-reload.

The retrain itself runs in a GitHub Actions cron (retrain_cycle.py),
not here. This server is I/O + coordination only.

Endpoints:
    GET  /health
    GET  /model/version
    GET  /model/policy.onnx
    GET  /model/policy.json
    POST /trajectory
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from airhockey.storage import HFBucketStore


HF_REPO = os.environ.get("HF_REPO_ID", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ONNX_DIR = Path(os.environ.get("AIRHOCKEY_ONNX_DIR", "onnx"))
FLUSH_THRESHOLD = int(os.environ.get("AIRHOCKEY_FLUSH_THRESHOLD", "200"))
FLUSH_INTERVAL_SEC = int(os.environ.get("AIRHOCKEY_FLUSH_INTERVAL", "1800"))
MAX_BUFFER = int(os.environ.get("AIRHOCKEY_MAX_BUFFER", "20000"))
POLL_INTERVAL_SEC = int(os.environ.get("AIRHOCKEY_POLL_INTERVAL", "60"))


class Step(BaseModel):
    obs: list[float] = Field(..., description="10-D normalized observation")
    action: list[float] = Field(..., description="2-D normalized action")
    reward: float
    done: bool


class Trajectory(BaseModel):
    steps: list[Step]
    client_id: Optional[str] = None
    model_version: int


class VersionResponse(BaseModel):
    version: int
    n_buffered: int
    last_flush_unix: float


class ServerState:
    """Shared mutable state guarded by a single lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.buffer: deque[dict] = deque(maxlen=MAX_BUFFER)
        self.version = 0
        self.last_flush = 0.0
        self.store: Optional[HFBucketStore] = None

    def init_storage(self) -> None:
        try:
            self.store = HFBucketStore()
            print("HF Buckets storage client initialized")
        except Exception as e:
            print(f"WARN: HF Buckets unavailable: {e}")
            self.store = None


state = ServerState()


def eager_download_model() -> None:
    """Pull policy.onnx + policy.json from HF Hub before the server
    accepts requests, so /model/policy.onnx works from second 0.
    """
    if not HF_REPO:
        print("HF_REPO_ID not set; skipping eager model download")
        return
    try:
        from huggingface_hub import hf_hub_download
        ONNX_DIR.mkdir(parents=True, exist_ok=True)
        for fname in ("policy.onnx", "policy.json"):
            hf_hub_download(
                repo_id=HF_REPO,
                filename=fname,
                token=HF_TOKEN,
                local_dir=str(ONNX_DIR),
            )
        print(f"Eagerly downloaded model from {HF_REPO}")
    except Exception as e:
        print(f"Eager download failed (server will still start): {e}")


def buffer_flush_loop() -> None:
    """Flush the in-memory buffer to HF Buckets when it reaches
    FLUSH_THRESHOLD entries or after FLUSH_INTERVAL_SEC seconds."""
    while True:
        time.sleep(10.0)
        try:
            with state.lock:
                ready = (
                    len(state.buffer) >= FLUSH_THRESHOLD
                    or (
                        state.buffer
                        and time.time() - state.last_flush > FLUSH_INTERVAL_SEC
                    )
                )
                if not ready:
                    continue
                steps = [dict(s) for s in state.buffer]
                state.buffer.clear()
            if state.store is None:
                # No bucket configured; drop the data so memory doesn't grow.
                state.last_flush = time.time()
                continue
            key = state.store.flush_buffer(steps)
            state.last_flush = time.time()
            print(f"Flushed {len(steps)} steps to {key}")
        except Exception as e:
            print(f"flush error: {e}")


def hf_hub_poll_loop() -> None:
    """Poll HF Hub every POLL_INTERVAL_SEC and re-download the ONNX
    when the repo SHA changes."""
    if not HF_REPO:
        print("HF_REPO_ID not set; skipping Hub polling")
        return
    last_sha: Optional[str] = None
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi(token=HF_TOKEN)
            info = api.repo_info(HF_REPO)
            sha = info.sha
            if sha == last_sha:
                continue
            print(f"New model revision detected: {sha}")
            ONNX_DIR.mkdir(parents=True, exist_ok=True)
            for fname in ("policy.onnx", "policy.json"):
                hf_hub_download(
                    repo_id=HF_REPO,
                    filename=fname,
                    token=HF_TOKEN,
                    local_dir=str(ONNX_DIR),
                )
            last_sha = sha
            state.version += 1
            print(f"Hot-reloaded model to internal version {state.version}")
        except Exception as e:
            print(f"hub poll error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.init_storage()
    state.version = 1
    state.last_flush = time.time()
    eager_download_model()
    threading.Thread(target=buffer_flush_loop, daemon=True).start()
    threading.Thread(target=hf_hub_poll_loop, daemon=True).start()
    print(f"Server up. ONNX dir={ONNX_DIR}, repo={HF_REPO or '(unset)'}")
    yield


app = FastAPI(title="Air Hockey RL Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "version": state.version}


@app.get("/model/version", response_model=VersionResponse)
def model_version():
    return VersionResponse(
        version=state.version,
        n_buffered=len(state.buffer),
        last_flush_unix=state.last_flush,
    )


@app.get("/model/policy.onnx")
def model_onnx():
    p = ONNX_DIR / "policy.onnx"
    if not p.exists():
        raise HTTPException(status_code=404, detail="model not yet downloaded")
    return FileResponse(p, media_type="application/octet-stream")


@app.get("/model/policy.json")
def model_meta():
    p = ONNX_DIR / "policy.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="metadata not yet downloaded")
    return FileResponse(p, media_type="application/json")


@app.post("/trajectory")
def submit_trajectory(traj: Trajectory):
    with state.lock:
        for s in traj.steps:
            state.buffer.append(s.model_dump())
        n = len(state.buffer)
    return {"ok": True, "n_buffered": n, "version": state.version}
