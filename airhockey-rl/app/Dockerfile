# Hugging Face Spaces Docker SDK image for the air hockey RL backend.
#
# Builds a small FastAPI server that:
#   • serves the trained ONNX model to the browser,
#   • buffers human-vs-AI gameplay trajectories,
#   • flushes them periodically to HF Buckets,
#   • hot-reloads the served model when a new HF Hub revision appears.
#
# The server does NOT train anything. Training runs offline in Colab
# (see scripts/colab_train.md) and the periodic retrain is a GitHub
# Actions cron job that invokes airhockey/retrain_cycle.py. Neither is
# part of this Docker image — which is why we don't install torch,
# onnx, gymnasium, or any of the training-side dependencies here.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# The server only needs airhockey/storage.py for the HF Buckets client.
# Copy only what we actually use at runtime.
COPY pyproject.toml /app/pyproject.toml
COPY airhockey/__init__.py /app/airhockey/__init__.py
COPY airhockey/storage.py /app/airhockey/storage.py
COPY app /app/app

# Install ONLY the server's runtime deps. No torch, no gymnasium.
RUN pip install --upgrade pip && \
    pip install \
      "fastapi>=0.110" \
      "uvicorn[standard]>=0.27" \
      "pydantic>=2" \
      "huggingface_hub>=0.20" \
      "boto3>=1.34" \
      "numpy>=1.26"

# Model files land here at runtime via the HF Hub poller
RUN mkdir -p /app/onnx

# HF Spaces injects PORT; default 7860
EXPOSE 7860
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
