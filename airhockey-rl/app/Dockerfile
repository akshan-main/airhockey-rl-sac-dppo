# HF Spaces Docker SDK image for the airhockey-rl backend.
# Runs only the FastAPI server — no training deps (torch, gymnasium,
# onnx) are installed here. Training runs offline on Colab; the
# periodic retrain runs in a GitHub Actions workflow.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY airhockey/__init__.py /app/airhockey/__init__.py
COPY airhockey/storage.py /app/airhockey/storage.py
COPY app /app/app

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
