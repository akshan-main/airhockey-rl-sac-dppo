"""Upload the trained model + ONNX export + model card to HF Hub.

Run after a successful training + ONNX export:
    python scripts/upload_to_hub.py

Required environment:
    HF_TOKEN     Hugging Face token with write scope on the target repo
    HF_REPO_ID   target repo, e.g. akshan-main/airhockey-dppo
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


REPO_ID = os.environ.get("HF_REPO_ID", "akshan-main/airhockey-dppo")
TOKEN = os.environ.get("HF_TOKEN")
CKPT = "ckpt/dppo.pt"
ONNX_DIR = "onnx"
CARD = "MODEL_CARD.md"


def main() -> None:
    if not TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is required")

    api = HfApi(token=TOKEN)
    create_repo(REPO_ID, repo_type="model", exist_ok=True, token=TOKEN)
    print(f"Target repo: {REPO_ID}")

    files = [
        (CKPT, "dppo.pt"),
        (f"{ONNX_DIR}/policy.onnx", "policy.onnx"),
        (f"{ONNX_DIR}/policy.json", "policy.json"),
        (CARD, "README.md"),
    ]

    for local, remote in files:
        if not Path(local).exists():
            print(f"  ! missing: {local}")
            continue
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=REPO_ID,
            commit_message=f"Upload {remote}",
        )
        print(f"  ✓ uploaded {local} → {REPO_ID}/{remote}")

    print(f"\nDone. View at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
