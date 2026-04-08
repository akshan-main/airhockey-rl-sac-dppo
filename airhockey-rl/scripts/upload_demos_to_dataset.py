"""Upload data/demos.npz + DATASET_CARD.md to a HF Dataset repo.

Run after `python -m airhockey.collect`:
    python scripts/upload_demos_to_dataset.py

Required environment:
    HF_TOKEN          write-scope token
    HF_DATASET_REPO   target dataset repo, e.g. akshan-main/airhockey-demos
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


REPO_ID = os.environ.get("HF_DATASET_REPO", "akshan-main/airhockey-demos")
TOKEN = os.environ.get("HF_TOKEN")
DEMOS = "data/demos.npz"
CARD = "DATASET_CARD.md"


def main() -> None:
    if not TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is required")

    api = HfApi(token=TOKEN)
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True, token=TOKEN)
    print(f"Target repo: {REPO_ID}")

    if not Path(DEMOS).exists():
        raise FileNotFoundError(
            f"{DEMOS} not found. Run `python -m airhockey.collect` first."
        )

    api.upload_file(
        path_or_fileobj=DEMOS,
        path_in_repo="demos.npz",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Upload demonstration trajectories",
    )
    print(f"  ✓ uploaded {DEMOS} → {REPO_ID}/demos.npz")

    api.upload_file(
        path_or_fileobj=CARD,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Upload dataset card",
    )
    print(f"  ✓ uploaded {CARD} → {REPO_ID}/README.md")

    print(f"\nDone. View at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
