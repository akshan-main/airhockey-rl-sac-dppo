"""HF Buckets storage client.

HF Buckets is Hugging Face's S3-compatible object storage. We use the
standard boto3 client pointed at the HF endpoint. Same code would work
against AWS S3, MinIO, Cloudflare R2, or any other S3-compatible store
just by changing the endpoint URL.

Layout in the bucket:
    buffered/YYYY-MM-DD/HH-MM-SS-<uuid>.npz   raw trajectory shards
    processed/                                  marker objects (empty body)
                                                 created after a shard has
                                                 been consumed by the trainer

Lifecycle policy on the HF bucket dashboard auto-deletes anything under
buffered/ older than 7 days, so disposed data goes away on its own.

Environment:
    HF_BUCKET_NAME       name of the bucket
    HF_BUCKET_ENDPOINT   e.g. https://buckets.hf.co
    HF_BUCKET_KEY_ID     access key id
    HF_BUCKET_SECRET     secret access key
"""
from __future__ import annotations

import io
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

try:
    import boto3
    from botocore.config import Config
except ImportError as e:  # pragma: no cover
    boto3 = None
    Config = None


@dataclass
class StorageConfig:
    bucket: str = os.environ.get("HF_BUCKET_NAME", "airhockey-rl-buffer")
    endpoint: str = os.environ.get("HF_BUCKET_ENDPOINT", "https://buckets.hf.co")
    key_id: str = os.environ.get("HF_BUCKET_KEY_ID", "")
    secret: str = os.environ.get("HF_BUCKET_SECRET", "")
    region: str = os.environ.get("HF_BUCKET_REGION", "auto")


class HFBucketStore:
    """Thin S3 wrapper for the air hockey buffered trajectories."""

    def __init__(self, cfg: StorageConfig | None = None):
        if boto3 is None:
            raise ImportError("boto3 is required: pip install boto3")
        self.cfg = cfg or StorageConfig()
        self.client = boto3.client(
            "s3",
            endpoint_url=self.cfg.endpoint,
            aws_access_key_id=self.cfg.key_id,
            aws_secret_access_key=self.cfg.secret,
            region_name=self.cfg.region,
            config=Config(signature_version="s3v4", retries={"max_attempts": 3}),
        )

    # ── Write ─────────────────────────────────────────────────
    def flush_buffer(self, steps: list[dict]) -> str:
        """Serialize a list of step dicts to .npz and upload as a new shard.
        Returns the object key written."""
        if not steps:
            return ""
        obs = np.array([s["obs"] for s in steps], dtype=np.float32)
        act = np.array([s["action"] for s in steps], dtype=np.float32)
        rew = np.array([s["reward"] for s in steps], dtype=np.float32)
        done = np.array([s["done"] for s in steps], dtype=np.float32)
        buf = io.BytesIO()
        np.savez_compressed(buf, obs=obs, act=act, reward=rew, done=done)
        buf.seek(0)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d/%H-%M-%S")
        key = f"buffered/{ts}-{uuid.uuid4().hex[:8]}.npz"
        self.client.put_object(
            Bucket=self.cfg.bucket,
            Key=key,
            Body=buf.read(),
            ContentType="application/octet-stream",
        )
        return key

    # ── Read ──────────────────────────────────────────────────
    def list_unprocessed_shards(self) -> list[str]:
        """List shard keys under buffered/ that have no matching processed/
        marker. Caller is responsible for ordering / batching."""
        all_buf = self._list_prefix("buffered/")
        processed = set(self._list_prefix("processed/"))
        # Marker key for shard X is processed/X
        return [k for k in all_buf if f"processed/{k}" not in processed]

    def download_shard(self, key: str) -> dict[str, np.ndarray]:
        obj = self.client.get_object(Bucket=self.cfg.bucket, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        d = np.load(buf)
        return {k: d[k] for k in d.files}

    def mark_processed(self, key: str) -> None:
        marker = f"processed/{key}"
        self.client.put_object(
            Bucket=self.cfg.bucket,
            Key=marker,
            Body=b"",
            ContentType="text/plain",
        )

    # ── Internals ─────────────────────────────────────────────
    def _list_prefix(self, prefix: str) -> list[str]:
        keys: list[str] = []
        token = None
        while True:
            kw = {"Bucket": self.cfg.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kw)
            for o in resp.get("Contents", []):
                keys.append(o["Key"])
            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break
        return keys
