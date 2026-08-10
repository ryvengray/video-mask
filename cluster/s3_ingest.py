"""S3 task ingestion and just-in-time presigned URL materialization."""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from cluster.local_ingest import DEFAULT_ALGORITHM, DEFAULT_ARGS, VIDEO_SUFFIXES
from cluster.store import ClusterStore


class S3Ingestor:
    """Scan a source bucket and create idempotent tasks for video objects.

    Credentials are used only on the Controller.  Workers receive short-lived
    GET/PUT URLs when they claim a task, so URLs do not expire while queued.
    """

    def __init__(self, store: ClusterStore, source_bucket: str, source_prefix: str,
                 output_bucket: str, output_prefix: str, region: str,
                 profile: str | None = None, poll_seconds: int = 60,
                 presign_seconds: int = 86400, client: Any | None = None):
        self.store = store
        self.source_bucket = source_bucket
        self.source_prefix = source_prefix.strip("/")
        self.output_bucket = output_bucket
        self.output_prefix = output_prefix.strip("/")
        self.poll_seconds = poll_seconds
        self.presign_seconds = presign_seconds
        self._lock = threading.Lock()
        self._last_scan = 0.0
        if client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - deployment dependency guard
                raise RuntimeError("S3 mode requires boto3; install requirements-cluster.txt") from exc
            session = boto3.Session(profile_name=profile or None, region_name=region or None)
            client = session.client("s3", region_name=region or None)
        self.client = client

    def output_key(self, source_key: str) -> str:
        relative = source_key
        prefix = f"{self.source_prefix}/" if self.source_prefix else ""
        if prefix and source_key.startswith(prefix):
            relative = source_key[len(prefix):]
        path = PurePosixPath(relative)
        filename = f"masked_{path.stem}.mp4"
        parts = [part for part in (self.output_prefix, str(path.parent), filename) if part and part != "."]
        return "/".join(parts)

    @staticmethod
    def _not_found(exc: Exception) -> bool:
        response = getattr(exc, "response", {}) or {}
        code = str((response.get("Error") or {}).get("Code", ""))
        return code in {"404", "NoSuchKey", "NotFound"}

    def output_exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.output_bucket, Key=key)
            return True
        except Exception as exc:
            if self._not_found(exc):
                return False
            raise

    def scan_if_due(self) -> int:
        with self._lock:
            if time.monotonic() - self._last_scan < self.poll_seconds:
                return 0
            created = self.scan()
            self._last_scan = time.monotonic()
            return created

    def scan(self) -> int:
        created = 0
        paginator = self.client.get_paginator("list_objects_v2")
        prefix = f"{self.source_prefix}/" if self.source_prefix else ""
        for page in paginator.paginate(Bucket=self.source_bucket, Prefix=prefix):
            for object_info in page.get("Contents", []):
                key = str(object_info["Key"])
                path = PurePosixPath(key)
                if path.suffix.lower() not in VIDEO_SUFFIXES or path.name.startswith("masked_"):
                    continue
                if int(object_info.get("Size") or 0) <= 0:
                    continue
                output_key = self.output_key(key)
                if self.output_exists(output_key):
                    continue
                version = str(object_info.get("VersionId") or object_info.get("ETag") or object_info.get("LastModified") or "")
                task_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"s3:{self.source_bucket}:{key}:{version}"))
                try:
                    self.store.create_task({
                        "task_id": task_id,
                        # Stored as an identifier only.  A fresh HTTPS URL is issued at claim time.
                        "source_url": f"s3://{self.source_bucket}/{key}",
                        "source_object_key": key,
                        "source_size_bytes": int(object_info.get("Size") or 0),
                        "algorithm": DEFAULT_ALGORITHM,
                        "arguments": DEFAULT_ARGS,
                        "output_object_key": output_key,
                    })
                    created += 1
                except sqlite3.IntegrityError:
                    pass
        return created

    def materialize(self, task: dict[str, Any]) -> dict[str, Any]:
        """Return a task copy with fresh presigned URLs for a Worker."""
        parsed = urlparse(str(task.get("source_url") or ""))
        if parsed.scheme != "s3" or parsed.netloc != self.source_bucket:
            return task
        source_key = parsed.path.lstrip("/")
        output_key = str(task.get("output_object_key") or self.output_key(source_key))
        result = dict(task)
        result["source_url"] = self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.source_bucket, "Key": source_key},
            ExpiresIn=self.presign_seconds,
        )
        result["output_upload_url"] = self.client.generate_presigned_url(
            "put_object", Params={"Bucket": self.output_bucket, "Key": output_key},
            ExpiresIn=self.presign_seconds,
        )
        return result
