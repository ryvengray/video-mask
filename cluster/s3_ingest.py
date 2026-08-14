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

    MULTIPART_PART_SIZE = 64 * 1024 * 1024

    def __init__(self, store: ClusterStore, source_bucket: str, source_prefix: str,
                 output_bucket: str, output_prefix: str, source_region: str,
                 output_region: str | None = None, profile: str | None = None,
                 poll_seconds: int = 60, presign_seconds: int = 86400,
                 client: Any | None = None, output_client: Any | None = None):
        self.store = store
        self.source_bucket = source_bucket
        self.source_prefix = source_prefix.strip("/")
        self.output_bucket = output_bucket
        self.output_prefix = output_prefix.strip("/")
        self.source_region = source_region
        self.output_region = output_region or source_region
        self.poll_seconds = poll_seconds
        self.presign_seconds = presign_seconds
        self._lock = threading.Lock()
        self._last_scan = 0.0
        if client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - deployment dependency guard
                raise RuntimeError("S3 mode requires boto3; install requirements-cluster.txt") from exc
            session = boto3.Session(profile_name=profile or None)

            def regional_client(region: str):
                # Pre-sign against the Bucket's actual regional endpoint.
                endpoint_url = f"https://s3.{region}.amazonaws.com" if region else None
                return session.client("s3", region_name=region or None, endpoint_url=endpoint_url)

            client = regional_client(self.source_region)
            output_client = regional_client(self.output_region)
        self.source_client = client
        self.output_client = output_client or client

    def output_key(self, source_key: str) -> str:
        """Build an output key while preserving the source-relative directory.

        For example, with ``source_prefix=source/inbox`` and
        ``output_prefix=outputs``, ``source/inbox/s/s/m.mp4`` becomes
        ``outputs/s/s/masked_m.mp4``.  Keeping the relative path prevents
        identically named videos in different source folders from overwriting
        one another in the output bucket.
        """
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
            self.output_client.head_object(Bucket=self.output_bucket, Key=key)
            return True
        except Exception as exc:
            if self._not_found(exc):
                return False
            raise

    def validate_bucket_regions(self) -> None:
        """Fail fast when a configured signing Region does not match its Bucket."""
        for label, bucket, region, client in (
            ("source", self.source_bucket, self.source_region, self.source_client),
            ("output", self.output_bucket, self.output_region, self.output_client),
        ):
            try:
                location = client.get_bucket_location(Bucket=bucket).get("LocationConstraint")
            except Exception as exc:
                raise RuntimeError(f"cannot determine {label} bucket region for {bucket}: {exc}") from exc
            bucket_region = "us-east-1" if location in (None, "") else str(location)
            if bucket_region != region:
                raise RuntimeError(
                    f"{label} bucket {bucket} is in {bucket_region}, but its configured region is {region}"
                )

    def scan_if_due(self) -> int:
        with self._lock:
            if time.monotonic() - self._last_scan < self.poll_seconds:
                return 0
            created = self._scan()
            self._last_scan = time.monotonic()
            return created

    def scan(self) -> int:
        """Scan immediately, serializing with scheduled and request-triggered scans."""
        with self._lock:
            created = self._scan()
            self._last_scan = time.monotonic()
            return created

    def _scan(self) -> int:
        """Scan while ``_lock`` is held by the caller."""
        created = 0
        paginator = self.source_client.get_paginator("list_objects_v2")
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
        result["source_url"] = self.source_client.generate_presigned_url(
            "get_object", Params={"Bucket": self.source_bucket, "Key": source_key},
            ExpiresIn=self.presign_seconds,
        )
        result["output_upload_url"] = self.output_client.generate_presigned_url(
            "put_object", Params={"Bucket": self.output_bucket, "Key": output_key},
            ExpiresIn=self.presign_seconds,
        )
        return result

    def materialize_upload_url(self, task: dict[str, Any]) -> dict[str, str]:
        """Issue a fresh output PUT URL without exposing the source URL again."""
        parsed = urlparse(str(task.get("source_url") or ""))
        if parsed.scheme != "s3" or parsed.netloc != self.source_bucket:
            return {"output_upload_url": str(task.get("output_upload_url") or "")}
        source_key = parsed.path.lstrip("/")
        output_key = str(task.get("output_object_key") or self.output_key(source_key))
        return {
            "output_upload_url": self.output_client.generate_presigned_url(
                "put_object", Params={"Bucket": self.output_bucket, "Key": output_key},
                ExpiresIn=self.presign_seconds,
            ),
            "output_object_key": output_key,
        }

    def _output_key_for_task(self, task: dict[str, Any]) -> str:
        parsed = urlparse(str(task.get("source_url") or ""))
        if parsed.scheme != "s3" or parsed.netloc != self.source_bucket:
            raise ValueError("task does not use the configured S3 source bucket")
        return str(task.get("output_object_key") or self.output_key(parsed.path.lstrip("/")))

    def initiate_multipart_upload(self, task: dict[str, Any]) -> dict[str, Any]:
        output_key = self._output_key_for_task(task)
        response = self.output_client.create_multipart_upload(Bucket=self.output_bucket, Key=output_key)
        upload_id = str(response.get("UploadId") or "")
        if not upload_id:
            raise RuntimeError("S3 did not return an UploadId for multipart upload")
        return {"upload_id": upload_id, "part_size": self.MULTIPART_PART_SIZE,
                "output_object_key": output_key}

    def multipart_part_url(self, task: dict[str, Any], upload_id: str, part_number: int) -> dict[str, str]:
        if not 1 <= part_number <= 10_000:
            raise ValueError("multipart part_number must be 1..10000")
        return {"upload_part_url": self.output_client.generate_presigned_url(
            "upload_part", Params={"Bucket": self.output_bucket, "Key": self._output_key_for_task(task),
                                   "UploadId": upload_id, "PartNumber": part_number},
            ExpiresIn=self.presign_seconds,
        )}

    def complete_multipart_upload(self, task: dict[str, Any], upload_id: str,
                                  parts: list[dict[str, Any]]) -> dict[str, str]:
        if not parts:
            raise ValueError("multipart upload requires at least one part")
        normalized = []
        for expected, part in enumerate(parts, start=1):
            number, etag = int(part.get("part_number") or 0), str(part.get("etag") or "")
            if number != expected or not etag:
                raise ValueError("multipart parts must be ordered, consecutive, and include ETags")
            normalized.append({"PartNumber": number, "ETag": etag})
        output_key = self._output_key_for_task(task)
        self.output_client.complete_multipart_upload(
            Bucket=self.output_bucket, Key=output_key, UploadId=upload_id,
            MultipartUpload={"Parts": normalized},
        )
        return {"output_object_key": output_key}

    def abort_multipart_upload(self, task: dict[str, Any], upload_id: str) -> None:
        self.output_client.abort_multipart_upload(
            Bucket=self.output_bucket, Key=self._output_key_for_task(task), UploadId=upload_id,
        )
