#!/usr/bin/env python3
"""FastAPI controller for pull-based video-mask workers."""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import random
import shutil
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("Install cluster requirements: pip install -r requirements-cluster.txt") from exc

from cluster.store import ClusterStore
from cluster.local_ingest import DEFAULT_ALGORITHM, DEFAULT_ARGS, VIDEO_SUFFIXES, LocalIngestor
from cluster.s3_ingest import S3Ingestor


logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).with_name("web")
STATUS_TONES = {
    "ready": "success", "completed": "success",
    "busy": "active", "assigned": "active", "downloading": "active",
    "processing": "active", "uploading": "active",
    "pending": "pending", "cancelling": "warning", "cancelled": "warning",
    "offline": "muted", "failed": "danger",
}
TASK_FILTER_STATUSES = (
    "pending", "assigned", "downloading", "processing", "uploading",
    "cancelling", "completed", "failed", "cancelled",
)
FACE_REVIEW_LEASE_SECONDS = 300
FACE_ANNOTATION_FILTERS = ("has_face", "no_face", "unlabelled")
MANUAL_SOURCE_PREFIX = "test/manual"
MANUAL_UPLOAD_MAX_BYTES = 20 * 1024 * 1024 * 1024
FRAME_PREVIEW_HEIGHT = 240
FRAME_PREVIEW_PER_MINUTE = 2
FRAME_PREVIEW_MAX_IMAGES = 24


def frame_preview_timestamps(task_id: str, duration_seconds: float) -> list[float]:
    """Return stable, spread-out low-resolution frame sample times."""
    if duration_seconds <= 0:
        return []
    minute_count = max(1, int((duration_seconds + 59) // 60))
    bucket_count = min(minute_count, FRAME_PREVIEW_MAX_IMAGES // FRAME_PREVIEW_PER_MINUTE)
    randomizer = random.Random(task_id)
    samples: list[float] = []
    for bucket in range(bucket_count):
        start = duration_seconds * bucket / bucket_count
        end = duration_seconds * (bucket + 1) / bucket_count
        padding = min(2.0, max(0.0, (end - start) / 8))
        for _ in range(FRAME_PREVIEW_PER_MINUTE):
            samples.append(randomizer.uniform(start + padding, max(start + padding, end - padding)))
    return sorted(min(duration_seconds - 0.1, max(0.0, value)) for value in samples)


def statistics_window(start_value: str | None, end_value: str | None) -> tuple[float, float]:
    """Parse dashboard date controls, defaulting to the most recent three days."""
    local_timezone = dt.datetime.now().astimezone().tzinfo

    def parse(value: str | None, fallback: float) -> float:
        if not value:
            return fallback
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("statistics dates must use ISO 8601 format") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=local_timezone)
        return parsed.timestamp()

    end_at = parse(end_value, time.time())
    start_at = parse(start_value, end_at - 3 * 24 * 3600)
    if end_at <= start_at:
        raise ValueError("statistics end time must be after its start time")
    if end_at - start_at > 90 * 24 * 3600:
        raise ValueError("statistics range cannot exceed 90 days")
    return start_at, end_at


def datetime_local(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%dT%H:%M")


class WorkerRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=128)
    token: str = Field(min_length=16)
    capabilities: dict[str, Any] = Field(default_factory=dict)


class HeartbeatRequest(WorkerRequest):
    status: str = "ready"


class ProgressRequest(WorkerRequest):
    status: str
    progress: dict[str, Any] = Field(default_factory=dict)


class TaskLogsRequest(WorkerRequest):
    lines: list[str] = Field(min_length=1, max_length=50)


class FinishRequest(WorkerRequest):
    output_sha256: str | None = None
    output_duration_seconds: float | None = None
    error_message: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)


class MultipartPart(BaseModel):
    part_number: int = Field(ge=1, le=10_000)
    etag: str = Field(min_length=1, max_length=512)


class MultipartUploadRequest(WorkerRequest):
    upload_id: str = Field(min_length=1, max_length=2048)


class MultipartPartUrlRequest(MultipartUploadRequest):
    part_number: int = Field(ge=1, le=10_000)


class MultipartCompleteRequest(MultipartUploadRequest):
    parts: list[MultipartPart] = Field(min_length=1, max_length=10_000)


class TaskRequest(BaseModel):
    source_url: str
    output_upload_url: str | None = None
    source_object_key: str | None = None
    source_size_bytes: int | None = None
    source_sha256: str | None = None
    source_duration_seconds: float | None = None
    algorithm: str = DEFAULT_ALGORITHM
    arguments: list[str] = Field(default_factory=lambda: list(DEFAULT_ARGS))
    output_object_key: str | None = None
    max_attempts: int = Field(default=3, ge=1, le=20)


class WorkerProvisionRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=128)
    token: str = Field(min_length=16)


class S3IngestSwitchRequest(BaseModel):
    enabled: bool


class FaceReviewRequest(BaseModel):
    reviewer_id: str = Field(min_length=16, max_length=128)


class FaceAnnotationRequest(FaceReviewRequest):
    has_face: bool


def create_app(database: Path, admin_token: str, stale_after_seconds: int = 90,
               local_source_dir: Path | None = None, local_output_dir: Path | None = None,
               s3_ingestor: S3Ingestor | None = None,
               s3_config: dict[str, Any] | None = None) -> FastAPI:
    store = ClusterStore(database)
    if s3_ingestor and s3_config:
        raise ValueError("s3_ingestor and s3_config are mutually exclusive")
    if s3_config:
        s3_ingestor = S3Ingestor(store, **s3_config)
    local_ingestor = (LocalIngestor(store, local_source_dir, local_output_dir)
                      if local_source_dir and local_output_dir else None)
    # This switch only controls discovering *new* objects from S3.  Existing
    # S3 tasks remain claimable and still receive fresh signed download/upload
    # URLs, so pausing ingestion never strands work already in the queue.
    s3_ingest_enabled = bool(s3_ingestor) and store.boolean_setting("s3_ingest_enabled", True)

    def s3_ingest_is_enabled() -> bool:
        return bool(s3_ingestor) and s3_ingest_enabled

    def model_data(model: BaseModel) -> dict[str, Any]:
        # FastAPI 0.110 uses Pydantic v2, while some supported deployments
        # still resolve Pydantic v1.
        return model.model_dump() if hasattr(model, "model_dump") else model.dict()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        scan_task: asyncio.Task[None] | None = None
        if s3_ingestor:
            if s3_ingest_is_enabled():
                s3_ingestor.validate_bucket_regions()

            async def scan_s3_forever() -> None:
                """Run S3 ingestion and lease recovery without HTTP traffic."""
                while True:
                    try:
                        if s3_ingest_is_enabled():
                            # boto3 and SQLite work are synchronous; keep them off the API event loop.
                            requeued = await asyncio.to_thread(store.requeue_stale, stale_after_seconds)
                            if requeued:
                                logger.info("Recovered %d stale or orphaned task(s)", requeued)
                            created = await asyncio.to_thread(s3_ingestor.scan)
                            if created:
                                logger.info("S3 scan created %d task(s)", created)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # A temporary IAM, DNS, or S3 problem must not permanently stop ingestion.
                        logger.exception("S3 scan failed; will retry")
                    await asyncio.sleep(s3_ingestor.poll_seconds)

            scan_task = asyncio.create_task(scan_s3_forever(), name="video-mask-s3-ingest")
        try:
            yield
        finally:
            if scan_task:
                scan_task.cancel()
                with suppress(asyncio.CancelledError):
                    await scan_task
            store.close()

    app = FastAPI(title="Video Mask Cluster Controller", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=WEB_DIR / "templates")

    def require_admin(authorization: str | None) -> None:
        expected = f"Bearer {admin_token}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid admin token")

    def worker_capabilities(capabilities: dict[str, Any], http_request: Request) -> dict[str, Any]:
        """Record the private address seen by the Controller, not a user-supplied address."""
        result = dict(capabilities)
        if http_request.client and http_request.client.host:
            result["controller_seen_ip"] = http_request.client.host
        return result

    @app.exception_handler(PermissionError)
    async def permission_error(_, exc: PermissionError):
        return JSONResponse(status_code=401, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error(_, exc: ValueError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        requeued = store.requeue_stale(stale_after_seconds)
        ingested = local_ingestor.scan() if local_ingestor else 0
        s3_ingested = s3_ingestor.scan_if_due() if s3_ingest_is_enabled() else 0
        return {"status": "ok", "requeued": str(requeued), "ingested": str(ingested),
                "s3_ingested": str(s3_ingested),
                "s3_ingest_enabled": str(s3_ingest_is_enabled()).lower()}

    @app.post("/api/workers/register")
    def register_worker(request: WorkerRequest, http_request: Request):
        return store.register_worker(request.worker_id, request.token,
                                     worker_capabilities(request.capabilities, http_request))

    @app.post("/api/workers/{worker_id}/heartbeat")
    def heartbeat(worker_id: str, request: HeartbeatRequest, http_request: Request):
        if request.worker_id != worker_id:
            raise ValueError("worker id does not match path")
        worker = store.heartbeat(worker_id, request.token, request.status,
                                 worker_capabilities(request.capabilities, http_request))
        current_task_id = worker.get("current_task_id")
        task = store.task(str(current_task_id)) if current_task_id else None
        return {**worker, "cancel_requested": bool(task and task.get("status") == "cancelling")}

    @app.post("/api/workers/{worker_id}/claim")
    def claim(worker_id: str, request: WorkerRequest):
        if request.worker_id != worker_id:
            raise ValueError("worker id does not match path")
        store.requeue_stale(stale_after_seconds)
        if local_ingestor:
            local_ingestor.scan()
        if s3_ingest_is_enabled():
            s3_ingestor.scan_if_due()
        task = store.claim(worker_id, request.token)
        return {"task": s3_ingestor.materialize(task) if task and s3_ingestor else task}

    @app.post("/api/workers/{worker_id}/tasks/{task_id}/upload-url")
    def refresh_upload_url(worker_id: str, task_id: str, request: WorkerRequest):
        if request.worker_id != worker_id:
            raise ValueError("worker id does not match path")
        task = store.active_task_for_worker(worker_id, request.token, task_id)
        if s3_ingestor:
            return s3_ingestor.materialize_upload_url(task)
        return {"output_upload_url": str(task.get("output_upload_url") or ""),
                "output_object_key": str(task.get("output_object_key") or "")}

    @app.post("/api/workers/{worker_id}/tasks/{task_id}/multipart/start")
    def start_multipart_upload(worker_id: str, task_id: str, request: WorkerRequest):
        if request.worker_id != worker_id:
            raise ValueError("worker id does not match path")
        task = store.active_task_for_worker(worker_id, request.token, task_id)
        if not s3_ingestor:
            raise ValueError("multipart upload is available only for S3 tasks")
        return s3_ingestor.initiate_multipart_upload(task)

    @app.post("/api/workers/{worker_id}/tasks/{task_id}/multipart/part-url")
    def multipart_part_url(worker_id: str, task_id: str, request: MultipartPartUrlRequest):
        if request.worker_id != worker_id:
            raise ValueError("worker id does not match path")
        task = store.active_task_for_worker(worker_id, request.token, task_id)
        if not s3_ingestor:
            raise ValueError("multipart upload is available only for S3 tasks")
        return s3_ingestor.multipart_part_url(task, request.upload_id, request.part_number)

    @app.post("/api/workers/{worker_id}/tasks/{task_id}/multipart/complete")
    def complete_multipart_upload(worker_id: str, task_id: str, request: MultipartCompleteRequest):
        if request.worker_id != worker_id:
            raise ValueError("worker id does not match path")
        task = store.active_task_for_worker(worker_id, request.token, task_id)
        if not s3_ingestor:
            raise ValueError("multipart upload is available only for S3 tasks")
        return s3_ingestor.complete_multipart_upload(task, request.upload_id,
                                                      [model_data(part) for part in request.parts])

    @app.post("/api/workers/{worker_id}/tasks/{task_id}/multipart/abort")
    def abort_multipart_upload(worker_id: str, task_id: str, request: MultipartUploadRequest):
        if request.worker_id != worker_id:
            raise ValueError("worker id does not match path")
        task = store.active_task_for_worker(worker_id, request.token, task_id)
        if not s3_ingestor:
            raise ValueError("multipart upload is available only for S3 tasks")
        s3_ingestor.abort_multipart_upload(task, request.upload_id)
        return {"status": "aborted"}

    @app.post("/api/tasks/{task_id}/progress")
    def progress(task_id: str, request: ProgressRequest):
        return store.progress(request.worker_id, request.token, task_id, request.status, request.progress)

    @app.post("/api/tasks/{task_id}/logs")
    def append_task_logs(task_id: str, request: TaskLogsRequest) -> dict[str, int]:
        return {"appended": store.append_task_logs(request.worker_id, request.token, task_id, request.lines)}

    @app.post("/api/tasks/{task_id}/complete")
    def complete(task_id: str, request: FinishRequest):
        return store.finish(request.worker_id, request.token, task_id, True, model_data(request))

    @app.post("/api/tasks/{task_id}/fail")
    def fail(task_id: str, request: FinishRequest):
        return store.finish(request.worker_id, request.token, task_id, False, model_data(request))

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        task = store.task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task does not exist")
        return task

    def task_playback_url(task_id: str, file: str) -> str:
        """Create a short-lived S3 URL without exposing it to the browser."""
        task = store.task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task does not exist")
        if not s3_ingestor:
            raise HTTPException(status_code=409, detail="playback URLs are only available for S3 tasks")
        if file == "input":
            source_key = str(task.get("source_object_key") or "")
            if not source_key:
                raise HTTPException(status_code=404, detail="task has no source object key")
            url = s3_ingestor.source_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": s3_ingestor.source_bucket, "Key": source_key},
                ExpiresIn=s3_ingestor.presign_seconds,
            )
        elif file == "output":
            output_key = str(task.get("output_object_key") or "")
            if not output_key:
                raise HTTPException(status_code=404, detail="task has no output object key")
            url = s3_ingestor.output_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": s3_ingestor.output_bucket, "Key": output_key},
                ExpiresIn=s3_ingestor.presign_seconds,
            )
        else:
            raise HTTPException(status_code=400, detail="file must be 'input' or 'output'")
        return url

    def task_playback_filename(task: dict[str, Any], file: str) -> str:
        key = task.get("output_object_key") if file == "output" else task.get("source_object_key")
        filename = Path(str(key or "")).name
        return filename or ("output.mp4" if file == "output" else "input.mp4")

    frame_preview_root = database.parent / "frame-previews"
    frame_preview_jobs: set[tuple[str, str, str]] = set()
    frame_preview_lock = threading.Lock()

    def frame_preview_dir(task_id: str, file: str) -> Path:
        # Task IDs originate from an API, so keep cache paths independent of
        # their spelling and never allow a path component to be user-controlled.
        return frame_preview_root / hashlib.sha256(task_id.encode("utf-8")).hexdigest() / file

    def frame_preview_manifest(directory: Path) -> dict[str, Any]:
        try:
            return json.loads((directory / "manifest.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def write_frame_preview_manifest(directory: Path, payload: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "manifest.json.partial"
        temporary.write_text(json.dumps(payload, separators=(",", ":")))
        temporary.replace(directory / "manifest.json")

    def frame_preview_response(task_id: str, file: str, manifest: dict[str, Any]) -> dict[str, Any]:
        frames = manifest.get("frames") or []
        return {
            "state": str(manifest.get("state") or "running"),
            "file": file,
            "frames": [{
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "url": f"/api/tasks/{urllib.parse.quote(task_id, safe='')}/frame-previews/files/"
                       f"{urllib.parse.quote(str(frame.get('filename') or ''), safe='')}?file={file}",
            } for frame in frames if frame.get("filename")],
            "error": str(manifest.get("error") or ""),
        }

    def generate_frame_previews(task_id: str, file: str, source_url: str, fingerprint: str,
                                duration_hint: float | None) -> None:
        directory = frame_preview_dir(task_id, file)
        job_key = (task_id, file, fingerprint)
        manifest: dict[str, Any] = {"state": "running", "fingerprint": fingerprint, "frames": []}
        try:
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True, exist_ok=True)
            duration_seconds = duration_hint
            if not isinstance(duration_seconds, (int, float)) or duration_seconds <= 0:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", source_url],
                    text=True, capture_output=True, check=False, timeout=60,
                )
                try:
                    duration_seconds = float(probe.stdout.strip())
                except ValueError as exc:
                    raise RuntimeError("unable to determine video duration for frame preview") from exc
            timestamps = frame_preview_timestamps(task_id, float(duration_seconds))
            if not timestamps:
                raise RuntimeError("video duration is empty")
            manifest["duration_seconds"] = duration_seconds
            write_frame_preview_manifest(directory, manifest)
            for index, timestamp in enumerate(timestamps, start=1):
                filename = f"frame-{index:02d}.jpg"
                image = directory / filename
                command = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
                    "-i", source_url, "-map", "0:v:0", "-frames:v", "1",
                    "-vf", f"scale=-2:{FRAME_PREVIEW_HEIGHT}", "-q:v", "7", "-y", str(image),
                ]
                completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=90)
                if completed.returncode:
                    logger.warning("Frame preview %s/%s at %.1fs failed: %s", task_id, file, timestamp,
                                   completed.stderr[-500:].strip())
                    continue
                manifest["frames"].append({"filename": filename, "timestamp_seconds": round(timestamp, 1)})
                write_frame_preview_manifest(directory, manifest)
            if not manifest["frames"]:
                raise RuntimeError("ffmpeg could not extract any video frames")
            manifest["state"] = "ready"
            write_frame_preview_manifest(directory, manifest)
        except Exception as exc:
            logger.warning("Frame preview generation failed for %s: %s", task_id, exc)
            manifest.update({"state": "error", "error": str(exc)[:500]})
            write_frame_preview_manifest(directory, manifest)
        finally:
            with frame_preview_lock:
                frame_preview_jobs.discard(job_key)

    @app.get("/api/tasks/{task_id}/frame-previews")
    def get_frame_previews(task_id: str) -> dict[str, Any]:
        task = store.task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task does not exist")
        if task.get("status") != "completed":
            raise HTTPException(status_code=409, detail="frame previews are available after task completion")
        if not s3_ingestor:
            raise HTTPException(status_code=409, detail="frame previews are only available for S3 tasks")
        file = "output" if task.get("output_object_key") else "input"
        fingerprint = "|".join(str(value or "") for value in (
            file, task.get("output_object_key") if file == "output" else task.get("source_object_key"),
            task.get("output_sha256") if file == "output" else task.get("source_sha256"),
            task.get("finished_at") if file == "output" else task.get("created_at"),
        ))
        directory = frame_preview_dir(task_id, file)
        manifest = frame_preview_manifest(directory)
        if manifest.get("fingerprint") == fingerprint and manifest.get("state") in {"running", "ready", "error"}:
            return frame_preview_response(task_id, file, manifest)
        job_key = (task_id, file, fingerprint)
        with frame_preview_lock:
            if job_key not in frame_preview_jobs:
                frame_preview_jobs.add(job_key)
                duration_hint = (task.get("output_duration_seconds") if file == "output"
                                 else task.get("source_duration_seconds"))
                threading.Thread(
                    target=generate_frame_previews,
                    args=(task_id, file, task_playback_url(task_id, file), fingerprint, duration_hint),
                    name=f"frame-preview-{task_id[:8]}", daemon=True,
                ).start()
        return frame_preview_response(task_id, file, {"state": "running", "frames": []})

    @app.get("/api/tasks/{task_id}/frame-previews/files/{filename}")
    def get_frame_preview_file(task_id: str, filename: str, file: str = "output"):
        if file not in {"input", "output"} or filename != Path(filename).name or not filename.endswith(".jpg"):
            raise HTTPException(status_code=404, detail="frame preview does not exist")
        image = frame_preview_dir(task_id, file) / filename
        if not image.is_file():
            raise HTTPException(status_code=404, detail="frame preview does not exist")
        return FileResponse(image, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/tasks/{task_id}/play-url")
    def get_task_play_url(task_id: str, file: str = "input") -> dict[str, str]:
        # Keep the signed S3 URL on the Controller. The browser requests this
        # relative URL through Nginx, so S3 sees the Controller as the client.
        if file not in {"input", "output"}:
            raise HTTPException(status_code=400, detail="file must be 'input' or 'output'")
        task = store.task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task does not exist")
        if not s3_ingestor:
            raise HTTPException(status_code=409, detail="playback URLs are only available for S3 tasks")
        return {"url": f"/api/tasks/{urllib.parse.quote(task_id, safe='')}/play?file={file}"}

    @app.get("/api/tasks/{task_id}/play", name="proxy_task_playback")
    def proxy_task_playback(request: Request, task_id: str, file: str = "input"):
        """Stream an S3 video through the Controller while preserving seek support."""
        task = store.task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task does not exist")
        url = task_playback_url(task_id, file)
        upstream_headers = {"User-Agent": "video-mask-controller-playback/1.0"}
        for header in ("range", "if-range"):
            value = request.headers.get(header)
            if value:
                upstream_headers[header.title()] = value
        try:
            upstream = urllib.request.urlopen(urllib.request.Request(url, headers=upstream_headers), timeout=30)
        except urllib.error.HTTPError as exc:
            raise HTTPException(status_code=exc.code, detail="S3 playback request failed") from exc
        except urllib.error.URLError as exc:
            raise HTTPException(status_code=502, detail="unable to reach S3 for playback") from exc

        response_headers: dict[str, str] = {}
        for header in ("Accept-Ranges", "Content-Length", "Content-Range", "Content-Type", "ETag", "Last-Modified"):
            value = upstream.headers.get(header)
            if value:
                response_headers[header] = value
        filename = task_playback_filename(task, file)
        ascii_filename = filename.encode("ascii", "ignore").decode("ascii") or "video.mp4"
        ascii_filename = ascii_filename.replace('"', "_").replace("\\", "_")
        response_headers["Content-Disposition"] = (
            f'inline; filename="{ascii_filename}"; '
            f"filename*=UTF-8''{urllib.parse.quote(filename, safe='')}"
        )

        def stream_video():
            try:
                while chunk := upstream.read(1024 * 1024):
                    yield chunk
            finally:
                upstream.close()

        media_type = response_headers.pop("Content-Type", None)
        return StreamingResponse(
            stream_video(),
            status_code=upstream.getcode(),
            headers=response_headers,
            media_type=media_type,
        )

    @app.post("/api/tasks")
    def create_task(request: TaskRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            return store.create_task(model_data(request))
        except (ValueError, sqlite3.IntegrityError) as exc:  # type: ignore[name-defined]
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/tasks/{task_id}/retry")
    def retry_task(task_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            return store.retry_task(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/tasks/{task_id}/restart")
    def restart_task(task_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            return store.restart_task(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/tasks/restart-completed")
    def restart_completed_tasks(authorization: str | None = Header(default=None)) -> dict[str, int | float]:
        require_admin(authorization)
        return store.restart_completed_tasks()

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel_task(task_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            return store.cancel_task(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/dashboard/tasks/{task_id}/restart")
    def dashboard_restart_task(task_id: str) -> dict[str, Any]:
        """Restart one completed task from the Nginx-protected dashboard."""
        task = store.task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task does not exist")
        if task.get("status") != "completed":
            raise HTTPException(status_code=409, detail="only completed tasks can be restarted here")
        return store.restart_task(task_id)

    @app.post("/api/dashboard/tasks/{task_id}/cancel")
    def dashboard_cancel_task(task_id: str) -> dict[str, Any]:
        """Request cancellation of an active task from the Nginx-protected dashboard."""
        task = store.task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task does not exist")
        if task.get("status") not in {"assigned", "downloading", "processing", "uploading"}:
            raise HTTPException(status_code=409, detail="only active tasks can be cancelled here")
        return store.cancel_task(task_id)

    @app.put("/api/dashboard/manual-tasks")
    async def create_dashboard_manual_task(
        request: Request,
        filename: str = Header(alias="X-Video-Mask-Filename"),
        algorithm: str = Header(alias="X-Video-Mask-Algorithm"),
        arguments: str = Header(alias="X-Video-Mask-Arguments"),
    ) -> dict[str, Any]:
        """Upload a dashboard-selected source video and queue it immediately.

        The dashboard is protected by the outer Nginx authentication.  Keeping
        the upload on this same route avoids exposing an S3 upload URL and its
        CORS requirements to the browser.
        """
        if not s3_ingestor:
            raise HTTPException(status_code=409, detail="manual upload requires S3 storage")
        filename = Path(filename.replace("\\", "/")).name
        if not filename or Path(filename).suffix.lower() not in VIDEO_SUFFIXES:
            raise HTTPException(status_code=400, detail="choose a supported video file")
        selected_algorithm = algorithm.strip()
        if (not selected_algorithm or len(selected_algorithm) > 255
                or Path(selected_algorithm).name != selected_algorithm):
            raise HTTPException(status_code=400, detail="algorithm must be a filename in the Worker source directory")
        try:
            selected_arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="algorithm parameters must be a JSON array of strings") from exc
        if (not isinstance(selected_arguments, list) or len(selected_arguments) > 128
                or not all(isinstance(value, str) and len(value) <= 4096 for value in selected_arguments)):
            raise HTTPException(status_code=400, detail="algorithm parameters must be a JSON array of at most 128 strings")
        source_key = f"{MANUAL_SOURCE_PREFIX}/{uuid.uuid4().hex}_{filename}"
        upload_root = database.parent / "manual-uploads"
        upload_root.mkdir(parents=True, exist_ok=True)
        temporary_source = upload_root / f"{uuid.uuid4().hex}.upload"
        source_size_bytes = 0
        try:
            with temporary_source.open("xb") as handle:
                async for block in request.stream():
                    source_size_bytes += len(block)
                    if source_size_bytes > MANUAL_UPLOAD_MAX_BYTES:
                        raise HTTPException(status_code=413, detail="uploaded file exceeds the 20 GiB limit")
                    handle.write(block)
        except HTTPException:
            temporary_source.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temporary_source.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="unable to receive uploaded file") from exc
        if not source_size_bytes:
            temporary_source.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="uploaded file is empty")

        content_type = request.headers.get("content-type", "")
        extra_args = {"ContentType": content_type} if content_type else None
        try:
            await asyncio.to_thread(
                s3_ingestor.source_client.upload_file,
                str(temporary_source),
                s3_ingestor.source_bucket,
                source_key,
                ExtraArgs=extra_args,
            )
        except Exception as exc:
            logger.exception("Dashboard manual source upload failed for %s", filename)
            raise HTTPException(status_code=502, detail="unable to upload the source video to S3") from exc
        finally:
            temporary_source.unlink(missing_ok=True)

        try:
            task = store.create_task({
                "source_url": f"s3://{s3_ingestor.source_bucket}/{source_key}",
                "source_object_key": source_key,
                "source_size_bytes": source_size_bytes,
                "algorithm": selected_algorithm,
                "arguments": selected_arguments,
                "output_object_key": s3_ingestor.output_key(source_key),
            })
        except Exception as exc:
            # The source object is unique to this failed request; remove it so
            # it cannot become an untracked charge or later be picked up by a scan.
            try:
                await asyncio.to_thread(
                    s3_ingestor.source_client.delete_object,
                    Bucket=s3_ingestor.source_bucket,
                    Key=source_key,
                )
            except Exception:
                logger.exception("Unable to remove failed manual upload %s", source_key)
            if isinstance(exc, (ValueError, sqlite3.IntegrityError)):
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise
        return {"task": task, "source_object_key": source_key}

    @app.get("/api/dashboard/tasks/{task_id}/logs")
    def dashboard_task_logs(task_id: str, limit: int = 1000) -> dict[str, Any]:
        if store.task(task_id) is None:
            raise HTTPException(status_code=404, detail="task does not exist")
        return {"task_id": task_id, "logs": store.task_logs(task_id, limit)}

    @app.post("/api/face-reviews/claim")
    def claim_face_review(request: FaceReviewRequest) -> dict[str, Any]:
        # The dashboard's outer access control identifies authorized reviewers;
        # reviewer_id is a browser-session lease owner, not an account identity.
        task = store.claim_next_face_review(request.reviewer_id, FACE_REVIEW_LEASE_SECONDS)
        if task is None:
            return {"task": None, "lease_seconds": FACE_REVIEW_LEASE_SECONDS}
        task_id = str(task["task_id"])
        playback_file = "output" if task.get("output_object_key") else "input"
        return {
            "task": task,
            "lease_seconds": FACE_REVIEW_LEASE_SECONDS,
            "playback_url": f"/api/tasks/{urllib.parse.quote(task_id, safe='')}/play?file={playback_file}",
            "playback_file": playback_file,
        }

    @app.post("/api/face-reviews/{task_id}/open")
    def open_face_review(task_id: str, request: FaceReviewRequest) -> dict[str, Any]:
        """Reserve a particular completed video so its label can be viewed or edited."""
        try:
            task = store.claim_face_review(task_id, request.reviewer_id, FACE_REVIEW_LEASE_SECONDS)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"task": task, "lease_seconds": FACE_REVIEW_LEASE_SECONDS}

    @app.post("/api/face-reviews/{task_id}/heartbeat")
    def renew_face_review(task_id: str, request: FaceReviewRequest) -> dict[str, Any]:
        try:
            task = store.renew_face_review(task_id, request.reviewer_id, FACE_REVIEW_LEASE_SECONDS)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"task": task, "lease_seconds": FACE_REVIEW_LEASE_SECONDS}

    @app.post("/api/face-reviews/{task_id}/release")
    def release_face_review(task_id: str, request: FaceReviewRequest) -> dict[str, bool]:
        return {"released": store.release_face_review(task_id, request.reviewer_id)}

    @app.put("/api/face-reviews/{task_id}/annotation")
    def annotate_face_review(task_id: str, request: FaceAnnotationRequest) -> dict[str, Any]:
        try:
            return store.annotate_face(task_id, request.reviewer_id, request.has_face)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/face-reviews/status")
    def get_face_review_status(task_ids: str = "") -> dict[str, Any]:
        selected = tuple(value for value in task_ids.split(",") if value)[:100]
        return store.face_review_status(selected or None)

    @app.delete("/api/tasks")
    def purge_tasks(confirm: str = "", authorization: str | None = Header(default=None)) -> dict[str, int]:
        """Delete every task record after an explicit destructive-action confirmation."""
        require_admin(authorization)
        if confirm != "DELETE_ALL_TASKS":
            raise HTTPException(status_code=400, detail="set confirm=DELETE_ALL_TASKS to delete all tasks")
        try:
            return store.purge_tasks()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/admin/workers")
    def provision_worker(request: WorkerProvisionRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return store.provision_worker(request.worker_id, request.token)

    @app.post("/api/admin/workers/{worker_id}/retire")
    def retire_worker(worker_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return store.retire_worker(worker_id)

    @app.get("/api/admin/s3-ingest")
    def get_s3_ingest_switch() -> dict[str, bool]:
        # Dashboard access is protected by the same outer proxy authentication
        # as the UI control, so this browser-facing switch has no token prompt.
        return {"configured": bool(s3_ingestor), "enabled": s3_ingest_is_enabled()}

    @app.put("/api/admin/s3-ingest")
    def set_s3_ingest_switch(request: S3IngestSwitchRequest) -> dict[str, bool]:
        nonlocal s3_ingest_enabled
        if not s3_ingestor:
            raise HTTPException(status_code=409, detail="S3 ingestion is not configured on this Controller")
        if request.enabled and not s3_ingest_enabled:
            # Validate before persisting the enabled state, so a bad Bucket
            # Region or missing permission cannot leave a misleading switch.
            s3_ingestor.validate_bucket_regions()
        s3_ingest_enabled = store.set_boolean_setting("s3_ingest_enabled", request.enabled)
        logger.warning("S3 ingestion %s by administrator", "enabled" if s3_ingest_enabled else "disabled")
        return {"configured": True, "enabled": s3_ingest_enabled}

    @app.post("/api/admin/s3-ingest/scan")
    def scan_s3_ingest_once() -> dict[str, bool | int]:
        """Run one S3 discovery pass even while scheduled ingestion is paused."""
        if not s3_ingestor:
            raise HTTPException(status_code=409, detail="S3 ingestion is not configured on this Controller")
        created = s3_ingestor.scan()
        logger.warning("S3 ingestion manual scan created %d task(s)", created)
        return {"configured": True, "enabled": s3_ingest_is_enabled(), "created": created}

    @app.get("/api/tasks")
    def list_tasks(limit: int = 100, offset: int = 0, status: str | None = None,
                   authorization: str | None = Header(default=None)):
        require_admin(authorization)
        page_size = max(1, min(limit, 1000))
        page_offset = max(0, offset)
        statuses = tuple(value.strip() for value in (status or "").split(",") if value.strip()) or None
        return {
            "tasks": store.list_tasks(page_size, page_offset, statuses),
            "total": store.count_tasks(statuses),
            "oldest_created_at": store.oldest_task_created_at(statuses) if statuses else None,
            "limit": page_size,
            "offset": page_offset,
        }

    @app.get("/api/workers")
    def list_workers(limit: int = 100, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        store.requeue_stale(stale_after_seconds)
        return {"workers": store.list_workers(max(1, min(limit, 1000)))}

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, page: int = 1, task_status: str | None = None, q: str = "",
                  statistics_start: str | None = None, statistics_end: str | None = None):
        store.requeue_stale(stale_after_seconds)
        try:
            stats_start, stats_end = statistics_window(statistics_start, statistics_end)
            processing_statistics = store.processing_statistics(stats_start, stats_end)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        selected_task_status = task_status.strip() if task_status else ""
        if selected_task_status and selected_task_status not in TASK_FILTER_STATUSES:
            raise HTTPException(status_code=400, detail="unknown task status filter")
        selected_statuses = (selected_task_status,) if selected_task_status else None
        selected_face_annotations = tuple(dict.fromkeys(
            value.strip() for value in request.query_params.getlist("face_annotation") if value.strip()
        ))
        unknown_face_annotations = set(selected_face_annotations) - set(FACE_ANNOTATION_FILTERS)
        if unknown_face_annotations:
            raise HTTPException(status_code=400, detail="unknown face annotation filter")
        selected_search = q.strip()[:200] or None
        task_filter_query = urllib.parse.urlencode([
            *([("task_status", selected_task_status)] if selected_task_status else []),
            *(("face_annotation", value) for value in selected_face_annotations),
            *([("q", selected_search)] if selected_search else []),
        ])
        page_size = 50
        total_tasks = store.count_tasks(selected_statuses, selected_search, selected_face_annotations or None)
        total_pages = max(1, (total_tasks + page_size - 1) // page_size)
        page = min(max(1, page), total_pages)
        rows = store.list_tasks(page_size, (page - 1) * page_size, selected_statuses, selected_search,
                                selected_face_annotations or None)

        def byte_size(value: Any) -> str:
            try:
                size = int(value)
            except (TypeError, ValueError):
                return "-"
            for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
                if size < 1024 or unit == "TiB":
                    return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
                size /= 1024
            return "-"

        def seconds(value: Any) -> str:
            try:
                elapsed = float(value)
            except (TypeError, ValueError):
                return "-"
            if elapsed < 60:
                return f"{elapsed:.1f}s"
            minutes, remainder = divmod(round(elapsed), 60)
            hours, minutes = divmod(minutes, 60)
            return f"{hours}h {minutes}m {remainder}s" if hours else f"{minutes}m {remainder}s"

        def last_seen(timestamp: float | None) -> str:
            if not timestamp:
                return "-"
            return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

        workers = store.list_workers(1000)
        task_status_counts = {
            status: store.count_tasks((status,))
            for status in TASK_FILTER_STATUSES
        }

        def task_view(task: dict[str, Any]) -> dict[str, Any]:
            progress = task.get("progress") or {}
            worker_id = str(task.get("assigned_worker_id") or "").strip()
            worker_name, marker, worker_slot = worker_id.rpartition("-slot-")
            if not (marker and worker_name and worker_slot.isdigit()):
                worker_name, worker_slot = worker_id or "-", ""
            processing = progress.get("processing_seconds", progress.get("elapsed_seconds"))
            started, restarted, finished = task.get("started_at"), task.get("restarted_at"), task.get("finished_at")
            face_annotation = task.get("face_annotation")
            review_active = (face_annotation is None and task.get("face_review_owner")
                             and float(task.get("face_review_lease_until") or 0) > time.time())
            manual_label = ("👍 Face" if face_annotation == 1 else "👎 No face"
                            if face_annotation == 0 else "👀 Reviewing" if review_active else "Unlabelled")
            if isinstance(processing, (int, float)):
                process_label = f"Process: {seconds(processing)}"
            elif task.get("status") in {"processing", "uploading", "cancelling"}:
                processing_started = progress.get("processing_started_at", started)
                running_for = time.time() - processing_started if isinstance(processing_started, (int, float)) else None
                process_label = f"Process: running {seconds(max(0, running_for))}"
            else:
                process_label = "Process: -"
            lifetime = (f"Task lifetime: {seconds(finished - started)}"
                        if isinstance(started, (int, float)) and isinstance(finished, (int, float)) else "")
            return {
                "task_id": str(task["task_id"]), "status": str(task["status"]),
                "worker_name": worker_name, "worker_slot": worker_slot,
                "source_name": str(task.get("source_object_key") or task.get("source_url") or "-"),
                "input_size": byte_size(progress.get("input_bytes", task.get("source_size_bytes"))),
                "output_name": str(task.get("output_object_key") or progress.get("output_filename") or "-"),
                "output_meta": f"{byte_size(progress.get('output_bytes'))} · {seconds(task.get('output_duration_seconds', progress.get('output_duration_seconds')))}",
                "output_hash": str(task.get("output_sha256") or ""), "process_label": process_label,
                "lifetime": lifetime, "attempt_count": task["attempt_count"],
                "created_at": last_seen(task.get("created_at")),
                "execution_started_label": "Restarted" if isinstance(restarted, (int, float)) else "Started",
                "execution_started_at": last_seen(
                    restarted if isinstance(restarted, (int, float))
                    else progress.get("processing_started_at", started)
                ),
                "execution_finished_at": last_seen(finished),
                "manual_label": manual_label,
                "manual_label_state": "labelled" if face_annotation is not None else "reviewing" if review_active else "unlabelled",
                "error_message": str(task.get("error_message") or "-"),
            }

        task_rows = [task_view(task) for task in rows]
        def base_worker_id(worker_id: str) -> str:
            prefix, marker, _ = str(worker_id).rpartition("-slot-")
            return prefix if marker and prefix else str(worker_id)

        def slot_label(worker: dict[str, Any]) -> str:
            capabilities = worker.get("capabilities") or {}
            if capabilities.get("slot") is not None:
                return f"slot-{capabilities['slot']}"
            _, marker, slot = str(worker["worker_id"]).rpartition("-slot-")
            return f"slot-{slot}" if marker and slot.isdigit() else str(worker["worker_id"])

        def host_address(caps: dict[str, Any]) -> str:
            address = caps.get("controller_seen_ip")
            if not address:
                addresses = caps.get("ip_addresses") or []
                address = ", ".join(str(value) for value in addresses) or "-"
            return str(address)

        ACTIVE_SLOT = {"busy", "assigned", "downloading", "processing", "uploading", "cancelling"}

        def server_groups() -> list[dict[str, Any]]:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for worker in workers:
                grouped.setdefault(base_worker_id(str(worker["worker_id"])), []).append(worker)
            result: list[dict[str, Any]] = []
            for base, slots in grouped.items():
                caps = slots[0].get("capabilities") or {}
                statuses = [str(slot.get("status") or "unknown") for slot in slots]
                online = any(status != "offline" for status in statuses)
                result.append({
                    "base_id": base,
                    "hostname": str(caps.get("hostname") or base),
                    "address": host_address(caps),
                    "gpu": str(caps.get("gpu") or "-"),
                    "slots": slots,
                    "slot_count": len(slots),
                    "ready": sum(status == "ready" for status in statuses),
                    "busy": sum(status in ACTIVE_SLOT for status in statuses),
                    "online": online,
                    "last_seen": last_seen(max((slot.get("last_seen_at") or 0) for slot in slots)),
                    "status": "online" if online else "offline",
                    "slots": [{
                        "label": slot_label(slot), "status": str(slot.get("status") or "unknown"),
                        "current_task_id": str(slot.get("current_task_id") or "-"),
                        "last_seen": last_seen(slot.get("last_seen_at")),
                    } for slot in slots],
                })
            result.sort(key=lambda group: (not group["online"], group["hostname"].lower(), group["base_id"]))
            return result

        groups = server_groups()
        active_workers = sum(1 for group in groups if group["online"])

        worker_status_counts: dict[str, int] = {}
        for worker in workers:
            status = str(worker.get("status") or "unknown")
            worker_status_counts[status] = worker_status_counts.get(status, 0) + 1
        active_tasks = sum(task_status_counts[status] for status in
                           ("assigned", "downloading", "processing", "uploading", "cancelling"))
        worker_filter_options = sorted(worker_status_counts.items())
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "active_tasks": active_tasks,
                "active_workers": active_workers,
                "groups": groups,
                "next_page": page + 1 if page < total_pages else None,
                "page": page,
                "page_size": page_size,
                "previous_page": page - 1 if page > 1 else None,
                "status_tones": STATUS_TONES,
                "s3_ingest_configured": bool(s3_ingestor),
                "s3_ingest_enabled": s3_ingest_is_enabled(),
                "manual_task_default_algorithm": DEFAULT_ALGORITHM,
                "manual_task_default_arguments": DEFAULT_ARGS,
                "processing_statistics": processing_statistics,
                "statistics_range_hours": round((stats_end - stats_start) / 3600),
                "statistics_start_input": datetime_local(stats_start),
                "statistics_end_input": datetime_local(stats_end),
                "task_rows": task_rows,
                "task_filter_options": [(status, task_status_counts[status]) for status in TASK_FILTER_STATUSES],
                "task_status_counts": task_status_counts,
                "selected_task_status": selected_task_status,
                "face_annotation_filter_options": (
                    ("has_face", "👍 Has face"),
                    ("no_face", "👎 No face"),
                    ("unlabelled", "Unlabelled"),
                ),
                "selected_face_annotations": selected_face_annotations,
                "selected_search": selected_search or "",
                "task_filter_query": task_filter_query,
                "task_total_all": sum(task_status_counts.values()),
                "total_pages": total_pages,
                "total_tasks": total_tasks,
                "updated_at": dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                "worker_filter_options": worker_filter_options,
            },
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-mask cluster controller")
    parser.add_argument("--database", type=Path, default=Path("cluster-controller.sqlite3"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--admin-token", default=os.environ.get("VIDEO_MASK_ADMIN_TOKEN"))
    parser.add_argument("--stale-after", type=int, default=90)
    parser.add_argument("--local-source-dir", type=Path,
                        help="Testing only: scan local videos and issue file:// tasks")
    parser.add_argument("--local-output-dir", type=Path,
                        help="Testing only: write completed files to this directory")
    parser.add_argument("--s3-source-bucket")
    parser.add_argument("--s3-source-prefix", default="")
    parser.add_argument("--s3-output-bucket")
    parser.add_argument("--s3-output-prefix", default="outputs/")
    parser.add_argument("--s3-source-region")
    parser.add_argument("--s3-output-region")
    parser.add_argument("--s3-profile")
    parser.add_argument("--s3-poll-seconds", type=int, default=60)
    parser.add_argument("--s3-presign-seconds", type=int, default=86400)
    args = parser.parse_args()
    if not args.admin_token or len(args.admin_token) < 16:
        raise SystemExit("Set --admin-token or VIDEO_MASK_ADMIN_TOKEN (at least 16 characters)")
    if bool(args.local_source_dir) != bool(args.local_output_dir):
        raise SystemExit("--local-source-dir and --local-output-dir must be supplied together")
    s3_options = (args.s3_source_bucket, args.s3_output_bucket, args.s3_source_region)
    if any(s3_options) and not all(s3_options):
        raise SystemExit("--s3-source-bucket, --s3-output-bucket and --s3-source-region must be supplied together")
    if args.s3_poll_seconds < 1 or not 1 <= args.s3_presign_seconds <= 604800:
        raise SystemExit("--s3-poll-seconds must be positive; --s3-presign-seconds must be 1..604800")
    s3_config = None
    if args.s3_source_bucket:
        s3_config = {
            "source_bucket": args.s3_source_bucket,
            "source_prefix": args.s3_source_prefix,
            "output_bucket": args.s3_output_bucket,
            "output_prefix": args.s3_output_prefix,
            "source_region": args.s3_source_region,
            "output_region": args.s3_output_region or args.s3_source_region,
            "profile": args.s3_profile,
            "poll_seconds": args.s3_poll_seconds,
            "presign_seconds": args.s3_presign_seconds,
        }
    import uvicorn
    uvicorn.run(create_app(args.database, args.admin_token, args.stale_after,
                           args.local_source_dir, args.local_output_dir,
                           s3_config=s3_config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
