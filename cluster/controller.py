#!/usr/bin/env python3
"""FastAPI controller for pull-based video-mask workers."""
from __future__ import annotations

import argparse
import datetime as dt
import html
import hmac
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, Header, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("Install cluster requirements: pip install -r requirements-cluster.txt") from exc

from cluster.store import ClusterStore
from cluster.local_ingest import LocalIngestor
from cluster.s3_ingest import S3Ingestor


class WorkerRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=128)
    token: str = Field(min_length=16)
    capabilities: dict[str, Any] = Field(default_factory=dict)


class HeartbeatRequest(WorkerRequest):
    status: str = "ready"


class ProgressRequest(WorkerRequest):
    status: str
    progress: dict[str, Any] = Field(default_factory=dict)


class FinishRequest(WorkerRequest):
    output_sha256: str | None = None
    output_duration_seconds: float | None = None
    error_message: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)


class TaskRequest(BaseModel):
    source_url: str
    output_upload_url: str | None = None
    source_object_key: str | None = None
    source_size_bytes: int | None = None
    source_sha256: str | None = None
    source_duration_seconds: float | None = None
    algorithm: str = "video_mask_batch_fish_v1.py"
    arguments: list[str] = Field(default_factory=lambda: [
        "--fisheye", "--fisheye-device", "pico4", "--face-size", "640",
        "--face-int", "5", "--frame-skip", "3", "--face-model", "yolov8",
    ])
    output_object_key: str | None = None
    max_attempts: int = Field(default=3, ge=1, le=20)


class WorkerProvisionRequest(BaseModel):
    worker_id: str = Field(min_length=1, max_length=128)
    token: str = Field(min_length=16)


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

    def model_data(model: BaseModel) -> dict[str, Any]:
        # FastAPI 0.110 uses Pydantic v2, while some supported deployments
        # still resolve Pydantic v1.
        return model.model_dump() if hasattr(model, "model_dump") else model.dict()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        store.close()

    app = FastAPI(title="Video Mask Cluster Controller", lifespan=lifespan)

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
        s3_ingested = s3_ingestor.scan_if_due() if s3_ingestor else 0
        return {"status": "ok", "requeued": str(requeued), "ingested": str(ingested),
                "s3_ingested": str(s3_ingested)}

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
        if s3_ingestor:
            s3_ingestor.scan_if_due()
        task = store.claim(worker_id, request.token)
        return {"task": s3_ingestor.materialize(task) if task and s3_ingestor else task}

    @app.post("/api/tasks/{task_id}/progress")
    def progress(task_id: str, request: ProgressRequest):
        return store.progress(request.worker_id, request.token, task_id, request.status, request.progress)

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

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel_task(task_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            return store.cancel_task(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/admin/workers")
    def provision_worker(request: WorkerProvisionRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return store.provision_worker(request.worker_id, request.token)

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
            "limit": page_size,
            "offset": page_offset,
        }

    @app.get("/api/workers")
    def list_workers(limit: int = 100, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        store.requeue_stale(stale_after_seconds)
        return {"workers": store.list_workers(max(1, min(limit, 1000)))}

    @app.get("/", response_class=HTMLResponse)
    def dashboard(page: int = 1):
        store.requeue_stale(stale_after_seconds)
        page_size = 50
        total_tasks = store.count_tasks()
        total_pages = max(1, (total_tasks + page_size - 1) // page_size)
        page = min(max(1, page), total_pages)
        rows = store.list_tasks(page_size, (page - 1) * page_size)

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

        def task_file_info(task: dict[str, Any]) -> str:
            progress = task.get("progress") or {}
            source_name = task.get("source_object_key") or task.get("source_url") or "-"
            output_name = task.get("output_object_key") or progress.get("output_filename") or "-"
            input_size = progress.get("input_bytes", task.get("source_size_bytes"))
            output_size = progress.get("output_bytes")
            output_duration = seconds(task.get("output_duration_seconds"))
            output_hash = task.get("output_sha256")
            hash_line = f"<br><small>SHA-256: {html.escape(str(output_hash)[:16])}…</small>" if output_hash else ""
            return (
                f"<small>Input</small><br>{html.escape(str(source_name))}<br><small>{byte_size(input_size)}</small>"
                f"<hr><small>Output</small><br>{html.escape(str(output_name))}<br>"
                f"<small>{byte_size(output_size)} · {output_duration}</small>{hash_line}"
            )

        def task_runtime(task: dict[str, Any]) -> str:
            progress = task.get("progress") or {}
            processing = progress.get("processing_seconds", progress.get("elapsed_seconds"))
            started, finished = task.get("started_at"), task.get("finished_at")
            end_to_end = None
            if isinstance(started, (int, float)) and isinstance(finished, (int, float)):
                end_to_end = finished - started
            if isinstance(processing, (int, float)):
                process_label = f"Process: {seconds(processing)}"
            elif task.get("status") in {"processing", "uploading", "cancelling"}:
                processing_started = progress.get("processing_started_at", started)
                running_for = time.time() - processing_started if isinstance(processing_started, (int, float)) else None
                process_label = f"Process: running {seconds(max(0, running_for))}"
            else:
                process_label = "Process: -"
            parts = [process_label]
            if end_to_end is not None:
                parts.append(f"Total: {seconds(end_to_end)}")
            return "<br>".join(parts)

        def last_seen(timestamp: float | None) -> str:
            if not timestamp:
                return "-"
            return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

        task_items = "".join(
            f"<tr><td>{html.escape(str(task['task_id']))}</td><td>{html.escape(str(task['status']))}</td>"
            f"<td>{html.escape(str(task['assigned_worker_id'] or '-'))}</td>"
            f"<td>{task_file_info(task)}</td><td>{task_runtime(task)}</td>"
            f"<td>{html.escape(str(task.get('algorithm') or '-'))}<br><small>"
            f"{html.escape(' '.join(str(value) for value in (task.get('arguments') or [])))}</small></td>"
            f"<td>{task['attempt_count']}</td><td>{last_seen(task.get('finished_at'))}</td>"
            f"<td>{html.escape(str(task.get('error_message') or '-'))}</td></tr>"
            for task in rows
        ) or "<tr><td colspan='9'>No tasks</td></tr>"
        workers = store.list_workers(100)

        def slot_for(worker: dict[str, Any]) -> str:
            capabilities = worker.get("capabilities") or {}
            if capabilities.get("slot") is not None:
                return str(capabilities["slot"])
            _, marker, slot = str(worker["worker_id"]).rpartition("-slot-")
            return slot if marker and slot.isdigit() else "-"

        def host_for(worker: dict[str, Any]) -> str:
            capabilities = worker.get("capabilities") or {}
            hostname = str(capabilities.get("hostname") or "-")
            address = capabilities.get("controller_seen_ip")
            if not address:
                addresses = capabilities.get("ip_addresses") or []
                address = ", ".join(str(value) for value in addresses) or "-"
            return f"{html.escape(hostname)}<br><small>{html.escape(str(address))}</small>"

        worker_items = "".join(
            f"<tr><td>{html.escape(str(worker['worker_id']))}</td><td>{html.escape(str(worker['status']))}</td>"
            f"<td>{host_for(worker)}</td><td>{html.escape(slot_for(worker))}</td>"
            f"<td>{html.escape(str((worker.get('capabilities') or {}).get('gpu', '-')))}</td>"
            f"<td>{html.escape(str((worker.get('capabilities') or {}).get('cuda_available', '-')))}</td>"
            f"<td>{html.escape(str((worker.get('capabilities') or {}).get('pid', '-')))}</td>"
            f"<td>{html.escape(str(worker.get('current_task_id') or '-'))}</td>"
            f"<td>{last_seen(worker.get('last_seen_at'))}</td></tr>"
            for worker in workers
        ) or "<tr><td colspan='9'>No workers registered</td></tr>"
        previous_link = (f'<a href="/?page={page - 1}">Previous</a>' if page > 1 else "Previous")
        next_link = (f'<a href="/?page={page + 1}">Next</a>' if page < total_pages else "Next")
        return """<!doctype html><title>Video Mask Cluster</title>
        <style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse;width:100%;margin:0 0 2rem}td,th{border:1px solid #ddd;padding:.5rem;text-align:left;vertical-align:top;word-break:break-word}th{white-space:nowrap}h2{margin-top:2rem}hr{border:0;border-top:1px solid #ddd;margin:.4rem 0}.pagination{display:flex;gap:1rem;align-items:center;margin:.5rem 0 1rem}</style>
        <h1>Video Mask Cluster</h1><h2>Workers</h2><table><tr><th>Worker</th><th>Status</th><th>Host / private IP</th><th>Slot</th><th>GPU</th><th>CUDA</th><th>PID</th><th>Current task</th><th>Last heartbeat</th></tr>""" + worker_items + "</table>" + \
            f"<h2>Tasks ({total_tasks})</h2><div class='pagination'>{previous_link}<span>Page {page} / {total_pages} · {page_size} per page</span>{next_link}</div>" + \
            "<table><tr><th>Task</th><th>Status</th><th>Worker</th><th>Files</th><th>Time</th><th>Algorithm / arguments</th><th>Attempts</th><th>Finished</th><th>Error</th></tr>" + task_items + "</table>"

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
    parser.add_argument("--s3-region")
    parser.add_argument("--s3-profile")
    parser.add_argument("--s3-poll-seconds", type=int, default=60)
    parser.add_argument("--s3-presign-seconds", type=int, default=86400)
    args = parser.parse_args()
    if not args.admin_token or len(args.admin_token) < 16:
        raise SystemExit("Set --admin-token or VIDEO_MASK_ADMIN_TOKEN (at least 16 characters)")
    if bool(args.local_source_dir) != bool(args.local_output_dir):
        raise SystemExit("--local-source-dir and --local-output-dir must be supplied together")
    s3_options = (args.s3_source_bucket, args.s3_output_bucket, args.s3_region)
    if any(s3_options) and not all(s3_options):
        raise SystemExit("--s3-source-bucket, --s3-output-bucket and --s3-region must be supplied together")
    if args.s3_poll_seconds < 1 or not 1 <= args.s3_presign_seconds <= 604800:
        raise SystemExit("--s3-poll-seconds must be positive; --s3-presign-seconds must be 1..604800")
    s3_config = None
    if args.s3_source_bucket:
        s3_config = {
            "source_bucket": args.s3_source_bucket,
            "source_prefix": args.s3_source_prefix,
            "output_bucket": args.s3_output_bucket,
            "output_prefix": args.s3_output_prefix,
            "region": args.s3_region,
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
