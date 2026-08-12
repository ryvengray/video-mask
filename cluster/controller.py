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
        if s3_ingestor:
            s3_ingestor.validate_bucket_regions()
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

    @app.post("/api/workers/{worker_id}/tasks/{task_id}/upload-url")
    def refresh_upload_url(worker_id: str, task_id: str, request: WorkerRequest):
        if request.worker_id != worker_id:
            raise ValueError("worker id does not match path")
        task = store.active_task_for_worker(worker_id, request.token, task_id)
        if s3_ingestor:
            return s3_ingestor.materialize_upload_url(task)
        return {"output_upload_url": str(task.get("output_upload_url") or ""),
                "output_object_key": str(task.get("output_object_key") or "")}

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
            output_duration = seconds(task.get("output_duration_seconds", progress.get("output_duration_seconds")))
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
                parts.append(f"Task lifetime: {seconds(end_to_end)}")
            return "<br>".join(parts)

        def last_seen(timestamp: float | None) -> str:
            if not timestamp:
                return "-"
            return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

        workers = store.list_workers(1000)
        task_status_counts = {
            status: store.count_tasks((status,))
            for status in ("pending", "assigned", "downloading", "processing", "uploading", "cancelling", "completed", "failed", "cancelled")
        }

        def status_badge(status: Any) -> str:
            value = str(status or "unknown")
            tone = {
                "ready": "success", "completed": "success",
                "busy": "active", "assigned": "active", "downloading": "active",
                "processing": "active", "uploading": "active",
                "pending": "pending", "cancelling": "warning", "cancelled": "warning",
                "offline": "muted", "failed": "danger",
            }.get(value, "muted")
            return f"<span class='badge {tone}'>{html.escape(value)}</span>"

        task_items = "".join(
            f"<tr><td class='mono task-id' title='{html.escape(str(task['task_id']))}'>{html.escape(str(task['task_id']))}</td>"
            f"<td>{status_badge(task['status'])}</td>"
            f"<td class='mono'>{html.escape(str(task['assigned_worker_id'] or '-'))}</td>"
            f"<td>{task_file_info(task)}</td><td>{task_runtime(task)}</td>"
            f"<td>{html.escape(str(task.get('algorithm') or '-'))}<br><small>"
            f"{html.escape(' '.join(str(value) for value in (task.get('arguments') or [])))}</small></td>"
            f"<td>{task['attempt_count']}</td><td>{last_seen(task.get('finished_at'))}</td>"
            f"<td>{html.escape(str(task.get('error_message') or '-'))}</td></tr>"
            for task in rows
        ) or "<tr><td colspan='9'>No tasks</td></tr>"
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
            f"<tr><td class='mono'>{html.escape(str(worker['worker_id']))}</td><td>{status_badge(worker['status'])}</td>"
            f"<td>{host_for(worker)}</td><td>{html.escape(slot_for(worker))}</td>"
            f"<td>{html.escape(str((worker.get('capabilities') or {}).get('gpu', '-')))}</td>"
            f"<td>{html.escape(str((worker.get('capabilities') or {}).get('cuda_available', '-')))}</td>"
            f"<td>{html.escape(str((worker.get('capabilities') or {}).get('pid', '-')))}</td>"
            f"<td>{html.escape(str(worker.get('current_task_id') or '-'))}</td>"
            f"<td>{last_seen(worker.get('last_seen_at'))}</td></tr>"
            for worker in workers
        ) or "<tr><td colspan='9'>No workers registered</td></tr>"
        worker_status_counts: dict[str, int] = {}
        for worker in workers:
            status = str(worker.get("status") or "unknown")
            worker_status_counts[status] = worker_status_counts.get(status, 0) + 1
        ready_slots = worker_status_counts.get("ready", 0)
        active_tasks = sum(task_status_counts[status] for status in
                           ("assigned", "downloading", "processing", "uploading", "cancelling"))
        worker_summary = " ".join(
            f"<span class='summary-count'>{count} {html.escape(status)}</span>"
            for status, count in sorted(worker_status_counts.items())
        ) or "No registered slots"
        worker_filter_options = "<option value=''>All statuses</option>" + "".join(
            f"<option value='{html.escape(status, quote=True)}'>{html.escape(status)} ({count})</option>"
            for status, count in sorted(worker_status_counts.items())
        )
        fleet_worker_rows = "".join(
            f"<tr data-worker-status='{html.escape(str(worker.get('status') or 'unknown'), quote=True)}'>"
            f"<td class='mono'>{html.escape(str(worker['worker_id']))}</td><td>{status_badge(worker['status'])}</td>"
            f"<td>{host_for(worker)}</td><td>{html.escape(str((worker.get('capabilities') or {}).get('gpu', '-')))}</td>"
            f"<td class='mono'>{html.escape(str(worker.get('current_task_id') or '-'))}</td>"
            f"<td>{last_seen(worker.get('last_seen_at'))}</td></tr>"
            for worker in workers
        ) or "<tr><td colspan='6'>No workers registered</td></tr>"
        previous_link = (f'<a href="/?page={page - 1}">Previous</a>' if page > 1 else "Previous")
        next_link = (f'<a href="/?page={page + 1}">Next</a>' if page < total_pages else "Next")
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60"><title>Video Mask · Operations</title>
<style>
:root{{--ink:#202124;--muted:#5f6368;--line:#dadce0;--canvas:#f8f9fa;--surface:#fff;--blue:#1a73e8;--blue-soft:#e8f0fe;--green:#137333;--green-soft:#e6f4ea;--red:#b3261e;--red-soft:#fce8e6;--yellow:#b06000;--yellow-soft:#fef7e0}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--canvas);color:var(--ink);font:14px/1.45 Arial,Roboto,"Helvetica Neue",sans-serif}}
.topbar{{height:64px;background:var(--surface);border-bottom:1px solid var(--line);display:flex;align-items:center;padding:0 32px;gap:14px}} .brand{{font-size:20px;font-weight:500;letter-spacing:-.2px}} .brand-dot{{height:10px;width:10px;border-radius:50%;background:var(--blue)}} .updated{{margin-left:auto;color:var(--muted);font-size:12px}}
.shell{{max-width:1800px;margin:0 auto;padding:26px 32px 40px}} h1,h2{{margin:0;font-weight:500}} h2{{font-size:18px}} .section-head{{display:flex;align-items:center;gap:12px;margin:0 0 12px}} .subtle{{color:var(--muted);font-size:13px}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:16px;margin:0 0 26px}} .metric{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:17px 18px;min-height:92px;box-shadow:0 1px 2px rgba(60,64,67,.08)}} .metric-label{{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.55px}} .metric-value{{font-size:29px;font-weight:500;margin-top:5px}} .metric-note{{color:var(--muted);font-size:12px}}
.operations-grid{{display:grid;grid-template-columns:minmax(0,1.9fr) minmax(360px,.9fr);gap:20px;align-items:start}} .panel{{background:var(--surface);border:1px solid var(--line);border-radius:8px;box-shadow:0 1px 2px rgba(60,64,67,.08)}} .table-scroll{{overflow:auto}} .task-table{{max-height:calc(100vh - 310px);min-height:420px}} .worker-table{{height:calc(100vh - 252px);min-height:420px;max-height:760px;border-top:1px solid var(--line)}}
table{{border-collapse:separate;border-spacing:0;width:100%;font-size:13px}} th{{position:sticky;top:0;z-index:1;background:#f8f9fa;color:#3c4043;font-size:11px;text-transform:uppercase;letter-spacing:.5px;text-align:left;padding:12px 14px;border-bottom:1px solid var(--line);white-space:nowrap}} td{{padding:12px 14px;vertical-align:top;border-bottom:1px solid #edf0f2;max-width:300px;word-break:break-word}} tr:last-child td{{border-bottom:0}} tbody tr:hover{{background:#f8fbff}} small{{color:var(--muted);font-size:12px}} hr{{border:0;border-top:1px solid #edf0f2;margin:8px 0}} .mono{{font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace;font-size:12px}} .task-id{{max-width:170px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.badge{{display:inline-flex;align-items:center;border-radius:12px;font-size:12px;font-weight:600;padding:3px 9px;white-space:nowrap}} .success{{color:var(--green);background:var(--green-soft)}} .active{{color:#174ea6;background:var(--blue-soft)}} .pending{{color:#5f6368;background:#f1f3f4}} .warning{{color:var(--yellow);background:var(--yellow-soft)}} .danger{{color:var(--red);background:var(--red-soft)}} .muted{{color:#5f6368;background:#f1f3f4}}
.pagination{{display:flex;justify-content:flex-end;align-items:center;gap:16px;padding:12px 14px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}} .pagination a{{color:var(--blue);text-decoration:none;font-weight:500}} .pagination a:hover{{text-decoration:underline}}
.fleet-title{{padding:16px 18px 12px}} .fleet-title h2{{margin-bottom:4px}} .fleet-summary{{padding:0 18px 14px;color:var(--muted);font-size:12px;border-bottom:1px solid var(--line)}} .fleet-controls{{display:flex;align-items:center;gap:8px;margin-left:auto}} .fleet-filter{{appearance:auto;border:1px solid var(--line);background:var(--surface);border-radius:4px;color:var(--ink);font:12px Arial,Roboto,sans-serif;padding:6px 24px 6px 8px}} .summary-count{{margin-right:10px;white-space:nowrap}} .fleet-panel th,.fleet-panel td{{padding:10px 12px}} .fleet-panel td{{max-width:180px}} .fleet-panel .mono{{font-size:11px}}
@media(max-width:1120px){{.operations-grid{{grid-template-columns:1fr}}.worker-table{{height:420px;min-height:0;max-height:420px}}}} @media(max-width:850px){{.topbar{{padding:0 18px}}.shell{{padding:20px 18px}}.metrics{{grid-template-columns:repeat(2,minmax(130px,1fr));gap:10px}}.metric{{padding:13px}}.task-table{{min-height:360px}}.updated{{display:none}}}}
</style></head><body>
<header class="topbar"><span class="brand-dot"></span><span class="brand">Video Mask</span><span class="subtle">Operations</span><span class="updated">Auto-refreshes every 60 seconds · {html.escape(dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))}</span></header>
<main class="shell">
  <section class="metrics" aria-label="Cluster summary">
    <div class="metric"><div class="metric-label">Queued</div><div class="metric-value">{task_status_counts['pending']}</div><div class="metric-note">Awaiting a slot</div></div>
    <div class="metric"><div class="metric-label">Active tasks</div><div class="metric-value">{active_tasks}</div><div class="metric-note">Download, process, upload</div></div>
    <div class="metric"><div class="metric-label">Ready slots</div><div class="metric-value">{ready_slots}</div><div class="metric-note">of {len(workers)} registered</div></div>
    <div class="metric"><div class="metric-label">Failed tasks</div><div class="metric-value">{task_status_counts['failed']}</div><div class="metric-note">Action may be required</div></div>
  </section>
  <section class="operations-grid">
    <div><div class="section-head"><h2>Tasks</h2><span class="subtle">{total_tasks} total</span></div><section class="panel"><div class="table-scroll task-table"><table><thead><tr><th>Task</th><th>Status</th><th>Worker</th><th>Files</th><th>Time</th><th>Algorithm / arguments</th><th>Attempts</th><th>Finished</th><th>Error</th></tr></thead><tbody>{task_items}</tbody></table></div><div class="pagination">{previous_link}<span>Page {page} of {total_pages} · {page_size} per page</span>{next_link}</div></section></div>
    <aside><div class="section-head"><h2>Worker fleet</h2><span class="subtle">Live slots</span><label class="fleet-controls"><span class="subtle">Status</span><select id="worker-status-filter" class="fleet-filter" aria-label="Filter workers by status">{worker_filter_options}</select></label></div><section class="panel fleet-panel"><div class="fleet-title"><h2>{len(workers)} registered slots</h2><span class="subtle">Slot health and host details</span></div><div class="fleet-summary">{worker_summary}</div><div class="table-scroll worker-table"><table><thead><tr><th>Worker</th><th>Status</th><th>Host / private IP</th><th>GPU</th><th>Current task</th><th>Last heartbeat</th></tr></thead><tbody id="worker-fleet-rows">{fleet_worker_rows}</tbody></table></div></section></aside>
  </section>
</main><script>
const workerFilter=document.getElementById('worker-status-filter');
const workerRows=document.querySelectorAll('#worker-fleet-rows tr[data-worker-status]');
workerFilter.addEventListener('change',()=>{{const selected=workerFilter.value;workerRows.forEach(row=>{{row.hidden=Boolean(selected)&&row.dataset.workerStatus!==selected;}});}});
</script></body></html>"""

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
