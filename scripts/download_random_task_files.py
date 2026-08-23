#!/usr/bin/env python3
"""Download a random set of task videos through the Controller playback API.

The script keeps one local background job at a time.  It reads task IDs and
object keys from a downloaded Controller SQLite snapshot, but never contacts
S3 directly: every byte is requested from the Controller ``/play`` endpoint.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_STATE_DIR = Path.home() / ".video-mask-task-download"
CHUNK_BYTES = 256 * 1024


class DownloadCancelled(Exception):
    """Raised when a background download is asked to stop."""


def parse_header(value: str) -> tuple[str, str]:
    name, separator, header_value = value.partition(":")
    if not separator or not name.strip() or not header_value.strip():
        raise argparse.ArgumentTypeError("headers must use 'Name: value' format")
    return name.strip(), header_value.strip()


def basic_auth_header(credentials_file: str) -> tuple[str, str]:
    try:
        credentials = Path(credentials_file).expanduser().read_text().strip()
    except OSError as exc:
        raise ValueError(f"unable to read --basic-auth-file: {exc}") from exc
    username, separator, password = credentials.partition(":")
    if not separator or not username or not password:
        raise ValueError("--basic-auth-file must contain one 'username:password' line")
    token = base64.b64encode(credentials.encode("utf-8")).decode("ascii")
    return "Authorization", f"Basic {token}"


def state_paths(state_dir: Path) -> tuple[Path, Path, Path]:
    return state_dir / "download.lock", state_dir / "state.json", state_dir / "job.json"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def process_running(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def safe_relative_key(key: str, task_id: str) -> Path:
    """Preserve object-key folders locally without allowing path escapes."""
    parts = PurePosixPath(key).parts
    if not key or PurePosixPath(key).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        return Path("invalid-object-key") / task_id / "video.bin"
    return Path(*parts)


def select_tasks(database: Path, file_kind: str, count: int, statuses: tuple[str, ...]) -> list[dict[str, Any]]:
    if not database.is_file():
        raise ValueError(f"database does not exist: {database}")
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(tasks)")}
        required = {"task_id", "status", "source_object_key", "output_object_key", "source_sha256", "output_sha256"}
        if not required <= columns:
            raise ValueError("database does not contain the Controller tasks table")
        key_column = "source_object_key" if file_kind == "input" else "output_object_key"
        sha_column = "source_sha256" if file_kind == "input" else "output_sha256"
        where = [f"{key_column} IS NOT NULL", f"{key_column} <> ''"]
        values: list[Any] = []
        if statuses:
            where.append("status IN (" + ", ".join("?" for _ in statuses) + ")")
            values.extend(statuses)
        rows = connection.execute(
            f"SELECT task_id, {key_column} AS object_key, {sha_column} AS expected_sha256 "
            f"FROM tasks WHERE {' AND '.join(where)} ORDER BY RANDOM() LIMIT ?",
            [*values, count],
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def build_request(controller: str, task_id: str, file_kind: str, headers: dict[str, str], insecure: bool) -> urllib.request.Request:
    query = urllib.parse.urlencode({"file": file_kind, "download": "true"})
    url = f"{controller.rstrip('/')}/api/tasks/{urllib.parse.quote(task_id, safe='')}/play?{query}"
    request_headers = {"User-Agent": "video-mask-task-downloader/1.0", **headers}
    return urllib.request.Request(url, headers=request_headers)


def worker(job_path: Path, state_path: Path, lock_fd: int) -> int:
    # The inherited descriptor owns the exclusive lock for this worker's full
    # lifetime.  Keep a reference so it is not accidentally closed.
    _lock_handle = os.fdopen(lock_fd, "a")
    job = read_json(job_path)
    if job is None:
        raise RuntimeError("download job configuration is missing or invalid")
    cancelled = False

    def request_cancel(_signal: int, _frame: Any) -> None:
        nonlocal cancelled
        cancelled = True
        raise DownloadCancelled

    signal.signal(signal.SIGTERM, request_cancel)
    signal.signal(signal.SIGINT, request_cancel)

    state: dict[str, Any] = {
        "status": "running", "pid": os.getpid(), "started_at": time.time(),
        "controller": job["controller"], "file": job.get("file_label", job["file"]), "destination": job["destination"],
        "total_tasks": len(job["tasks"]), "completed_tasks": 0, "skipped_tasks": 0,
        "failed_tasks": 0, "bytes_downloaded": 0, "errors": [], "current": None,
    }
    write_json(state_path, state)
    headers = dict(job.get("headers") or {})
    ssl_context = None
    if job.get("insecure"):
        import ssl
        ssl_context = ssl._create_unverified_context()  # nosec B323: explicit CLI opt-in for private Controller TLS

    try:
        for task in job["tasks"]:
            task_id = str(task["task_id"])
            relative_path = safe_relative_key(str(task["object_key"]), task_id)
            output_path = Path(job["destination"]) / relative_path
            partial_path = output_path.with_name(output_path.name + ".part")
            if output_path.is_file():
                state["skipped_tasks"] += 1
                state["current"] = {"task_id": task_id, "path": str(relative_path), "state": "already exists"}
                write_json(state_path, state)
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            state["current"] = {
                "task_id": task_id, "path": str(relative_path), "downloaded_bytes": 0, "total_bytes": None,
            }
            write_json(state_path, state)
            digest = hashlib.sha256()
            last_update = 0.0
            try:
                request = build_request(job["controller"], task_id, job["file"], headers, bool(job.get("insecure")))
                with contextlib.closing(urllib.request.urlopen(request, timeout=60, context=ssl_context)) as response:
                    total_bytes = response.headers.get("Content-Length")
                    state["current"]["total_bytes"] = int(total_bytes) if total_bytes and total_bytes.isdigit() else None
                    with partial_path.open("wb") as handle:
                        while block := response.read(CHUNK_BYTES):
                            if cancelled:
                                raise DownloadCancelled
                            handle.write(block)
                            digest.update(block)
                            state["bytes_downloaded"] += len(block)
                            state["current"]["downloaded_bytes"] += len(block)
                            if time.monotonic() - last_update >= 0.5:
                                write_json(state_path, state)
                                last_update = time.monotonic()
                expected_sha256 = str(task.get("expected_sha256") or "").lower()
                actual_sha256 = digest.hexdigest()
                if expected_sha256 and expected_sha256 != actual_sha256:
                    partial_path.unlink(missing_ok=True)
                    raise RuntimeError(f"SHA-256 mismatch (expected {expected_sha256}, got {actual_sha256})")
                partial_path.replace(output_path)
                state["completed_tasks"] += 1
            except DownloadCancelled:
                partial_path.unlink(missing_ok=True)
                raise
            except (OSError, urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as exc:
                partial_path.unlink(missing_ok=True)
                state["failed_tasks"] += 1
                state["errors"].append({"task_id": task_id, "path": str(relative_path), "error": str(exc)[:500]})
            finally:
                write_json(state_path, state)
        state["status"] = "completed"
        state["finished_at"] = time.time()
        state["current"] = None
        write_json(state_path, state)
        return 0
    except DownloadCancelled:
        state["status"] = "cancelled"
        state["finished_at"] = time.time()
        write_json(state_path, state)
        return 130
    except Exception as exc:
        state["status"] = "failed"
        state["finished_at"] = time.time()
        state["errors"].append({"error": str(exc)[:500]})
        write_json(state_path, state)
        return 1
    finally:
        _lock_handle.close()


def print_status(state_dir: Path) -> int:
    _lock_path, state_path, _job_path = state_paths(state_dir)
    state = read_json(state_path)
    if state is None:
        print("No download has been started from this state directory.")
        return 0
    active = state.get("status") == "running" and process_running(state.get("pid"))
    print(f"Status: {'running' if active else state.get('status', 'unknown')}")
    print(f"Files: {state.get('completed_tasks', 0)} completed, {state.get('skipped_tasks', 0)} skipped, "
          f"{state.get('failed_tasks', 0)} failed, {state.get('total_tasks', 0)} total")
    print(f"Downloaded: {state.get('bytes_downloaded', 0):,} bytes")
    current = state.get("current") or {}
    if current:
        total = current.get("total_bytes")
        progress = f"/{total:,}" if isinstance(total, int) else ""
        print(f"Current: {current.get('path', '-') } ({current.get('downloaded_bytes', 0):,}{progress} bytes)")
    if state.get("errors"):
        print(f"Latest error: {state['errors'][-1].get('error', 'unknown error')}")
    return 0


def cancel(state_dir: Path) -> int:
    _lock_path, state_path, _job_path = state_paths(state_dir)
    state = read_json(state_path)
    if not state or state.get("status") != "running" or not process_running(state.get("pid")):
        print("No active download to cancel.")
        return 0
    pid = int(state["pid"])
    os.killpg(pid, signal.SIGTERM)
    print(f"Cancellation requested for download process {pid}.")
    return 0


def start(args: argparse.Namespace) -> int:
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    database = Path(args.database).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()
    state_dir = Path(args.state_dir).expanduser().resolve()
    lock_path, state_path, job_path = state_paths(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        existing = read_json(state_path) or {}
        raise ValueError(f"a download is already running (PID {existing.get('pid', 'unknown')}); use 'status' or 'cancel'") from exc
    try:
        statuses = tuple(dict.fromkeys(args.status or ["completed"]))
        controller_file = "input" if args.file == "source" else "output"
        tasks = select_tasks(database, controller_file, args.count, statuses)
        if not tasks:
            raise ValueError("no matching downloadable tasks were found in the SQLite snapshot")
        headers = dict(args.header)
        if args.basic_auth_file:
            if any(name.lower() == "authorization" for name in headers):
                raise ValueError("use either --header Authorization or --basic-auth-file, not both")
            name, value = basic_auth_header(args.basic_auth_file)
            headers[name] = value
        job = {
            "controller": args.controller.rstrip("/"), "file": controller_file, "file_label": args.file,
            "destination": str(destination), "headers": headers, "insecure": args.insecure,
            "tasks": tasks,
        }
        write_json(job_path, job)
        log_path = state_dir / "download.log"
        with log_path.open("a") as log_file:
            command = [sys.executable, str(Path(__file__).resolve()), "_worker",
                       "--job", str(job_path), "--state", str(state_path), "--lock-fd", str(lock_handle.fileno())]
            process = subprocess.Popen(
                command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=(lock_handle.fileno(),), close_fds=True,
            )
        initial_state = {
            "status": "starting", "pid": process.pid, "started_at": time.time(),
            "total_tasks": len(tasks), "completed_tasks": 0, "skipped_tasks": 0,
            "failed_tasks": 0, "bytes_downloaded": 0, "current": None, "errors": [],
        }
        # Do not clobber a state record the child may already have written.
        current_state = read_json(state_path) or {}
        if not current_state or current_state.get("status") == "starting":
            write_json(state_path, initial_state)
        print(f"Started background download PID {process.pid}: {len(tasks)} {args.file} file(s).")
        print(f"Destination: {destination}")
        print(f"Progress: {Path(__file__).name} status --state-dir {state_dir}")
        return 0
    finally:
        lock_handle.close()


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Download random task video files through a Controller")
    subcommands = command.add_subparsers(dest="command", required=True)
    start_parser = subcommands.add_parser("start", help="Start one background download batch")
    start_parser.add_argument("--database", required=True, help="Downloaded Controller SQLite snapshot")
    start_parser.add_argument("--controller", required=True, help="Controller base URL, e.g. https://controller.example.com")
    start_parser.add_argument("--count", type=int, required=True, help="Number of random tasks to select")
    start_parser.add_argument("--destination", default="./task-downloads", help="Root directory for downloaded object-key paths")
    start_parser.add_argument("--file", choices=("source", "output"), default="source", help="Download source video (default) or processed output")
    start_parser.add_argument("--status", action="append", default=None,
                              help="Task status to include; repeat for multiple statuses (default: completed)")
    start_parser.add_argument("--header", action="append", default=[], type=parse_header,
                              help="HTTP header for Controller/Nginx auth; repeat as needed, e.g. 'Authorization: Bearer …'")
    start_parser.add_argument("--basic-auth-file",
                              help="File containing one Controller Nginx 'username:password' line (recommended for Basic Auth)")
    start_parser.add_argument("--insecure", action="store_true", help="Allow an untrusted HTTPS certificate")
    start_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR), help="Local state/lock directory")
    for name, help_text in (("status", "Show progress for the current or latest batch"), ("cancel", "Request cancellation of the active batch")):
        action_parser = subcommands.add_parser(name, help=help_text)
        action_parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR), help="Local state/lock directory")
    worker_parser = subcommands.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--job", required=True)
    worker_parser.add_argument("--state", required=True)
    worker_parser.add_argument("--lock-fd", type=int, required=True)
    return command


def main() -> int:
    args = parser().parse_args()
    if args.command == "start":
        return start(args)
    if args.command == "status":
        return print_status(Path(args.state_dir).expanduser().resolve())
    if args.command == "cancel":
        return cancel(Path(args.state_dir).expanduser().resolve())
    if args.command == "_worker":
        return worker(Path(args.job), Path(args.state), args.lock_fd)
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
