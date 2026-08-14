#!/usr/bin/env python3
"""Pull-based worker for a video-mask cluster."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ARGS = ["--fisheye", "--fisheye-device", "pico4", "--face-size", "640",
                "--face-conf", "0.35", "--face-int", "5", "--frame-skip", "2",
                "--face-model", "yolov8"]
DEFAULT_ALGORITHM = "video_mask_batch_fish_v1.py"
SINGLE_PUT_MAX_BYTES = 5 * 1024 * 1024 * 1024


def request_json(url: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duration(path: Path) -> float | None:
    process = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                             text=True, capture_output=True, check=False)
    try:
        return float(process.stdout.strip())
    except ValueError:
        return None


class Worker:
    def __init__(self, args: argparse.Namespace):
        self.controller = args.controller.rstrip("/")
        self.worker_id = args.worker_id
        self.token = args.token
        self.work_dir = args.work_dir.resolve()
        self.completed_output_dir = args.completed_output_dir.resolve()
        self.algorithm = args.algorithm.resolve()
        self.python = args.python
        self.poll_seconds = args.poll_seconds
        self.extra_args = args.extra_arg or DEFAULT_ARGS
        self.allow_local_files = args.allow_local_files
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.completed_output_dir.mkdir(parents=True, exist_ok=True)

    def payload(self, **extra: Any) -> dict[str, Any]:
        return {"worker_id": self.worker_id, "token": self.token, **extra}

    def capabilities(self) -> dict[str, Any]:
        """Describe this process so the Controller dashboard can identify it."""
        hostname = socket.gethostname()
        info: dict[str, Any] = {
            "algorithm": DEFAULT_ALGORITHM,
            "pid": os.getpid(),
            "hostname": hostname,
        }
        prefix, marker, slot = self.worker_id.rpartition("-slot-")
        if marker and prefix and slot.isdigit():
            info["slot"] = int(slot)
        try:
            addresses = socket.gethostbyname_ex(hostname)[2]
            info["ip_addresses"] = sorted({address for address in addresses if not address.startswith("127.")})
        except OSError:
            pass
        try:
            import torch
            info.update({"cuda_available": torch.cuda.is_available(), "torch": torch.__version__})
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
        except Exception as exc:
            info["cuda_error"] = str(exc)
        return info

    def api(self, suffix: str, payload: dict[str, Any]) -> dict[str, Any]:
        return request_json(self.controller + suffix, payload)

    def report(self, task_id: str, status: str, **progress: Any) -> None:
        self.api(f"/api/tasks/{task_id}/progress", self.payload(status=status, progress=progress))

    def busy_heartbeat(self, stop: threading.Event, capabilities: dict[str, Any],
                       cancel_requested: threading.Event) -> None:
        """Keep long downloads/encodes leased to this Worker."""
        while not stop.wait(15):
            try:
                response = self.api(f"/api/workers/{self.worker_id}/heartbeat",
                                    self.payload(status="busy", capabilities=capabilities))
                if response.get("cancel_requested"):
                    cancel_requested.set()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"Worker heartbeat failed: {exc}", file=sys.stderr, flush=True)

    def download(self, url: str, destination: Path) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "file":
            if not self.allow_local_files:
                raise RuntimeError("file:// tasks require --allow-local-files")
            local_source = Path(urllib.parse.unquote(parsed.path))
            shutil.copy2(local_source, destination)
            return
        partial = destination.with_suffix(destination.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        partial.replace(destination)

    def upload(self, url: str, source: Path) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == "file":
            if not self.allow_local_files:
                raise RuntimeError("file:// tasks require --allow-local-files")
            destination = Path(urllib.parse.unquote(parsed.path))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return
        # curl streams the file from disk.  Do not use read_bytes() here: long
        # videos must never be loaded into a Worker's RAM for upload.
        result = subprocess.run(
            ["curl", "--fail-with-body", "--silent", "--show-error", "--upload-file", str(source), url],
            text=True, capture_output=True, check=False,
        )
        if result.returncode != 0:
            diagnostic = "\n".join(part for part in (result.stderr, result.stdout) if part).strip()
            raise RuntimeError("output upload failed: " + diagnostic[-3000:])

    @staticmethod
    def upload_part(url: str, source: Path, offset: int, size: int) -> str:
        """Stream one byte range to a pre-signed S3 UploadPart URL."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https":
            raise RuntimeError("multipart upload URL must use HTTPS")
        target = urllib.parse.urlunparse(("", "", parsed.path or "/", parsed.params, parsed.query, ""))
        connection = http.client.HTTPSConnection(parsed.netloc, timeout=120)
        try:
            connection.putrequest("PUT", target, skip_host=True)
            connection.putheader("Host", parsed.netloc)
            connection.putheader("Content-Length", str(size))
            connection.endheaders()
            with source.open("rb") as handle:
                handle.seek(offset)
                remaining = size
                while remaining:
                    block = handle.read(min(1024 * 1024, remaining))
                    if not block:
                        raise RuntimeError("output file changed during multipart upload")
                    connection.send(block)
                    remaining -= len(block)
            response = connection.getresponse()
            body = response.read().decode("utf-8", errors="replace")
            if not 200 <= response.status < 300:
                raise RuntimeError(f"multipart part upload failed: HTTP {response.status} {body[-2000:]}")
            etag = response.getheader("ETag")
            if not etag:
                raise RuntimeError("multipart part upload response did not include an ETag")
            return etag
        finally:
            connection.close()

    def upload_multipart(self, task: dict[str, Any], source: Path,
                         cancel_requested: threading.Event) -> None:
        task_id = str(task["task_id"])
        started = self.api(f"/api/workers/{self.worker_id}/tasks/{task_id}/multipart/start", self.payload())
        upload_id, part_size = str(started["upload_id"]), int(started["part_size"])
        size = source.stat().st_size
        parts: list[dict[str, Any]] = []
        completed = False
        try:
            for part_number, offset in enumerate(range(0, size, part_size), start=1):
                if cancel_requested.is_set():
                    raise RuntimeError("cancelled by administrator")
                length = min(part_size, size - offset)
                for attempt in range(1, 4):
                    try:
                        signed = self.api(
                            f"/api/workers/{self.worker_id}/tasks/{task_id}/multipart/part-url",
                            self.payload(upload_id=upload_id, part_number=part_number),
                        )
                        etag = self.upload_part(str(signed["upload_part_url"]), source, offset, length)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        print(f"[{task_id}] multipart part {part_number} failed; retrying", file=sys.stderr, flush=True)
                parts.append({"part_number": part_number, "etag": etag})
                self.report(task_id, "uploading", phase="multipart_uploading",
                            uploaded_bytes=min(offset + length, size), output_bytes=size,
                            multipart_part=part_number)
            self.api(f"/api/workers/{self.worker_id}/tasks/{task_id}/multipart/complete",
                     self.payload(upload_id=upload_id, parts=parts))
            completed = True
        finally:
            if not completed:
                try:
                    self.api(f"/api/workers/{self.worker_id}/tasks/{task_id}/multipart/abort",
                             self.payload(upload_id=upload_id))
                except Exception as exc:
                    print(f"[{task_id}] failed to abort multipart upload: {exc}", file=sys.stderr, flush=True)

    def algorithm_command(self, source: Path, output_dir: Path, arguments: list[str]) -> list[str]:
        """Build an invocation for either a development script or a release binary.

        The normal source deployment supplies a ``.py`` algorithm and therefore
        needs Python. A release supplies an executable wrapper around compiled
        proprietary modules and must be invoked directly; attempting to run it
        as a Python script makes an otherwise valid Worker fail at task claim time.
        """
        shared = [str(source), "--out-dir", str(output_dir), *arguments]
        if self.algorithm.suffix.lower() == ".py":
            return [self.python, "-u", str(self.algorithm), *shared]
        if not self.algorithm.is_file():
            raise RuntimeError(f"algorithm executable does not exist: {self.algorithm}")
        if not os.access(self.algorithm, os.X_OK):
            raise RuntimeError(f"algorithm executable is not executable: {self.algorithm}")
        return [str(self.algorithm), *shared]

    def persist_output(self, output: Path, task_id: str) -> Path:
        """Keep a completed result after the task work directory is cleaned."""
        destination = self.completed_output_dir / output.name
        if destination.exists():
            destination = self.completed_output_dir / f"{output.stem}_{task_id[:8]}{output.suffix}"
        shutil.copy2(output, destination)
        return destination

    def run_task(self, task: dict[str, Any]) -> None:
        task_id = task["task_id"]
        directory = self.work_dir / task_id
        directory.mkdir(parents=True, exist_ok=True)
        source_name = Path(urllib.parse.urlparse(task["source_url"]).path).name or "input.mp4"
        source = directory / source_name
        output_dir = directory / "output"
        output_dir.mkdir(exist_ok=True)
        heartbeat_stop = threading.Event()
        cancel_requested = threading.Event()
        heartbeat = threading.Thread(target=self.busy_heartbeat,
                                     args=(heartbeat_stop, self.capabilities(), cancel_requested), daemon=True)
        heartbeat.start()
        phase = "initializing"
        output: Path | None = None
        elapsed: float | None = None
        try:
            phase = "downloading"
            self.report(task_id, "downloading", phase="downloading")
            self.download(task["source_url"], source)
            if task.get("source_sha256") and sha256(source) != task["source_sha256"]:
                raise RuntimeError("downloaded source checksum does not match task")
            phase = "processing"
            self.report(task_id, "processing", phase="processing", source_bytes=source.stat().st_size,
                        processing_started_at=time.time())
            command = self.algorithm_command(
                source, output_dir, task.get("arguments") or self.extra_args
            )
            started = time.monotonic()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, start_new_session=True,
                                       env={**os.environ, "PYTHONUNBUFFERED": "1"})

            cancel_monitor_stop = threading.Event()

            def stop_cancelled_process() -> None:
                while not cancel_monitor_stop.wait(0.5):
                    if cancel_requested.is_set() and process.poll() is None:
                        print(f"[{task_id}] cancellation requested; stopping algorithm", flush=True)
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        return

            cancel_monitor = threading.Thread(target=stop_cancelled_process, daemon=True)
            cancel_monitor.start()
            tail: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[{task_id}] {line}", flush=True)
                    tail = (tail + [line])[-20:]
            exit_code = process.wait()
            cancel_monitor_stop.set()
            cancel_monitor.join(timeout=1)
            if cancel_requested.is_set():
                raise RuntimeError("cancelled by administrator")
            if exit_code != 0:
                raise RuntimeError("algorithm failed: " + " | ".join(tail)[-1500:])
            candidates = sorted(output_dir.glob("masked_*.mp4"))
            if not candidates:
                raise RuntimeError("algorithm completed without an output mp4")
            output = candidates[0]
            elapsed = round(time.monotonic() - started, 1)
            output_url = task.get("output_upload_url")
            if output_url:
                phase = "uploading"
                self.report(task_id, "uploading", phase="uploading", elapsed_seconds=elapsed)
                if output.stat().st_size > SINGLE_PUT_MAX_BYTES:
                    self.upload_multipart(task, output, cancel_requested)
                else:
                    try:
                        self.upload(output_url, output)
                    except RuntimeError as first_upload_error:
                        # Long-running tasks may reach S3 after their original PUT
                        # URL expires. Refresh only the output URL: the completed
                        # video stays local, so no download or inference is repeated.
                        print(f"[{task_id}] upload failed; requesting a fresh upload URL", flush=True)
                        try:
                            refreshed = self.api(
                                f"/api/workers/{self.worker_id}/tasks/{task_id}/upload-url", self.payload()
                            )
                        except Exception as refresh_error:
                            raise RuntimeError(
                                "output upload failed and the refreshed URL request failed; "
                                f"first error: {first_upload_error}; refresh error: {refresh_error}"
                            ) from refresh_error
                        refreshed_url = str(refreshed.get("output_upload_url") or "")
                        if not refreshed_url:
                            raise RuntimeError(
                                "output upload failed and Controller did not provide a refreshed URL: "
                                + str(first_upload_error)
                            ) from first_upload_error
                        self.report(task_id, "uploading", phase="uploading_retry", elapsed_seconds=elapsed,
                                    upload_retry_reason=str(first_upload_error)[-500:])
                        try:
                            self.upload(refreshed_url, output)
                        except RuntimeError as retry_error:
                            raise RuntimeError(
                                "output upload failed after refreshed URL: " + str(retry_error)
                            ) from retry_error
                completed_output = output
                # Do not save a sensitive presigned URL in the Controller database.
                output_location = task.get("output_object_key") or "uploaded"
            else:
                phase = "saving_local_output"
                self.report(task_id, "uploading", phase="saving_local_output", elapsed_seconds=elapsed)
                completed_output = self.persist_output(output, task_id)
                output_location = str(completed_output)
            self.api(f"/api/tasks/{task_id}/complete", self.payload(
                output_sha256=sha256(completed_output), output_duration_seconds=duration(completed_output),
                progress={
                    "input_filename": source.name,
                    "input_bytes": source.stat().st_size,
                    "output_filename": completed_output.name,
                    "output_bytes": completed_output.stat().st_size,
                    "processing_seconds": elapsed,
                    # Retain the existing field for older Controller databases/API consumers.
                    "elapsed_seconds": elapsed,
                    "output_location": output_location,
                },
            ))
        except Exception as exc:
            message = f"{phase}: {exc}"
            print(f"[{task_id}] ERROR: {message}", file=sys.stderr, flush=True)
            failure_progress: dict[str, Any] = {
                "input_filename": source.name,
                "input_bytes": source.stat().st_size if source.exists() else None,
            }
            if elapsed is not None:
                failure_progress.update({"processing_seconds": elapsed, "elapsed_seconds": elapsed})
            if output is not None and output.exists():
                failure_progress.update({
                    "output_filename": output.name,
                    "output_bytes": output.stat().st_size,
                    "output_duration_seconds": duration(output),
                })
            try:
                self.api(f"/api/tasks/{task_id}/fail", self.payload(
                    error_message=message[:3000], progress=failure_progress,
                ))
            except Exception as report_error:
                print(f"[{task_id}] failed to report error: {report_error}", file=sys.stderr, flush=True)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2)
            shutil.rmtree(directory, ignore_errors=True)

    def serve(self) -> None:
        capabilities = self.capabilities()
        self.api("/api/workers/register", self.payload(capabilities=capabilities))
        print(f"Worker {self.worker_id} registered with {self.controller}", flush=True)
        while True:
            try:
                self.api(f"/api/workers/{self.worker_id}/heartbeat", self.payload(status="ready", capabilities=capabilities))
                response = self.api(f"/api/workers/{self.worker_id}/claim", self.payload(capabilities=capabilities))
                if response.get("task"):
                    self.run_task(response["task"])
                    continue
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"Controller unavailable: {exc}", file=sys.stderr, flush=True)
            time.sleep(self.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-mask cluster worker agent")
    parser.add_argument("--controller", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/home/ubuntu/work/video-mask"))
    parser.add_argument("--completed-output-dir", type=Path, default=Path("/home/ubuntu/outputs"),
                        help="Persistent output directory used when a task has no output_upload_url")
    parser.add_argument("--algorithm", type=Path, default=Path(__file__).resolve().parents[1] / DEFAULT_ALGORITHM)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--allow-local-files", action="store_true",
                        help="Allow file:// input/output URLs for single-machine testing only")
    args = parser.parse_args()
    if args.poll_seconds < 1:
        raise SystemExit("--poll-seconds must be positive")
    Worker(args).serve()


if __name__ == "__main__":
    main()
